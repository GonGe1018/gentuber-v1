"""
diffusion_engine_ip_adapter.py — IP-Adapter + ControlNet + LCM inference worker.

Architecture
------------
- StableDiffusionControlNetPipeline with LCM scheduler
- ControlNet conditioned on OpenPose skeleton (pose guide)
- IP-Adapter Plus for character appearance preservation
- TAESD for fast VAE decoding
- CLIP image embeddings cached at startup (zero per-frame cost)

The key advantage over img2img approaches: character appearance (IP-Adapter)
and pose (ControlNet) are injected through independent paths, so they don't
conflict. ControlNet gets full txt2img freedom to reshape the pose.
"""

import math
import queue
import threading
from typing import Optional

import numpy as np
import torch
from diffusers import (
    AutoencoderTiny,
    ControlNetModel,
    LCMScheduler,
    StableDiffusionControlNetPipeline,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image


class DiffusionEngineIPAdapter:
    """
    Parameters
    ----------
    cfg : config.Config
    in_queue  : queue.Queue  — receives np.ndarray control maps or (ctrl, source) tuples
    out_queue : queue.Queue  — puts np.ndarray generated frames (RGB, HxWx3)
    """

    def __init__(self, cfg, in_queue: queue.Queue, out_queue: queue.Queue):
        self.cfg = cfg
        self.in_queue = in_queue
        self.out_queue = out_queue
        self._pipe = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def load(self) -> None:
        cfg = self.cfg
        dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        device = cfg.device
        H, W = cfg.output_height, cfg.output_width

        # ── 1. ControlNet (OpenPose) ──────────────────────────────────────
        print("[IPAdapter] Loading ControlNet …")
        controlnet = ControlNetModel.from_pretrained(
            cfg.controlnet_model_id,
            torch_dtype=dtype,
        )

        # ── 2. Base pipeline (anime model) ────────────────────────────────
        print("[IPAdapter] Loading base pipeline …")
        pipe = StableDiffusionControlNetPipeline.from_pretrained(
            cfg.lcm_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )

        # ── 3. LCM-LoRA + scheduler ──────────────────────────────────────
        print("[IPAdapter] Loading LCM-LoRA …")
        pipe.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe.fuse_lora()
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        # ── 4. TAESD (fast VAE decode) ────────────────────────────────────
        print("[IPAdapter] Loading TAESD …")
        pipe.vae = AutoencoderTiny.from_pretrained(
            cfg.taesd_model_id,
            torch_dtype=dtype,
        )

        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        # ── 5. IP-Adapter Plus ────────────────────────────────────────────
        ip_weight = getattr(cfg, "ip_adapter_weight", "ip-adapter-plus_sd15.bin")
        ip_scale = getattr(cfg, "ip_adapter_scale", 0.5)
        print(f"[IPAdapter] Loading IP-Adapter: {ip_weight}, scale={ip_scale} …")
        pipe.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name=ip_weight,
        )
        pipe.set_ip_adapter_scale(ip_scale)

        # ── 6. Performance optimizations ──────────────────────────────────
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.controlnet = pipe.controlnet.to(memory_format=torch.channels_last)
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

        # ── 7. Pre-compute text embeddings ────────────────────────────────
        print("[IPAdapter] Pre-computing text embeddings …")
        do_cfg = cfg.guidance_scale > 1.0
        with torch.inference_mode():
            self._prompt_embeds, self._neg_embeds = pipe.encode_prompt(
                prompt=cfg.prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=cfg.negative_prompt if do_cfg else None,
            )

        # ── 8. Cache CLIP image embeddings from reference ─────────────────
        ref_path = getattr(cfg, "reference_image", "")
        if ref_path:
            print(f"[IPAdapter] Encoding reference image: {ref_path} …")
            ref_pil = Image.open(ref_path).convert("RGB").resize((W, H))
            with torch.inference_mode():
                self._ip_embeds = pipe.prepare_ip_adapter_image_embeds(
                    ip_adapter_image=ref_pil,
                    ip_adapter_image_embeds=None,
                    device=device,
                    num_images_per_prompt=1,
                    do_classifier_free_guidance=do_cfg,
                )
            print(
                f"[IPAdapter] Reference embeddings cached: {[e.shape for e in self._ip_embeds]}"
            )
        else:
            self._ip_embeds = None
            print("[IPAdapter] WARNING: no reference image configured")

        # ── 9. Unload CLIP image encoder to free VRAM ─────────────────────
        if hasattr(pipe, "image_encoder") and pipe.image_encoder is not None:
            del pipe.image_encoder
            pipe.image_encoder = None
        if hasattr(pipe, "feature_extractor") and pipe.feature_extractor is not None:
            del pipe.feature_extractor
            pipe.feature_extractor = None
        torch.cuda.empty_cache()
        print("[IPAdapter] CLIP encoder unloaded (VRAM freed)")

        self._pipe = pipe
        self._device = device
        self._dtype = dtype
        self._H, self._W = H, W

        # Pre-allocate pinned buffer for ctrl upload
        self._pinned_buf = torch.empty(
            (1, 3, H, W), dtype=torch.float16, pin_memory=True
        )
        self._transfer_stream = torch.cuda.Stream()

        # Compute actual steps (need steps >= 1 after scheduler)
        self._steps = max(cfg.num_inference_steps, 4)  # IP-Adapter needs >= 4 steps
        self._cn_scale = getattr(cfg, "controlnet_conditioning_scale", 1.0)

        # ── 10. Warmup ────────────────────────────────────────────────────
        print(f"[IPAdapter] Warming up (steps={self._steps}) …")
        dummy = torch.zeros((1, 3, H, W), dtype=dtype, device=device).to(
            memory_format=torch.channels_last
        )
        with torch.inference_mode():
            for _ in range(4):
                pipe(
                    prompt_embeds=self._prompt_embeds,
                    negative_prompt_embeds=self._neg_embeds,
                    image=dummy,
                    ip_adapter_image_embeds=self._ip_embeds,
                    num_inference_steps=self._steps,
                    guidance_scale=cfg.guidance_scale,
                    controlnet_conditioning_scale=self._cn_scale,
                    width=W,
                    height=H,
                    output_type="pt",
                )
        torch.cuda.synchronize()

        print("[IPAdapter] Ready.")

    def start(self) -> "DiffusionEngineIPAdapter":
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="diffusion-ip"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        try:
            self.in_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)

    def _worker(self) -> None:
        cfg = self.cfg
        pipe = self._pipe
        torch.set_num_threads(2)
        generator = torch.Generator(device=self._device).manual_seed(
            getattr(cfg, "seed", 42)
        )

        copy_stream = torch.cuda.Stream()
        prev_gpu_frame: torch.Tensor | None = None

        while self._running:
            # Drain queue — keep freshest control map
            control_map = None
            try:
                while True:
                    item = self.in_queue.get_nowait()
                    if item is None:
                        self._running = False
                        break
                    control_map = item[0] if isinstance(item, tuple) else item
            except queue.Empty:
                pass

            if not self._running:
                break

            if control_map is None:
                try:
                    item = self.in_queue.get(timeout=0.5)
                    if item is None:
                        self._running = False
                        break
                    control_map = item[0] if isinstance(item, tuple) else item
                except queue.Empty:
                    if prev_gpu_frame is not None:
                        copy_stream.synchronize()
                        frame = (
                            prev_gpu_frame[0].permute(1, 2, 0).cpu().numpy().clip(0, 1)
                        )
                        frame = (frame * 255).astype(np.uint8)
                        if self.out_queue.full():
                            try:
                                self.out_queue.get_nowait()
                            except queue.Empty:
                                pass
                        self.out_queue.put(frame)
                        prev_gpu_frame = None
                    continue

            # ── Upload control map to GPU ─────────────────────────────────
            if (
                control_map.dtype == np.float16
                and control_map.ndim == 3
                and control_map.shape[0] == 3
            ):
                ctrl_np = control_map
            else:
                ctrl_np = control_map.transpose(2, 0, 1).astype(np.float16) / 255.0
            self._pinned_buf[0].copy_(torch.from_numpy(ctrl_np), non_blocking=False)
            with torch.cuda.stream(self._transfer_stream):
                ctrl_tensor = self._pinned_buf.to(
                    device=self._device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            torch.cuda.current_stream().wait_stream(self._transfer_stream)

            # ── GPU inference: txt2img + ControlNet + IP-Adapter ──────────
            with torch.inference_mode():
                result = pipe(
                    prompt_embeds=self._prompt_embeds,
                    negative_prompt_embeds=self._neg_embeds,
                    image=ctrl_tensor,
                    ip_adapter_image_embeds=self._ip_embeds,
                    num_inference_steps=self._steps,
                    guidance_scale=cfg.guidance_scale,
                    controlnet_conditioning_scale=self._cn_scale,
                    width=self._W,
                    height=self._H,
                    generator=generator,
                    output_type="pt",
                )
            gpu_frame = result.images

            # ── Async D2H copy ────────────────────────────────────────────
            if prev_gpu_frame is not None:
                with torch.cuda.stream(copy_stream):
                    cpu_frame = (
                        prev_gpu_frame[0]
                        .permute(1, 2, 0)
                        .to(device="cpu", non_blocking=True)
                    )
                copy_stream.synchronize()
                frame = (cpu_frame.float().numpy().clip(0, 1) * 255).astype(np.uint8)
                if self.out_queue.full():
                    try:
                        self.out_queue.get_nowait()
                    except queue.Empty:
                        pass
                self.out_queue.put(frame)

            prev_gpu_frame = gpu_frame
