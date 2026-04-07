"""
diffusion_engine_ip_adapter.py — IP-Adapter + ControlNet + LCM inference worker.

Architecture
------------
- StableDiffusionControlNetPipeline (txt2img) with LCM scheduler
- ControlNet conditioned on OpenPose skeleton (pose guide)
- IP-Adapter Plus for character appearance preservation
- TAESD for fast VAE encode/decode
- CLIP image embeddings cached at startup (zero per-frame cost)
- Temporal feedback via manual latent blending:
    prev_output → VAE encode → blend with fixed noise → run ALL steps
    ControlNet guides pose at every step (no step skipping).
"""

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

        # ── 2. Base pipeline (txt2img — feedback via manual latent blend) ─
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

        # ── 4. TAESD (fast VAE encode + decode) ──────────────────────────
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
                f"[IPAdapter] Reference embeddings cached: "
                f"{[e.shape for e in self._ip_embeds]}"
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

        # Steps and scales
        self._steps = max(cfg.num_inference_steps, 4)
        self._cn_scale = getattr(cfg, "controlnet_conditioning_scale", 1.0)

        # Temporal feedback: noise ratio blended with previous latent
        #   0.3 = 30% noise + 70% previous frame latent (recommended)
        #   0.5 = 50/50
        #   1.0 = pure noise (no feedback, each frame independent)
        self._feedback_alpha = getattr(cfg, "temporal_feedback_strength", 0.3)
        print(
            f"[IPAdapter] Temporal feedback: alpha={self._feedback_alpha}, "
            f"steps={self._steps} (all steps run regardless)"
        )

        # ── 10. Warmup ────────────────────────────────────────────────────
        print(f"[IPAdapter] Warming up (steps={self._steps}) …")
        dummy = torch.zeros(1, 3, H, W, dtype=dtype, device=device).to(
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
        seed = getattr(cfg, "seed", 42)
        generator = torch.Generator(device=self._device).manual_seed(
            seed if seed >= 0 else torch.randint(0, 2**31, (1,)).item()
        )

        lH, lW = self._H // 8, self._W // 8
        alpha = self._feedback_alpha

        # Fixed noise for deterministic baseline
        fixed_noise = torch.randn(
            (1, 4, lH, lW),
            dtype=self._dtype,
            device=self._device,
            generator=generator,
        )

        copy_stream = torch.cuda.Stream()
        prev_gpu_frame: torch.Tensor | None = None  # for async D2H copy
        prev_output_gpu: torch.Tensor | None = None  # (1,3,H,W) for feedback

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

            # ── Build starting latents ────────────────────────────────────
            with torch.inference_mode():
                if prev_output_gpu is not None and alpha < 1.0:
                    # Encode previous output → latent space
                    # TAESD encode expects [-1, 1] input; pipeline output is [0, 1]
                    prev_for_vae = (prev_output_gpu * 2.0 - 1.0).to(
                        memory_format=torch.channels_last
                    )
                    prev_latent = pipe.vae.encode(prev_for_vae).latents

                    # Blend: α*noise + (1-α)*prev_latent
                    # α=0.3 → 30% noise, 70% previous structure
                    start_latents = alpha * fixed_noise + (1.0 - alpha) * prev_latent
                else:
                    # First frame: pure fixed noise
                    start_latents = fixed_noise.clone()

                # ── Run full pipeline with custom latents (ALL steps) ─────
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
                    latents=start_latents,
                    output_type="pt",
                )

            gpu_frame = result.images
            # Keep on GPU for next frame's feedback
            prev_output_gpu = gpu_frame

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
