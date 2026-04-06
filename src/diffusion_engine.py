"""
diffusion_engine.py — Async LCM + ControlNet + TAESD inference worker.

Architecture
------------
- StableDiffusionControlNetPipeline with LCM scheduler
- ControlNet conditioned on OpenPose skeleton map
- TAESD (Tiny AutoEncoder) replaces the full VAE for ~10x faster decoding
- Runs in a dedicated thread; communicates via queues so the pose stage
  and display stage are never blocked waiting for GPU work.

Stream-batch strategy
---------------------
The worker loop picks the LATEST available control map from the input
queue (dropping stale frames) and immediately starts the next inference
without waiting for the display thread to consume the previous result.
This keeps GPU utilisation high and end-to-end latency low.
"""

import queue
import threading
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderTiny,
    ControlNetModel,
    LCMScheduler,
    StableDiffusionControlNetImg2ImgPipeline,
    StableDiffusionControlNetPipeline,
)


class DiffusionEngine:
    """
    Parameters
    ----------
    cfg : config.Config
    in_queue  : queue.Queue  — receives np.ndarray control maps (RGB, HxWx3)
    out_queue : queue.Queue  — puts np.ndarray generated frames (RGB, HxWx3)
    """

    def __init__(self, cfg, in_queue: queue.Queue, out_queue: queue.Queue):
        self.cfg = cfg
        self.in_queue = in_queue
        self.out_queue = out_queue
        self._pipe: Optional[StableDiffusionControlNetPipeline] = None
        self._thread: Optional[threading.Thread] = None
        self._running = False

    # ── lifecycle ────────────────────────────────────────────────────────────

    def load(self) -> None:
        """Download / load models onto GPU.  Call once before start()."""
        cfg = self.cfg
        dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        device = cfg.device

        self._use_source = getattr(cfg, "img2img_input", "noise") == "camera"
        self._use_reference = getattr(cfg, "img2img_input", "noise") == "reference"
        self._strength = getattr(cfg, "img2img_strength", 0.5)

        print("[DiffusionEngine] Loading ControlNet …")
        controlnet = ControlNetModel.from_pretrained(
            cfg.controlnet_model_id,
            torch_dtype=dtype,
        )

        print("[DiffusionEngine] Loading base pipeline …")
        if self._use_source or self._use_reference:
            pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
                cfg.base_model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            )
            mode = "reference" if self._use_reference else "camera"
            print(f"[DiffusionEngine] img2img mode={mode}, strength={self._strength}")
        else:
            pipe = StableDiffusionControlNetPipeline.from_pretrained(
                cfg.base_model_id,
                controlnet=controlnet,
                torch_dtype=dtype,
                safety_checker=None,
            )

        # Swap in LCM scheduler for 4-step inference
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

        # Replace full VAE with Tiny VAE (~10x faster decode, minor quality loss)
        print("[DiffusionEngine] Loading TAESD …")
        pipe.vae = AutoencoderTiny.from_pretrained(
            cfg.taesd_model_id,
            torch_dtype=dtype,
        )

        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        # cuDNN auto-tuner + TF32 (Blackwell/Ampere: free ~10% speedup)
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        # Flash Attention via PyTorch SDPA backend
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        # channels_last: ~15% faster on Ampere/Ada/Blackwell
        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.controlnet = pipe.controlnet.to(memory_format=torch.channels_last)
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

        # Use PyTorch native SDPA (faster than attention slicing on sm_120)
        try:
            pipe.unet.set_attn_processor(
                __import__("diffusers").models.attention_processor.AttnProcessor2_0()
            )
            print("[DiffusionEngine] SDPA attention enabled")
        except Exception:
            pipe.enable_attention_slicing()
            print("[DiffusionEngine] Fallback: attention slicing")

        # Pre-compute text embeddings once — skip CLIP every frame
        print("[DiffusionEngine] Pre-computing text embeddings …")
        do_cfg = cfg.guidance_scale > 1.0
        with torch.inference_mode():
            self._prompt_embeds, self._neg_embeds = pipe.encode_prompt(
                prompt=cfg.prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=do_cfg,
                negative_prompt=cfg.negative_prompt if do_cfg else None,
            )

        self._pipe = pipe
        self._device = device
        self._dtype = dtype

        # Pre-allocate pinned memory staging buffer for zero-copy CPU->GPU transfer
        H, W = cfg.output_height, cfg.output_width
        self._H, self._W = H, W
        self._pinned_buf = torch.empty(
            (1, 3, H, W), dtype=torch.float16, pin_memory=True
        )
        # Source frame pinned buffer (for camera img2img mode)
        if self._use_source:
            self._pinned_src = torch.empty(
                (1, 3, H, W), dtype=torch.float16, pin_memory=True
            )
        self._transfer_stream = torch.cuda.Stream()

        # Pre-encode reference image if configured
        self._ref_tensor = None
        if self._use_reference:
            ref_path = getattr(cfg, "reference_image", "")
            if ref_path:
                import cv2 as _cv2

                ref_bgr = _cv2.imread(ref_path)
                if ref_bgr is None:
                    print(
                        f"[DiffusionEngine] WARNING: cannot read reference: {ref_path}"
                    )
                    self._use_reference = False
                else:
                    ref_rgb = _cv2.cvtColor(ref_bgr, _cv2.COLOR_BGR2RGB)
                    ref_rgb = _cv2.resize(ref_rgb, (W, H))
                    # Normalize to [0, 1] for img2img pipeline input
                    self._ref_tensor = (
                        torch.from_numpy(ref_rgb)
                        .float()
                        .div(255.0)
                        .permute(2, 0, 1)
                        .unsqueeze(0)
                        .to(
                            device=device,
                            dtype=dtype,
                            memory_format=torch.channels_last,
                        )
                    )
                    print(f"[DiffusionEngine] Reference cached: {ref_path}")

        # Warmup: run several inferences so cudnn.benchmark finishes tuning
        print("[DiffusionEngine] Warming up (cudnn tuning) ...")
        dummy = torch.zeros((1, 3, H, W), dtype=dtype, device=device).to(
            memory_format=torch.channels_last
        )
        # img2img needs enough steps so that steps*strength >= 1
        warmup_steps = cfg.num_inference_steps
        if self._use_source or self._use_reference:
            import math

            warmup_steps = max(warmup_steps, math.ceil(1.0 / self._strength))
        with torch.inference_mode():
            for _ in range(8):
                if self._use_source or self._use_reference:
                    pipe(
                        prompt_embeds=self._prompt_embeds,
                        negative_prompt_embeds=self._neg_embeds,
                        image=dummy,  # source frame
                        control_image=dummy,  # skeleton
                        strength=self._strength,
                        num_inference_steps=warmup_steps,
                        guidance_scale=cfg.guidance_scale,
                        output_type="pt",
                    )
                else:
                    pipe(
                        prompt_embeds=self._prompt_embeds,
                        negative_prompt_embeds=self._neg_embeds,
                        image=dummy,
                        num_inference_steps=warmup_steps,
                        guidance_scale=cfg.guidance_scale,
                        width=W,
                        height=H,
                        output_type="pt",
                    )
        torch.cuda.synchronize()
        self._actual_steps = warmup_steps

        print("[DiffusionEngine] Ready.")

    def start(self) -> "DiffusionEngine":
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="diffusion"
        )
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        # Unblock the worker if it's waiting on the queue
        try:
            self.in_queue.put_nowait(None)
        except queue.Full:
            pass
        if self._thread:
            self._thread.join(timeout=5.0)

    # ── worker loop ──────────────────────────────────────────────────────────

    def _worker(self) -> None:
        cfg = self.cfg
        torch.set_num_threads(2)
        generator = torch.Generator(device=cfg.device).manual_seed(42)

        copy_stream = torch.cuda.Stream()
        prev_gpu_frame: torch.Tensor | None = None

        while self._running:
            # Drain queue — keep only the freshest item
            control_map = None
            source_map = None
            try:
                while True:
                    item = self.in_queue.get_nowait()
                    if item is None:
                        self._running = False
                        break
                    if isinstance(item, tuple):
                        control_map, source_map = item[0], item[1]
                    else:
                        control_map = item
            except queue.Empty:
                pass

            if not self._running:
                break

            if control_map is None:
                try:
                    item = self.in_queue.get(timeout=0.05)
                    if item is None:
                        self._running = False
                        break
                    if isinstance(item, tuple):
                        control_map, source_map = item[0], item[1]
                    else:
                        control_map = item
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
                    device=cfg.device,
                    non_blocking=True,
                    memory_format=torch.channels_last,
                )
            torch.cuda.current_stream().wait_stream(self._transfer_stream)

            # ── Upload source frame if available ──────────────────────────
            src_tensor = None
            if self._use_reference and self._ref_tensor is not None:
                src_tensor = self._ref_tensor
            elif self._use_source and source_map is not None:
                self._pinned_src[0].copy_(
                    torch.from_numpy(source_map), non_blocking=False
                )
                with torch.cuda.stream(self._transfer_stream):
                    src_tensor = self._pinned_src.to(
                        device=cfg.device,
                        non_blocking=True,
                        memory_format=torch.channels_last,
                    )
                torch.cuda.current_stream().wait_stream(self._transfer_stream)
                # Normalize from [-1,1] to [0,1] for the pipeline
                src_tensor = (src_tensor + 1.0) * 0.5

            # ── GPU inference ─────────────────────────────────────────────
            with torch.inference_mode():
                if src_tensor is not None:
                    result = self._pipe(
                        prompt_embeds=self._prompt_embeds,
                        negative_prompt_embeds=self._neg_embeds,
                        image=src_tensor,  # source frame
                        control_image=ctrl_tensor,  # skeleton
                        strength=self._strength,
                        num_inference_steps=self._actual_steps,
                        guidance_scale=cfg.guidance_scale,
                        generator=generator,
                        output_type="pt",
                    )
                else:
                    result = self._pipe(
                        prompt_embeds=self._prompt_embeds,
                        negative_prompt_embeds=self._neg_embeds,
                        image=ctrl_tensor,
                        num_inference_steps=self._actual_steps,
                        guidance_scale=cfg.guidance_scale,
                        width=cfg.output_width,
                        height=cfg.output_height,
                        generator=generator,
                        output_type="pt",
                    )
            gpu_frame = result.images

            # ── Async copy previous frame to CPU while GPU is free ────────
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
