"""
diffusion_engine_lcm_graph.py — KohakuV2 + LCM-LoRA + T2I-Adapter + CUDA graph.

Same architecture as DiffusionEngineSDTurboGraph but uses an anime-specific
SD1.5 model with LCM-LoRA fused in for better visual quality.

Model: KBlueLeaf/kohaku-v2.1 + latent-consistency/lcm-lora-sdv1-5
Adapter: TencentARC/t2iadapter_openpose_sd14v1
"""

import queue
import threading
from typing import Optional

import numpy as np

import cv2
import torch
from diffusers import (
    AutoencoderTiny,
    LCMScheduler,
    StableDiffusionAdapterPipeline,
    StableDiffusionPipeline,
    T2IAdapter,
)
from diffusers.models.attention_processor import AttnProcessor2_0

LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"
ANIME_MODEL_ID = "KBlueLeaf/kohaku-v2.1"


class DiffusionEngineLCMGraph:
    """
    KohakuV2 + LCM-LoRA + T2I-Adapter with CUDA graph.

    Drop-in replacement for DiffusionEngineSDTurboGraph with better
    anime-style output quality at equivalent throughput.

    Parameters
    ----------
    cfg : config.Config
    in_queue  : queue.Queue
    out_queue : queue.Queue
    model_id  : str  — any SD1.5-compatible model (default: KohakuV2)
    """

    def __init__(
        self,
        cfg,
        in_queue: queue.Queue,
        out_queue: queue.Queue,
        model_id: str = None,
    ):
        self.cfg = cfg
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.model_id = model_id or getattr(cfg, "lcm_model_id", None) or ANIME_MODEL_ID
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def load(self) -> None:
        cfg = self.cfg
        dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        device = cfg.device
        H, W = cfg.output_height, cfg.output_width

        print(f"[LCMGraph] Loading {self.model_id} + LCM-LoRA ...")
        base = StableDiffusionPipeline.from_pretrained(
            self.model_id, torch_dtype=dtype, safety_checker=None
        )
        base.load_lora_weights(LCM_LORA_ID)
        base.fuse_lora()

        print("[LCMGraph] Loading T2I-Adapter ...")
        adapter = T2IAdapter.from_pretrained(
            cfg.t2i_adapter_model_id, torch_dtype=dtype
        )

        pipe = StableDiffusionAdapterPipeline(
            vae=base.vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            unet=base.unet,
            adapter=adapter,
            scheduler=LCMScheduler.from_config(base.scheduler.config),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        del base

        pipe.vae = AutoencoderTiny.from_pretrained(
            cfg.taesd_model_id, torch_dtype=dtype
        )
        pipe = pipe.to(device)
        pipe.set_progress_bar_config(disable=True)

        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
        pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
        adapter = adapter.to(memory_format=torch.channels_last)

        try:
            pipe.unet.set_attn_processor(AttnProcessor2_0())
            print("[LCMGraph] SDPA attention enabled")
        except Exception:
            pipe.enable_attention_slicing()

        self._pipe = pipe
        self._adapter = adapter
        self._device = device
        self._dtype = dtype
        self._H, self._W = H, W

        print("[LCMGraph] Pre-computing text embeddings ...")
        with torch.inference_mode():
            self._prompt_embeds, _ = pipe.encode_prompt(
                prompt=cfg.prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=None,
            )

        self._pinned_buf = torch.empty(
            (1, 3, H, W), dtype=torch.float16, pin_memory=True
        )
        self._transfer_stream = torch.cuda.Stream()

        self._build_graph()

        # Pre-compute scheduler timesteps for worker (CPU to avoid device sync in hot loop)
        self._pipe.scheduler.set_timesteps(
            max(1, self.cfg.num_inference_steps), device=device
        )
        self._timesteps = self._pipe.scheduler.timesteps.cpu()
        self._sigma_0 = float(self._pipe.scheduler.init_noise_sigma)

        print("[LCMGraph] Ready.")

    def _build_graph(self) -> None:
        cfg = self.cfg
        device = self._device
        dtype = self._dtype
        H, W = self._H, self._W
        lH, lW = H // 8, W // 8
        pipe = self._pipe
        adapter = self._adapter

        self._static_latents = torch.zeros(
            (1, 4, lH, lW), dtype=dtype, device=device
        ).to(memory_format=torch.channels_last)
        self._static_ctrl = torch.zeros((1, 3, H, W), dtype=dtype, device=device).to(
            memory_format=torch.channels_last
        )
        # LCM 1-step: t=999
        self._static_timestep = torch.tensor([999], dtype=torch.long, device=device)

        # Precompute denoising constants for t=999 (baked into CUDA graph).
        # LCM prediction_type="epsilon": x0 = (latents - sqrt(1-a)*noise) / sqrt(a)
        # For 1-step LCM the final output IS x0 (going straight to t=0).
        t_val = 999
        alpha_prod = pipe.scheduler.alphas_cumprod[t_val].to(device=device, dtype=dtype)
        self._sqrt_alpha = alpha_prod.sqrt()
        self._sqrt_one_minus_alpha = (1.0 - alpha_prod).sqrt()

        print("[LCMGraph] Warming up for CUDA graph capture ...")
        with torch.inference_mode():
            for _ in range(12):
                a = adapter(self._static_ctrl)
                noise_pred = pipe.unet(
                    self._static_latents,
                    self._static_timestep,
                    self._prompt_embeds,
                    down_intrablock_additional_residuals=a,
                    return_dict=False,
                )[0]
                x0 = (
                    self._static_latents.float()
                    - self._sqrt_one_minus_alpha.float() * noise_pred.float()
                ) / self._sqrt_alpha.float()
                pipe.vae.decode(
                    x0.to(dtype) / pipe.vae.config.scaling_factor, return_dict=False
                )
        torch.cuda.synchronize()

        print("[LCMGraph] Capturing CUDA graph (adapter+UNet+denoise+VAE) ...")
        self._graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(self._graph):
                self._adapter_state = adapter(self._static_ctrl)
                self._static_noise_pred = pipe.unet(
                    self._static_latents,
                    self._static_timestep,
                    self._prompt_embeds,
                    down_intrablock_additional_residuals=self._adapter_state,
                    return_dict=False,
                )[0]
                # Compute x0 in float32 to avoid float16 catastrophic cancellation.
                # At t=999: sqrt_alpha≈0.068, so dividing by it amplifies fp16 rounding errors.
                _x0 = (
                    self._static_latents.float()
                    - self._sqrt_one_minus_alpha.float()
                    * self._static_noise_pred.float()
                ) / self._sqrt_alpha.float()
                _x0_half = _x0.to(dtype)
                # Store x0 for img2img feedback (next frame reuses this)
                self._static_x0 = _x0_half.clone()
                self._static_decoded = pipe.vae.decode(
                    _x0_half / pipe.vae.config.scaling_factor, return_dict=False
                )[0]
        torch.cuda.synchronize()
        print(f"[LCMGraph] Graph captured ({lH}x{lW} latents)")

    def start(self) -> "DiffusionEngineLCMGraph":
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="lcm-graph"
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
        # Explicitly release CUDA graph and static GPU tensors so the
        # caller can del+empty_cache between benchmark runs without OOM
        self._graph = None
        self._static_latents = None
        self._static_ctrl = None
        self._static_decoded = None
        self._pipe = None
        self._adapter = None

    def _worker(self) -> None:
        cfg = self.cfg
        device = self._device
        dtype = self._dtype
        pipe = self._pipe
        H, W = self._H, self._W
        lH, lW = H // 8, W // 8
        steps = max(1, cfg.num_inference_steps)

        torch.set_num_threads(2)
        seed = getattr(cfg, "seed", 42)
        generator = torch.Generator(device=device).manual_seed(
            seed if seed >= 0 else torch.randint(0, 2**31, (1,)).item()
        )

        NOISE_RING = 64
        noise_ring = [
            torch.randn((1, 4, lH, lW), dtype=dtype, device=device, generator=generator)
            for _ in range(NOISE_RING)
        ]
        noise_idx = 0

        pinned_A = torch.empty((1, 3, H, W), dtype=torch.float16, pin_memory=True)
        pinned_B = torch.empty((1, 3, H, W), dtype=torch.float16, pin_memory=True)
        gpu_A = torch.empty(
            (1, 3, H, W), dtype=dtype, device=device, memory_format=torch.channels_last
        )
        gpu_B = torch.empty(
            (1, 3, H, W), dtype=dtype, device=device, memory_format=torch.channels_last
        )

        pinned_out = torch.empty((H, W, 3), dtype=torch.float32, pin_memory=True)
        copy_stream = torch.cuda.Stream()

        # Use pre-computed scheduler state from load()
        timesteps = self._timesteps
        sigma_0 = self._sigma_0

        # Control map jitter threshold — skip generation if change is below this
        ctrl_threshold = getattr(cfg, "ctrl_jitter_threshold", 0.015)

        # Source-guided img2img mode (StreamDiffusion style)
        use_source_img2img = getattr(cfg, "img2img_input", "noise") == "camera"

        # Pinned + GPU buffers for source frame upload
        if use_source_img2img:
            pinned_src = torch.empty((1, 3, H, W), dtype=torch.float16, pin_memory=True)
            gpu_src = torch.empty(
                (1, 3, H, W),
                dtype=dtype,
                device=device,
                memory_format=torch.channels_last,
            )

        def upload_ctrl(ctrl_map, pinned, gpu_buf):
            """Upload CHW float16 control map to GPU on transfer_stream."""
            pinned[0].copy_(torch.from_numpy(ctrl_map), non_blocking=False)
            with torch.cuda.stream(self._transfer_stream):
                gpu_buf.copy_(pinned, non_blocking=True)

        def upload_source(src_map):
            """Upload CHW float16 source frame to GPU on transfer_stream."""
            pinned_src[0].copy_(torch.from_numpy(src_map), non_blocking=False)
            with torch.cuda.stream(self._transfer_stream):
                gpu_src.copy_(pinned_src, non_blocking=True)

        def get_queue_item(last_ctrl, last_source):
            """Drain queue, return (ctrl, source, ctrl_changed, source_changed)."""
            item = None
            try:
                while True:
                    raw = self.in_queue.get_nowait()
                    if raw is None:
                        self._running = False
                        return None, None, False, False
                    item = raw
            except queue.Empty:
                pass
            if item is None:
                return last_ctrl, last_source, False, False

            # Unpack tuple or plain ctrl
            if isinstance(item, tuple):
                ctrl, source = item
            else:
                ctrl, source = item, None

            ctrl_changed = True
            ctrl_diff = float(
                np.abs(ctrl.astype(np.float32) - last_ctrl.astype(np.float32)).mean()
            )
            if ctrl_diff < ctrl_threshold:
                ctrl_changed = False

            source_changed = False
            if source is not None and last_source is not None:
                src_diff = float(
                    np.abs(
                        source.astype(np.float32) - last_source.astype(np.float32)
                    ).mean()
                )
                if src_diff > 0.001:
                    source_changed = True
            elif source is not None:
                source_changed = True

            return ctrl, source, ctrl_changed, source_changed

        # Bootstrap
        first_ctrl = None
        first_source = None
        while first_ctrl is None and self._running:
            try:
                raw = self.in_queue.get(timeout=0.1)
                if raw is None:
                    self._running = False
                    return
                if isinstance(raw, tuple):
                    first_ctrl, first_source = raw
                else:
                    first_ctrl = raw
            except queue.Empty:
                continue
        if not self._running:
            return

        upload_ctrl(first_ctrl, pinned_A, gpu_A)
        torch.cuda.current_stream().wait_stream(self._transfer_stream)
        self._static_ctrl.copy_(gpu_A)

        if use_source_img2img and first_source is not None:
            upload_source(first_source)
            torch.cuda.current_stream().wait_stream(self._transfer_stream)

        last_ctrl = first_ctrl
        last_source = first_source
        next_pinned, next_gpu = pinned_B, gpu_B
        next_ctrl_ready = False

        # Temporal coherence: when enabled, use a single fixed noise
        # so only pose conditioning changes between frames.
        temporal_blend = getattr(cfg, "temporal_blend", 0.5)
        use_fixed_noise = temporal_blend < 1.0
        if use_fixed_noise:
            fixed_noise = noise_ring[0]

        # Precompute alpha values for img2img (both source and feedback paths)
        img2img_strength = getattr(cfg, "img2img_strength", 0.5)
        use_img2img = img2img_strength < 1.0 or use_source_img2img
        prev_x0 = None  # for feedback path only
        has_prev_frame = False

        if use_img2img:
            t_feedback = int(999 * img2img_strength)
            t_feedback = max(1, min(999, t_feedback))
            alpha_fb = pipe.scheduler.alphas_cumprod[t_feedback].to(
                device=device, dtype=dtype
            )
            sqrt_alpha_fb = alpha_fb.sqrt()
            sqrt_one_minus_alpha_fb = (1.0 - alpha_fb).sqrt()
            t_feedback_tensor = torch.tensor(
                [t_feedback], dtype=torch.long, device=device
            )
            print(
                f"[LCMGraph] img2img: input={cfg.img2img_input}, "
                f"strength={img2img_strength}, t={t_feedback}, "
                f"alpha={alpha_fb.item():.4f}"
            )

        while self._running:
            with torch.inference_mode():
                if use_fixed_noise:
                    noise = fixed_noise
                else:
                    noise = noise_ring[noise_idx]
                    noise_idx = (noise_idx + 1) % NOISE_RING

                # ── Get next item from queue ──────────────────────────────
                next_ctrl, next_source, ctrl_changed, source_changed = get_queue_item(
                    last_ctrl, last_source
                )
                if next_ctrl is None:
                    break
                anything_changed = ctrl_changed or source_changed

                # If nothing changed and we have a frame, skip generation
                if not anything_changed and has_prev_frame:
                    pass  # reuse pinned_out from last iteration

                elif use_source_img2img and next_source is not None:
                    # ── Source-guided img2img (StreamDiffusion style) ──────
                    # Upload ctrl if changed
                    if ctrl_changed:
                        upload_ctrl(next_ctrl, next_pinned, next_gpu)
                        torch.cuda.current_stream().wait_stream(self._transfer_stream)
                        self._static_ctrl.copy_(next_gpu)
                        next_pinned, next_gpu = (
                            (pinned_A, gpu_A)
                            if next_gpu is gpu_B
                            else (pinned_B, gpu_B)
                        )
                    last_ctrl = next_ctrl

                    # Upload and encode source frame
                    if source_changed or not has_prev_frame:
                        upload_source(next_source)
                        torch.cuda.current_stream().wait_stream(self._transfer_stream)
                        last_source = next_source

                    # VAE encode: (1,3,H,W) [-1,1] → latent (1,4,lH,lW)
                    source_latent = pipe.vae.encode(
                        gpu_src.to(memory_format=torch.channels_last)
                    ).latents

                    # Forward diffuse: noised = sqrt(α)*latent + sqrt(1-α)*noise
                    noised = (
                        sqrt_alpha_fb * source_latent + sqrt_one_minus_alpha_fb * noise
                    ).to(memory_format=torch.channels_last)

                    # UNet denoise with T2I-Adapter pose guide
                    adapter_state = self._adapter(self._static_ctrl)
                    noise_pred = pipe.unet(
                        noised,
                        t_feedback_tensor,
                        self._prompt_embeds,
                        down_intrablock_additional_residuals=adapter_state,
                        return_dict=False,
                    )[0]

                    # epsilon → x0 in float32
                    x0 = (
                        noised.float()
                        - sqrt_one_minus_alpha_fb.float() * noise_pred.float()
                    ) / sqrt_alpha_fb.float()
                    x0_half = x0.to(dtype)

                    decoded = pipe.vae.decode(
                        x0_half / pipe.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]

                    copy_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(copy_stream):
                        frame_gpu = (decoded[0].permute(1, 2, 0).float() + 1.0) * 0.5
                        pinned_out.copy_(frame_gpu.clamp(0, 1), non_blocking=True)
                    has_prev_frame = True

                elif use_img2img and prev_x0 is not None:
                    # ── img2img feedback path (eager, legacy) ─────────────
                    if ctrl_changed:
                        upload_ctrl(next_ctrl, next_pinned, next_gpu)
                        torch.cuda.current_stream().wait_stream(self._transfer_stream)
                        self._static_ctrl.copy_(next_gpu)
                        next_pinned, next_gpu = (
                            (pinned_A, gpu_A)
                            if next_gpu is gpu_B
                            else (pinned_B, gpu_B)
                        )
                    last_ctrl = next_ctrl

                    noised = (
                        sqrt_alpha_fb * prev_x0 + sqrt_one_minus_alpha_fb * noise
                    ).to(memory_format=torch.channels_last)

                    adapter_state = self._adapter(self._static_ctrl)
                    noise_pred = pipe.unet(
                        noised,
                        t_feedback_tensor,
                        self._prompt_embeds,
                        down_intrablock_additional_residuals=adapter_state,
                        return_dict=False,
                    )[0]

                    x0 = (
                        noised.float()
                        - sqrt_one_minus_alpha_fb.float() * noise_pred.float()
                    ) / sqrt_alpha_fb.float()
                    prev_x0 = x0.to(dtype)

                    decoded = pipe.vae.decode(
                        prev_x0 / pipe.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]

                    copy_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(copy_stream):
                        frame_gpu = (decoded[0].permute(1, 2, 0).float() + 1.0) * 0.5
                        pinned_out.copy_(frame_gpu.clamp(0, 1), non_blocking=True)
                    has_prev_frame = True

                elif steps == 1:
                    # ── Fast path: CUDA graph (noise-based) ───────────────
                    self._static_latents.copy_(
                        noise.to(memory_format=torch.channels_last)
                    )

                    if ctrl_changed:
                        upload_ctrl(next_ctrl, next_pinned, next_gpu)
                        next_ctrl_ready = True
                    last_ctrl = next_ctrl

                    self._graph.replay()
                    has_prev_frame = True

                    if use_img2img and not use_source_img2img:
                        prev_x0 = self._static_x0.clone()

                    if next_ctrl_ready:
                        torch.cuda.current_stream().wait_stream(self._transfer_stream)
                        self._static_ctrl.copy_(next_gpu)
                        next_pinned, next_gpu = (
                            (pinned_A, gpu_A)
                            if next_gpu is gpu_B
                            else (pinned_B, gpu_B)
                        )
                        next_ctrl_ready = False

                    copy_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(copy_stream):
                        frame_gpu = (
                            self._static_decoded[0].permute(1, 2, 0).float() + 1.0
                        ) * 0.5
                        pinned_out.copy_(frame_gpu.clamp(0, 1), non_blocking=True)

                else:
                    # ── Multi-step: eager ──────────────────────────────────
                    if ctrl_changed:
                        upload_ctrl(next_ctrl, next_pinned, next_gpu)
                        torch.cuda.current_stream().wait_stream(self._transfer_stream)
                        self._static_ctrl.copy_(next_gpu)
                        next_pinned, next_gpu = (
                            (pinned_A, gpu_A)
                            if next_gpu is gpu_B
                            else (pinned_B, gpu_B)
                        )
                    last_ctrl = next_ctrl

                    pipe.scheduler.set_timesteps(steps, device=device)
                    latents = noise * sigma_0
                    adapter_state = self._adapter(self._static_ctrl)

                    for t in timesteps:
                        noise_pred = pipe.unet(
                            latents.to(memory_format=torch.channels_last),
                            t,
                            self._prompt_embeds,
                            down_intrablock_additional_residuals=adapter_state,
                            return_dict=False,
                        )[0]
                        latents = pipe.scheduler.step(
                            noise_pred, t, latents, return_dict=False
                        )[0]

                    decoded = pipe.vae.decode(
                        latents / pipe.vae.config.scaling_factor,
                        return_dict=False,
                    )[0]

                    copy_stream.wait_stream(torch.cuda.current_stream())
                    with torch.cuda.stream(copy_stream):
                        frame_gpu = (decoded[0].permute(1, 2, 0).float() + 1.0) * 0.5
                        pinned_out.copy_(frame_gpu.clamp(0, 1), non_blocking=True)
                    has_prev_frame = True

            copy_stream.synchronize()
            frame = cv2.convertScaleAbs(pinned_out.numpy(), alpha=255)

            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put(frame)

            if not self._running:
                break
