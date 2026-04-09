"""
diffusion_engine_ip_adapter.py — IP-Adapter + ControlNet + LCM inference worker.

Architecture
------------
- StableDiffusionControlNetImg2ImgPipeline with LCM scheduler
- ControlNet conditioned on OpenPose skeleton (pose guide)
- IP-Adapter Plus for character appearance preservation
- TAESD for fast VAE encode/decode
- CLIP image embeddings cached at startup (zero per-frame cost)
- Temporal feedback: previous frame as img2img input with enough steps
  so ControlNet can guide pose at multiple denoising steps.

Key insight: img2img strength controls the starting timestep, and
total_steps is set high enough so actual_steps = floor(total * strength)
gives ControlNet enough chances to guide the pose.
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
    StableDiffusionControlNetImg2ImgPipeline,
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
        self._pipe_txt2img = None
        self._pipe_img2img = None
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

        # ── 2a. txt2img pipeline (first frame only) ──────────────────────
        print("[IPAdapter] Loading txt2img pipeline …")
        pipe_txt = StableDiffusionControlNetPipeline.from_pretrained(
            cfg.lcm_model_id,
            controlnet=controlnet,
            torch_dtype=dtype,
            safety_checker=None,
        )

        # ── 3. LCM-LoRA + scheduler ──────────────────────────────────────
        print("[IPAdapter] Loading LCM-LoRA …")
        pipe_txt.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
        pipe_txt.fuse_lora()
        pipe_txt.scheduler = LCMScheduler.from_config(pipe_txt.scheduler.config)

        # ── 4. TAESD ─────────────────────────────────────────────────────
        print("[IPAdapter] Loading TAESD …")
        pipe_txt.vae = AutoencoderTiny.from_pretrained(
            cfg.taesd_model_id,
            torch_dtype=dtype,
        )

        pipe_txt = pipe_txt.to(device)
        pipe_txt.set_progress_bar_config(disable=True)

        # ── 5. IP-Adapter Plus ────────────────────────────────────────────
        ip_weight = getattr(cfg, "ip_adapter_weight", "ip-adapter-plus_sd15.bin")
        ip_scale = getattr(cfg, "ip_adapter_scale", 0.5)
        print(f"[IPAdapter] Loading IP-Adapter: {ip_weight}, scale={ip_scale} …")
        pipe_txt.load_ip_adapter(
            "h94/IP-Adapter",
            subfolder="models",
            weight_name=ip_weight,
        )
        pipe_txt.set_ip_adapter_scale(ip_scale)

        # ── 2b. img2img pipeline (shares components with txt2img) ─────────
        print("[IPAdapter] Creating img2img pipeline …")
        pipe_img = StableDiffusionControlNetImg2ImgPipeline(
            vae=pipe_txt.vae,
            text_encoder=pipe_txt.text_encoder,
            tokenizer=pipe_txt.tokenizer,
            unet=pipe_txt.unet,
            controlnet=pipe_txt.controlnet,
            scheduler=pipe_txt.scheduler,
            safety_checker=None,
            feature_extractor=None,
            image_encoder=getattr(pipe_txt, "image_encoder", None),
        )
        pipe_img.set_progress_bar_config(disable=True)

        # ── 6. Performance optimizations ──────────────────────────────────
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)

        pipe_txt.unet = pipe_txt.unet.to(memory_format=torch.channels_last)
        pipe_txt.controlnet = pipe_txt.controlnet.to(memory_format=torch.channels_last)
        pipe_txt.vae = pipe_txt.vae.to(memory_format=torch.channels_last)

        # ── 7. Pre-compute text embeddings ────────────────────────────────
        print("[IPAdapter] Pre-computing text embeddings …")
        do_cfg = cfg.guidance_scale > 1.0
        with torch.inference_mode():
            self._prompt_embeds, self._neg_embeds = pipe_txt.encode_prompt(
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
                self._ip_embeds = pipe_txt.prepare_ip_adapter_image_embeds(
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
        if hasattr(pipe_txt, "image_encoder") and pipe_txt.image_encoder is not None:
            del pipe_txt.image_encoder
            pipe_txt.image_encoder = None
        if (
            hasattr(pipe_txt, "feature_extractor")
            and pipe_txt.feature_extractor is not None
        ):
            del pipe_txt.feature_extractor
            pipe_txt.feature_extractor = None
        pipe_img.image_encoder = None
        pipe_img.feature_extractor = None
        torch.cuda.empty_cache()
        print("[IPAdapter] CLIP encoder unloaded (VRAM freed)")

        self._pipe_txt2img = pipe_txt
        self._pipe_img2img = pipe_img
        self._device = device
        self._dtype = dtype
        self._H, self._W = H, W

        # Pre-allocate pinned buffer for ctrl upload
        self._pinned_buf = torch.empty(
            (1, 3, H, W), dtype=torch.float16, pin_memory=True
        )
        self._transfer_stream = torch.cuda.Stream()

        # Steps and scales
        self._cn_scale = getattr(cfg, "controlnet_conditioning_scale", 1.0)

        # Temporal feedback strength
        self._feedback_strength = getattr(cfg, "temporal_feedback_strength", 0.3)

        # txt2img steps (first frame)
        self._txt2img_steps = max(cfg.num_inference_steps, 4)

        # img2img steps: ensure actual_steps = floor(total * strength) >= 4
        # so ControlNet gets enough chances to guide pose
        desired_actual = 4
        self._img2img_total_steps = max(
            self._txt2img_steps,
            math.ceil(desired_actual / self._feedback_strength),
        )
        actual = int(self._img2img_total_steps * self._feedback_strength)
        print(
            f"[IPAdapter] Temporal feedback: strength={self._feedback_strength}, "
            f"img2img total_steps={self._img2img_total_steps}, "
            f"actual_steps~{actual}"
        )

        # ── 10. Warmup ────────────────────────────────────────────────────
        print(f"[IPAdapter] Warming up …")
        dummy = torch.zeros(1, 3, H, W, dtype=dtype, device=device).to(
            memory_format=torch.channels_last
        )
        with torch.inference_mode():
            # Warmup txt2img
            for _ in range(2):
                pipe_txt(
                    prompt_embeds=self._prompt_embeds,
                    negative_prompt_embeds=self._neg_embeds,
                    image=dummy,
                    ip_adapter_image_embeds=self._ip_embeds,
                    num_inference_steps=self._txt2img_steps,
                    guidance_scale=cfg.guidance_scale,
                    controlnet_conditioning_scale=self._cn_scale,
                    width=W,
                    height=H,
                    output_type="pt",
                )
            # Warmup img2img
            for _ in range(2):
                pipe_img(
                    prompt_embeds=self._prompt_embeds,
                    negative_prompt_embeds=self._neg_embeds,
                    image=dummy,
                    control_image=dummy,
                    ip_adapter_image_embeds=self._ip_embeds,
                    strength=self._feedback_strength,
                    num_inference_steps=self._img2img_total_steps,
                    guidance_scale=cfg.guidance_scale,
                    controlnet_conditioning_scale=self._cn_scale,
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
        pipe_txt = self._pipe_txt2img
        pipe_img = self._pipe_img2img
        torch.set_num_threads(2)
        seed = getattr(cfg, "seed", 42)
        generator = torch.Generator(device=self._device).manual_seed(
            seed if seed >= 0 else torch.randint(0, 2**31, (1,)).item()
        )

        lH, lW = self._H // 8, self._W // 8

        # Fixed noise for first frame
        fixed_latents = torch.randn(
            (1, 4, lH, lW),
            dtype=self._dtype,
            device=self._device,
            generator=generator,
        )

        # Adaptive feedback parameters
        base_strength = self._feedback_strength  # 0.3 = default minimum
        max_strength = cfg.motion_max_strength
        # ctrl_diff thresholds (measured: jitter ~0.005, small move ~0.01, big move ~0.025)
        motion_lo = cfg.motion_lo
        motion_hi = cfg.motion_hi
        # If control map is nearly empty (person left), reset
        pose_empty_threshold = cfg.pose_empty_threshold

        copy_stream = torch.cuda.Stream()
        prev_gpu_frame: torch.Tensor | None = None
        prev_latent: torch.Tensor | None = None  # latent-level feedback
        prev_ctrl_np: np.ndarray | None = None  # for motion detection
        frame_count: int = 0  # for periodic reset
        periodic_reset = cfg.periodic_reset_frames

        # Get scheduler for manual noise injection
        scheduler = pipe_txt.scheduler

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

            # ── Adaptive feedback: measure motion ─────────────────────────
            pose_energy = float(np.abs(ctrl_np).mean())
            if pose_energy < pose_empty_threshold:
                # No person detected → reset, next frame starts fresh
                prev_latent = None
                prev_ctrl_np = None

            # Periodic reset to prevent latent drift
            frame_count += 1
            if periodic_reset > 0 and frame_count % periodic_reset == 0:
                prev_latent = None

            if prev_ctrl_np is not None and prev_latent is not None:
                ctrl_diff = float(
                    np.abs(
                        ctrl_np.astype(np.float32) - prev_ctrl_np.astype(np.float32)
                    ).mean()
                )
                # Linear interpolation: motion_lo → base, motion_hi → max
                t = (ctrl_diff - motion_lo) / (motion_hi - motion_lo)
                t = max(0.0, min(1.0, t))
                adaptive_strength = base_strength + t * (max_strength - base_strength)
            else:
                adaptive_strength = 1.0  # first frame or after reset

            prev_ctrl_np = ctrl_np.copy()

            # ── GPU inference ─────────────────────────────────────────────
            with torch.inference_mode():
                if prev_latent is None or adaptive_strength >= 0.99:
                    # First frame or large motion: txt2img (full freedom)
                    result = pipe_txt(
                        prompt_embeds=self._prompt_embeds,
                        negative_prompt_embeds=self._neg_embeds,
                        image=ctrl_tensor,
                        ip_adapter_image_embeds=self._ip_embeds,
                        num_inference_steps=self._txt2img_steps,
                        guidance_scale=cfg.guidance_scale,
                        controlnet_conditioning_scale=self._cn_scale,
                        width=self._W,
                        height=self._H,
                        latents=fixed_latents.clone(),
                        output_type="latent",
                    )
                    denoised_latent = result.images
                else:
                    # Latent-level feedback: add noise to prev latent directly
                    # (bypasses VAE encode → no color drift)
                    actual_steps = self._txt2img_steps  # 4 steps

                    # Set timesteps for actual_steps
                    scheduler.set_timesteps(actual_steps, device=self._device)
                    timesteps = scheduler.timesteps

                    # Pick how many steps to actually run based on strength
                    # strength=0.3 → run 1-2 steps (light refinement)
                    # strength=0.85 → run 3-4 steps (heavy change)
                    num_denoise_steps = max(1, round(actual_steps * adaptive_strength))
                    # Take only the last N timesteps (lower noise levels)
                    run_timesteps = timesteps[-num_denoise_steps:]
                    start_timestep = run_timesteps[0:1]

                    # Add noise to previous latent at the starting timestep
                    noise = torch.randn_like(prev_latent)
                    noised_latent = scheduler.add_noise(
                        prev_latent, noise, start_timestep
                    )

                    # Manual denoising loop with ControlNet
                    latent = noised_latent
                    do_cfg = cfg.guidance_scale > 1.0

                    # IP-Adapter embeds must be passed as list of tensors
                    if do_cfg:
                        ip_embeds_loop = self._ip_embeds
                        prompt_embeds_cfg = torch.cat(
                            [self._neg_embeds, self._prompt_embeds]
                        )
                    else:
                        ip_embeds_loop = [e[:1] for e in self._ip_embeds]
                        prompt_embeds_cfg = self._prompt_embeds

                    for i, t in enumerate(run_timesteps):
                        latent_input = torch.cat([latent] * 2) if do_cfg else latent
                        latent_input = scheduler.scale_model_input(latent_input, t)
                        ctrl_input = (
                            torch.cat([ctrl_tensor] * 2) if do_cfg else ctrl_tensor
                        )

                        # ControlNet
                        down_samples, mid_sample = pipe_txt.controlnet(
                            latent_input,
                            t,
                            encoder_hidden_states=prompt_embeds_cfg,
                            controlnet_cond=ctrl_input,
                            conditioning_scale=self._cn_scale,
                            return_dict=False,
                        )

                        # UNet with ControlNet residuals + IP-Adapter
                        added_cond_kwargs = {"image_embeds": ip_embeds_loop}
                        noise_pred = pipe_txt.unet(
                            latent_input,
                            t,
                            encoder_hidden_states=prompt_embeds_cfg,
                            down_block_additional_residuals=down_samples,
                            mid_block_additional_residual=mid_sample,
                            added_cond_kwargs=added_cond_kwargs,
                            return_dict=False,
                        )[0]

                        if do_cfg:
                            pred_uncond, pred_text = noise_pred.chunk(2)
                            noise_pred = pred_uncond + cfg.guidance_scale * (
                                pred_text - pred_uncond
                            )

                        latent = scheduler.step(
                            noise_pred, t, latent, return_dict=False
                        )[0]

                    denoised_latent = latent

                # Save latent for next frame (no VAE encode needed)
                prev_latent = denoised_latent.detach()

                # VAE decode for display only
                decoded = pipe_txt.vae.decode(
                    denoised_latent / pipe_txt.vae.config.scaling_factor
                ).sample
                gpu_frame = (decoded / 2 + 0.5).clamp(0, 1)

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
