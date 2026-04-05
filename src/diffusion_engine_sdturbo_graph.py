"""
diffusion_engine_sdturbo_graph.py — SD-Turbo + T2I-Adapter with CUDA graph.

CUDA graph captures the UNet forward pass and replays it with near-zero
Python overhead, giving ~27% speedup over eager mode.

Architecture:
  - Static input tensors updated in-place before each graph replay
  - Manual inference loop (bypasses diffusers pipeline.__call__)
  - T2I-Adapter conditioning injected before UNet via down_intrablock_additional_residuals
"""

import queue
import threading
from typing import Optional

import cv2
import numpy as np
import torch
from diffusers import (
    AutoencoderTiny,
    StableDiffusionAdapterPipeline,
    T2IAdapter,
)
from diffusers.models.attention_processor import AttnProcessor2_0


class DiffusionEngineSDTurboGraph:
    """
    SD-Turbo + T2I-Adapter with CUDA graph on the UNet forward pass.

    Parameters
    ----------
    cfg : config.Config
    in_queue  : queue.Queue
    out_queue : queue.Queue
    """

    def __init__(self, cfg, in_queue: queue.Queue, out_queue: queue.Queue):
        self.cfg = cfg
        self.in_queue = in_queue
        self.out_queue = out_queue
        self._thread: Optional[threading.Thread] = None
        self._running = False

    def load(self) -> None:
        cfg = self.cfg
        dtype = torch.float16 if cfg.dtype == "float16" else torch.float32
        device = cfg.device
        H, W = cfg.output_height, cfg.output_width

        print("[GraphEngine] Loading T2I-Adapter ...")
        self._adapter = T2IAdapter.from_pretrained(
            cfg.t2i_adapter_model_id, torch_dtype=dtype
        ).to(device)

        print("[GraphEngine] Loading SD-Turbo ...")
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            adapter=self._adapter,
            torch_dtype=dtype,
            safety_checker=None,
        )
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
        self._adapter = self._adapter.to(memory_format=torch.channels_last)

        try:
            pipe.unet.set_attn_processor(AttnProcessor2_0())
            print("[GraphEngine] SDPA attention enabled")
        except Exception:
            pipe.enable_attention_slicing()

        self._pipe = pipe
        self._device = device
        self._dtype = dtype
        self._H, self._W = H, W

        # Pre-compute text embeddings
        print("[GraphEngine] Pre-computing text embeddings ...")
        with torch.inference_mode():
            self._prompt_embeds, _ = pipe.encode_prompt(
                prompt=cfg.prompt,
                device=device,
                num_images_per_prompt=1,
                do_classifier_free_guidance=False,
                negative_prompt=None,
            )

        # Pinned memory for H2D transfer
        self._pinned_buf = torch.empty(
            (1, 3, H, W), dtype=torch.float16, pin_memory=True
        )
        self._transfer_stream = torch.cuda.Stream()

        # Build CUDA graph
        self._build_graph()
        print("[GraphEngine] Ready.")

    def _build_graph(self) -> None:
        """Capture adapter + UNet + VAE decode as a single CUDA graph."""
        cfg = self.cfg
        device = self._device
        dtype = self._dtype
        H, W = self._H, self._W
        lH, lW = H // 8, W // 8

        pipe = self._pipe
        adapter = self._adapter

        # Static tensors — updated in-place before each replay
        self._static_latents = torch.zeros(
            (1, 4, lH, lW), dtype=dtype, device=device
        ).to(memory_format=torch.channels_last)
        self._static_ctrl = torch.zeros((1, 3, H, W), dtype=dtype, device=device).to(
            memory_format=torch.channels_last
        )
        self._static_timestep = torch.tensor([999], dtype=torch.long, device=device)

        # Warmup (cuDNN tuning + graph pool allocation)
        print("[GraphEngine] Warming up for CUDA graph capture ...")
        with torch.inference_mode():
            for _ in range(12):
                adapter_state = adapter(self._static_ctrl)
                noise_pred = pipe.unet(
                    self._static_latents,
                    self._static_timestep,
                    self._prompt_embeds,
                    down_intrablock_additional_residuals=adapter_state,
                    return_dict=False,
                )[0]
                # SD-Turbo 1-step: denoised = noise_pred (model predicts x0)
                denoised = noise_pred / pipe.vae.config.scaling_factor
                pipe.vae.decode(denoised, return_dict=False)
        torch.cuda.synchronize()

        # Capture: adapter + UNet + scheduler (trivial) + VAE decode
        print("[GraphEngine] Capturing CUDA graph (UNet + VAE) ...")
        self._graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(self._graph):
                self._adapter_state = adapter(self._static_ctrl)
                self._static_unet_out = pipe.unet(
                    self._static_latents,
                    self._static_timestep,
                    self._prompt_embeds,
                    down_intrablock_additional_residuals=self._adapter_state,
                    return_dict=False,
                )[0]
                # SD-Turbo 1-step: denoised = noise_pred (x0 prediction)
                _denoised = self._static_unet_out / pipe.vae.config.scaling_factor
                self._static_decoded = pipe.vae.decode(_denoised, return_dict=False)[0]
        torch.cuda.synchronize()
        print(f"[GraphEngine] Graph captured (UNet + VAE, {lH}x{lW} latents)")

    def start(self) -> "DiffusionEngineSDTurboGraph":
        self._running = True
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="graph-engine"
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
        device = self._device
        dtype = self._dtype
        pipe = self._pipe
        H, W = self._H, self._W
        lH, lW = H // 8, W // 8

        torch.set_num_threads(2)
        seed = getattr(cfg, "seed", 42)
        generator = torch.Generator(device=device).manual_seed(
            seed if seed >= 0 else torch.randint(0, 2**31, (1,)).item()
        )

        # Pre-generate a ring of noise tensors to avoid randn on the hot path
        NOISE_RING = 64
        noise_ring = [
            torch.randn((1, 4, lH, lW), dtype=dtype, device=device, generator=generator)
            for _ in range(NOISE_RING)
        ]
        noise_idx = 0

        # Double-buffer H2D: while graph replays on frame N, upload frame N+1
        # Two pinned CPU buffers + two GPU staging buffers (ping-pong)
        pinned_A = torch.empty((1, 3, H, W), dtype=torch.float16, pin_memory=True)
        pinned_B = torch.empty((1, 3, H, W), dtype=torch.float16, pin_memory=True)
        gpu_A = torch.empty(
            (1, 3, H, W), dtype=dtype, device=device, memory_format=torch.channels_last
        )
        gpu_B = torch.empty(
            (1, 3, H, W), dtype=dtype, device=device, memory_format=torch.channels_last
        )

        # Async D2H: pinned output buffer
        pinned_out = torch.empty((H, W, 3), dtype=torch.float32, pin_memory=True)
        copy_stream = torch.cuda.Stream()

        t = int(pipe.scheduler.timesteps[0].cpu())
        sigma = float(pipe.scheduler.sigmas[0])

        def upload_ctrl(ctrl_map, pinned, gpu_buf):
            """Upload control map to GPU on transfer_stream.
            Accepts either:
              - uint8 HWC (H,W,3) — raw from pose thread (legacy)
              - float16 CHW (3,H,W) — pre-processed by pose thread (fast path)
            """
            if (
                ctrl_map.dtype == np.float16
                and ctrl_map.ndim == 3
                and ctrl_map.shape[0] == 3
            ):
                # Already preprocessed: CHW float16 in [0,1]
                pinned[0].copy_(torch.from_numpy(ctrl_map), non_blocking=False)
            else:
                # Legacy: HWC uint8 — preprocess here
                np_ctrl = ctrl_map.transpose(2, 0, 1).astype(np.float16) / 255.0
                pinned[0].copy_(torch.from_numpy(np_ctrl), non_blocking=False)
            with torch.cuda.stream(self._transfer_stream):
                gpu_buf.copy_(pinned, non_blocking=True)
            return self._transfer_stream

        def get_control_map(last_ctrl):
            """Return freshest control map. Falls back to last_ctrl if queue empty."""
            ctrl = None
            try:
                while True:
                    item = self.in_queue.get_nowait()
                    if item is None:
                        self._running = False
                        return None
                    ctrl = item
            except queue.Empty:
                pass
            # Reuse last frame if pose thread hasn't produced a new one yet
            return ctrl if ctrl is not None else last_ctrl

        # Bootstrap: wait for first frame (can't reuse nothing)
        first_ctrl = None
        while first_ctrl is None and self._running:
            try:
                first_ctrl = self.in_queue.get(timeout=0.1)
                if first_ctrl is None:
                    self._running = False
                    return
            except queue.Empty:
                continue
        if not self._running:
            return

        upload_ctrl(first_ctrl, pinned_A, gpu_A)
        torch.cuda.current_stream().wait_stream(self._transfer_stream)
        self._static_ctrl.copy_(gpu_A)

        last_ctrl = first_ctrl
        # Ping-pong state
        next_pinned, next_gpu = pinned_B, gpu_B
        next_ctrl_ready = False

        while self._running:
            with torch.inference_mode():
                # ── Prepare noisy latents ─────────────────────────────────
                noise = noise_ring[noise_idx]
                noise_idx = (noise_idx + 1) % NOISE_RING
                self._static_latents.copy_(
                    (noise * sigma).to(memory_format=torch.channels_last)
                )
                # _static_timestep is constant (999) — no fill needed

                # ── Prefetch next control map (reuse last if pose not ready) ─
                next_ctrl = get_control_map(last_ctrl)
                if next_ctrl is not None:
                    upload_ctrl(next_ctrl, next_pinned, next_gpu)
                    next_ctrl_ready = True
                    last_ctrl = next_ctrl

                # ── Replay CUDA graph (adapter + UNet + VAE) ─────────────
                self._graph.replay()

                # ── Swap in prefetched ctrl for next iteration ────────────
                if next_ctrl_ready:
                    torch.cuda.current_stream().wait_stream(self._transfer_stream)
                    self._static_ctrl.copy_(next_gpu)
                    next_pinned, next_gpu = (
                        (pinned_A, gpu_A) if next_gpu is gpu_B else (pinned_B, gpu_B)
                    )
                    next_ctrl_ready = False

                # ── Async D2H ─────────────────────────────────────────────
                with torch.cuda.stream(copy_stream):
                    frame_gpu = (
                        self._static_decoded[0].permute(1, 2, 0).float() + 1.0
                    ) * 0.5
                    # nan_to_num guards against NaN from degenerate inputs
                    pinned_out.copy_(
                        frame_gpu.nan_to_num(0.0).clamp(0, 1), non_blocking=True
                    )

            torch.cuda.current_stream().wait_stream(copy_stream)
            frame = cv2.convertScaleAbs(pinned_out.numpy(), alpha=255)

            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put(frame)

            if not self._running:
                break
