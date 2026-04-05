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
        self.out_queue = out_queue
        self.model_id = model_id
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
        # LCM 1-step: use t=999 (same as SD-Turbo)
        self._static_timestep = torch.tensor([999], dtype=torch.long, device=device)

        print("[LCMGraph] Warming up for CUDA graph capture ...")
        with torch.inference_mode():
            for _ in range(12):
                a = adapter(self._static_ctrl)
                u = pipe.unet(
                    self._static_latents,
                    self._static_timestep,
                    self._prompt_embeds,
                    down_intrablock_additional_residuals=a,
                    return_dict=False,
                )[0]
                d = u / pipe.vae.config.scaling_factor
                pipe.vae.decode(d, return_dict=False)
        torch.cuda.synchronize()

        print("[LCMGraph] Capturing CUDA graph (adapter+UNet+VAE) ...")
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
                _d = self._static_unet_out / pipe.vae.config.scaling_factor
                self._static_decoded = pipe.vae.decode(_d, return_dict=False)[0]
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

    def _worker(self) -> None:
        cfg = self.cfg
        device = self._device
        dtype = self._dtype
        pipe = self._pipe
        H, W = self._H, self._W
        lH, lW = H // 8, W // 8
        steps = max(1, cfg.num_inference_steps)

        torch.set_num_threads(2)
        generator = torch.Generator(device=device).manual_seed(42)

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

        def upload_ctrl(ctrl_map, pinned, gpu_buf):
            if (
                ctrl_map.dtype == np.float16
                and ctrl_map.ndim == 3
                and ctrl_map.shape[0] == 3
            ):
                pinned[0].copy_(torch.from_numpy(ctrl_map), non_blocking=False)
            else:
                np_ctrl = ctrl_map.transpose(2, 0, 1).astype(np.float16) / 255.0
                pinned[0].copy_(torch.from_numpy(np_ctrl), non_blocking=False)
            with torch.cuda.stream(self._transfer_stream):
                gpu_buf.copy_(pinned, non_blocking=True)

        def get_control_map(last_ctrl):
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
            return ctrl if ctrl is not None else last_ctrl

        # Bootstrap
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
        next_pinned, next_gpu = pinned_B, gpu_B
        next_ctrl_ready = False

        while self._running:
            with torch.inference_mode():
                noise = noise_ring[noise_idx]
                noise_idx = (noise_idx + 1) % NOISE_RING

                if steps == 1:
                    # ── Fast path: CUDA graph ─────────────────────────────
                    self._static_latents.copy_(
                        (noise * sigma_0).to(memory_format=torch.channels_last)
                    )
                    self._static_timestep.fill_(int(timesteps[0]))

                    next_ctrl = get_control_map(last_ctrl)
                    if next_ctrl is not None:
                        upload_ctrl(next_ctrl, next_pinned, next_gpu)
                        next_ctrl_ready = True
                        last_ctrl = next_ctrl

                    self._graph.replay()

                    if next_ctrl_ready:
                        torch.cuda.current_stream().wait_stream(self._transfer_stream)
                        self._static_ctrl.copy_(next_gpu)
                        next_pinned, next_gpu = (
                            (pinned_A, gpu_A)
                            if next_gpu is gpu_B
                            else (pinned_B, gpu_B)
                        )
                        next_ctrl_ready = False

                    with torch.cuda.stream(copy_stream):
                        frame_gpu = (
                            self._static_decoded[0].permute(1, 2, 0).float() + 1.0
                        ) * 0.5
                        pinned_out.copy_(
                            frame_gpu.nan_to_num(0.0).clamp(0, 1), non_blocking=True
                        )

                else:
                    # ── Multi-step: eager (CUDA graph can't capture loops) ─
                    next_ctrl = get_control_map(last_ctrl)
                    if next_ctrl is not None:
                        upload_ctrl(next_ctrl, next_pinned, next_gpu)
                        torch.cuda.current_stream().wait_stream(self._transfer_stream)
                        self._static_ctrl.copy_(next_gpu)
                        next_pinned, next_gpu = (
                            (pinned_A, gpu_A)
                            if next_gpu is gpu_B
                            else (pinned_B, gpu_B)
                        )
                        last_ctrl = next_ctrl

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
                        latents / pipe.vae.config.scaling_factor, return_dict=False
                    )[0]

                    with torch.cuda.stream(copy_stream):
                        frame_gpu = (decoded[0].permute(1, 2, 0).float() + 1.0) * 0.5
                        pinned_out.copy_(
                            frame_gpu.nan_to_num(0.0).clamp(0, 1), non_blocking=True
                        )

            torch.cuda.current_stream().wait_stream(copy_stream)
            frame = (pinned_out.numpy() * 255).astype(np.uint8)

            if self.out_queue.full():
                try:
                    self.out_queue.get_nowait()
                except queue.Empty:
                    pass
            self.out_queue.put(frame)

            if not self._running:
                break
