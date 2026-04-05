"""
scripts/profile_worker.py — Per-op timing breakdown of the graph engine hot path.

Measures each step in the worker loop to find the next bottleneck.

Usage:
    uv run python scripts/profile_worker.py
    uv run python scripts/profile_worker.py --engine lcm_graph
    uv run python scripts/profile_worker.py --engine sdturbo_graph
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
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

from config import cfg
from src.diffusion_engine_lcm_graph import ANIME_MODEL_ID
from src.pose_extractor import PoseExtractor

N_WARMUP = 20
N_BENCH = 200
W, H = cfg.output_width, cfg.output_height
lH, lW = H // 8, W // 8
dtype = torch.float16
device = cfg.device


def _setup(pipe, adapter):
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)
    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    adapter = adapter.to(memory_format=torch.channels_last)
    pipe.unet.set_attn_processor(AttnProcessor2_0())


def load_sdturbo():
    adapter = T2IAdapter.from_pretrained(
        cfg.t2i_adapter_model_id, torch_dtype=dtype
    ).to(device)
    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        "stabilityai/sd-turbo", adapter=adapter, torch_dtype=dtype, safety_checker=None
    )
    pipe.vae = AutoencoderTiny.from_pretrained(cfg.taesd_model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _setup(pipe, adapter)
    with torch.inference_mode():
        pe, _ = pipe.encode_prompt(cfg.prompt, device, 1, False, None)
    t = int(pipe.scheduler.timesteps[0].cpu())
    sigma = float(pipe.scheduler.sigmas[0])
    return pipe, adapter, pe, t, sigma


def load_lcm():
    adapter = T2IAdapter.from_pretrained(
        cfg.t2i_adapter_model_id, torch_dtype=dtype
    ).to(device)
    base = StableDiffusionPipeline.from_pretrained(
        ANIME_MODEL_ID, torch_dtype=dtype, safety_checker=None
    )
    base.load_lora_weights("latent-consistency/lcm-lora-sdv1-5")
    base.fuse_lora()
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
    pipe.vae = AutoencoderTiny.from_pretrained(cfg.taesd_model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    _setup(pipe, adapter)
    with torch.inference_mode():
        pe, _ = pipe.encode_prompt(cfg.prompt, device, 1, False, None)
    pipe.scheduler.set_timesteps(1, device=device)
    t = int(pipe.scheduler.timesteps[0].cpu())
    sigma = float(pipe.scheduler.init_noise_sigma)  # 1.0 for LCM
    return pipe, adapter, pe, t, sigma


def bench_op(name, fn, n=N_BENCH):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / n * 1000
    print(f"  {name:<40}: {ms:6.2f} ms")
    return ms


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--engine", choices=["lcm_graph", "sdturbo_graph"], default="lcm_graph"
    )
    args = p.parse_args()

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {W}x{H}  engine: {args.engine}\n")
    print("Loading ...")

    if args.engine == "lcm_graph":
        pipe, adapter, pe, t, sigma = load_lcm()
    else:
        pipe, adapter, pe, t, sigma = load_sdturbo()

    # Static tensors
    static_latents = torch.zeros((1, 4, lH, lW), dtype=dtype, device=device).to(
        memory_format=torch.channels_last
    )
    static_ctrl = torch.zeros((1, 3, H, W), dtype=dtype, device=device).to(
        memory_format=torch.channels_last
    )
    static_timestep = torch.tensor([t], dtype=torch.long, device=device)

    # Build CUDA graph
    print("Building CUDA graph ...")
    for _ in range(12):
        with torch.inference_mode():
            a = adapter(static_ctrl)
            u = pipe.unet(
                static_latents,
                static_timestep,
                pe,
                down_intrablock_additional_residuals=a,
                return_dict=False,
            )[0]
            d = u / pipe.vae.config.scaling_factor
            pipe.vae.decode(d, return_dict=False)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(graph):
            _a = adapter(static_ctrl)
            _u = pipe.unet(
                static_latents,
                static_timestep,
                pe,
                down_intrablock_additional_residuals=_a,
                return_dict=False,
            )[0]
            _d = _u / pipe.vae.config.scaling_factor
            _decoded = pipe.vae.decode(_d, return_dict=False)[0]
    torch.cuda.synchronize()

    # Pre-generate noise ring
    gen = torch.Generator(device=device).manual_seed(42)
    noise_ring = [
        torch.randn((1, 4, lH, lW), dtype=dtype, device=device, generator=gen)
        for _ in range(64)
    ]

    # Pinned buffers
    pinned_ctrl = torch.empty((1, 3, H, W), dtype=torch.float16, pin_memory=True)
    pinned_out = torch.empty((H, W, 3), dtype=torch.float32, pin_memory=True)
    gpu_ctrl = torch.empty(
        (1, 3, H, W), dtype=dtype, device=device, memory_format=torch.channels_last
    )
    copy_stream = torch.cuda.Stream()
    xfer_stream = torch.cuda.Stream()

    dummy_ctrl_np = np.zeros((H, W, 3), dtype=np.uint8)
    extractor = PoseExtractor(width=W, height=H, detect_hands=False)

    print(f"\nPer-op breakdown ({N_BENCH} iterations each):\n")

    bench_op(
        "extractor.preprocess(ctrl_map)",
        lambda: extractor.preprocess(dummy_ctrl_np),
    )

    np_ctrl = extractor.preprocess(dummy_ctrl_np)
    bench_op(
        "pinned_ctrl[0].copy_(from_numpy)",
        lambda: pinned_ctrl[0].copy_(torch.from_numpy(np_ctrl), non_blocking=False),
    )

    def h2d():
        with torch.cuda.stream(xfer_stream):
            gpu_ctrl.copy_(pinned_ctrl, non_blocking=True)
        torch.cuda.current_stream().wait_stream(xfer_stream)

    bench_op("H2D ctrl transfer (pinned -> GPU)", lambda: h2d())

    bench_op("static_ctrl.copy_(gpu_ctrl)", lambda: static_ctrl.copy_(gpu_ctrl))

    idx = [0]

    def noise_step():
        n = noise_ring[idx[0] % 64]
        idx[0] += 1
        # LCM: sigma==1.0, skip multiply
        if sigma == 1.0:
            static_latents.copy_(n.to(memory_format=torch.channels_last))
        else:
            static_latents.copy_((n * sigma).to(memory_format=torch.channels_last))

    bench_op("noise ring + static_latents.copy_", lambda: noise_step())

    bench_op("graph.replay() [adapter+UNet+VAE]", lambda: graph.replay())

    def d2h():
        with torch.cuda.stream(copy_stream):
            frame_gpu = (_decoded[0].permute(1, 2, 0).float() + 1.0) * 0.5
            pinned_out.copy_(frame_gpu.clamp(0, 1), non_blocking=True)
        torch.cuda.current_stream().wait_stream(copy_stream)

    bench_op("D2H decoded frame (GPU -> pinned CPU)", lambda: d2h())

    bench_op(
        "cv2.convertScaleAbs(pinned_out, alpha=255)",
        lambda: cv2.convertScaleAbs(pinned_out.numpy(), alpha=255),
    )

    print()

    def full_iter():
        np_c = extractor.preprocess(dummy_ctrl_np)
        pinned_ctrl[0].copy_(torch.from_numpy(np_c), non_blocking=False)
        with torch.cuda.stream(xfer_stream):
            gpu_ctrl.copy_(pinned_ctrl, non_blocking=True)
        n = noise_ring[0]
        if sigma == 1.0:
            static_latents.copy_(n.to(memory_format=torch.channels_last))
        else:
            static_latents.copy_((n * sigma).to(memory_format=torch.channels_last))
        torch.cuda.current_stream().wait_stream(xfer_stream)
        static_ctrl.copy_(gpu_ctrl)
        graph.replay()
        with torch.cuda.stream(copy_stream):
            fg = (_decoded[0].permute(1, 2, 0).float() + 1.0) * 0.5
            pinned_out.copy_(fg.nan_to_num(0.0).clamp(0, 1), non_blocking=True)
        torch.cuda.current_stream().wait_stream(copy_stream)
        cv2.convertScaleAbs(pinned_out.numpy(), alpha=255)

    bench_op("FULL hot path (sequential)", lambda: full_iter())


if __name__ == "__main__":
    main()
