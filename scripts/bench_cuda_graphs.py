"""
scripts/bench_cuda_graphs.py — Test CUDA graph capture on UNet forward pass.

CUDA graphs replay a captured sequence of GPU ops with near-zero CPU overhead,
eliminating Python dispatch cost between kernels.

Usage:
    uv run python scripts/bench_cuda_graphs.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from diffusers import (
    AutoencoderTiny,
    StableDiffusionAdapterPipeline,
    T2IAdapter,
)
from PIL import Image

from config import cfg

N_WARMUP = 10
N_BENCH = 40
W, H = cfg.output_width, cfg.output_height


def load_pipe(dtype=torch.float16):
    adapter = T2IAdapter.from_pretrained(cfg.t2i_adapter_model_id, torch_dtype=dtype)
    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        adapter=adapter,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.vae = AutoencoderTiny.from_pretrained(cfg.taesd_model_id, torch_dtype=dtype)
    pipe = pipe.to(cfg.device)
    pipe.set_progress_bar_config(disable=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)

    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

    try:
        pipe.unet.set_attn_processor(
            __import__("diffusers").models.attention_processor.AttnProcessor2_0()
        )
    except Exception:
        pipe.enable_attention_slicing()

    with torch.inference_mode():
        pe, ne = pipe.encode_prompt(cfg.prompt, cfg.device, 1, False, None)

    return pipe, pe, ne


def bench_baseline(pipe, pe, ne) -> float:
    dummy = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            pipe(
                prompt_embeds=pe,
                negative_prompt_embeds=ne,
                image=dummy,
                num_inference_steps=1,
                guidance_scale=0.0,
                width=W,
                height=H,
                output_type="np",
            )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N_BENCH):
            pipe(
                prompt_embeds=pe,
                negative_prompt_embeds=ne,
                image=dummy,
                num_inference_steps=1,
                guidance_scale=0.0,
                width=W,
                height=H,
                output_type="np",
            )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_BENCH * 1000


def bench_graphed_unet(pipe, pe, ne) -> float:
    """
    Capture UNet forward as a CUDA graph.
    The rest of the pipeline (scheduler, VAE) runs normally.
    """
    device = cfg.device
    dtype = torch.float16

    # Build static inputs matching what the pipeline passes to UNet
    # SD-Turbo 1-step: no CFG, so batch=1
    latent_h = H // 8
    latent_w = W // 8
    static_latents = torch.zeros(1, 4, latent_h, latent_w, dtype=dtype, device=device)
    static_timestep = torch.tensor([999], dtype=torch.long, device=device)
    static_encoder_hs = pe  # (1, seq, 768)

    # Warmup to stabilise cuDNN
    with torch.inference_mode():
        for _ in range(N_WARMUP):
            pipe.unet(static_latents, static_timestep, static_encoder_hs)
    torch.cuda.synchronize()

    # Capture graph
    g = torch.cuda.CUDAGraph()
    with torch.inference_mode():
        with torch.cuda.graph(g):
            static_out = pipe.unet(static_latents, static_timestep, static_encoder_hs)
    torch.cuda.synchronize()
    print(f"  [Graph] Captured UNet graph ({latent_h}x{latent_w} latents)")

    # Benchmark graph replay
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N_BENCH):
            g.replay()
    torch.cuda.synchronize()
    unet_ms = (time.perf_counter() - t0) / N_BENCH * 1000

    # Benchmark UNet without graph for comparison
    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N_BENCH):
            pipe.unet(static_latents, static_timestep, static_encoder_hs)
    torch.cuda.synchronize()
    unet_eager_ms = (time.perf_counter() - t0) / N_BENCH * 1000

    print(f"  UNet eager : {unet_eager_ms:.1f} ms")
    print(
        f"  UNet graph : {unet_ms:.1f} ms  ({(unet_eager_ms - unet_ms) / unet_eager_ms * 100:.1f}% faster)"
    )

    return unet_ms, unet_eager_ms


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {W}x{H}, SD-Turbo 1-step, {N_BENCH} runs\n")

    print("Loading pipeline ...")
    pipe, pe, ne = load_pipe()

    print("\n[1] Baseline (full pipeline, no graph):")
    ms = bench_baseline(pipe, pe, ne)
    print(f"  Full pipeline: {ms:.1f} ms  ({1000 / ms:.1f} FPS)")

    print("\n[2] CUDA graph on UNet only:")
    unet_graph_ms, unet_eager_ms = bench_graphed_unet(pipe, pe, ne)

    overhead_ms = ms - unet_eager_ms
    projected_ms = unet_graph_ms + overhead_ms
    print(f"\n  Pipeline overhead (VAE+scheduler+etc): {overhead_ms:.1f} ms")
    print(
        f"  Projected full pipeline with graph   : {projected_ms:.1f} ms  ({1000 / projected_ms:.1f} FPS)"
    )
    print(
        f"  Potential gain: {ms - projected_ms:.1f} ms  ({(ms - projected_ms) / ms * 100:.1f}%)"
    )


if __name__ == "__main__":
    main()
