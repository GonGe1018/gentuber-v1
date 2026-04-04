"""
scripts/bench_torch_compile.py — Test torch.compile on UNet + adapter.

torch.compile (Inductor backend) fuses ops and generates optimised CUDA
kernels. Combined with CUDA graphs it may push past 50 FPS.

Usage:
    uv run python scripts/bench_torch_compile.py
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
from diffusers.models.attention_processor import AttnProcessor2_0

from config import cfg

N_WARMUP = 15
N_BENCH = 40
W, H = cfg.output_width, cfg.output_height


def load_components(dtype=torch.float16):
    device = cfg.device
    adapter = T2IAdapter.from_pretrained(
        cfg.t2i_adapter_model_id, torch_dtype=dtype
    ).to(device)
    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        "stabilityai/sd-turbo",
        adapter=adapter,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.vae = AutoencoderTiny.from_pretrained(cfg.taesd_model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)

    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    adapter = adapter.to(memory_format=torch.channels_last)
    pipe.unet.set_attn_processor(AttnProcessor2_0())

    with torch.inference_mode():
        pe, _ = pipe.encode_prompt(cfg.prompt, device, 1, False, None)

    return pipe, adapter, pe


def bench_unet(pipe, adapter, pe, use_compile: bool) -> float:
    device = cfg.device
    dtype = torch.float16
    lH, lW = H // 8, W // 8

    unet = pipe.unet
    adapter_ = adapter

    if use_compile:
        print("  Compiling UNet + adapter (first call will be slow) ...")
        unet = torch.compile(unet, mode="reduce-overhead", fullgraph=False)
        adapter_ = torch.compile(adapter_, mode="reduce-overhead", fullgraph=False)

    latents = torch.zeros((1, 4, lH, lW), dtype=dtype, device=device).to(
        memory_format=torch.channels_last
    )
    ctrl = torch.zeros((1, 3, H, W), dtype=dtype, device=device).to(
        memory_format=torch.channels_last
    )
    timestep = torch.tensor([999], dtype=torch.long, device=device)

    with torch.inference_mode():
        for i in range(N_WARMUP):
            adapter_state = adapter_(ctrl)
            unet(
                latents,
                timestep,
                pe,
                down_intrablock_additional_residuals=adapter_state,
                return_dict=False,
            )
            if i == 0 and use_compile:
                torch.cuda.synchronize()
                print("  First compiled call done.")
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N_BENCH):
            adapter_state = adapter_(ctrl)
            unet(
                latents,
                timestep,
                pe,
                down_intrablock_additional_residuals=adapter_state,
                return_dict=False,
            )
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / N_BENCH * 1000


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {W}x{H}, {N_BENCH} runs\n")

    print("Loading pipeline ...")
    pipe, adapter, pe = load_components()

    print("\n[1] Eager UNet + adapter:")
    ms_eager = bench_unet(pipe, adapter, pe, use_compile=False)
    print(f"  {ms_eager:.1f} ms  ({1000 / ms_eager:.1f} FPS)")

    print("\n[2] torch.compile UNet + adapter (mode=reduce-overhead):")
    ms_compile = bench_unet(pipe, adapter, pe, use_compile=True)
    print(f"  {ms_compile:.1f} ms  ({1000 / ms_compile:.1f} FPS)")

    gain = (ms_eager - ms_compile) / ms_eager * 100
    print(
        f"\n  Speedup: {gain:+.1f}%  ({ms_eager - ms_compile:.1f} ms saved per frame)"
    )


if __name__ == "__main__":
    main()
