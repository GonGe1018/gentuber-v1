"""
scripts/bench_fp8.py — Benchmark FP8 UNet on Blackwell (sm_120).

RTX 5070 Ti (Blackwell) has native FP8 tensor cores via torch.float8_e4m3fn.
We use torch._scaled_mm (FP8 GEMM) by replacing Linear layers with FP8 versions.

Note: FP8 is incompatible with CUDA graphs (static shapes required for
scaled_mm), so this benchmarks eager mode only.

Usage:
    uv run python scripts/bench_fp8.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import numpy as np
from diffusers import (
    AutoencoderTiny,
    StableDiffusionAdapterPipeline,
    T2IAdapter,
)
from diffusers.models.attention_processor import AttnProcessor2_0

from config import cfg

N_WARMUP = 10
N_BENCH = 30
W, H = cfg.output_width, cfg.output_height
lH, lW = H // 8, W // 8
device = cfg.device


class FP8Linear(nn.Module):
    """
    Drop-in replacement for nn.Linear using FP8 GEMM (torch._scaled_mm).

    cuBLASLt requires:
      - A (input): row-major float8  shape (M, K)
      - B (weight): column-major float8  shape (K, N)
        = weight stored as (N, K) row-major, accessed via .t() (non-contiguous view)
    """

    def __init__(self, linear: nn.Linear):
        super().__init__()
        # Store as (out, in) fp8 row-major — .t() in forward gives column-major (in, out)
        self.register_buffer("weight_fp8", linear.weight.data.to(torch.float8_e4m3fn))
        self.bias = linear.bias
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.register_buffer(
            "scale_a", torch.ones(1, dtype=torch.float32, device=device)
        )
        self.register_buffer(
            "scale_b", torch.ones(1, dtype=torch.float32, device=device)
        )

    def forward(self, x):
        orig_shape = x.shape
        # Ensure row-major contiguous input
        x_2d = x.reshape(-1, self.in_features)
        if not x_2d.is_contiguous():
            x_2d = x_2d.contiguous()
        x_fp8 = x_2d.to(torch.float8_e4m3fn)
        # weight_fp8.t() is (in, out) column-major — exactly what cuBLASLt needs
        out = torch._scaled_mm(
            x_fp8,
            self.weight_fp8.t(),  # non-contiguous column-major view
            scale_a=self.scale_a,
            scale_b=self.scale_b,
            out_dtype=torch.float16,
        )
        if self.bias is not None:
            out = out + self.bias
        return out.reshape(*orig_shape[:-1], self.out_features)


def quantize_unet_fp8(unet):
    """Replace all eligible Linear layers in UNet with FP8 versions."""
    replaced = 0
    for name, module in list(unet.named_modules()):
        if not isinstance(module, nn.Linear):
            continue
        # Skip small layers and non-multiple-of-16 shapes
        if module.in_features % 16 != 0 or module.out_features % 16 != 0:
            continue
        if module.in_features < 64:
            continue
        parent = unet
        parts = name.split(".")
        for p in parts[:-1]:
            parent = getattr(parent, p)
        setattr(parent, parts[-1], FP8Linear(module).to(device))
        replaced += 1
    return replaced


def load_pipe(fp8=False):
    dtype = torch.float16
    adapter = T2IAdapter.from_pretrained(
        cfg.t2i_adapter_model_id, torch_dtype=dtype
    ).to(device)
    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        "stabilityai/sd-turbo", adapter=adapter, torch_dtype=dtype, safety_checker=None
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

    try:
        pipe.unet.set_attn_processor(AttnProcessor2_0())
    except Exception:
        pass

    if fp8:
        n = quantize_unet_fp8(pipe.unet)
        print(f"  Replaced {n} Linear layers with FP8")

    with torch.inference_mode():
        pe, _ = pipe.encode_prompt(cfg.prompt, device, 1, False, None)

    return pipe, adapter, pe


def bench(pipe, adapter, pe, label):
    dtype = torch.float16
    latents = torch.zeros((1, 4, lH, lW), dtype=dtype, device=device).to(
        memory_format=torch.channels_last
    )
    ctrl = torch.zeros((1, 3, H, W), dtype=dtype, device=device).to(
        memory_format=torch.channels_last
    )
    timestep = torch.tensor([999], dtype=torch.long, device=device)

    with torch.inference_mode():
        for _ in range(N_WARMUP):
            a = adapter(ctrl)
            pipe.unet(
                latents,
                timestep,
                pe,
                down_intrablock_additional_residuals=a,
                return_dict=False,
            )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N_BENCH):
            a = adapter(ctrl)
            pipe.unet(
                latents,
                timestep,
                pe,
                down_intrablock_additional_residuals=a,
                return_dict=False,
            )
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_BENCH * 1000
    print(f"  {label:<30}: {ms:6.1f} ms  ({1000 / ms:.1f} FPS)")
    return ms


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    cap = torch.cuda.get_device_capability()
    print(f"Compute capability: sm_{cap[0]}{cap[1]}")
    print(f"FP8 supported: {cap >= (8, 9)}")
    print(f"Resolution: {W}x{H}, UNet+adapter only, {N_BENCH} runs\n")

    if not hasattr(torch, "float8_e4m3fn"):
        print("torch.float8_e4m3fn not available (requires PyTorch >= 2.1)")
        return

    print("Loading float16 baseline ...")
    pipe_fp16, adapter_fp16, pe = load_pipe(fp8=False)
    ms_fp16 = bench(pipe_fp16, adapter_fp16, pe, "float16 (baseline)")

    del pipe_fp16, adapter_fp16
    torch.cuda.empty_cache()

    print("\nLoading FP8 quantized UNet ...")
    try:
        pipe_fp8, adapter_fp8, pe2 = load_pipe(fp8=True)
        ms_fp8 = bench(pipe_fp8, adapter_fp8, pe2, "FP8 (torch._scaled_mm)")
        gain = (ms_fp16 - ms_fp8) / ms_fp16 * 100
        print(f"\n  Speedup: {gain:+.1f}%  ({ms_fp16 - ms_fp8:.1f} ms saved)")
    except Exception as e:
        print(f"  FP8 failed: {e}")


if __name__ == "__main__":
    main()
