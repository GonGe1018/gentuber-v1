"""
scripts/bench_int8.py — Benchmark INT8 quantized UNet on Blackwell.

Blackwell (sm_120) has INT8 tensor cores. bitsandbytes replaces Linear
layers with INT8 versions, potentially halving UNet compute.

Note: INT8 quantization is incompatible with CUDA graphs (dynamic dispatch),
so this benchmarks eager mode only to measure the potential speedup.

Usage:
    uv run python scripts/bench_int8.py
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

N_WARMUP = 10
N_BENCH = 30
W, H = cfg.output_width, cfg.output_height
lH, lW = H // 8, W // 8
device = cfg.device


def load_pipe(dtype=torch.float16, load_in_8bit=False):
    adapter = T2IAdapter.from_pretrained(
        cfg.t2i_adapter_model_id, torch_dtype=dtype
    ).to(device)

    if load_in_8bit:
        try:
            pipe = StableDiffusionAdapterPipeline.from_pretrained(
                "stabilityai/sd-turbo",
                adapter=adapter,
                torch_dtype=dtype,
                safety_checker=None,
                load_in_8bit=True,
                device_map="auto",
            )
        except Exception:
            # Fallback: load normally then quantize UNet with bitsandbytes
            import bitsandbytes as bnb

            pipe = StableDiffusionAdapterPipeline.from_pretrained(
                "stabilityai/sd-turbo",
                adapter=adapter,
                torch_dtype=dtype,
                safety_checker=None,
            )
            pipe = pipe.to(device)
            # Replace Linear layers in UNet with Int8 versions
            for name, module in pipe.unet.named_modules():
                if (
                    isinstance(module, torch.nn.Linear)
                    and module.weight.shape[0] % 16 == 0
                ):
                    parent = pipe.unet
                    parts = name.split(".")
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    int8_linear = bnb.nn.Linear8bitLt(
                        module.in_features,
                        module.out_features,
                        bias=module.bias is not None,
                        has_fp16_weights=False,
                    )
                    int8_linear.weight = bnb.nn.Int8Params(
                        module.weight.data, requires_grad=False, has_fp16_weights=False
                    )
                    if module.bias is not None:
                        int8_linear.bias = module.bias
                    setattr(parent, parts[-1], int8_linear)
    else:
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            "stabilityai/sd-turbo",
            adapter=adapter,
            torch_dtype=dtype,
            safety_checker=None,
        )
        pipe = pipe.to(device)

    pipe.vae = AutoencoderTiny.from_pretrained(
        cfg.taesd_model_id, torch_dtype=dtype
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)

    if not load_in_8bit:
        pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

    try:
        pipe.unet.set_attn_processor(AttnProcessor2_0())
    except Exception:
        pass

    with torch.inference_mode():
        pe, _ = pipe.encode_prompt(cfg.prompt, device, 1, False, None)

    return pipe, adapter, pe


def bench(pipe, adapter, pe, label) -> float:
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
    print(f"Resolution: {W}x{H}, UNet+adapter only, {N_BENCH} runs\n")

    print("Loading float16 pipeline ...")
    pipe_fp16, adapter_fp16, pe = load_pipe(dtype=torch.float16)
    ms_fp16 = bench(pipe_fp16, adapter_fp16, pe, "float16 (baseline)")

    del pipe_fp16, adapter_fp16
    torch.cuda.empty_cache()

    print("\nLoading INT8 pipeline (bitsandbytes) ...")
    try:
        import bitsandbytes

        pipe_int8, adapter_int8, pe2 = load_pipe(dtype=torch.float16, load_in_8bit=True)
        ms_int8 = bench(pipe_int8, adapter_int8, pe2, "INT8 (bitsandbytes)")
        gain = (ms_fp16 - ms_int8) / ms_fp16 * 100
        print(f"\n  Speedup: {gain:+.1f}%")
    except ImportError:
        print("  bitsandbytes not installed. Run: uv add bitsandbytes")
    except Exception as e:
        print(f"  INT8 failed: {e}")


if __name__ == "__main__":
    main()
