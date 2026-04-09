"""
scripts/bench_sdturbo.py — Benchmark SD-Turbo + T2I-Adapter vs LCM.

SD-Turbo is a single-step adversarially-trained model that may be faster
than LCM since it doesn't need the LCM scheduler overhead.

Usage:
    uv run python scripts/bench_sdturbo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from diffusers import (
    AutoencoderTiny,
    LCMScheduler,
    StableDiffusionAdapterPipeline,
    T2IAdapter,
)
from PIL import Image

from config import cfg

N_WARMUP = 8
N_BENCH = 30
W, H = 384, 384


def bench(model_id: str, scheduler_lcm: bool, label: str) -> float:
    dtype = torch.float16
    device = cfg.device

    adapter = T2IAdapter.from_pretrained(cfg.t2i_adapter_model_id, torch_dtype=dtype)
    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        model_id,
        adapter=adapter,
        torch_dtype=dtype,
        safety_checker=None,
        variant="fp16" if "turbo" not in model_id else None,
    )
    if scheduler_lcm:
        pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
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
        pipe.unet.set_attn_processor(
            __import__("diffusers").models.attention_processor.AttnProcessor2_0()
        )
    except Exception:
        pipe.enable_attention_slicing()

    with torch.inference_mode():
        pe, ne = pipe.encode_prompt(cfg.prompt, device, 1, False, None)

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
    ms = (time.perf_counter() - t0) / N_BENCH * 1000

    del pipe, adapter
    torch.cuda.empty_cache()
    return ms


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {W}x{H}, 1 step, {N_BENCH} runs\n")

    configs = [
        ("SimianLuo/LCM_Dreamshaper_v7", True, "LCM Dreamshaper v7"),
        ("stabilityai/sd-turbo", False, "SD-Turbo"),
    ]

    for model_id, use_lcm, label in configs:
        print(f"  Testing {label} ...")
        try:
            ms = bench(model_id, use_lcm, label)
            print(f"  {label:<25}: {ms:.1f} ms  ({1000 / ms:.1f} FPS)\n")
        except Exception as e:
            print(f"  {label:<25}: FAILED -- {e}\n")


if __name__ == "__main__":
    main()
