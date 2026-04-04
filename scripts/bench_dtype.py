"""
scripts/bench_dtype.py — Compare float16 vs bfloat16 on current GPU.

Blackwell (sm_120) has native bf16 tensor cores which may be faster.

Usage:
    uv run --no-sync python scripts/bench_dtype.py
"""

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
    T2IAdapter,
)
from PIL import Image

from config import cfg

N = 20
W, H = cfg.output_width, cfg.output_height


def bench_dtype(dtype_str: str) -> float:
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16
    device = cfg.device

    adapter = T2IAdapter.from_pretrained(cfg.t2i_adapter_model_id, torch_dtype=dtype)
    pipe = StableDiffusionAdapterPipeline.from_pretrained(
        cfg.base_model_id, adapter=adapter, torch_dtype=dtype, safety_checker=None
    )
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

    # Warmup
    with torch.inference_mode():
        for _ in range(6):
            pipe(
                prompt_embeds=pe,
                negative_prompt_embeds=ne,
                image=dummy,
                num_inference_steps=1,
                guidance_scale=1.0,
                width=W,
                height=H,
                output_type="np",
            )
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    with torch.inference_mode():
        for _ in range(N):
            pipe(
                prompt_embeds=pe,
                negative_prompt_embeds=ne,
                image=dummy,
                num_inference_steps=1,
                guidance_scale=1.0,
                width=W,
                height=H,
                output_type="np",
            )
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N * 1000

    del pipe, adapter
    torch.cuda.empty_cache()
    return ms


def main():
    print(f"[DTypeBench] GPU: {torch.cuda.get_device_name(0)}")
    print(f"[DTypeBench] Resolution: {W}x{H}, 1 step, {N} runs each\n")

    for dt in ["float16", "bfloat16"]:
        print(f"  Testing {dt} ...")
        ms = bench_dtype(dt)
        print(f"  {dt:10s}: {ms:.1f} ms/frame  ({1000 / ms:.1f} FPS)\n")


if __name__ == "__main__":
    main()
