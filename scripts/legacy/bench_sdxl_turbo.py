"""
scripts/bench_sdxl_turbo.py — Benchmark SDXL-Turbo isolated FPS.

SDXL-Turbo is a 2.6B param adversarial model (vs 860M for SD-Turbo).
Better quality but likely slower. Testing before building a full engine.

Note: SDXL-Turbo uses native 512x512 resolution (not 384x384).

Usage:
    uv run python scripts/bench_sdxl_turbo.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch
from diffusers import AutoencoderTiny, StableDiffusionXLPipeline
from PIL import Image

from config import cfg

N_WARMUP = 6
N_BENCH = 20
SIZES = [(384, 384), (512, 512)]
device = cfg.device


def bench(W: int, H: int) -> float:
    dtype = torch.float16
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/sdxl-turbo",
        torch_dtype=dtype,
        variant="fp16",
        safety_checker=None,
    )
    # SDXL-Turbo has its own TAESD-XL
    try:
        taesd_xl = AutoencoderTiny.from_pretrained(
            "madebyollin/taesdxl", torch_dtype=dtype
        )
        pipe.vae = taesd_xl
        print(f"  Using TAESDXL")
    except Exception:
        print(f"  Using full VAE (TAESDXL not available)")

    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)

    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

    try:
        from diffusers.models.attention_processor import AttnProcessor2_0

        pipe.unet.set_attn_processor(AttnProcessor2_0())
        print(f"  SDPA enabled")
    except Exception:
        pass

    with torch.inference_mode():
        pe, ne, pp, np_ = pipe.encode_prompt(
            prompt=cfg.prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
        )

    dummy = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))

    with torch.inference_mode():
        for _ in range(N_WARMUP):
            pipe(
                prompt_embeds=pe,
                negative_prompt_embeds=ne,
                pooled_prompt_embeds=pp,
                negative_pooled_prompt_embeds=np_,
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
                pooled_prompt_embeds=pp,
                negative_pooled_prompt_embeds=np_,
                num_inference_steps=1,
                guidance_scale=0.0,
                width=W,
                height=H,
                output_type="np",
            )
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_BENCH * 1000

    del pipe
    torch.cuda.empty_cache()
    return ms


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"SDXL-Turbo (no T2I-Adapter), {N_BENCH} runs\n")

    for W, H in SIZES:
        print(f"  {W}x{H} ...")
        try:
            ms = bench(W, H)
            print(f"  {W}x{H}: {ms:.1f} ms  ({1000 / ms:.1f} FPS)\n")
        except Exception as e:
            print(f"  {W}x{H}: FAILED -- {e}\n")


if __name__ == "__main__":
    main()
