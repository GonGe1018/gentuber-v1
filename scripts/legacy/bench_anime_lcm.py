"""
scripts/bench_anime_lcm.py — Benchmark anime SD1.5 + LCM-LoRA + T2I-Adapter.

SD-Turbo produces generic output. Anime-specific SD1.5 models (KohakuV2,
Anything V5) with LCM-LoRA give much better anime style at similar speed.

Tests: KohakuV2 + LCM-LoRA vs SD-Turbo (isolated UNet+adapter FPS)

Usage:
    uv run python scripts/bench_anime_lcm.py
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
from diffusers.models.attention_processor import AttnProcessor2_0

from config import cfg

N_WARMUP = 10
N_BENCH = 30
W, H = cfg.output_width, cfg.output_height
lH, lW = H // 8, W // 8
device = cfg.device
dtype = torch.float16

ANIME_MODELS = [
    ("KBlueLeaf/kohaku-v2.1", True, "KohakuV2 + LCM-LoRA"),
    ("Lykon/dreamshaper-8", True, "DreamShaper8 + LCM-LoRA"),
    ("stabilityai/sd-turbo", False, "SD-Turbo (baseline)"),
]
LCM_LORA_ID = "latent-consistency/lcm-lora-sdv1-5"


def load_pipe(model_id: str, use_lcm_lora: bool):
    adapter = T2IAdapter.from_pretrained(
        cfg.t2i_adapter_model_id, torch_dtype=dtype
    ).to(device)

    if use_lcm_lora:
        # Load LoRA via StableDiffusionPipeline (which supports load_lora_weights),
        # fuse it into the UNet, then hand the UNet to the adapter pipeline.
        from diffusers import StableDiffusionPipeline

        base = StableDiffusionPipeline.from_pretrained(
            model_id, torch_dtype=dtype, safety_checker=None
        )
        base.load_lora_weights(LCM_LORA_ID)
        base.fuse_lora()
        unet = base.unet
        pipe = StableDiffusionAdapterPipeline(
            vae=base.vae,
            text_encoder=base.text_encoder,
            tokenizer=base.tokenizer,
            unet=unet,
            adapter=adapter,
            scheduler=LCMScheduler.from_config(base.scheduler.config),
            safety_checker=None,
            feature_extractor=None,
            requires_safety_checker=False,
        )
        del base
    else:
        pipe = StableDiffusionAdapterPipeline.from_pretrained(
            model_id,
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

    try:
        pipe.unet.set_attn_processor(AttnProcessor2_0())
    except Exception:
        pass

    with torch.inference_mode():
        pe, ne = pipe.encode_prompt(
            cfg.prompt,
            device,
            1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
        )

    return pipe, adapter, pe, ne


def bench_unet(pipe, adapter, pe, label: str) -> float:
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
    print(f"  {label:<35}: {ms:6.1f} ms  ({1000 / ms:.1f} FPS)")
    return ms


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {W}x{H}, UNet+adapter only, {N_BENCH} runs\n")

    for model_id, use_lcm, label in ANIME_MODELS:
        print(f"  Loading {label} ...")
        try:
            pipe, adapter, pe, ne = load_pipe(model_id, use_lcm)
            bench_unet(pipe, adapter, pe, label)
            del pipe, adapter
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"  {label}: FAILED -- {e}")
        print()


if __name__ == "__main__":
    main()
