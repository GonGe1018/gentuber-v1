"""
scripts/bench_lcm_steps.py — LCM-LoRA quality vs speed at 1/2/4 steps.

Benchmarks KohakuV2 + LCM-LoRA at different step counts to find the
best quality/speed tradeoff.

Usage:
    uv run python scripts/bench_lcm_steps.py
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
    StableDiffusionPipeline,
    T2IAdapter,
)
from diffusers.models.attention_processor import AttnProcessor2_0
from PIL import Image

from config import cfg

N_WARMUP = 6
N_BENCH = 20
W, H = cfg.output_width, cfg.output_height
device = cfg.device
dtype = torch.float16


def load_pipe():
    adapter = T2IAdapter.from_pretrained(
        cfg.t2i_adapter_model_id, torch_dtype=dtype
    ).to(device)

    base = StableDiffusionPipeline.from_pretrained(
        "KBlueLeaf/kohaku-v2.1", torch_dtype=dtype, safety_checker=None
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

    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.enable_flash_sdp(True)

    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)
    adapter = adapter.to(memory_format=torch.channels_last)
    pipe.unet.set_attn_processor(AttnProcessor2_0())

    with torch.inference_mode():
        pe, _ = pipe.encode_prompt(
            cfg.prompt, device, 1, do_classifier_free_guidance=False
        )

    return pipe, adapter, pe


def bench(pipe, adapter, pe, steps: int) -> float:
    dummy_ctrl = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))

    with torch.inference_mode():
        for _ in range(N_WARMUP):
            pipe(
                prompt_embeds=pe,
                image=dummy_ctrl,
                num_inference_steps=steps,
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
                image=dummy_ctrl,
                num_inference_steps=steps,
                guidance_scale=0.0,
                width=W,
                height=H,
                output_type="np",
            )
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / N_BENCH * 1000
    print(f"  {steps} step(s): {ms:6.1f} ms  ({1000 / ms:.1f} FPS)")
    return ms


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"KohakuV2 + LCM-LoRA, {W}x{H}, {N_BENCH} runs\n")
    print("Loading ...")
    pipe, adapter, pe = load_pipe()

    for steps in [1, 2, 4]:
        bench(pipe, adapter, pe, steps)


if __name__ == "__main__":
    main()
