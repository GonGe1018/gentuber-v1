"""
scripts/test_stage2_optimized.py — Benchmark diffusion optimizations.

Applies in order:
  1. Pre-computed text embeddings (skip CLIP every frame)
  2. channels_last memory layout
  3. torch.compile on UNet
  4. Step sweep: 1 / 2 / 4 steps

Usage:
    uv run --no-sync python scripts/test_stage2_optimized.py
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
    ControlNetModel,
    LCMScheduler,
    StableDiffusionControlNetPipeline,
)
from PIL import Image

from config import cfg
from src.pose_extractor import PoseExtractor


def bench(
    pipe,
    prompt_embeds,
    neg_embeds,
    pil_ctrl,
    steps: int,
    n: int = 8,
    prompt: str | None = None,
    negative_prompt: str | None = None,
) -> float:
    """Return average ms per inference over n runs."""
    kwargs = dict(
        image=pil_ctrl,
        num_inference_steps=steps,
        guidance_scale=cfg.guidance_scale,
        width=cfg.output_width,
        height=cfg.output_height,
        output_type="np",
    )
    if prompt_embeds is not None:
        kwargs["prompt_embeds"] = prompt_embeds
        kwargs["negative_prompt_embeds"] = neg_embeds
    else:
        kwargs["prompt"] = prompt or cfg.prompt
        kwargs["negative_prompt"] = negative_prompt or cfg.negative_prompt

    # warmup
    with torch.inference_mode():
        pipe(**kwargs)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        with torch.inference_mode():
            pipe(**kwargs)
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n * 1000


def main() -> None:
    dtype = torch.float16
    device = cfg.device

    # ── Extract skeleton ──────────────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.video_source)
    ok, frame_bgr = cap.read()
    cap.release()
    frame_bgr = cv2.resize(frame_bgr, (cfg.capture_width, cfg.capture_height))
    ext = PoseExtractor(width=cfg.capture_width, height=cfg.capture_height)
    skeleton_rgb, _ = ext.process(frame_bgr)
    ext.close()
    pil_ctrl = Image.fromarray(skeleton_rgb)

    # ── Load pipeline ─────────────────────────────────────────────────────
    print("[Opt] Loading models …")
    controlnet = ControlNetModel.from_pretrained(
        cfg.controlnet_model_id, torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.base_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = AutoencoderTiny.from_pretrained(cfg.taesd_model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
        print("[Opt] xformers enabled")
    except Exception:
        print("[Opt] xformers not available")

    # ── Baseline (4 steps, no optimisations) ─────────────────────────────
    ms = bench(
        pipe,
        None,
        None,
        pil_ctrl,
        steps=4,
        prompt=cfg.prompt,
        negative_prompt=cfg.negative_prompt,
    )
    print(f"\nBaseline (4 steps, raw prompt):  {ms:.1f} ms  ({1000 / ms:.1f} FPS)")

    # ── Opt 1: pre-compute text embeddings ───────────────────────────────
    with torch.inference_mode():
        prompt_embeds, neg_embeds = pipe.encode_prompt(
            prompt=cfg.prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=(cfg.guidance_scale > 1.0),
            negative_prompt=cfg.negative_prompt,
        )

    for steps in [4, 2, 1]:
        ms = bench(pipe, prompt_embeds, neg_embeds, pil_ctrl, steps=steps)
        print(
            f"Pre-embed + {steps} step(s):           {ms:.1f} ms  ({1000 / ms:.1f} FPS)"
        )

    # ── Opt 2: channels_last ──────────────────────────────────────────────
    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.controlnet = pipe.controlnet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

    for steps in [4, 2, 1]:
        ms = bench(pipe, prompt_embeds, neg_embeds, pil_ctrl, steps=steps)
        print(
            f"channels_last + {steps} step(s):       {ms:.1f} ms  ({1000 / ms:.1f} FPS)"
        )

    # ── Opt 3: torch.compile ──────────────────────────────────────────────
    print("\n[Opt] Compiling UNet (this takes ~60s first time) …")
    pipe.unet = torch.compile(pipe.unet, mode="reduce-overhead", fullgraph=True)

    for steps in [4, 2, 1]:
        ms = bench(pipe, prompt_embeds, neg_embeds, pil_ctrl, steps=steps, n=5)
        print(
            f"compiled + {steps} step(s):            {ms:.1f} ms  ({1000 / ms:.1f} FPS)"
        )

    # ── Save best output ──────────────────────────────────────────────────
    with torch.inference_mode():
        result = pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            image=pil_ctrl,
            num_inference_steps=2,
            guidance_scale=cfg.guidance_scale,
            width=cfg.output_width,
            height=cfg.output_height,
            output_type="np",
        )
    out = (result.images[0] * 255).astype(np.uint8)
    cv2.imwrite(
        "assets/stage2_optimized_output.png", cv2.cvtColor(out, cv2.COLOR_RGB2BGR)
    )
    print("\nOutput saved → assets/stage2_optimized_output.png")


if __name__ == "__main__":
    main()
