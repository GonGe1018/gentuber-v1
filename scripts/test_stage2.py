"""
scripts/test_stage2.py — Stage 2: single diffusion inference smoke test.

Loads the LCM + ControlNet + TAESD pipeline, runs one inference on a
skeleton frame extracted from the test video, and saves the result.

Usage:
    uv run python scripts/test_stage2.py
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np
from PIL import Image

from config import cfg
from src.pose_extractor import PoseExtractor


def main() -> None:
    # ── Extract one skeleton frame ────────────────────────────────────────
    cap = cv2.VideoCapture(cfg.video_source)
    ok, frame_bgr = cap.read()
    cap.release()
    if not ok:
        raise RuntimeError(f"Cannot read {cfg.video_source}")

    frame_bgr = cv2.resize(frame_bgr, (cfg.capture_width, cfg.capture_height))
    ext = PoseExtractor(width=cfg.capture_width, height=cfg.capture_height)
    skeleton_rgb, _ = ext.process(frame_bgr)
    ext.close()

    cv2.imwrite(
        "assets/stage2_skeleton_input.png",
        cv2.cvtColor(skeleton_rgb, cv2.COLOR_RGB2BGR),
    )
    print("[Stage2] Skeleton saved → assets/stage2_skeleton_input.png")

    # ── Load pipeline ─────────────────────────────────────────────────────
    import torch
    from diffusers import (
        AutoencoderTiny,
        ControlNetModel,
        LCMScheduler,
        StableDiffusionControlNetPipeline,
    )

    dtype = torch.float16 if cfg.dtype == "float16" else torch.float32

    print("[Stage2] Loading ControlNet …")
    controlnet = ControlNetModel.from_pretrained(
        cfg.controlnet_model_id, torch_dtype=dtype
    )

    print("[Stage2] Loading base pipeline …")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.base_model_id,
        controlnet=controlnet,
        torch_dtype=dtype,
        safety_checker=None,
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)

    print("[Stage2] Loading TAESD …")
    pipe.vae = AutoencoderTiny.from_pretrained(cfg.taesd_model_id, torch_dtype=dtype)

    pipe = pipe.to(cfg.device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    try:
        pipe.enable_xformers_memory_efficient_attention()
    except Exception:
        pass

    # ── Warmup ────────────────────────────────────────────────────────────
    print("[Stage2] Warmup inference …")
    pil_ctrl = Image.fromarray(skeleton_rgb)
    with torch.inference_mode():
        pipe(
            prompt=cfg.prompt,
            image=pil_ctrl,
            num_inference_steps=cfg.num_inference_steps,
            guidance_scale=cfg.guidance_scale,
            width=cfg.output_width,
            height=cfg.output_height,
            output_type="np",
        )

    # ── Timed inference ───────────────────────────────────────────────────
    print("[Stage2] Timed inference …")
    N = 5
    t0 = time.perf_counter()
    for _ in range(N):
        with torch.inference_mode():
            result = pipe(
                prompt=cfg.prompt,
                negative_prompt=cfg.negative_prompt,
                image=pil_ctrl,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                width=cfg.output_width,
                height=cfg.output_height,
                output_type="np",
            )
    elapsed = time.perf_counter() - t0
    avg_ms = elapsed / N * 1000

    out_frame = (result.images[0] * 255).astype(np.uint8)
    cv2.imwrite("assets/stage2_output.png", cv2.cvtColor(out_frame, cv2.COLOR_RGB2BGR))

    print(f"\n[Stage2] Results:")
    print(f"  Steps          : {cfg.num_inference_steps}")
    print(f"  Avg latency    : {avg_ms:.1f} ms/frame")
    print(f"  Throughput     : {1000 / avg_ms:.1f} FPS")
    print(f"  Output saved   → assets/stage2_output.png")


if __name__ == "__main__":
    main()
