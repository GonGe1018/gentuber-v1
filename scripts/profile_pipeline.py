"""
scripts/profile_pipeline.py — Per-stage timing breakdown.

Measures: queue-wait / H2D-transfer / inference / D2H-copy
to identify the exact bottleneck.

Usage:
    uv run --no-sync python scripts/profile_pipeline.py
"""

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from config import cfg
from src.capture import VideoCapture
from src.diffusion_engine import DiffusionEngine
from src.pose_extractor import PoseExtractor

N_FRAMES = 40


def pose_worker(capture, extractor, pose_queue, stop_event):
    while not stop_event.is_set():
        frame_bgr = capture.read(timeout=0.1)
        if frame_bgr is None:
            continue
        control_map, _ = extractor.process(frame_bgr)
        if pose_queue.full():
            try:
                pose_queue.get_nowait()
            except queue.Empty:
                pass
        pose_queue.put(control_map)


def main():
    pose_queue = queue.Queue(maxsize=cfg.pose_queue_size)
    out_queue = queue.Queue(maxsize=cfg.output_queue_size)

    capture = VideoCapture(
        cfg.video_source,
        width=cfg.capture_width,
        height=cfg.capture_height,
        queue_size=2,
        loop=True,
    )
    extractor = PoseExtractor(width=cfg.capture_width, height=cfg.capture_height)

    # Load pipeline manually for fine-grained timing
    dtype = torch.float16
    device = cfg.device

    from diffusers import (
        AutoencoderTiny,
        ControlNetModel,
        LCMScheduler,
        StableDiffusionControlNetPipeline,
    )
    from PIL import Image

    print("[Profile] Loading models ...")
    controlnet = ControlNetModel.from_pretrained(
        cfg.controlnet_model_id, torch_dtype=dtype
    )
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        cfg.base_model_id, controlnet=controlnet, torch_dtype=dtype, safety_checker=None
    )
    pipe.scheduler = LCMScheduler.from_config(pipe.scheduler.config)
    pipe.vae = AutoencoderTiny.from_pretrained(cfg.taesd_model_id, torch_dtype=dtype)
    pipe = pipe.to(device)
    pipe.set_progress_bar_config(disable=True)
    pipe.enable_attention_slicing()
    pipe.unet = pipe.unet.to(memory_format=torch.channels_last)
    pipe.controlnet = pipe.controlnet.to(memory_format=torch.channels_last)
    pipe.vae = pipe.vae.to(memory_format=torch.channels_last)

    with torch.inference_mode():
        prompt_embeds, neg_embeds = pipe.encode_prompt(
            prompt=cfg.prompt,
            device=device,
            num_images_per_prompt=1,
            do_classifier_free_guidance=False,
            negative_prompt=None,
        )

    pinned = torch.empty(
        (1, 3, cfg.output_height, cfg.output_width),
        dtype=torch.float16,
        pin_memory=True,
    )
    generator = torch.Generator(device=device).manual_seed(42)

    # Warmup
    dummy = torch.zeros(1, 3, 512, 512, dtype=torch.float16, device=device)
    with torch.inference_mode():
        pipe(
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=neg_embeds,
            image=dummy,
            num_inference_steps=1,
            guidance_scale=1.0,
            width=512,
            height=512,
            output_type="pt",
        )
    torch.cuda.synchronize()

    stop_event = threading.Event()
    capture.start()
    pose_thread = threading.Thread(
        target=pose_worker,
        args=(capture, extractor, pose_queue, stop_event),
        daemon=True,
    )
    pose_thread.start()

    t_wait = t_h2d = t_infer = t_d2h = 0.0
    collected = 0

    print(f"[Profile] Timing {N_FRAMES} frames ...")
    while collected < N_FRAMES:
        # Queue wait
        t0 = time.perf_counter()
        try:
            ctrl = pose_queue.get(timeout=2.0)
        except queue.Empty:
            break
        t1 = time.perf_counter()

        # H2D transfer
        ctrl_np = ctrl.transpose(2, 0, 1).astype(np.float16) / 255.0
        pinned[0].copy_(torch.from_numpy(ctrl_np), non_blocking=False)
        ctrl_gpu = pinned.to(
            device=device, non_blocking=False, memory_format=torch.channels_last
        )
        torch.cuda.synchronize()
        t2 = time.perf_counter()

        # Inference
        with torch.inference_mode():
            result = pipe(
                prompt_embeds=prompt_embeds,
                negative_prompt_embeds=neg_embeds,
                image=ctrl_gpu,
                num_inference_steps=cfg.num_inference_steps,
                guidance_scale=cfg.guidance_scale,
                width=cfg.output_width,
                height=cfg.output_height,
                generator=generator,
                output_type="pt",
            )
        torch.cuda.synchronize()
        t3 = time.perf_counter()

        # D2H copy
        frame = (
            result.images[0].permute(1, 2, 0).cpu().float().numpy().clip(0, 1) * 255
        ).astype(np.uint8)
        torch.cuda.synchronize()
        t4 = time.perf_counter()

        t_wait += t1 - t0
        t_h2d += t2 - t1
        t_infer += t3 - t2
        t_d2h += t4 - t3
        collected += 1

    stop_event.set()
    capture.stop()
    extractor.close()

    n = max(collected, 1)
    total_ms = (t_wait + t_h2d + t_infer + t_d2h) / n * 1000
    print(f"\n[Profile] Per-frame breakdown ({collected} frames):")
    print(f"  Queue wait   : {t_wait / n * 1000:6.1f} ms")
    print(f"  H2D transfer : {t_h2d / n * 1000:6.1f} ms")
    print(f"  Inference    : {t_infer / n * 1000:6.1f} ms")
    print(f"  D2H copy     : {t_d2h / n * 1000:6.1f} ms")
    print(f"  Total        : {total_ms:6.1f} ms  ({1000 / total_ms:.1f} FPS)")


if __name__ == "__main__":
    main()
