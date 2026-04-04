"""
scripts/bench_all.py — Comprehensive benchmark across all config combinations.

Tests: backend x resolution x steps
Outputs a summary table.

Usage:
    uv run python scripts/bench_all.py
"""

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from config import Config
from src.capture import VideoCapture
from src.interpolator import FrameInterpolator
from src.pose_extractor import PoseExtractor

N_WARMUP = 20
N_BENCH = 60


def pose_worker(capture, extractor, pose_queue, stop_event):
    while not stop_event.is_set():
        frame_bgr = capture.read(timeout=0.1)
        if frame_bgr is None:
            continue
        ctrl_map, _ = extractor.process(frame_bgr)
        ctrl = extractor.preprocess(ctrl_map)
        if pose_queue.full():
            try:
                pose_queue.get_nowait()
            except queue.Empty:
                pass
        pose_queue.put(ctrl)


def run_bench(backend: str, size: int, steps: int) -> float:
    """Returns steady-state FPS (after warmup frames)."""
    cfg = Config()
    cfg.engine_backend = backend
    cfg.capture_width = cfg.capture_height = size
    cfg.output_width = cfg.output_height = size
    cfg.num_inference_steps = steps
    cfg.detect_hands = False  # consistent baseline

    pose_queue = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=4)

    capture = VideoCapture(
        cfg.video_source, width=size, height=size, queue_size=2, loop=True
    )
    extractor = PoseExtractor(width=size, height=size, detect_hands=False)

    if backend == "sdturbo_graph":
        from src.diffusion_engine_sdturbo_graph import DiffusionEngineSDTurboGraph

        engine = DiffusionEngineSDTurboGraph(
            cfg=cfg, in_queue=pose_queue, out_queue=out_queue
        )
    elif backend == "sdturbo":
        from src.diffusion_engine_sdturbo import DiffusionEngineSDTurbo

        engine = DiffusionEngineSDTurbo(
            cfg=cfg, in_queue=pose_queue, out_queue=out_queue
        )
    elif backend == "t2i":
        from src.diffusion_engine_t2i import DiffusionEngineT2I

        engine = DiffusionEngineT2I(cfg=cfg, in_queue=pose_queue, out_queue=out_queue)
    else:
        from src.diffusion_engine import DiffusionEngine

        engine = DiffusionEngine(cfg=cfg, in_queue=pose_queue, out_queue=out_queue)

    engine.load()

    stop_event = threading.Event()
    capture.start()
    pt = threading.Thread(
        target=pose_worker,
        args=(capture, extractor, pose_queue, stop_event),
        daemon=True,
    )
    pt.start()
    engine.start()

    # Warmup
    for _ in range(N_WARMUP):
        try:
            out_queue.get(timeout=5.0)
        except queue.Empty:
            break

    # Benchmark
    t0 = time.perf_counter()
    collected = 0
    for _ in range(N_BENCH):
        try:
            out_queue.get(timeout=5.0)
        except queue.Empty:
            break
        collected += 1
    elapsed = time.perf_counter() - t0

    stop_event.set()
    engine.stop()
    capture.stop()
    extractor.close()

    import torch

    torch.cuda.empty_cache()

    return collected / elapsed if elapsed > 0 else 0.0


def main():
    import torch

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"{'Backend':<12} {'Size':>6} {'Steps':>6} {'FPS':>8}")
    print("-" * 38)

    configs = [
        ("sdturbo_graph", 256, 1),
        ("sdturbo_graph", 384, 1),
        ("sdturbo_graph", 512, 1),
        ("sdturbo", 256, 1),
        ("sdturbo", 384, 1),
        ("t2i", 384, 1),
        ("t2i", 384, 2),
        ("controlnet", 384, 1),
        ("controlnet", 384, 2),
    ]

    for backend, size, steps in configs:
        fps = run_bench(backend, size, steps)
        print(f"{backend:<12} {size:>6} {steps:>6} {fps:>8.1f}")

    print("-" * 38)
    print("Done.")


if __name__ == "__main__":
    main()
