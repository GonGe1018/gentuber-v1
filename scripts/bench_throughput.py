"""
scripts/bench_throughput.py — Pure engine throughput (no disk I/O).

Measures how fast the engine produces frames without any writer bottleneck.

Usage:
    uv run python scripts/bench_throughput.py
"""

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from src.capture import VideoCapture
from src.diffusion_engine_sdturbo_graph import DiffusionEngineSDTurboGraph
from src.pose_extractor import PoseExtractor

N_WARMUP = 30
N_BENCH = 300

SIZES = [256, 384, 512]


def pose_worker(capture, extractor, pose_queue, stop_event):
    while not stop_event.is_set():
        frame_bgr = capture.read(timeout=0.1)
        if frame_bgr is None:
            continue
        ctrl, _ = extractor.process(frame_bgr)
        if pose_queue.full():
            try:
                pose_queue.get_nowait()
            except queue.Empty:
                pass
        pose_queue.put(ctrl)


def bench_size(size: int) -> float:
    cfg.capture_width = cfg.capture_height = size
    cfg.output_width = cfg.output_height = size

    pose_queue = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=512)  # large — never blocks engine

    capture = VideoCapture(
        cfg.video_source, width=size, height=size, queue_size=2, loop=True
    )
    extractor = PoseExtractor(width=size, height=size, detect_hands=False)
    engine = DiffusionEngineSDTurboGraph(
        cfg=cfg, in_queue=pose_queue, out_queue=out_queue
    )

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
            out_queue.get(timeout=2.0)
        except queue.Empty:
            break

    # Benchmark — just drain, no processing
    t0 = time.perf_counter()
    collected = 0
    for _ in range(N_BENCH):
        try:
            out_queue.get(timeout=2.0)
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
    print(f"Backend: sdturbo_graph, {N_BENCH} frames each\n")
    print(f"{'Size':>8}  {'FPS':>8}  {'ms/frame':>10}")
    print("-" * 32)

    for size in SIZES:
        fps = bench_size(size)
        print(f"{size:>8}  {fps:>8.1f}  {1000 / fps:>10.1f}")

    print("-" * 32)
    print("Done.")


if __name__ == "__main__":
    main()
