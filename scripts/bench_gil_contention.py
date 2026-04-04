"""
scripts/bench_gil_contention.py — Measure GIL contention from pose thread.

Compares engine FPS with and without the pose thread running concurrently.
If there's a significant gap, multiprocessing would help.

Usage:
    uv run python scripts/bench_gil_contention.py
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
from src.diffusion_engine_sdturbo_graph import DiffusionEngineSDTurboGraph
from src.pose_extractor import PoseExtractor

N_WARMUP = 30
N_BENCH = 200
W, H = cfg.output_width, cfg.output_height


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


def bench(with_pose_thread: bool) -> float:
    pose_queue = queue.Queue(maxsize=512)  # large — never blocks engine
    out_queue = queue.Queue(maxsize=512)

    capture = VideoCapture(cfg.video_source, width=W, height=H, queue_size=2, loop=True)
    extractor = PoseExtractor(width=W, height=H, detect_hands=False)
    engine = DiffusionEngineSDTurboGraph(
        cfg=cfg, in_queue=pose_queue, out_queue=out_queue
    )

    engine.load()

    stop_event = threading.Event()
    capture.start()

    if with_pose_thread:
        pt = threading.Thread(
            target=pose_worker,
            args=(capture, extractor, pose_queue, stop_event),
            daemon=True,
        )
        pt.start()
    else:
        # Pre-fill queue with dummy frames so engine never blocks
        dummy = np.zeros((3, H, W), dtype=np.float16)
        for _ in range(512):
            pose_queue.put(dummy)

        # Refill thread
        def refill():
            while not stop_event.is_set():
                if pose_queue.qsize() < 64:
                    for _ in range(64):
                        pose_queue.put(dummy)
                time.sleep(0.001)

        rt = threading.Thread(target=refill, daemon=True)
        rt.start()

    engine.start()

    # Warmup
    for _ in range(N_WARMUP):
        try:
            out_queue.get(timeout=2.0)
        except queue.Empty:
            break

    # Benchmark
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
    torch.cuda.empty_cache()

    return collected / elapsed if elapsed > 0 else 0.0


def main():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {W}x{H}, {N_BENCH} frames each\n")

    print("  [1] Engine only (no pose thread, dummy ctrl maps) ...")
    fps_no_pose = bench(with_pose_thread=False)
    print(f"      {fps_no_pose:.1f} FPS\n")

    print("  [2] Engine + pose thread (real MediaPipe) ...")
    fps_with_pose = bench(with_pose_thread=True)
    print(f"      {fps_with_pose:.1f} FPS\n")

    contention = (fps_no_pose - fps_with_pose) / fps_no_pose * 100
    print(f"  GIL contention overhead: {contention:.1f}%")
    if contention > 5:
        print("  -> Multiprocessing would help.")
    else:
        print("  -> GIL contention is negligible; GPU is the bottleneck.")


if __name__ == "__main__":
    main()
