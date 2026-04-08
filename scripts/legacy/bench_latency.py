"""
scripts/bench_latency.py — End-to-end latency measurement (pose → display).

Records wall-clock time when each ctrl map enters the pose queue and when
each output frame exits the engine queue, then matches them by index.

Usage:
    uv run python scripts/bench_latency.py
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
from src.diffusion_engine_lcm_graph import DiffusionEngineLCMGraph, ANIME_MODEL_ID
from src.pose_extractor import PoseExtractor

N_WARMUP = 30
N_BENCH = 200
W, H = cfg.output_width, cfg.output_height


def main():
    pose_queue = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=4)

    capture = VideoCapture(cfg.video_source, width=W, height=H, queue_size=2, loop=True)
    extractor = PoseExtractor(width=W, height=H, detect_hands=False)
    engine = DiffusionEngineLCMGraph(
        cfg=cfg,
        in_queue=pose_queue,
        out_queue=out_queue,
        model_id=getattr(cfg, "lcm_model_id", None) or ANIME_MODEL_ID,
    )

    engine.load()

    pose_times = []
    stop_event = threading.Event()

    def pose_worker():
        while not stop_event.is_set():
            frame_bgr = capture.read(timeout=0.1)
            if frame_bgr is None:
                continue
            ctrl_map, _ = extractor.process(frame_bgr)
            ctrl = extractor.preprocess(ctrl_map)
            t = time.perf_counter()
            if pose_queue.full():
                try:
                    pose_queue.get_nowait()
                except queue.Empty:
                    pass
            pose_queue.put(ctrl)
            pose_times.append(t)

    capture.start()
    pt = threading.Thread(target=pose_worker, daemon=True)
    pt.start()
    engine.start()

    # Warmup
    for _ in range(N_WARMUP):
        try:
            out_queue.get(timeout=2.0)
        except queue.Empty:
            break

    pose_times.clear()

    # Collect output timestamps
    out_times = []
    for _ in range(N_BENCH):
        try:
            out_queue.get(timeout=2.0)
            out_times.append(time.perf_counter())
        except queue.Empty:
            break

    stop_event.set()
    engine.stop()
    capture.stop()
    extractor.close()
    torch.cuda.empty_cache()

    if not out_times or not pose_times:
        print("Not enough samples.")
        return

    # Match each output to the most recent pose that preceded it
    latencies = []
    pi = 0
    for t_out in out_times:
        while pi < len(pose_times) - 1 and pose_times[pi + 1] <= t_out:
            pi += 1
        if pi < len(pose_times) and pose_times[pi] <= t_out:
            latencies.append((t_out - pose_times[pi]) * 1000)

    if not latencies:
        print("No latency samples matched.")
        return

    s = sorted(latencies)
    duration = out_times[-1] - out_times[0]
    print(f"\nGPU: {torch.cuda.get_device_name(0)}")
    print(f"Resolution: {W}x{H}, {len(latencies)} samples\n")
    print(f"  End-to-end latency (pose submission -> output received):")
    print(f"    avg : {sum(s) / len(s):.1f} ms")
    print(f"    p50 : {s[len(s) // 2]:.1f} ms")
    print(f"    p95 : {s[int(len(s) * 0.95)]:.1f} ms")
    print(f"    p99 : {s[int(len(s) * 0.99)]:.1f} ms")
    print(f"    min : {s[0]:.1f} ms")
    print(f"    max : {s[-1]:.1f} ms")
    print(f"\n  Output FPS : {len(out_times) / duration:.1f}")
    print(f"  Pose FPS   : {len(pose_times) / duration:.1f}")


if __name__ == "__main__":
    main()
