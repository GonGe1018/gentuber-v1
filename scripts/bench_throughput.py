"""
scripts/bench_throughput.py — Pure engine throughput (no disk I/O).

Measures how fast the engine produces frames without any writer bottleneck.

Usage:
    uv run python scripts/bench_throughput.py
    uv run python scripts/bench_throughput.py --engine lcm_graph
    uv run python scripts/bench_throughput.py --engine sdturbo_graph
    uv run python scripts/bench_throughput.py --engine lcm_graph --size 384
"""

import argparse
import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from src.capture import VideoCapture
from src.pose_extractor import PoseExtractor

N_WARMUP = 30
N_BENCH = 200

SIZES = [256, 384, 512]


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


def bench_size(size: int, engine_name: str) -> float:
    cfg.capture_width = cfg.capture_height = size
    cfg.output_width = cfg.output_height = size

    pose_queue = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=512)

    capture = VideoCapture(
        cfg.video_source, width=size, height=size, queue_size=2, loop=True
    )
    extractor = PoseExtractor(width=size, height=size, detect_hands=False)

    if engine_name == "lcm_graph":
        from src.diffusion_engine_lcm_graph import (
            DiffusionEngineLCMGraph,
            ANIME_MODEL_ID,
        )

        model_id = getattr(cfg, "lcm_model_id", None) or ANIME_MODEL_ID
        engine = DiffusionEngineLCMGraph(
            cfg=cfg, in_queue=pose_queue, out_queue=out_queue, model_id=model_id
        )
        print(f"  model: {model_id}")
    else:
        from src.diffusion_engine_sdturbo_graph import DiffusionEngineSDTurboGraph

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

    import gc
    import torch

    del engine
    gc.collect()
    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    return collected / elapsed if elapsed > 0 else 0.0


def main():
    import subprocess
    import torch

    p = argparse.ArgumentParser()
    p.add_argument(
        "--engine", choices=["lcm_graph", "sdturbo_graph"], default="lcm_graph"
    )
    p.add_argument(
        "--size",
        type=int,
        choices=[256, 384, 512],
        default=None,
        help="Run a single size (used internally by subprocess mode)",
    )
    p.add_argument("--_single", action="store_true", help=argparse.SUPPRESS)
    args = p.parse_args()

    # Single-size mode: actually run the benchmark
    if args._single or args.size is not None:
        sizes = [args.size] if args.size else SIZES
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"Backend: {args.engine}, {N_BENCH} frames each\n")
        print(f"{'Size':>8}  {'FPS':>8}  {'ms/frame':>10}")
        print("-" * 32)
        for size in sizes:
            fps = bench_size(size, args.engine)
            print(f"{size:>8}  {fps:>8.1f}  {1000 / fps:>10.1f}")
        print("-" * 32)
        print("Done.")
        return

    # Multi-size mode: spawn a fresh subprocess per size to avoid CUDA graph OOM
    import sys

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Backend: {args.engine}, {N_BENCH} frames each\n")
    print(f"{'Size':>8}  {'FPS':>8}  {'ms/frame':>10}")
    print("-" * 32)

    for size in SIZES:
        result = subprocess.run(
            [
                sys.executable,
                __file__,
                "--engine",
                args.engine,
                "--size",
                str(size),
                "--_single",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            print(
                f"  [ERROR size={size}] {result.stderr.splitlines()[-1] if result.stderr else 'unknown'}"
            )
            continue
        for line in result.stdout.splitlines():
            stripped = line.strip()
            if stripped.startswith("model:"):
                print(f"  {stripped}")
            elif stripped and stripped[0].isdigit():
                parts = stripped.split()
                if len(parts) == 3:
                    print(f"{parts[0]:>8}  {parts[1]:>8}  {parts[2]:>10}")

    print("-" * 32)
    print("Done.")


if __name__ == "__main__":
    main()
