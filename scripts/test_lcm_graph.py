"""
scripts/test_lcm_graph.py — Benchmark LCM graph engine (KohakuV2 + LCM-LoRA).

Usage:
    uv run python scripts/test_lcm_graph.py
"""

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import cfg
from src.capture import VideoCapture
from src.diffusion_engine_lcm_graph import DiffusionEngineLCMGraph
from src.interpolator import FrameInterpolator
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


def main():
    pose_queue = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=512)

    capture = VideoCapture(cfg.video_source, width=W, height=H, queue_size=2, loop=True)
    extractor = PoseExtractor(width=W, height=H, detect_hands=False)
    engine = DiffusionEngineLCMGraph(cfg=cfg, in_queue=pose_queue, out_queue=out_queue)

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

    fps = collected / elapsed if elapsed > 0 else 0
    print(f"\n[LCMGraph] {W}x{H}: {fps:.1f} FPS  ({1000 / fps:.1f} ms/frame)")


if __name__ == "__main__":
    main()
