"""
scripts/test_graph_engine.py — Benchmark CUDA graph engine vs eager SD-Turbo.

Usage:
    uv run python scripts/test_graph_engine.py
"""

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from config import cfg
from src.capture import VideoCapture
from src.diffusion_engine_sdturbo_graph import DiffusionEngineSDTurboGraph
from src.interpolator import FrameInterpolator
from src.pose_extractor import PoseExtractor

N_FRAMES = 300
OUTPUT = Path("assets/graph_engine_output.mp4")
TEST_WIDTH = 384
TEST_HEIGHT = 384


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
    cfg.capture_width = TEST_WIDTH
    cfg.capture_height = TEST_HEIGHT
    cfg.output_width = TEST_WIDTH
    cfg.output_height = TEST_HEIGHT

    print(f"[GraphTest] {N_FRAMES} frames @ {TEST_WIDTH}x{TEST_HEIGHT}")

    pose_queue = queue.Queue(maxsize=cfg.pose_queue_size)
    out_queue = queue.Queue(maxsize=cfg.output_queue_size)

    capture = VideoCapture(
        cfg.video_source, width=TEST_WIDTH, height=TEST_HEIGHT, queue_size=2, loop=True
    )
    extractor = PoseExtractor(width=TEST_WIDTH, height=TEST_HEIGHT, detect_hands=False)
    engine = DiffusionEngineSDTurboGraph(
        cfg=cfg, in_queue=pose_queue, out_queue=out_queue
    )
    interp = FrameInterpolator(alpha=cfg.interp_alpha)

    engine.load()

    stop_event = threading.Event()
    capture.start()
    pose_thread = threading.Thread(
        target=pose_worker,
        args=(capture, extractor, pose_queue, stop_event),
        daemon=True,
    )
    pose_thread.start()
    engine.start()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, 15, (TEST_WIDTH, TEST_HEIGHT))

    collected = 0
    t_start = time.perf_counter()
    frame_times = []

    print("[GraphTest] Collecting frames ...")
    while collected < N_FRAMES:
        try:
            frame_rgb = out_queue.get(timeout=5.0)
        except queue.Empty:
            print("[GraphTest] Timeout")
            break

        t_now = time.perf_counter()
        frame_times.append(t_now)
        frame_rgb = interp.blend(frame_rgb)
        writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
        collected += 1

        if collected % 30 == 0:
            elapsed = t_now - t_start
            print(f"  frame {collected:3d}/{N_FRAMES}  FPS: {collected / elapsed:.1f}")

    writer.release()
    stop_event.set()
    engine.stop()
    capture.stop()
    extractor.close()

    total = time.perf_counter() - t_start
    avg_fps = collected / total if total > 0 else 0
    if len(frame_times) > 1:
        gaps = [
            frame_times[i + 1] - frame_times[i] for i in range(len(frame_times) - 1)
        ]
        avg_gap_ms = sum(gaps) / len(gaps) * 1000
    else:
        avg_gap_ms = 0

    print(f"\n[GraphTest] Done.")
    print(f"  Frames : {collected}  Total: {total:.1f}s")
    print(f"  Avg FPS: {avg_fps:.1f}  Avg gap: {avg_gap_ms:.1f} ms")
    print(f"  Output : {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
