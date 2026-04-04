"""
scripts/quality_check.py — Save side-by-side comparison frames.

Produces a video: [input frame | skeleton | generated anime frame]

Usage:
    uv run --no-sync python scripts/quality_check.py
"""

import queue
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2
import numpy as np

from config import cfg
from src.capture import VideoCapture
from src.diffusion_engine_lcm_graph import DiffusionEngineLCMGraph
from src.interpolator import FrameInterpolator
from src.pose_extractor import PoseExtractor

N_FRAMES = 60
OUTPUT = Path("assets/quality_check.mp4")
W, H = 384, 384


def pose_worker(capture, extractor, pose_queue, raw_queue, stop_event):
    while not stop_event.is_set():
        frame_bgr = capture.read(timeout=0.1)
        if frame_bgr is None:
            continue
        control_map, _ = extractor.process(frame_bgr)
        ctrl = extractor.preprocess(control_map)
        item = (frame_bgr.copy(), control_map)
        for q in [pose_queue, raw_queue]:
            if q.full():
                try:
                    q.get_nowait()
                except queue.Empty:
                    pass
        pose_queue.put(ctrl)
        raw_queue.put(item)


def main():
    cfg.capture_width = W
    cfg.capture_height = H
    cfg.output_width = W
    cfg.output_height = H

    pose_queue = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=4)
    raw_queue = queue.Queue(maxsize=8)

    capture = VideoCapture(cfg.video_source, width=W, height=H, queue_size=2, loop=True)
    extractor = PoseExtractor(width=W, height=H)
    engine = DiffusionEngineLCMGraph(cfg=cfg, in_queue=pose_queue, out_queue=out_queue)
    interp = FrameInterpolator(alpha=cfg.interp_alpha)

    engine.load()

    stop_event = threading.Event()
    capture.start()
    pose_thread = threading.Thread(
        target=pose_worker,
        args=(capture, extractor, pose_queue, raw_queue, stop_event),
        daemon=True,
    )
    pose_thread.start()
    engine.start()

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(OUTPUT), fourcc, 10, (W * 3, H))

    collected = 0
    t_start = time.perf_counter()
    print(f"[QCheck] Collecting {N_FRAMES} frames ...")

    while collected < N_FRAMES:
        try:
            frame_rgb = out_queue.get(timeout=5.0)
        except queue.Empty:
            print("[QCheck] Timeout")
            break

        frame_rgb = interp.blend(frame_rgb)

        # Get matching raw+skeleton (best effort)
        raw_bgr = skeleton_rgb = None
        try:
            raw_bgr, skeleton_rgb = raw_queue.get_nowait()
        except queue.Empty:
            pass

        if raw_bgr is None:
            raw_bgr = np.zeros((H, W, 3), dtype=np.uint8)
        if skeleton_rgb is None:
            skeleton_rgb = np.zeros((H, W, 3), dtype=np.uint8)

        skeleton_bgr = cv2.cvtColor(skeleton_rgb, cv2.COLOR_RGB2BGR)
        output_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

        # Labels
        for img, label in [
            (raw_bgr, "Input"),
            (skeleton_bgr, "Skeleton"),
            (output_bgr, "Anime"),
        ]:
            cv2.putText(
                img, label, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2
            )

        side_by_side = np.concatenate([raw_bgr, skeleton_bgr, output_bgr], axis=1)
        writer.write(side_by_side)
        collected += 1

        if collected % 10 == 0:
            elapsed = time.perf_counter() - t_start
            print(f"  frame {collected}/{N_FRAMES}  FPS: {collected / elapsed:.1f}")

    writer.release()
    stop_event.set()
    engine.stop()
    capture.stop()
    extractor.close()

    # Save a single representative PNG
    if collected > 0:
        cap2 = cv2.VideoCapture(str(OUTPUT))
        cap2.set(cv2.CAP_PROP_POS_FRAMES, collected // 2)
        ok, thumb = cap2.read()
        cap2.release()
        if ok:
            cv2.imwrite("assets/quality_check_thumb.png", thumb)
            print(f"  Thumbnail -> assets/quality_check_thumb.png")

    total = time.perf_counter() - t_start
    print(
        f"\n[QCheck] Done. {collected} frames in {total:.1f}s ({collected / total:.1f} FPS)"
    )
    print(f"  Output -> {OUTPUT.resolve()}")


if __name__ == "__main__":
    main()
