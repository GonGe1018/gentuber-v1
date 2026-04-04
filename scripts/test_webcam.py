"""
scripts/test_webcam.py — Live webcam test with display window.

Runs the full pipeline on webcam input and shows the output in real time.
Press 'q' to quit.

Usage:
    uv run python scripts/test_webcam.py
    uv run python scripts/test_webcam.py --cam 1   # second webcam
"""

import argparse
import queue
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import cv2

from config import cfg
from src.capture import VideoCapture
from src.diffusion_engine_sdturbo_graph import DiffusionEngineSDTurboGraph
from src.interpolator import FrameInterpolator
from src.pose_extractor import PoseExtractor
from src.renderer import Renderer


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
    p = argparse.ArgumentParser()
    p.add_argument("--cam", type=int, default=0, help="Webcam index")
    p.add_argument("--size", type=int, default=384, choices=[256, 384, 512])
    args = p.parse_args()

    cfg.video_source = args.cam
    cfg.capture_width = cfg.capture_height = args.size
    cfg.output_width = cfg.output_height = args.size

    print(f"[WebcamTest] cam={args.cam}  size={args.size}x{args.size}")
    print("[WebcamTest] Press 'q' to quit.\n")

    pose_queue = queue.Queue(maxsize=2)
    out_queue = queue.Queue(maxsize=4)

    capture = VideoCapture(
        cfg.video_source, width=args.size, height=args.size, queue_size=2, loop=False
    )
    extractor = PoseExtractor(
        width=args.size, height=args.size, detect_hands=cfg.detect_hands
    )
    engine = DiffusionEngineSDTurboGraph(
        cfg=cfg, in_queue=pose_queue, out_queue=out_queue
    )
    interp = FrameInterpolator(alpha=cfg.interp_alpha)
    renderer = Renderer(
        title="Realtime Live2D — Webcam",
        show_fps=True,
        show_skeleton=cfg.show_skeleton_overlay,
    )

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

    last_skeleton = None

    try:
        while True:
            frame_rgb = None
            try:
                while True:
                    frame_rgb = out_queue.get_nowait()
            except queue.Empty:
                pass

            if frame_rgb is None:
                try:
                    frame_rgb = out_queue.get(timeout=0.05)
                except queue.Empty:
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                    continue

            frame_rgb = interp.blend(frame_rgb)

            try:
                last_skeleton = (
                    pose_queue.queue[-1] if pose_queue.queue else last_skeleton
                )
            except Exception:
                pass

            alive = renderer.show(frame_rgb, skeleton_rgb=last_skeleton)
            if not alive:
                break

    except KeyboardInterrupt:
        pass
    finally:
        stop_event.set()
        engine.stop()
        capture.stop()
        extractor.close()
        renderer.close()
        print("[WebcamTest] Done.")


if __name__ == "__main__":
    main()
