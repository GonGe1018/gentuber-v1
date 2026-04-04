"""
main.py -- Pipeline orchestration for realtime-live2d.

Thread layout
─────────────
  [VideoCapture thread]  -- reads video/webcam into ring buffer
        |
  [Pose thread]          -- MediaPipe pose extraction (CPU)
        |  RGB control maps  ->  pose_queue
  [DiffusionEngine thread] -- LCM+ControlNet+TAESD on GPU
        |  RGB frames  ->  out_queue
  [Main thread]          -- interpolation + OpenCV display

Usage
-----
    uv run --no-sync python main.py
    uv run --no-sync python main.py --source 0          # webcam
    uv run --no-sync python main.py --steps 2           # better quality
    uv run --no-sync python main.py --source video.mp4 --steps 1
"""

import argparse
import queue
import sys
import threading
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import cfg
from src.capture import VideoCapture
from src.diffusion_engine import DiffusionEngine
from src.diffusion_engine_t2i import DiffusionEngineT2I
from src.diffusion_engine_sdturbo import DiffusionEngineSDTurbo
from src.diffusion_engine_sdturbo_graph import DiffusionEngineSDTurboGraph
from src.interpolator import FrameInterpolator
from src.pose_extractor import PoseExtractor
from src.renderer import Renderer


def parse_args():
    p = argparse.ArgumentParser(description="Realtime Live2D pipeline")
    p.add_argument(
        "--source",
        default=None,
        help="Video file path or webcam index (default: config.video_source)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=None,
        help="LCM inference steps 1-4 (default: config.num_inference_steps)",
    )
    p.add_argument("--prompt", default=None, help="Override generation prompt")
    p.add_argument(
        "--no-skeleton", action="store_true", help="Disable skeleton overlay on output"
    )
    p.add_argument(
        "--no-interp", action="store_true", help="Disable temporal frame interpolation"
    )
    p.add_argument(
        "--backend",
        choices=["sdturbo_graph", "sdturbo", "t2i", "controlnet"],
        default=None,
        help="sdturbo_graph (~46 FPS), sdturbo (~24 FPS), t2i (~23 FPS), controlnet (~18 FPS)",
    )
    p.add_argument(
        "--size",
        choices=["256", "384", "512"],
        default=None,
        help="Output resolution: 256 (~26 FPS), 384 (~23 FPS), 512 (~15 FPS)",
    )
    p.add_argument(
        "--no-hands",
        action="store_true",
        help="Skip hand landmark detection (saves ~6ms/frame)",
    )
    p.add_argument(
        "--quality",
        choices=["fast", "balanced", "quality"],
        default=None,
        help=(
            "fast: 256px, no-hands (~80 FPS)  "
            "balanced: 384px (~47 FPS, default)  "
            "quality: 512px (~30 FPS)"
        ),
    )
    return p.parse_args()


def pose_worker(
    capture: VideoCapture,
    extractor: PoseExtractor,
    pose_queue: queue.Queue,
    stop_event: threading.Event,
) -> None:
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


def main() -> None:
    args = parse_args()

    # Apply CLI overrides to config
    if args.source is not None:
        try:
            cfg.video_source = int(args.source)
        except ValueError:
            cfg.video_source = args.source
    if args.steps is not None:
        cfg.num_inference_steps = max(1, min(4, args.steps))
    if args.prompt is not None:
        cfg.prompt = args.prompt
    if args.no_skeleton:
        cfg.show_skeleton_overlay = False
    if args.no_interp:
        cfg.interp_alpha = 1.0
    if args.backend is not None:
        cfg.engine_backend = args.backend
    if args.size is not None:
        s = int(args.size)
        cfg.capture_width = cfg.capture_height = s
        cfg.output_width = cfg.output_height = s
    if args.no_hands:
        cfg.detect_hands = False
    if args.quality is not None:
        presets = {
            "fast": {"size": 256, "detect_hands": False},
            "balanced": {"size": 384, "detect_hands": True},
            "quality": {"size": 512, "detect_hands": True},
        }
        p = presets[args.quality]
        s = p["size"]
        cfg.capture_width = cfg.capture_height = s
        cfg.output_width = cfg.output_height = s
        cfg.detect_hands = p["detect_hands"]

    print("=" * 60)
    print("  Realtime Live2D -- MVP Pipeline")
    print("=" * 60)
    print(f"  Source  : {cfg.video_source}")
    print(f"  Device  : {cfg.device}  dtype={cfg.dtype}")
    print(f"  Steps   : {cfg.num_inference_steps}  guidance={cfg.guidance_scale}")
    print(f"  Prompt  : {cfg.prompt[:60]}...")
    print("=" * 60)

    pose_queue = queue.Queue(maxsize=cfg.pose_queue_size)
    out_queue = queue.Queue(maxsize=cfg.output_queue_size)

    capture = VideoCapture(
        source=cfg.video_source,
        width=cfg.capture_width,
        height=cfg.capture_height,
        queue_size=cfg.capture_queue_size,
        loop=True,
    )
    extractor = PoseExtractor(
        width=cfg.capture_width,
        height=cfg.capture_height,
        detect_hands=cfg.detect_hands,
    )
    # Select engine backend from config
    if cfg.engine_backend == "sdturbo_graph":
        engine = DiffusionEngineSDTurboGraph(
            cfg=cfg, in_queue=pose_queue, out_queue=out_queue
        )
    elif cfg.engine_backend == "sdturbo":
        engine = DiffusionEngineSDTurbo(
            cfg=cfg, in_queue=pose_queue, out_queue=out_queue
        )
    elif cfg.engine_backend == "t2i":
        engine = DiffusionEngineT2I(cfg=cfg, in_queue=pose_queue, out_queue=out_queue)
    else:
        engine = DiffusionEngine(cfg=cfg, in_queue=pose_queue, out_queue=out_queue)
    interp = FrameInterpolator(alpha=cfg.interp_alpha)
    renderer = Renderer(
        title=cfg.window_title,
        show_fps=cfg.show_fps,
        show_skeleton=cfg.show_skeleton_overlay,
    )

    # Load + warmup models (blocks until ready)
    engine.load()

    stop_event = threading.Event()
    capture.start()

    pose_thread = threading.Thread(
        target=pose_worker,
        args=(capture, extractor, pose_queue, stop_event),
        daemon=True,
        name="pose",
    )
    pose_thread.start()
    engine.start()

    print("\n[Main] Pipeline running -- press 'q' in the window to quit.\n")

    last_skeleton: np.ndarray | None = None

    try:
        while True:
            # Drain queue -- always display the freshest generated frame
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
                    if last_skeleton is not None:
                        bgr = cv2.cvtColor(last_skeleton, cv2.COLOR_RGB2BGR)
                        cv2.imshow(cfg.window_title, bgr)
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
        print("\n[Main] Interrupted.")
    finally:
        print("[Main] Shutting down ...")
        stop_event.set()
        engine.stop()
        capture.stop()
        extractor.close()
        renderer.close()
        print("[Main] Done.")


if __name__ == "__main__":
    main()
