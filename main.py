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
import os
import queue
import sys
import threading
import warnings
from pathlib import Path

# Suppress noisy third-party warnings before any imports
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*local_dir_use_symlinks.*")
warnings.filterwarnings("ignore", message=".*safety_checker.*")
warnings.filterwarnings("ignore", message=".*decode_latents.*")

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from config import cfg
from src.capture import VideoCapture
from src.diffusion_engine import DiffusionEngine
from src.diffusion_engine_t2i import DiffusionEngineT2I
from src.diffusion_engine_sdturbo import DiffusionEngineSDTurbo
from src.diffusion_engine_sdturbo_graph import DiffusionEngineSDTurboGraph
from src.diffusion_engine_lcm_graph import DiffusionEngineLCMGraph
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
    p.add_argument("--negative-prompt", default=None, help="Override negative prompt")
    p.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Noise seed for reproducible output (-1 = random)",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Anime model for lcm_graph backend (e.g. 'KBlueLeaf/kohaku-v2.1', 'Lykon/dreamshaper-8')",
    )
    p.add_argument(
        "--no-skeleton", action="store_true", help="Disable skeleton overlay on output"
    )
    p.add_argument(
        "--no-interp", action="store_true", help="Disable temporal frame interpolation"
    )
    p.add_argument(
        "--backend",
        choices=["lcm_graph", "sdturbo_graph", "sdturbo", "t2i", "controlnet"],
        default=None,
        help="lcm_graph (~60 FPS, best quality), sdturbo_graph (~63 FPS), sdturbo (~27 FPS), t2i (~27 FPS), controlnet (~20 FPS)",
    )
    p.add_argument(
        "--size",
        choices=["256", "384", "512"],
        default=None,
        help="Output resolution: 256 (~103 FPS), 384 (~60 FPS), 512 (~37 FPS)",
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
            "fast: 256px, no-hands (~124 FPS)  "
            "balanced: 384px (~73 FPS, default)  "
            "quality: 512px (~49 FPS)"
        ),
    )
    p.add_argument(
        "--max-fps",
        type=float,
        default=60.0,
        help="Cap display refresh rate (default: 60, 0=uncapped)",
    )
    p.add_argument(
        "--temporal",
        type=float,
        default=None,
        help="Temporal latent blending (0.0=frozen, 1.0=no coherence, default: 0.5)",
    )
    p.add_argument(
        "--strength",
        type=float,
        default=None,
        help="img2img feedback strength (0.0=frozen, 1.0=no feedback, default: 0.5)",
    )
    p.add_argument(
        "--reference",
        default=None,
        help="Reference character image for img2img (default: config.reference_image)",
    )
    return p.parse_args()


def pose_worker(
    capture: VideoCapture,
    extractor: PoseExtractor,
    pose_queue: queue.Queue,
    stop_event: threading.Event,
    skeleton_queue: queue.Queue | None = None,
    send_source: bool = False,
) -> None:
    while not stop_event.is_set():
        frame_bgr = capture.read(timeout=0.1)
        if frame_bgr is None:
            continue
        control_map, _ = extractor.process(frame_bgr)
        # Pre-process here (pose thread has spare capacity) to keep
        # the diffusion engine hot path free of numpy overhead
        ctrl_preprocessed = extractor.preprocess(control_map)

        if send_source:
            source_preprocessed = extractor.preprocess_source(frame_bgr)
            item = (ctrl_preprocessed, source_preprocessed)
        else:
            item = ctrl_preprocessed

        if pose_queue.full():
            try:
                pose_queue.get_nowait()
            except queue.Empty:
                pass
        pose_queue.put(item)
        # Keep latest raw HWC uint8 for skeleton overlay display
        if skeleton_queue is not None:
            if skeleton_queue.full():
                try:
                    skeleton_queue.get_nowait()
                except queue.Empty:
                    pass
            skeleton_queue.put(control_map)


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
    if args.negative_prompt is not None:
        cfg.negative_prompt = args.negative_prompt
    if args.seed is not None:
        cfg.seed = args.seed
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

    if args.model is not None:
        cfg.lcm_model_id = args.model
    if args.temporal is not None:
        cfg.temporal_blend = max(0.0, min(1.0, args.temporal))
    if args.strength is not None:
        cfg.img2img_strength = max(0.0, min(1.0, args.strength))
    if args.reference is not None:
        cfg.reference_image = args.reference
        cfg.img2img_input = "reference"

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
    skeleton_queue = queue.Queue(maxsize=2)  # HWC uint8 for display overlay

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
    if cfg.engine_backend == "lcm_graph":
        from src.diffusion_engine_lcm_graph import ANIME_MODEL_ID

        engine = DiffusionEngineLCMGraph(
            cfg=cfg,
            in_queue=pose_queue,
            out_queue=out_queue,
            model_id=getattr(cfg, "lcm_model_id", None) or ANIME_MODEL_ID,
        )
    elif cfg.engine_backend == "sdturbo_graph":
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
        max_fps=args.max_fps,
    )

    # Load + warmup models (blocks until ready)
    engine.load()

    stop_event = threading.Event()
    capture.start()

    pose_thread = threading.Thread(
        target=pose_worker,
        args=(capture, extractor, pose_queue, stop_event, skeleton_queue),
        kwargs={"send_source": getattr(cfg, "img2img_input", "noise") == "camera"},
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

            # Get latest skeleton for overlay (HWC uint8 from skeleton_queue)
            try:
                while True:
                    last_skeleton = skeleton_queue.get_nowait()
            except queue.Empty:
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
