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
import time
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
        choices=[
            "ip_adapter",
            "lcm_graph",
            "sdturbo_graph",
            "sdturbo",
            "t2i",
            "controlnet",
        ],
        default=None,
        help="ip_adapter (~6-8 FPS, best character), lcm_graph (~60 FPS), sdturbo_graph (~63 FPS), sdturbo (~27 FPS), t2i (~27 FPS), controlnet (~20 FPS)",
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
        "--half-body",
        action="store_true",
        help="VTuber mode: upper body only (remove legs from skeleton)",
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
    p.add_argument(
        "--cn-scale",
        type=float,
        default=None,
        help="ControlNet conditioning scale (0.5=weak, 1.5=strong pose guide, default: 1.5)",
    )
    p.add_argument(
        "--ip-scale",
        type=float,
        default=None,
        help="IP-Adapter scale (0.3=light, 0.5=balanced, 0.7=strong character, default: 0.5)",
    )
    p.add_argument(
        "--guidance",
        type=float,
        default=None,
        help="Guidance scale (1.0=CFG-free, 1.5=recommended with IP-Adapter, default: 1.0)",
    )
    p.add_argument(
        "--feedback",
        type=float,
        default=None,
        help="Temporal feedback strength (0.3=strong coherence, 1.0=no feedback, default: 0.3)",
    )
    p.add_argument(
        "--output",
        "-o",
        default=None,
        help="Save output to mp4 file and exit (no GUI). e.g. --output result.mp4",
    )
    p.add_argument(
        "--no-gui",
        action="store_true",
        help="Skip settings GUI, use CLI args and config.py defaults directly",
    )
    return p.parse_args()


def pose_worker(
    capture: VideoCapture,
    extractor: PoseExtractor,
    pose_queue: queue.Queue,
    stop_event: threading.Event,
    skeleton_queue: queue.Queue | None = None,
    source_display_queue: queue.Queue | None = None,
    send_source: bool = False,
) -> None:
    while not stop_event.is_set():
        frame_bgr = capture.read(timeout=0.1)
        if frame_bgr is None:
            continue
        control_map, _ = extractor.process(frame_bgr)
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

        # Send original frame for side-by-side display
        if source_display_queue is not None:
            # Convert BGR to RGB for display
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            if source_display_queue.full():
                try:
                    source_display_queue.get_nowait()
                except queue.Empty:
                    pass
            source_display_queue.put(frame_rgb)
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

    # ── Settings GUI (unless --no-gui or --output) ────────────────────────
    gui_output = None
    if not args.no_gui and args.output is None:
        from src.settings_gui import show_settings_gui

        gui_output = show_settings_gui(cfg)

    # Apply CLI overrides (take priority over GUI settings)
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
    if args.half_body:
        cfg.half_body = True
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
    if args.cn_scale is not None:
        cfg.controlnet_conditioning_scale = args.cn_scale
    if args.ip_scale is not None:
        cfg.ip_adapter_scale = args.ip_scale
    if args.guidance is not None:
        cfg.guidance_scale = max(1.0, min(3.0, args.guidance))
    if args.feedback is not None:
        cfg.temporal_feedback_strength = max(0.0, min(1.0, args.feedback))

    # Resolve output: CLI --output takes priority over GUI setting
    output_path = args.output or gui_output

    # Auto-switch prompt for half-body mode
    if cfg.half_body and args.prompt is None:
        cfg.prompt = cfg.half_body_prompt

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
        loop=False if args.output else True,
    )
    extractor = PoseExtractor(
        width=cfg.capture_width,
        height=cfg.capture_height,
        detect_hands=cfg.detect_hands,
        half_body=cfg.half_body,
    )
    # Select engine backend from config
    if cfg.engine_backend == "ip_adapter":
        from src.diffusion_engine_ip_adapter import DiffusionEngineIPAdapter

        engine = DiffusionEngineIPAdapter(
            cfg=cfg, in_queue=pose_queue, out_queue=out_queue
        )
    elif cfg.engine_backend == "lcm_graph":
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

    # Load + warmup models (blocks until ready)
    engine.load()

    stop_event = threading.Event()
    capture.start()

    source_display_queue = queue.Queue(maxsize=2)

    pose_thread = threading.Thread(
        target=pose_worker,
        args=(
            capture,
            extractor,
            pose_queue,
            stop_event,
            skeleton_queue,
            source_display_queue,
        ),
        kwargs={"send_source": getattr(cfg, "img2img_input", "noise") == "camera"},
        daemon=True,
        name="pose",
    )
    pose_thread.start()
    engine.start()

    # ── Headless mode: save to mp4 and exit ───────────────────────────────
    if output_path:
        src_fps = capture.fps
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            output_path,
            fourcc,
            src_fps,
            (cfg.output_width, cfg.output_height),
        )
        print(f"\n[Main] Recording to {output_path} ({src_fps:.1f} FPS) ...")

        count = 0
        t0 = time.perf_counter()
        try:
            while True:
                try:
                    frame_rgb = out_queue.get(timeout=5)
                except queue.Empty:
                    break
                writer.write(cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR))
                count += 1
                if not capture._running and out_queue.empty():
                    break
        except KeyboardInterrupt:
            print("\n[Main] Interrupted.")
        finally:
            elapsed = time.perf_counter() - t0
            writer.release()
            stop_event.set()

            # Re-encode to H.264 for browser/GitHub compatibility
            h264_path = output_path.rsplit(".", 1)[0] + "_h264.mp4"
            try:
                import subprocess

                subprocess.run(
                    [
                        "ffmpeg",
                        "-y",
                        "-i",
                        output_path,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "fast",
                        "-crf",
                        "18",
                        "-pix_fmt",
                        "yuv420p",
                        "-movflags",
                        "+faststart",
                        h264_path,
                    ],
                    check=True,
                    capture_output=True,
                )
                import shutil

                shutil.move(h264_path, output_path)
                print(f"[Main] Re-encoded to H.264: {output_path}")
            except (FileNotFoundError, subprocess.CalledProcessError):
                print(f"[Main] ffmpeg not found, keeping mp4v: {output_path}")
            engine.stop()
            capture.stop()
            extractor.close()
            print(
                f"[Main] Saved {count} frames in {elapsed:.1f}s "
                f"({count / elapsed:.1f} gen FPS) -> {output_path}"
            )
        return

    # ── GUI mode ──────────────────────────────────────────────────────────
    renderer = Renderer(
        title=cfg.window_title,
        show_fps=cfg.show_fps,
        show_skeleton=cfg.show_skeleton_overlay,
        max_fps=args.max_fps,
    )

    print("\n[Main] Pipeline running -- press 'q' in the window to quit.\n")

    last_skeleton: np.ndarray | None = None
    last_frame: np.ndarray | None = None  # keep last good frame to avoid flicker
    last_source: np.ndarray | None = None  # original source frame

    try:
        while True:
            # Drain source display queue — keep freshest
            try:
                while True:
                    last_source = source_display_queue.get_nowait()
            except queue.Empty:
                pass

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
                    # No new frame — re-show last frame to avoid flicker
                    if last_frame is not None:
                        alive = renderer.show(
                            last_frame,
                            skeleton_rgb=last_skeleton,
                            source_rgb=last_source,
                        )
                        if not alive:
                            break
                    else:
                        if cv2.waitKey(1) & 0xFF == ord("q"):
                            break
                    continue

            frame_rgb = interp.blend(frame_rgb)
            last_frame = frame_rgb

            # Get latest skeleton for overlay (HWC uint8 from skeleton_queue)
            try:
                while True:
                    last_skeleton = skeleton_queue.get_nowait()
            except queue.Empty:
                pass

            alive = renderer.show(
                frame_rgb, skeleton_rgb=last_skeleton, source_rgb=last_source
            )
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
