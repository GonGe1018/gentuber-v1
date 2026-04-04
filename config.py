"""
Central configuration for the realtime-live2d pipeline.
Edit values here to tune performance vs quality trade-offs.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Input: int for webcam index (e.g. 0), str for video file path
    video_source: str = "assets/test_input.mp4"

    # Resolution presets:
    #   256x256 -> ~26 FPS  (fast, lower quality)
    #   384x384 -> ~23 FPS  (recommended balance)
    #   512x512 -> ~15 FPS  (highest quality)
    capture_width: int = 384
    capture_height: int = 384
    output_width: int = 384
    output_height: int = 384

    # Diffusion backend:
    #   "sdturbo_graph" -- SD-Turbo + T2I-Adapter + CUDA graph (~46 FPS @ 384, recommended)
    #   "sdturbo"       -- SD-Turbo + T2I-Adapter eager (~24 FPS @ 384)
    #   "t2i"           -- LCM + T2I-Adapter (~23 FPS @ 384)
    #   "controlnet"    -- LCM + ControlNet  (~18 FPS @ 384)
    engine_backend: str = "sdturbo_graph"

    # Model IDs
    base_model_id: str = "SimianLuo/LCM_Dreamshaper_v7"
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_openpose"
    t2i_adapter_model_id: str = "TencentARC/t2iadapter_openpose_sd14v1"
    taesd_model_id: str = "madebyollin/taesd"

    # LCM inference steps
    #   1 step  -> ~15 FPS end-to-end on RTX 5070 Ti  (lower quality)
    #   2 steps -> ~9  FPS                             (better quality)
    #   4 steps -> ~5  FPS                             (best quality)
    num_inference_steps: int = 1
    guidance_scale: float = 1.0  # LCM works best at 1.0 (CFG-free)

    prompt: str = (
        "anime girl, full body, colorful outfit, white background, "
        "high quality, detailed, 2d illustration"
    )
    negative_prompt: str = "blurry, low quality, realistic, 3d, photo"

    # Pipeline queue depths (keep small to minimise latency)
    capture_queue_size: int = 2
    pose_queue_size: int = 2
    output_queue_size: int = 4

    # Pose extraction
    # detect_hands=False saves ~6ms/frame (hand model skipped)
    detect_hands: bool = True
    #   0.0 = no smoothing (raw output)
    #   0.3 = recommended (reduces flicker without adding lag)
    #   1.0 = always show latest frame
    interp_alpha: float = 0.3

    # Hardware
    device: str = "cuda"  # "cuda" | "cpu" | "mps"
    dtype: str = "float16"  # "float16" | "bfloat16" | "float32"

    # Display
    show_skeleton_overlay: bool = True
    show_fps: bool = True
    window_title: str = "Realtime Live2D"


cfg = Config()
