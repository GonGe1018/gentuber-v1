"""
Central configuration for the realtime-live2d pipeline.
Edit values here to tune performance vs quality trade-offs.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Input: int for webcam index (e.g. 0), str for video file path
    video_source: str = "assets/test_input.mp4"

    # Resolution presets (sdturbo_graph backend, RTX 5070 Ti):
    #   256x256 -> ~124 FPS  (fast, lower quality)
    #   384x384 -> ~73 FPS   (recommended balance)
    #   512x512 -> ~49 FPS   (highest quality)
    capture_width: int = 384
    capture_height: int = 384
    output_width: int = 384
    output_height: int = 384

    # Diffusion backend:
    #   "lcm_graph"     -- KohakuV2 + LCM-LoRA + CUDA graph (~73 FPS @ 384, best quality)
    #   "sdturbo_graph" -- SD-Turbo + T2I-Adapter + CUDA graph (~73 FPS @ 384)
    #   "sdturbo"       -- SD-Turbo + T2I-Adapter eager (~25 FPS @ 384)
    #   "t2i"           -- LCM + T2I-Adapter (~25 FPS @ 384)
    #   "controlnet"    -- LCM + ControlNet  (~19 FPS @ 384)
    engine_backend: str = "lcm_graph"

    # Anime model for lcm_graph backend (any SD1.5-compatible HuggingFace model)
    # Good options:
    #   "KBlueLeaf/kohaku-v2.1"  -- clean anime style (default)
    #   "Lykon/dreamshaper-8"    -- painterly, slightly faster
    lcm_model_id: str = "KBlueLeaf/kohaku-v2.1"

    # Base model for t2i / controlnet backends (LCM-finetuned SD1.5)
    base_model_id: str = "SimianLuo/LCM_Dreamshaper_v7"
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_openpose"
    t2i_adapter_model_id: str = "TencentARC/t2iadapter_openpose_sd14v1"
    taesd_model_id: str = "madebyollin/taesd"

    # LCM inference steps (only applies to t2i / controlnet backends)
    #   1 step  -> ~25 FPS  (lower quality)
    #   2 steps -> ~15 FPS  (better quality)
    #   4 steps -> ~8  FPS  (best quality)
    num_inference_steps: int = 1
    guidance_scale: float = 1.0  # LCM works best at 1.0 (CFG-free)

    prompt: str = (
        "masterpiece, best quality, anime girl, full body, "
        "colorful outfit, white background, 2d illustration, "
        "sharp focus, vibrant colors"
    )
    negative_prompt: str = (
        "lowres, bad anatomy, bad hands, missing fingers, "
        "blurry, low quality, realistic, 3d, photo, watermark"
    )

    # Noise seed for reproducible output (42 = fixed, -1 = random each run)
    seed: int = 42

    # Pipeline queue depths (keep small to minimise latency)
    capture_queue_size: int = 2
    pose_queue_size: int = 2
    output_queue_size: int = 4

    # Pose extraction
    # detect_hands=False saves ~6ms/frame (hand model skipped)
    detect_hands: bool = True

    # Temporal smoothing: blend ratio between prev and current frame
    #   0.0 = no smoothing (raw output)
    #   0.3 = recommended (reduces flicker without adding lag)
    #   1.0 = always show latest frame
    interp_alpha: float = 0.3

    # Temporal latent blending: mix previous denoised latent into next frame's noise
    #   0.0 = fully reuse previous latent (frozen, no variation)
    #   1.0 = fully new noise each frame (current default, no temporal coherence)
    #   0.5 = recommended balance (smooth transitions, still responsive to pose)
    temporal_blend: float = 0.5

    # img2img feedback strength: how much noise to add to previous frame's latent
    #   0.0 = no noise (frozen image, ignores new pose)
    #   1.0 = full noise (no feedback, same as txt2img)
    #   0.5-0.7 = recommended (preserves previous structure, adapts to new pose)
    img2img_strength: float = 0.7

    # Hardware
    device: str = "cuda"  # "cuda" | "cpu" | "mps"
    dtype: str = "float16"  # "float16" | "bfloat16" | "float32"

    # Display
    show_skeleton_overlay: bool = True
    show_fps: bool = True
    window_title: str = "Realtime Live2D"


cfg = Config()
