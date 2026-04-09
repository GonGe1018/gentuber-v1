"""
Central configuration for the gentuber-v1 pipeline.
Edit values here to tune performance vs quality trade-offs.
"""

from dataclasses import dataclass


@dataclass
class Config:
    # Input: int for webcam index (e.g. 0), str for video file path
    video_source: str = "0"

    # Resolution presets (sdturbo_graph backend, RTX 5070 Ti):
    #   256x256 -> ~124 FPS  (fast, lower quality)
    #   384x384 -> ~73 FPS   (recommended balance)
    #   512x512 -> ~49 FPS   (highest quality)
    capture_width: int = 256
    capture_height: int = 256
    output_width: int = 256
    output_height: int = 256

    # Diffusion backend
    engine_backend: str = "ip_adapter"

    # Base model
    lcm_model_id: str = "KBlueLeaf/kohaku-v2.1"
    controlnet_model_id: str = "lllyasviel/control_v11p_sd15_openpose"
    taesd_model_id: str = "madebyollin/taesd"

    # Inference
    num_inference_steps: int = 4
    guidance_scale: float = 1.2

    prompt: str = (
        "1girl, solo, black hair, short hair, bob cut, straight bangs, "
        "white collared shirt, red bow tie, dark brown pleated skirt, "
        "standing, full body, simple background, white background, "
        "flat color, cel shading, anime coloring, clean lineart, "
        "masterpiece, best quality, highres"
    )
    half_body_prompt: str = (
        "1girl, solo, black hair, short hair, bob cut, straight bangs, "
        "white collared shirt, red bow tie, "
        "upper body, portrait, simple background, white background, "
        "flat color, cel shading, anime coloring, clean lineart, "
        "masterpiece, best quality, highres"
    )
    negative_prompt: str = (
        "lowres, bad anatomy, bad hands, missing fingers, extra digits, "
        "blurry, low quality, worst quality, normal quality, "
        "realistic, 3d, photo, watermark, signature, text, "
        "long hair, flowing hair, hair blowing, wind, floating hair, messy hair, "
        "blue hair, sailor collar, sailor uniform, "
        "glowing, lens flare, light particles, sparkle, bloom, "
        "gradient background, detailed background, scenery, outdoors, "
        "multiple girls, extra limbs, deformed, ugly, duplicate, 2girls, multiple body, multiple face, multiple head"
    )

    # Noise seed for reproducible output (42 = fixed, -1 = random each run)
    seed: int = 42

    # Pipeline queue depths (keep small to minimise latency)
    capture_queue_size: int = 2
    pose_queue_size: int = 2
    output_queue_size: int = 4

    # Pose extraction
    detect_hands: bool = True
    half_body: bool = False  # VTuber mode: upper body only

    # Temporal smoothing (disabled by default — no_interp=true)
    interp_alpha: float = 0.3
    no_interp: bool = True

    # Temporal latent blending
    temporal_blend: float = 0.5

    # img2img feedback
    img2img_strength: float = 0.5
    img2img_input: str = "reference"

    # Reference character image
    reference_image: str = "assets/reference.png"

    # Control map jitter threshold
    ctrl_jitter_threshold: float = 0.015

    # ControlNet conditioning scale
    controlnet_conditioning_scale: float = 1.8

    # IP-Adapter
    ip_adapter_scale: float = 0.6
    ip_adapter_weight: str = "ip-adapter-plus_sd15.bin"

    # Temporal feedback
    temporal_feedback_strength: float = 0.2

    # Adaptive motion thresholds
    #   motion_lo: below this ctrl_diff = jitter, use base feedback strength
    #   motion_hi: above this = large motion, use max_strength (near full reset)
    #   motion_max_strength: cap for adaptive strength (0.85 = strong re-denoise)
    #   pose_empty_threshold: if pose energy < this, treat as no person → reset
    motion_lo: float = 0.008
    motion_hi: float = 0.04
    motion_max_strength: float = 0.85
    pose_empty_threshold: float = 0.001

    # Hardware
    device: str = "cuda"  # "cuda" | "cpu" | "mps"
    dtype: str = "float16"  # "float16" | "bfloat16" | "float32"

    # Display
    show_skeleton_overlay: bool = True
    show_fps: bool = True
    window_title: str = "GenTuber v1"


cfg = Config()
