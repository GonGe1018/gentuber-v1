# GenTuber v1

[한국어](README_KO.md)

Real-time anime character animation driven by webcam or video input. Your body movements are captured via MediaPipe pose estimation, then an anime character mimics your poses in real-time using Stable Diffusion.

## Demo

https://github.com/user-attachments/assets/demo_sidebyside_mixkit.mp4

https://github.com/user-attachments/assets/demo_sidebyside_pexels.mp4

Source → Skeleton → Generated (256px, ~13 FPS on RTX 5070 Ti)

## Architecture

```
[Webcam / Video] → [MediaPipe Pose] → [Diffusion Engine (GPU)] → [Display / MP4]
                     skeleton extraction    IP-Adapter (character)
                                            ControlNet (pose)
                                            LCM (speed)
```

Each stage runs in its own thread with bounded queues — latency never accumulates.

## Key Features

- **IP-Adapter + ControlNet**: Character appearance and pose are controlled through independent paths. No conflict between style preservation and pose accuracy.
- **Temporal Feedback**: Previous frame's latent feeds into the next generation for consistent style across frames.
- **Fixed Noise**: Same latent tensor reused every frame so only pose changes affect the output.
- **Half-Body (VTuber) Mode**: Upper body only skeleton for VTuber-style bust shots.
- **Headless Recording**: `--output result.mp4` processes a video file and saves the result without GUI.

## Requirements

- Python 3.12+
- CUDA 12.8+
- NVIDIA GPU (tested on RTX 5070 Ti, 16GB VRAM)

## Setup

```bash
git clone https://github.com/GonGe1018/gentuber-v1
cd gentuber-v1
uv sync
```

All dependencies including PyTorch CUDA wheels are resolved automatically by `uv sync`.

## Quick Start

```bash
# Webcam (real-time, default IP-Adapter backend)
uv run gentuber --source 0

# Video file → MP4 output (no GUI)
uv run gentuber --source input.mp4 --output result.mp4

# Use a custom character reference image
uv run gentuber --source 0 --reference my_character.png
```

## Backends

The default (and only active) backend is `ip_adapter` — IP-Adapter Plus for character appearance + ControlNet for pose + LCM-LoRA for speed + latent-level temporal feedback.

| Backend | FPS (384px) | Description |
|---|---|---|
| `ip_adapter` | ~13 | Character-driven animation with latent feedback |

> Legacy backends (lcm_graph, sdturbo_graph, controlnet, etc.) are archived in `src/legacy/`. See [src/legacy/README.md](src/legacy/README.md).

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--source` | `assets/test_input.mp4` | Video file path or webcam index (0, 1, ...) |
| `--output`, `-o` | — | Save to MP4 and exit (headless, no GUI) |
| `--reference` | `assets/reference.png` | Character reference image for IP-Adapter |
| `--backend` | `ip_adapter` | Diffusion backend |
| `--steps` | `4` | Inference steps |
| `--size` | `384` | Output resolution: `256` / `384` / `512` |
| `--ip-scale` | `0.5` | IP-Adapter strength (0.3=light, 0.7=strong character) |
| `--cn-scale` | `2.0` | ControlNet pose strength |
| `--feedback` | `0.3` | Temporal feedback (0.3=strong coherence, 1.0=no feedback) |
| `--seed` | `42` | Noise seed (-1 = random) |
| `--prompt` | see config.py | Generation prompt |
| `--half-body` | off | VTuber mode: upper body only |
| `--quality` | — | Preset: `fast` / `balanced` / `quality` |
| `--max-fps` | `60` | Display refresh cap |
| `--no-skeleton` | off | Hide skeleton overlay |
| `--no-interp` | off | Disable temporal smoothing |
| `--no-hands` | off | Skip hand detection |

## Examples

```bash
# Strong character preservation, moderate pose
uv run gentuber --source 0 --ip-scale 0.6 --cn-scale 1.0

# Strong pose following, lighter character
uv run gentuber --source 0 --ip-scale 0.4 --cn-scale 2.0

# No temporal feedback (each frame independent)
uv run gentuber --source 0 --feedback 1.0

# Batch process a dance video
uv run gentuber --source dance.mp4 -o dance_anime.mp4 --steps 4

# VTuber half-body mode
uv run gentuber --source 0 --half-body
```

## How IP-Adapter Backend Works

```
Frame 1:  Fixed noise → UNet + ControlNet(pose₁) + IP-Adapter(character) → Latent₁ → Decode → Output₁
                                                                              ↓
Frame 2:  Latent₁ + noise → UNet + ControlNet(pose₂) + IP-Adapter(character) → Latent₂ → Decode → Output₂
                                                                                  ↓
Frame 3:  Latent₂ + noise → UNet + ControlNet(pose₃) + IP-Adapter(character) → Latent₃ → Decode → Output₃
```

- **IP-Adapter Plus**: Injects character appearance via CLIP patch embeddings into cross-attention (cached at startup, zero per-frame cost)
- **ControlNet**: Guides pose via OpenPose skeleton at every denoising step
- **Latent Feedback**: Previous frame's latent reused directly (no VAE encode cycle → no color drift)
- **Adaptive Motion**: Small motion = fewer denoise steps (fast), large motion = full txt2img reset (clean)

## Configuration

Edit `config.py` for persistent defaults:

```python
cfg.video_source = 0                        # webcam
cfg.engine_backend = "ip_adapter"           # best character consistency
cfg.reference_image = "my_character.png"    # your character
cfg.ip_adapter_scale = 0.6                  # character strength
cfg.controlnet_conditioning_scale = 1.8     # pose strength
cfg.temporal_feedback_strength = 0.2        # style coherence
cfg.output_width = cfg.output_height = 384  # resolution
```

## Models

All models are downloaded automatically on first run to `~/.cache/huggingface/`:

| Model | Size | Purpose |
|---|---|---|
| `KBlueLeaf/kohaku-v2.1` | ~2 GB | Anime SD1.5 base model |
| `latent-consistency/lcm-lora-sdv1-5` | ~200 MB | LCM-LoRA for fast inference |
| `h94/IP-Adapter` (ip-adapter-plus_sd15) | ~98 MB | Character appearance preservation |
| `h94/IP-Adapter` (image_encoder) | ~1.2 GB | CLIP ViT-H-14 (unloaded after caching) |
| `lllyasviel/control_v11p_sd15_openpose` | ~361 MB | ControlNet OpenPose |
| `TencentARC/t2iadapter_openpose_sd14v1` | ~77 MB | T2I-Adapter (legacy) |
| `madebyollin/taesd` | ~5 MB | Tiny VAE decoder |
| MediaPipe pose/hand models | ~14 MB | Pose estimation |

## Project Structure

```
main.py                                  # Pipeline orchestration + CLI
config.py                                # All tunable parameters
src/
  diffusion_engine_ip_adapter.py         # IP-Adapter + ControlNet + LCM (default)
  capture.py                             # Threaded video capture (FPS-synced)
  pose_extractor.py                      # MediaPipe → OpenPose skeleton
  interpolator.py                        # Temporal frame blending
  renderer.py                            # OpenCV display + FPS overlay
  settings_gui.py                        # Tkinter settings panel
  legacy/                                # Archived backends (see legacy/README.md)
assets/
  reference.png                          # Default character reference
  models/                                # MediaPipe model files
```

## Version History

| Branch | Description |
|---|---|
| `master` | Current — IP-Adapter + ControlNet + latent feedback |
| `dev/v0.0.7-latent-feedback` | Latent-level temporal feedback |
| `dev/v0.0.6-vtuber-halfbody` | VTuber half-body mode |
| `dev/v0.0.5-ip-adapter-gui` | IP-Adapter + GUI settings panel |
| `dev/v0.0.3-reference-img2img` | Reference image img2img mode |
| `dev/v0.0.2-source-img2img` | Camera frame img2img (StreamDiffusion style) |
| `dev/v0.0.1-noise-based` | Pure noise + T2I-Adapter |
| `dev/v0.0.1-alpha` | Initial prototype |

## License

MIT
