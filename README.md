# Realtime Live2D

[한국어](README_KO.md)

Real-time anime character animation driven by webcam or video input. Your body movements are captured via MediaPipe pose estimation, then an anime character mimics your poses in real-time using Stable Diffusion.

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
- **Temporal Feedback**: Previous frame feeds into the next generation for consistent style across frames.
- **Fixed Noise**: Same latent tensor reused every frame so only pose changes affect the output.
- **Multiple Backends**: From ~5 FPS (best quality) to ~73 FPS (fastest), pick your tradeoff.
- **Headless Recording**: `--output result.mp4` processes a video file and saves the result without GUI.

## Requirements

- Python 3.12+
- CUDA 12.8+
- NVIDIA GPU (tested on RTX 5070 Ti, 16GB VRAM)

## Setup

```bash
git clone https://github.com/GonGe1018/realtime-live2d
cd realtime-live2d
uv sync
```

All dependencies including PyTorch CUDA wheels are resolved automatically by `uv sync`.

## Quick Start

```bash
# Webcam (real-time, default IP-Adapter backend)
uv run live2d --source 0

# Video file → MP4 output (no GUI)
uv run live2d --source input.mp4 --output result.mp4

# Use a custom character reference image
uv run live2d --source 0 --reference my_character.png
```

## Backends

| Backend | FPS (384px) | Character Consistency | Pose Accuracy | Use Case |
|---|---|---|---|---|
| `ip_adapter` | ~5-17 | Best | Good | Character-driven animation |
| `lcm_graph` | ~60 | Low | Good | Fast prototyping |
| `sdturbo_graph` | ~63 | Low | Good | Fast prototyping |
| `controlnet` | ~19 | Medium | Good | ControlNet experiments |
| `sdturbo` | ~25 | Low | Good | Eager mode debugging |
| `t2i` | ~25 | Low | Good | T2I-Adapter experiments |

The `ip_adapter` backend uses IP-Adapter Plus for character appearance + ControlNet for pose + LCM-LoRA for speed + temporal feedback from the previous frame.

## CLI Reference

| Flag | Default | Description |
|---|---|---|
| `--source` | `assets/test_input.mp4` | Video file path or webcam index (0, 1, ...) |
| `--output`, `-o` | — | Save to MP4 and exit (headless, no GUI) |
| `--reference` | `assets/reference.png` | Character reference image for IP-Adapter |
| `--backend` | `ip_adapter` | `ip_adapter` / `lcm_graph` / `sdturbo_graph` / `sdturbo` / `t2i` / `controlnet` |
| `--steps` | `1` (4 for ip_adapter) | Inference steps |
| `--size` | `384` | Output resolution: `256` / `384` / `512` |
| `--ip-scale` | `0.5` | IP-Adapter strength (0.3=light, 0.7=strong character) |
| `--cn-scale` | `1.5` | ControlNet pose strength |
| `--feedback` | `0.3` | Temporal feedback (0.3=strong coherence, 1.0=no feedback) |
| `--strength` | `0.5` | img2img strength for legacy backends |
| `--seed` | `42` | Noise seed (-1 = random) |
| `--prompt` | see config.py | Generation prompt |
| `--quality` | — | Preset: `fast` / `balanced` / `quality` |
| `--max-fps` | `60` | Display refresh cap |
| `--no-skeleton` | off | Hide skeleton overlay |
| `--no-interp` | off | Disable temporal smoothing |
| `--no-hands` | off | Skip hand detection |

## Examples

```bash
# Strong character preservation, moderate pose
uv run live2d --source 0 --ip-scale 0.6 --cn-scale 1.0

# Strong pose following, lighter character
uv run live2d --source 0 --ip-scale 0.4 --cn-scale 1.5

# No temporal feedback (each frame independent)
uv run live2d --source 0 --feedback 1.0

# Batch process a dance video
uv run live2d --source dance.mp4 -o dance_anime.mp4 --steps 4

# Fast backend for prototyping (~60 FPS)
uv run live2d --source 0 --backend lcm_graph
```

## How IP-Adapter Backend Works

```
Frame 1:  Fixed noise → UNet + ControlNet(pose₁) + IP-Adapter(character) → Output₁
                                                                              ↓
Frame 2:  Output₁ + noise → UNet + ControlNet(pose₂) + IP-Adapter(character) → Output₂
                                                                                  ↓
Frame 3:  Output₂ + noise → UNet + ControlNet(pose₃) + IP-Adapter(character) → Output₃
```

- **IP-Adapter Plus**: Injects character appearance via CLIP patch embeddings into cross-attention (cached at startup, zero per-frame cost)
- **ControlNet**: Guides pose via OpenPose skeleton at every denoising step
- **Temporal Feedback**: Previous output feeds back as img2img input, preserving style continuity
- **Fixed Noise**: First frame uses a deterministic latent for reproducible baseline

## Configuration

Edit `config.py` for persistent defaults:

```python
cfg.video_source = 0                        # webcam
cfg.engine_backend = "ip_adapter"           # best character consistency
cfg.reference_image = "my_character.png"    # your character
cfg.ip_adapter_scale = 0.5                  # character strength
cfg.controlnet_conditioning_scale = 1.5     # pose strength
cfg.temporal_feedback_strength = 0.3        # style coherence
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
| `TencentARC/t2iadapter_openpose_sd14v1` | ~77 MB | T2I-Adapter (lcm_graph backend) |
| `madebyollin/taesd` | ~5 MB | Tiny VAE decoder |
| MediaPipe pose/hand models | ~14 MB | Pose estimation |

## Project Structure

```
main.py                                  # Pipeline orchestration + CLI
config.py                                # All tunable parameters
src/
  diffusion_engine_ip_adapter.py         # IP-Adapter + ControlNet + LCM (default)
  diffusion_engine_lcm_graph.py          # KohakuV2 + LCM-LoRA + CUDA graph
  diffusion_engine_sdturbo_graph.py      # SD-Turbo + T2I-Adapter + CUDA graph
  diffusion_engine.py                    # LCM + ControlNet (eager)
  diffusion_engine_t2i.py               # LCM + T2I-Adapter (eager)
  diffusion_engine_sdturbo.py            # SD-Turbo + T2I-Adapter (eager)
  capture.py                             # Threaded video capture (FPS-synced)
  pose_extractor.py                      # MediaPipe → OpenPose skeleton
  interpolator.py                        # Temporal frame blending
  renderer.py                            # OpenCV display + FPS overlay
assets/
  reference.png                          # Default character reference
  models/                                # MediaPipe model files
```

## Version History

| Branch | Description |
|---|---|
| `master` | Current — IP-Adapter + ControlNet + temporal feedback |
| `dev/v0.0.3-reference-img2img` | Reference image img2img mode |
| `dev/v0.0.2-source-img2img` | Camera frame img2img (StreamDiffusion style) |
| `dev/v0.0.1-noise-based` | Pure noise + T2I-Adapter |
| `dev/v0.0.1-alpha` | Initial prototype |

## License

MIT
