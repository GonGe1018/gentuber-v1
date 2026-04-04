# realtime-live2d

Real-time anime character animation from a webcam or video file using pose-conditioned diffusion.

**RTX 5070 Ti benchmarks (384×384, 1 step):**

| Backend | FPS |
|---|---|
| SD-Turbo + T2I-Adapter | ~24 |
| LCM + T2I-Adapter | ~23 |
| LCM + ControlNet | ~18 |

## How it works

```
VideoCapture → PoseExtractor (MediaPipe) → DiffusionEngine (GPU) → Display
```

Each stage runs in its own thread. Queues are bounded and drop stale frames so latency never accumulates.

## Setup

Requires Python 3.12, CUDA 12.8, and an NVIDIA GPU (tested on RTX 5070 Ti).

```powershell
git clone https://github.com/yourname/realtime-live2d
cd realtime-live2d
uv sync
```

`uv sync` installs everything including PyTorch cu128 via direct wheel URLs — no manual pip steps needed.

## Run

```powershell
# Default (SD-Turbo, 384x384, webcam or test video)
uv run live2d

# Webcam
uv run live2d --source 0

# Fast mode (256x256, ~26 FPS)
uv run live2d --size 256

# Quality mode (512x512, 2 steps)
uv run live2d --size 512 --steps 2

# Custom prompt
uv run live2d --prompt "samurai warrior, detailed armor, white background"

# Switch backend
uv run live2d --backend t2i
uv run live2d --backend controlnet
```

Press `q` in the display window to quit.

Or use the convenience script:

```powershell
.\run.ps1 --source 0 --size 384
```

## CLI reference

| Flag | Default | Description |
|---|---|---|
| `--source` | `assets/test_input.mp4` | Video file path or webcam index |
| `--steps` | `1` | LCM inference steps (1–4) |
| `--size` | `384` | Output resolution: 256 / 384 / 512 |
| `--backend` | `sdturbo` | `sdturbo` / `t2i` / `controlnet` |
| `--prompt` | see config.py | Generation prompt |
| `--no-skeleton` | off | Hide skeleton overlay |
| `--no-interp` | off | Disable temporal smoothing |
| `--no-hands` | off | Skip hand landmark detection |

## Configuration

Edit `config.py` to change defaults without CLI flags:

```python
cfg.video_source = 0          # webcam
cfg.num_inference_steps = 2   # better quality
cfg.output_width = 512        # higher resolution
cfg.prompt = "..."            # your prompt
cfg.detect_hands = False      # skip hand detection
```

## Benchmarks

Run the full benchmark suite:

```powershell
uv run python scripts/bench_all.py
```

Per-stage profiling:

```powershell
uv run python scripts/profile_pipeline.py
```

## Models

Downloaded automatically on first run to `~/.cache/huggingface/`:

| Model | Size | Purpose |
|---|---|---|
| `stabilityai/sd-turbo` | 3.1 GB | Base diffusion model |
| `TencentARC/t2iadapter_openpose_sd14v1` | 77 MB | Pose conditioning |
| `lllyasviel/control_v11p_sd15_openpose` | 361 MB | ControlNet (alternative) |
| `madebyollin/taesd` | 5 MB | Tiny VAE decoder |
| `pose_landmarker_lite.task` | ~5 MB | MediaPipe pose model |
| `hand_landmarker.task` | ~9 MB | MediaPipe hand model |

## Project structure

```
main.py                     # pipeline orchestration
config.py                   # all tunable parameters
src/
  capture.py                # threaded video capture
  pose_extractor.py         # MediaPipe → OpenPose skeleton map
  diffusion_engine.py       # LCM + ControlNet
  diffusion_engine_t2i.py   # LCM + T2I-Adapter
  diffusion_engine_sdturbo.py  # SD-Turbo + T2I-Adapter (default)
  interpolator.py           # temporal frame blending
  renderer.py               # OpenCV display + FPS overlay
scripts/
  test_stage1.py            # pose extraction benchmark
  test_stage3.py            # end-to-end pipeline benchmark
  bench_all.py              # full benchmark matrix
  profile_pipeline.py       # per-stage latency breakdown
  quality_check.py          # side-by-side comparison video
  make_test_video.py        # generate synthetic test input
```

## Optimisations applied

- **T2I-Adapter** instead of ControlNet (77M vs 361M params, ~25% faster)
- **SD-Turbo** single-step adversarial model (no scheduler overhead)
- **TAESD** tiny VAE decoder (~10x faster than full VAE)
- **Pre-computed text embeddings** (CLIP runs once at startup)
- **channels_last** memory layout (better tensor core utilisation)
- **PyTorch SDPA** (Flash Attention via `AttnProcessor2_0`)
- **TF32** on Blackwell/Ampere (~10% free speedup)
- **cuDNN benchmark** + warmup (tuned conv algorithms)
- **MediaPipe VIDEO mode** (temporal tracking, 54 FPS vs 40 FPS)
- **Pinned memory + async H2D transfer** (overlapped CPU→GPU copy)
- **Double-buffered GPU→CPU copy** (async D2H while next inference runs)
