# realtime-live2d

Real-time anime character animation from a webcam or video file using pose-conditioned diffusion.

**RTX 5070 Ti benchmarks (pure engine throughput, no I/O):**

| Backend | 256×256 | 384×384 | 512×512 |
|---|---|---|---|
| SD-Turbo + T2I + CUDA graph | **~124 FPS** | **~73 FPS** | **~49 FPS** |
| SD-Turbo + T2I (eager) | ~27 FPS | ~25 FPS | — |
| LCM + T2I-Adapter | — | ~25 FPS | ~15 FPS |
| LCM + ControlNet | — | ~19 FPS | — |

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
# Quality presets (easiest way to tune speed vs quality)
uv run live2d --quality fast       # 256px, no hands, ~124 FPS
uv run live2d --quality balanced   # 384px, default, ~73 FPS
uv run live2d --quality quality    # 512px, ~49 FPS

# Manual control
uv run live2d --source 0                   # webcam
uv run live2d --size 256                   # fast mode (~124 FPS)
uv run live2d --size 512 --steps 2         # quality mode (~25 FPS)
uv run live2d --backend sdturbo_graph      # default, ~73 FPS
uv run live2d --backend sdturbo            # eager, ~25 FPS
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
| `--quality` | — | `fast` (~124 FPS) / `balanced` (~73 FPS) / `quality` (~49 FPS) |
| `--backend` | `sdturbo_graph` | `sdturbo_graph` / `sdturbo` / `t2i` / `controlnet` |
| `--max-fps` | `60` | Cap display refresh rate (0 = uncapped) |
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
  diffusion_engine_sdturbo_graph.py  # SD-Turbo + T2I-Adapter + CUDA graph (default)
  interpolator.py           # temporal frame blending
  renderer.py               # OpenCV display + FPS overlay
scripts/
  test_stage1.py            # pose extraction benchmark
  test_stage3.py            # end-to-end pipeline benchmark
  bench_all.py              # full benchmark matrix
  bench_throughput.py       # pure engine throughput (no I/O)
  profile_worker.py         # per-op latency breakdown of engine hot path
  quality_check.py          # side-by-side comparison video
  make_test_video.py        # generate synthetic test input
```

## Optimisations applied

| Optimisation | Gain | Notes |
|---|---|---|
| CUDA graph (adapter+UNet+VAE) | **~2x** | Eliminates Python kernel-launch overhead |
| Noise ring (pre-generated) | ~1ms | Avoids `torch.randn` on hot path |
| Pose frame reuse | ~30% | Engine never waits for pose thread |
| Async D2H copy | ~0.5ms | Pinned memory + separate CUDA stream |
| T2I-Adapter vs ControlNet | ~25% | 77M vs 361M params |
| SD-Turbo vs LCM | ~5% | No scheduler overhead, CFG-free |
| TAESD vs full VAE | ~10x VAE | 5MB vs 335MB decoder |
| Pre-computed text embeddings | ~2ms | CLIP runs once at startup |
| `channels_last` memory layout | ~5% | Better tensor core utilisation |
| PyTorch SDPA (AttnProcessor2_0) | ~10% | Flash Attention via cuDNN |
| TF32 on Blackwell | ~5% | Free via `allow_tf32=True` |
| cuDNN benchmark + warmup | ~3% | Tuned conv algorithms |
| MediaPipe VIDEO mode | ~25% pose | Temporal tracking vs per-frame detect |
| pose_landmarker_lite | ~12% pose | 3MB vs 8MB model |
| ctrl preprocessing in pose thread | 0ms hot path | 3ms numpy work offloaded |
| `cv2.addWeighted` interpolation | ~0.5ms | SIMD uint8 vs float32 cast |

**What didn't work on Windows:**
- `torch.compile` — requires Triton (Linux only)
- INT8 (bitsandbytes) — falls back to dequantize+fp16, 2x slower
- FP8 (`torch._scaled_mm`) — per-call cast overhead > GEMM gain without calibrated scales

**Hard floor:** UNet+adapter CUDA graph replay = **15.6ms @ 384×384** (64 FPS theoretical max).
