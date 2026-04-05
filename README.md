# realtime-live2d

Real-time anime character animation from a webcam or video file using pose-conditioned diffusion.

**RTX 5070 Ti benchmarks (pure engine throughput, no I/O):**

| Backend | 256×256 | 384×384 | 512×512 | Quality |
|---|---|---|---|---|
| KohakuV2 + LCM-LoRA + CUDA graph | **~131 FPS** | **~73 FPS** | **~47 FPS** | ★★★★ anime |
| SD-Turbo + T2I + CUDA graph | **~131 FPS** | **~77 FPS** | **~51 FPS** | ★★★ generic |
| SD-Turbo + T2I (eager) | ~28 FPS | ~27 FPS | — | ★★★ generic |
| LCM + T2I-Adapter | — | ~27 FPS | ~15 FPS | ★★★ generic |
| LCM + ControlNet | — | ~19 FPS | ~11 FPS | ★★★ generic |

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
uv run live2d --quality fast       # 256px, no hands, ~120 FPS
uv run live2d --quality balanced   # 384px, default, ~73 FPS
uv run live2d --quality quality    # 512px, ~48 FPS

# Manual control
uv run live2d --source 0                   # webcam
uv run live2d --backend lcm_graph          # default, KohakuV2 anime, ~73 FPS
uv run live2d --backend sdturbo_graph      # SD-Turbo, ~73 FPS
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
| `--steps` | `1` | LCM inference steps (1–4, lcm_graph uses CUDA graph at 1, eager at 2+) |
| `--size` | `384` | Output resolution: 256 / 384 / 512 |
| `--model` | — | Anime model for `lcm_graph` (e.g. `Lykon/dreamshaper-8`) |
| `--quality` | — | `fast` (~132 FPS) / `balanced` (~73 FPS) / `quality` (~48 FPS) |
| `--backend` | `lcm_graph` | `lcm_graph` / `sdturbo_graph` / `sdturbo` / `t2i` / `controlnet` |
| `--max-fps` | `60` | Cap display refresh rate (0 = uncapped) |
| `--prompt` | see config.py | Generation prompt |
| `--negative-prompt` | see config.py | Negative prompt |
| `--no-skeleton` | off | Hide skeleton overlay |
| `--no-interp` | off | Disable temporal smoothing |
| `--no-hands` | off | Skip hand landmark detection |

## Configuration

Edit `config.py` to change defaults without CLI flags:

```python
cfg.video_source = 0          # webcam
cfg.num_inference_steps = 2   # better quality (t2i/controlnet only)
cfg.output_width = 512        # higher resolution
cfg.prompt = "..."            # your prompt
cfg.detect_hands = False      # skip hand detection
cfg.lcm_model_id = "Lykon/dreamshaper-8"  # alternative anime model
```

## Benchmarks

Run the full benchmark suite (eager backends):

```powershell
uv run python scripts/bench_all.py
```

Pure engine throughput (CUDA graph backends):

```powershell
uv run python scripts/bench_throughput.py --engine lcm_graph
uv run python scripts/bench_throughput.py --engine sdturbo_graph
```

Per-op hot path profiling:

```powershell
uv run python scripts/profile_worker.py
```

## Models

Downloaded automatically on first run to `~/.cache/huggingface/`:

| Model | Size | Purpose |
|---|---|---|
| `KBlueLeaf/kohaku-v2.1` | ~2 GB | Anime SD1.5 base (lcm_graph default) |
| `latent-consistency/lcm-lora-sdv1-5` | ~200 MB | LCM-LoRA weights (fused at startup) |
| `stabilityai/sd-turbo` | 3.1 GB | SD-Turbo base (sdturbo_graph) |
| `TencentARC/t2iadapter_openpose_sd14v1` | 77 MB | Pose conditioning (T2I-Adapter) |
| `lllyasviel/control_v11p_sd15_openpose` | 361 MB | ControlNet (alternative) |
| `madebyollin/taesd` | 5 MB | Tiny VAE decoder |
| `pose_landmarker_lite.task` | ~5 MB | MediaPipe pose model |
| `hand_landmarker.task` | ~9 MB | MediaPipe hand model |

## Project structure

```
main.py                     # pipeline orchestration
config.py                   # all tunable parameters
run.ps1                     # convenience launcher (uv sync + run)
src/
  capture.py                # threaded video capture
  pose_extractor.py         # MediaPipe -> OpenPose skeleton map
  diffusion_engine_lcm_graph.py              # KohakuV2 + LCM-LoRA + CUDA graph (default)
  diffusion_engine_sdturbo_graph.py          # SD-Turbo + T2I-Adapter + CUDA graph
  diffusion_engine_sdturbo.py        # SD-Turbo + T2I-Adapter eager
  diffusion_engine_t2i.py            # LCM + T2I-Adapter
  diffusion_engine.py                # LCM + ControlNet
  interpolator.py           # temporal frame blending (cv2.addWeighted)
  renderer.py               # OpenCV display + FPS overlay + rate cap
scripts/
  test_stage1.py            # pose extraction benchmark (54 FPS)
  test_stage3.py            # end-to-end pipeline benchmark
  test_graph_engine.py      # CUDA graph engine benchmark (sdturbo_graph)
  test_lcm_graph.py         # CUDA graph engine benchmark (lcm_graph)
  test_t2i_adapter.py       # T2I-Adapter engine benchmark
  test_webcam.py            # live webcam test with display
  bench_all.py              # full benchmark matrix (all backends x resolutions)
  bench_throughput.py       # pure engine throughput (no I/O bottleneck)
  bench_latency.py          # end-to-end latency (5.9ms avg @ 384x384)
  bench_cuda_graphs.py      # CUDA graph vs eager UNet comparison
  bench_sdturbo.py          # SD-Turbo vs LCM isolated comparison
  bench_anime_lcm.py        # anime model comparison (KohakuV2 vs DreamShaper8)
  bench_lcm_steps.py        # LCM step count vs speed tradeoff
  bench_gil_contention.py   # GIL contention measurement (0.3%)
  bench_int8.py             # INT8 quantization (slower on Windows)
  bench_fp8.py              # FP8 quantization (slower without calibration)
  bench_sdxl_turbo.py       # SDXL-Turbo (16 FPS, not viable)
  bench_torch_compile.py    # torch.compile (fails on Windows/no Triton)
  bench_dtype.py            # dtype comparison (fp16 vs bf16)
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
| float32 intermediate in preprocess | ~1.5ms pose | 2x faster than direct float16 cast |
| `cv2.addWeighted` interpolation | ~0.5ms | SIMD uint8 vs float32 cast |
| `cv2.convertScaleAbs` D2H cast | ~0.4ms | 15x faster than `(arr*255).astype(u8)` |
| CPU timestep (avoid device sync) | ~15ms | `int(t.cpu())` vs `int(t)` on CUDA tensor |
| Skip noise×sigma (LCM σ=1.0) | ~0.01ms | Identity multiply removed |
| Remove redundant `fill_()` | ~0.01ms | Constant timestep set once at capture |
| KohakuV2 + LCM-LoRA | quality | Anime-specific SD1.5, same throughput |

**What didn't work on Windows:**
- `torch.compile` — requires Triton (Linux only)
- INT8 (bitsandbytes) — falls back to dequantize+fp16, 2x slower
- FP8 (`torch._scaled_mm`) — per-call cast overhead > GEMM gain without calibrated scales

**Hard floor:** UNet+adapter CUDA graph replay = **16.6ms @ 384×384** (60 FPS theoretical max). End-to-end latency avg **5.8ms** (p95=11ms) due to pose frame reuse.
