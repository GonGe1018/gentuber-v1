# Legacy Backends

Archived diffusion engine backends that are no longer actively used. The current default is `ip_adapter` in `src/diffusion_engine_ip_adapter.py`.

## Engines

| File | Backend | Description |
|---|---|---|
| `diffusion_engine.py` | `controlnet` | LCM + ControlNet (eager mode) |
| `diffusion_engine_t2i.py` | `t2i` | LCM + T2I-Adapter (eager mode) |
| `diffusion_engine_sdturbo.py` | `sdturbo` | SD-Turbo + T2I-Adapter (eager mode) |
| `diffusion_engine_sdturbo_graph.py` | `sdturbo_graph` | SD-Turbo + T2I-Adapter + CUDA graph |
| `diffusion_engine_lcm_graph.py` | `lcm_graph` | KohakuV2 + LCM-LoRA + CUDA graph |

## Scripts

Related benchmark and test scripts are in `scripts/legacy/`.

## Restoring

To use a legacy backend, move it back to `src/` and update the imports in `main.py`.
