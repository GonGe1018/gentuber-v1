# run.ps1 — Setup and launch for realtime-live2d
# Usage: .\run.ps1 [args passed to main.py]
# Examples:
#   .\run.ps1                              # default (384x384, lcm_graph, ~73 FPS)
#   .\run.ps1 --source 0                   # webcam
#   .\run.ps1 --quality fast               # 256px, no hands (~121 FPS)
#   .\run.ps1 --quality quality            # 512px (~43 FPS)
#   .\run.ps1 --backend sdturbo_graph      # SD-Turbo backend (~72 FPS)
#   .\run.ps1 --steps 2                    # 2-step LCM (better quality, ~16 FPS)

$ErrorActionPreference = "Stop"

Write-Host "=== realtime-live2d ===" -ForegroundColor Cyan

# Sync all dependencies (torch cu128 wheels are pinned in pyproject.toml)
Write-Host "[1/2] Syncing dependencies..." -ForegroundColor Yellow
uv sync

# Launch pipeline
Write-Host "[2/2] Starting pipeline..." -ForegroundColor Green
uv run live2d @args
