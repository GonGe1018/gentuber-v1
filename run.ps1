# run.ps1 — Setup and launch for realtime-live2d
# Usage: .\run.ps1 [args passed to main.py]
# Examples:
#   .\run.ps1                              # default (384x384, T2I, 1 step)
#   .\run.ps1 --source 0                   # webcam
#   .\run.ps1 --size 256                   # fast mode (~26 FPS)
#   .\run.ps1 --size 512 --steps 2         # quality mode (~9 FPS)
#   .\run.ps1 --backend controlnet         # ControlNet backend

$ErrorActionPreference = "Stop"

Write-Host "=== realtime-live2d ===" -ForegroundColor Cyan

# Sync all dependencies (torch cu128 wheels are pinned in pyproject.toml)
Write-Host "[1/2] Syncing dependencies..." -ForegroundColor Yellow
uv sync

# Launch pipeline
Write-Host "[2/2] Starting pipeline..." -ForegroundColor Green
uv run live2d @args
