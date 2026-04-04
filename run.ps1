# run.ps1 — One-shot setup and launch for realtime-live2d
# Usage: .\run.ps1 [args passed to main.py]
# Example: .\run.ps1 --steps 2 --source 0

$ErrorActionPreference = "Stop"

Write-Host "=== realtime-live2d setup ===" -ForegroundColor Cyan

# 1. Sync managed deps (everything except torch)
Write-Host "[1/3] Syncing dependencies..." -ForegroundColor Yellow
uv sync

# 2. Install cu128 torch (bypasses lockfile hash check)
Write-Host "[2/3] Installing PyTorch cu128 for RTX 5070 Ti..." -ForegroundColor Yellow
uv pip install "torch==2.7.0+cu128" "torchvision==0.22.0+cu128" `
    --index-url https://download.pytorch.org/whl/cu128

# 3. Run pipeline
Write-Host "[3/3] Starting pipeline..." -ForegroundColor Green
uv run --no-sync python main.py @args
