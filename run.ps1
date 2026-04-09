# run.ps1 — Setup and launch for gentuber-v1
# Usage: .\run.ps1 [args passed to main.py]
# Examples:
#   .\run.ps1                              # default (256x256, webcam)
#   .\run.ps1 --source 0                   # webcam
#   .\run.ps1 --quality fast               # 256px, no hands
#   .\run.ps1 --quality quality            # 512px
#   .\run.ps1 --half-body                  # VTuber mode

$ErrorActionPreference = "Stop"

Write-Host "=== GenTuber v1 ===" -ForegroundColor Cyan

# Sync all dependencies (torch cu128 wheels are pinned in pyproject.toml)
Write-Host "[1/2] Syncing dependencies..." -ForegroundColor Yellow
uv sync

# Launch pipeline
Write-Host "[2/2] Starting pipeline..." -ForegroundColor Green
uv run gentuber @args
