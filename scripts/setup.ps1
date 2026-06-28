#!/usr/bin/env python3
"""
Repository setup script for Windows (PowerShell).
Run from the project root:

    powershell -ExecutionPolicy Bypass -File scripts/setup.ps1

What it does:
  1. Creates a virtual environment in ./venv
  2. Installs production dependencies
  3. Installs development dependencies
  4. Installs pre-commit hooks
"""

Write-Host "`n=====================================================" -ForegroundColor Cyan
Write-Host "  ISL Sign-to-Text — Developer Setup" -ForegroundColor Cyan
Write-Host "=====================================================`n" -ForegroundColor Cyan

# 1. Create venv if not exists
if (-not (Test-Path "venv")) {
    Write-Host "[1/4] Creating virtual environment..." -ForegroundColor Yellow
    python -m venv venv
} else {
    Write-Host "[1/4] Virtual environment already exists" -ForegroundColor Green
}

# 2. Install production deps
Write-Host "[2/4] Installing production dependencies..." -ForegroundColor Yellow
venv\Scripts\pip.exe install --upgrade pip
venv\Scripts\pip.exe install -r requirements.txt

# 3. Install dev deps
Write-Host "[3/4] Installing development dependencies..." -ForegroundColor Yellow
venv\Scripts\pip.exe install -r requirements-dev.txt

# 4. Install pre-commit hooks
Write-Host "[4/4] Installing pre-commit hooks..." -ForegroundColor Yellow
venv\Scripts\pre-commit install

Write-Host "`n====================================================" -ForegroundColor Green
Write-Host "  Setup complete!" -ForegroundColor Green
Write-Host "  Activate:  venv\Scripts\activate" -ForegroundColor White
Write-Host "  Run API:   python run_api.py" -ForegroundColor White
Write-Host "  Run tests: pytest tests/unit/ tests/api/" -ForegroundColor White
Write-Host "====================================================`n" -ForegroundColor Green
