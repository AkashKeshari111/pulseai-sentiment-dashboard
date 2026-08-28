<#
.SYNOPSIS
    One-command setup for PulseAI on Windows.

.DESCRIPTION
    Creates a virtual environment, installs dependencies, prepares the dataset,
    trains the baseline, and installs the dashboard's npm packages.

    Transformer fine-tuning is NOT run by default - it takes roughly two hours
    on CPU. Pass -Train to include it, or run it yourself later with:
        python -m src.train_transformer

.EXAMPLE
    .\scripts\quickstart.ps1
    .\scripts\quickstart.ps1 -Train -Synthetic
#>
[CmdletBinding()]
param(
    [switch]$Train,       # also fine-tune DistilBERT (~2 hours on CPU)
    [switch]$Synthetic,   # generate data locally instead of streaming from the Hub
    [switch]$SkipNpm
)

$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root

function Write-Step($message) {
    Write-Host ''
    Write-Host "=== $message" -ForegroundColor Cyan
}

# --- 1. Virtual environment --------------------------------------------------
Write-Step 'Python environment'
if (-not (Test-Path '.venv')) {
    python -m venv .venv
    Write-Host 'created .venv'
}
$python = Join-Path $root '.venv\Scripts\python.exe'

& $python -m pip install --upgrade pip --quiet
Write-Host 'installing dependencies (this takes a few minutes the first time)...'
& $python -m pip install -r requirements.txt --quiet
Write-Host 'dependencies installed' -ForegroundColor Green

# --- 2. Configuration --------------------------------------------------------
Write-Step 'Configuration'
if (-not (Test-Path '.env')) {
    Copy-Item '.env.example' '.env'
    Write-Host 'created .env from .env.example' -ForegroundColor Yellow
    Write-Host 'ACTION REQUIRED: put your MongoDB Atlas connection string in .env' -ForegroundColor Yellow
} else {
    Write-Host '.env already exists - leaving it alone'
}

# --- 3. Data -----------------------------------------------------------------
Write-Step 'Dataset'
if (Test-Path 'data\processed\train.csv') {
    Write-Host 'prepared splits already exist - skipping'
} elseif ($Synthetic) {
    & $python -m src.dataset --prepare --synthetic
} else {
    & $python -m src.dataset --prepare
}

# --- 4. Models ---------------------------------------------------------------
Write-Step 'Baseline model'
& $python -m src.train_baseline

if ($Train) {
    Write-Step 'Fine-tuning DistilBERT (this will take a while)'
    & $python -m src.train_transformer
} else {
    Write-Host ''
    Write-Host 'Skipped transformer fine-tuning. Run it with:' -ForegroundColor Yellow
    Write-Host '    .venv\Scripts\python.exe -m src.train_transformer' -ForegroundColor Yellow
}

# --- 5. Dashboard ------------------------------------------------------------
if (-not $SkipNpm) {
    Write-Step 'Dashboard dependencies'
    Push-Location dashboard
    npm install --no-audit --no-fund
    Pop-Location
}

# --- Done --------------------------------------------------------------------
Write-Host ''
Write-Host '======================================================' -ForegroundColor Green
Write-Host '  Setup complete' -ForegroundColor Green
Write-Host '======================================================' -ForegroundColor Green
Write-Host ''
Write-Host '  1. Put your Atlas connection string in .env'
Write-Host '  2. Seed demo data:  .venv\Scripts\python.exe -m api.seed --count 600'
Write-Host '  3. Start the API:   .venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8020'
Write-Host '  4. Start the UI:    cd dashboard; npm run dev'
Write-Host ''
Write-Host '  dashboard  http://localhost:5173'
Write-Host '  API docs   http://localhost:8020/docs'
Write-Host ''
