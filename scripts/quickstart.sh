#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# One-command setup for PulseAI on macOS / Linux.
#
#   ./scripts/quickstart.sh              # setup, no transformer training
#   ./scripts/quickstart.sh --train      # also fine-tune DistilBERT (~2h CPU)
#   ./scripts/quickstart.sh --synthetic  # generate data locally, no Hub access
# ---------------------------------------------------------------------------
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

TRAIN=false
SYNTHETIC=false
SKIP_NPM=false

for arg in "$@"; do
  case "$arg" in
    --train)     TRAIN=true ;;
    --synthetic) SYNTHETIC=true ;;
    --skip-npm)  SKIP_NPM=true ;;
    *) echo "unknown option: $arg" >&2; exit 1 ;;
  esac
done

step() { printf '\n\033[36m=== %s\033[0m\n' "$1"; }

# --- 1. Virtual environment -------------------------------------------------
step 'Python environment'
[ -d .venv ] || { python3 -m venv .venv; echo 'created .venv'; }
PYTHON="$ROOT/.venv/bin/python"

"$PYTHON" -m pip install --upgrade pip --quiet
echo 'installing dependencies (a few minutes the first time)...'
"$PYTHON" -m pip install -r requirements.txt --quiet
echo 'dependencies installed'

# --- 2. Configuration -------------------------------------------------------
step 'Configuration'
if [ ! -f .env ]; then
  cp .env.example .env
  printf '\033[33mcreated .env — put your MongoDB Atlas connection string in it\033[0m\n'
else
  echo '.env already exists — leaving it alone'
fi

# --- 3. Data ----------------------------------------------------------------
step 'Dataset'
if [ -f data/processed/train.csv ]; then
  echo 'prepared splits already exist — skipping'
elif [ "$SYNTHETIC" = true ]; then
  "$PYTHON" -m src.dataset --prepare --synthetic
else
  "$PYTHON" -m src.dataset --prepare
fi

# --- 4. Models --------------------------------------------------------------
step 'Baseline model'
"$PYTHON" -m src.train_baseline

if [ "$TRAIN" = true ]; then
  step 'Fine-tuning DistilBERT (this will take a while)'
  "$PYTHON" -m src.train_transformer
else
  printf '\n\033[33mSkipped transformer fine-tuning. Run it with:\n    .venv/bin/python -m src.train_transformer\033[0m\n'
fi

# --- 5. Dashboard -----------------------------------------------------------
if [ "$SKIP_NPM" = false ]; then
  step 'Dashboard dependencies'
  (cd dashboard && npm install --no-audit --no-fund)
fi

cat <<'BANNER'

======================================================
  Setup complete
======================================================

  1. Put your Atlas connection string in .env
  2. Seed demo data:  .venv/bin/python -m api.seed --count 600
  3. Start the API:   .venv/bin/python -m uvicorn api.main:app --reload --port 8020
  4. Start the UI:    cd dashboard && npm run dev

  dashboard  http://localhost:5173
  API docs   http://localhost:8020/docs

BANNER
