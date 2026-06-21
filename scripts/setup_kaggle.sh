#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-kaggle.txt

echo "Kaggle/T4 setup complete."
echo "Use: VLA_BACKEND=openvla-4bit ENV_BACKEND=simpler python -m src.loop_runner --config config/kaggle.yaml"
