#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

python3 -m venv .venv
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements-gpu.txt

echo "GPU dependencies installed."
echo ""
echo "Next steps:"
echo "1. Install flash-attn: pip install flash-attn==2.6.1 --no-build-isolation"
echo "2. Clone SIMPLER: git clone https://github.com/simpler-env/SimplerEnv ../SimplerEnv"
echo "3. Install SIMPLER: cd ../SimplerEnv && pip install -e ."
echo "4. Set VLA_BACKEND=openvla and ENV_BACKEND=simpler in .env"
echo "5. Run: python -m src.loop_runner"
