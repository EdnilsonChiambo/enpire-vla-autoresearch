#!/usr/bin/env bash
# ManiSkill3 SIMPLER install for Kaggle Python 3.12 (main branch needs sapien 2.2.2 / py3.10).
set -euo pipefail

SIMPLER_DIR="${SIMPLER_DIR:-/kaggle/working/SimplerEnv}"

if [ -d "$SIMPLER_DIR/.git" ]; then
  git -C "$SIMPLER_DIR" fetch --depth 1 origin maniskill3
  git -C "$SIMPLER_DIR" checkout maniskill3
else
  git clone -b maniskill3 --depth 1 https://github.com/simpler-env/SimplerEnv.git "$SIMPLER_DIR"
fi

python -m pip install -q "git+https://github.com/haosulab/ManiSkill.git" "tyro==0.8.5" "torch==2.3.1"
python -m pip install -q -e "$SIMPLER_DIR"

python -c "import gymnasium as gym; import mani_skill; import simpler_env; print('SIMPLER ManiSkill3 OK')"
