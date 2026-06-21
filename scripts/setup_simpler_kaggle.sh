#!/usr/bin/env bash
# ManiSkill3 SIMPLER install for Kaggle Python 3.12 (main branch needs sapien 2.2.2 / py3.10).
set -euo pipefail

SIMPLER_DIR="${SIMPLER_DIR:-/kaggle/working/SimplerEnv}"

if [ -d "$SIMPLER_DIR/.git" ]; then
  current_branch="$(git -C "$SIMPLER_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || true)"
  if [ "$current_branch" = "maniskill3" ]; then
    git -C "$SIMPLER_DIR" pull --ff-only origin maniskill3 || true
  else
    # Shallow main clone leaves no local `maniskill3` ref; fetch then create branch from FETCH_HEAD.
    git -C "$SIMPLER_DIR" fetch --depth 1 origin maniskill3
    git -C "$SIMPLER_DIR" checkout -B maniskill3 FETCH_HEAD
  fi
else
  git clone -b maniskill3 --depth 1 https://github.com/simpler-env/SimplerEnv.git "$SIMPLER_DIR"
fi

python -m pip install -q "git+https://github.com/haosulab/ManiSkill.git" "tyro==0.8.5" "torch==2.3.1"
python -m pip install -q -e "$SIMPLER_DIR"

# ManiSkill can upgrade transformers; OpenVLA needs 4.40.2 + tokenizers 0.19.x
python -m pip install -q -r requirements-kaggle.txt

python -c "import gymnasium as gym; import mani_skill; import simpler_env; print('SIMPLER ManiSkill3 OK')"
