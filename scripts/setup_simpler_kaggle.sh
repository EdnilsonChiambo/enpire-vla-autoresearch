#!/usr/bin/env bash
# Full SIMPLER install for Kaggle / GPU notebooks.
set -euo pipefail

SIMPLER_DIR="${SIMPLER_DIR:-/kaggle/working/SimplerEnv}"

if [ ! -d "$SIMPLER_DIR/.git" ]; then
  git clone --recurse-submodules --depth 1 https://github.com/simpler-env/SimplerEnv.git "$SIMPLER_DIR"
else
  git -C "$SIMPLER_DIR" submodule update --init --recursive
fi

python -m pip install -q "numpy>=1.26,<2.0"
python -m pip install -q -e "$SIMPLER_DIR/ManiSkill2_real2sim"
python -m pip install -q -e "$SIMPLER_DIR"

python -c "import simpler_env; import mani_skill2_real2sim; print('SIMPLER OK')"
