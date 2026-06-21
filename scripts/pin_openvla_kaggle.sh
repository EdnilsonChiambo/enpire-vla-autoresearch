#!/usr/bin/env bash
# Force OpenVLA-compatible transformers on Kaggle when pip resolver refuses downgrade.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"

python -m pip install -q \
  "transformers==4.40.2" \
  "tokenizers>=0.19,<0.20" \
  --force-reinstall \
  --no-deps

python -m pip install -q -r "$ROOT/requirements-kaggle.txt"

python -c "
import transformers
from transformers import AutoModelForVision2Seq
print('transformers', transformers.__version__, 'OK')
"
