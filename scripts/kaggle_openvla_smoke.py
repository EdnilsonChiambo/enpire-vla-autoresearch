"""Instrumented smoke test for OpenVLA 4-bit (run on Kaggle GPU)."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Ensure repo root is on sys.path when run as: python scripts/kaggle_openvla_smoke.py
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

import numpy as np

from src.debug_log import debug_log


def main() -> int:
    debug_log("H1", "smoke:entry", "smoke test started", {"cwd": os.getcwd()})

    try:
        import torch
        from transformers import __version__ as transformers_version

        debug_log(
            "H2",
            "smoke:torch",
            "torch environment",
            {
                "cuda_available": torch.cuda.is_available(),
                "cuda_device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                "torch_version": torch.__version__,
                "transformers_version": transformers_version,
            },
        )
    except Exception as exc:
        debug_log("H1", "smoke:torch_fail", "torch import failed", {"error": str(exc), "type": type(exc).__name__})
        raise

    try:
        import accelerate
        import bitsandbytes as bnb

        debug_log(
            "H1",
            "smoke:deps",
            "gpu deps versions",
            {
                "accelerate": accelerate.__version__,
                "bitsandbytes": bnb.__version__,
            },
        )
    except Exception as exc:
        debug_log("H1", "smoke:deps_fail", "dependency import failed", {"error": str(exc), "type": type(exc).__name__})
        raise

    policy_setup = os.environ.get("OPENVLA_UNNORM_KEY", "bridge_orig")
    debug_log("H4", "smoke:config", "adapter config", {"policy_setup": policy_setup})

    try:
        from src.vla.openvla_adapter_4bit import OpenVLA4BitAdapter

        debug_log("H3", "smoke:load_start", "loading OpenVLA4BitAdapter")
        adapter = OpenVLA4BitAdapter(policy_setup=policy_setup)
        debug_log("H3", "smoke:load_ok", "model loaded")
    except Exception as exc:
        debug_log(
            "H3",
            "smoke:load_fail",
            "model load failed",
            {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()[-800:]},
        )
        raise

    try:
        debug_log("H5", "smoke:predict_start", "running predict_action")
        action = adapter.predict_action(np.zeros((224, 224, 3), dtype=np.uint8), "pick up the spoon")
        debug_log("H5", "smoke:predict_ok", "predict_action succeeded", {"action_shape": list(action.shape)})
        print("OpenVLA 4-bit OK. Action shape:", action.shape)
        return 0
    except Exception as exc:
        debug_log(
            "H5",
            "smoke:predict_fail",
            "predict_action failed",
            {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()[-800:]},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
