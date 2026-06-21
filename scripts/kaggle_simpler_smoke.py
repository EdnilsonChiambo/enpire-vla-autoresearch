"""Instrumented smoke test for SIMPLER ManiSkill3 (run on Kaggle GPU)."""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

from src.debug_log import debug_log


def main() -> int:
    simpler_dir = os.environ.get("SIMPLER_DIR", "/kaggle/working/SimplerEnv")
    debug_log("H1", "simpler_smoke:entry", "smoke test started", {"simpler_dir": simpler_dir})

    try:
        import simpler_env

        debug_log(
            "H1",
            "simpler_smoke:import_ok",
            "simpler_env import succeeded",
            {"file": getattr(simpler_env, "__file__", None)},
        )
    except ModuleNotFoundError as exc:
        debug_log(
            "H1",
            "simpler_smoke:import_fail",
            "simpler_env not found in fresh python",
            {"error": str(exc), "sys_path_head": sys.path[:5]},
        )
        raise

    try:
        from src.env.maniskill_assets import configure_maniskill_runtime, ensure_maniskill3_assets

        configure_maniskill_runtime()
        ensure_maniskill3_assets("widowx_spoon_on_towel")

        import gymnasium as gym
        import mani_skill
        import mani_skill.envs.tasks.digital_twins.bridge_dataset_eval  # noqa: F401

        debug_log("H2", "simpler_smoke:deps_ok", "mani_skill deps imported")
    except Exception as exc:
        debug_log(
            "H2",
            "simpler_smoke:deps_fail",
            "mani_skill import failed",
            {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()[-800:]},
        )
        raise

    try:
        env = gym.make("PutSpoonOnTableClothInScene-v1", obs_mode="rgb+segmentation", num_envs=1)
        obs, _info = env.reset()
        debug_log(
            "H3",
            "simpler_smoke:env_ok",
            "gym env reset succeeded",
            {"obs_keys": list(obs.keys()) if isinstance(obs, dict) else str(type(obs))},
        )
        env.close()
        print("SIMPLER ManiSkill3 env OK")
        return 0
    except Exception as exc:
        debug_log(
            "H3",
            "simpler_smoke:env_fail",
            "gym env reset failed",
            {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()[-1200:]},
        )
        raise


if __name__ == "__main__":
    raise SystemExit(main())
