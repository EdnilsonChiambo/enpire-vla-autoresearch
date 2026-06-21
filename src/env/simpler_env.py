from __future__ import annotations

import numpy as np


class SimplerEnvWrapper:
    """SIMPLER environment wrapper (requires SimplerEnv installation + GPU)."""

    DEFAULT_TASKS = {
        "widowx_spoon_on_towel": "pick up the spoon",
        "widowx_carrot_on_plate": "pick up the carrot",
        "widowx_stack_cube": "stack the green block on the yellow block",
    }

    def __init__(self, task: str = "widowx_spoon_on_towel"):
        self.task_name = task
        self.env = None
        self.action_dim = 7
        self.max_episode_steps = 120
        self._load_env()

    def _load_env(self) -> None:
        try:
            import simpler_env  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "SIMPLER is not fully installed. Run:\n"
                "  git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv\n"
                "  pip install 'numpy>=1.26,<2.0'  # use 1.24.4 only on Python 3.10/3.11\n"
                "  pip install -e SimplerEnv/ManiSkill2_real2sim\n"
                "  pip install -e SimplerEnv\n"
                f"Original error: {exc}"
            ) from exc

        self.env = simpler_env.make(self.task_name)

    @property
    def name(self) -> str:
        return "simpler"

    def reset(self):
        obs, info = self.env.reset()
        return self._extract_image(obs), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        success = bool(info.get("success", reward > 0))
        info = dict(info)
        info["success"] = success
        return self._extract_image(obs), float(reward), done, info

    def close(self):
        if self.env is not None:
            self.env.close()

    def _extract_image(self, obs) -> np.ndarray:
        if isinstance(obs, dict):
            for key in ("image", "rgb", "agentview_image", "image_primary"):
                if key in obs:
                    return np.asarray(obs[key], dtype=np.uint8)
            if "images" in obs:
                images = obs["images"]
                if isinstance(images, dict) and images:
                    return np.asarray(next(iter(images.values())), dtype=np.uint8)
        return np.asarray(obs, dtype=np.uint8)
