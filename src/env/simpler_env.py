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
                "SIMPLER is not installed. Clone https://github.com/simpler-env/SimplerEnv "
                "and run `pip install -e .` inside that repo."
            ) from exc

        # SimplerEnv API may vary by version; try common entry points.
        if hasattr(simpler_env, "make"):
            self.env = simpler_env.make(self.task_name)
        else:
            from simpler_env.utils.env.observation_utils import get_image_from_maniskill2_obs_dict  # type: ignore

            self._get_image = get_image_from_maniskill2_obs_dict
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
