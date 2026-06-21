from __future__ import annotations

import os

import numpy as np


MANISKILL3_ENV_IDS = {
    "widowx_spoon_on_towel": "PutSpoonOnTableClothInScene-v1",
    "widowx_carrot_on_plate": "PutCarrotOnPlateInScene-v1",
    "widowx_stack_cube": "StackGreenCubeOnYellowCubeBakedTexInScene-v1",
    "widowx_put_eggplant_in_basket": "PutEggplantInBasketScene-v1",
}


def _to_bool(value) -> bool:
    if hasattr(value, "any"):
        return bool(value.any())
    if hasattr(value, "item"):
        return bool(value.item())
    return bool(value)


def _to_float(value) -> float:
    if hasattr(value, "mean"):
        return float(value.mean())
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


class SimplerEnvWrapper:
    """SIMPLER wrapper supporting ManiSkill2 (py3.10) and ManiSkill3 (py3.12/Kaggle)."""

    def __init__(
        self,
        task: str = "widowx_spoon_on_towel",
        simpler_backend: str | None = None,
    ):
        self.task_name = task
        self.backend = (simpler_backend or os.getenv("SIMPLER_BACKEND", "maniskill3")).lower()
        self.env = None
        self.action_dim = 7
        self.max_episode_steps = 120
        self._get_image_ms3 = None
        self._load_env()

    def _load_env(self) -> None:
        if self.backend in ("maniskill3", "ms3"):
            self._load_maniskill3()
        elif self.backend in ("maniskill2", "ms2", "main"):
            self._load_maniskill2()
        else:
            raise ValueError(f"Unknown simpler_backend: {self.backend}")

    def _load_maniskill3(self) -> None:
        try:
            import gymnasium as gym
            import mani_skill.envs.tasks.digital_twins.bridge_dataset_eval  # noqa: F401
            from simpler_env.utils.env.observation_utils import get_image_from_maniskill3_obs_dict
        except ImportError as exc:
            raise ImportError(
                "ManiSkill3 SIMPLER is not installed. On Kaggle (Python 3.12) run:\n"
                "  git clone -b maniskill3 --depth 1 https://github.com/simpler-env/SimplerEnv\n"
                "  pip install git+https://github.com/haosulab/ManiSkill.git tyro==0.8.5\n"
                "  pip install -e SimplerEnv\n"
                f"Original error: {exc}"
            ) from exc

        env_id = MANISKILL3_ENV_IDS.get(self.task_name, "PutSpoonOnTableClothInScene-v1")
        self._get_image_ms3 = get_image_from_maniskill3_obs_dict
        self.env = gym.make(env_id, obs_mode="rgb", num_envs=1)

    def _load_maniskill2(self) -> None:
        try:
            import simpler_env  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "ManiSkill2 SIMPLER requires Python 3.10/3.11. Run:\n"
                "  git clone --recurse-submodules https://github.com/simpler-env/SimplerEnv\n"
                "  pip install 'numpy>=1.26,<2.0'\n"
                "  pip install -e SimplerEnv/ManiSkill2_real2sim\n"
                "  pip install -e SimplerEnv\n"
                f"Original error: {exc}"
            ) from exc

        self.env = simpler_env.make(self.task_name)

    @property
    def name(self) -> str:
        return f"simpler-{self.backend}"

    def reset(self):
        obs, info = self.env.reset()
        if self.backend in ("maniskill3", "ms3"):
            return self._get_image_ms3(self.env, obs), info
        return self._extract_image(obs), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        obs, reward, terminated, truncated, info = self.env.step(action)
        done = _to_bool(terminated) or _to_bool(truncated)

        if self.backend in ("maniskill3", "ms3"):
            success = _to_bool(info.get("success", False)) if isinstance(info, dict) else False
            image = self._get_image_ms3(self.env, obs)
        else:
            success = bool(info.get("success", _to_float(reward) > 0))
            image = self._extract_image(obs)

        info = dict(info) if isinstance(info, dict) else {}
        info["success"] = success
        return image, _to_float(reward), done, info

    def close(self):
        if self.env is not None:
            self.env.close()

    @staticmethod
    def _extract_image(obs) -> np.ndarray:
        if isinstance(obs, dict):
            for key in ("image", "rgb", "agentview_image", "image_primary"):
                if key in obs:
                    return np.asarray(obs[key], dtype=np.uint8)
            if "images" in obs:
                images = obs["images"]
                if isinstance(images, dict) and images:
                    return np.asarray(next(iter(images.values())), dtype=np.uint8)
        return np.asarray(obs, dtype=np.uint8)
