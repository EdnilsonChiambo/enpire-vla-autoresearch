from __future__ import annotations

from dataclasses import dataclass

import gymnasium as gym
import gym_pusht  # noqa: F401
import numpy as np


@dataclass
class EpisodeResult:
    success: bool
    steps: int
    total_reward: float


class PushTEnvWrapper:
    """Gym-PushT wrapper exposing pixel observations and binary success."""

    def __init__(self):
        self.env = gym.make(
            "gym_pusht/PushT-v0",
            obs_type="pixels",
            render_mode="rgb_array",
        )
        self.action_dim = 2
        self.max_episode_steps = 300

    @property
    def name(self) -> str:
        return "pusht"

    def reset(self):
        obs, info = self.env.reset()
        return self._extract_image(obs), info

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32).reshape(-1)[:2]
        obs, reward, terminated, truncated, info = self.env.step(action)
        image = self._extract_image(obs)
        done = terminated or truncated
        success = bool(reward >= 1.0 or info.get("success", terminated))
        info = dict(info)
        info["success"] = success
        return image, float(reward), done, info

    def close(self):
        self.env.close()

    @staticmethod
    def _extract_image(obs) -> np.ndarray:
        if isinstance(obs, dict):
            obs = obs.get("pixels", obs.get("image", next(iter(obs.values()))))
        image = np.asarray(obs)
        if image.dtype != np.uint8:
            image = (np.clip(image, 0, 1) * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)
        return image
