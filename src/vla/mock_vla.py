from __future__ import annotations

import re

import numpy as np

from src.vla.base import VLAAdapter


class MockVLAAdapter(VLAAdapter):
    """CPU-friendly adapter for validating the autoresearch loop without a GPU."""

    def __init__(self, action_dim: int = 2, seed: int = 0):
        self._rng = np.random.default_rng(seed)
        self._action_dim = action_dim
        self._step = 0

    @property
    def name(self) -> str:
        return "mock"

    def predict_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        self._step += 1
        instruction = instruction.lower()

        if self._action_dim == 2:
            # PushT: bias toward center; stronger bias when prompt mentions alignment/care.
            center = np.array([256.0, 256.0], dtype=np.float32)
            noise = self._rng.normal(0, 40, size=2).astype(np.float32)
            action = center + noise
            if any(word in instruction for word in ("carefully", "align", "slow", "precise")):
                action = 0.7 * action + 0.3 * center
            return np.clip(action, 0, 512)

        # Generic 7-DoF fallback for SIMPLER-shaped actions.
        action = self._rng.normal(0, 0.05, size=self._action_dim).astype(np.float32)
        if "handle" in instruction or "grip" in instruction:
            action[-1] = 1.0
        return action


class HeuristicPushTVLA(MockVLAAdapter):
    """Optional heuristic that reacts to prompt keywords for more realistic mock evals."""

    KEYWORD_BIAS = {
        "left": np.array([-30.0, 0.0]),
        "right": np.array([30.0, 0.0]),
        "up": np.array([0.0, -30.0]),
        "down": np.array([0.0, 30.0]),
        "center": np.array([0.0, 0.0]),
    }

    def predict_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        base = super().predict_action(image, instruction)
        bias = np.zeros(2, dtype=np.float32)
        for keyword, delta in self.KEYWORD_BIAS.items():
            if keyword in instruction.lower():
                bias += delta.astype(np.float32)
        return np.clip(base + bias, 0, 512)


def extract_prompt_keywords(instruction: str) -> list[str]:
    return re.findall(r"[a-z]+", instruction.lower())
