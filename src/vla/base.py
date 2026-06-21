from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class VLAAdapter(ABC):
    @abstractmethod
    def predict_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        """Return an action vector for the current observation and instruction."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier for logging."""
