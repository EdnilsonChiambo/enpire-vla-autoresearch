from __future__ import annotations

import numpy as np

from src.vla.base import VLAAdapter


class OctoAdapter(VLAAdapter):
    """Octo VLA adapter (GPU recommended). Phase 2 scaffold."""

    def __init__(self, model_id: str = "hf://rail-berkeley/octo-small-1.5"):
        self.model_id = model_id
        self._model = None
        self._rng = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import jax
            from octo.model.octo_model import OctoModel
        except ImportError as exc:
            raise ImportError(
                "Octo requires JAX and the octo package. See README for setup instructions."
            ) from exc

        self._model = OctoModel.load_pretrained(self.model_id)
        self._rng = jax.random.PRNGKey(0)

    @property
    def name(self) -> str:
        return "octo"

    def predict_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        import jax

        if self._model is None:
            raise RuntimeError("Octo model is not loaded.")

        observation = {"image_primary": image[None, ...]}
        task = self._model.create_tasks(texts=[instruction])
        self._rng, subkey = jax.random.split(self._rng)
        action = self._model.sample_actions(observation, task, rng=subkey)
        return np.asarray(action, dtype=np.float32).reshape(-1)
