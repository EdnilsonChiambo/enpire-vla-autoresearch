from __future__ import annotations

import numpy as np

from src.vla.base import VLAAdapter


class OpenVLAAdapter(VLAAdapter):
    """OpenVLA inference adapter for SIMPLER / bridge-style tasks (GPU required)."""

    def __init__(
        self,
        model_id: str = "openvla/openvla-7b",
        policy_setup: str = "widowx_bridge",
    ):
        self.model_id = model_id
        self.policy_setup = policy_setup
        self._model = None
        self._processor = None
        self._load_model()

    def _load_model(self) -> None:
        try:
            import torch
            from transformers import AutoProcessor

            from src.vla.openvla_common import get_openvla_auto_model_class

            AutoModelForVision2Seq = get_openvla_auto_model_class()
        except ImportError as exc:
            raise ImportError(
                "OpenVLA requires GPU dependencies. Install with: pip install -r requirements-gpu.txt"
            ) from exc

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        if torch.cuda.is_available():
            self._model = self._model.to("cuda:0")

    @property
    def name(self) -> str:
        return "openvla"

    def predict_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        from src.vla.openvla_common import predict_openvla_action

        if self._model is None or self._processor is None:
            raise RuntimeError("OpenVLA model is not loaded.")

        return predict_openvla_action(
            self._model,
            self._processor,
            image,
            instruction,
            self.policy_setup,
        )
