from __future__ import annotations

import numpy as np

from src.vla.base import VLAAdapter
from src.vla.openvla_common import predict_openvla_action


class OpenVLA4BitAdapter(VLAAdapter):
    """OpenVLA-7B with 4-bit quantization for 16GB GPUs (Kaggle T4, Colab T4)."""

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
            from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
        except ImportError as exc:
            raise ImportError(
                "OpenVLA 4-bit requires GPU dependencies. "
                "Install with: pip install -r requirements-kaggle.txt"
            ) from exc

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
        self._model = AutoModelForVision2Seq.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=bnb_config,
            low_cpu_mem_usage=True,
            trust_remote_code=True,
            device_map="auto",
            attn_implementation="sdpa",
        )

    @property
    def name(self) -> str:
        return "openvla-4bit"

    def predict_action(self, image: np.ndarray, instruction: str) -> np.ndarray:
        if self._model is None or self._processor is None:
            raise RuntimeError("OpenVLA 4-bit model is not loaded.")

        return predict_openvla_action(
            self._model,
            self._processor,
            image,
            instruction,
            self.policy_setup,
        )
