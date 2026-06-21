from __future__ import annotations

import numpy as np

from src.debug_log import debug_log
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
            from transformers import AutoProcessor, BitsAndBytesConfig

            from src.vla.openvla_common import get_openvla_auto_model_class

            AutoModelForVision2Seq = get_openvla_auto_model_class()
        except ImportError as exc:
            debug_log("H1", "openvla_4bit:_load_model", "import failed", {"error": str(exc)})
            raise ImportError(
                "OpenVLA 4-bit requires GPU dependencies. "
                "Install with: pip install -r requirements-kaggle.txt "
                "(re-run after ManiSkill install if transformers was upgraded)."
            ) from exc

        debug_log(
            "H2",
            "openvla_4bit:_load_model",
            "pre-load env",
            {
                "cuda_available": torch.cuda.is_available(),
                "model_id": self.model_id,
                "policy_setup": self.policy_setup,
            },
        )

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4",
        )

        try:
            self._processor = AutoProcessor.from_pretrained(self.model_id, trust_remote_code=True)
            debug_log("H3", "openvla_4bit:_load_model", "processor loaded")
            self._model = AutoModelForVision2Seq.from_pretrained(
                self.model_id,
                torch_dtype=torch.bfloat16,
                quantization_config=bnb_config,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
                device_map="auto",
                attn_implementation="sdpa",
            )
            debug_log(
                "H3",
                "openvla_4bit:_load_model",
                "model loaded",
                {"device_map": getattr(self._model, "hf_device_map", None)},
            )
        except Exception as exc:
            import traceback

            debug_log(
                "H3",
                "openvla_4bit:_load_model",
                "from_pretrained failed",
                {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()[-800:]},
            )
            raise

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
