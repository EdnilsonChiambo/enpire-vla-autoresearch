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
            from transformers import AutoModelForVision2Seq, AutoProcessor
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
        import torch
        from PIL import Image

        if self._model is None or self._processor is None:
            raise RuntimeError("OpenVLA model is not loaded.")

        pil_image = Image.fromarray(image.astype(np.uint8))
        prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

        inputs = self._processor(prompt, pil_image, return_tensors="pt")
        if torch.cuda.is_available():
            inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

        with torch.inference_mode():
            action = self._model.predict_action(**inputs, unnorm_key=self.policy_setup)

        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        return np.asarray(action, dtype=np.float32).reshape(-1)
