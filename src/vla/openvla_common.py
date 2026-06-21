from __future__ import annotations

import numpy as np


def predict_openvla_action(model, processor, image: np.ndarray, instruction: str, policy_setup: str):
    import traceback

    import torch
    from PIL import Image

    from src.debug_log import debug_log

    pil_image = Image.fromarray(image.astype(np.uint8))
    prompt = f"In: What action should the robot take to {instruction.lower()}?\nOut:"

    try:
        inputs = processor(prompt, pil_image, return_tensors="pt")
        debug_log(
            "H5",
            "openvla_common:predict",
            "processor output",
            {"input_keys": list(inputs.keys()), "policy_setup": policy_setup},
        )
        device = getattr(model, "device", None)
        if device is None and hasattr(model, "hf_device_map"):
            device = next(model.parameters()).device

        debug_log("H5", "openvla_common:predict", "resolved device", {"device": str(device)})

        if device is not None:
            cast_inputs = {}
            for key, value in inputs.items():
                if torch.is_floating_point(value):
                    cast_inputs[key] = value.to(device, dtype=torch.bfloat16)
                else:
                    cast_inputs[key] = value.to(device)
            inputs = cast_inputs

        with torch.inference_mode():
            action = model.predict_action(**inputs, unnorm_key=policy_setup)

        if hasattr(action, "cpu"):
            action = action.cpu().numpy()
        result = np.asarray(action, dtype=np.float32).reshape(-1)
        debug_log("H5", "openvla_common:predict", "success", {"action_shape": list(result.shape)})
        return result
    except Exception as exc:
        debug_log(
            "H5",
            "openvla_common:predict",
            "predict failed",
            {"error": str(exc), "type": type(exc).__name__, "traceback": traceback.format_exc()[-800:]},
        )
        raise
