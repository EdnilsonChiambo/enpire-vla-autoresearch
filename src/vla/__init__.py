from src.vla.base import VLAAdapter
from src.vla.mock_vla import MockVLAAdapter

__all__ = ["VLAAdapter", "MockVLAAdapter", "create_vla_adapter"]


def create_vla_adapter(
    backend: str,
    model_id: str | None = None,
    policy_setup: str | None = None,
    action_dim: int = 2,
    **kwargs,
) -> VLAAdapter:
    backend = backend.lower()
    if backend == "mock":
        return MockVLAAdapter(action_dim=action_dim)
    if backend == "openvla":
        from src.vla.openvla_adapter import OpenVLAAdapter

        return OpenVLAAdapter(
            model_id=model_id or "openvla/openvla-7b",
            policy_setup=policy_setup or "widowx_bridge",
        )
    if backend == "octo":
        from src.vla.octo_adapter import OctoAdapter

        return OctoAdapter(model_id=model_id or "hf://rail-berkeley/octo-small-1.5")
    raise ValueError(f"Unknown VLA backend: {backend}")
