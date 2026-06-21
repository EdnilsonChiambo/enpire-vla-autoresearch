from src.env.pusht_env import PushTEnvWrapper
from src.env.simpler_env import SimplerEnvWrapper

__all__ = ["PushTEnvWrapper", "SimplerEnvWrapper", "create_env"]


def create_env(backend: str, simpler_task: str | None = None, **kwargs):
    backend = backend.lower()
    if backend == "pusht":
        return PushTEnvWrapper()
    if backend == "simpler":
        return SimplerEnvWrapper(task=simpler_task or "widowx_spoon_on_towel")
    raise ValueError(f"Unknown env backend: {backend}")
