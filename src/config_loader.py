from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "config" / "default.yaml"


def load_config(config_path: Path | None = None) -> dict[str, Any]:
    load_dotenv(PROJECT_ROOT / ".env")
    path = config_path or DEFAULT_CONFIG_PATH
    with open(path, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    config["env"]["backend"] = os.getenv("ENV_BACKEND", config["env"]["backend"])
    config["vla"]["backend"] = os.getenv("VLA_BACKEND", config["vla"]["backend"])
    return config


def resolve_path(relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path
