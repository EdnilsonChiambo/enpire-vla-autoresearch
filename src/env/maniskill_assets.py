from __future__ import annotations

import os

# Bridge SIMPLER ManiSkill3 tasks share this asset bundle.
MANISKILL3_TASK_ASSETS: dict[str, list[str]] = {
    "widowx_spoon_on_towel": ["bridge_v2_real2sim"],
    "widowx_carrot_on_plate": ["bridge_v2_real2sim"],
    "widowx_stack_cube": ["bridge_v2_real2sim"],
    "widowx_put_eggplant_in_basket": ["bridge_v2_real2sim"],
}


def configure_maniskill_runtime() -> None:
    """Kaggle/subprocess-safe defaults for ManiSkill asset downloads."""
    os.environ.setdefault("MS_SKIP_ASSET_DOWNLOAD_PROMPT", "1")
    if os.path.isdir("/kaggle/working"):
        os.environ.setdefault("MS_ASSET_DIR", "/kaggle/working/.maniskill")


def ensure_maniskill3_assets(task: str) -> None:
    """Download ManiSkill task assets without interactive prompts."""
    configure_maniskill_runtime()

    import mani_skill.envs  # noqa: F401 — registers DATA_SOURCES
    from mani_skill.utils.assets.data import DATA_SOURCES, is_data_source_downloaded
    from mani_skill.utils.download_asset import download

    asset_uids = MANISKILL3_TASK_ASSETS.get(task, ["bridge_v2_real2sim"])
    for uid in asset_uids:
        if is_data_source_downloaded(uid):
            continue
        print(f"Downloading ManiSkill asset '{uid}' (one-time, ~few GB)...")
        download(DATA_SOURCES[uid], verbose=True, non_interactive=True)
        print(f"ManiSkill asset '{uid}' ready.")
