"""YAML/JSON configuration loader."""

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str) -> dict[str, Any]:
    """
    Load a YAML or JSON configuration file.

    Parameters
    ----------
    path : str
        Path to .yaml or .json file.

    Returns
    -------
    dict
    """
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(config_path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
