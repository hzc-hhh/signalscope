"""YAML/JSON configuration loader."""

from pathlib import Path
from typing import Any, Dict, Optional

import yaml


def load_config(path: str) -> Dict[str, Any]:
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

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}
