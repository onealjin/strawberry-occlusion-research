"""Configuration loading helpers."""

from pathlib import Path
from typing import Any

import yaml


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML configuration file and require a mapping at the top level."""

    with Path(path).open("r", encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file) or {}

    if not isinstance(config, dict):
        raise ValueError("Configuration files must contain a top-level mapping.")

    return config
