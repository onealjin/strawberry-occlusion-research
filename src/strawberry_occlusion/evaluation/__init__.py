"""Checkpoint evaluation utilities for semantic segmentation."""

from importlib import import_module
from types import ModuleType

__all__ = ["cutline", "segmentation"]


def __getattr__(name: str) -> ModuleType:
    """Load evaluation submodules lazily to keep CPU geometry imports lightweight."""

    if name in __all__:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
