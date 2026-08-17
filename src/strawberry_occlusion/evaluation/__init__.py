"""Checkpoint evaluation utilities for semantic segmentation."""

from importlib import import_module
from types import ModuleType

__all__ = ["cutline", "segmentation"]
_LAZY_MODULES = frozenset(
    (*__all__, "action_failure_signals", "occlusion_robustness", "selective_action")
)


def __getattr__(name: str) -> ModuleType:
    """Load evaluation submodules lazily to keep CPU geometry imports lightweight."""

    if name in _LAZY_MODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
