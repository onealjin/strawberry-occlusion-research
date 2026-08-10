"""Headless visualization helpers for segmentation quality assurance."""

from importlib import import_module
from types import ModuleType

__all__ = ["cutline", "segmentation"]
_LAZY_MODULES = frozenset((*__all__, "occlusion_robustness"))


def __getattr__(name: str) -> ModuleType:
    """Load visualization submodules lazily to avoid evaluation import cycles."""

    if name in _LAZY_MODULES:
        return import_module(f"{__name__}.{name}")
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
