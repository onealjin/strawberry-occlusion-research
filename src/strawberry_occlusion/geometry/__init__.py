"""Geometry primitives for attachment zones and cutlines."""

from strawberry_occlusion.geometry.mask_cutline import (
    MaskCutlineParameters,
    MaskCutlineResult,
    clip_infinite_line_to_image,
    estimate_visible_mask_cutline,
)
from strawberry_occlusion.geometry.primitives import (
    LineSegment,
    Point,
    distance,
    midpoint,
)

__all__ = [
    "LineSegment",
    "MaskCutlineParameters",
    "MaskCutlineResult",
    "Point",
    "clip_infinite_line_to_image",
    "distance",
    "estimate_visible_mask_cutline",
    "midpoint",
]
