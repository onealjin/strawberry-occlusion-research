"""Geometry primitives for attachment zones and cutlines."""

from strawberry_occlusion.geometry.fixed_axis_cutline import (
    FixedAxisCutlineParameters,
    FixedAxisCutlineResult,
    ProjectedPointDiagnostics,
    estimate_fixed_axis_cutline,
)
from strawberry_occlusion.geometry.fixed_axis_search_cutline import (
    FixedAxisCandidateTable,
    FixedAxisSearchCutlineParameters,
    FixedAxisSearchCutlineResult,
    calculate_fixed_axis_candidate_curve,
    estimate_fixed_axis_search_cutline,
)
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
    "FixedAxisCutlineParameters",
    "FixedAxisCutlineResult",
    "FixedAxisCandidateTable",
    "FixedAxisSearchCutlineParameters",
    "FixedAxisSearchCutlineResult",
    "LineSegment",
    "MaskCutlineParameters",
    "MaskCutlineResult",
    "Point",
    "ProjectedPointDiagnostics",
    "clip_infinite_line_to_image",
    "calculate_fixed_axis_candidate_curve",
    "distance",
    "estimate_fixed_axis_cutline",
    "estimate_fixed_axis_search_cutline",
    "estimate_visible_mask_cutline",
    "midpoint",
]
