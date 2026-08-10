"""Deterministic public-safe augmentation utilities."""

from strawberry_occlusion.augmentation.synthetic_occluder import (
    MATCHED_REGIONS,
    MatchedOccluderSet,
    OccluderPlacement,
    SyntheticOccluder,
    apply_occluder,
    build_matched_occluder_set,
    derive_attachment_roi,
    generate_synthetic_occluder,
)

__all__ = [
    "MATCHED_REGIONS",
    "MatchedOccluderSet",
    "OccluderPlacement",
    "SyntheticOccluder",
    "apply_occluder",
    "build_matched_occluder_set",
    "derive_attachment_roi",
    "generate_synthetic_occluder",
]
