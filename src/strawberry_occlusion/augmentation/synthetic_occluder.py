"""Deterministic Level-1 synthetic occluders and matched region placement.

The procedural object is generated once for a sample/severity/seed set. Region
conditions translate that exact discrete RGB template without clipping,
resizing, rotating, or regenerating it.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Literal

import numpy as np

from strawberry_occlusion.geometry import FixedAxisCutlineResult


RegionName = Literal["attachment", "calyx_tip", "flesh_far", "background"]
MATCHED_REGIONS: tuple[RegionName, ...] = (
    "attachment",
    "calyx_tip",
    "flesh_far",
    "background",
)


@dataclass(frozen=True)
class SyntheticOccluder:
    """One immutable opaque synthetic appearance in local template coordinates."""

    occluder_id: str
    template_sha256: str
    target_area_pixels: int
    rotation_degrees: float
    scale_pixels: float
    opacity: float
    mask: np.ndarray
    rgb: np.ndarray

    @property
    def area_pixels(self) -> int:
        """Return the exact number of opaque template pixels."""

        return int(np.count_nonzero(self.mask))


@dataclass(frozen=True)
class OccluderPlacement:
    """An integer translation of a synthetic template into image coordinates."""

    region: RegionName
    top: int
    left: int
    area_pixels: int
    template_sha256: str


@dataclass(frozen=True)
class MatchedOccluderSet:
    """Complete matched translations or structured region-unavailability reasons."""

    complete: bool
    occluder: SyntheticOccluder
    placements: dict[RegionName, OccluderPlacement]
    unavailable_reasons: dict[RegionName, str]
    attachment_roi: np.ndarray


def generate_synthetic_occluder(
    *,
    sample_id: str,
    severity_fraction: float,
    seed: int,
    foreground_area_pixels: int,
    profile_name: str = "m4_v1",
) -> SyntheticOccluder:
    """Create an exact-area, tapered leaf-like, public-safe opaque template."""

    if not isinstance(sample_id, str) or not sample_id:
        raise ValueError("sample_id must be a non-empty string")
    if (
        isinstance(severity_fraction, bool)
        or not isinstance(severity_fraction, (int, float))
        or not math.isfinite(float(severity_fraction))
        or not 0.0 < float(severity_fraction) < 1.0
    ):
        raise ValueError("severity_fraction must be finite and in (0, 1)")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if (
        isinstance(foreground_area_pixels, bool)
        or not isinstance(foreground_area_pixels, int)
        or foreground_area_pixels <= 0
    ):
        raise ValueError("foreground_area_pixels must be a positive integer")

    target_area = max(1, int(round(foreground_area_pixels * severity_fraction)))
    identity_material = (
        f"{profile_name}|{sample_id}|{severity_fraction:.12g}|{seed}|{target_area}"
    )
    identity_digest = hashlib.sha256(identity_material.encode("utf-8")).hexdigest()
    rng_seed = int(identity_digest[:16], 16)
    rng = np.random.default_rng(rng_seed)

    aspect_ratio = float(rng.uniform(1.75, 2.35))
    fill_fraction = 0.60
    height = max(
        3, int(math.ceil(math.sqrt(target_area / (aspect_ratio * fill_fraction))))
    )
    width = max(3, int(math.ceil(height * aspect_ratio)))
    while height * width < target_area:
        width += 1

    y, x = np.indices((height, width), dtype=np.float64)
    centre_x = (width - 1) / 2.0
    centre_y = (height - 1) / 2.0
    normalized_x = (x - centre_x) / max(centre_x, 1.0)
    normalized_y = (y - centre_y) / max(centre_y, 1.0)
    rotation_degrees = float(rng.uniform(-24.0, 24.0))
    angle = math.radians(rotation_degrees)
    rotated_x = normalized_x * math.cos(angle) + normalized_y * math.sin(angle)
    rotated_y = -normalized_x * math.sin(angle) + normalized_y * math.cos(angle)

    taper = np.clip(1.0 - 0.38 * rotated_x, 0.48, 1.25)
    radial_score = 1.0 - (rotated_x**2 + (rotated_y / taper) ** 2)
    edge_ripple = 0.025 * np.sin(7.0 * rotated_y + rng.uniform(0.0, 2.0 * math.pi))
    score = radial_score + edge_ripple + rng.normal(0.0, 0.0025, (height, width))
    centre_distance = (x - centre_x) ** 2 + (y - centre_y) ** 2
    order = np.lexsort((x.ravel(), y.ravel(), centre_distance.ravel(), -score.ravel()))
    selected = order[:target_area]
    mask = np.zeros((height, width), dtype=bool)
    mask.ravel()[selected] = True

    # Deterministic non-photorealistic green/brown surface with a central vein.
    longitudinal = (rotated_x + 1.0) / 2.0
    vein = np.exp(-((rotated_y / 0.09) ** 2))
    texture_noise = rng.normal(0.0, 5.0, (height, width))
    red = 54.0 + 42.0 * longitudinal + 24.0 * vein + texture_noise
    green = 112.0 + 46.0 * (1.0 - longitudinal) + 20.0 * vein + texture_noise
    blue = 38.0 + 18.0 * (1.0 - longitudinal) + 8.0 * vein + texture_noise / 2.0
    rgb = np.rint(np.stack((red, green, blue), axis=2)).clip(0, 255).astype(np.uint8)
    rgb[~mask] = 0

    digest = hashlib.sha256()
    digest.update(np.asarray(mask, dtype=np.uint8).tobytes(order="C"))
    digest.update(rgb.tobytes(order="C"))
    template_hash = digest.hexdigest()
    occluder_id = f"occ_{identity_digest[:12]}_{template_hash[:12]}"
    scale_pixels = math.sqrt(target_area / math.pi)
    mask.setflags(write=False)
    rgb.setflags(write=False)
    return SyntheticOccluder(
        occluder_id=occluder_id,
        template_sha256=template_hash,
        target_area_pixels=target_area,
        rotation_degrees=rotation_degrees,
        scale_pixels=scale_pixels,
        opacity=1.0,
        mask=mask,
        rgb=rgb,
    )


def derive_attachment_roi(
    result: FixedAxisCutlineResult,
    *,
    dilation_radius_pixels: int = 5,
) -> np.ndarray:
    """Return a fixed clean contact ROI derived only from manual-mask geometry."""

    if not isinstance(result, FixedAxisCutlineResult):
        raise TypeError("result must be a FixedAxisCutlineResult")
    if (
        isinstance(dilation_radius_pixels, bool)
        or not isinstance(dilation_radius_pixels, int)
        or dilation_radius_pixels < 0
    ):
        raise ValueError("dilation_radius_pixels must be a non-negative integer")
    contact = np.asarray(result.contact_mask, dtype=bool)
    roi = _square_dilation(contact, radius=dilation_radius_pixels)
    roi.setflags(write=False)
    return roi


def build_matched_occluder_set(
    clean_mask: np.ndarray,
    clean_v2a: FixedAxisCutlineResult,
    occluder: SyntheticOccluder,
    *,
    attachment_roi_dilation_pixels: int,
    background_separation_pixels: int,
) -> MatchedOccluderSet:
    """Translate one template to all frozen regions or return an incomplete set."""

    mask = _validate_semantic_mask(clean_mask)
    if not isinstance(clean_v2a, FixedAxisCutlineResult):
        raise TypeError("clean_v2a must be a FixedAxisCutlineResult")
    if (clean_v2a.image_height, clean_v2a.image_width) != mask.shape:
        raise ValueError("clean_v2a dimensions must match clean_mask")
    if not isinstance(occluder, SyntheticOccluder):
        raise TypeError("occluder must be a SyntheticOccluder")
    if occluder.area_pixels != occluder.target_area_pixels:
        raise ValueError("occluder mask area does not match its target area")

    attachment_roi = derive_attachment_roi(
        clean_v2a,
        dilation_radius_pixels=attachment_roi_dilation_pixels,
    )
    reasons: dict[RegionName, str] = {}
    placements: dict[RegionName, OccluderPlacement] = {}
    if not np.any(clean_v2a.contact_mask):
        reasons["attachment"] = "attachment_contact_unavailable"
    else:
        placement = _target_centred_placement(
            "attachment",
            attachment_roi,
            occluder,
            ranking="centroid",
        )
        if placement is None:
            reasons["attachment"] = "attachment_valid_translation_unavailable"
        else:
            placements["attachment"] = placement

    separation = max(3.0, float(math.ceil(occluder.scale_pixels)))
    axis = clean_v2a.normalized_removal_axis
    anchor_projection = (
        _point_projection(clean_v2a.baseline_v1_contact_anchor, axis)
        if clean_v2a.baseline_v1_contact_anchor is not None
        else None
    )

    distal_calyx = (
        np.asarray(clean_v2a.selected_calyx_mask, dtype=bool) & ~attachment_roi
    )
    if anchor_projection is not None:
        distal_calyx &= _projection_distance_mask(
            mask.shape,
            axis,
            reference=anchor_projection,
            minimum_distance=separation,
            direction="outward",
        )
    placement = _target_centred_placement(
        "calyx_tip",
        distal_calyx,
        occluder,
        ranking="outward",
        axis=axis,
        forbidden_overlap_mask=attachment_roi,
    )
    if placement is None:
        reasons["calyx_tip"] = "distal_calyx_support_unavailable"
    else:
        placements["calyx_tip"] = placement

    flesh_far = np.asarray(clean_v2a.selected_flesh_mask, dtype=bool) & ~attachment_roi
    if anchor_projection is not None:
        flesh_far &= _projection_distance_mask(
            mask.shape,
            axis,
            reference=anchor_projection,
            minimum_distance=separation,
            direction="inward",
        )
    interior_flesh = _binary_erosion(flesh_far, radius=1)
    placement = _target_centred_placement(
        "flesh_far",
        interior_flesh if np.any(interior_flesh) else flesh_far,
        occluder,
        ranking="inward",
        axis=axis,
        forbidden_overlap_mask=attachment_roi,
    )
    if placement is None:
        reasons["flesh_far"] = "separated_flesh_support_unavailable"
    else:
        placements["flesh_far"] = placement

    foreground = np.isin(mask, (1, 2))
    forbidden_background = _square_dilation(
        foreground,
        radius=background_separation_pixels,
    )
    placement = _background_placement(
        forbidden_background,
        occluder,
    )
    if placement is None:
        reasons["background"] = "separated_background_placement_unavailable"
    else:
        placements["background"] = placement

    regions_by_location: dict[tuple[int, int], list[RegionName]] = {}
    for region, placement in placements.items():
        regions_by_location.setdefault((placement.top, placement.left), []).append(
            region
        )
    duplicated_regions = {
        region
        for regions in regions_by_location.values()
        if len(regions) > 1
        for region in regions
    }
    for region in duplicated_regions:
        reasons[region] = "placement_not_spatially_distinct"
        placements.pop(region)

    complete = len(placements) == len(MATCHED_REGIONS) and not reasons
    _validate_matched_placements(
        mask,
        attachment_roi,
        occluder,
        placements,
    )
    return MatchedOccluderSet(
        complete=complete,
        occluder=occluder,
        placements=placements,
        unavailable_reasons=reasons,
        attachment_roi=attachment_roi,
    )


def apply_occluder(
    image: np.ndarray,
    occluder: SyntheticOccluder,
    placement: OccluderPlacement,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply one opaque template translation and return image plus full mask."""

    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if not np.issubdtype(array.dtype, np.number):
        raise TypeError("image must use a numeric dtype")
    if placement.template_sha256 != occluder.template_sha256:
        raise ValueError("placement template hash does not match occluder")
    height, width = occluder.mask.shape
    if (
        placement.top < 0
        or placement.left < 0
        or placement.top + height > array.shape[0]
        or placement.left + width > array.shape[1]
    ):
        raise ValueError("occluder placement would clip at image bounds")

    output = np.array(array, copy=True)
    full_mask = np.zeros(array.shape[:2], dtype=bool)
    row_slice = slice(placement.top, placement.top + height)
    column_slice = slice(placement.left, placement.left + width)
    local_output = output[row_slice, column_slice]
    if np.issubdtype(output.dtype, np.floating):
        texture = occluder.rgb.astype(output.dtype) / 255.0
    else:
        texture = occluder.rgb.astype(output.dtype)
    local_output[occluder.mask] = texture[occluder.mask]
    full_mask[row_slice, column_slice] = occluder.mask
    if int(np.count_nonzero(full_mask)) != occluder.area_pixels:
        raise ValueError("translated occluder area changed unexpectedly")
    return output, full_mask


def _target_centred_placement(
    region: RegionName,
    target: np.ndarray,
    occluder: SyntheticOccluder,
    *,
    ranking: str,
    axis: tuple[float, float] = (1.0, 0.0),
    forbidden_overlap_mask: np.ndarray | None = None,
) -> OccluderPlacement | None:
    candidates = np.argwhere(target)
    if not len(candidates):
        return None
    local_anchor = _template_anchor(occluder.mask)
    valid: list[tuple[float, float, int, int]] = []
    centroid_y, centroid_x = candidates.mean(axis=0)
    image_height, image_width = target.shape
    template_height, template_width = occluder.mask.shape
    forbidden = None
    if forbidden_overlap_mask is not None:
        forbidden = np.asarray(forbidden_overlap_mask, dtype=bool)
        if forbidden.shape != target.shape:
            raise ValueError("forbidden_overlap_mask must match the target shape")
    for y_value, x_value in candidates:
        y = int(y_value)
        x = int(x_value)
        top = y - local_anchor[0]
        left = x - local_anchor[1]
        if (
            top < 0
            or left < 0
            or top + template_height > image_height
            or left + template_width > image_width
        ):
            continue
        if forbidden is not None and np.any(
            forbidden[
                top : top + template_height,
                left : left + template_width,
            ][occluder.mask]
        ):
            continue
        projection = x * axis[0] + y * axis[1]
        centroid_distance = (x - centroid_x) ** 2 + (y - centroid_y) ** 2
        if ranking == "outward":
            key = (-projection, centroid_distance, y, x)
        elif ranking == "inward":
            key = (projection, centroid_distance, y, x)
        else:
            key = (centroid_distance, 0.0, y, x)
        valid.append(key)
    if not valid:
        return None
    selected = min(valid)
    y = int(selected[2])
    x = int(selected[3])
    return OccluderPlacement(
        region=region,
        top=y - local_anchor[0],
        left=x - local_anchor[1],
        area_pixels=occluder.area_pixels,
        template_sha256=occluder.template_sha256,
    )


def _background_placement(
    forbidden: np.ndarray,
    occluder: SyntheticOccluder,
) -> OccluderPlacement | None:
    template_height, template_width = occluder.mask.shape
    image_height, image_width = forbidden.shape
    if template_height > image_height or template_width > image_width:
        return None
    integral = np.pad(
        forbidden.astype(np.int64).cumsum(axis=0).cumsum(axis=1),
        ((1, 0), (1, 0)),
    )
    window_sums = (
        integral[template_height:, template_width:]
        - integral[:-template_height, template_width:]
        - integral[template_height:, :-template_width]
        + integral[:-template_height, :-template_width]
    )
    locations = np.argwhere(window_sums == 0)
    if not len(locations):
        return None
    top, left = (int(value) for value in locations[0])
    return OccluderPlacement(
        region="background",
        top=top,
        left=left,
        area_pixels=occluder.area_pixels,
        template_sha256=occluder.template_sha256,
    )


def _validate_matched_placements(
    clean_mask: np.ndarray,
    attachment_roi: np.ndarray,
    occluder: SyntheticOccluder,
    placements: dict[RegionName, OccluderPlacement],
) -> None:
    if any(
        placement.template_sha256 != occluder.template_sha256
        or placement.area_pixels != occluder.area_pixels
        for placement in placements.values()
    ):
        raise ValueError("matched placements changed occluder identity or area")
    dummy = np.zeros((*clean_mask.shape, 3), dtype=np.uint8)
    foreground = np.isin(clean_mask, (1, 2))
    for region, placement in placements.items():
        _, translated = apply_occluder(dummy, occluder, placement)
        if int(np.count_nonzero(translated)) != occluder.area_pixels:
            raise ValueError("matched placement changed visible occluder area")
        attachment_overlap = int(np.count_nonzero(translated & attachment_roi))
        foreground_overlap = int(np.count_nonzero(translated & foreground))
        if region in {"calyx_tip", "flesh_far"} and attachment_overlap != 0:
            raise ValueError(
                f"{region} occluder footprint must not overlap the attachment ROI"
            )
        if region == "background" and foreground_overlap != 0:
            raise ValueError(
                "background occluder footprint must not overlap foreground"
            )


def _template_anchor(mask: np.ndarray) -> tuple[int, int]:
    coordinates = np.argwhere(mask)
    centre = np.asarray([(mask.shape[0] - 1) / 2.0, (mask.shape[1] - 1) / 2.0])
    distances = ((coordinates - centre) ** 2).sum(axis=1)
    selected = coordinates[int(np.argmin(distances))]
    return int(selected[0]), int(selected[1])


def _projection_distance_mask(
    shape: tuple[int, int],
    axis: tuple[float, float],
    *,
    reference: float,
    minimum_distance: float,
    direction: Literal["inward", "outward"],
) -> np.ndarray:
    y, x = np.indices(shape, dtype=np.float64)
    projections = x * axis[0] + y * axis[1]
    if direction == "outward":
        return projections >= reference + minimum_distance
    return projections <= reference - minimum_distance


def _point_projection(point: object, axis: tuple[float, float]) -> float:
    return float(getattr(point, "x") * axis[0] + getattr(point, "y") * axis[1])


def _square_dilation(mask: np.ndarray, *, radius: int) -> np.ndarray:
    if radius == 0:
        return np.array(mask, dtype=bool, copy=True)
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.zeros((height, width), dtype=bool)
    for delta_y in range(2 * radius + 1):
        for delta_x in range(2 * radius + 1):
            output |= padded[delta_y : delta_y + height, delta_x : delta_x + width]
    return output


def _binary_erosion(mask: np.ndarray, *, radius: int) -> np.ndarray:
    if radius == 0:
        return np.array(mask, dtype=bool, copy=True)
    height, width = mask.shape
    padded = np.pad(mask, radius, mode="constant", constant_values=False)
    output = np.ones((height, width), dtype=bool)
    for delta_y in range(2 * radius + 1):
        for delta_x in range(2 * radius + 1):
            output &= padded[delta_y : delta_y + height, delta_x : delta_x + width]
    return output


def _validate_semantic_mask(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError("clean_mask must have shape [H, W]")
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype, np.integer
    ):
        raise TypeError("clean_mask must use an integer dtype")
    if not np.all(np.isin(array, (0, 1, 2))):
        raise ValueError("clean_mask must contain only class IDs 0, 1, and 2")
    return array
