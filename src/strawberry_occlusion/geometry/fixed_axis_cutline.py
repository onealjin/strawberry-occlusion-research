"""Experimental v2a fixed-axis robust-contact cutline geometry."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from math import acos, ceil, degrees, hypot, isfinite
from numbers import Real
from typing import Literal

import numpy as np

from strawberry_occlusion.geometry.mask_cutline import (
    MaskCutlineResult,
    _label_components,
    clip_infinite_line_to_image,
    estimate_visible_mask_cutline,
)
from strawberry_occlusion.geometry.primitives import LineSegment, Point


@dataclass(frozen=True)
class FixedAxisCutlineParameters:
    """Parameters controlling experimental v2a cut-coordinate estimation."""

    removal_axis: tuple[float, float]
    projection_quantile: float = 0.95
    support_band_width_pixels: float = 5.0
    signed_offset_pixels: float = 0.0
    calyx_dilation_radius: int = 1
    component_connectivity: Literal[4, 8] = 8


@dataclass(frozen=True)
class FixedAxisCutlineResult:
    """Geometry and diagnostics for one experimental v2a estimate.

    ``cut_margin_to_flesh_extent`` is the selected Flesh component's maximum
    projection minus the final cut coordinate. Positive values put the cut
    inside the selected Flesh projection extent, zero puts it at the outer
    extent, and negative values put it beyond that extent. These values are
    diagnostics and do not classify a cut as good or accurate.
    """

    status: Literal["ok", "failed"]
    failure_code: str | None
    failure_reason: str | None
    parameters: FixedAxisCutlineParameters
    image_width: int
    image_height: int
    normalized_removal_axis: tuple[float, float]
    flesh_pixel_count: int
    calyx_pixel_count: int
    flesh_component_count: int
    calyx_component_count: int
    selected_flesh_component: int | None
    selected_calyx_component: int | None
    selected_flesh_pixel_count: int
    selected_calyx_pixel_count: int
    contact_pixel_count: int
    support_pixel_count: int
    support_lateral_extent_pixels: float | None
    flesh_centroid: Point | None
    baseline_v1_contact_anchor: Point | None
    baseline_v1_fruit_to_attachment_direction: tuple[float, float] | None
    baseline_v1_cutline: LineSegment | None
    unshifted_cut_coordinate: float | None
    final_cut_coordinate: float | None
    signed_offset_pixels: float
    diagnostic_reference_point: Point | None
    candidate_cutline: LineSegment | None
    final_cutline: LineSegment | None
    minimum_contact_projection: float | None
    maximum_contact_projection: float | None
    selected_contact_projection: float | None
    axis_disagreement_angle_degrees: float | None
    maximum_selected_flesh_projection: float | None
    cut_margin_to_flesh_extent: float | None
    candidate_final_lines_coincide: bool | None
    flesh_mask: np.ndarray = field(repr=False, compare=False)
    calyx_mask: np.ndarray = field(repr=False, compare=False)
    selected_flesh_mask: np.ndarray = field(repr=False, compare=False)
    selected_calyx_mask: np.ndarray = field(repr=False, compare=False)
    contact_mask: np.ndarray = field(repr=False, compare=False)
    support_mask: np.ndarray = field(repr=False, compare=False)

    @property
    def succeeded(self) -> bool:
        """Return whether a final clipped fixed-axis cutline was produced."""

        return self.status == "ok"

    @property
    def contact_band(self) -> np.ndarray:
        """Return the selected v1 contact mask under its baseline name."""

        return self.contact_mask

    @property
    def selected_contact_mask(self) -> np.ndarray:
        """Return the original complete selected contact mask."""

        return self.contact_mask


def estimate_fixed_axis_cutline(
    mask: np.ndarray,
    *,
    removal_axis: Sequence[Real] | np.ndarray,
    projection_quantile: float = 0.95,
    support_band_width_pixels: float = 5.0,
    signed_offset_pixels: float = 0.0,
    calyx_dilation_radius: int = 1,
    component_connectivity: Literal[4, 8] = 8,
) -> FixedAxisCutlineResult:
    """Estimate one v2a cut coordinate along a configured removal axis.

    The removal axis is normalized internally and points from retained Flesh
    toward removed Calyx. Contact pixels use pixel-centre coordinates ``(x, y)``.
    Their projections are sorted, and the deterministic non-interpolated order
    statistic at index ``min(n - 1, ceil(projection_quantile * n) - 1)`` is
    selected. Consequently, the unshifted coordinate is always the projection
    of an actual contact pixel.

    The support mask contains selected contact pixels satisfying ``projection >=
    unshifted_cut_coordinate - support_band_width_pixels``. The support band is
    diagnostic only. It cannot affect the selected or final cut coordinate,
    whose line equation is ``dot(pixel, axis) = coordinate``.
    Coordinates remain in source-image pixels; no machine-coordinate conversion
    or calibration is performed.
    """

    original_axis, normalized_axis = _validate_removal_axis(removal_axis)
    quantile = _validate_finite_real(
        projection_quantile,
        name="projection_quantile",
    )
    if not 0.0 < quantile <= 1.0:
        raise ValueError("projection_quantile must satisfy 0 < value <= 1")
    support_width = _validate_finite_real(
        support_band_width_pixels,
        name="support_band_width_pixels",
    )
    if support_width < 0.0:
        raise ValueError("support_band_width_pixels must be non-negative")
    signed_offset = _validate_finite_real(
        signed_offset_pixels,
        name="signed_offset_pixels",
    )

    baseline = estimate_visible_mask_cutline(
        mask,
        calyx_dilation_radius=calyx_dilation_radius,
        signed_offset=0.0,
        component_connectivity=component_connectivity,
    )
    parameters = FixedAxisCutlineParameters(
        removal_axis=original_axis,
        projection_quantile=quantile,
        support_band_width_pixels=support_width,
        signed_offset_pixels=signed_offset,
        calyx_dilation_radius=baseline.parameters.calyx_dilation_radius,
        component_connectivity=baseline.parameters.component_connectivity,
    )
    selected_flesh_mask = _component_mask(
        baseline.flesh_mask,
        label=baseline.selected_flesh_component,
        connectivity=baseline.parameters.component_connectivity,
    )
    selected_calyx_mask = _component_mask(
        baseline.calyx_mask,
        label=baseline.selected_calyx_component,
        connectivity=baseline.parameters.component_connectivity,
    )
    empty_support = _readonly_boolean_copy(np.zeros(mask.shape, dtype=bool))
    common = _common_result_values(
        baseline,
        parameters=parameters,
        normalized_axis=normalized_axis,
        selected_flesh_mask=selected_flesh_mask,
        selected_calyx_mask=selected_calyx_mask,
    )

    if baseline.contact_pixel_count == 0:
        return FixedAxisCutlineResult(
            **common,
            status="failed",
            failure_code=baseline.failure_code,
            failure_reason=baseline.failure_reason,
            support_pixel_count=0,
            support_lateral_extent_pixels=None,
            unshifted_cut_coordinate=None,
            final_cut_coordinate=None,
            diagnostic_reference_point=None,
            candidate_cutline=None,
            final_cutline=None,
            minimum_contact_projection=None,
            maximum_contact_projection=None,
            selected_contact_projection=None,
            axis_disagreement_angle_degrees=None,
            maximum_selected_flesh_projection=None,
            cut_margin_to_flesh_extent=None,
            candidate_final_lines_coincide=None,
            support_mask=empty_support,
        )

    contact_coordinates = _xy_coordinates(baseline.contact_band)
    axis_array = np.asarray(normalized_axis, dtype=np.float64)
    contact_projections = contact_coordinates @ axis_array
    sorted_projections = np.sort(contact_projections)
    selected_index = min(
        len(sorted_projections) - 1,
        ceil(quantile * len(sorted_projections)) - 1,
    )
    unshifted_coordinate = float(sorted_projections[selected_index])
    final_coordinate = unshifted_coordinate + signed_offset

    support_selection = contact_projections >= (unshifted_coordinate - support_width)
    support_coordinates = contact_coordinates[support_selection]
    support_indices = support_coordinates.astype(np.intp)
    support_mask_array = np.zeros(mask.shape, dtype=bool)
    support_mask_array[support_indices[:, 1], support_indices[:, 0]] = True
    support_mask = _readonly_boolean_copy(support_mask_array)
    support_centroid_array = support_coordinates.mean(axis=0)
    centroid_projection = float(support_centroid_array @ axis_array)
    reference_array = (
        support_centroid_array + (final_coordinate - centroid_projection) * axis_array
    )
    reference_point = Point(
        x=float(reference_array[0]),
        y=float(reference_array[1]),
    )
    perpendicular = (-normalized_axis[1], normalized_axis[0])
    candidate_cutline = clip_infinite_line_to_image(
        _point_on_line(unshifted_coordinate, normalized_axis),
        perpendicular,
        width=baseline.image_width,
        height=baseline.image_height,
    )

    lateral_projections = support_coordinates @ np.asarray(
        perpendicular,
        dtype=np.float64,
    )
    support_lateral_extent = float(
        lateral_projections.max() - lateral_projections.min()
    )
    disagreement = _axis_disagreement_angle(
        normalized_axis,
        baseline.fruit_to_attachment_direction,
    )
    flesh_coordinates = _xy_coordinates(selected_flesh_mask)
    maximum_flesh_projection = float((flesh_coordinates @ axis_array).max())
    cut_margin = maximum_flesh_projection - final_coordinate
    geometry = {
        "support_pixel_count": len(support_coordinates),
        "support_lateral_extent_pixels": support_lateral_extent,
        "unshifted_cut_coordinate": unshifted_coordinate,
        "final_cut_coordinate": final_coordinate,
        "diagnostic_reference_point": reference_point,
        "minimum_contact_projection": float(sorted_projections[0]),
        "maximum_contact_projection": float(sorted_projections[-1]),
        "selected_contact_projection": unshifted_coordinate,
        "axis_disagreement_angle_degrees": disagreement,
        "maximum_selected_flesh_projection": maximum_flesh_projection,
        "cut_margin_to_flesh_extent": cut_margin,
        "support_mask": support_mask,
    }
    if candidate_cutline is None:
        return FixedAxisCutlineResult(
            **common,
            **geometry,
            status="failed",
            failure_code="candidate_line_outside_image",
            failure_reason="Candidate fixed-axis cutline does not cross the image bounds.",
            candidate_cutline=None,
            final_cutline=None,
            candidate_final_lines_coincide=None,
        )

    final_cutline = clip_infinite_line_to_image(
        _point_on_line(final_coordinate, normalized_axis),
        perpendicular,
        width=baseline.image_width,
        height=baseline.image_height,
    )
    if final_cutline is None:
        return FixedAxisCutlineResult(
            **common,
            **geometry,
            status="failed",
            failure_code="offset_line_outside_image",
            failure_reason=(
                "Final offset fixed-axis cutline does not cross the image bounds; "
                f"signed offset was {signed_offset}."
            ),
            candidate_cutline=candidate_cutline,
            final_cutline=None,
            candidate_final_lines_coincide=None,
        )

    return FixedAxisCutlineResult(
        **common,
        **geometry,
        status="ok",
        failure_code=None,
        failure_reason=None,
        candidate_cutline=candidate_cutline,
        final_cutline=final_cutline,
        candidate_final_lines_coincide=_segments_coincide(
            candidate_cutline,
            final_cutline,
        ),
    )


def _validate_removal_axis(
    removal_axis: Sequence[Real] | np.ndarray,
) -> tuple[tuple[float, float], tuple[float, float]]:
    if isinstance(removal_axis, (str, bytes)):
        raise ValueError("removal_axis must be a two-dimensional finite vector")
    try:
        raw_axis = np.asarray(removal_axis)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "removal_axis must be a two-dimensional finite vector"
        ) from error
    if raw_axis.shape != (2,):
        raise ValueError("removal_axis must be a two-dimensional finite vector")
    raw_values = raw_axis.tolist()
    if any(
        isinstance(value, (bool, np.bool_)) or not isinstance(value, Real)
        for value in raw_values
    ):
        raise ValueError("removal_axis must contain two finite real values")
    try:
        axis = raw_axis.astype(np.float64)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(
            "removal_axis must be a two-dimensional finite vector"
        ) from error
    if not np.all(np.isfinite(axis)):
        raise ValueError("removal_axis must contain only finite values")
    maximum_magnitude = max(abs(float(axis[0])), abs(float(axis[1])))
    if maximum_magnitude == 0.0:
        raise ValueError("removal_axis must be non-zero")
    scaled_x = float(axis[0]) / maximum_magnitude
    scaled_y = float(axis[1]) / maximum_magnitude
    scaled_length = hypot(scaled_x, scaled_y)
    original = (float(axis[0]), float(axis[1]))
    normalized = (scaled_x / scaled_length, scaled_y / scaled_length)
    return original, normalized


def _validate_finite_real(value: float, *, name: str) -> float:
    if (
        isinstance(value, bool)
        or not isinstance(value, Real)
        or not isfinite(float(value))
    ):
        raise ValueError(f"{name} must be a finite real number")
    return float(value)


def _component_mask(
    binary_mask: np.ndarray,
    *,
    label: int | None,
    connectivity: Literal[4, 8],
) -> np.ndarray:
    if label is None:
        return _readonly_boolean_copy(np.zeros(binary_mask.shape, dtype=bool))
    labels, components = _label_components(binary_mask, connectivity=connectivity)
    if label < 1 or label > len(components):
        raise RuntimeError("baseline v1 returned an invalid selected component label")
    return _readonly_boolean_copy(labels == label)


def _common_result_values(
    baseline: MaskCutlineResult,
    *,
    parameters: FixedAxisCutlineParameters,
    normalized_axis: tuple[float, float],
    selected_flesh_mask: np.ndarray,
    selected_calyx_mask: np.ndarray,
) -> dict[str, object]:
    return {
        "parameters": parameters,
        "image_width": baseline.image_width,
        "image_height": baseline.image_height,
        "normalized_removal_axis": normalized_axis,
        "flesh_pixel_count": baseline.flesh_pixel_count,
        "calyx_pixel_count": baseline.calyx_pixel_count,
        "flesh_component_count": baseline.flesh_component_count,
        "calyx_component_count": baseline.calyx_component_count,
        "selected_flesh_component": baseline.selected_flesh_component,
        "selected_calyx_component": baseline.selected_calyx_component,
        "selected_flesh_pixel_count": baseline.selected_flesh_pixel_count,
        "selected_calyx_pixel_count": baseline.selected_calyx_pixel_count,
        "contact_pixel_count": baseline.contact_pixel_count,
        "flesh_centroid": baseline.flesh_centroid,
        "baseline_v1_contact_anchor": baseline.attachment_anchor,
        "baseline_v1_fruit_to_attachment_direction": (
            baseline.fruit_to_attachment_direction
        ),
        "baseline_v1_cutline": baseline.candidate_cutline,
        "signed_offset_pixels": parameters.signed_offset_pixels,
        "flesh_mask": baseline.flesh_mask,
        "calyx_mask": baseline.calyx_mask,
        "selected_flesh_mask": selected_flesh_mask,
        "selected_calyx_mask": selected_calyx_mask,
        "contact_mask": baseline.contact_band,
    }


def _xy_coordinates(binary_mask: np.ndarray) -> np.ndarray:
    coordinates_yx = np.argwhere(binary_mask)
    return coordinates_yx[:, ::-1].astype(np.float64, copy=False)


def _point_on_line(
    coordinate: float,
    axis: tuple[float, float],
) -> Point:
    return Point(x=coordinate * axis[0], y=coordinate * axis[1])


def _axis_disagreement_angle(
    configured_axis: tuple[float, float],
    baseline_direction: tuple[float, float] | None,
) -> float | None:
    if baseline_direction is None:
        return None
    cosine = (
        configured_axis[0] * baseline_direction[0]
        + configured_axis[1] * baseline_direction[1]
    )
    return degrees(acos(min(1.0, max(-1.0, cosine))))


def _segments_coincide(first: LineSegment, second: LineSegment) -> bool:
    return first == second or (first.start == second.end and first.end == second.start)


def _readonly_boolean_copy(mask: np.ndarray) -> np.ndarray:
    copied = np.array(mask, dtype=bool, copy=True)
    copied.setflags(write=False)
    return copied
