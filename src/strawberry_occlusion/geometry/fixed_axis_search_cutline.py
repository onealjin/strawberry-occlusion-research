"""Experimental v2b outermost-feasible fixed-axis cut-coordinate search."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field, replace
from math import floor
from numbers import Real
from typing import Any, Literal

import numpy as np

from strawberry_occlusion.geometry.fixed_axis_cutline import (
    FixedAxisCutlineResult,
    _point_on_line,
    _project_point_to_fixed_line,
    _readonly_boolean_copy,
    _validate_finite_real,
    _validate_removal_axis,
    estimate_fixed_axis_cutline,
)
from strawberry_occlusion.geometry.mask_cutline import (
    _validate_mask,
    clip_infinite_line_to_image,
)
from strawberry_occlusion.geometry.primitives import LineSegment, Point


@dataclass(frozen=True)
class FixedAxisSearchCutlineParameters:
    """Research-only parameters controlling experimental v2b search."""

    removal_axis: tuple[float, float]
    projection_quantile: float = 0.95
    support_band_width_pixels: float = 5.0
    signed_offset_pixels: float = 0.0
    calyx_dilation_radius: int = 1
    component_connectivity: Literal[4, 8] = 8
    candidate_step_pixels: float = 1.0
    inward_search_margin_pixels: float = 10.0
    outward_search_margin_pixels: float = 0.0
    blade_band_half_width_pixels: float = 1.0
    lateral_window_half_width_pixels: float = 64.0
    minimum_attachment_evidence_fraction: float = 0.50
    minimum_flesh_band_pixels: int = 1
    minimum_calyx_band_pixels: int = 1


@dataclass(frozen=True)
class FixedAxisCandidateTable:
    """Compact read-only v2b candidate curve in configured-axis order.

    Candidate coordinates are finite, strictly increasing, and unique. Feasible
    block IDs are zero-based in the same order and use ``-1`` for infeasible
    candidates.
    """

    search_interval: tuple[float, float] | None
    image_projection_interval: tuple[float, float]
    lateral_reference_coordinate: float | None
    flesh_centroid_lateral_reference_coordinate: float | None
    coordinate: np.ndarray = field(repr=False, compare=False)
    local_contact_count: np.ndarray = field(repr=False, compare=False)
    local_flesh_count: np.ndarray = field(repr=False, compare=False)
    local_calyx_count: np.ndarray = field(repr=False, compare=False)
    local_v2a_support_count: np.ndarray = field(repr=False, compare=False)
    selected_flesh_loss_ratio: np.ndarray = field(repr=False, compare=False)
    selected_calyx_retention_ratio: np.ndarray = field(
        repr=False,
        compare=False,
    )
    retained_contact_ratio: np.ndarray = field(repr=False, compare=False)
    feasible: np.ndarray = field(repr=False, compare=False)
    feasible_block_id: np.ndarray = field(repr=False, compare=False)

    def __len__(self) -> int:
        return len(self.coordinate)


@dataclass(frozen=True)
class FixedAxisSearchCutlineResult:
    """Structured v2b geometry and selected-pair research diagnostics."""

    status: Literal["ok", "failed"]
    failure_code: str | None
    failure_reason: str | None
    parameters: FixedAxisSearchCutlineParameters
    image_width: int
    image_height: int
    normalized_removal_axis: tuple[float, float]
    blade_tangent: tuple[float, float]
    flesh_component_count: int
    calyx_component_count: int
    selected_flesh_component: int | None
    selected_calyx_component: int | None
    selected_flesh_pixel_count: int
    selected_calyx_pixel_count: int
    contact_pixel_count: int
    v1_attachment_anchor: Point | None
    v1_anchor_projection: float | None
    v2a_cut_coordinate: float | None
    support_centroid_reference_point: Point | None
    flesh_centroid_projected_reference_point: Point | None
    attachment_anchor_projected_reference_point: Point | None
    lateral_reference_coordinate: float | None
    flesh_centroid_lateral_reference_coordinate: float | None
    search_interval: tuple[float, float] | None
    candidate_count: int
    feasible_candidate_count: int
    feasible_block_count: int
    feasible_hull_start: float | None
    feasible_hull_end: float | None
    selected_cut_coordinate: float | None
    candidate_cutline: LineSegment | None
    final_cutline: LineSegment | None
    diagnostic_reference_point: Point | None
    selected_candidate_index: int | None
    selected_block_id: int | None
    selected_block_start: float | None
    selected_block_end: float | None
    selected_block_candidate_count: int
    selected_block_is_singleton: bool | None
    maximum_local_contact_evidence: int
    selected_local_contact_count: int
    selected_local_flesh_count: int
    selected_local_calyx_count: int
    selected_local_v2a_support_count: int
    selected_flesh_loss_count: int | None
    selected_flesh_loss_ratio: float | None
    selected_calyx_retention_count: int | None
    selected_calyx_retention_ratio: float | None
    retained_contact_count: int | None
    retained_contact_ratio: float | None
    coordinate_shift_from_v1: float | None
    coordinate_shift_from_v2a: float | None
    cut_margin_to_flesh_extent: float | None
    candidate_final_lines_coincide: bool | None
    candidate_table: FixedAxisCandidateTable = field(repr=False, compare=False)
    selected_flesh_mask: np.ndarray = field(repr=False, compare=False)
    selected_calyx_mask: np.ndarray = field(repr=False, compare=False)
    contact_mask: np.ndarray = field(repr=False, compare=False)
    v2a_support_mask: np.ndarray = field(repr=False, compare=False)
    selected_local_window_mask: np.ndarray = field(repr=False, compare=False)
    selected_local_contact_mask: np.ndarray = field(repr=False, compare=False)
    selected_local_flesh_mask: np.ndarray = field(repr=False, compare=False)
    selected_local_calyx_mask: np.ndarray = field(repr=False, compare=False)
    selected_local_v2a_support_mask: np.ndarray = field(
        repr=False,
        compare=False,
    )

    @property
    def succeeded(self) -> bool:
        """Return whether v2b selected and clipped a feasible fixed line."""

        return self.status == "ok"


CandidateCurveFunction = Callable[..., FixedAxisCandidateTable]


def estimate_fixed_axis_search_cutline(
    mask: np.ndarray,
    *,
    removal_axis: Sequence[Real] | np.ndarray,
    projection_quantile: float = 0.95,
    support_band_width_pixels: float = 5.0,
    signed_offset_pixels: float = 0.0,
    calyx_dilation_radius: int = 1,
    component_connectivity: Literal[4, 8] = 8,
    candidate_step_pixels: float = 1.0,
    inward_search_margin_pixels: float = 10.0,
    outward_search_margin_pixels: float = 0.0,
    blade_band_half_width_pixels: float = 1.0,
    lateral_window_half_width_pixels: float = 64.0,
    minimum_attachment_evidence_fraction: float = 0.50,
    minimum_flesh_band_pixels: int = 1,
    minimum_calyx_band_pixels: int = 1,
    v2a_result: FixedAxisCutlineResult | None = None,
    candidate_curve_function: CandidateCurveFunction | None = None,
) -> FixedAxisSearchCutlineResult:
    """Select the most outward feasible fixed-axis research cut coordinate.

    The v2a signed offset is already incorporated in its final coordinate and
    is not applied again. Candidate coordinates are strictly ordered and
    unique, so the greatest feasible coordinate normally determines selection
    without further tie-breaks. Candidate evidence uses one-time selected-pixel
    ``s`` and ``t`` projections. Lateral filtering and sorting occur once per
    mask; vectorized ``searchsorted`` queries then produce all candidate counts.
    No full image mask is created independently for each candidate.

    This is a geometric research proxy, not true attachment severance, cutline
    accuracy, or production-ready machine geometry.
    """

    mask_array = _validate_mask(mask)
    search_parameters = _validate_search_parameters(
        candidate_step_pixels=candidate_step_pixels,
        inward_search_margin_pixels=inward_search_margin_pixels,
        outward_search_margin_pixels=outward_search_margin_pixels,
        blade_band_half_width_pixels=blade_band_half_width_pixels,
        lateral_window_half_width_pixels=lateral_window_half_width_pixels,
        minimum_attachment_evidence_fraction=(minimum_attachment_evidence_fraction),
        minimum_flesh_band_pixels=minimum_flesh_band_pixels,
        minimum_calyx_band_pixels=minimum_calyx_band_pixels,
    )
    if v2a_result is None:
        v2a = estimate_fixed_axis_cutline(
            mask_array,
            removal_axis=removal_axis,
            projection_quantile=projection_quantile,
            support_band_width_pixels=support_band_width_pixels,
            signed_offset_pixels=signed_offset_pixels,
            calyx_dilation_radius=calyx_dilation_radius,
            component_connectivity=component_connectivity,
        )
    else:
        v2a = _validate_supplied_v2a_result(
            v2a_result,
            mask_shape=mask_array.shape,
            removal_axis=removal_axis,
            projection_quantile=projection_quantile,
            support_band_width_pixels=support_band_width_pixels,
            signed_offset_pixels=signed_offset_pixels,
            calyx_dilation_radius=calyx_dilation_radius,
            component_connectivity=component_connectivity,
        )
    parameters = FixedAxisSearchCutlineParameters(
        removal_axis=v2a.parameters.removal_axis,
        projection_quantile=v2a.parameters.projection_quantile,
        support_band_width_pixels=v2a.parameters.support_band_width_pixels,
        signed_offset_pixels=v2a.parameters.signed_offset_pixels,
        calyx_dilation_radius=v2a.parameters.calyx_dilation_radius,
        component_connectivity=v2a.parameters.component_connectivity,
        **search_parameters,
    )
    tangent = (-v2a.normalized_removal_axis[1], v2a.normalized_removal_axis[0])
    empty_table = _empty_candidate_table(
        image_projection_interval=_image_projection_interval(
            width=v2a.image_width,
            height=v2a.image_height,
            axis=v2a.normalized_removal_axis,
        )
    )
    prerequisite_missing = (
        v2a.final_cut_coordinate is None
        or v2a.baseline_v1_contact_anchor is None
        or not np.any(v2a.selected_flesh_mask)
        or not np.any(v2a.selected_calyx_mask)
        or not np.any(v2a.contact_mask)
    )
    if prerequisite_missing:
        return _result(
            v2a,
            parameters=parameters,
            tangent=tangent,
            candidate_table=empty_table,
            status="failed",
            failure_code="v2a_prerequisite_failed",
            failure_reason=(
                "V2b requires a v2a cut coordinate, selected component pair, "
                "and selected contact evidence."
            ),
        )

    curve_function = candidate_curve_function or calculate_fixed_axis_candidate_curve
    candidate_table = curve_function(
        v2a,
        candidate_step_pixels=parameters.candidate_step_pixels,
        inward_search_margin_pixels=parameters.inward_search_margin_pixels,
        outward_search_margin_pixels=parameters.outward_search_margin_pixels,
        blade_band_half_width_pixels=parameters.blade_band_half_width_pixels,
        lateral_window_half_width_pixels=(parameters.lateral_window_half_width_pixels),
        minimum_attachment_evidence_fraction=(
            parameters.minimum_attachment_evidence_fraction
        ),
        minimum_flesh_band_pixels=parameters.minimum_flesh_band_pixels,
        minimum_calyx_band_pixels=parameters.minimum_calyx_band_pixels,
    )
    candidate_table = _validate_candidate_table(candidate_table)
    maximum_contact = (
        int(candidate_table.local_contact_count.max()) if len(candidate_table) else 0
    )
    if maximum_contact == 0:
        candidate_table = replace(
            candidate_table,
            feasible=_readonly_array(np.zeros(len(candidate_table)), dtype=bool),
        )
    candidate_table = _with_feasible_block_ids(
        candidate_table,
        candidate_step_pixels=parameters.candidate_step_pixels,
    )
    feasible_indices = np.flatnonzero(candidate_table.feasible)
    if not len(feasible_indices):
        return _result(
            v2a,
            parameters=parameters,
            tangent=tangent,
            candidate_table=candidate_table,
            status="failed",
            failure_code="no_feasible_fixed_axis_cut",
            failure_reason=(
                "No fixed-axis candidate satisfied the configured local "
                "attachment-evidence, Flesh-band, and Calyx-band thresholds."
            ),
            maximum_local_contact_evidence=maximum_contact,
        )

    selected_index = int(feasible_indices[-1])
    selected_coordinate = float(candidate_table.coordinate[selected_index])
    selected_block = _selected_block_diagnostics(candidate_table, selected_index)
    candidate_cutline = clip_infinite_line_to_image(
        _point_on_line(selected_coordinate, v2a.normalized_removal_axis),
        tangent,
        width=v2a.image_width,
        height=v2a.image_height,
    )
    if candidate_cutline is None:
        return _result(
            v2a,
            parameters=parameters,
            tangent=tangent,
            candidate_table=candidate_table,
            status="failed",
            failure_code="candidate_line_outside_image",
            failure_reason="Selected v2b fixed line does not cross the image bounds.",
            selected_candidate_index=selected_index,
            selected_cut_coordinate=selected_coordinate,
            maximum_local_contact_evidence=maximum_contact,
            **selected_block,
        )

    diagnostic_reference = _project_point_to_fixed_line(
        v2a.baseline_v1_contact_anchor,
        coordinate=selected_coordinate,
        axis=v2a.normalized_removal_axis,
    )
    selected_masks = _selected_local_masks(
        shape=mask_array.shape,
        coordinate=selected_coordinate,
        axis=v2a.normalized_removal_axis,
        tangent=tangent,
        lateral_reference=float(candidate_table.lateral_reference_coordinate),
        blade_band_half_width=parameters.blade_band_half_width_pixels,
        lateral_window_half_width=parameters.lateral_window_half_width_pixels,
        selected_flesh_mask=v2a.selected_flesh_mask,
        selected_calyx_mask=v2a.selected_calyx_mask,
        contact_mask=v2a.contact_mask,
        support_mask=v2a.support_mask,
    )
    maximum_flesh_projection = v2a.maximum_selected_flesh_projection
    return _result(
        v2a,
        parameters=parameters,
        tangent=tangent,
        candidate_table=candidate_table,
        status="ok",
        failure_code=None,
        failure_reason=None,
        selected_candidate_index=selected_index,
        selected_cut_coordinate=selected_coordinate,
        candidate_cutline=candidate_cutline,
        final_cutline=candidate_cutline,
        diagnostic_reference_point=diagnostic_reference,
        maximum_local_contact_evidence=maximum_contact,
        selected_local_contact_count=int(
            candidate_table.local_contact_count[selected_index]
        ),
        selected_local_flesh_count=int(
            candidate_table.local_flesh_count[selected_index]
        ),
        selected_local_calyx_count=int(
            candidate_table.local_calyx_count[selected_index]
        ),
        selected_local_v2a_support_count=int(
            candidate_table.local_v2a_support_count[selected_index]
        ),
        selected_flesh_loss_count=_flesh_loss_count(
            v2a.selected_flesh_mask,
            coordinate=selected_coordinate,
            axis=v2a.normalized_removal_axis,
        ),
        selected_flesh_loss_ratio=float(
            candidate_table.selected_flesh_loss_ratio[selected_index]
        ),
        selected_calyx_retention_count=_retained_count(
            v2a.selected_calyx_mask,
            coordinate=selected_coordinate,
            axis=v2a.normalized_removal_axis,
        ),
        selected_calyx_retention_ratio=float(
            candidate_table.selected_calyx_retention_ratio[selected_index]
        ),
        retained_contact_count=_retained_count(
            v2a.contact_mask,
            coordinate=selected_coordinate,
            axis=v2a.normalized_removal_axis,
        ),
        retained_contact_ratio=float(
            candidate_table.retained_contact_ratio[selected_index]
        ),
        coordinate_shift_from_v1=(
            selected_coordinate
            - _point_projection(
                v2a.baseline_v1_contact_anchor,
                v2a.normalized_removal_axis,
            )
        ),
        coordinate_shift_from_v2a=(
            selected_coordinate - float(v2a.final_cut_coordinate)
        ),
        cut_margin_to_flesh_extent=(
            maximum_flesh_projection - selected_coordinate
            if maximum_flesh_projection is not None
            else None
        ),
        candidate_final_lines_coincide=True,
        **selected_block,
        **selected_masks,
    )


def calculate_fixed_axis_candidate_curve(
    v2a_result: FixedAxisCutlineResult,
    *,
    candidate_step_pixels: float = 1.0,
    inward_search_margin_pixels: float = 10.0,
    outward_search_margin_pixels: float = 0.0,
    blade_band_half_width_pixels: float = 1.0,
    lateral_window_half_width_pixels: float = 64.0,
    minimum_attachment_evidence_fraction: float = 0.50,
    minimum_flesh_band_pixels: int = 1,
    minimum_calyx_band_pixels: int = 1,
) -> FixedAxisCandidateTable:
    """Return vectorized local evidence and selected-pair proxies by coordinate."""

    parameters = _validate_search_parameters(
        candidate_step_pixels=candidate_step_pixels,
        inward_search_margin_pixels=inward_search_margin_pixels,
        outward_search_margin_pixels=outward_search_margin_pixels,
        blade_band_half_width_pixels=blade_band_half_width_pixels,
        lateral_window_half_width_pixels=lateral_window_half_width_pixels,
        minimum_attachment_evidence_fraction=(minimum_attachment_evidence_fraction),
        minimum_flesh_band_pixels=minimum_flesh_band_pixels,
        minimum_calyx_band_pixels=minimum_calyx_band_pixels,
    )
    if not isinstance(v2a_result, FixedAxisCutlineResult):
        raise TypeError("v2a_result must be a FixedAxisCutlineResult")
    anchor = v2a_result.baseline_v1_contact_anchor
    v2a_coordinate = v2a_result.final_cut_coordinate
    if anchor is None or v2a_coordinate is None:
        raise ValueError("v2a_result must contain an anchor and final cut coordinate")
    axis = v2a_result.normalized_removal_axis
    tangent = (-axis[1], axis[0])
    anchor_projection = _point_projection(anchor, axis)
    lateral_reference = _point_projection(anchor, tangent)
    flesh_lateral_reference = _point_projection(v2a_result.flesh_centroid, tangent)
    image_interval = _image_projection_interval(
        width=v2a_result.image_width,
        height=v2a_result.image_height,
        axis=axis,
    )
    search_start = max(
        min(anchor_projection, v2a_coordinate)
        - parameters["inward_search_margin_pixels"],
        image_interval[0],
    )
    search_end = min(
        v2a_coordinate + parameters["outward_search_margin_pixels"],
        image_interval[1],
    )
    if search_start > search_end:
        return _empty_candidate_table(
            image_projection_interval=image_interval,
            lateral_reference=lateral_reference,
            flesh_lateral_reference=flesh_lateral_reference,
        )
    candidates = _candidate_coordinates(
        search_start,
        search_end,
        step=parameters["candidate_step_pixels"],
    )
    candidates = np.asarray(
        [
            coordinate
            for coordinate in candidates
            if clip_infinite_line_to_image(
                _point_on_line(float(coordinate), axis),
                tangent,
                width=v2a_result.image_width,
                height=v2a_result.image_height,
            )
            is not None
        ],
        dtype=np.float64,
    )
    if not len(candidates):
        return _empty_candidate_table(
            image_projection_interval=image_interval,
            lateral_reference=lateral_reference,
            flesh_lateral_reference=flesh_lateral_reference,
            search_interval=(search_start, search_end),
        )

    flesh_s, flesh_t = _mask_projections(v2a_result.selected_flesh_mask, axis, tangent)
    calyx_s, calyx_t = _mask_projections(v2a_result.selected_calyx_mask, axis, tangent)
    contact_s, contact_t = _mask_projections(v2a_result.contact_mask, axis, tangent)
    support_s, support_t = _mask_projections(v2a_result.support_mask, axis, tangent)
    lateral_half_width = parameters["lateral_window_half_width_pixels"]
    blade_half_width = parameters["blade_band_half_width_pixels"]
    local_flesh = _local_band_counts(
        flesh_s[np.abs(flesh_t - lateral_reference) <= lateral_half_width],
        candidates,
        half_width=blade_half_width,
    )
    local_calyx = _local_band_counts(
        calyx_s[np.abs(calyx_t - lateral_reference) <= lateral_half_width],
        candidates,
        half_width=blade_half_width,
    )
    local_contact = _local_band_counts(
        contact_s[np.abs(contact_t - lateral_reference) <= lateral_half_width],
        candidates,
        half_width=blade_half_width,
    )
    local_support = _local_band_counts(
        support_s[np.abs(support_t - lateral_reference) <= lateral_half_width],
        candidates,
        half_width=blade_half_width,
    )
    sorted_flesh = np.sort(flesh_s)
    sorted_calyx = np.sort(calyx_s)
    sorted_contact = np.sort(contact_s)
    flesh_loss = len(sorted_flesh) - np.searchsorted(
        sorted_flesh,
        candidates,
        side="right",
    )
    calyx_retained = np.searchsorted(sorted_calyx, candidates, side="right")
    contact_retained = np.searchsorted(sorted_contact, candidates, side="right")
    flesh_loss_ratio = _safe_ratio(flesh_loss, len(sorted_flesh))
    calyx_retention_ratio = _safe_ratio(calyx_retained, len(sorted_calyx))
    retained_contact_ratio = _safe_ratio(contact_retained, len(sorted_contact))
    maximum_contact = int(local_contact.max()) if len(local_contact) else 0
    evidence_threshold = (
        parameters["minimum_attachment_evidence_fraction"] * maximum_contact
    )
    feasible = (
        (maximum_contact > 0)
        & (local_contact >= evidence_threshold)
        & (local_flesh >= parameters["minimum_flesh_band_pixels"])
        & (local_calyx >= parameters["minimum_calyx_band_pixels"])
    )
    table = FixedAxisCandidateTable(
        search_interval=(search_start, search_end),
        image_projection_interval=image_interval,
        lateral_reference_coordinate=lateral_reference,
        flesh_centroid_lateral_reference_coordinate=flesh_lateral_reference,
        coordinate=_readonly_array(candidates, dtype=np.float64),
        local_contact_count=_readonly_array(local_contact, dtype=np.int64),
        local_flesh_count=_readonly_array(local_flesh, dtype=np.int64),
        local_calyx_count=_readonly_array(local_calyx, dtype=np.int64),
        local_v2a_support_count=_readonly_array(local_support, dtype=np.int64),
        selected_flesh_loss_ratio=_readonly_array(
            flesh_loss_ratio,
            dtype=np.float64,
        ),
        selected_calyx_retention_ratio=_readonly_array(
            calyx_retention_ratio,
            dtype=np.float64,
        ),
        retained_contact_ratio=_readonly_array(
            retained_contact_ratio,
            dtype=np.float64,
        ),
        feasible=_readonly_array(feasible, dtype=bool),
        feasible_block_id=_readonly_array(
            np.full(len(candidates), -1),
            dtype=np.int64,
        ),
    )
    return _with_feasible_block_ids(
        table,
        candidate_step_pixels=parameters["candidate_step_pixels"],
    )


def _validate_search_parameters(
    *,
    candidate_step_pixels: float,
    inward_search_margin_pixels: float,
    outward_search_margin_pixels: float,
    blade_band_half_width_pixels: float,
    lateral_window_half_width_pixels: float,
    minimum_attachment_evidence_fraction: float,
    minimum_flesh_band_pixels: int,
    minimum_calyx_band_pixels: int,
) -> dict[str, float | int]:
    values = {
        "candidate_step_pixels": _validate_finite_real(
            candidate_step_pixels,
            name="candidate_step_pixels",
        ),
        "inward_search_margin_pixels": _validate_finite_real(
            inward_search_margin_pixels,
            name="inward_search_margin_pixels",
        ),
        "outward_search_margin_pixels": _validate_finite_real(
            outward_search_margin_pixels,
            name="outward_search_margin_pixels",
        ),
        "blade_band_half_width_pixels": _validate_finite_real(
            blade_band_half_width_pixels,
            name="blade_band_half_width_pixels",
        ),
        "lateral_window_half_width_pixels": _validate_finite_real(
            lateral_window_half_width_pixels,
            name="lateral_window_half_width_pixels",
        ),
        "minimum_attachment_evidence_fraction": _validate_finite_real(
            minimum_attachment_evidence_fraction,
            name="minimum_attachment_evidence_fraction",
        ),
    }
    if values["candidate_step_pixels"] <= 0.0:
        raise ValueError("candidate_step_pixels must be greater than zero")
    for name in (
        "inward_search_margin_pixels",
        "outward_search_margin_pixels",
        "blade_band_half_width_pixels",
        "lateral_window_half_width_pixels",
    ):
        if values[name] < 0.0:
            raise ValueError(f"{name} must be non-negative")
    fraction = values["minimum_attachment_evidence_fraction"]
    if not 0.0 < fraction <= 1.0:
        raise ValueError(
            "minimum_attachment_evidence_fraction must satisfy 0 < value <= 1"
        )
    for name, value in (
        ("minimum_flesh_band_pixels", minimum_flesh_band_pixels),
        ("minimum_calyx_band_pixels", minimum_calyx_band_pixels),
    ):
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"{name} must be a non-negative integer")
        values[name] = value
    return values


def _validate_supplied_v2a_result(
    result: FixedAxisCutlineResult,
    *,
    mask_shape: tuple[int, int],
    removal_axis: Sequence[Real] | np.ndarray,
    projection_quantile: float,
    support_band_width_pixels: float,
    signed_offset_pixels: float,
    calyx_dilation_radius: int,
    component_connectivity: int,
) -> FixedAxisCutlineResult:
    if not isinstance(result, FixedAxisCutlineResult):
        raise TypeError("v2a_result must be a FixedAxisCutlineResult")
    if (result.image_height, result.image_width) != mask_shape:
        raise ValueError("v2a_result dimensions must match the supplied mask")
    _, normalized_axis = _validate_removal_axis(removal_axis)
    if not np.allclose(
        normalized_axis,
        result.normalized_removal_axis,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("removal_axis must match the supplied v2a_result")
    supplied_values = {
        "projection_quantile": _validate_finite_real(
            projection_quantile,
            name="projection_quantile",
        ),
        "support_band_width_pixels": _validate_finite_real(
            support_band_width_pixels,
            name="support_band_width_pixels",
        ),
        "signed_offset_pixels": _validate_finite_real(
            signed_offset_pixels,
            name="signed_offset_pixels",
        ),
    }
    expected_values = {
        "projection_quantile": result.parameters.projection_quantile,
        "support_band_width_pixels": result.parameters.support_band_width_pixels,
        "signed_offset_pixels": result.parameters.signed_offset_pixels,
    }
    for name, value in supplied_values.items():
        if value != expected_values[name]:
            raise ValueError(f"{name} must match the supplied v2a_result")
    if (
        isinstance(calyx_dilation_radius, bool)
        or not isinstance(calyx_dilation_radius, int)
        or calyx_dilation_radius != result.parameters.calyx_dilation_radius
    ):
        raise ValueError("calyx_dilation_radius must match the supplied v2a_result")
    if (
        isinstance(component_connectivity, bool)
        or not isinstance(component_connectivity, int)
        or component_connectivity != result.parameters.component_connectivity
    ):
        raise ValueError("component_connectivity must match the supplied v2a_result")
    return result


def _candidate_coordinates(start: float, end: float, *, step: float) -> np.ndarray:
    count = floor((end - start) / step) + 1
    coordinates = start + step * np.arange(count, dtype=np.float64)
    tolerance = 1e-12 * max(1.0, abs(start), abs(end))
    if end - float(coordinates[-1]) > tolerance:
        coordinates = np.append(coordinates, end)
    else:
        coordinates[-1] = end
    return coordinates


def _mask_projections(
    binary_mask: np.ndarray,
    axis: tuple[float, float],
    tangent: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]:
    coordinates_yx = np.argwhere(binary_mask)
    x = coordinates_yx[:, 1].astype(np.float64)
    y = coordinates_yx[:, 0].astype(np.float64)
    return x * axis[0] + y * axis[1], x * tangent[0] + y * tangent[1]


def _local_band_counts(
    projections: np.ndarray,
    candidates: np.ndarray,
    *,
    half_width: float,
) -> np.ndarray:
    sorted_projections = np.sort(projections)
    lower = np.searchsorted(
        sorted_projections,
        candidates - half_width,
        side="left",
    )
    upper = np.searchsorted(
        sorted_projections,
        candidates + half_width,
        side="right",
    )
    return upper - lower


def _validate_candidate_table(
    table: FixedAxisCandidateTable,
) -> FixedAxisCandidateTable:
    if not isinstance(table, FixedAxisCandidateTable):
        raise TypeError(
            "candidate_curve_function must return a FixedAxisCandidateTable"
        )
    if table.lateral_reference_coordinate is None:
        raise ValueError("candidate curve must provide lateral_reference_coordinate")
    _validate_finite_real(
        table.lateral_reference_coordinate,
        name="candidate curve lateral_reference_coordinate",
    )

    coordinates = np.asarray(table.coordinate)
    if coordinates.ndim != 1 or not np.all(np.isfinite(coordinates)):
        raise ValueError("candidate curve coordinates must be a finite 1D array")
    if len(coordinates) > 1 and np.any(np.diff(coordinates) <= 0.0):
        raise ValueError(
            "candidate curve coordinates must be strictly ordered and unique"
        )
    for field_name in (
        "local_contact_count",
        "local_flesh_count",
        "local_calyx_count",
        "local_v2a_support_count",
        "selected_flesh_loss_ratio",
        "selected_calyx_retention_ratio",
        "retained_contact_ratio",
        "feasible",
        "feasible_block_id",
    ):
        values = np.asarray(getattr(table, field_name))
        if values.ndim != 1 or len(values) != len(coordinates):
            raise ValueError(
                f"candidate curve {field_name} must match the coordinate array"
            )
    return table


def _with_feasible_block_ids(
    table: FixedAxisCandidateTable,
    *,
    candidate_step_pixels: float,
) -> FixedAxisCandidateTable:
    block_ids = np.full(len(table), -1, dtype=np.int64)
    block_id = -1
    previous_feasible_index: int | None = None
    scale = max(
        1.0,
        candidate_step_pixels,
        *(abs(float(value)) for value in table.coordinate),
    )
    tolerance = 1e-12 * scale
    for index in range(len(table)):
        if not bool(table.feasible[index]):
            previous_feasible_index = None
            continue
        starts_new_block = previous_feasible_index is None
        if previous_feasible_index is not None:
            coordinate_gap = float(
                table.coordinate[index] - table.coordinate[previous_feasible_index]
            )
            starts_new_block = coordinate_gap > candidate_step_pixels + tolerance
        if starts_new_block:
            block_id += 1
        block_ids[index] = block_id
        previous_feasible_index = index
    return replace(
        table,
        feasible_block_id=_readonly_array(block_ids, dtype=np.int64),
    )


def _selected_block_diagnostics(
    table: FixedAxisCandidateTable,
    selected_index: int,
) -> dict[str, int | float | bool]:
    block_id = int(table.feasible_block_id[selected_index])
    if block_id < 0:
        raise ValueError("selected candidate must belong to a feasible block")
    block_indices = np.flatnonzero(table.feasible_block_id == block_id)
    return {
        "selected_block_id": block_id,
        "selected_block_start": float(table.coordinate[block_indices[0]]),
        "selected_block_end": float(table.coordinate[block_indices[-1]]),
        "selected_block_candidate_count": int(len(block_indices)),
        "selected_block_is_singleton": bool(len(block_indices) == 1),
    }


def _safe_ratio(numerator: np.ndarray, denominator: int) -> np.ndarray:
    if denominator == 0:
        return np.zeros_like(numerator, dtype=np.float64)
    return numerator / denominator


def _selected_local_masks(
    *,
    shape: tuple[int, int],
    coordinate: float,
    axis: tuple[float, float],
    tangent: tuple[float, float],
    lateral_reference: float,
    blade_band_half_width: float,
    lateral_window_half_width: float,
    selected_flesh_mask: np.ndarray,
    selected_calyx_mask: np.ndarray,
    contact_mask: np.ndarray,
    support_mask: np.ndarray,
) -> dict[str, np.ndarray]:
    pixel_y, pixel_x = np.indices(shape, dtype=np.float64)
    axis_projection = pixel_x * axis[0] + pixel_y * axis[1]
    lateral_projection = pixel_x * tangent[0] + pixel_y * tangent[1]
    window = (np.abs(axis_projection - coordinate) <= blade_band_half_width) & (
        np.abs(lateral_projection - lateral_reference) <= lateral_window_half_width
    )
    return {
        "selected_local_window_mask": _readonly_boolean_copy(window),
        "selected_local_contact_mask": _readonly_boolean_copy(window & contact_mask),
        "selected_local_flesh_mask": _readonly_boolean_copy(
            window & selected_flesh_mask
        ),
        "selected_local_calyx_mask": _readonly_boolean_copy(
            window & selected_calyx_mask
        ),
        "selected_local_v2a_support_mask": _readonly_boolean_copy(
            window & support_mask
        ),
    }


def _result(
    v2a: FixedAxisCutlineResult,
    *,
    parameters: FixedAxisSearchCutlineParameters,
    tangent: tuple[float, float],
    candidate_table: FixedAxisCandidateTable,
    status: Literal["ok", "failed"],
    failure_code: str | None,
    failure_reason: str | None,
    selected_candidate_index: int | None = None,
    selected_cut_coordinate: float | None = None,
    candidate_cutline: LineSegment | None = None,
    final_cutline: LineSegment | None = None,
    diagnostic_reference_point: Point | None = None,
    selected_block_id: int | None = None,
    selected_block_start: float | None = None,
    selected_block_end: float | None = None,
    selected_block_candidate_count: int = 0,
    selected_block_is_singleton: bool | None = None,
    maximum_local_contact_evidence: int = 0,
    selected_local_contact_count: int = 0,
    selected_local_flesh_count: int = 0,
    selected_local_calyx_count: int = 0,
    selected_local_v2a_support_count: int = 0,
    selected_flesh_loss_count: int | None = None,
    selected_flesh_loss_ratio: float | None = None,
    selected_calyx_retention_count: int | None = None,
    selected_calyx_retention_ratio: float | None = None,
    retained_contact_count: int | None = None,
    retained_contact_ratio: float | None = None,
    coordinate_shift_from_v1: float | None = None,
    coordinate_shift_from_v2a: float | None = None,
    cut_margin_to_flesh_extent: float | None = None,
    candidate_final_lines_coincide: bool | None = None,
    selected_local_window_mask: np.ndarray | None = None,
    selected_local_contact_mask: np.ndarray | None = None,
    selected_local_flesh_mask: np.ndarray | None = None,
    selected_local_calyx_mask: np.ndarray | None = None,
    selected_local_v2a_support_mask: np.ndarray | None = None,
) -> FixedAxisSearchCutlineResult:
    empty = _readonly_boolean_copy(
        np.zeros((v2a.image_height, v2a.image_width), dtype=bool)
    )
    anchor_projection = (
        _point_projection(
            v2a.baseline_v1_contact_anchor,
            v2a.normalized_removal_axis,
        )
        if v2a.baseline_v1_contact_anchor is not None
        else None
    )
    flesh_lateral = (
        _point_projection(v2a.flesh_centroid, tangent)
        if v2a.flesh_centroid is not None
        else None
    )
    feasible_indices = np.flatnonzero(candidate_table.feasible)
    feasible_block_ids = candidate_table.feasible_block_id[feasible_indices]
    return FixedAxisSearchCutlineResult(
        status=status,
        failure_code=failure_code,
        failure_reason=failure_reason,
        parameters=parameters,
        image_width=v2a.image_width,
        image_height=v2a.image_height,
        normalized_removal_axis=v2a.normalized_removal_axis,
        blade_tangent=tangent,
        flesh_component_count=v2a.flesh_component_count,
        calyx_component_count=v2a.calyx_component_count,
        selected_flesh_component=v2a.selected_flesh_component,
        selected_calyx_component=v2a.selected_calyx_component,
        selected_flesh_pixel_count=v2a.selected_flesh_pixel_count,
        selected_calyx_pixel_count=v2a.selected_calyx_pixel_count,
        contact_pixel_count=v2a.contact_pixel_count,
        v1_attachment_anchor=v2a.baseline_v1_contact_anchor,
        v1_anchor_projection=anchor_projection,
        v2a_cut_coordinate=v2a.final_cut_coordinate,
        support_centroid_reference_point=v2a.diagnostic_reference_point,
        flesh_centroid_projected_reference_point=(
            v2a.flesh_centroid_projected_reference_point
        ),
        attachment_anchor_projected_reference_point=(
            v2a.attachment_anchor_projected_reference_point
        ),
        lateral_reference_coordinate=(candidate_table.lateral_reference_coordinate),
        flesh_centroid_lateral_reference_coordinate=(
            candidate_table.flesh_centroid_lateral_reference_coordinate
            if candidate_table.flesh_centroid_lateral_reference_coordinate is not None
            else flesh_lateral
        ),
        search_interval=candidate_table.search_interval,
        candidate_count=len(candidate_table),
        feasible_candidate_count=int(len(feasible_indices)),
        feasible_block_count=int(len(np.unique(feasible_block_ids))),
        feasible_hull_start=(
            float(candidate_table.coordinate[feasible_indices[0]])
            if len(feasible_indices)
            else None
        ),
        feasible_hull_end=(
            float(candidate_table.coordinate[feasible_indices[-1]])
            if len(feasible_indices)
            else None
        ),
        selected_cut_coordinate=selected_cut_coordinate,
        candidate_cutline=candidate_cutline,
        final_cutline=final_cutline,
        diagnostic_reference_point=diagnostic_reference_point,
        selected_candidate_index=selected_candidate_index,
        selected_block_id=selected_block_id,
        selected_block_start=selected_block_start,
        selected_block_end=selected_block_end,
        selected_block_candidate_count=selected_block_candidate_count,
        selected_block_is_singleton=selected_block_is_singleton,
        maximum_local_contact_evidence=maximum_local_contact_evidence,
        selected_local_contact_count=selected_local_contact_count,
        selected_local_flesh_count=selected_local_flesh_count,
        selected_local_calyx_count=selected_local_calyx_count,
        selected_local_v2a_support_count=selected_local_v2a_support_count,
        selected_flesh_loss_count=selected_flesh_loss_count,
        selected_flesh_loss_ratio=selected_flesh_loss_ratio,
        selected_calyx_retention_count=selected_calyx_retention_count,
        selected_calyx_retention_ratio=selected_calyx_retention_ratio,
        retained_contact_count=retained_contact_count,
        retained_contact_ratio=retained_contact_ratio,
        coordinate_shift_from_v1=coordinate_shift_from_v1,
        coordinate_shift_from_v2a=coordinate_shift_from_v2a,
        cut_margin_to_flesh_extent=cut_margin_to_flesh_extent,
        candidate_final_lines_coincide=candidate_final_lines_coincide,
        candidate_table=candidate_table,
        selected_flesh_mask=v2a.selected_flesh_mask,
        selected_calyx_mask=v2a.selected_calyx_mask,
        contact_mask=v2a.contact_mask,
        v2a_support_mask=v2a.support_mask,
        selected_local_window_mask=_mask_or_empty(
            selected_local_window_mask,
            empty,
        ),
        selected_local_contact_mask=_mask_or_empty(
            selected_local_contact_mask,
            empty,
        ),
        selected_local_flesh_mask=_mask_or_empty(
            selected_local_flesh_mask,
            empty,
        ),
        selected_local_calyx_mask=_mask_or_empty(
            selected_local_calyx_mask,
            empty,
        ),
        selected_local_v2a_support_mask=(
            _mask_or_empty(selected_local_v2a_support_mask, empty)
        ),
    )


def _empty_candidate_table(
    *,
    image_projection_interval: tuple[float, float],
    lateral_reference: float | None = None,
    flesh_lateral_reference: float | None = None,
    search_interval: tuple[float, float] | None = None,
) -> FixedAxisCandidateTable:
    return FixedAxisCandidateTable(
        search_interval=search_interval,
        image_projection_interval=image_projection_interval,
        lateral_reference_coordinate=lateral_reference,
        flesh_centroid_lateral_reference_coordinate=flesh_lateral_reference,
        coordinate=_readonly_array([], dtype=np.float64),
        local_contact_count=_readonly_array([], dtype=np.int64),
        local_flesh_count=_readonly_array([], dtype=np.int64),
        local_calyx_count=_readonly_array([], dtype=np.int64),
        local_v2a_support_count=_readonly_array([], dtype=np.int64),
        selected_flesh_loss_ratio=_readonly_array([], dtype=np.float64),
        selected_calyx_retention_ratio=_readonly_array([], dtype=np.float64),
        retained_contact_ratio=_readonly_array([], dtype=np.float64),
        feasible=_readonly_array([], dtype=bool),
        feasible_block_id=_readonly_array([], dtype=np.int64),
    )


def _image_projection_interval(
    *,
    width: int,
    height: int,
    axis: tuple[float, float],
) -> tuple[float, float]:
    corners = (
        0.0,
        (width - 1) * axis[0],
        (height - 1) * axis[1],
        (width - 1) * axis[0] + (height - 1) * axis[1],
    )
    return min(corners), max(corners)


def _point_projection(point: Point | None, axis: tuple[float, float]) -> float:
    if point is None:
        raise ValueError("point is required")
    return point.x * axis[0] + point.y * axis[1]


def _flesh_loss_count(
    binary_mask: np.ndarray,
    *,
    coordinate: float,
    axis: tuple[float, float],
) -> int:
    projections, _ = _mask_projections(binary_mask, axis, (-axis[1], axis[0]))
    return int(np.count_nonzero(projections > coordinate))


def _retained_count(
    binary_mask: np.ndarray,
    *,
    coordinate: float,
    axis: tuple[float, float],
) -> int:
    projections, _ = _mask_projections(binary_mask, axis, (-axis[1], axis[0]))
    return int(np.count_nonzero(projections <= coordinate))


def _readonly_array(values: Any, *, dtype: np.dtype[Any] | type[Any]) -> np.ndarray:
    copied = np.array(values, dtype=dtype, copy=True)
    copied.setflags(write=False)
    return copied


def _mask_or_empty(mask: np.ndarray | None, empty: np.ndarray) -> np.ndarray:
    return empty if mask is None else mask
