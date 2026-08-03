from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np
import pytest

import strawberry_occlusion.geometry.fixed_axis_search_cutline as search_module
from strawberry_occlusion.evaluation.cutline import calculate_line_side_proxies
from strawberry_occlusion.geometry import (
    FixedAxisSearchCutlineParameters,
    Point,
    calculate_fixed_axis_candidate_curve,
    estimate_fixed_axis_cutline,
    estimate_fixed_axis_search_cutline,
    estimate_visible_mask_cutline,
)


def _outward_mask() -> np.ndarray:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:20, 3:17] = 1
    mask[9:12, 17:20] = 2
    return mask


def _draped_mask() -> np.ndarray:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:20, 3:17] = 1
    mask[8:11, 8:19] = 2
    return mask


def _tip_mask() -> np.ndarray:
    mask = np.zeros((30, 30), dtype=np.uint8)
    mask[8:25, 3:18] = 1
    mask[4:9, 17:24] = 1
    mask[13:20, 18:25] = 2
    mask[4:14, 24] = 2
    return mask


def _directional_mask(axis: tuple[float, float]) -> np.ndarray:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[5:15, 5:15] = 1
    if axis == (1.0, 0.0):
        mask[8:12, 15:18] = 2
    elif axis == (-1.0, 0.0):
        mask[8:12, 2:5] = 2
    elif axis == (0.0, 1.0):
        mask[15:18, 8:12] = 2
    elif axis == (0.0, -1.0):
        mask[2:5, 8:12] = 2
    else:
        mask[12:16, 12:16] = 2
    return mask


def _projection(point: Point, axis: Sequence[float]) -> float:
    return point.x * axis[0] + point.y * axis[1]


def _result_with_feasible_indices(
    feasible_indices: Sequence[int],
):
    mask = _outward_mask()
    v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))

    def candidate_curve(v2a_result, **parameters):
        table = calculate_fixed_axis_candidate_curve(v2a_result, **parameters)
        feasible = np.zeros(len(table), dtype=bool)
        feasible[list(feasible_indices)] = True
        feasible.setflags(write=False)
        return replace(table, feasible=feasible)

    return estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        v2a_result=v2a,
        candidate_curve_function=candidate_curve,
    )


def test_v2a_centred_point_formulas_and_diagnostics() -> None:
    mask = _draped_mask()
    result = estimate_fixed_axis_cutline(mask, removal_axis=(2.0, 1.0))
    axis = result.normalized_removal_axis
    coordinate = result.final_cut_coordinate
    assert result.succeeded and coordinate is not None

    pairs = (
        (
            result.flesh_centroid,
            result.flesh_centroid_projected_reference_point,
            result.flesh_centroid_projected_reference_diagnostics,
        ),
        (
            result.baseline_v1_contact_anchor,
            result.attachment_anchor_projected_reference_point,
            result.attachment_anchor_projected_reference_diagnostics,
        ),
    )
    tangent = (-axis[1], axis[0])
    for original, projected, diagnostics in pairs:
        assert original is not None and projected is not None
        expected_displacement = coordinate - _projection(original, axis)
        assert projected.x == pytest.approx(
            original.x + expected_displacement * axis[0]
        )
        assert projected.y == pytest.approx(
            original.y + expected_displacement * axis[1]
        )
        assert _projection(projected, axis) == pytest.approx(coordinate)
        assert _projection(projected, tangent) == pytest.approx(
            _projection(original, tangent)
        )
        assert diagnostics is not None
        assert diagnostics.nearest_in_bounds_semantic_class in (0, 1, 2)
        assert diagnostics.distance_to_selected_calyx_mask is not None
        assert diagnostics.distance_to_selected_contact_mask is not None


def test_v2a_existing_support_reference_and_coordinate_are_unchanged() -> None:
    mask = _outward_mask()
    first = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    second = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))

    assert first.final_cut_coordinate == second.final_cut_coordinate == 16.0
    assert first.diagnostic_reference_point == second.diagnostic_reference_point


def test_tip_case_moves_inward_to_stronger_central_attachment_evidence() -> None:
    mask = _tip_mask()
    v1 = estimate_visible_mask_cutline(mask)
    v2a = estimate_fixed_axis_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        projection_quantile=0.95,
    )
    v2b = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        projection_quantile=0.95,
        lateral_window_half_width_pixels=4.0,
        minimum_attachment_evidence_fraction=0.60,
        v2a_result=v2a,
    )

    assert v2a.final_cut_coordinate == 23.0
    assert v2b.succeeded
    assert v2b.selected_cut_coordinate is not None
    assert v2b.selected_cut_coordinate < v2a.final_cut_coordinate
    assert v2b.selected_local_contact_count == v2b.maximum_local_contact_evidence
    v2a_nearest_index = int(np.argmin(np.abs(v2b.candidate_table.coordinate - 23.0)))
    assert (
        v2b.selected_local_contact_count
        > v2b.candidate_table.local_contact_count[v2a_nearest_index]
    )
    assert (
        v2b.selected_flesh_loss_ratio
        < calculate_line_side_proxies(mask, v1)["flesh_loss_proxy_ratio"]
    )
    line = v2b.final_cutline
    assert line is not None
    direction = (line.end.x - line.start.x, line.end.y - line.start.y)
    assert direction[0] == pytest.approx(0.0)


def test_zero_default_outward_margin_never_selects_beyond_v2a() -> None:
    mask = _outward_mask()
    v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    v2b = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        v2a_result=v2a,
    )

    assert v2b.succeeded
    assert (
        FixedAxisSearchCutlineParameters((1.0, 0.0)).outward_search_margin_pixels == 0.0
    )
    assert v2b.parameters.outward_search_margin_pixels == 0.0
    assert v2b.search_interval is not None
    assert v2b.search_interval[1] <= v2a.final_cut_coordinate
    assert v2b.selected_cut_coordinate <= v2a.final_cut_coordinate


def test_explicit_positive_outward_margin_remains_a_research_override() -> None:
    mask = _outward_mask()
    v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    default_table = calculate_fixed_axis_candidate_curve(v2a)
    result = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        outward_search_margin_pixels=5.0,
        v2a_result=v2a,
    )

    assert default_table.search_interval is not None
    assert default_table.search_interval[1] == v2a.final_cut_coordinate
    assert result.parameters.outward_search_margin_pixels == 5.0
    assert result.search_interval is not None
    assert result.search_interval[1] == v2a.final_cut_coordinate + 5.0
    assert result.selected_cut_coordinate > v2a.final_cut_coordinate


def test_inward_drape_lowers_flesh_loss_from_v1_and_retains_evidence() -> None:
    mask = _draped_mask()
    v1 = estimate_visible_mask_cutline(mask)
    v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    v2b = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        v2a_result=v2a,
    )
    v1_proxies = calculate_line_side_proxies(mask, v1)

    assert v2b.succeeded
    assert v2b.selected_flesh_loss_ratio is not None
    assert v2b.selected_flesh_loss_ratio < v1_proxies["flesh_loss_proxy_ratio"]
    assert v2b.selected_local_contact_count > 0
    assert v2b.retained_contact_ratio is not None
    assert v2b.retained_contact_ratio > 0.0


@pytest.mark.parametrize(
    "axis",
    ((1.0, 0.0), (-1.0, 0.0), (0.0, 1.0), (0.0, -1.0), (1.3, 0.7)),
)
def test_axes_are_normalized_and_final_line_has_fixed_orientation(
    axis: tuple[float, float],
) -> None:
    result = estimate_fixed_axis_search_cutline(
        _directional_mask(axis),
        removal_axis=axis,
        lateral_window_half_width_pixels=20.0,
    )

    assert result.succeeded
    assert np.linalg.norm(result.normalized_removal_axis) == pytest.approx(1.0)
    line = result.final_cutline
    assert line is not None
    line_direction = np.array((line.end.x - line.start.x, line.end.y - line.start.y))
    assert np.dot(line_direction, result.normalized_removal_axis) == pytest.approx(
        0.0,
        abs=1e-10,
    )


@pytest.mark.parametrize(
    ("name", "value"),
    (
        ("candidate_step_pixels", 0.0),
        ("candidate_step_pixels", -1.0),
        ("inward_search_margin_pixels", -1.0),
        ("outward_search_margin_pixels", float("nan")),
        ("blade_band_half_width_pixels", -0.1),
        ("lateral_window_half_width_pixels", float("inf")),
        ("minimum_attachment_evidence_fraction", 0.0),
        ("minimum_attachment_evidence_fraction", 1.1),
        ("minimum_flesh_band_pixels", -1),
        ("minimum_calyx_band_pixels", 1.5),
    ),
)
def test_search_parameter_validation(name: str, value: float) -> None:
    with pytest.raises((TypeError, ValueError), match=name):
        estimate_fixed_axis_search_cutline(
            _outward_mask(),
            removal_axis=(1.0, 0.0),
            **{name: value},
        )


def test_supplied_v2a_configuration_must_match_search_configuration() -> None:
    mask = _outward_mask()
    v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))

    with pytest.raises(ValueError, match="removal_axis must match"):
        estimate_fixed_axis_search_cutline(
            mask,
            removal_axis=(-1.0, 0.0),
            v2a_result=v2a,
        )
    with pytest.raises(ValueError, match="projection_quantile must match"):
        estimate_fixed_axis_search_cutline(
            mask,
            removal_axis=(2.0, 0.0),
            projection_quantile=0.8,
            v2a_result=v2a,
        )


def test_candidate_interval_is_clipped_inclusive_and_step_is_deterministic() -> None:
    v2a = estimate_fixed_axis_cutline(_outward_mask(), removal_axis=(1.0, 0.0))
    table = calculate_fixed_axis_candidate_curve(
        v2a,
        candidate_step_pixels=2.5,
        inward_search_margin_pixels=100.0,
        outward_search_margin_pixels=100.0,
    )

    assert table.search_interval == (0.0, 23.0)
    assert table.coordinate[0] == 0.0
    assert table.coordinate[-1] == 23.0
    assert np.all(np.diff(table.coordinate) > 0.0)
    repeated = calculate_fixed_axis_candidate_curve(
        v2a,
        candidate_step_pixels=2.5,
        inward_search_margin_pixels=100.0,
        outward_search_margin_pixels=100.0,
    )
    np.testing.assert_array_equal(table.coordinate, repeated.coordinate)
    np.testing.assert_array_equal(table.feasible, repeated.feasible)


def test_evidence_fraction_semantics_and_selected_output_are_deterministic() -> None:
    mask = _tip_mask()
    v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    arguments = {
        "lateral_window_half_width_pixels": 4.0,
        "minimum_attachment_evidence_fraction": 0.8,
    }
    table = calculate_fixed_axis_candidate_curve(v2a, **arguments)
    maximum_contact = int(table.local_contact_count.max())
    threshold = 0.8 * maximum_contact
    expected = (
        (maximum_contact > 0)
        & (table.local_contact_count >= threshold)
        & (table.local_flesh_count >= 1)
        & (table.local_calyx_count >= 1)
    )
    np.testing.assert_array_equal(table.feasible, expected)

    first = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        v2a_result=v2a,
        **arguments,
    )
    second = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        v2a_result=v2a,
        **arguments,
    )
    assert first.selected_candidate_index == second.selected_candidate_index
    assert first.selected_cut_coordinate == second.selected_cut_coordinate


def test_one_contiguous_feasible_block_has_exact_selected_block_diagnostics() -> None:
    result = _result_with_feasible_indices((1, 2, 3))
    table = result.candidate_table

    assert result.succeeded
    assert result.feasible_candidate_count == 3
    assert result.feasible_block_count == 1
    assert result.feasible_hull_start == table.coordinate[1]
    assert result.feasible_hull_end == table.coordinate[3]
    assert result.selected_block_id == 0
    assert result.selected_block_start == table.coordinate[1]
    assert result.selected_block_end == table.coordinate[3]
    assert result.selected_block_candidate_count == 3
    assert result.selected_block_is_singleton is False


def test_multiple_feasible_blocks_preserve_hull_gaps_and_candidate_block_ids() -> None:
    result = _result_with_feasible_indices((1, 2, 4, 7, 8))
    table = result.candidate_table

    assert result.succeeded
    assert result.feasible_candidate_count == 5
    assert result.feasible_block_count == 3
    assert result.feasible_hull_start == table.coordinate[1]
    assert result.feasible_hull_end == table.coordinate[8]
    np.testing.assert_array_equal(
        table.feasible_block_id[:10],
        np.array([-1, 0, 0, -1, 1, -1, -1, 2, 2, -1]),
    )
    assert not bool(table.feasible[3])
    assert not bool(table.feasible[5])
    assert not bool(table.feasible[6])
    assert result.selected_block_id == 2
    assert result.selected_block_candidate_count == 2


def test_outermost_singleton_selected_block_can_be_smaller_than_largest_block() -> None:
    result = _result_with_feasible_indices((1, 2, 3, 5))
    table = result.candidate_table

    assert result.succeeded
    assert result.selected_candidate_index == 5
    assert result.selected_block_id == 1
    assert result.selected_block_start == table.coordinate[5]
    assert result.selected_block_end == table.coordinate[5]
    assert result.selected_block_candidate_count == 1
    assert result.selected_block_is_singleton is True
    assert np.count_nonzero(table.feasible_block_id == 0) == 3
    assert result.selected_block_candidate_count < np.count_nonzero(
        table.feasible_block_id == 0
    )


def test_local_counts_and_selected_pair_proxies_match_direct_pixel_calculation() -> (
    None
):
    mask = _tip_mask()
    result = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        lateral_window_half_width_pixels=4.0,
        minimum_attachment_evidence_fraction=0.6,
    )
    index = result.selected_candidate_index
    coordinate = result.selected_cut_coordinate
    assert result.succeeded and index is not None and coordinate is not None

    assert result.selected_local_contact_count == int(
        np.count_nonzero(result.selected_local_contact_mask)
    )
    assert result.selected_local_flesh_count == int(
        np.count_nonzero(result.selected_local_flesh_mask)
    )
    assert result.selected_local_calyx_count == int(
        np.count_nonzero(result.selected_local_calyx_mask)
    )
    flesh_coordinates = np.argwhere(result.selected_flesh_mask)
    flesh_loss = int(np.count_nonzero(flesh_coordinates[:, 1] > coordinate))
    assert result.selected_flesh_loss_count == flesh_loss
    assert result.selected_flesh_loss_ratio == pytest.approx(
        flesh_loss / len(flesh_coordinates)
    )
    threshold = (
        result.parameters.minimum_attachment_evidence_fraction
        * result.maximum_local_contact_evidence
    )
    assert result.candidate_table.local_contact_count[index] >= threshold
    assert bool(result.candidate_table.feasible[index])


def test_no_feasible_candidate_is_structured_and_does_not_fall_back() -> None:
    result = estimate_fixed_axis_search_cutline(
        _outward_mask(),
        removal_axis=(1.0, 0.0),
        minimum_calyx_band_pixels=100,
    )

    assert result.status == "failed"
    assert result.failure_code == "no_feasible_fixed_axis_cut"
    assert result.selected_cut_coordinate is None
    assert result.final_cutline is None
    assert not np.any(result.candidate_table.feasible)
    assert result.feasible_candidate_count == 0
    assert result.feasible_block_count == 0
    assert result.feasible_hull_start is None
    assert result.feasible_hull_end is None


def test_zero_local_contact_evidence_has_no_feasible_or_malformed_values() -> None:
    mask = _outward_mask()
    v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    anchor = v2a.baseline_v1_contact_anchor
    assert anchor is not None
    shifted_reference = replace(
        v2a,
        baseline_v1_contact_anchor=Point(anchor.x, -100.0),
    )

    result = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        lateral_window_half_width_pixels=0.0,
        v2a_result=shifted_reference,
    )

    assert result.status == "failed"
    assert result.failure_code == "no_feasible_fixed_axis_cut"
    assert result.maximum_local_contact_evidence == 0
    assert result.feasible_candidate_count == 0
    assert result.feasible_block_count == 0
    assert not np.any(result.candidate_table.feasible)
    for values in (
        result.candidate_table.selected_flesh_loss_ratio,
        result.candidate_table.selected_calyx_retention_ratio,
        result.candidate_table.retained_contact_ratio,
    ):
        assert np.all(np.isfinite(values))


def test_injectable_candidate_curve_requires_lateral_reference_coordinate() -> None:
    mask = _outward_mask()
    v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))

    def missing_reference(v2a_result, **parameters):
        table = calculate_fixed_axis_candidate_curve(v2a_result, **parameters)
        return replace(table, lateral_reference_coordinate=None)

    with pytest.raises(
        ValueError,
        match="candidate curve must provide lateral_reference_coordinate",
    ):
        estimate_fixed_axis_search_cutline(
            mask,
            removal_axis=(1.0, 0.0),
            v2a_result=v2a,
            candidate_curve_function=missing_reference,
        )


def test_v1_and_v2a_numerical_regression_safety_across_v2b_margin_modes() -> None:
    mask = _outward_mask()
    expected_v1 = estimate_visible_mask_cutline(mask)
    expected_v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))

    estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        v2a_result=expected_v2a,
    )
    estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        outward_search_margin_pixels=5.0,
        v2a_result=expected_v2a,
    )
    repeated_v1 = estimate_visible_mask_cutline(mask)
    repeated_v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))

    assert (
        expected_v1.attachment_anchor
        == repeated_v1.attachment_anchor
        == Point(
            16.0,
            10.0,
        )
    )
    assert expected_v1.final_cutline == repeated_v1.final_cutline
    assert (
        expected_v2a.final_cut_coordinate == repeated_v2a.final_cut_coordinate == 16.0
    )
    assert expected_v2a.final_cutline == repeated_v2a.final_cutline


def test_v2a_failure_is_a_structured_v2b_prerequisite_failure() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[2:8, 2:8] = 1
    result = estimate_fixed_axis_search_cutline(mask, removal_axis=(1.0, 0.0))

    assert result.status == "failed"
    assert result.failure_code == "v2a_prerequisite_failed"
    assert result.candidate_count == 0


def test_candidate_curve_helper_is_called_once_and_one_final_full_mask_is_built(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    mask = _outward_mask()
    v2a = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    curve_calls = 0
    indices_calls = 0
    original_curve = calculate_fixed_axis_candidate_curve
    original_indices = search_module.np.indices

    def curve(*args: object, **kwargs: object):
        nonlocal curve_calls
        curve_calls += 1
        return original_curve(*args, **kwargs)

    def indices(*args: object, **kwargs: object):
        nonlocal indices_calls
        indices_calls += 1
        return original_indices(*args, **kwargs)

    monkeypatch.setattr(search_module.np, "indices", indices)
    result = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        v2a_result=v2a,
        candidate_curve_function=curve,
    )

    assert result.succeeded
    assert curve_calls == 1
    assert indices_calls == 1


def test_no_input_mutation_and_all_candidate_and_mask_diagnostics_are_read_only() -> (
    None
):
    mask = _tip_mask()
    original = mask.copy()
    result = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        lateral_window_half_width_pixels=4.0,
    )

    np.testing.assert_array_equal(mask, original)
    arrays = (
        result.candidate_table.coordinate,
        result.candidate_table.local_contact_count,
        result.candidate_table.feasible,
        result.selected_flesh_mask,
        result.selected_calyx_mask,
        result.contact_mask,
        result.v2a_support_mask,
        result.selected_local_window_mask,
        result.selected_local_contact_mask,
        result.selected_local_flesh_mask,
        result.selected_local_calyx_mask,
        result.selected_local_v2a_support_mask,
    )
    assert all(not array.flags.writeable for array in arrays)


def test_selected_reference_projects_anchor_and_shifts_are_exact() -> None:
    result = estimate_fixed_axis_search_cutline(
        _tip_mask(),
        removal_axis=(1.0, 0.0),
        lateral_window_half_width_pixels=4.0,
    )
    anchor = result.v1_attachment_anchor
    reference = result.diagnostic_reference_point
    coordinate = result.selected_cut_coordinate
    assert result.succeeded
    assert anchor is not None and reference is not None and coordinate is not None
    assert _projection(reference, result.normalized_removal_axis) == pytest.approx(
        coordinate
    )
    assert _projection(reference, result.blade_tangent) == pytest.approx(
        _projection(anchor, result.blade_tangent)
    )
    assert result.coordinate_shift_from_v1 == pytest.approx(
        coordinate - result.v1_anchor_projection
    )
    assert result.coordinate_shift_from_v2a == pytest.approx(
        coordinate - result.v2a_cut_coordinate
    )
