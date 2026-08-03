from dataclasses import FrozenInstanceError, replace
from importlib import import_module
from math import ceil, hypot, sqrt

import numpy as np
import pytest

from strawberry_occlusion.geometry import (
    FixedAxisCutlineParameters,
    FixedAxisCutlineResult,
    estimate_fixed_axis_cutline,
    estimate_visible_mask_cutline,
)
from strawberry_occlusion.visualization.fixed_axis_cutline import (
    BASELINE_ANCHOR_COLOR,
    CANDIDATE_LINE_COLOR,
    CONTACT_COLOR,
    FINAL_LINE_COLOR,
    FLESH_CENTROID_COLOR,
    REFERENCE_POINT_COLOR,
    REMOVAL_AXIS_COLOR,
    SUPPORT_COLOR,
    _coincidence_text,
    create_fixed_axis_cutline_visualization,
)


def _outward_calyx_mask() -> np.ndarray:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:20, 3:17] = 1
    mask[9:12, 17:20] = 2
    return mask


def _inward_draped_calyx_mask() -> np.ndarray:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:20, 3:17] = 1
    mask[8:11, 8:19] = 2
    return mask


def _isolated_outer_contact_mask() -> np.ndarray:
    mask = np.zeros((32, 20), dtype=np.uint8)
    mask[3:28, 2:11] = 1
    mask[28, 2:16] = 1
    mask[26:29, 15] = 1
    mask[3:26, 11:13] = 2
    mask[25, 13:15] = 2
    return mask


def _flesh_loss_proxy(
    mask: np.ndarray,
    *,
    axis: tuple[float, float],
    coordinate: float,
) -> int:
    pixel_y, pixel_x = np.indices(mask.shape, dtype=np.float64)
    projections = pixel_x * axis[0] + pixel_y * axis[1]
    return int(np.count_nonzero((mask == 1) & (projections > coordinate)))


def _assert_fixed_line_equation(result: FixedAxisCutlineResult) -> None:
    assert result.final_cutline is not None
    assert result.final_cut_coordinate is not None
    axis = np.asarray(result.normalized_removal_axis)
    for point in (result.final_cutline.start, result.final_cutline.end):
        assert np.dot((point.x, point.y), axis) == pytest.approx(
            result.final_cut_coordinate,
            abs=1e-9,
        )
    line_direction = np.asarray(
        (
            result.final_cutline.end.x - result.final_cutline.start.x,
            result.final_cutline.end.y - result.final_cutline.start.y,
        )
    )
    assert np.dot(line_direction, axis) == pytest.approx(0.0, abs=1e-9)


def test_public_api_exposes_frozen_parameter_and_result_types() -> None:
    result = estimate_fixed_axis_cutline(
        _outward_calyx_mask(),
        removal_axis=(1.0, 0.0),
    )

    assert isinstance(result, FixedAxisCutlineResult)
    assert isinstance(result.parameters, FixedAxisCutlineParameters)
    with pytest.raises(FrozenInstanceError):
        result.final_cut_coordinate = 4.0  # type: ignore[misc]


def test_outward_calyx_stays_at_outer_attachment_without_deep_shift() -> None:
    result = estimate_fixed_axis_cutline(
        _outward_calyx_mask(),
        removal_axis=(1.0, 0.0),
    )

    assert result.succeeded
    assert result.unshifted_cut_coordinate == 16.0
    assert result.final_cut_coordinate == 16.0
    assert result.maximum_selected_flesh_projection == 16.0
    assert result.cut_margin_to_flesh_extent == 0.0
    assert result.candidate_final_lines_coincide is True
    assert result.candidate_cutline == result.final_cutline


def test_inward_drape_moves_v1_anchor_inward_but_v2a_uses_outer_contact() -> None:
    mask = _inward_draped_calyx_mask()
    baseline = estimate_visible_mask_cutline(mask)
    result = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))

    assert baseline.succeeded
    assert baseline.attachment_anchor == result.baseline_v1_contact_anchor
    assert baseline.attachment_anchor is not None
    assert result.final_cut_coordinate is not None
    baseline_axis_coordinate = baseline.attachment_anchor.x
    assert baseline_axis_coordinate < result.final_cut_coordinate
    assert result.final_cut_coordinate == result.maximum_contact_projection

    baseline_loss = _flesh_loss_proxy(
        mask,
        axis=(1.0, 0.0),
        coordinate=baseline_axis_coordinate,
    )
    v2a_loss = _flesh_loss_proxy(
        mask,
        axis=(1.0, 0.0),
        coordinate=result.final_cut_coordinate,
    )
    assert baseline_loss > 0
    assert v2a_loss < baseline_loss


def test_v2a_orientation_does_not_rotate_when_calyx_shape_changes() -> None:
    outward = estimate_fixed_axis_cutline(
        _outward_calyx_mask(),
        removal_axis=(1.0, 0.0),
    )
    draped = estimate_fixed_axis_cutline(
        _inward_draped_calyx_mask(),
        removal_axis=(1.0, 0.0),
    )

    for result in (outward, draped):
        assert result.final_cutline is not None
        assert result.final_cutline.start.x == pytest.approx(
            result.final_cut_coordinate
        )
        assert result.final_cutline.end.x == pytest.approx(result.final_cut_coordinate)
        _assert_fixed_line_equation(result)


@pytest.mark.parametrize(
    ("axis", "normalized"),
    (
        ((1.0, 0.0), (1.0, 0.0)),
        ((-1.0, 0.0), (-1.0, 0.0)),
        ((0.0, 1.0), (0.0, 1.0)),
        ((0.0, -1.0), (0.0, -1.0)),
        ((3.0, 4.0), (0.6, 0.8)),
    ),
    ids=("positive-x", "negative-x", "positive-y", "negative-y", "diagonal"),
)
def test_supported_removal_axes_produce_fixed_perpendicular_lines(
    axis: tuple[float, float],
    normalized: tuple[float, float],
) -> None:
    result = estimate_fixed_axis_cutline(
        _outward_calyx_mask(),
        removal_axis=axis,
    )

    assert result.succeeded
    assert result.normalized_removal_axis == pytest.approx(normalized)
    assert hypot(*result.normalized_removal_axis) == pytest.approx(1.0)
    _assert_fixed_line_equation(result)


def test_axis_normalization_preserves_all_fixed_axis_geometry() -> None:
    unit = estimate_fixed_axis_cutline(
        _inward_draped_calyx_mask(),
        removal_axis=(1.0, 0.0),
    )
    scaled = estimate_fixed_axis_cutline(
        _inward_draped_calyx_mask(),
        removal_axis=(2.0, 0.0),
    )

    assert scaled.normalized_removal_axis == unit.normalized_removal_axis
    assert scaled.unshifted_cut_coordinate == unit.unshifted_cut_coordinate
    assert scaled.final_cut_coordinate == unit.final_cut_coordinate
    assert scaled.candidate_cutline == unit.candidate_cutline
    assert scaled.final_cutline == unit.final_cutline
    assert scaled.diagnostic_reference_point == unit.diagnostic_reference_point
    np.testing.assert_array_equal(scaled.support_mask, unit.support_mask)


@pytest.mark.parametrize(
    "axis",
    (
        (0.0, 0.0),
        (1.0,),
        (1.0, 2.0, 3.0),
        ((1.0, 0.0),),
        (np.nan, 0.0),
        (np.inf, 0.0),
        (1.0 + 2.0j, 0.0),
        (True, False),
        "10",
    ),
)
def test_invalid_removal_axes_are_rejected(axis: object) -> None:
    with pytest.raises(ValueError, match="removal_axis"):
        estimate_fixed_axis_cutline(
            _outward_calyx_mask(),
            removal_axis=axis,  # type: ignore[arg-type]
        )


@pytest.mark.parametrize("quantile", (0.05, 0.5, 0.95, 1.0))
def test_projection_quantile_uses_exact_noninterpolated_order_statistic(
    quantile: float,
) -> None:
    result = estimate_fixed_axis_cutline(
        _inward_draped_calyx_mask(),
        removal_axis=(0.0, 1.0),
        projection_quantile=quantile,
    )
    coordinates = np.argwhere(result.contact_mask)
    sorted_projections = np.sort(coordinates[:, 0].astype(np.float64))
    expected_index = min(
        len(sorted_projections) - 1,
        ceil(quantile * len(sorted_projections)) - 1,
    )

    assert result.unshifted_cut_coordinate == sorted_projections[expected_index]
    assert result.selected_contact_projection in sorted_projections


@pytest.mark.parametrize(
    "quantile",
    (0.0, -0.1, 1.0001, np.nan, np.inf, True, "0.95"),
)
def test_invalid_projection_quantiles_are_rejected(quantile: object) -> None:
    with pytest.raises(ValueError, match="projection_quantile"):
        estimate_fixed_axis_cutline(
            _outward_calyx_mask(),
            removal_axis=(1.0, 0.0),
            projection_quantile=quantile,  # type: ignore[arg-type]
        )


def test_duplicate_contact_projections_and_repeated_output_are_deterministic() -> None:
    mask = _outward_calyx_mask()
    original = mask.copy()

    first = estimate_fixed_axis_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        projection_quantile=0.37,
    )
    second = estimate_fixed_axis_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        projection_quantile=0.37,
    )

    contact_x = np.argwhere(first.contact_mask)[:, 1]
    assert len(np.unique(contact_x)) == 1
    assert first == second
    assert first.unshifted_cut_coordinate == float(contact_x[0])
    np.testing.assert_array_equal(first.support_mask, second.support_mask)
    np.testing.assert_array_equal(mask, original)


def test_high_quantile_ignores_one_isolated_extreme_but_one_follows_maximum() -> None:
    mask = _isolated_outer_contact_mask()

    robust = estimate_fixed_axis_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        projection_quantile=0.95,
    )
    maximum = estimate_fixed_axis_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        projection_quantile=1.0,
    )

    assert robust.contact_pixel_count == 25
    assert robust.unshifted_cut_coordinate == 10.0
    assert robust.maximum_contact_projection == 15.0
    assert robust.unshifted_cut_coordinate < robust.maximum_contact_projection
    assert maximum.unshifted_cut_coordinate == maximum.maximum_contact_projection
    assert maximum.unshifted_cut_coordinate == 15.0


def test_support_threshold_and_width_are_diagnostic_only() -> None:
    mask = _inward_draped_calyx_mask()
    narrow = estimate_fixed_axis_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        support_band_width_pixels=0.0,
    )
    wide = estimate_fixed_axis_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        support_band_width_pixels=20.0,
    )

    assert narrow.unshifted_cut_coordinate == wide.unshifted_cut_coordinate
    assert narrow.final_cut_coordinate == wide.final_cut_coordinate
    assert narrow.candidate_cutline == wide.candidate_cutline
    assert narrow.final_cutline == wide.final_cutline
    assert narrow.support_pixel_count < wide.support_pixel_count
    for result in (narrow, wide):
        coordinates = np.argwhere(result.support_mask)
        projections = coordinates[:, 1].astype(np.float64)
        threshold = (
            result.unshifted_cut_coordinate
            - result.parameters.support_band_width_pixels
        )
        assert np.all(projections >= threshold)
        assert np.all(result.support_mask <= result.contact_mask)


def test_zero_width_support_keeps_selected_pixel_for_irrational_diagonal_axis() -> None:
    result = estimate_fixed_axis_cutline(
        _inward_draped_calyx_mask(),
        removal_axis=(1.0, sqrt(2.0)),
        projection_quantile=0.73,
        support_band_width_pixels=0.0,
    )
    contact_coordinates = np.argwhere(result.contact_mask)[:, ::-1].astype(np.float64)
    contact_projections = contact_coordinates @ np.asarray(
        result.normalized_removal_axis
    )
    selected_index = min(
        len(contact_projections) - 1,
        ceil(result.parameters.projection_quantile * len(contact_projections)) - 1,
    )
    selected_projection = np.sort(contact_projections)[selected_index]
    selected_coordinates = contact_coordinates[
        contact_projections == selected_projection
    ].astype(np.intp)

    assert result.support_pixel_count > 0
    assert result.unshifted_cut_coordinate == selected_projection
    assert any(
        result.support_mask[coordinate_y, coordinate_x]
        for coordinate_x, coordinate_y in selected_coordinates
    )
    assert not result.support_mask.flags.writeable


def test_diagnostic_reference_point_lies_on_final_line() -> None:
    result = estimate_fixed_axis_cutline(
        _inward_draped_calyx_mask(),
        removal_axis=(1.0, 1.0),
        signed_offset_pixels=0.75,
    )

    assert result.diagnostic_reference_point is not None
    assert result.final_cut_coordinate is not None
    reference = result.diagnostic_reference_point
    projection = np.dot(
        (reference.x, reference.y),
        result.normalized_removal_axis,
    )
    assert projection == pytest.approx(result.final_cut_coordinate, abs=1e-10)


def test_signed_offset_shifts_coordinate_exactly_without_rotating_line() -> None:
    candidate = estimate_fixed_axis_cutline(
        _outward_calyx_mask(),
        removal_axis=(1.0, 0.0),
    )
    shifted = estimate_fixed_axis_cutline(
        _outward_calyx_mask(),
        removal_axis=(1.0, 0.0),
        signed_offset_pixels=2.5,
    )

    assert candidate.final_cut_coordinate is not None
    assert shifted.final_cut_coordinate == candidate.final_cut_coordinate + 2.5
    assert shifted.signed_offset_pixels == 2.5
    assert shifted.final_cutline is not None
    assert shifted.final_cutline.start.x == pytest.approx(18.5)
    assert shifted.final_cutline.end.x == pytest.approx(18.5)
    assert shifted.candidate_final_lines_coincide is False
    _assert_fixed_line_equation(shifted)


@pytest.mark.parametrize(
    ("present_class", "failure_code"),
    ((2, "no_flesh"), (1, "no_calyx")),
)
def test_missing_classes_return_baseline_style_structured_failures(
    present_class: int,
    failure_code: str,
) -> None:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:6, 3:6] = present_class

    result = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))

    assert result.status == "failed"
    assert result.failure_code == failure_code
    assert result.failure_reason
    assert result.contact_pixel_count == 0
    assert result.candidate_cutline is None
    assert result.final_cutline is None


def test_no_contact_returns_baseline_style_structured_failure() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[12:18, 2:8] = 1
    mask[1:4, 15:18] = 2

    result = estimate_fixed_axis_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        calyx_dilation_radius=2,
    )

    assert result.failure_code == "no_contact"
    assert "dilation radius 2" in (result.failure_reason or "")
    assert result.support_pixel_count == 0


def test_candidate_line_clipping_failure_is_structured(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = import_module("strawberry_occlusion.geometry.fixed_axis_cutline")
    monkeypatch.setattr(
        module,
        "clip_infinite_line_to_image",
        lambda *args, **kwargs: None,
    )

    result = estimate_fixed_axis_cutline(
        _outward_calyx_mask(),
        removal_axis=(1.0, 0.0),
    )

    assert result.failure_code == "candidate_line_outside_image"
    assert result.candidate_cutline is None
    assert result.final_cutline is None


def test_final_offset_line_outside_image_is_structured() -> None:
    result = estimate_fixed_axis_cutline(
        _outward_calyx_mask(),
        removal_axis=(1.0, 0.0),
        signed_offset_pixels=100.0,
    )

    assert result.failure_code == "offset_line_outside_image"
    assert result.candidate_cutline is not None
    assert result.final_cut_coordinate == 116.0
    assert result.final_cutline is None
    assert result.diagnostic_reference_point is not None


def test_fixed_axis_can_succeed_when_v1_direction_is_degenerate() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1
    mask[1, 2] = 2
    baseline = estimate_visible_mask_cutline(mask)

    result = estimate_fixed_axis_cutline(mask, removal_axis=(0.0, -1.0))

    assert baseline.failure_code == "degenerate_direction"
    assert result.succeeded
    assert result.baseline_v1_fruit_to_attachment_direction is None
    assert result.axis_disagreement_angle_degrees is None
    assert result.final_cut_coordinate == -2.0
    _assert_fixed_line_equation(result)


def test_selected_pair_and_contact_are_exactly_reused_from_baseline_v1() -> None:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:8, 2:6] = 1
    mask[2:4, 3:5] = 2
    mask[13:21, 12:22] = 1
    mask[10:13, 14:20] = 2
    baseline = estimate_visible_mask_cutline(mask)

    result = estimate_fixed_axis_cutline(mask, removal_axis=(0.0, 1.0))

    assert result.selected_flesh_component == baseline.selected_flesh_component
    assert result.selected_calyx_component == baseline.selected_calyx_component
    assert result.selected_flesh_pixel_count == baseline.selected_flesh_pixel_count
    assert result.selected_calyx_pixel_count == baseline.selected_calyx_pixel_count
    assert result.baseline_v1_contact_anchor == baseline.attachment_anchor
    np.testing.assert_array_equal(result.contact_mask, baseline.contact_band)


def test_input_is_unchanged_and_all_diagnostic_masks_are_read_only() -> None:
    mask = _inward_draped_calyx_mask()
    original = mask.copy()

    result = estimate_fixed_axis_cutline(mask, removal_axis=(sqrt(2.0), 0.0))

    np.testing.assert_array_equal(mask, original)
    diagnostic_masks = (
        result.flesh_mask,
        result.calyx_mask,
        result.selected_flesh_mask,
        result.selected_calyx_mask,
        result.contact_mask,
        result.support_mask,
    )
    assert all(
        not diagnostic_mask.flags.writeable for diagnostic_mask in diagnostic_masks
    )
    for diagnostic_mask in diagnostic_masks:
        with pytest.raises(ValueError):
            diagnostic_mask[0, 0] = True


@pytest.mark.parametrize(
    ("parameter", "value"),
    (
        ("support_band_width_pixels", -1.0),
        ("support_band_width_pixels", np.inf),
        ("signed_offset_pixels", np.nan),
        ("signed_offset_pixels", True),
    ),
)
def test_other_invalid_finite_parameters_are_rejected(
    parameter: str,
    value: object,
) -> None:
    with pytest.raises(ValueError, match=parameter):
        estimate_fixed_axis_cutline(
            _outward_calyx_mask(),
            removal_axis=(1.0, 0.0),
            **{parameter: value},
        )


def test_visualization_is_headless_deterministic_and_shows_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    mask = _inward_draped_calyx_mask()
    original = mask.copy()
    result = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))

    first = create_fixed_axis_cutline_visualization(mask, result)
    second = create_fixed_axis_cutline_visualization(mask, result)

    assert first.mode == "RGB"
    assert first.size == second.size
    assert first.tobytes() == second.tobytes()
    assert "signed offset is zero" in _coincidence_text(result)
    rendered_colors = {
        tuple(color) for color in np.unique(np.asarray(first).reshape(-1, 3), axis=0)
    }
    for expected_color in (
        CONTACT_COLOR,
        SUPPORT_COLOR,
        FLESH_CENTROID_COLOR,
        BASELINE_ANCHOR_COLOR,
        REMOVAL_AXIS_COLOR,
        CANDIDATE_LINE_COLOR,
        FINAL_LINE_COLOR,
        REFERENCE_POINT_COLOR,
    ):
        assert expected_color in rendered_colors
    np.testing.assert_array_equal(mask, original)


def test_visualization_rejects_mismatched_result_dimensions() -> None:
    mask = _outward_calyx_mask()
    result = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    mismatched = replace(result, image_width=result.image_width + 1)

    with pytest.raises(ValueError, match="result dimensions"):
        create_fixed_axis_cutline_visualization(mask, mismatched)
