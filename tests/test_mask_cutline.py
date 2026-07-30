from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

import strawberry_occlusion.visualization as visualization_package
from strawberry_occlusion.geometry import (
    LineSegment,
    Point,
    clip_infinite_line_to_image,
    estimate_visible_mask_cutline,
)
from strawberry_occlusion.visualization.cutline import (
    STATUS_HEIGHT,
    TITLE_HEIGHT,
    create_mask_cutline_visualization,
)


def test_visualization_package_exports_cutline_module() -> None:
    assert visualization_package.__all__ == ["cutline", "segmentation"]
    assert (
        visualization_package.cutline.create_mask_cutline_visualization
        is create_mask_cutline_visualization
    )


def _vertical_attachment_mask() -> np.ndarray:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[4:10, 3:9] = 1
    mask[2:4, 5:7] = 2
    return mask


def _horizontal_attachment_mask() -> np.ndarray:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 2:7] = 1
    mask[5:7, 7:9] = 2
    return mask


def _assert_segment_in_bounds(
    segment: LineSegment,
    *,
    width: int,
    height: int,
) -> None:
    for point in (segment.start, segment.end):
        assert 0.0 <= point.x <= width - 1
        assert 0.0 <= point.y <= height - 1
    assert segment.start.x in (0.0, width - 1) or segment.start.y in (0.0, height - 1)
    assert segment.end.x in (0.0, width - 1) or segment.end.y in (0.0, height - 1)


def test_simple_touching_shapes_produce_contact_anchor_and_horizontal_cutline() -> None:
    result = estimate_visible_mask_cutline(_vertical_attachment_mask())

    assert result.succeeded
    assert result.failure_reason is None
    assert result.flesh_component_count == 1
    assert result.calyx_component_count == 1
    assert result.contact_pixel_count == 4
    assert result.contact_component_count == 1
    assert result.flesh_centroid == Point(x=5.5, y=6.5)
    assert result.attachment_anchor == Point(x=5.5, y=4.0)
    assert result.fruit_to_attachment_direction == pytest.approx((0.0, -1.0))
    assert result.candidate_cutline == LineSegment(
        start=Point(0.0, 4.0),
        end=Point(11.0, 4.0),
    )
    assert result.final_cutline == result.candidate_cutline


def test_horizontal_direction_produces_vertical_cutline() -> None:
    result = estimate_visible_mask_cutline(_horizontal_attachment_mask())

    assert result.succeeded
    assert result.fruit_to_attachment_direction == pytest.approx((1.0, 0.0))
    assert result.attachment_anchor == Point(x=6.0, y=5.5)
    assert result.final_cutline == LineSegment(
        start=Point(6.0, 0.0),
        end=Point(6.0, 11.0),
    )


def test_rotated_shapes_produce_perpendicular_clipped_line() -> None:
    y, x = np.mgrid[:31, :31]
    flesh = (x - 12) ** 2 + (y - 18) ** 2 <= 6**2
    calyx = (x - 20) ** 2 + (y - 10) ** 2 <= 5**2
    mask = np.zeros((31, 31), dtype=np.uint8)
    mask[flesh] = 1
    mask[calyx] = 2

    result = estimate_visible_mask_cutline(mask)

    assert result.succeeded
    direction = np.asarray(result.fruit_to_attachment_direction)
    segment = result.final_cutline
    assert segment is not None
    line_direction = np.asarray(
        [segment.end.x - segment.start.x, segment.end.y - segment.start.y]
    )
    assert float(np.dot(direction, line_direction)) == pytest.approx(0.0, abs=1e-9)
    assert direction[0] > 0
    assert direction[1] < 0
    _assert_segment_in_bounds(segment, width=31, height=31)


@pytest.mark.parametrize(
    ("missing_class", "failure_code"),
    ((2, "no_calyx"), (1, "no_flesh")),
)
def test_missing_semantic_classes_return_clear_failures(
    missing_class: int,
    failure_code: str,
) -> None:
    mask = _vertical_attachment_mask()
    mask[mask == missing_class] = 0

    result = estimate_visible_mask_cutline(mask)

    assert not result.succeeded
    assert result.status == "failed"
    assert result.failure_code == failure_code
    assert result.failure_reason
    assert result.final_cutline is None


def test_non_touching_components_return_no_contact_failure() -> None:
    mask = np.zeros((20, 20), dtype=np.uint8)
    mask[12:18, 2:8] = 1
    mask[1:4, 15:18] = 2

    result = estimate_visible_mask_cutline(mask, calyx_dilation_radius=2)

    assert result.failure_code == "no_contact"
    assert "dilation radius 2" in (result.failure_reason or "")
    assert result.contact_pixel_count == 0


def test_multiple_components_select_pair_with_largest_contact() -> None:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:8, 2:6] = 1
    mask[2:4, 3:5] = 2
    mask[13:21, 12:22] = 1
    mask[10:13, 14:20] = 2

    result = estimate_visible_mask_cutline(mask)

    assert result.succeeded
    assert result.flesh_component_count == 2
    assert result.calyx_component_count == 2
    assert result.selected_flesh_pixel_count == 80
    assert result.selected_calyx_pixel_count == 18
    assert result.contact_pixel_count == 8
    assert result.attachment_anchor is not None
    assert result.attachment_anchor.y == pytest.approx(13.0)


def test_four_connectivity_keeps_diagonally_touching_flesh_regions_separate() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:6, 2:4] = 1
    mask[6:8, 4:6] = 1
    mask[2:4, 2:4] = 2

    result = estimate_visible_mask_cutline(mask, component_connectivity=4)

    assert result.succeeded
    assert result.parameters.component_connectivity == 4
    assert result.flesh_component_count == 2
    assert result.selected_flesh_pixel_count == 4
    assert result.contact_component_count == 1


def test_disconnected_contact_regions_are_counted_and_share_one_centroid() -> None:
    mask = np.zeros((15, 15), dtype=np.uint8)
    mask[4:13, 2:4] = 1
    mask[4:13, 10:12] = 1
    mask[11:13, 2:12] = 1
    mask[2:4, 3:11] = 2

    result = estimate_visible_mask_cutline(mask)

    assert result.succeeded
    assert result.contact_pixel_count == 4
    assert result.contact_component_count == 2
    assert result.attachment_anchor == Point(x=6.5, y=4.0)


def test_tiny_contact_region_is_handled() -> None:
    mask = np.zeros((10, 10), dtype=np.uint8)
    mask[4:9, 4:9] = 1
    mask[2:4, 2:4] = 2

    result = estimate_visible_mask_cutline(mask)

    assert result.succeeded
    assert result.contact_pixel_count == 1
    assert result.attachment_anchor == Point(x=4.0, y=4.0)


def test_contact_at_flesh_centroid_returns_degenerate_direction_failure() -> None:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1
    mask[1, 2] = 2

    result = estimate_visible_mask_cutline(mask)

    assert result.failure_code == "degenerate_direction"
    assert result.contact_component_count == 1
    assert result.flesh_centroid == Point(x=2.0, y=2.0)
    assert result.attachment_anchor == result.flesh_centroid
    assert result.candidate_cutline is None
    assert result.final_cutline is None


@pytest.mark.parametrize(
    ("point", "direction", "expected"),
    (
        (
            Point(2.0, 2.0),
            (1.0, 0.0),
            LineSegment(Point(0.0, 2.0), Point(4.0, 2.0)),
        ),
        (
            Point(2.0, 2.0),
            (0.0, 1.0),
            LineSegment(Point(2.0, 0.0), Point(2.0, 4.0)),
        ),
        (
            Point(2.0, 2.0),
            (1.0, 1.0),
            LineSegment(Point(0.0, 0.0), Point(4.0, 4.0)),
        ),
    ),
    ids=("horizontal", "vertical", "diagonal"),
)
def test_clip_infinite_line_to_image_directly(
    point: Point,
    direction: tuple[float, float],
    expected: LineSegment,
) -> None:
    assert (
        clip_infinite_line_to_image(
            point,
            direction,
            width=5,
            height=5,
        )
        == expected
    )


def test_signed_offset_moves_along_fruit_to_attachment_direction() -> None:
    mask = _vertical_attachment_mask()
    positive = estimate_visible_mask_cutline(mask, signed_offset=2.0)
    negative = estimate_visible_mask_cutline(mask, signed_offset=-2.0)

    assert positive.offset_anchor == Point(x=5.5, y=2.0)
    assert negative.offset_anchor == Point(x=5.5, y=6.0)
    assert positive.final_cutline is not None
    assert negative.final_cutline is not None
    assert positive.final_cutline.start.y == pytest.approx(2.0)
    assert negative.final_cutline.start.y == pytest.approx(6.0)


def test_large_offset_returns_clear_clipping_failure() -> None:
    result = estimate_visible_mask_cutline(
        _vertical_attachment_mask(),
        signed_offset=20.0,
    )

    assert result.failure_code == "offset_line_outside_image"
    assert result.candidate_cutline is not None
    assert result.offset_anchor == Point(x=5.5, y=-16.0)
    assert result.final_cutline is None


def test_mask_validation_rejects_bad_dimensions_dtype_and_values() -> None:
    with pytest.raises(ValueError, match=r"shape \[H, W\]"):
        estimate_visible_mask_cutline(np.zeros((2, 2, 1), dtype=np.uint8))
    with pytest.raises(TypeError, match="integer dtype"):
        estimate_visible_mask_cutline(np.zeros((3, 3), dtype=np.float32))
    invalid = np.zeros((3, 3), dtype=np.uint8)
    invalid[1, 1] = 3
    with pytest.raises(ValueError, match="only class IDs 0, 1, and 2"):
        estimate_visible_mask_cutline(invalid)


def test_zero_calyx_dilation_radius_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        estimate_visible_mask_cutline(
            _vertical_attachment_mask(),
            calyx_dilation_radius=0,
        )


def test_estimation_is_deterministic_and_does_not_mutate_input() -> None:
    mask = _vertical_attachment_mask()
    original = mask.copy()

    first = estimate_visible_mask_cutline(mask, signed_offset=1.25)
    second = estimate_visible_mask_cutline(mask, signed_offset=1.25)

    assert first == second
    np.testing.assert_array_equal(first.flesh_mask, second.flesh_mask)
    np.testing.assert_array_equal(first.calyx_mask, second.calyx_mask)
    np.testing.assert_array_equal(first.contact_band, second.contact_band)
    np.testing.assert_array_equal(mask, original)
    assert not first.flesh_mask.flags.writeable
    assert not first.contact_band.flags.writeable


def test_visualization_is_headless_deterministic_and_contains_six_panels(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    mask = _vertical_attachment_mask()
    original = mask.copy()
    result = estimate_visible_mask_cutline(mask, signed_offset=1.0)

    first = create_mask_cutline_visualization(mask, result)
    second = create_mask_cutline_visualization(mask, result)
    first_path = tmp_path / "first.png"
    second_path = tmp_path / "second.png"
    first.save(first_path, format="PNG")
    second.save(second_path, format="PNG")

    height, width = mask.shape
    assert first.size == (
        3 * width,
        2 * (height + TITLE_HEIGHT) + STATUS_HEIGHT,
    )
    assert first.mode == "RGB"
    assert first_path.read_bytes() == second_path.read_bytes()
    np.testing.assert_array_equal(mask, original)


def test_visualization_rejects_mismatched_result_dimensions() -> None:
    mask = _vertical_attachment_mask()
    result = estimate_visible_mask_cutline(mask)
    mismatched = replace(result, image_width=result.image_width + 1)

    with pytest.raises(ValueError, match="result dimensions"):
        create_mask_cutline_visualization(mask, mismatched)
