"""Headless v1-versus-v2a-versus-v2b comparison visualization."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil, hypot
from typing import Any

import numpy as np
from PIL import Image, ImageDraw

from strawberry_occlusion.evaluation.cutline import flesh_boundary_mask
from strawberry_occlusion.geometry import (
    FixedAxisCutlineResult,
    FixedAxisSearchCutlineResult,
    LineSegment,
    MaskCutlineResult,
    Point,
)
from strawberry_occlusion.visualization.fixed_axis_cutline import _rgb_array


TITLE_HEIGHT = 24
MIN_PANEL_WIDTH = 420
MIN_PANEL_HEIGHT = 280
PANEL_TITLES = (
    "Source RGB",
    "Semantic mask + Flesh boundary",
    "Complete selected contact",
    "v2a robust support",
    "Baseline v1 geometry",
    "Fixed-axis v2a geometry",
    "Fixed-axis v2b search geometry",
    "Combined RGB overlay",
    "Candidate curve, metrics, and status",
)
FLESH_COLOR = (150, 24, 24)
CALYX_COLOR = (30, 180, 60)
FLESH_BOUNDARY_COLOR = (255, 215, 0)
CONTACT_COLOR = (255, 0, 255)
SUPPORT_COLOR = (170, 80, 255)
FLESH_CENTROID_COLOR = (255, 80, 80)
V1_DIRECTION_COLOR = (255, 140, 0)
V1_LINE_COLOR = (255, 0, 180)
V1_ANCHOR_COLOR = (255, 105, 180)
V2A_AXIS_COLOR = (40, 120, 255)
V2A_CANDIDATE_COLOR = (0, 255, 255)
V2A_FINAL_COLOR = (255, 255, 255)
V2A_REFERENCE_COLOR = (255, 255, 0)
V2A_FLESH_REFERENCE_COLOR = (80, 255, 140)
V2A_ATTACHMENT_REFERENCE_COLOR = (255, 190, 0)
V2B_LINE_COLOR = (0, 255, 100)
V2B_REFERENCE_COLOR = (0, 220, 255)
V2B_WINDOW_COLOR = (35, 100, 65)


def create_fixed_axis_comparison_visualization(
    mask: np.ndarray,
    v1_result: MaskCutlineResult,
    v2a_result: FixedAxisCutlineResult,
    v2b_result: FixedAxisSearchCutlineResult,
    *,
    image: Image.Image | np.ndarray | None = None,
    metrics: Mapping[str, Any] | None = None,
) -> Image.Image:
    """Render nine deterministic panels in source-image coordinates."""

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"mask must have shape [H, W], got {mask_array.shape}")
    height, width = mask_array.shape
    expected_shape = (height, width)
    if (v1_result.image_height, v1_result.image_width) != expected_shape:
        raise ValueError("v1 result dimensions must match the supplied mask")
    if (v2a_result.image_height, v2a_result.image_width) != expected_shape:
        raise ValueError("v2a result dimensions must match the supplied mask")
    if (v2b_result.image_height, v2b_result.image_width) != expected_shape:
        raise ValueError("v2b result dimensions must match the supplied mask")

    semantic = _semantic_panel(v1_result)
    source = (
        semantic.copy() if image is None else _rgb_array(image, shape=expected_shape)
    )
    contact = semantic.copy()
    contact[v1_result.contact_band] = CONTACT_COLOR
    support = contact.copy()
    support[v2a_result.support_mask] = SUPPORT_COLOR
    v1_geometry = _v1_geometry_panel(source, v1_result)
    v2a_geometry = _v2a_geometry_panel(source, v2a_result)
    v2b_geometry = _v2b_geometry_panel(source, v2b_result)
    combined = _combined_geometry_panel(
        source,
        v1_result,
        v2a_result,
        v2b_result,
    )
    image_panels = (
        source,
        semantic,
        contact,
        support,
        v1_geometry,
        v2a_geometry,
        v2b_geometry,
        combined,
    )

    scale = max(
        1,
        ceil(MIN_PANEL_WIDTH / width),
        ceil(MIN_PANEL_HEIGHT / height),
    )
    display_width = width * scale
    display_height = height * scale
    canvas = Image.new(
        "RGB",
        (3 * display_width, 3 * (display_height + TITLE_HEIGHT)),
        color=(0, 0, 0),
    )
    for index, panel_array in enumerate(image_panels):
        _paste_image_panel(
            canvas,
            panel_array,
            title=PANEL_TITLES[index],
            index=index,
            display_width=display_width,
            display_height=display_height,
        )
    metric_panel = _metric_panel(
        v1_result,
        v2a_result,
        v2b_result,
        metrics=metrics or {},
        width=display_width,
        height=display_height + TITLE_HEIGHT,
    )
    canvas.paste(
        metric_panel,
        (2 * display_width, 2 * (display_height + TITLE_HEIGHT)),
    )
    return canvas


def _semantic_panel(result: MaskCutlineResult) -> np.ndarray:
    panel = np.zeros(
        (result.image_height, result.image_width, 3),
        dtype=np.uint8,
    )
    panel[result.flesh_mask] = FLESH_COLOR
    panel[result.calyx_mask] = CALYX_COLOR
    panel[flesh_boundary_mask(result.flesh_mask)] = FLESH_BOUNDARY_COLOR
    return panel


def _v1_geometry_panel(
    base: np.ndarray,
    result: MaskCutlineResult,
) -> np.ndarray:
    panel = Image.fromarray(np.array(base, dtype=np.uint8, copy=True))
    draw = ImageDraw.Draw(panel)
    line_width, marker_radius = _drawing_sizes(result.image_width, result.image_height)
    if result.flesh_centroid is not None:
        _draw_marker(
            draw,
            result.flesh_centroid,
            color=FLESH_CENTROID_COLOR,
            radius=marker_radius,
        )
    if result.flesh_centroid is not None and result.attachment_anchor is not None:
        _draw_segment(
            draw,
            LineSegment(result.flesh_centroid, result.attachment_anchor),
            color=V1_DIRECTION_COLOR,
            width=line_width,
        )
    if result.attachment_anchor is not None:
        _draw_marker(
            draw,
            result.attachment_anchor,
            color=V1_ANCHOR_COLOR,
            radius=marker_radius,
        )
    if result.final_cutline is not None:
        _draw_dashed_segment(
            draw,
            result.final_cutline,
            color=V1_LINE_COLOR,
            width=max(1, line_width + 1),
        )
    return np.asarray(panel, dtype=np.uint8)


def _v2a_geometry_panel(
    base: np.ndarray,
    result: FixedAxisCutlineResult,
) -> np.ndarray:
    panel_array = np.array(base, dtype=np.uint8, copy=True)
    panel_array[result.support_mask] = SUPPORT_COLOR
    panel = Image.fromarray(panel_array)
    draw = ImageDraw.Draw(panel)
    line_width, marker_radius = _drawing_sizes(result.image_width, result.image_height)
    axis_origin = result.flesh_centroid or result.diagnostic_reference_point
    if axis_origin is not None:
        _draw_axis_arrow(
            draw,
            axis_origin,
            result.normalized_removal_axis,
            length=max(5.0, min(result.image_width, result.image_height) * 0.3),
            color=V2A_AXIS_COLOR,
            width=line_width,
            marker_radius=marker_radius,
        )
    if result.candidate_cutline is not None:
        _draw_segment(
            draw,
            result.candidate_cutline,
            color=V2A_CANDIDATE_COLOR,
            width=max(3, line_width + 2),
        )
    if result.final_cutline is not None:
        _draw_segment(
            draw,
            result.final_cutline,
            color=V2A_FINAL_COLOR,
            width=max(1, line_width),
        )
    if result.diagnostic_reference_point is not None:
        _draw_marker(
            draw,
            result.diagnostic_reference_point,
            color=V2A_REFERENCE_COLOR,
            radius=marker_radius,
        )
    if result.flesh_centroid_projected_reference_point is not None:
        _draw_ring(
            draw,
            result.flesh_centroid_projected_reference_point,
            color=V2A_FLESH_REFERENCE_COLOR,
            radius=max(2, marker_radius + 2),
        )
    if result.attachment_anchor_projected_reference_point is not None:
        _draw_ring(
            draw,
            result.attachment_anchor_projected_reference_point,
            color=V2A_ATTACHMENT_REFERENCE_COLOR,
            radius=max(3, marker_radius + 4),
        )
    return np.asarray(panel, dtype=np.uint8)


def _v2b_geometry_panel(
    base: np.ndarray,
    result: FixedAxisSearchCutlineResult,
) -> np.ndarray:
    panel_array = np.array(base, dtype=np.uint8, copy=True)
    panel_array[result.selected_local_window_mask] = V2B_WINDOW_COLOR
    panel_array[result.selected_local_flesh_mask] = FLESH_COLOR
    panel_array[result.selected_local_calyx_mask] = CALYX_COLOR
    panel_array[result.selected_local_contact_mask] = CONTACT_COLOR
    panel = Image.fromarray(panel_array)
    draw = ImageDraw.Draw(panel)
    line_width, marker_radius = _drawing_sizes(result.image_width, result.image_height)
    if result.final_cutline is not None:
        _draw_segment(
            draw,
            result.final_cutline,
            color=V2B_LINE_COLOR,
            width=max(2, line_width + 1),
        )
    if result.diagnostic_reference_point is not None:
        _draw_marker(
            draw,
            result.diagnostic_reference_point,
            color=V2B_REFERENCE_COLOR,
            radius=max(1, marker_radius + 1),
        )
        _draw_axis_arrow(
            draw,
            result.diagnostic_reference_point,
            result.normalized_removal_axis,
            length=max(5.0, min(result.image_width, result.image_height) * 0.3),
            color=V2A_AXIS_COLOR,
            width=line_width,
            marker_radius=marker_radius,
        )
    return np.asarray(panel, dtype=np.uint8)


def _combined_geometry_panel(
    base: np.ndarray,
    v1_result: MaskCutlineResult,
    v2a_result: FixedAxisCutlineResult,
    v2b_result: FixedAxisSearchCutlineResult,
) -> np.ndarray:
    panel = Image.fromarray(np.array(base, dtype=np.uint8, copy=True))
    draw = ImageDraw.Draw(panel)
    line_width, marker_radius = _drawing_sizes(
        v1_result.image_width,
        v1_result.image_height,
    )
    if v1_result.final_cutline is not None:
        _draw_dashed_segment(
            draw,
            v1_result.final_cutline,
            color=V1_LINE_COLOR,
            width=max(1, line_width + 1),
        )
    if v2a_result.candidate_cutline is not None:
        _draw_segment(
            draw,
            v2a_result.candidate_cutline,
            color=V2A_CANDIDATE_COLOR,
            width=max(3, line_width + 2),
        )
    if v2a_result.final_cutline is not None:
        _draw_segment(
            draw,
            v2a_result.final_cutline,
            color=V2A_FINAL_COLOR,
            width=max(1, line_width),
        )
    if v2b_result.final_cutline is not None:
        _draw_segment(
            draw,
            v2b_result.final_cutline,
            color=V2B_LINE_COLOR,
            width=max(2, line_width + 1),
        )
    if v1_result.attachment_anchor is not None:
        _draw_marker(
            draw,
            v1_result.attachment_anchor,
            color=V1_ANCHOR_COLOR,
            radius=marker_radius,
        )
    if v2a_result.diagnostic_reference_point is not None:
        _draw_marker(
            draw,
            v2a_result.diagnostic_reference_point,
            color=V2A_REFERENCE_COLOR,
            radius=marker_radius,
        )
    if v2b_result.diagnostic_reference_point is not None:
        _draw_marker(
            draw,
            v2b_result.diagnostic_reference_point,
            color=V2B_REFERENCE_COLOR,
            radius=max(1, marker_radius + 1),
        )
    return np.asarray(panel, dtype=np.uint8)


def _paste_image_panel(
    canvas: Image.Image,
    panel_array: np.ndarray,
    *,
    title: str,
    index: int,
    display_width: int,
    display_height: int,
) -> None:
    panel = Image.new(
        "RGB",
        (display_width, display_height + TITLE_HEIGHT),
        color=(0, 0, 0),
    )
    ImageDraw.Draw(panel).text((5, 5), title, fill=(255, 255, 255))
    panel_image = Image.fromarray(panel_array).resize(
        (display_width, display_height),
        resample=Image.Resampling.NEAREST,
    )
    panel.paste(panel_image, (0, TITLE_HEIGHT))
    column = index % 3
    row = index // 3
    canvas.paste(
        panel,
        (column * display_width, row * (display_height + TITLE_HEIGHT)),
    )


def _metric_panel(
    v1_result: MaskCutlineResult,
    v2a_result: FixedAxisCutlineResult,
    v2b_result: FixedAxisSearchCutlineResult,
    *,
    metrics: Mapping[str, Any],
    width: int,
    height: int,
) -> Image.Image:
    panel = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(panel)
    draw.text((5, 5), PANEL_TITLES[-1], fill=(255, 255, 255))
    plot_bottom = _draw_candidate_curve(
        draw,
        v2b_result,
        left=7,
        top=TITLE_HEIGHT + 3,
        width=width - 14,
        height=max(55, min(78, height // 3)),
    )
    v1_reason = v1_result.failure_reason or "finite geometry; quality not assessed"
    v2a_reason = v2a_result.failure_reason or "finite geometry; quality not assessed"
    v2b_reason = (
        v2b_result.failure_reason or "finite proxy geometry; quality not assessed"
    )
    lines = (
        f"v1={v1_result.status}: {_truncate(v1_reason)}",
        f"v2a={v2a_result.status}: {_truncate(v2a_reason)}",
        f"v2b={v2b_result.status}: {_truncate(v2b_reason)}",
        (
            "whole-mask Flesh loss v1/v2a/v2b="
            f"{_format_number(metrics.get('v1_flesh_loss_ratio'))}/"
            f"{_format_number(metrics.get('v2a_flesh_loss_ratio'))}/"
            f"{_format_number(metrics.get('v2b_whole_mask_flesh_loss_ratio'))}"
        ),
        (
            "whole-mask Calyx retained v1/v2a/v2b="
            f"{_format_number(metrics.get('v1_calyx_retention_ratio'))}/"
            f"{_format_number(metrics.get('v2a_calyx_retention_ratio'))}/"
            f"{_format_number(metrics.get('v2b_whole_mask_calyx_retention_ratio'))}"
        ),
        (
            "coordinates v1 projection/v2a/v2b="
            f"{_format_number(metrics.get('v1_anchor_projection_on_removal_axis'))}/"
            f"{_format_number(metrics.get('v2a_final_cut_coordinate'))}/"
            f"{_format_number(metrics.get('v2b_cut_coordinate'))} px"
        ),
        (
            "v2b local contact/max="
            f"{v2b_result.selected_local_contact_count}/"
            f"{v2b_result.maximum_local_contact_evidence} | margin="
            f"{_format_number(metrics.get('v2b_cut_margin_to_flesh_extent'))} px"
        ),
        (
            "v2a axis disagreement="
            f"{_format_number(metrics.get('v2a_axis_disagreement_degrees'))} deg | "
            f"q={v2a_result.parameters.projection_quantile:g}"
        ),
        _coincidence_text(v2a_result),
        "v2a/v2b fixed orientation; diagnostic point placement is not the scalar coordinate.",
        "status=ok means finite proxy geometry, not proven cut quality.",
    )
    for index, line in enumerate(lines):
        draw.text((6, plot_bottom + 4 + 14 * index), line, fill=(255, 255, 255))
    _draw_legend(
        draw,
        top=plot_bottom + 8 + 14 * len(lines),
        width=width,
    )
    return panel


def _draw_legend(
    draw: ImageDraw.ImageDraw,
    *,
    top: int,
    width: int,
) -> None:
    items = (
        ("v1 line", V1_LINE_COLOR),
        ("v2a line", V2A_FINAL_COLOR),
        ("v2b line", V2B_LINE_COLOR),
        ("support ref", V2A_REFERENCE_COLOR),
        ("Flesh ref", V2A_FLESH_REFERENCE_COLOR),
        ("anchor ref", V2A_ATTACHMENT_REFERENCE_COLOR),
        ("v2b ref", V2B_REFERENCE_COLOR),
        ("fixed axis", V2A_AXIS_COLOR),
    )
    column_width = width // 4
    for index, (label, color) in enumerate(items):
        column = index % 4
        row = index // 4
        x = 5 + column * column_width
        y = top + row * 14
        draw.rectangle((x, y + 2, x + 7, y + 9), fill=color)
        draw.text((x + 10, y), label, fill=(235, 235, 235))


def _draw_candidate_curve(
    draw: ImageDraw.ImageDraw,
    result: FixedAxisSearchCutlineResult,
    *,
    left: int,
    top: int,
    width: int,
    height: int,
) -> int:
    """Draw compact local-contact evidence; returns the panel y after it."""

    right = left + width
    bottom = top + height
    draw.rectangle((left, top, right, bottom), outline=(110, 110, 110))
    table = result.candidate_table
    if not len(table):
        draw.text(
            (left + 4, top + 4), "candidate curve unavailable", fill=(220, 220, 220)
        )
        return bottom
    coordinates = table.coordinate
    evidence = table.local_contact_count
    minimum = float(coordinates.min())
    maximum = float(coordinates.max())
    maximum_evidence = max(1, int(evidence.max()))

    def map_x(value: float) -> float:
        if maximum == minimum:
            return left + width / 2
        return left + (value - minimum) / (maximum - minimum) * width

    points = [
        (
            map_x(float(coordinate)),
            bottom - float(count) / maximum_evidence * (height - 10),
        )
        for coordinate, count in zip(coordinates, evidence, strict=True)
    ]
    if len(points) > 1:
        draw.line(points, fill=CONTACT_COLOR, width=2)
    for index, point in enumerate(points):
        color = V2B_LINE_COLOR if bool(table.feasible[index]) else CONTACT_COLOR
        draw.ellipse(
            (point[0] - 2, point[1] - 2, point[0] + 2, point[1] + 2),
            fill=color,
        )
    markers = (
        (result.v1_anchor_projection, V1_LINE_COLOR),
        (result.v2a_cut_coordinate, V2A_FINAL_COLOR),
        (result.selected_cut_coordinate, V2B_LINE_COLOR),
    )
    for coordinate, color in markers:
        if coordinate is not None and minimum <= coordinate <= maximum:
            x = map_x(float(coordinate))
            draw.line((x, top, x, bottom), fill=color, width=1)
    draw.text(
        (left + 4, top + 2),
        "contact evidence; feasible=green | v1=magenta v2a=white v2b=green",
        fill=(230, 230, 230),
    )
    return bottom


def _coincidence_text(result: FixedAxisCutlineResult) -> str:
    if (
        result.signed_offset_pixels == 0.0
        and result.candidate_cutline is not None
        and result.final_cutline is not None
    ):
        return "v2a candidate and final lines coincide (signed offset is zero)"
    value = result.candidate_final_lines_coincide
    rendered = "n/a" if value is None else ("yes" if value else "no")
    return f"v2a candidate/final lines coincide={rendered}"


def _drawing_sizes(width: int, height: int) -> tuple[int, int]:
    shortest_side = min(width, height)
    return max(1, shortest_side // 160), max(1, shortest_side // 80)


def _draw_axis_arrow(
    draw: ImageDraw.ImageDraw,
    start: Point,
    axis: tuple[float, float],
    *,
    length: float,
    color: tuple[int, int, int],
    width: int,
    marker_radius: int,
) -> None:
    end = Point(x=start.x + length * axis[0], y=start.y + length * axis[1])
    _draw_segment(
        draw,
        LineSegment(start=start, end=end),
        color=color,
        width=width,
    )
    arrow_length = max(2.0, marker_radius * 2.5)
    arrow_width = max(1.5, marker_radius * 1.5)
    perpendicular = (-axis[1], axis[0])
    base_x = end.x - arrow_length * axis[0]
    base_y = end.y - arrow_length * axis[1]
    draw.polygon(
        (
            (end.x, end.y),
            (
                base_x + arrow_width * perpendicular[0],
                base_y + arrow_width * perpendicular[1],
            ),
            (
                base_x - arrow_width * perpendicular[0],
                base_y - arrow_width * perpendicular[1],
            ),
        ),
        fill=color,
    )


def _draw_dashed_segment(
    draw: ImageDraw.ImageDraw,
    segment: LineSegment,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    delta_x = segment.end.x - segment.start.x
    delta_y = segment.end.y - segment.start.y
    length = hypot(delta_x, delta_y)
    if length == 0.0:
        return
    dash_length = max(2.0, length / 16.0)
    position = 0.0
    while position < length:
        dash_end = min(position + dash_length, length)
        start_fraction = position / length
        end_fraction = dash_end / length
        draw.line(
            (
                (
                    segment.start.x + start_fraction * delta_x,
                    segment.start.y + start_fraction * delta_y,
                ),
                (
                    segment.start.x + end_fraction * delta_x,
                    segment.start.y + end_fraction * delta_y,
                ),
            ),
            fill=color,
            width=width,
        )
        position += 2.0 * dash_length


def _draw_segment(
    draw: ImageDraw.ImageDraw,
    segment: LineSegment,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    draw.line(
        (
            (segment.start.x, segment.start.y),
            (segment.end.x, segment.end.y),
        ),
        fill=color,
        width=width,
    )


def _draw_marker(
    draw: ImageDraw.ImageDraw,
    point: Point,
    *,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    draw.ellipse(
        (
            point.x - radius,
            point.y - radius,
            point.x + radius,
            point.y + radius,
        ),
        fill=color,
    )


def _draw_ring(
    draw: ImageDraw.ImageDraw,
    point: Point,
    *,
    color: tuple[int, int, int],
    radius: int,
) -> None:
    draw.ellipse(
        (
            point.x - radius,
            point.y - radius,
            point.x + radius,
            point.y + radius,
        ),
        outline=color,
        width=1,
    )


def _format_number(value: Any) -> str:
    return "n/a" if value is None else f"{float(value):.4g}"


def _truncate(value: str, *, limit: int = 54) -> str:
    return value if len(value) <= limit else value[: limit - 3] + "..."
