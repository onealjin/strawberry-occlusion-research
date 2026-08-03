"""Headless diagnostics for experimental v2a fixed-axis cutlines."""

from __future__ import annotations

from math import ceil

import numpy as np
from PIL import Image, ImageDraw

from strawberry_occlusion.geometry import (
    FixedAxisCutlineResult,
    LineSegment,
    Point,
)


TITLE_HEIGHT = 24
STATUS_HEIGHT = 104
MIN_PANEL_WIDTH = 240
PANEL_TITLES = (
    "Selected Flesh / Calyx",
    "Complete selected contact",
    "Robust outer-contact support",
    "v1 angle + configured axis",
    "v2a fixed candidate line",
    "v2a fixed final line",
)
FLESH_COLOR = (150, 24, 24)
CALYX_COLOR = (30, 180, 60)
CONTACT_COLOR = (255, 0, 255)
SUPPORT_COLOR = (255, 215, 0)
FLESH_CENTROID_COLOR = (0, 160, 255)
BASELINE_ANCHOR_COLOR = (255, 255, 255)
REMOVAL_AXIS_COLOR = (255, 128, 0)
BASELINE_LINE_COLOR = (128, 220, 255)
CANDIDATE_LINE_COLOR = (0, 255, 255)
FINAL_LINE_COLOR = (255, 255, 255)
REFERENCE_POINT_COLOR = (255, 64, 64)


def create_fixed_axis_cutline_visualization(
    mask: np.ndarray,
    result: FixedAxisCutlineResult,
    *,
    image: Image.Image | np.ndarray | None = None,
) -> Image.Image:
    """Render deterministic v2a geometry, support, and scalar diagnostics."""

    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(f"mask must have shape [H, W], got {mask_array.shape}")
    height, width = mask_array.shape
    if (result.image_height, result.image_width) != (height, width):
        raise ValueError("result dimensions must match the supplied mask")

    selected_panel = _selected_components_panel(result)
    contact_panel = selected_panel.copy()
    contact_panel[result.contact_mask] = CONTACT_COLOR
    support_panel = contact_panel.copy()
    support_panel[result.support_mask] = SUPPORT_COLOR
    geometry_base = (
        selected_panel if image is None else _rgb_array(image, shape=mask_array.shape)
    )
    comparison_panel = _geometry_panel(
        geometry_base,
        result,
        show_baseline=True,
        show_candidate=False,
        show_final=False,
    )
    candidate_panel = _geometry_panel(
        geometry_base,
        result,
        show_baseline=True,
        show_candidate=True,
        show_final=False,
    )
    final_panel = _geometry_panel(
        geometry_base,
        result,
        show_baseline=True,
        show_candidate=True,
        show_final=True,
    )
    panels = (
        selected_panel,
        contact_panel,
        support_panel,
        comparison_panel,
        candidate_panel,
        final_panel,
    )

    scale = max(1, ceil(MIN_PANEL_WIDTH / width))
    display_width = width * scale
    display_height = height * scale
    canvas = Image.new(
        "RGB",
        (
            3 * display_width,
            2 * (display_height + TITLE_HEIGHT) + STATUS_HEIGHT,
        ),
        color=(0, 0, 0),
    )
    for index, (title, panel_array) in enumerate(
        zip(PANEL_TITLES, panels, strict=True)
    ):
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

    _draw_status_footer(
        canvas,
        result,
        top=2 * (display_height + TITLE_HEIGHT),
    )
    return canvas


def _selected_components_panel(result: FixedAxisCutlineResult) -> np.ndarray:
    panel = np.zeros(
        (result.image_height, result.image_width, 3),
        dtype=np.uint8,
    )
    panel[result.selected_flesh_mask] = FLESH_COLOR
    panel[result.selected_calyx_mask] = CALYX_COLOR
    return panel


def _geometry_panel(
    base: np.ndarray,
    result: FixedAxisCutlineResult,
    *,
    show_baseline: bool,
    show_candidate: bool,
    show_final: bool,
) -> np.ndarray:
    panel = Image.fromarray(np.array(base, dtype=np.uint8, copy=True))
    draw = ImageDraw.Draw(panel)
    shortest_side = min(result.image_width, result.image_height)
    line_width = max(1, shortest_side // 160)
    marker_radius = max(1, shortest_side // 80)

    if show_baseline and result.baseline_v1_cutline is not None:
        _draw_segment(
            draw,
            result.baseline_v1_cutline,
            fill=BASELINE_LINE_COLOR,
            width=line_width,
        )
    if result.flesh_centroid is not None:
        _draw_marker(
            draw,
            result.flesh_centroid,
            color=FLESH_CENTROID_COLOR,
            radius=marker_radius,
        )
        _draw_removal_axis(
            draw,
            result.flesh_centroid,
            result.normalized_removal_axis,
            length=max(4.0, shortest_side * 0.3),
            width=line_width,
            marker_radius=marker_radius,
        )
    if result.baseline_v1_contact_anchor is not None:
        _draw_marker(
            draw,
            result.baseline_v1_contact_anchor,
            color=BASELINE_ANCHOR_COLOR,
            radius=marker_radius,
        )
    if show_candidate and result.candidate_cutline is not None:
        _draw_segment(
            draw,
            result.candidate_cutline,
            fill=CANDIDATE_LINE_COLOR,
            width=max(line_width, 2 if show_final else 1),
        )
    if show_final and result.final_cutline is not None:
        _draw_segment(
            draw,
            result.final_cutline,
            fill=FINAL_LINE_COLOR,
            width=line_width,
        )
    if show_final and result.diagnostic_reference_point is not None:
        _draw_marker(
            draw,
            result.diagnostic_reference_point,
            color=REFERENCE_POINT_COLOR,
            radius=marker_radius,
        )
    return np.asarray(panel, dtype=np.uint8)


def _draw_removal_axis(
    draw: ImageDraw.ImageDraw,
    start: Point,
    axis: tuple[float, float],
    *,
    length: float,
    width: int,
    marker_radius: int,
) -> None:
    end = Point(
        x=start.x + length * axis[0],
        y=start.y + length * axis[1],
    )
    _draw_segment(
        draw,
        LineSegment(start=start, end=end),
        fill=REMOVAL_AXIS_COLOR,
        width=width,
    )
    _draw_marker(
        draw,
        end,
        color=REMOVAL_AXIS_COLOR,
        radius=marker_radius,
    )


def _draw_segment(
    draw: ImageDraw.ImageDraw,
    segment: LineSegment,
    *,
    fill: tuple[int, int, int],
    width: int,
) -> None:
    draw.line(
        (
            (segment.start.x, segment.start.y),
            (segment.end.x, segment.end.y),
        ),
        fill=fill,
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


def _draw_status_footer(
    image: Image.Image,
    result: FixedAxisCutlineResult,
    *,
    top: int,
) -> None:
    draw = ImageDraw.Draw(image)
    reason = result.failure_reason or (
        "finite geometry produced; geometric quality not assessed"
    )
    lines = (
        (
            f"v2a status={result.status} | {reason} | axis="
            f"({_format_number(result.normalized_removal_axis[0])}, "
            f"{_format_number(result.normalized_removal_axis[1])})"
        ),
        (
            f"projection quantile={result.parameters.projection_quantile:g} | "
            f"unshifted coordinate={_format_number(result.unshifted_cut_coordinate)} | "
            f"final coordinate={_format_number(result.final_cut_coordinate)} | "
            f"signed offset={result.signed_offset_pixels:g} px"
        ),
        (
            f"{_coincidence_text(result)} | axis disagreement="
            f"{_format_number(result.axis_disagreement_angle_degrees)} deg | "
            f"cut margin={_format_number(result.cut_margin_to_flesh_extent)} px"
        ),
        (
            f"complete contact={result.contact_pixel_count} px | outer support="
            f"{result.support_pixel_count} px, lateral extent="
            f"{_format_number(result.support_lateral_extent_pixels)} px | "
            "margin > 0 means inside selected Flesh extent; no quality classification"
        ),
    )
    for index, line in enumerate(lines):
        draw.text((5, top + 5 + 22 * index), line, fill=(255, 255, 255))


def _coincidence_text(result: FixedAxisCutlineResult) -> str:
    if (
        result.signed_offset_pixels == 0.0
        and result.candidate_cutline is not None
        and result.final_cutline is not None
    ):
        return "candidate/final lines coincide=yes (signed offset is zero)"
    value = result.candidate_final_lines_coincide
    rendered = "n/a" if value is None else ("yes" if value else "no")
    return f"candidate/final lines coincide={rendered}"


def _format_number(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.4g}"


def _rgb_array(
    image: Image.Image | np.ndarray,
    *,
    shape: tuple[int, int],
) -> np.ndarray:
    if isinstance(image, Image.Image):
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
    else:
        array = np.asarray(image)
        if array.ndim != 3 or array.shape[2] != 3:
            raise ValueError("image must have RGB shape [H, W, 3]")
        if not np.issubdtype(array.dtype, np.integer):
            raise TypeError("image must use an integer dtype")
        if np.any(array < 0) or np.any(array > 255):
            raise ValueError("image values must be in [0, 255]")
        array = array.astype(np.uint8, copy=True)
    if array.shape[:2] != shape:
        raise ValueError("image dimensions must match the mask")
    return array
