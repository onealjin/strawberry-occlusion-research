"""Headless diagnostics for visible-mask cutline geometry."""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from strawberry_occlusion.geometry import LineSegment, MaskCutlineResult, Point
from strawberry_occlusion.visualization.segmentation import colorize_mask


TITLE_HEIGHT = 24
STATUS_HEIGHT = 52
PANEL_TITLES = (
    "RGB / colour mask",
    "Flesh mask",
    "Calyx mask",
    "Contact band",
    "Candidate geometry",
    "Final offset cutline",
)
CONTACT_COLOR = (255, 0, 255)
FLESH_CENTROID_COLOR = (0, 128, 255)
ATTACHMENT_ANCHOR_COLOR = (255, 255, 0)
DIRECTION_AXIS_COLOR = (255, 128, 0)
CANDIDATE_CUTLINE_COLOR = (0, 255, 255)
FINAL_CUTLINE_COLOR = (255, 255, 255)
OFFSET_ANCHOR_COLOR = (255, 0, 255)


def create_mask_cutline_visualization(
    mask: np.ndarray,
    result: MaskCutlineResult,
    *,
    image: Image.Image | np.ndarray | None = None,
) -> Image.Image:
    """Render six deterministic panels and a compact status footer."""

    mask_array = np.asarray(mask)
    colour_mask = colorize_mask(mask_array)
    height, width = mask_array.shape
    if (result.image_height, result.image_width) != (height, width):
        raise ValueError("result dimensions must match the supplied mask")
    input_panel = (
        colour_mask if image is None else _rgb_array(image, shape=mask_array.shape)
    )
    flesh_panel = _binary_panel(result.flesh_mask, color=(255, 0, 0))
    calyx_panel = _binary_panel(result.calyx_mask, color=(0, 255, 0))
    contact_panel = _binary_panel(result.contact_band, color=CONTACT_COLOR)
    candidate_panel = _geometry_panel(
        colour_mask,
        result,
        include_candidate=True,
        include_final=False,
    )
    final_panel = _geometry_panel(
        input_panel,
        result,
        include_candidate=False,
        include_final=True,
    )
    panels = (
        input_panel,
        flesh_panel,
        calyx_panel,
        contact_panel,
        candidate_panel,
        final_panel,
    )

    canvas = Image.new(
        "RGB",
        (3 * width, 2 * (height + TITLE_HEIGHT) + STATUS_HEIGHT),
        color=(0, 0, 0),
    )
    for index, (title, panel_array) in enumerate(
        zip(PANEL_TITLES, panels, strict=True)
    ):
        panel = Image.new("RGB", (width, height + TITLE_HEIGHT), color=(0, 0, 0))
        ImageDraw.Draw(panel).text((4, 5), title, fill=(255, 255, 255))
        panel.paste(Image.fromarray(panel_array), (0, TITLE_HEIGHT))
        column = index % 3
        row = index // 3
        canvas.paste(panel, (column * width, row * (height + TITLE_HEIGHT)))

    footer_top = 2 * (height + TITLE_HEIGHT)
    _draw_status_footer(canvas, result, top=footer_top)
    return canvas


def _geometry_panel(
    base: np.ndarray,
    result: MaskCutlineResult,
    *,
    include_candidate: bool,
    include_final: bool,
) -> np.ndarray:
    panel = Image.fromarray(np.array(base, dtype=np.uint8, copy=True))
    draw = ImageDraw.Draw(panel)
    line_width = max(1, min(result.image_width, result.image_height) // 200)
    marker_radius = max(1, min(result.image_width, result.image_height) // 100)

    if result.flesh_centroid is not None and result.attachment_anchor is not None:
        _draw_segment(
            draw,
            LineSegment(result.flesh_centroid, result.attachment_anchor),
            fill=DIRECTION_AXIS_COLOR,
            width=line_width,
        )
        _draw_marker(
            draw,
            result.flesh_centroid,
            color=FLESH_CENTROID_COLOR,
            radius=marker_radius,
        )
        _draw_marker(
            draw,
            result.attachment_anchor,
            color=ATTACHMENT_ANCHOR_COLOR,
            radius=marker_radius,
        )
    if include_candidate and result.candidate_cutline is not None:
        _draw_segment(
            draw,
            result.candidate_cutline,
            fill=CANDIDATE_CUTLINE_COLOR,
            width=line_width,
        )
    if include_final and result.final_cutline is not None:
        _draw_segment(
            draw,
            result.final_cutline,
            fill=FINAL_CUTLINE_COLOR,
            width=max(1, line_width + 1),
        )
    if include_final and result.offset_anchor is not None:
        _draw_marker(
            draw,
            result.offset_anchor,
            color=OFFSET_ANCHOR_COLOR,
            radius=marker_radius,
        )
    return np.asarray(panel, dtype=np.uint8)


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
    result: MaskCutlineResult,
    *,
    top: int,
) -> None:
    draw = ImageDraw.Draw(image)
    reason = result.failure_reason or "cutline estimated"
    summary = (
        f"status={result.status} | {reason} | "
        f"dilation={result.parameters.calyx_dilation_radius} | "
        f"offset={result.parameters.signed_offset:g} | "
        f"components F/C={result.flesh_component_count}/"
        f"{result.calyx_component_count} | "
        f"contact pixels/components={result.contact_pixel_count}/"
        f"{result.contact_component_count}"
    )
    draw.text((5, top + 4), summary, fill=(255, 255, 255))
    legend = (
        ("centroid", FLESH_CENTROID_COLOR),
        ("anchor", ATTACHMENT_ANCHOR_COLOR),
        ("axis", DIRECTION_AXIS_COLOR),
        ("candidate", CANDIDATE_CUTLINE_COLOR),
        ("final", FINAL_CUTLINE_COLOR),
    )
    x = 5
    for label, color in legend:
        draw.rectangle((x, top + 28, x + 10, top + 38), fill=color)
        draw.text((x + 14, top + 26), label, fill=(255, 255, 255))
        x += 76


def _binary_panel(mask: np.ndarray, *, color: tuple[int, int, int]) -> np.ndarray:
    panel = np.zeros((*mask.shape, 3), dtype=np.uint8)
    panel[mask] = color
    return panel


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
