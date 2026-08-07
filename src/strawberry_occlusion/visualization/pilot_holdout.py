"""Headless diagnostics for the seven-image pilot holdout workflow."""

from __future__ import annotations

from collections.abc import Mapping
from math import ceil
from textwrap import wrap
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from strawberry_occlusion.evaluation.segmentation import (
    ERROR_PALETTE,
    diagnostic_masks,
)
from strawberry_occlusion.geometry import (
    FixedAxisCutlineResult,
    FixedAxisSearchCutlineResult,
    LineSegment,
)
from strawberry_occlusion.visualization.fixed_axis_comparison import (
    V2A_FINAL_COLOR,
    V2B_LINE_COLOR,
)
from strawberry_occlusion.visualization.segmentation import (
    PALETTE,
    colorize_mask,
    overlay_mask,
)


TITLE_HEIGHT = 24
MIN_PANEL_WIDTH = 240
MIN_PANEL_HEIGHT = 160
PANEL_TITLES = (
    "Evaluation RGB",
    "Ground-truth semantic mask",
    "Predicted semantic mask",
    "Ground-truth/prediction errors",
    "Calyx false positives/negatives",
    "Normalized entropy",
    "Ground-truth-mask v2a/v2b geometry",
    "Predicted-mask v2a/v2b geometry",
    "Pilot metadata and metrics",
)
INELIGIBLE_MESSAGE = "segmentation-only: not eligible for current fixed-axis aggregate"


def create_pilot_holdout_visualization(
    image: Image.Image | np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    normalized_entropy: np.ndarray,
    *,
    sample_id: str,
    metadata: Mapping[str, str],
    segmentation_metrics: Mapping[str, Any],
    fixed_axis_eligible: bool,
    ground_truth_v2a: FixedAxisCutlineResult | None = None,
    ground_truth_v2b: FixedAxisSearchCutlineResult | None = None,
    predicted_v2a: FixedAxisCutlineResult | None = None,
    predicted_v2b: FixedAxisSearchCutlineResult | None = None,
) -> Image.Image:
    """Render one deterministic lossless nine-panel pilot diagnostic.

    The supplied image and masks must already use the checkpoint evaluation
    dimensions. Geometry is drawn in those same coordinates; this function never
    rotates, crops, or changes semantic class IDs.
    """

    image_array = _rgb_array(image)
    target_array = _class_mask(target, name="target")
    predicted_array = _class_mask(predicted, name="predicted")
    confidence_array = _unit_interval(confidence, name="confidence")
    entropy_array = _unit_interval(normalized_entropy, name="normalized_entropy")
    shape = target_array.shape
    if predicted_array.shape != shape:
        raise ValueError("target and predicted masks must have matching dimensions")
    if image_array.shape[:2] != shape:
        raise ValueError("image and mask dimensions must match")
    if confidence_array.shape != shape or entropy_array.shape != shape:
        raise ValueError("confidence and entropy dimensions must match the masks")

    errors, calyx_errors = _error_panels(
        image_array,
        target_array,
        predicted_array,
    )
    entropy_panel = _entropy_panel(entropy_array)
    if fixed_axis_eligible:
        _require_geometry(
            shape,
            ground_truth_v2a,
            ground_truth_v2b,
            predicted_v2a,
            predicted_v2b,
        )
        ground_truth_geometry = _geometry_panel(
            image_array,
            target_array,
            ground_truth_v2a,
            ground_truth_v2b,
        )
        predicted_geometry = _geometry_panel(
            image_array,
            predicted_array,
            predicted_v2a,
            predicted_v2b,
        )
    else:
        ground_truth_geometry = _message_panel(shape, INELIGIBLE_MESSAGE)
        predicted_geometry = _message_panel(shape, INELIGIBLE_MESSAGE)

    panels = (
        image_array,
        colorize_mask(target_array),
        colorize_mask(predicted_array),
        errors,
        calyx_errors,
        entropy_panel,
        ground_truth_geometry,
        predicted_geometry,
    )
    height, width = shape
    scale = max(
        1,
        ceil(MIN_PANEL_WIDTH / width),
        ceil(MIN_PANEL_HEIGHT / height),
    )
    display_width = width * scale
    display_height = height * scale
    panel_height = display_height + TITLE_HEIGHT
    canvas = Image.new(
        "RGB",
        (3 * display_width, 3 * panel_height),
        color=(0, 0, 0),
    )
    for index, panel_array in enumerate(panels):
        panel = _titled_image_panel(
            PANEL_TITLES[index],
            panel_array,
            display_width=display_width,
            display_height=display_height,
        )
        canvas.paste(panel, ((index % 3) * display_width, (index // 3) * panel_height))

    metric_panel = _metric_panel(
        width=display_width,
        height=panel_height,
        sample_id=sample_id,
        metadata=metadata,
        metrics=segmentation_metrics,
        fixed_axis_eligible=fixed_axis_eligible,
        ground_truth_v2a=ground_truth_v2a,
        ground_truth_v2b=ground_truth_v2b,
        predicted_v2a=predicted_v2a,
        predicted_v2b=predicted_v2b,
    )
    canvas.paste(metric_panel, (2 * display_width, 2 * panel_height))
    return canvas


def _error_panels(
    image: np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    masks = diagnostic_masks(
        torch.from_numpy(predicted.astype(np.int64, copy=False)),
        torch.from_numpy(target.astype(np.int64, copy=False)),
    )
    overlay = image.astype(np.float64, copy=True)
    for name in (
        "flesh_false_positive",
        "flesh_false_negative",
        "calyx_false_positive",
        "calyx_false_negative",
    ):
        selected = masks[name].numpy()
        color = np.asarray(ERROR_PALETTE[name], dtype=np.float64)
        overlay[selected] = 0.25 * overlay[selected] + 0.75 * color

    calyx = np.zeros_like(image)
    correct_calyx = (target == 2) & (predicted == 2)
    calyx[correct_calyx] = PALETTE[2]
    calyx[masks["calyx_false_positive"].numpy()] = ERROR_PALETTE["calyx_false_positive"]
    calyx[masks["calyx_false_negative"].numpy()] = ERROR_PALETTE["calyx_false_negative"]
    return np.rint(overlay).clip(0, 255).astype(np.uint8), calyx


def _entropy_panel(entropy: np.ndarray) -> np.ndarray:
    red = np.rint(entropy * 255.0).astype(np.uint8)
    blue = np.rint((1.0 - entropy) * 180.0).astype(np.uint8)
    green = np.rint((1.0 - np.abs(entropy - 0.5) * 2.0) * 120.0).astype(np.uint8)
    return np.stack((red, green, blue), axis=2)


def _require_geometry(
    shape: tuple[int, int],
    *results: FixedAxisCutlineResult | FixedAxisSearchCutlineResult | None,
) -> None:
    if any(result is None for result in results):
        raise ValueError("eligible samples require ground-truth and predicted geometry")
    for result in results:
        assert result is not None
        if (result.image_height, result.image_width) != shape:
            raise ValueError("geometry dimensions must match the evaluation masks")


def _geometry_panel(
    image: np.ndarray,
    mask: np.ndarray,
    v2a: FixedAxisCutlineResult | None,
    v2b: FixedAxisSearchCutlineResult | None,
) -> np.ndarray:
    if v2a is None or v2b is None:
        raise ValueError("geometry results are required")
    base = overlay_mask(image, mask, alpha=0.35)
    panel = Image.fromarray(base)
    draw = ImageDraw.Draw(panel)
    line_width = max(1, round(min(mask.shape) / 180))
    if v2a.final_cutline is not None:
        _draw_segment(draw, v2a.final_cutline, color=V2A_FINAL_COLOR, width=line_width)
    if v2b.final_cutline is not None:
        _draw_segment(
            draw,
            v2b.final_cutline,
            color=V2B_LINE_COLOR,
            width=max(2, line_width + 1),
        )
    status = f"v2a={_structured_status(v2a)}  v2b={_structured_status(v2b)}"
    draw.rectangle((0, 0, min(mask.shape[1] - 1, 410), 15), fill=(0, 0, 0))
    draw.text((3, 2), status[:68], fill=(255, 255, 255))
    return np.asarray(panel, dtype=np.uint8)


def _draw_segment(
    draw: ImageDraw.ImageDraw,
    segment: LineSegment,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    draw.line(
        (
            segment.start.x,
            segment.start.y,
            segment.end.x,
            segment.end.y,
        ),
        fill=color,
        width=width,
    )


def _structured_status(
    result: FixedAxisCutlineResult | FixedAxisSearchCutlineResult,
) -> str:
    return (
        "ok" if result.status == "ok" else f"failed:{result.failure_code or 'unknown'}"
    )


def _message_panel(shape: tuple[int, int], message: str) -> np.ndarray:
    height, width = shape
    panel = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(panel)
    lines = wrap(message, width=max(18, width // 7))
    y = max(4, height // 2 - 7 * len(lines))
    for line in lines:
        draw.text((5, y), line, fill=(255, 255, 255))
        y += 14
    return np.asarray(panel, dtype=np.uint8)


def _titled_image_panel(
    title: str,
    panel_array: np.ndarray,
    *,
    display_width: int,
    display_height: int,
) -> Image.Image:
    panel = Image.new(
        "RGB",
        (display_width, display_height + TITLE_HEIGHT),
        color=(0, 0, 0),
    )
    ImageDraw.Draw(panel).text((4, 5), title, fill=(255, 255, 255))
    resized = Image.fromarray(panel_array).resize(
        (display_width, display_height),
        resample=Image.Resampling.NEAREST,
    )
    panel.paste(resized, (0, TITLE_HEIGHT))
    return panel


def _metric_panel(
    *,
    width: int,
    height: int,
    sample_id: str,
    metadata: Mapping[str, str],
    metrics: Mapping[str, Any],
    fixed_axis_eligible: bool,
    ground_truth_v2a: FixedAxisCutlineResult | None,
    ground_truth_v2b: FixedAxisSearchCutlineResult | None,
    predicted_v2a: FixedAxisCutlineResult | None,
    predicted_v2b: FixedAxisSearchCutlineResult | None,
) -> Image.Image:
    panel = Image.new("RGB", (width, height), color=(18, 18, 18))
    draw = ImageDraw.Draw(panel)
    draw.text((5, 5), PANEL_TITLES[-1], fill=(255, 255, 255))
    lines = [
        f"sample_id={sample_id}",
        f"orientation={metadata.get('orientation_category', '') or 'unknown'}",
        f"fixed_axis_eligible={metadata.get('fixed_axis_eligible', '') or 'blank'}",
        f"occlusion={metadata.get('occlusion_category', '') or 'unknown'}",
        f"annotation_qa={metadata.get('annotation_qa_status', '') or 'unknown'}",
        (
            "IoU Flesh/Calyx="
            f"{_format_metric(metrics.get('iou_flesh'))}/"
            f"{_format_metric(metrics.get('iou_calyx'))}"
        ),
        (
            "Dice Flesh/Calyx="
            f"{_format_metric(metrics.get('dice_flesh'))}/"
            f"{_format_metric(metrics.get('dice_calyx'))}"
        ),
        (
            "Calyx precision/recall="
            f"{_format_metric(metrics.get('calyx_precision'))}/"
            f"{_format_metric(metrics.get('calyx_recall'))}"
        ),
        (
            "mean max-confidence/entropy="
            f"{_format_metric(metrics.get('mean_maximum_softmax_confidence'))}/"
            f"{_format_metric(metrics.get('mean_normalized_entropy'))}"
        ),
    ]
    if fixed_axis_eligible:
        assert ground_truth_v2a is not None
        assert ground_truth_v2b is not None
        assert predicted_v2a is not None
        assert predicted_v2b is not None
        lines.extend(
            (
                f"GT v2a/v2b={_structured_status(ground_truth_v2a)}/"
                f"{_structured_status(ground_truth_v2b)}",
                f"Pred v2a/v2b={_structured_status(predicted_v2a)}/"
                f"{_structured_status(predicted_v2b)}",
                "white=v2a, green=v2b; mask geometry only",
            )
        )
    else:
        lines.append(INELIGIBLE_MESSAGE)

    y = TITLE_HEIGHT + 3
    characters = max(22, width // 7)
    for line in lines:
        for wrapped_line in wrap(line, width=characters) or [""]:
            draw.text((6, y), wrapped_line, fill=(255, 255, 255))
            y += 14
    return panel


def _format_metric(value: Any) -> str:
    if value is None:
        return "n/a"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.3f}" if np.isfinite(number) else "n/a"


def _rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("image must use an integer dtype")
    if np.any(array < 0) or np.any(array > 255):
        raise ValueError("image values must be in [0, 255]")
    return array.astype(np.uint8, copy=True)


def _class_mask(mask: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [H, W]")
    if not np.all(np.isin(array, (0, 1, 2))):
        raise ValueError(f"{name} must contain only class IDs 0, 1, and 2")
    return array.astype(np.uint8, copy=False)


def _unit_interval(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2 or not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must be a finite [H, W] array")
    if np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} values must be in [0, 1]")
    return array
