"""Headless Milestone 4 matched-quartet visualizations and aggregate plots."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from textwrap import wrap
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "strawberry-occlusion-matplotlib"),
)

import matplotlib
import numpy as np
from PIL import Image, ImageDraw

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402

from strawberry_occlusion.geometry import LineSegment  # noqa: E402
from strawberry_occlusion.visualization.fixed_axis_comparison import (  # noqa: E402
    V2A_FINAL_COLOR,
    V2B_LINE_COLOR,
)


PathLike = str | Path
REGION_ORDER = ("attachment", "calyx_tip", "flesh_far", "background")
PLOT_FILENAMES = (
    "stability_vs_severity_v2a.png",
    "stability_vs_severity_v2b.png",
    "gt_reference_error_vs_severity_v2a.png",
    "gt_reference_error_vs_severity_v2b.png",
    "structured_failure_vs_severity.png",
    "silent_drift_vs_severity.png",
    "region_ablation.png",
    "iou_vs_cut_stability.png",
)


def create_matched_quartet_visualization(
    clean_rgb: np.ndarray,
    conditions: Mapping[str, Mapping[str, Any]],
    *,
    sample_id: str,
    severity_fraction: float,
    seed: int,
    occluder_id: str,
) -> Image.Image:
    """Render clean plus four matched translations with clean/perturbed geometry."""

    clean = _rgb_array(clean_rgb)
    if set(conditions) != set(REGION_ORDER):
        raise ValueError("conditions must contain the four frozen regions")
    panels: list[Image.Image] = []
    clean_results = conditions["attachment"].get("clean_results")
    if not isinstance(clean_results, Mapping):
        raise ValueError("conditions must provide clean_results")
    panels.append(
        _panel(
            clean,
            "Clean RGB",
            footer=(
                f"sample={sample_id} | severity={severity_fraction:.2f} | seed={seed}\n"
                f"occluder={occluder_id}"
            ),
            clean_results=clean_results,
            occluded_results=None,
        )
    )
    for region in REGION_ORDER:
        condition = conditions[region]
        image = _rgb_array(condition.get("image"))
        if image.shape != clean.shape:
            raise ValueError("all matched-condition images must match clean_rgb")
        metrics = condition.get("metrics")
        occluded_results = condition.get("occluded_results")
        if not isinstance(metrics, Mapping) or not isinstance(
            occluded_results, Mapping
        ):
            raise ValueError("conditions must provide metrics and occluded_results")
        lines = []
        for method in ("v2a", "v2b"):
            row = metrics.get(method, {})
            lines.append(
                f"{method}: Δ vs clean={_format(row.get('stability_signed'))} px | "
                f"GT err={_format(row.get('gt_reference_abs'))} px | "
                f"status={row.get('method_status', 'n/a')}"
            )
        panels.append(
            _panel(
                image,
                region.replace("_", " ").title(),
                footer=(
                    f"severity={severity_fraction:.2f} | seed={seed} | "
                    f"occluder={occluder_id}\n" + "\n".join(lines)
                ),
                clean_results=clean_results,
                occluded_results=occluded_results,
            )
        )
    panel_width = panels[0].width
    panel_height = panels[0].height
    canvas = Image.new(
        "RGB", (panel_width * len(panels), panel_height), color=(0, 0, 0)
    )
    for index, panel in enumerate(panels):
        canvas.paste(panel, (index * panel_width, 0))
    return canvas


def create_occlusion_robustness_plots(
    trial_rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Sequence[Mapping[str, Any]]],
    output_root: PathLike,
) -> dict[str, str]:
    """Write the eight deterministic frozen-profile aggregate plots."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / name for name in PLOT_FILENAMES}
    condition = list(analysis.get("condition_summary", ()))
    _line_by_region(
        condition,
        paths["stability_vs_severity_v2a.png"],
        method="v2a",
        field="mean_stability_abs",
        ylabel="Group-weighted mean |c_occ - c_clean| (pixels)",
        title="v2a action stability under matched occlusion",
    )
    _line_by_region(
        condition,
        paths["stability_vs_severity_v2b.png"],
        method="v2b",
        field="mean_stability_abs",
        ylabel="Group-weighted mean |c_occ - c_clean| (pixels)",
        title="v2b action stability under matched occlusion",
    )
    _line_by_region(
        condition,
        paths["gt_reference_error_vs_severity_v2a.png"],
        method="v2a",
        field="mean_gt_reference_abs",
        ylabel="Group-weighted mean |c_occ - c_gt| (pixels)",
        title="v2a GT-reference error under matched occlusion",
    )
    _line_by_region(
        condition,
        paths["gt_reference_error_vs_severity_v2b.png"],
        method="v2b",
        field="mean_gt_reference_abs",
        ylabel="Group-weighted mean |c_occ - c_gt| (pixels)",
        title="v2b GT-reference error under matched occlusion",
    )
    _status_plot(condition, paths["structured_failure_vs_severity.png"])
    _silent_drift_plot(condition, paths["silent_drift_vs_severity.png"])
    _region_ablation_plot(
        analysis.get("paired_region_differences", ()),
        paths["region_ablation.png"],
    )
    _iou_stability_plot(trial_rows, paths["iou_vs_cut_stability.png"])
    return {name: path.name for name, path in paths.items()}


def _panel(
    image: np.ndarray,
    title: str,
    *,
    footer: str,
    clean_results: Mapping[str, Any],
    occluded_results: Mapping[str, Any] | None,
) -> Image.Image:
    height, width = image.shape[:2]
    scale = max(1, int(np.ceil(240 / width)), int(np.ceil(160 / height)))
    display_width = width * scale
    display_height = height * scale
    title_height = 24
    footer_lines = sum(
        max(1, len(wrap(line, width=max(28, display_width // 7))))
        for line in footer.splitlines()
    )
    footer_height = max(54, 8 + 14 * footer_lines)
    panel = Image.new(
        "RGB",
        (display_width, title_height + display_height + footer_height),
        color=(16, 16, 16),
    )
    draw = ImageDraw.Draw(panel)
    draw.text((5, 5), title, fill=(255, 255, 255))
    overlay = Image.fromarray(image.copy())
    overlay_draw = ImageDraw.Draw(overlay)
    for method, color in (("v2a", V2A_FINAL_COLOR), ("v2b", V2B_LINE_COLOR)):
        clean = clean_results.get(method)
        segment = getattr(clean, "final_cutline", None)
        if segment is not None:
            _draw_dashed(overlay_draw, segment, color=color, width=1)
        if occluded_results is not None:
            occluded = occluded_results.get(method)
            segment = getattr(occluded, "final_cutline", None)
            if segment is not None:
                _draw_segment(
                    overlay_draw,
                    segment,
                    color=color,
                    width=2 if method == "v2b" else 1,
                )
    overlay = overlay.resize((display_width, display_height), Image.Resampling.NEAREST)
    panel.paste(overlay, (0, title_height))
    y = title_height + display_height + 5
    for source_line in footer.splitlines():
        for line in wrap(source_line, width=max(28, display_width // 7)) or [""]:
            draw.text((5, y), line, fill=(238, 238, 238))
            y += 14
    return panel


def _line_by_region(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    method: str,
    field: str,
    ylabel: str,
    title: str,
) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    plotted = False
    for region in REGION_ORDER:
        selected = sorted(
            (
                row
                for row in rows
                if row.get("method") == method and row.get("region") == region
            ),
            key=lambda row: _number(row.get("severity_fraction"), default=0.0),
        )
        points = [
            (_number(row.get("severity_fraction")), _number(row.get(field)))
            for row in selected
            if _number(row.get("severity_fraction")) is not None
            and _number(row.get(field)) is not None
        ]
        if not points:
            continue
        plotted = True
        axis.plot(
            [point[0] for point in points],
            [point[1] for point in points],
            marker="o",
            label=region.replace("_", " "),
        )
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No complete finite conditions",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel("Occluder area / clean visible foreground area")
    axis.set_ylabel(ylabel)
    axis.set_title(
        f"{title}\nn_groups={_maximum_group_count(selected_rows=rows, method=method)}"
    )
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _status_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.8), constrained_layout=True, sharey=True
    )
    for axis, method in zip(axes, ("v2a", "v2b"), strict=True):
        _plot_condition_field(
            axis, rows, method=method, field="structured_failure_rate"
        )
        axis.set_title(method)
        axis.set_xlabel("Occluder area / foreground area")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Group-weighted structured failure rate")
    figure.suptitle(
        "Structured no-cut/failure response by severity "
        f"(n_groups={_maximum_group_count(selected_rows=rows)})"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _silent_drift_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axes = plt.subplots(
        1, 2, figsize=(12.0, 4.8), constrained_layout=True, sharey=True
    )
    colors = {2: "#1f77b4", 5: "#ff7f0e", 10: "#d62728"}
    for axis, method in zip(axes, ("v2a", "v2b"), strict=True):
        for threshold in (2, 5, 10):
            points = []
            for severity in (0.01, 0.02, 0.04):
                values = [
                    _number(row.get(f"silent_drift_{threshold}_rate"))
                    for row in rows
                    if row.get("method") == method
                    and _number(row.get("severity_fraction")) == severity
                ]
                finite = [value for value in values if value is not None]
                if finite:
                    points.append((severity, float(np.mean(finite))))
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    marker="o",
                    color=colors[threshold],
                    label=f"> {threshold} px",
                )
        axis.set_title(method)
        axis.set_xlabel("Occluder area / foreground area")
        axis.grid(alpha=0.25)
        if axis.lines:
            axis.legend()
    axes[0].set_ylabel("Group-weighted descriptive silent-drift rate")
    figure.suptitle(
        "Finite apparently valid coordinates with stability drift "
        f"(n_groups={_maximum_group_count(selected_rows=rows)})"
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_condition_field(
    axis: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
    field: str,
) -> None:
    for region in REGION_ORDER:
        points = sorted(
            (
                (_number(row.get("severity_fraction")), _number(row.get(field)))
                for row in rows
                if row.get("method") == method and row.get("region") == region
            ),
            key=lambda point: -1.0 if point[0] is None else point[0],
        )
        points = [
            point for point in points if point[0] is not None and point[1] is not None
        ]
        if points:
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                label=region.replace("_", " "),
            )
    if axis.lines:
        axis.legend(fontsize=8)


def _region_ablation_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(9.0, 5.2), constrained_layout=True)
    labels = []
    values = []
    for method in ("v2a", "v2b"):
        for control in ("background", "flesh_far", "calyx_tip"):
            selected = [
                _number(
                    row.get("stability_abs_auc_difference_attachment_minus_control")
                )
                for row in rows
                if row.get("method") == method
                and row.get("comparison") == f"attachment_minus_{control}"
            ]
            finite = [value for value in selected if value is not None]
            labels.append(f"{method}\natt - {control.replace('_', ' ')}")
            values.append(float(np.mean(finite)) if finite else np.nan)
    x = np.arange(len(labels))
    axis.bar(
        x,
        np.nan_to_num(values, nan=0.0),
        color=["#4c78a8"] * 3 + ["#59a14f"] * 3,
    )
    for index, value in enumerate(values):
        if np.isnan(value):
            axis.text(index, 0.0, "n/a", ha="center", va="bottom")
    axis.axhline(0.0, color="black", linewidth=1)
    axis.set_xticks(x, labels)
    axis.set_ylabel("Mean group AUC difference in stability_abs (pixels)")
    axis.set_title(
        "Matched region ablation: attachment minus control\n"
        f"n_groups={len({str(row.get('group_id')) for row in rows if row.get('group_id')})}"
    )
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _iou_stability_plot(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    figure, axis = plt.subplots(figsize=(7.5, 5.5), constrained_layout=True)
    plotted = False
    for method, color in (("v2a", "#4c78a8"), ("v2b", "#59a14f")):
        points = [
            (
                _number(row.get("mean_foreground_iou")),
                _number(row.get("stability_abs")),
            )
            for row in rows
            if row.get("method") == method
        ]
        points = [
            point for point in points if point[0] is not None and point[1] is not None
        ]
        if points:
            plotted = True
            axis.scatter(
                [point[0] for point in points],
                [point[1] for point in points],
                s=20,
                alpha=0.65,
                color=color,
                label=method,
            )
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No paired finite coordinates",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel("Visible mean foreground IoU (occluder pixels excluded)")
    axis.set_ylabel("|c_occ - c_clean| (pixels)")
    n_groups = len(
        {
            str(row.get("group_id"))
            for row in rows
            if row.get("group_id") not in (None, "")
        }
    )
    axis.set_title(
        "Visible segmentation quality versus action stability\n"
        f"n_groups={n_groups}; points are repeated perturbation measurements, "
        "not independent statistical units"
    )
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _draw_segment(
    draw: ImageDraw.ImageDraw,
    segment: LineSegment,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    draw.line(
        (segment.start.x, segment.start.y, segment.end.x, segment.end.y),
        fill=color,
        width=width,
    )


def _draw_dashed(
    draw: ImageDraw.ImageDraw,
    segment: LineSegment,
    *,
    color: tuple[int, int, int],
    width: int,
) -> None:
    delta_x = segment.end.x - segment.start.x
    delta_y = segment.end.y - segment.start.y
    length = float(np.hypot(delta_x, delta_y))
    if length == 0.0:
        return
    dash = max(2.0, length / 18.0)
    position = 0.0
    while position < length:
        end = min(position + dash, length)
        start_fraction = position / length
        end_fraction = end / length
        draw.line(
            (
                segment.start.x + start_fraction * delta_x,
                segment.start.y + start_fraction * delta_y,
                segment.start.x + end_fraction * delta_x,
                segment.start.y + end_fraction * delta_y,
            ),
            fill=color,
            width=width,
        )
        position += 2.0 * dash


def _rgb_array(value: Any) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError("visualization images must use an integer dtype")
    if np.any(array < 0) or np.any(array > 255):
        raise ValueError("visualization image values must be in [0, 255]")
    return array.astype(np.uint8, copy=True)


def _number(value: Any, *, default: float | None = None) -> float | None:
    if value is None or value == "":
        return default
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if np.isfinite(number) else default


def _maximum_group_count(
    *,
    selected_rows: Sequence[Mapping[str, Any]],
    method: str | None = None,
) -> int:
    counts = [
        _number(row.get("n_groups"))
        for row in selected_rows
        if method is None or row.get("method") == method
    ]
    finite = [int(value) for value in counts if value is not None]
    return max(finite, default=0)


def _format(value: Any) -> str:
    number = _number(value)
    return "n/a" if number is None else f"{number:.2f}"
