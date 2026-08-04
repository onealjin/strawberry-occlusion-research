"""Headless aggregate plots for fixed-axis-v2b parameter sensitivity."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "strawberry-occlusion-matplotlib"),
)

import matplotlib
import numpy as np

matplotlib.use("Agg", force=True)
from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import MaxNLocator  # noqa: E402


PathLike = str | Path


def create_fixed_axis_sensitivity_plots(
    configuration_rows: Sequence[Mapping[str, Any]],
    stability_rows: Sequence[Mapping[str, Any]],
    output_root: PathLike,
) -> dict[str, str]:
    """Write the four deterministic sensitivity figures and return their paths."""

    if not configuration_rows:
        raise ValueError("configuration_rows must not be empty")
    plots_root = Path(output_root)
    plots_root.mkdir(parents=True, exist_ok=True)
    paths = {
        "aggregate_metric_heatmaps": plots_root / "aggregate_metric_heatmaps.png",
        "status_and_fragmentation_heatmaps": plots_root
        / "status_and_fragmentation_heatmaps.png",
        "coordinate_stability": plots_root / "coordinate_stability.png",
        "configuration_tradeoffs": plots_root / "configuration_tradeoffs.png",
    }
    _plot_aggregate_metric_heatmaps(
        configuration_rows,
        paths["aggregate_metric_heatmaps"],
    )
    _plot_status_and_fragmentation_heatmaps(
        configuration_rows,
        paths["status_and_fragmentation_heatmaps"],
    )
    _plot_coordinate_stability(stability_rows, paths["coordinate_stability"])
    _plot_configuration_tradeoffs(
        configuration_rows,
        paths["configuration_tradeoffs"],
    )
    return {name: path.name for name, path in paths.items()}


def _plot_aggregate_metric_heatmaps(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    evidence_values, width_values = _grid_axes(rows)
    panels = (
        ("success_rate", "Success rate", 2, (0.0, 1.0)),
        (
            "median_whole_mask_flesh_loss_ratio",
            "Median whole-mask Flesh-loss proxy",
            4,
            None,
        ),
        (
            "median_whole_mask_calyx_retention_ratio",
            "Median whole-mask Calyx-retention proxy",
            4,
            None,
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.5), constrained_layout=True)
    for axis, (field, title, minimum_decimals, fixed_limits) in zip(
        axes,
        panels,
        strict=True,
    ):
        matrix = _metric_matrix(rows, evidence_values, width_values, field)
        minimum, maximum = (
            fixed_limits if fixed_limits is not None else _finite_panel_limits(matrix)
        )
        _draw_heatmap(
            axis,
            matrix,
            evidence_values=evidence_values,
            width_values=width_values,
            title=title,
            value_format=(
                f".{_distinguishing_decimal_places(matrix, minimum_decimals)}f"
            ),
            color_map="viridis",
            minimum=minimum,
            maximum=maximum,
        )
    figure.suptitle(
        "Fixed-axis-v2b sensitivity: geometric mask proxies, not physical outcomes",
        fontsize=12,
    )
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_status_and_fragmentation_heatmaps(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    evidence_values, width_values = _grid_axes(rows)
    panels = (
        ("fragmented_set_rate", "Fragmented feasible-set rate", ".2f"),
        ("singleton_selected_block_rate", "Singleton selection rate", ".2f"),
        ("structured_failure_count", "Structured failure count", ".0f"),
        ("median_feasible_block_count", "Median feasible-block count", ".1f"),
    )
    figure, axes = plt.subplots(2, 2, figsize=(9.5, 8.0), constrained_layout=True)
    for axis, (field, title, value_format) in zip(
        axes.flat,
        panels,
        strict=True,
    ):
        matrix = _metric_matrix(rows, evidence_values, width_values, field)
        _draw_heatmap(
            axis,
            matrix,
            evidence_values=evidence_values,
            width_values=width_values,
            title=title,
            value_format=value_format,
            color_map="magma",
        )
    figure.suptitle("Status and feasible-set fragmentation summary", fontsize=12)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_configuration_tradeoffs(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.0, 6.0), constrained_layout=True)
    plotted = False
    for row in rows:
        flesh = _finite_or_none(row.get("median_whole_mask_flesh_loss_ratio"))
        calyx = _finite_or_none(row.get("median_whole_mask_calyx_retention_ratio"))
        if flesh is None or calyx is None:
            continue
        plotted = True
        baseline = bool(row.get("is_committed_baseline"))
        axis.scatter(
            flesh,
            calyx,
            s=75 if baseline else 45,
            marker="*" if baseline else "o",
            color="#d62728" if baseline else "#1f77b4",
            zorder=3,
        )
        axis.annotate(
            str(row["configuration_id"]),
            (flesh, calyx),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
        )
    if not plotted:
        axis.text(
            0.5,
            0.5,
            "No successful finite proxy pairs",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel(
        "Median whole-mask Flesh-loss proxy\n"
        "Geometric visible-mask proxies; neither axis is a confirmed\n"
        "physical outcome."
    )
    axis.set_ylabel("Median whole-mask Calyx-retention proxy")
    axis.set_title("Configuration proxy trade-offs (no combined score)")
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _plot_coordinate_stability(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    ordered_rows = sorted(rows, key=lambda row: str(row["sample_id"]))
    height = max(4.5, 0.32 * max(1, len(ordered_rows)) + 1.8)
    figure, (coordinate_axis, failure_axis) = plt.subplots(
        1,
        2,
        figsize=(12.0, height),
        gridspec_kw={"width_ratios": (4.5, 1.0)},
        sharey=True,
        constrained_layout=True,
    )
    positions = np.arange(len(ordered_rows), dtype=np.float64)
    baseline_plotted = False
    for position, row in zip(positions, ordered_rows, strict=True):
        minimum = _finite_or_none(row.get("coordinate_minimum"))
        maximum = _finite_or_none(row.get("coordinate_maximum"))
        median_value = _finite_or_none(row.get("coordinate_median"))
        if minimum is None or maximum is None or median_value is None:
            continue
        coordinate_axis.hlines(
            position,
            minimum,
            maximum,
            color="#1f77b4",
            linewidth=2.0,
        )
        coordinate_axis.scatter(
            median_value,
            position,
            color="#1f77b4",
            s=22,
            zorder=3,
        )
        baseline_coordinate = _finite_or_none(row.get("committed_baseline_coordinate"))
        if baseline_coordinate is not None:
            baseline_plotted = True
            coordinate_axis.scatter(
                baseline_coordinate,
                position,
                marker="*",
                color="#d62728",
                edgecolor="black",
                linewidth=0.4,
                s=65,
                zorder=4,
            )
    failure_counts = [int(row["failure_count"]) for row in ordered_rows]
    colors = ["#d62728" if count else "#bdbdbd" for count in failure_counts]
    failure_axis.barh(positions, failure_counts, color=colors, height=0.6)
    failure_axis.set_xlabel("Failures")
    failure_axis.set_xlim(0.0, max(1.0, max(failure_counts, default=0) + 0.5))
    failure_axis.xaxis.set_major_locator(MaxNLocator(integer=True))
    failure_axis.grid(axis="x", alpha=0.2)
    if any(failure_counts):
        failure_axis.legend(
            handles=[Patch(facecolor="#d62728", label="Structured failures")],
            loc="lower right",
            fontsize=8,
        )

    coordinate_axis.set_yticks(positions)
    coordinate_axis.set_yticklabels(
        [str(row["sample_id"]) for row in ordered_rows],
        fontsize=8,
    )
    coordinate_axis.invert_yaxis()
    coordinate_axis.set_xlabel("Configured-axis cut coordinate (px)")
    coordinate_axis.set_title("Successful-coordinate range and median")
    coordinate_axis.grid(axis="x", alpha=0.25)
    coordinate_legend = [
        Line2D(
            [0],
            [0],
            color="#1f77b4",
            marker="o",
            markersize=4,
            linewidth=2,
            label="Successful coordinate range and median",
        )
    ]
    if baseline_plotted:
        coordinate_legend.append(
            Line2D(
                [0],
                [0],
                color="#d62728",
                marker="*",
                markeredgecolor="black",
                linewidth=0,
                markersize=9,
                label="Committed baseline e050_w064",
            )
        )
    coordinate_axis.legend(handles=coordinate_legend, loc="best", fontsize=8)
    failure_axis.set_title("Structured failures")
    figure.suptitle("Per-sample coordinate stability across configurations")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _grid_axes(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[list[float], list[float]]:
    evidence_values = sorted(
        {float(row["minimum_attachment_evidence_fraction"]) for row in rows}
    )
    width_values = sorted(
        {float(row["lateral_window_half_width_pixels"]) for row in rows}
    )
    return evidence_values, width_values


def _metric_matrix(
    rows: Sequence[Mapping[str, Any]],
    evidence_values: Sequence[float],
    width_values: Sequence[float],
    field: str,
) -> np.ndarray:
    lookup = {
        (
            float(row["minimum_attachment_evidence_fraction"]),
            float(row["lateral_window_half_width_pixels"]),
        ): _finite_or_none(row.get(field))
        for row in rows
    }
    return np.asarray(
        [
            [
                (
                    np.nan
                    if lookup.get((evidence, width)) is None
                    else lookup[(evidence, width)]
                )
                for width in width_values
            ]
            for evidence in evidence_values
        ],
        dtype=np.float64,
    )


def _draw_heatmap(
    axis: Any,
    matrix: np.ndarray,
    *,
    evidence_values: Sequence[float],
    width_values: Sequence[float],
    title: str,
    value_format: str,
    color_map: str,
    minimum: float | None = None,
    maximum: float | None = None,
) -> None:
    masked = np.ma.masked_invalid(matrix)
    color_map_instance = matplotlib.colormaps[color_map].with_extremes(bad="#d9d9d9")
    image = axis.imshow(
        masked,
        aspect="auto",
        origin="lower",
        cmap=color_map_instance,
        vmin=minimum,
        vmax=maximum,
    )
    axis.set_xticks(range(len(width_values)))
    axis.set_xticklabels([format(value, "g") for value in width_values])
    axis.set_yticks(range(len(evidence_values)))
    axis.set_yticklabels([format(value, "g") for value in evidence_values])
    axis.set_xlabel("Lateral half-width (px)")
    axis.set_ylabel("Minimum attachment evidence fraction")
    axis.set_title(title, fontsize=10)
    for row_index in range(matrix.shape[0]):
        for column_index in range(matrix.shape[1]):
            value = matrix[row_index, column_index]
            label = "n/a" if not np.isfinite(value) else format(value, value_format)
            axis.text(
                column_index,
                row_index,
                label,
                ha="center",
                va="center",
                color=(
                    "black"
                    if not np.isfinite(value)
                    else _annotation_text_color(image, float(value))
                ),
                fontsize=8,
            )
    axis.figure.colorbar(image, ax=axis, shrink=0.78)


def _finite_panel_limits(matrix: np.ndarray) -> tuple[float | None, float | None]:
    finite = matrix[np.isfinite(matrix)]
    if not len(finite):
        return None, None
    minimum = float(finite.min())
    maximum = float(finite.max())
    if minimum == maximum:
        padding = max(abs(minimum) * 0.01, 1e-6)
    else:
        padding = max((maximum - minimum) * 0.05, 1e-9)
    return minimum - padding, maximum + padding


def _distinguishing_decimal_places(
    matrix: np.ndarray,
    minimum_decimal_places: int,
) -> int:
    finite_unique = np.unique(matrix[np.isfinite(matrix)])
    decimal_places = minimum_decimal_places
    while decimal_places < 15 and len(
        {format(float(value), f".{decimal_places}f") for value in finite_unique}
    ) < len(finite_unique):
        decimal_places += 1
    return decimal_places


def _annotation_text_color(image: Any, value: float) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance >= 0.5 else "white"


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if np.isfinite(numeric) else None
