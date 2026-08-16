"""Deterministic group-level visualizations for the M5-v0 analysis."""

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


PathLike = str | Path
PLOT_FILENAMES = (
    "entropy_vs_action_instability.png",
    "global_vs_attachment_association.png",
    "geometry_support_vs_action_instability.png",
)


def create_action_failure_signal_plots(
    analysis: Mapping[str, Sequence[Mapping[str, Any]]],
    output_root: PathLike,
) -> dict[str, str]:
    """Write three preliminary figures using group means as plotted points."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    group_rows = analysis.get("per_group_signal_summary", ())
    associations = analysis.get("association_table", ())
    paths = {name: root / name for name in PLOT_FILENAMES}
    _entropy_scatter(group_rows, paths["entropy_vs_action_instability.png"])
    _association_comparison(
        associations,
        paths["global_vs_attachment_association.png"],
    )
    _geometry_scatter(
        group_rows,
        paths["geometry_support_vs_action_instability.png"],
    )
    return {name: f"plots/{name}" for name in PLOT_FILENAMES}


def _entropy_scatter(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    signals = (
        ("predictive_entropy_global", "Global predictive entropy"),
        ("predictive_entropy_attachment", "Predicted-attachment entropy"),
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
        constrained_layout=True,
        sharey=True,
    )
    for axis, (signal, title) in zip(axes, signals, strict=True):
        selected = _group_points(rows, signal=signal)
        _draw_group_scatter(axis, selected)
        axis.set_title(f"{title}\nn_groups={len(selected)}")
        axis.set_xlabel("Mean current-observation signal")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean |c_occ - c_clean| (pixels)")
    figure.suptitle("v2b group-level action-instability diagnostics")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _association_comparison(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
) -> None:
    signals = (
        ("predictive_entropy_global", "Entropy\nglobal"),
        ("predictive_entropy_attachment", "Entropy\nattachment"),
        ("max_probability_global", "Max prob.\nglobal"),
        ("max_probability_attachment", "Max prob.\nattachment"),
        ("probability_margin_global", "Margin\nglobal"),
        ("probability_margin_attachment", "Margin\nattachment"),
    )
    selected = {
        str(row.get("signal")): row
        for row in rows
        if row.get("method") == "v2b" and row.get("region_scope") == "all"
    }
    values = [
        _number(selected.get(name, {}).get("spearman_rho")) for name, _ in signals
    ]
    plotted = [value if value is not None else 0.0 for value in values]
    colors = ["#4c78a8", "#f58518"] * 3
    figure, axis = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    bars = axis.bar(np.arange(len(signals)), plotted, color=colors)
    for bar, value in zip(bars, values, strict=True):
        if value is None:
            axis.text(
                bar.get_x() + bar.get_width() / 2,
                0.02,
                "undefined",
                ha="center",
                va="bottom",
                rotation=90,
                fontsize=8,
            )
    axis.axhline(0.0, color="black", linewidth=0.8)
    axis.set_xticks(np.arange(len(signals)), [label for _, label in signals])
    axis.set_ylim(-1.05, 1.05)
    axis.set_ylabel("Spearman rho across group means")
    axis.set_title(
        "Global versus predicted-attachment signal association\n"
        "descriptive v2b development analysis"
    )
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _geometry_scatter(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    signals = (
        (
            "v2b_selected_contact_evidence_fraction",
            "Selected contact evidence fraction",
        ),
        ("v2b_selected_block_relative_size", "Selected feasible-block relative size"),
    )
    figure, axes = plt.subplots(
        1,
        2,
        figsize=(11.0, 4.8),
        constrained_layout=True,
        sharey=True,
    )
    for axis, (signal, title) in zip(axes, signals, strict=True):
        selected = _group_points(rows, signal=signal)
        _draw_group_scatter(axis, selected)
        axis.set_title(f"{title}\nn_groups={len(selected)}")
        axis.set_xlabel("Mean current-v2b support signal")
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Mean |c_occ - c_clean| (pixels)")
    figure.suptitle("Geometry evidence support versus v2b action instability")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _group_points(
    rows: Sequence[Mapping[str, Any]],
    *,
    signal: str,
) -> list[tuple[float, float]]:
    points = []
    for row in rows:
        if (
            row.get("method") != "v2b"
            or row.get("region_scope") != "all"
            or row.get("signal") != signal
        ):
            continue
        x = _number(row.get("mean_signal"))
        y = _number(row.get("mean_stability_abs"))
        if x is not None and y is not None:
            points.append((x, y))
    return points


def _draw_group_scatter(axis: Any, points: Sequence[tuple[float, float]]) -> None:
    if points:
        axis.scatter(
            [point[0] for point in points],
            [point[1] for point in points],
            color="#4c78a8",
            alpha=0.85,
            edgecolors="white",
            linewidths=0.5,
        )
    else:
        axis.text(
            0.5,
            0.5,
            "No pairwise-complete groups",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None
