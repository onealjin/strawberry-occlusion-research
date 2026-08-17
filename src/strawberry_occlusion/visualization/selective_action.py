"""Deterministic, visibly stamped M5-v1 selective-action figures."""

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
    "primary_geometry_risk_vs_drift.png",
    "selective_action_curve.png",
    "common_group_selective_action_curve.png",
)
DEVELOPMENT_BANNER = "DESCRIPTIVE — DEVELOPMENT REUSE — NON-CONFIRMATORY"
DEVELOPMENT_COMPARATOR_NOTE = (
    "Comparator shown for context; the geometry signal was selected on this "
    "development cohort."
)
DEVELOPMENT_ASSOCIATION_NOTE = (
    "Association is an integrity/re-expression check on development reuse."
)
_SCORE_STYLES = {
    "geometry_risk": ("Geometry risk", "#4c78a8"),
    "perception_risk": ("Attachment entropy comparator", "#f58518"),
}


def create_selective_action_plots(
    curve_rows: Sequence[Mapping[str, Any]],
    group_rows: Sequence[Mapping[str, Any]],
    association_rows: Sequence[Mapping[str, Any]],
    output_root: PathLike,
    *,
    method: str,
    region: str,
    n_groups: int,
    cohort_role: str,
) -> dict[str, str]:
    """Write the primary and fixed-evaluation-population secondary curves."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    paths = {name: root / name for name in PLOT_FILENAMES}
    _geometry_association(
        group_rows,
        association_rows,
        paths["primary_geometry_risk_vs_drift.png"],
        method=method,
        region=region,
        n_groups=n_groups,
        cohort_role=cohort_role,
    )
    _primary_curve(
        curve_rows,
        paths["selective_action_curve.png"],
        method=method,
        region=region,
        n_groups=n_groups,
        cohort_role=cohort_role,
    )
    _common_group_curve(
        curve_rows,
        paths["common_group_selective_action_curve.png"],
        method=method,
        region=region,
        n_groups=n_groups,
        cohort_role=cohort_role,
    )
    return {name: f"plots/{name}" for name in PLOT_FILENAMES}


def _geometry_association(
    group_rows: Sequence[Mapping[str, Any]],
    association_rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    method: str,
    region: str,
    n_groups: int,
    cohort_role: str,
) -> None:
    selected = [
        row for row in group_rows if str(row.get("score_name")) == "geometry_risk"
    ]
    association = next(
        row for row in association_rows if str(row.get("score_name")) == "geometry_risk"
    )
    x = [_number(row.get("group_mean_risk")) for row in selected]
    y = [_number(row.get("group_mean_stability_abs")) for row in selected]
    if any(value is None for value in (*x, *y)):
        raise ValueError("Geometry association figure requires finite group means")
    figure, axis = plt.subplots(figsize=(8.8, 6.4))
    axis.scatter(
        x,
        y,
        color="#4c78a8",
        alpha=0.88,
        edgecolors="white",
        linewidths=0.6,
    )
    rho = _number(association.get("spearman_rho"))
    ci_lower = _number(association.get("ci_lower"))
    ci_upper = _number(association.get("ci_upper"))
    rho_text = "undefined" if rho is None else f"{rho:.4f}"
    ci_text = (
        "undefined"
        if ci_lower is None or ci_upper is None
        else f"[{ci_lower:.4f}, {ci_upper:.4f}]"
    )
    axis.text(
        0.02,
        0.98,
        f"Group-first Spearman rho = {rho_text}\n95% group-bootstrap CI = {ci_text}",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "edgecolor": "#cccccc", "alpha": 0.9},
    )
    axis.set_xlabel("Group mean geometry_risk")
    axis.set_ylabel("Group mean stability_abs (pixels)")
    axis.set_title("M5-v1 primary geometry-risk association")
    axis.grid(alpha=0.25)
    _stamp(
        figure,
        method=method,
        region=region,
        group_text=f"n groups = {n_groups}",
        cohort_role=cohort_role,
        development_note=DEVELOPMENT_ASSOCIATION_NOTE,
    )
    figure.subplots_adjust(bottom=0.24, top=0.88, left=0.12, right=0.97)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _primary_curve(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    method: str,
    region: str,
    n_groups: int,
    cohort_role: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 6.4))
    for score_name, (label, color) in _SCORE_STYLES.items():
        selected = _score_rows(rows, score_name)
        x = np.asarray(
            [_number(row.get("realized_observation_coverage")) for row in selected],
            dtype=np.float64,
        )
        y = np.asarray(
            [_number(row.get("group_balanced_mean_abs_drift")) for row in selected],
            dtype=np.float64,
        )
        low = np.asarray(
            [_number(row.get("group_cluster_bootstrap_ci_low")) for row in selected],
            dtype=np.float64,
        )
        high = np.asarray(
            [_number(row.get("group_cluster_bootstrap_ci_high")) for row in selected],
            dtype=np.float64,
        )
        random_mean = np.asarray(
            [_number(row.get("random_reference_mean")) for row in selected],
            dtype=np.float64,
        )
        random_low = np.asarray(
            [_number(row.get("random_reference_p05")) for row in selected],
            dtype=np.float64,
        )
        random_high = np.asarray(
            [_number(row.get("random_reference_p95")) for row in selected],
            dtype=np.float64,
        )
        axis.plot(x, y, marker="o", linewidth=2.0, color=color, label=label)
        axis.vlines(x, low, high, color=color, alpha=0.45, linewidth=1.2)
        axis.plot(
            x,
            random_mean,
            linestyle="--",
            linewidth=1.2,
            color=color,
            alpha=0.65,
            label=f"{label} random reference",
        )
        axis.fill_between(x, random_low, random_high, color=color, alpha=0.08)
    axis.set_xlabel(
        "Realized observation coverage\n"
        "(global-threshold policies indexed by requested 1.00–0.50 grid)"
    )
    axis.set_ylabel("Group-balanced mean retained |c_occ - c_clean| (pixels)")
    axis.set_title(
        "M5-v1 primary selective-action curve\n"
        "Realized coverage may exceed requested coverage after ceil rounding/ties"
    )
    axis.text(
        0.01,
        0.01,
        "Vertical ranges: 95% group-bootstrap CI\n"
        "Shaded random ranges: descriptive p05–p95 (not a CI)",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        fontsize=8,
        color="#444444",
    )
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    _stamp(
        figure,
        method=method,
        region=region,
        group_text=f"n groups = {n_groups}",
        cohort_role=cohort_role,
        development_note=DEVELOPMENT_COMPARATOR_NOTE,
    )
    figure.subplots_adjust(bottom=0.27, top=0.82, left=0.11, right=0.97)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _common_group_curve(
    rows: Sequence[Mapping[str, Any]],
    path: Path,
    *,
    method: str,
    region: str,
    n_groups: int,
    cohort_role: str,
) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 6.4))
    common_count = int(rows[0].get("common_retention_group_count", 0)) if rows else 0
    plotted = False
    for score_name, (label, color) in _SCORE_STYLES.items():
        selected = _score_rows(rows, score_name)
        points = [
            (
                _number(row.get("realized_observation_coverage")),
                _number(row.get("common_group_balanced_mean_abs_drift")),
            )
            for row in selected
        ]
        defined = [(x, y) for x, y in points if x is not None and y is not None]
        if not defined:
            continue
        plotted = True
        axis.plot(
            [point[0] for point in defined],
            [point[1] for point in defined],
            marker="o",
            linewidth=2.0,
            color=color,
            label=label,
        )
    if not plotted:
        axis.text(
            0.5,
            0.5,
            f"Undefined: fewer than 3 common retention groups (n={common_count})",
            ha="center",
            va="center",
            transform=axis.transAxes,
        )
    axis.set_xlabel("Realized observation coverage under the primary global thresholds")
    axis.set_ylabel("Common-group balanced mean retained drift (pixels)")
    axis.set_title(
        "M5-v1 common retention-stable group curve\n"
        f"Evaluation restricted after policy application; common groups={common_count}"
    )
    axis.grid(alpha=0.25)
    if plotted:
        axis.legend(fontsize=9)
    _stamp(
        figure,
        method=method,
        region=region,
        group_text=(
            f"total targeted groups = {n_groups} | "
            f"common retention groups = {common_count}"
        ),
        cohort_role=cohort_role,
        development_note=DEVELOPMENT_COMPARATOR_NOTE,
    )
    figure.subplots_adjust(bottom=0.27, top=0.82, left=0.11, right=0.97)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def _score_rows(
    rows: Sequence[Mapping[str, Any]], score_name: str
) -> list[Mapping[str, Any]]:
    selected = [row for row in rows if str(row.get("score")) == score_name]
    return sorted(
        selected,
        key=lambda row: float(row["realized_observation_coverage"]),
    )


def _stamp(
    figure: Any,
    *,
    method: str,
    region: str,
    group_text: str,
    cohort_role: str,
    development_note: str | None,
) -> None:
    figure.text(
        0.5,
        0.09,
        f"method = {method} | region = {region} | {group_text} | "
        f"cohort role = {cohort_role}",
        ha="center",
        va="center",
        fontsize=9,
    )
    if cohort_role == "development_reuse":
        if development_note is not None:
            figure.text(
                0.5,
                0.055,
                development_note,
                ha="center",
                va="center",
                fontsize=8.5,
                color="#555555",
            )
        figure.text(
            0.5,
            0.02,
            DEVELOPMENT_BANNER,
            ha="center",
            va="center",
            fontsize=10,
            color="#9c2f2f",
            weight="bold",
        )


def _number(value: Any) -> float | None:
    if value in (None, ""):
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if np.isfinite(numeric) else None
