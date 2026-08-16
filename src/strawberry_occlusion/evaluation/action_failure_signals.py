"""Milestone 5-v0 action-instability signal extraction and group-aware analysis.

Primary risk signals in this module use only the current model probabilities and
the frozen geometry diagnostics produced from the current predicted mask.  The
clean coordinate enters only when the evaluation target is attached to a row.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from strawberry_occlusion.augmentation.synthetic_occluder import (
    MATCHED_REGIONS,
    _square_dilation,
)
from strawberry_occlusion.evaluation.occlusion_robustness import (
    METHODS,
    PROFILE_NAME as M4_PROFILE_NAME,
    _assert_no_absolute_paths,
    _as_bool,
    _csv_value,
    _finite_or_none,
    _install_staged_directory,
    _read_csv,
    _validate_output_destination,
    _write_csv,
    _write_json,
    benchmark_profile,
    run_occlusion_robustness,
)
from strawberry_occlusion.geometry import (
    FixedAxisCutlineResult,
    FixedAxisSearchCutlineResult,
)


PathLike = str | Path
PROFILE_NAME = "m5_v0"
ATTACHMENT_ROI_DILATION_PIXELS = 5
REGION_SCOPES = ("all", *MATCHED_REGIONS)
MINIMUM_GROUPS_FOR_SPEARMAN = 3


@dataclass(frozen=True)
class SignalDefinition:
    """Frozen public metadata for one current-observation signal."""

    name: str
    family: str
    spatial_scope: str
    higher_value_interpretation: str
    method_scope: str = "all"


SIGNAL_DEFINITIONS = (
    SignalDefinition(
        "predictive_entropy_global",
        "perception_uncertainty",
        "full_preprocessed_image",
        "more predictive uncertainty",
    ),
    SignalDefinition(
        "max_probability_global",
        "perception_confidence",
        "full_preprocessed_image",
        "more conventional probability confidence",
    ),
    SignalDefinition(
        "probability_margin_global",
        "perception_confidence",
        "full_preprocessed_image",
        "larger top-one versus top-two separation",
    ),
    SignalDefinition(
        "predictive_entropy_attachment",
        "perception_uncertainty",
        "predicted_attachment_roi",
        "more predictive uncertainty",
    ),
    SignalDefinition(
        "max_probability_attachment",
        "perception_confidence",
        "predicted_attachment_roi",
        "more conventional probability confidence",
    ),
    SignalDefinition(
        "probability_margin_attachment",
        "perception_confidence",
        "predicted_attachment_roi",
        "larger top-one versus top-two separation",
    ),
    SignalDefinition(
        "v2b_candidate_count",
        "geometry_support",
        "current_v2b_geometry",
        "more searched candidates; risk direction not pre-assumed",
        "v2b",
    ),
    SignalDefinition(
        "v2b_feasible_candidate_count",
        "geometry_support",
        "current_v2b_geometry",
        "more feasible candidate support",
        "v2b",
    ),
    SignalDefinition(
        "v2b_feasible_candidate_fraction",
        "geometry_support",
        "current_v2b_geometry",
        "larger feasible share of the searched interval",
        "v2b",
    ),
    SignalDefinition(
        "v2b_feasible_block_count",
        "geometry_support",
        "current_v2b_geometry",
        "more disjoint feasible blocks; risk direction not pre-assumed",
        "v2b",
    ),
    SignalDefinition(
        "v2b_selected_block_width_pixels",
        "geometry_support",
        "current_v2b_geometry",
        "wider selected feasible coordinate block",
        "v2b",
    ),
    SignalDefinition(
        "v2b_selected_block_relative_size",
        "geometry_support",
        "current_v2b_geometry",
        "larger selected share of all feasible candidates",
        "v2b",
    ),
    SignalDefinition(
        "v2b_selected_block_is_singleton",
        "geometry_support",
        "current_v2b_geometry",
        "one denotes a singleton selected feasible block",
        "v2b",
    ),
    SignalDefinition(
        "v2b_maximum_local_contact_evidence",
        "geometry_support",
        "current_v2b_geometry",
        "stronger maximum local contact evidence",
        "v2b",
    ),
    SignalDefinition(
        "v2b_selected_local_contact_count",
        "geometry_support",
        "current_v2b_geometry",
        "more local contact evidence at the selected coordinate",
        "v2b",
    ),
    SignalDefinition(
        "v2b_selected_contact_evidence_fraction",
        "geometry_support",
        "current_v2b_geometry",
        "selected contact evidence closer to the observed maximum",
        "v2b",
    ),
    SignalDefinition(
        "v2b_selected_local_flesh_count",
        "geometry_support",
        "current_v2b_geometry",
        "more selected local Flesh-band support",
        "v2b",
    ),
    SignalDefinition(
        "v2b_selected_local_calyx_count",
        "geometry_support",
        "current_v2b_geometry",
        "more selected local Calyx-band support",
        "v2b",
    ),
    SignalDefinition(
        "v2b_selected_local_v2a_support_count",
        "geometry_support",
        "current_v2b_geometry",
        "more selected local v2a support",
        "v2b",
    ),
    SignalDefinition(
        "v2b_support_pixel_count",
        "geometry_support",
        "current_v2b_geometry",
        "more total predicted v2a support pixels",
        "v2b",
    ),
)
SIGNAL_NAMES = tuple(definition.name for definition in SIGNAL_DEFINITIONS)
_SIGNAL_BY_NAME = {definition.name: definition for definition in SIGNAL_DEFINITIONS}

IDENTIFIER_FIELDS = (
    "sample_id",
    "group_id",
    "session_id",
    "profile",
    "source_profile",
    "matched_set_id",
    "perturbation_id",
    "severity_fraction",
    "seed",
    "region",
    "method",
)
OUTCOME_FIELDS = (
    "method_status",
    "structured_failure_code",
    "finite_cut_returned",
    "c_clean",
    "c_occ",
    "stability_abs",
    "primary_outcome_defined",
)
ROI_FIELDS = (
    "attachment_roi_definition",
    "attachment_roi_dilation_pixels",
    "attachment_roi_pixel_count",
    "attachment_roi_defined",
    "attachment_roi_status",
)
OBSERVATION_FIELDS = (*IDENTIFIER_FIELDS, *OUTCOME_FIELDS, *ROI_FIELDS, *SIGNAL_NAMES)
GROUP_SUMMARY_FIELDS = (
    "group_id",
    "method",
    "region_scope",
    "signal",
    "signal_family",
    "paired_observation_count",
    "mean_signal",
    "mean_stability_abs",
)
DESCRIPTIVE_FIELDS = (
    "method",
    "region_scope",
    "signal",
    "signal_family",
    "spatial_scope",
    "n_groups_total",
    "n_groups_with_signal",
    "observation_count",
    "defined_signal_observation_count",
    "missing_signal_observation_count",
    "mean_group_mean_signal",
    "median_group_mean_signal",
    "minimum_group_mean_signal",
    "maximum_group_mean_signal",
)
ASSOCIATION_FIELDS = (
    "method",
    "region_scope",
    "signal",
    "signal_family",
    "spatial_scope",
    "higher_value_interpretation",
    "n_groups_total",
    "n_groups_with_defined_outcome",
    "n_groups_pairwise_complete",
    "pairwise_observation_count",
    "missing_signal_observation_count",
    "missing_outcome_observation_count",
    "spearman_rho",
    "undefined_reason",
)


def predictive_entropy(probabilities: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return per-pixel categorical entropy in natural-log units."""

    values = _validated_probability_array(probabilities)
    terms = np.zeros_like(values, dtype=np.float64)
    positive = values > 0.0
    terms[positive] = values[positive] * np.log(values[positive])
    entropy = -terms.sum(axis=0)
    entropy.setflags(write=False)
    return entropy


def probability_margin(probabilities: torch.Tensor | np.ndarray) -> np.ndarray:
    """Return the per-pixel top-one minus top-two softmax probability margin."""

    values = _validated_probability_array(probabilities)
    ordered = np.sort(values, axis=0)
    margin = ordered[-1] - ordered[-2]
    margin.setflags(write=False)
    return margin


def derive_predicted_attachment_roi(
    v2a_result: FixedAxisCutlineResult,
    *,
    dilation_radius_pixels: int = ATTACHMENT_ROI_DILATION_PIXELS,
) -> np.ndarray:
    """Dilate the current prediction's v2a contact mask by a square radius.

    The caller must supply the v2a result produced from the same current
    predicted mask as the probabilities being aggregated.  No manual mask,
    clean counterpart, or future observation enters this operation.
    """

    if not isinstance(v2a_result, FixedAxisCutlineResult):
        raise TypeError("v2a_result must be a FixedAxisCutlineResult")
    if (
        isinstance(dilation_radius_pixels, bool)
        or not isinstance(dilation_radius_pixels, int)
        or dilation_radius_pixels < 0
    ):
        raise ValueError("dilation_radius_pixels must be a non-negative integer")
    contact = np.asarray(v2a_result.contact_mask, dtype=bool)
    if contact.shape != (v2a_result.image_height, v2a_result.image_width):
        raise ValueError("v2a contact-mask dimensions must match its result dimensions")
    roi = _square_dilation(contact, radius=dilation_radius_pixels)
    roi.setflags(write=False)
    return roi


def aggregate_probability_signals(
    probabilities: torch.Tensor | np.ndarray,
    attachment_roi: np.ndarray,
) -> dict[str, float | None]:
    """Aggregate global and predicted-attachment probability signals."""

    values = _validated_probability_array(probabilities)
    roi = np.asarray(attachment_roi)
    if roi.dtype != np.bool_ or roi.shape != values.shape[1:]:
        raise ValueError("attachment_roi must be a boolean [H, W] array")
    entropy = predictive_entropy(values)
    maximum = values.max(axis=0)
    margin = probability_margin(values)
    return {
        "predictive_entropy_global": float(entropy.mean()),
        "max_probability_global": float(maximum.mean()),
        "probability_margin_global": float(margin.mean()),
        "predictive_entropy_attachment": _masked_array_mean(entropy, roi),
        "max_probability_attachment": _masked_array_mean(maximum, roi),
        "probability_margin_attachment": _masked_array_mean(margin, roi),
    }


def geometry_support_signals(
    v2b_result: FixedAxisSearchCutlineResult,
) -> dict[str, float | int | bool | None]:
    """Expose existing current-v2b support diagnostics without changing selection."""

    if not isinstance(v2b_result, FixedAxisSearchCutlineResult):
        raise TypeError("v2b_result must be a FixedAxisSearchCutlineResult")
    selected_width = (
        float(v2b_result.selected_block_end - v2b_result.selected_block_start)
        if v2b_result.selected_block_start is not None
        and v2b_result.selected_block_end is not None
        else None
    )
    return {
        "v2b_candidate_count": v2b_result.candidate_count,
        "v2b_feasible_candidate_count": v2b_result.feasible_candidate_count,
        "v2b_feasible_candidate_fraction": _ratio_or_none(
            v2b_result.feasible_candidate_count,
            v2b_result.candidate_count,
        ),
        "v2b_feasible_block_count": v2b_result.feasible_block_count,
        "v2b_selected_block_width_pixels": selected_width,
        "v2b_selected_block_relative_size": _ratio_or_none(
            v2b_result.selected_block_candidate_count,
            v2b_result.feasible_candidate_count,
        ),
        "v2b_selected_block_is_singleton": v2b_result.selected_block_is_singleton,
        "v2b_maximum_local_contact_evidence": (
            v2b_result.maximum_local_contact_evidence
        ),
        "v2b_selected_local_contact_count": v2b_result.selected_local_contact_count,
        "v2b_selected_contact_evidence_fraction": _ratio_or_none(
            v2b_result.selected_local_contact_count,
            v2b_result.maximum_local_contact_evidence,
        ),
        "v2b_selected_local_flesh_count": v2b_result.selected_local_flesh_count,
        "v2b_selected_local_calyx_count": v2b_result.selected_local_calyx_count,
        "v2b_selected_local_v2a_support_count": (
            v2b_result.selected_local_v2a_support_count
        ),
        "v2b_support_pixel_count": int(np.count_nonzero(v2b_result.v2a_support_mask)),
    }


def extract_inference_time_signals(
    probabilities: torch.Tensor | np.ndarray,
    v2a_result: FixedAxisCutlineResult,
    v2b_result: FixedAxisSearchCutlineResult,
    *,
    attachment_roi_dilation_pixels: int = ATTACHMENT_ROI_DILATION_PIXELS,
) -> dict[str, Any]:
    """Extract signals using only one current model observation and its geometry."""

    roi = derive_predicted_attachment_roi(
        v2a_result,
        dilation_radius_pixels=attachment_roi_dilation_pixels,
    )
    roi_count = int(np.count_nonzero(roi))
    return {
        "attachment_roi_definition": (
            "square_dilation_of_current_predicted_v2a_contact_mask"
        ),
        "attachment_roi_dilation_pixels": attachment_roi_dilation_pixels,
        "attachment_roi_pixel_count": roi_count,
        "attachment_roi_defined": roi_count > 0,
        "attachment_roi_status": (
            "defined" if roi_count else "empty_predicted_attachment_contact_roi"
        ),
        **aggregate_probability_signals(probabilities, roi),
        **geometry_support_signals(v2b_result),
    }


def analyze_action_failure_observations(
    observation_rows: Sequence[Mapping[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate within group before computing descriptive rank associations."""

    rows = [dict(row) for row in observation_rows]
    group_rows: list[dict[str, Any]] = []
    descriptives: list[dict[str, Any]] = []
    associations: list[dict[str, Any]] = []
    for method in METHODS:
        for region_scope in REGION_SCOPES:
            selected = [
                row
                for row in rows
                if str(row.get("method")) == method
                and (region_scope == "all" or row.get("region") == region_scope)
            ]
            if not selected:
                continue
            total_groups = {str(row["group_id"]) for row in selected}
            for definition in SIGNAL_DEFINITIONS:
                if definition.method_scope not in ("all", method):
                    continue
                signal_groups: dict[str, list[float]] = defaultdict(list)
                paired_groups: dict[str, list[tuple[float, float]]] = defaultdict(list)
                groups_with_outcome: set[str] = set()
                defined_signal_count = 0
                missing_outcome_count = 0
                pairwise_count = 0
                for row in selected:
                    group_id = str(row["group_id"])
                    signal_value = _numeric_signal(row.get(definition.name))
                    outcome = _finite_or_none(row.get("stability_abs"))
                    if signal_value is not None:
                        signal_groups[group_id].append(signal_value)
                        defined_signal_count += 1
                    if outcome is not None:
                        groups_with_outcome.add(group_id)
                    else:
                        missing_outcome_count += 1
                    if signal_value is not None and outcome is not None:
                        paired_groups[group_id].append((signal_value, outcome))
                        pairwise_count += 1
                signal_group_means = [
                    float(np.mean(values)) for values in signal_groups.values()
                ]
                descriptives.append(
                    {
                        "method": method,
                        "region_scope": region_scope,
                        "signal": definition.name,
                        "signal_family": definition.family,
                        "spatial_scope": definition.spatial_scope,
                        "n_groups_total": len(total_groups),
                        "n_groups_with_signal": len(signal_groups),
                        "observation_count": len(selected),
                        "defined_signal_observation_count": defined_signal_count,
                        "missing_signal_observation_count": (
                            len(selected) - defined_signal_count
                        ),
                        "mean_group_mean_signal": _array_stat(
                            signal_group_means, np.mean
                        ),
                        "median_group_mean_signal": _array_stat(
                            signal_group_means, np.median
                        ),
                        "minimum_group_mean_signal": _array_stat(
                            signal_group_means, np.min
                        ),
                        "maximum_group_mean_signal": _array_stat(
                            signal_group_means, np.max
                        ),
                    }
                )
                current_group_rows = []
                for group_id in sorted(paired_groups):
                    pairs = paired_groups[group_id]
                    summary_row = {
                        "group_id": group_id,
                        "method": method,
                        "region_scope": region_scope,
                        "signal": definition.name,
                        "signal_family": definition.family,
                        "paired_observation_count": len(pairs),
                        "mean_signal": float(
                            np.mean([pair[0] for pair in pairs], dtype=np.float64)
                        ),
                        "mean_stability_abs": float(
                            np.mean([pair[1] for pair in pairs], dtype=np.float64)
                        ),
                    }
                    group_rows.append(summary_row)
                    current_group_rows.append(summary_row)
                rho, undefined_reason = _spearman_from_group_rows(current_group_rows)
                associations.append(
                    {
                        "method": method,
                        "region_scope": region_scope,
                        "signal": definition.name,
                        "signal_family": definition.family,
                        "spatial_scope": definition.spatial_scope,
                        "higher_value_interpretation": (
                            definition.higher_value_interpretation
                        ),
                        "n_groups_total": len(total_groups),
                        "n_groups_with_defined_outcome": len(groups_with_outcome),
                        "n_groups_pairwise_complete": len(current_group_rows),
                        "pairwise_observation_count": pairwise_count,
                        "missing_signal_observation_count": (
                            len(selected) - defined_signal_count
                        ),
                        "missing_outcome_observation_count": missing_outcome_count,
                        "spearman_rho": rho,
                        "undefined_reason": undefined_reason,
                    }
                )
    return {
        "per_group_signal_summary": group_rows,
        "signal_descriptive_statistics": descriptives,
        "association_table": associations,
    }


def run_action_failure_signals(
    development_manifest: PathLike,
    safe_data_root: PathLike,
    checkpoint_path: PathLike,
    output_root: PathLike,
    *,
    device: str | torch.device,
    profile: str = PROFILE_NAME,
    overwrite: bool = False,
    model_factory: Any = None,
) -> dict[str, Any]:
    """Rerun frozen M4 inference and write the preliminary M5-v0 analysis."""

    _require_profile(profile)
    destination = Path(output_root)
    _validate_output_destination(destination, overwrite=overwrite)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    observation_rows: list[dict[str, Any]] = []

    def collect_observation(
        *,
        sample: Any,
        severity_fraction: float,
        seed: int,
        region: str,
        matched_set_id: str,
        perturbation_id: str,
        prediction: torch.Tensor,
        probabilities: torch.Tensor | None,
        v2a_result: FixedAxisCutlineResult,
        v2b_result: FixedAxisSearchCutlineResult,
        method_rows: Mapping[str, Mapping[str, Any]],
    ) -> None:
        del prediction
        if probabilities is None:
            raise ValueError(
                "m5_v0 requires current-observation softmax probabilities; "
                "the selected inference function did not expose them"
            )
        inference_signals = extract_inference_time_signals(
            probabilities,
            v2a_result,
            v2b_result,
        )
        for method in METHODS:
            source = method_rows[method]
            geometry = {
                name: inference_signals[name] if method == "v2b" else None
                for name in SIGNAL_NAMES
                if _SIGNAL_BY_NAME[name].family == "geometry_support"
            }
            stability = _finite_or_none(source.get("stability_abs"))
            observation_rows.append(
                {
                    "sample_id": sample.sample_id,
                    "group_id": sample.group_id,
                    "session_id": sample.session_id,
                    "profile": PROFILE_NAME,
                    "source_profile": M4_PROFILE_NAME,
                    "matched_set_id": matched_set_id,
                    "perturbation_id": perturbation_id,
                    "severity_fraction": severity_fraction,
                    "seed": seed,
                    "region": region,
                    "method": method,
                    "method_status": source.get("method_status"),
                    "structured_failure_code": source.get("structured_failure_code"),
                    "finite_cut_returned": _as_bool(source.get("finite_cut_returned")),
                    "c_clean": _finite_or_none(source.get("c_clean")),
                    "c_occ": _finite_or_none(source.get("c_occ")),
                    "stability_abs": stability,
                    "primary_outcome_defined": stability is not None,
                    **{field: inference_signals[field] for field in ROI_FIELDS},
                    **{
                        name: inference_signals[name]
                        for name in SIGNAL_NAMES
                        if _SIGNAL_BY_NAME[name].family != "geometry_support"
                    },
                    **geometry,
                }
            )

    try:
        source_result = run_occlusion_robustness(
            development_manifest,
            safe_data_root,
            checkpoint_path,
            staging / "m4_reference",
            device=device,
            profile=M4_PROFILE_NAME,
            overwrite=False,
            visualization_sample_ids=(),
            model_factory=model_factory,
            observation_callback=collect_observation,
        )
        analysis = analyze_action_failure_observations(observation_rows)
        manifest, summary = _write_action_failure_artifacts(
            staging,
            observation_rows,
            analysis,
            source_manifest=source_result["manifest"],
        )
        _install_staged_directory(staging, destination, overwrite=overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"manifest": manifest, "summary": summary, "analysis": analysis}


def summarize_existing_action_failure_output(
    output_root: PathLike,
    *,
    profile: str = PROFILE_NAME,
) -> dict[str, Any]:
    """Regenerate deterministic group-aware M5 summaries from observation rows."""

    _require_profile(profile)
    root = Path(output_root)
    manifest_path = root / "manifest.json"
    source_manifest_path = root / "m4_reference" / "manifest.json"
    observation_path = root / "per_observation_signals.csv"
    for path in (manifest_path, source_manifest_path, observation_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required M5 artifact does not exist: {path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("profile") != PROFILE_NAME:
        raise ValueError("output root is not an m5_v0 result")
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    rows = _read_csv(observation_path)
    analysis = analyze_action_failure_observations(rows)
    _, summary = _write_action_failure_artifacts(
        root,
        rows,
        analysis,
        source_manifest=source_manifest,
        rewrite_manifest=False,
        write_observation_rows=False,
    )
    return {"summary": summary, "analysis": analysis}


def _write_action_failure_artifacts(
    root: Path,
    observation_rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Sequence[Mapping[str, Any]]],
    *,
    source_manifest: Mapping[str, Any],
    rewrite_manifest: bool = True,
    write_observation_rows: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from strawberry_occlusion.visualization.action_failure_signals import (
        create_action_failure_signal_plots,
    )

    if write_observation_rows:
        _write_observation_csv(
            root / "per_observation_signals.csv",
            observation_rows,
        )
    _write_csv(
        root / "per_group_signal_summary.csv",
        GROUP_SUMMARY_FIELDS,
        analysis["per_group_signal_summary"],
    )
    _write_csv(
        root / "signal_descriptive_statistics.csv",
        DESCRIPTIVE_FIELDS,
        analysis["signal_descriptive_statistics"],
    )
    _write_csv(
        root / "association_table.csv",
        ASSOCIATION_FIELDS,
        analysis["association_table"],
    )
    plots = create_action_failure_signal_plots(analysis, root / "plots")
    summary = _summary_payload(observation_rows, analysis, source_manifest)
    _assert_no_absolute_paths(summary)
    _write_json(root / "summary.json", summary)
    association_count = len(analysis["association_table"])
    configuration = frozen_m5_configuration(
        n_association_coefficients=association_count
    )
    _assert_no_absolute_paths(configuration)
    _write_json(root / "configuration.json", configuration)
    manifest = {
        "study": "Action-Aware Failure Signals Under Occlusion",
        "profile": PROFILE_NAME,
        "role": "preliminary_development_analysis",
        "source_profile": M4_PROFILE_NAME,
        "source_m4_counts": {
            key: source_manifest.get(key)
            for key in (
                "n_groups",
                "n_images",
                "n_sessions",
                "perturbation_evaluations",
            )
        },
        "m5_groups_represented": summary["counts"]["n_groups_represented"],
        "m5_perturbations_represented": summary["counts"][
            "n_perturbations_represented"
        ],
        "statistical_unit": "group_id",
        "perturbation_rows_are_statistical_n": False,
        "primary_outcome": "abs(c_occ - c_clean)",
        "calibrated_deployment_policy": False,
        "training_or_optimization_performed": False,
        "artifacts": {
            "configuration": "configuration.json",
            "per_observation_signals": "per_observation_signals.csv",
            "per_group_signal_summary": "per_group_signal_summary.csv",
            "signal_descriptive_statistics": "signal_descriptive_statistics.csv",
            "association_table": "association_table.csv",
            "summary": "summary.json",
            "plots": plots,
            "m4_reference": "m4_reference",
        },
    }
    _assert_no_absolute_paths(manifest)
    if rewrite_manifest:
        _write_json(root / "manifest.json", manifest)
    return manifest, summary


def frozen_m5_configuration(
    *,
    n_association_coefficients: int | None = None,
) -> dict[str, Any]:
    """Return the public, non-tunable M5-v0 configuration."""

    return {
        "profile": PROFILE_NAME,
        "source_benchmark": benchmark_profile(M4_PROFILE_NAME),
        "research_stage": "preliminary_development_analysis",
        "primary_outcome": "abs(c_occ - c_clean)",
        "inference_time_signal_inputs": [
            "current_model_probabilities",
            "current_predicted_mask_geometry",
        ],
        "prohibited_primary_signal_inputs": [
            "ground_truth_mask",
            "manual_mask_geometry",
            "clean_counterpart",
            "future_observation",
        ],
        "global_aggregation": "arithmetic mean over the full preprocessed image",
        "attachment_roi": {
            "source": "current predicted v2a contact mask",
            "dilation": "square/Chebyshev dilation",
            "dilation_radius_pixels": ATTACHMENT_ROI_DILATION_PIXELS,
            "empty_behavior": "localized signals are undefined and retained as missing",
        },
        "entropy": "natural-log categorical entropy -sum_c p_c log(p_c)",
        "probability_margin": "top-one minus top-two softmax probability",
        "signals": [definition.__dict__ for definition in SIGNAL_DEFINITIONS],
        "statistics": {
            "independent_unit": "group_id",
            "within_group_aggregation": (
                "arithmetic mean over pairwise-complete repeated perturbations"
            ),
            "association": "Spearman rank correlation of group means",
            "minimum_groups_for_reported_rho": MINIMUM_GROUPS_FOR_SPEARMAN,
            "n_association_coefficients": n_association_coefficients,
            "multiplicity_correction": None,
            "association_interpretation": "descriptive_not_confirmatory",
            "row_level_inference": False,
            "bootstrap": None,
        },
        "selective_action_analysis": "deferred_to_m5_v1",
        "training_or_optimization": False,
        "calibrated_policy": False,
    }


def _summary_payload(
    rows: Sequence[Mapping[str, Any]],
    analysis: Mapping[str, Sequence[Mapping[str, Any]]],
    source_manifest: Mapping[str, Any],
) -> dict[str, Any]:
    groups = {str(row["group_id"]) for row in rows}
    perturbations = {str(row["perturbation_id"]) for row in rows}
    missing_signals = {}
    for definition in SIGNAL_DEFINITIONS:
        applicable = [
            row
            for row in rows
            if definition.method_scope == "all"
            or row.get("method") == definition.method_scope
        ]
        defined = sum(
            _numeric_signal(row.get(definition.name)) is not None for row in applicable
        )
        missing_signals[definition.name] = {
            "applicable_observation_rows": len(applicable),
            "defined_observation_rows": defined,
            "missing_observation_rows": len(applicable) - defined,
        }
    return {
        "profile": PROFILE_NAME,
        "interpretation": (
            "Preliminary development-cohort failure-awareness analysis; not a "
            "calibrated deployment or abstention policy."
        ),
        "primary_outcome": "abs(c_occ - c_clean)",
        "independent_unit": "group_id",
        "counts": {
            "development_manifest_groups": source_manifest.get("n_groups"),
            "n_groups_represented": len(groups),
            "n_observation_method_rows": len(rows),
            "n_perturbations_represented": len(perturbations),
            "defined_primary_outcome_rows": sum(
                _finite_or_none(row.get("stability_abs")) is not None for row in rows
            ),
            "missing_primary_outcome_rows": sum(
                _finite_or_none(row.get("stability_abs")) is None for row in rows
            ),
            "structured_failure_rows": sum(
                str(row.get("method_status")) != "ok" for row in rows
            ),
        },
        "missing_signals": missing_signals,
        "n_association_coefficients": len(analysis["association_table"]),
        "claims": {
            "calibrated_confidence": False,
            "calibrated_failure_probability": False,
            "deployment_readiness": False,
            "generalization_beyond_current_development_cohort": False,
        },
    }


def _validated_probability_array(
    probabilities: torch.Tensor | np.ndarray,
) -> np.ndarray:
    if isinstance(probabilities, torch.Tensor):
        values = probabilities.detach().cpu().numpy()
    else:
        values = np.asarray(probabilities)
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] != 3:
        raise ValueError("probabilities must have shape [3, H, W]")
    if values.shape[1] == 0 or values.shape[2] == 0:
        raise ValueError("probability height and width must be greater than zero")
    if not np.all(np.isfinite(values)) or np.any((values < 0.0) | (values > 1.0)):
        raise ValueError("probabilities must contain finite values in [0, 1]")
    if not np.allclose(values.sum(axis=0), 1.0, atol=1e-5, rtol=1e-5):
        raise ValueError("probabilities must sum to one at every pixel")
    return values


def _write_observation_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write irreplaceable M5 observations with exact float round-trip precision."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=OBSERVATION_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    field: _observation_csv_value(row.get(field))
                    for field in OBSERVATION_FIELDS
                }
            )


def _observation_csv_value(value: Any) -> Any:
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return format(numeric, ".17g") if math.isfinite(numeric) else ""
    return _csv_value(value)


def _masked_array_mean(values: np.ndarray, mask: np.ndarray) -> float | None:
    selected = values[mask]
    return float(selected.mean()) if selected.size else None


def _ratio_or_none(numerator: int, denominator: int) -> float | None:
    return float(numerator / denominator) if denominator > 0 else None


def _numeric_signal(value: Any) -> float | None:
    if isinstance(value, (bool, np.bool_)):
        return float(value)
    normalized = str(value).strip().lower()
    if normalized in {"true", "false"}:
        return float(normalized == "true")
    return _finite_or_none(value)


def _array_stat(values: Sequence[float], function: Any) -> float | None:
    return float(function(np.asarray(values, dtype=np.float64))) if values else None


def _spearman_from_group_rows(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[float | None, str | None]:
    if len(rows) < MINIMUM_GROUPS_FOR_SPEARMAN:
        return None, "fewer_than_3_pairwise_complete_groups"
    signals = np.asarray([row["mean_signal"] for row in rows], dtype=np.float64)
    outcomes = np.asarray([row["mean_stability_abs"] for row in rows], dtype=np.float64)
    if np.all(signals == signals[0]):
        return None, "constant_group_mean_signal"
    if np.all(outcomes == outcomes[0]):
        return None, "constant_group_mean_outcome"
    signal_ranks = _average_ranks(signals)
    outcome_ranks = _average_ranks(outcomes)
    rho = float(np.corrcoef(signal_ranks, outcome_ranks)[0, 1])
    return rho, None


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=np.float64)
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = (start + end - 1) / 2.0 + 1.0
        start = end
    return ranks


def _require_profile(profile: str) -> None:
    if profile != PROFILE_NAME:
        raise ValueError(f"Only frozen profile {PROFILE_NAME!r} is supported")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run preliminary M5-v0 action-instability signal analysis."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    run = subparsers.add_parser("run")
    run.add_argument("--profile", choices=(PROFILE_NAME,), required=True)
    run.add_argument("--development-manifest", type=Path, required=True)
    run.add_argument("--safe-data-root", type=Path, required=True)
    run.add_argument("--checkpoint", type=Path, required=True)
    run.add_argument("--output-root", type=Path, required=True)
    run.add_argument("--device", required=True)
    run.add_argument("--overwrite", action="store_true")
    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--profile", choices=(PROFILE_NAME,), required=True)
    summarize.add_argument("--output-root", type=Path, required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the M5-v0 command-line workflow."""

    arguments = _build_argument_parser().parse_args(argv)
    if arguments.command == "summarize":
        result = summarize_existing_action_failure_output(
            arguments.output_root,
            profile=arguments.profile,
        )
        counts = result["summary"]["counts"]
        print(
            "Regenerated group-aware M5-v0 summaries for "
            f"{counts['n_groups_represented']} represented groups."
        )
        return 0
    result = run_action_failure_signals(
        arguments.development_manifest,
        arguments.safe_data_root,
        arguments.checkpoint,
        arguments.output_root,
        device=arguments.device,
        profile=arguments.profile,
        overwrite=arguments.overwrite,
    )
    counts = result["summary"]["counts"]
    print(
        f"Completed {PROFILE_NAME}: {counts['n_groups_represented']} represented "
        "groups; group_id is the independent unit."
    )
    print("This preliminary analysis is not a calibrated deployment policy.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
