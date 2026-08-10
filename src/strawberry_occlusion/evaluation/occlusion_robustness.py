"""Milestone 4 occlusion-induced action stability and failure awareness.

The canonical ``m4_v1`` workflow is inference-only. It reads only image and mask
paths explicitly named by a development-manifest CSV, constructs matched opaque
synthetic occluders from the clean manual mask, and keeps repeated perturbations
separate from the independent ``group_id`` analysis unit.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import math
import re
import shutil
import tempfile
import uuid
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from statistics import mean
from types import MappingProxyType
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn

from strawberry_occlusion.augmentation.synthetic_occluder import (
    MATCHED_REGIONS,
    MatchedOccluderSet,
    apply_occluder,
    build_matched_occluder_set,
    derive_attachment_roi,
    generate_synthetic_occluder,
)
from strawberry_occlusion.evaluation.pilot_holdout import (
    EXPECTED_CANONICAL_CHECKPOINT_EPOCH,
    FROZEN_FIXED_AXIS_CONFIGURATION,
    _run_frozen_geometry,
)
from strawberry_occlusion.evaluation.segmentation import (
    ModelFactory,
    _model_logits,
    _validate_logits,
    confidence_and_entropy,
    load_segmentation_checkpoint,
    metrics_from_confusion_matrix,
    preprocess_class_mask,
    preprocess_rgb_image,
    resolve_device,
)
from strawberry_occlusion.geometry import (
    FixedAxisCutlineResult,
    FixedAxisSearchCutlineResult,
    estimate_fixed_axis_cutline,
    estimate_fixed_axis_search_cutline,
)
from strawberry_occlusion.metrics import confusion_matrix
from strawberry_occlusion.visualization.occlusion_robustness import (
    create_matched_quartet_visualization,
    create_occlusion_robustness_plots,
)


PathLike = str | Path
GeometryResult = FixedAxisCutlineResult | FixedAxisSearchCutlineResult
InferenceFunction = Callable[..., tuple[torch.Tensor, torch.Tensor]]
V2aEstimator = Callable[..., FixedAxisCutlineResult]
V2bEstimator = Callable[..., FixedAxisSearchCutlineResult]

PROFILE_NAME = "m4_v1"
PROFILE_SEVERITIES = (0.01, 0.02, 0.04)
PROFILE_SEEDS = (0, 1)
SILENT_DRIFT_THRESHOLDS = (2, 5, 10)
BOOTSTRAP_SEED = 410_204
BOOTSTRAP_REPLICATES = 2_000
MINIMUM_STABLE_INTERVAL_GROUPS = 5
M4_V1_PLACEMENT_PARAMETERS = MappingProxyType(
    {
        "attachment_roi_dilation_pixels": 5,
        "background_foreground_separation_pixels": 2,
    }
)
COORDINATE_STABILITY_CONDITIONING = (
    "Coordinate-stability summaries are conditional on successful finite cut "
    "estimates and must be interpreted jointly with structured-failure rates."
)
SEVERITY_COVERAGE_CONDITIONING = (
    "AUC and bootstrap analyses requiring all three severity levels are "
    "conditional on groups with complete required severity coverage; missing "
    "perturbations are not imputed."
)
REQUIRED_MANIFEST_FIELDS = (
    "sample_id",
    "group_id",
    "session_id",
    "role",
    "image_path",
    "mask_path",
    "orientation_category",
    "fixed_axis_eligible",
    "annotation_qa_status",
)
FORBIDDEN_ROLES = frozenset({"test", "final_test", "locked_test"})
FORBIDDEN_PATH_COMPONENTS = frozenset({"final_test", "locked_test"})
ACCEPTED_QA_STATUSES = frozenset(
    {
        "accepted",
        "approved",
        "complete",
        "passed",
        "qa_passed",
        "reviewed",
        "reviewed_passed",
        "verified",
    }
)
METHODS = ("v2a", "v2b")
COMPARISON_REGIONS = ("background", "flesh_far", "calyx_tip")
SUMMARY_OUTCOMES = (
    "stability_abs",
    "gt_reference_abs",
    "excess_gt_reference_error",
    "structured_failure_rate",
    "silent_drift_2_rate",
    "silent_drift_5_rate",
    "silent_drift_10_rate",
)


@dataclass(frozen=True)
class DevelopmentSample:
    """One explicitly listed, validated manifest row."""

    sample_id: str
    group_id: str
    session_id: str
    role: str
    image_path: Path
    mask_path: Path
    orientation_category: str
    fixed_axis_eligible: bool
    annotation_qa_status: str
    resolved_image_path: Path
    resolved_mask_path: Path

    def snapshot_row(self) -> dict[str, Any]:
        """Return sanitized metadata containing relative paths only."""

        return {
            "sample_id": self.sample_id,
            "group_id": self.group_id,
            "session_id": self.session_id,
            "role": self.role,
            "image_path": self.image_path.as_posix(),
            "mask_path": self.mask_path.as_posix(),
            "orientation_category": self.orientation_category,
            "fixed_axis_eligible": self.fixed_axis_eligible,
            "annotation_qa_status": self.annotation_qa_status,
        }


TRIAL_FIELDS = (
    "sample_id",
    "group_id",
    "session_id",
    "orientation_category",
    "profile",
    "matched_set_id",
    "perturbation_id",
    "matched_set_complete",
    "severity_fraction",
    "seed",
    "region",
    "method",
    "occluder_id",
    "occluder_template_sha256",
    "occluder_area_pixels",
    "occluder_opacity",
    "occluder_rotation_degrees",
    "occluder_scale_pixels",
    "placement_top",
    "placement_left",
    "placed_mask_sha256",
    "occluder_attachment_roi_overlap_pixels",
    "occluder_foreground_overlap_pixels",
    "c_gt",
    "c_clean",
    "c_occ",
    "gt_status",
    "gt_failure_code",
    "clean_status",
    "clean_failure_code",
    "method_status",
    "structured_failure_code",
    "finite_cut_returned",
    "stability_signed",
    "stability_abs",
    "gt_reference_signed",
    "gt_reference_abs",
    "clean_gt_reference_abs",
    "excess_gt_reference_error",
    "silent_drift_2_pixels",
    "silent_drift_5_pixels",
    "silent_drift_10_pixels",
    "predicted_calyx_component_count",
    "attachment_contact_support_pixel_count",
    "support_pixel_count",
    "candidate_count",
    "feasible_candidate_count",
    "feasible_block_count",
    "selected_block_id",
    "selected_block_start",
    "selected_block_end",
    "selected_block_candidate_count",
    "selected_block_is_singleton",
    "selected_local_contact_count",
    "selected_local_flesh_count",
    "selected_local_calyx_count",
    "mean_normalized_entropy_global",
    "mean_normalized_entropy_attachment_roi",
    "mean_normalized_entropy_occluder",
    "prediction_inside_occluder_background_pixels",
    "prediction_inside_occluder_flesh_pixels",
    "prediction_inside_occluder_calyx_pixels",
    "visible_evaluated_pixel_count",
    "iou_background",
    "iou_flesh",
    "iou_calyx",
    "dice_background",
    "dice_flesh",
    "dice_calyx",
    "precision_background",
    "precision_flesh",
    "precision_calyx",
    "recall_background",
    "recall_flesh",
    "recall_calyx",
    "mean_foreground_iou",
    "mean_foreground_dice",
    "pixel_accuracy",
)

CLEAN_REFERENCE_FIELDS = (
    "sample_id",
    "group_id",
    "session_id",
    "fixed_axis_eligible",
    "method",
    "c_gt",
    "c_clean",
    "gt_status",
    "gt_failure_code",
    "clean_status",
    "clean_failure_code",
    "clean_gt_reference_abs",
    "clean_mean_normalized_entropy",
    "clean_mean_normalized_entropy_attachment_roi",
    "clean_iou_background",
    "clean_iou_flesh",
    "clean_iou_calyx",
    "clean_dice_background",
    "clean_dice_flesh",
    "clean_dice_calyx",
)

INCOMPLETE_FIELDS = (
    "sample_id",
    "group_id",
    "session_id",
    "profile",
    "matched_set_id",
    "severity_fraction",
    "seed",
    "occluder_id",
    "occluder_template_sha256",
    "occluder_area_pixels",
    "missing_regions",
    "attachment_reason",
    "calyx_tip_reason",
    "flesh_far_reason",
    "background_reason",
)

MATCHED_SET_COMPLETENESS_FIELDS = (
    "summary_level",
    "region",
    "severity_fraction",
    "attempted_matched_sets",
    "successfully_placed",
    "unavailable",
    "completion_proportion",
    "n_groups_attempted",
    "n_groups_represented_among_successful_placements",
    "structured_unavailable_reason_counts",
    "attempted_matched_quartets",
    "complete_matched_quartets",
    "incomplete_matched_quartets",
    "completion_fraction",
    "n_groups_with_complete_quartet",
)

COUNT_FIELDS = ("n_groups", "n_images", "n_sessions", "perturbation_evaluations")
AUC_FIELDS = tuple(f"{outcome}_auc" for outcome in SUMMARY_OUTCOMES)
PER_IMAGE_SUMMARY_FIELDS = (
    "sample_id",
    "group_id",
    "session_id",
    "method",
    "region",
    *COUNT_FIELDS,
    *AUC_FIELDS,
)
PER_GROUP_SUMMARY_FIELDS = (
    "group_id",
    "method",
    "region",
    *COUNT_FIELDS,
    *AUC_FIELDS,
)
CONDITION_SUMMARY_FIELDS = (
    "method",
    "region",
    "severity_fraction",
    *COUNT_FIELDS,
    "mean_stability_abs",
    "mean_gt_reference_abs",
    "mean_excess_gt_reference_error",
    "structured_failure_rate",
    "silent_drift_2_rate",
    "silent_drift_5_rate",
    "silent_drift_10_rate",
)
PAIRED_FIELDS = (
    "group_id",
    "method",
    "comparison",
    "comparison_priority",
    *COUNT_FIELDS,
    *(
        f"{outcome}_auc_difference_attachment_minus_control"
        for outcome in SUMMARY_OUTCOMES
    ),
)
BOOTSTRAP_FIELDS = (
    "method",
    "comparison",
    "comparison_priority",
    "outcome",
    *COUNT_FIELDS,
    "mean_paired_difference",
    "confidence_level",
    "lower_bound",
    "upper_bound",
    "benchmark_bootstrap_seed",
    "bootstrap_seed",
    "bootstrap_replicates",
    "stable_interval",
    "interval_note",
)
FAILURE_SUMMARY_FIELDS = (
    "method",
    "region",
    "severity_fraction",
    *COUNT_FIELDS,
    "structured_failure_rate",
    "finite_cut_rate",
    "silent_drift_2_rate",
    "silent_drift_5_rate",
    "silent_drift_10_rate",
)
SEGMENTATION_SUMMARY_FIELDS = (
    "region",
    "severity_fraction",
    *COUNT_FIELDS,
    "mean_iou_background",
    "mean_iou_flesh",
    "mean_iou_calyx",
    "mean_dice_background",
    "mean_dice_flesh",
    "mean_dice_calyx",
    "mean_foreground_iou",
    "mean_foreground_dice",
)


def benchmark_profile(profile: str = PROFILE_NAME) -> dict[str, Any]:
    """Return the public frozen Milestone 4 profile."""

    _require_profile(profile)
    return {
        "profile": PROFILE_NAME,
        "profile_version": 1,
        "regions": list(MATCHED_REGIONS),
        "severity_fractions": list(PROFILE_SEVERITIES),
        "severity_definition": (
            "Exact opaque template pixel area divided by clean visible "
            "Flesh+Calyx foreground area after nearest-neighbour preprocessing; "
            "the integer target is round(fraction * foreground pixels), minimum one"
        ),
        "seeds": list(PROFILE_SEEDS),
        "occluder": {
            "level": 1,
            "appearance": "deterministic tapered leaf-like procedural RGB texture",
            "opacity": 1.0,
            "matched_control": (
                "One template per sample/severity/seed; only integer translation "
                "changes across regions"
            ),
        },
        "placement": {
            **copy.deepcopy(dict(M4_V1_PLACEMENT_PARAMETERS)),
            "attachment": "clean manual-mask v2a contact ROI centroid",
            "calyx_tip": "distal selected Calyx support beyond the attachment ROI",
            "flesh_far": (
                "selected Flesh separated inward from attachment; eroded interior "
                "support preferred"
            ),
            "background": (
                "entire template bounding box on background separated from foreground"
            ),
        },
        "geometry": copy.deepcopy(FROZEN_FIXED_AXIS_CONFIGURATION),
        "preprocessing": {
            "rgb_resize": "checkpoint resolution with bilinear interpolation",
            "mask_resize": "checkpoint resolution with nearest-neighbour interpolation",
            "coordinate_space": "checkpoint-preprocessed image pixels",
        },
        "segmentation_metric_visibility": (
            "Ordinary metrics exclude every opaque synthetic-occluder pixel"
        ),
        "silent_drift_threshold_pixels": list(SILENT_DRIFT_THRESHOLDS),
        "silent_drift_definition": (
            "method status ok with a finite coordinate and stability_abs greater "
            "than the descriptive analysis threshold"
        ),
        "statistics": {
            "independent_unit": "group_id",
            "perturbations_are_repeated_measurements": True,
            "severity_auc": (
                "Trapezoidal area over group-aggregated low/medium/high values, "
                "with severity rescaled from 0 at 0.01 to 1 at 0.04"
            ),
            "bootstrap_unit": "group_id",
            "bootstrap_seed": BOOTSTRAP_SEED,
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "minimum_groups_for_stable_interval": MINIMUM_STABLE_INTERVAL_GROUPS,
            "row_level_hypothesis_tests": False,
            "severity_coverage_conditioning": SEVERITY_COVERAGE_CONDITIONING,
            "coordinate_stability_conditioning": COORDINATE_STABILITY_CONDITIONING,
        },
        "canonical_checkpoint_epoch": EXPECTED_CANONICAL_CHECKPOINT_EPOCH,
        "accepted_annotation_qa_statuses": sorted(ACCEPTED_QA_STATUSES),
        "training_or_optimization": False,
    }


def validate_development_manifest(
    development_manifest: PathLike,
    safe_data_root: PathLike | None = None,
    *,
    require_qa: bool = True,
) -> list[DevelopmentSample]:
    """Validate an explicit CSV without discovering any additional samples."""

    manifest_path = Path(development_manifest)
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Development manifest does not exist: {manifest_path}")
    root = (
        manifest_path.parent.resolve()
        if safe_data_root is None
        else Path(safe_data_root).resolve()
    )
    if not root.is_dir():
        raise FileNotFoundError(f"Safe data root does not exist: {root}")
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        fieldnames = tuple(reader.fieldnames or ())
        missing_fields = [
            field for field in REQUIRED_MANIFEST_FIELDS if field not in fieldnames
        ]
        if missing_fields:
            raise ValueError(
                "Development manifest is missing required columns: "
                + ", ".join(missing_fields)
            )
        raw_rows = list(reader)
    if not raw_rows:
        raise ValueError("Development manifest must contain at least one row")

    scalar_rows: list[dict[str, Any]] = []
    seen_sample_ids: set[str] = set()
    for row_number, raw in enumerate(raw_rows, start=2):
        row = {
            field: (raw.get(field) or "").strip() for field in REQUIRED_MANIFEST_FIELDS
        }
        sample_id = row["sample_id"]
        _validate_identifier(sample_id, name=f"row {row_number} sample_id")
        if sample_id in seen_sample_ids:
            raise ValueError(f"Duplicate sample_id {sample_id!r}")
        seen_sample_ids.add(sample_id)
        group_id = row["group_id"]
        if not group_id:
            raise ValueError(f"row {row_number} group_id must be non-empty")
        _validate_identifier(group_id, name=f"row {row_number} group_id")
        session_id = row["session_id"]
        if not session_id:
            raise ValueError(f"row {row_number} session_id must be non-empty")
        _validate_identifier(session_id, name=f"row {row_number} session_id")
        role = row["role"].lower()
        if role in FORBIDDEN_ROLES:
            raise ValueError(
                f"row {row_number} role {role!r} is forbidden for Milestone 4"
            )
        if role != "development":
            raise ValueError(
                f"row {row_number} role must equal 'development' for m4_v1"
            )
        fixed_axis_eligible = _parse_strict_boolean(
            row["fixed_axis_eligible"],
            name=f"row {row_number} fixed_axis_eligible",
        )
        qa_status = row["annotation_qa_status"].lower()
        if require_qa and qa_status not in ACCEPTED_QA_STATUSES:
            raise ValueError(
                f"row {row_number} annotation_qa_status is not reviewed/approved"
            )
        if qa_status and not re.fullmatch(r"[a-z0-9_-]+", qa_status):
            raise ValueError(
                f"row {row_number} annotation_qa_status must be a path-free token"
            )
        orientation = row["orientation_category"].lower()
        if not orientation or not re.fullmatch(r"[a-z0-9_-]+", orientation):
            raise ValueError(
                f"row {row_number} orientation_category must be a non-empty token"
            )
        scalar_rows.append(
            {
                **row,
                "role": role,
                "fixed_axis_eligible": fixed_axis_eligible,
                "annotation_qa_status": qa_status,
                "orientation_category": orientation,
                "row_number": row_number,
            }
        )

    # Validate all path strings and boundaries before checking/opening any listed file.
    path_rows: list[dict[str, Any]] = []
    for row in scalar_rows:
        path_values: dict[str, Path] = {}
        resolved_values: dict[str, Path] = {}
        for field in ("image_path", "mask_path"):
            value = str(row[field])
            relative = _validated_relative_path(
                value, context=f"row {row['row_number']} {field}"
            )
            unresolved = root / relative
            resolved = unresolved.resolve()
            try:
                resolved.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"row {row['row_number']} {field} escapes safe-data-root"
                ) from error
            forbidden = sorted(
                component
                for component in (part.lower() for part in resolved.parts)
                if component in FORBIDDEN_PATH_COMPONENTS
            )
            if forbidden:
                raise ValueError(
                    f"row {row['row_number']} {field} resolves inside forbidden "
                    f"directory component {forbidden[0]!r}"
                )
            path_values[field] = relative
            resolved_values[f"resolved_{field}"] = resolved
        path_rows.append({**row, **path_values, **resolved_values})

    for row in path_rows:
        if not row["resolved_image_path"].is_file():
            raise FileNotFoundError(
                f"Listed image does not exist for sample {row['sample_id']!r}"
            )
        if not row["resolved_mask_path"].is_file():
            raise FileNotFoundError(
                f"Listed mask does not exist for sample {row['sample_id']!r}"
            )

    return [
        DevelopmentSample(
            sample_id=str(row["sample_id"]),
            group_id=str(row["group_id"]),
            session_id=str(row["session_id"]),
            role=str(row["role"]),
            image_path=row["image_path"],
            mask_path=row["mask_path"],
            orientation_category=str(row["orientation_category"]),
            fixed_axis_eligible=bool(row["fixed_axis_eligible"]),
            annotation_qa_status=str(row["annotation_qa_status"]),
            resolved_image_path=row["resolved_image_path"],
            resolved_mask_path=row["resolved_mask_path"],
        )
        for row in path_rows
    ]


def masked_segmentation_metrics(
    predicted: torch.Tensor | np.ndarray,
    target: torch.Tensor | np.ndarray,
    occluder_mask: torch.Tensor | np.ndarray,
) -> dict[str, float | int]:
    """Calculate ordinary visible metrics excluding opaque occluder pixels."""

    predicted_tensor = torch.as_tensor(predicted, dtype=torch.long)
    target_tensor = torch.as_tensor(target, dtype=torch.long)
    ignored = torch.as_tensor(occluder_mask, dtype=torch.bool)
    if predicted_tensor.shape != target_tensor.shape or predicted_tensor.ndim != 2:
        raise ValueError("predicted and target must have matching [H, W] shapes")
    if ignored.shape != target_tensor.shape:
        raise ValueError("occluder_mask must match predicted and target shapes")
    visible_target = target_tensor.clone()
    visible_target[ignored] = 255
    matrix = confusion_matrix(
        predicted_tensor,
        visible_target,
        3,
        ignore_index=255,
    )
    metrics: dict[str, float | int] = dict(metrics_from_confusion_matrix(matrix))
    metrics["visible_evaluated_pixel_count"] = int((~ignored).sum().item())
    return metrics


def reference_error_metrics(
    c_gt: float | None,
    c_clean: float | None,
    c_occ: float | None,
) -> dict[str, float | None]:
    """Return the frozen Milestone 4 coordinate-difference conventions."""

    gt = _finite_or_none(c_gt)
    clean = _finite_or_none(c_clean)
    occluded = _finite_or_none(c_occ)
    stability_signed = (
        occluded - clean if occluded is not None and clean is not None else None
    )
    gt_signed = occluded - gt if occluded is not None and gt is not None else None
    clean_gt_abs = abs(clean - gt) if clean is not None and gt is not None else None
    gt_abs = abs(gt_signed) if gt_signed is not None else None
    return {
        "stability_signed": stability_signed,
        "stability_abs": abs(stability_signed)
        if stability_signed is not None
        else None,
        "gt_reference_signed": gt_signed,
        "gt_reference_abs": gt_abs,
        "clean_gt_reference_abs": clean_gt_abs,
        "excess_gt_reference_error": (
            gt_abs - clean_gt_abs
            if gt_abs is not None and clean_gt_abs is not None
            else None
        ),
    }


def silent_drift_flags(
    *,
    method_status: str,
    finite_cut_returned: bool,
    stability_abs: float | None,
) -> dict[str, bool]:
    """Return descriptive 2/5/10-pixel silent-drift proxies."""

    drift = _finite_or_none(stability_abs)
    eligible = method_status == "ok" and finite_cut_returned and drift is not None
    return {
        f"silent_drift_{threshold}_pixels": bool(eligible and drift > threshold)
        for threshold in SILENT_DRIFT_THRESHOLDS
    }


def run_occlusion_robustness(
    development_manifest: PathLike,
    safe_data_root: PathLike,
    checkpoint_path: PathLike,
    output_root: PathLike,
    *,
    device: str | torch.device,
    profile: str = PROFILE_NAME,
    overwrite: bool = False,
    visualization_sample_ids: Sequence[str] = (),
    model_factory: ModelFactory | None = None,
    inference_function: InferenceFunction | None = None,
    v2a_estimator: V2aEstimator = estimate_fixed_axis_cutline,
    v2b_estimator: V2bEstimator = estimate_fixed_axis_search_cutline,
) -> dict[str, Any]:
    """Run the canonical development-only mechanistic robustness benchmark."""

    profile_definition = benchmark_profile(profile)
    placement_parameters = profile_definition["placement"]
    samples = validate_development_manifest(development_manifest, safe_data_root)
    checkpoint_file = Path(checkpoint_path)
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_file}")
    destination = Path(output_root)
    _validate_output_destination(destination, overwrite=overwrite)
    selected_visualizations = _validate_visualization_ids(
        visualization_sample_ids,
        samples,
    )
    resolved_device = resolve_device(device)
    model, checkpoint = load_segmentation_checkpoint(
        checkpoint_file,
        device=resolved_device,
        model_factory=model_factory,
    )
    if checkpoint["completed_epoch"] != EXPECTED_CANONICAL_CHECKPOINT_EPOCH:
        raise ValueError(
            "m4_v1 requires the canonical epoch-92 checkpoint; "
            f"received completed_epoch={checkpoint['completed_epoch']}"
        )
    height = int(checkpoint["input_height"])
    width = int(checkpoint["input_width"])
    infer = inference_function or _infer_segmentation

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{destination.name}.staging-",
            dir=destination.parent,
        )
    )
    try:
        (staging / "plots").mkdir()
        (staging / "visualizations").mkdir()
        clean_rows: list[dict[str, Any]] = []
        trial_rows: list[dict[str, Any]] = []
        incomplete_rows: list[dict[str, Any]] = []
        model.eval()
        with torch.inference_mode():
            for sample in samples:
                clean_image, target = _load_listed_pair(
                    sample,
                    height=height,
                    width=width,
                )
                clean_prediction, clean_entropy = infer(
                    model,
                    clean_image,
                    device=resolved_device,
                )
                clean_prediction = _validated_prediction(
                    clean_prediction,
                    shape=target.shape,
                )
                clean_entropy = _validated_entropy(clean_entropy, shape=target.shape)
                clean_metrics = masked_segmentation_metrics(
                    clean_prediction,
                    target,
                    np.zeros(target.shape, dtype=bool),
                )
                if sample.fixed_axis_eligible:
                    gt_v2a, gt_v2b = _run_frozen_geometry(
                        target,
                        v2a_estimator=v2a_estimator,
                        v2b_estimator=v2b_estimator,
                    )
                    clean_v2a, clean_v2b = _run_frozen_geometry(
                        clean_prediction.numpy(),
                        v2a_estimator=v2a_estimator,
                        v2b_estimator=v2b_estimator,
                    )
                    geometry_by_method = {
                        "v2a": (gt_v2a, clean_v2a),
                        "v2b": (gt_v2b, clean_v2b),
                    }
                    clean_attachment_roi = derive_attachment_roi(
                        gt_v2a,
                        dilation_radius_pixels=int(
                            placement_parameters["attachment_roi_dilation_pixels"]
                        ),
                    )
                else:
                    gt_v2a = gt_v2b = clean_v2a = clean_v2b = None
                    geometry_by_method = {}
                    clean_attachment_roi = None

                for method in METHODS:
                    gt_result, clean_result = geometry_by_method.get(
                        method, (None, None)
                    )
                    clean_rows.append(
                        _clean_reference_row(
                            sample,
                            method,
                            gt_result,
                            clean_result,
                            clean_entropy,
                            clean_metrics,
                            clean_attachment_roi,
                        )
                    )
                if not sample.fixed_axis_eligible:
                    continue
                assert gt_v2a is not None
                foreground_area = int(np.count_nonzero(np.isin(target, (1, 2))))
                if foreground_area == 0:
                    raise ValueError(
                        f"Fixed-axis-eligible sample {sample.sample_id!r} has no foreground"
                    )
                clean_hwc = clean_image.permute(1, 2, 0).cpu().numpy()
                clean_rgb = np.rint(clean_hwc * 255.0).clip(0, 255).astype(np.uint8)
                for severity in PROFILE_SEVERITIES:
                    for seed in PROFILE_SEEDS:
                        matched_set_id = _stable_id(
                            "set",
                            sample.sample_id,
                            f"{severity:.12g}",
                            str(seed),
                        )
                        occluder = generate_synthetic_occluder(
                            sample_id=sample.sample_id,
                            severity_fraction=severity,
                            seed=seed,
                            foreground_area_pixels=foreground_area,
                            profile_name=PROFILE_NAME,
                        )
                        matched = build_matched_occluder_set(
                            target,
                            gt_v2a,
                            occluder,
                            attachment_roi_dilation_pixels=int(
                                placement_parameters["attachment_roi_dilation_pixels"]
                            ),
                            background_separation_pixels=int(
                                placement_parameters[
                                    "background_foreground_separation_pixels"
                                ]
                            ),
                        )
                        if not matched.complete:
                            incomplete_rows.append(
                                _incomplete_row(
                                    sample,
                                    matched,
                                    matched_set_id=matched_set_id,
                                    severity=severity,
                                    seed=seed,
                                )
                            )
                            continue
                        quartet_payload: dict[str, dict[str, Any]] = {}
                        for region in MATCHED_REGIONS:
                            placement = matched.placements[region]
                            occluded_hwc, full_occluder_mask = apply_occluder(
                                clean_hwc,
                                occluder,
                                placement,
                            )
                            overlap_diagnostics = _occluder_overlap_diagnostics(
                                region=region,
                                occluder_mask=full_occluder_mask,
                                clean_mask=target,
                                attachment_roi=matched.attachment_roi,
                            )
                            occluded_image = torch.from_numpy(
                                np.ascontiguousarray(occluded_hwc.transpose(2, 0, 1))
                            ).to(dtype=torch.float32)
                            prediction, entropy = infer(
                                model,
                                occluded_image,
                                device=resolved_device,
                            )
                            prediction = _validated_prediction(
                                prediction,
                                shape=target.shape,
                            )
                            entropy = _validated_entropy(entropy, shape=target.shape)
                            occ_v2a, occ_v2b = _run_frozen_geometry(
                                prediction.numpy(),
                                v2a_estimator=v2a_estimator,
                                v2b_estimator=v2b_estimator,
                            )
                            visible_metrics = masked_segmentation_metrics(
                                prediction,
                                target,
                                full_occluder_mask,
                            )
                            perturbation_id = _stable_id(
                                "perturbation",
                                matched_set_id,
                                region,
                            )
                            occ_results = {"v2a": occ_v2a, "v2b": occ_v2b}
                            method_rows: dict[str, dict[str, Any]] = {}
                            for method in METHODS:
                                gt_result, clean_result = geometry_by_method[method]
                                row = _trial_row(
                                    sample,
                                    method=method,
                                    gt_result=gt_result,
                                    clean_result=clean_result,
                                    occ_result=occ_results[method],
                                    severity=severity,
                                    seed=seed,
                                    region=region,
                                    matched_set_id=matched_set_id,
                                    perturbation_id=perturbation_id,
                                    matched=matched,
                                    placement=placement,
                                    full_occluder_mask=full_occluder_mask,
                                    prediction=prediction,
                                    entropy=entropy,
                                    visible_metrics=visible_metrics,
                                    overlap_diagnostics=overlap_diagnostics,
                                )
                                trial_rows.append(row)
                                method_rows[method] = row
                            quartet_payload[region] = {
                                "image": np.rint(occluded_hwc * 255.0)
                                .clip(0, 255)
                                .astype(np.uint8),
                                "clean_results": {
                                    "v2a": clean_v2a,
                                    "v2b": clean_v2b,
                                },
                                "occluded_results": occ_results,
                                "metrics": method_rows,
                            }
                        if sample.sample_id in selected_visualizations:
                            visualization = create_matched_quartet_visualization(
                                clean_rgb,
                                quartet_payload,
                                sample_id=sample.sample_id,
                                severity_fraction=severity,
                                seed=seed,
                                occluder_id=occluder.occluder_id,
                            )
                            relative = (
                                Path("visualizations")
                                / sample.sample_id
                                / f"severity_{severity:.2f}_seed_{seed}.png"
                            )
                            (staging / relative).parent.mkdir(
                                parents=True, exist_ok=True
                            )
                            visualization.save(staging / relative, format="PNG")

        metadata_rows = [sample.snapshot_row() for sample in samples]
        analysis = summarize_robustness_trials(
            trial_rows,
            metadata_rows,
            incomplete_rows=incomplete_rows,
        )
        _write_csv(
            staging / "development_metadata_snapshot.csv",
            REQUIRED_MANIFEST_FIELDS,
            metadata_rows,
        )
        _write_csv(
            staging / "clean_reference_per_image.csv",
            CLEAN_REFERENCE_FIELDS,
            clean_rows,
        )
        _write_csv(staging / "perturbation_trials.csv", TRIAL_FIELDS, trial_rows)
        _write_csv(
            staging / "incomplete_matched_sets.csv", INCOMPLETE_FIELDS, incomplete_rows
        )
        _write_analysis_artifacts(staging, analysis)
        create_occlusion_robustness_plots(trial_rows, analysis, staging / "plots")
        _write_json(staging / "benchmark_profile.json", profile_definition)
        counts = _analysis_counts(metadata_rows)
        counts["perturbation_evaluations"] = len(
            {str(row["perturbation_id"]) for row in trial_rows}
        )
        manifest = {
            "benchmark": "Occlusion-Induced Action Stability and Failure Awareness Benchmark",
            "profile": PROFILE_NAME,
            "development_manifest_authority": True,
            "role": "development",
            "class_mapping": {"background": 0, "Flesh": 1, "Calyx": 2},
            **counts,
            "fixed_axis_eligible_images": sum(
                sample.fixed_axis_eligible for sample in samples
            ),
            "incomplete_matched_set_count": len(incomplete_rows),
            "checkpoint": {
                "basename": checkpoint_file.name,
                "sha256": _sha256(checkpoint_file),
                "epoch": checkpoint["completed_epoch"],
                "model_name": checkpoint["model_name"],
                "preprocessing_height": height,
                "preprocessing_width": width,
            },
            "device_type": resolved_device.type,
            "statistical_unit": "group_id",
            "perturbation_evaluations_are_statistical_n": False,
            "analysis_conditioning": {
                "coordinate_stability": COORDINATE_STABILITY_CONDITIONING,
                "severity_coverage": SEVERITY_COVERAGE_CONDITIONING,
            },
            "placement_parameters": {
                key: placement_parameters[key] for key in M4_V1_PLACEMENT_PARAMETERS
            },
            "training_fine_tuning_optimization_performed": False,
            "artifacts": {
                "benchmark_profile": "benchmark_profile.json",
                "development_metadata_snapshot": "development_metadata_snapshot.csv",
                "clean_reference_per_image": "clean_reference_per_image.csv",
                "perturbation_trials": "perturbation_trials.csv",
                "incomplete_matched_sets": "incomplete_matched_sets.csv",
                "matched_set_completeness": "matched_set_completeness.csv",
                "per_image_summary": "per_image_summary.csv",
                "per_group_summary": "per_group_summary.csv",
                "condition_summary": "condition_summary.csv",
                "paired_region_differences": "paired_region_differences.csv",
                "bootstrap_intervals": "bootstrap_intervals.csv",
                "failure_awareness_summary": "failure_awareness_summary.csv",
                "segmentation_summary": "segmentation_summary.csv",
                "plots": "plots",
                "visualizations": "visualizations",
            },
        }
        _assert_no_absolute_paths(manifest)
        _write_json(staging / "manifest.json", manifest)
        _install_staged_directory(staging, destination, overwrite=overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"manifest": manifest, "analysis": analysis}


def summarize_robustness_trials(
    trial_rows: Sequence[Mapping[str, Any]],
    metadata_rows: Sequence[Mapping[str, Any]],
    *,
    incomplete_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, list[dict[str, Any]]]:
    """Aggregate repeats within images/groups before comparative inference."""

    eligible_rows = [
        row for row in trial_rows if _as_bool(row.get("matched_set_complete"))
    ]
    per_image = _curve_summary_rows(eligible_rows, unit="sample_id")
    per_group = _curve_summary_rows(eligible_rows, unit="group_id")
    condition = _condition_summary_rows(eligible_rows)
    paired = _paired_region_rows(eligible_rows)
    bootstrap = bootstrap_group_intervals(
        paired,
        metadata_rows=metadata_rows,
        seed=BOOTSTRAP_SEED,
        replicates=BOOTSTRAP_REPLICATES,
    )
    failure = _failure_summary_rows(eligible_rows)
    segmentation = _segmentation_summary_rows(eligible_rows)
    completeness = matched_set_completeness_rows(trial_rows, incomplete_rows)
    return {
        "per_image_summary": per_image,
        "per_group_summary": per_group,
        "condition_summary": condition,
        "paired_region_differences": paired,
        "bootstrap_intervals": bootstrap,
        "failure_awareness_summary": failure,
        "segmentation_summary": segmentation,
        "matched_set_completeness": completeness,
    }


def matched_set_completeness_rows(
    trial_rows: Sequence[Mapping[str, Any]],
    incomplete_rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Summarize deterministic placement attrition without imputing missing sets."""

    attempts: dict[str, dict[str, Any]] = {}
    for row in trial_rows:
        matched_set_id = str(row.get("matched_set_id") or "")
        if not matched_set_id:
            continue
        attempts.setdefault(
            matched_set_id,
            {
                "matched_set_id": matched_set_id,
                "group_id": str(row.get("group_id") or ""),
                "severity_fraction": _finite_or_none(row.get("severity_fraction")),
                "complete": True,
                "reasons": {},
            },
        )
    for row in incomplete_rows:
        matched_set_id = str(row.get("matched_set_id") or "")
        if not matched_set_id:
            continue
        if matched_set_id in attempts and attempts[matched_set_id]["complete"]:
            raise ValueError(
                "A matched set cannot be both complete and incomplete in artifacts"
            )
        reasons = {
            region: str(row.get(f"{region}_reason") or "")
            for region in MATCHED_REGIONS
            if row.get(f"{region}_reason") not in (None, "")
        }
        attempts[matched_set_id] = {
            "matched_set_id": matched_set_id,
            "group_id": str(row.get("group_id") or ""),
            "severity_fraction": _finite_or_none(row.get("severity_fraction")),
            "complete": False,
            "reasons": reasons,
        }

    output: list[dict[str, Any]] = []
    for severity in PROFILE_SEVERITIES:
        severity_attempts = [
            attempt
            for attempt in attempts.values()
            if attempt["severity_fraction"] == severity
        ]
        if not severity_attempts:
            continue
        attempted_groups = {
            attempt["group_id"] for attempt in severity_attempts if attempt["group_id"]
        }
        for region in MATCHED_REGIONS:
            successful = [
                attempt
                for attempt in severity_attempts
                if region not in attempt["reasons"]
            ]
            unavailable = [
                attempt for attempt in severity_attempts if region in attempt["reasons"]
            ]
            reason_counts: dict[str, int] = defaultdict(int)
            for attempt in unavailable:
                reason_counts[str(attempt["reasons"][region])] += 1
            output.append(
                {
                    "summary_level": "region",
                    "region": region,
                    "severity_fraction": severity,
                    "attempted_matched_sets": len(severity_attempts),
                    "successfully_placed": len(successful),
                    "unavailable": len(unavailable),
                    "completion_proportion": len(successful) / len(severity_attempts),
                    "n_groups_attempted": len(attempted_groups),
                    "n_groups_represented_among_successful_placements": len(
                        {
                            attempt["group_id"]
                            for attempt in successful
                            if attempt["group_id"]
                        }
                    ),
                    "structured_unavailable_reason_counts": json.dumps(
                        dict(sorted(reason_counts.items())),
                        sort_keys=True,
                        separators=(",", ":"),
                    ),
                    "attempted_matched_quartets": None,
                    "complete_matched_quartets": None,
                    "incomplete_matched_quartets": None,
                    "completion_fraction": None,
                    "n_groups_with_complete_quartet": None,
                }
            )
        complete = [attempt for attempt in severity_attempts if attempt["complete"]]
        output.append(
            {
                "summary_level": "quartet",
                "region": "all_regions",
                "severity_fraction": severity,
                "attempted_matched_sets": None,
                "successfully_placed": None,
                "unavailable": None,
                "completion_proportion": None,
                "n_groups_attempted": len(attempted_groups),
                "n_groups_represented_among_successful_placements": None,
                "structured_unavailable_reason_counts": None,
                "attempted_matched_quartets": len(severity_attempts),
                "complete_matched_quartets": len(complete),
                "incomplete_matched_quartets": len(severity_attempts) - len(complete),
                "completion_fraction": len(complete) / len(severity_attempts),
                "n_groups_with_complete_quartet": len(
                    {attempt["group_id"] for attempt in complete if attempt["group_id"]}
                ),
            }
        )
    return output


def bootstrap_group_intervals(
    paired_rows: Sequence[Mapping[str, Any]],
    *,
    metadata_rows: Sequence[Mapping[str, Any]] = (),
    seed: int = BOOTSTRAP_SEED,
    replicates: int = BOOTSTRAP_REPLICATES,
) -> list[dict[str, Any]]:
    """Bootstrap paired means by resampling independent group IDs only."""

    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("bootstrap seed must be a non-negative integer")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates < 1
    ):
        raise ValueError("bootstrap replicates must be a positive integer")
    output: list[dict[str, Any]] = []
    for method in METHODS:
        for control in COMPARISON_REGIONS:
            comparison = f"attachment_minus_{control}"
            comparison_rows = [
                row
                for row in paired_rows
                if row.get("method") == method and row.get("comparison") == comparison
            ]
            for outcome in SUMMARY_OUTCOMES:
                interval_seed = _derived_bootstrap_seed(
                    seed,
                    method=method,
                    comparison=comparison,
                    outcome=outcome,
                )
                rng = np.random.default_rng(interval_seed)
                field = f"{outcome}_auc_difference_attachment_minus_control"
                group_values: dict[str, list[float]] = defaultdict(list)
                for row in comparison_rows:
                    value = _finite_or_none(row.get(field))
                    if value is not None:
                        group_values[str(row["group_id"])].append(value)
                grouped = {
                    group_id: mean(values)
                    for group_id, values in sorted(group_values.items())
                }
                group_ids = tuple(grouped)
                values = np.asarray(
                    [grouped[group_id] for group_id in group_ids], dtype=np.float64
                )
                if len(group_ids):
                    bootstrap_means = np.empty(replicates, dtype=np.float64)
                    for index in range(replicates):
                        selected = rng.integers(0, len(group_ids), size=len(group_ids))
                        bootstrap_means[index] = float(values[selected].mean())
                    lower, upper = np.percentile(bootstrap_means, (2.5, 97.5))
                    estimate = float(values.mean())
                else:
                    lower = upper = estimate = None
                related = [
                    row
                    for row in comparison_rows
                    if str(row.get("group_id")) in grouped
                ]
                related_metadata = [
                    row for row in metadata_rows if str(row.get("group_id")) in grouped
                ]
                counts = _analysis_counts(related_metadata)
                counts["n_groups"] = len(group_ids)
                if not related_metadata:
                    counts["n_images"] = sum(
                        int(_finite_or_none(row.get("n_images")) or 0)
                        for row in related
                    )
                    counts["n_sessions"] = sum(
                        int(_finite_or_none(row.get("n_sessions")) or 0)
                        for row in related
                    )
                counts["perturbation_evaluations"] = sum(
                    int(_finite_or_none(row.get("perturbation_evaluations")) or 0)
                    for row in related
                )
                stable = len(group_ids) >= MINIMUM_STABLE_INTERVAL_GROUPS
                output.append(
                    {
                        "method": method,
                        "comparison": comparison,
                        "comparison_priority": (
                            "primary" if control == "background" else "secondary"
                        ),
                        "outcome": outcome,
                        **counts,
                        "mean_paired_difference": estimate,
                        "confidence_level": 0.95,
                        "lower_bound": None if lower is None else float(lower),
                        "upper_bound": None if upper is None else float(upper),
                        "benchmark_bootstrap_seed": seed,
                        "bootstrap_seed": interval_seed,
                        "bootstrap_replicates": replicates,
                        "stable_interval": stable,
                        "interval_note": (
                            "descriptive group bootstrap interval"
                            if stable
                            else "too few independent groups for a stable interval"
                        ),
                    }
                )
    return output


def summarize_existing_output(
    output_root: PathLike, *, profile: str = PROFILE_NAME
) -> dict[str, Any]:
    """Regenerate summaries and plots from an existing sanitized trial table."""

    _require_profile(profile)
    root = Path(output_root)
    trial_path = root / "perturbation_trials.csv"
    metadata_path = root / "development_metadata_snapshot.csv"
    incomplete_path = root / "incomplete_matched_sets.csv"
    if (
        not trial_path.is_file()
        or not metadata_path.is_file()
        or not incomplete_path.is_file()
    ):
        raise FileNotFoundError(
            "Existing output is missing trial, metadata, or incomplete-set CSV artifacts"
        )
    trials = _read_csv(trial_path)
    metadata = _read_csv(metadata_path)
    incomplete = _read_csv(incomplete_path)
    analysis = summarize_robustness_trials(
        trials,
        metadata,
        incomplete_rows=incomplete,
    )
    _write_analysis_artifacts(root, analysis)
    create_occlusion_robustness_plots(trials, analysis, root / "plots")
    return analysis


def _infer_segmentation(
    model: nn.Module,
    image: torch.Tensor,
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Infer from RGB only; no GT or placement mask enters this interface."""

    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError("inference image must have shape [3, H, W]")
    images = image.unsqueeze(0).to(device)
    logits = _model_logits(model(images))
    _validate_logits(logits, images)
    _, _, entropy = confidence_and_entropy(logits)
    return logits.argmax(dim=1)[0].cpu(), entropy[0].cpu()


def _load_listed_pair(
    sample: DevelopmentSample,
    *,
    height: int,
    width: int,
) -> tuple[torch.Tensor, np.ndarray]:
    # These are the only two sample paths opened by the runner.
    with (
        Image.open(sample.resolved_image_path) as image_file,
        Image.open(sample.resolved_mask_path) as mask_file,
    ):
        if image_file.size != mask_file.size:
            raise ValueError(
                f"Image/mask dimensions differ for listed sample {sample.sample_id!r}"
            )
        image = preprocess_rgb_image(
            image_file.convert("RGB"), height=height, width=width
        )
        mask = preprocess_class_mask(mask_file, height=height, width=width).numpy()
    return image, mask


def _clean_reference_row(
    sample: DevelopmentSample,
    method: str,
    gt_result: GeometryResult | None,
    clean_result: GeometryResult | None,
    clean_entropy: torch.Tensor,
    clean_metrics: Mapping[str, Any],
    attachment_roi: np.ndarray | None,
) -> dict[str, Any]:
    c_gt = _result_coordinate(gt_result)
    c_clean = _result_coordinate(clean_result)
    return {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "session_id": sample.session_id,
        "fixed_axis_eligible": sample.fixed_axis_eligible,
        "method": method,
        "c_gt": c_gt,
        "c_clean": c_clean,
        "gt_status": gt_result.status if gt_result is not None else "not_eligible",
        "gt_failure_code": gt_result.failure_code
        if gt_result is not None
        else "segmentation_only",
        "clean_status": clean_result.status
        if clean_result is not None
        else "not_eligible",
        "clean_failure_code": clean_result.failure_code
        if clean_result is not None
        else "segmentation_only",
        "clean_gt_reference_abs": (
            abs(c_clean - c_gt) if c_clean is not None and c_gt is not None else None
        ),
        "clean_mean_normalized_entropy": float(clean_entropy.mean().item()),
        "clean_mean_normalized_entropy_attachment_roi": (
            _masked_mean(clean_entropy, attachment_roi)
            if attachment_roi is not None
            else None
        ),
        **{
            f"clean_{field}": clean_metrics[field]
            for field in (
                "iou_background",
                "iou_flesh",
                "iou_calyx",
                "dice_background",
                "dice_flesh",
                "dice_calyx",
            )
        },
    }


def _trial_row(
    sample: DevelopmentSample,
    *,
    method: str,
    gt_result: GeometryResult,
    clean_result: GeometryResult,
    occ_result: GeometryResult,
    severity: float,
    seed: int,
    region: str,
    matched_set_id: str,
    perturbation_id: str,
    matched: MatchedOccluderSet,
    placement: Any,
    full_occluder_mask: np.ndarray,
    prediction: torch.Tensor,
    entropy: torch.Tensor,
    visible_metrics: Mapping[str, Any],
    overlap_diagnostics: Mapping[str, int],
) -> dict[str, Any]:
    c_gt = _result_coordinate(gt_result)
    c_clean = _result_coordinate(clean_result)
    c_occ = _result_coordinate(occ_result)
    differences = reference_error_metrics(c_gt, c_clean, c_occ)
    finite_cut = c_occ is not None and occ_result.status == "ok"
    silent = silent_drift_flags(
        method_status=occ_result.status,
        finite_cut_returned=finite_cut,
        stability_abs=differences["stability_abs"],
    )
    inside = prediction.numpy()[full_occluder_mask]
    attachment_roi = np.asarray(matched.attachment_roi, dtype=bool)
    return {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "session_id": sample.session_id,
        "orientation_category": sample.orientation_category,
        "profile": PROFILE_NAME,
        "matched_set_id": matched_set_id,
        "perturbation_id": perturbation_id,
        "matched_set_complete": True,
        "severity_fraction": severity,
        "seed": seed,
        "region": region,
        "method": method,
        "occluder_id": matched.occluder.occluder_id,
        "occluder_template_sha256": matched.occluder.template_sha256,
        "occluder_area_pixels": matched.occluder.area_pixels,
        "occluder_opacity": matched.occluder.opacity,
        "occluder_rotation_degrees": matched.occluder.rotation_degrees,
        "occluder_scale_pixels": matched.occluder.scale_pixels,
        "placement_top": placement.top,
        "placement_left": placement.left,
        "placed_mask_sha256": hashlib.sha256(
            np.asarray(full_occluder_mask, dtype=np.uint8).tobytes(order="C")
        ).hexdigest(),
        **dict(overlap_diagnostics),
        "c_gt": c_gt,
        "c_clean": c_clean,
        "c_occ": c_occ,
        "gt_status": gt_result.status,
        "gt_failure_code": gt_result.failure_code,
        "clean_status": clean_result.status,
        "clean_failure_code": clean_result.failure_code,
        "method_status": occ_result.status,
        "structured_failure_code": occ_result.failure_code,
        "finite_cut_returned": finite_cut,
        **differences,
        **silent,
        **_geometry_diagnostics(occ_result),
        "mean_normalized_entropy_global": float(entropy.mean().item()),
        "mean_normalized_entropy_attachment_roi": _masked_mean(entropy, attachment_roi),
        "mean_normalized_entropy_occluder": _masked_mean(entropy, full_occluder_mask),
        "prediction_inside_occluder_background_pixels": int(
            np.count_nonzero(inside == 0)
        ),
        "prediction_inside_occluder_flesh_pixels": int(np.count_nonzero(inside == 1)),
        "prediction_inside_occluder_calyx_pixels": int(np.count_nonzero(inside == 2)),
        **dict(visible_metrics),
    }


def _occluder_overlap_diagnostics(
    *,
    region: str,
    occluder_mask: np.ndarray,
    clean_mask: np.ndarray,
    attachment_roi: np.ndarray,
) -> dict[str, int]:
    translated = np.asarray(occluder_mask, dtype=bool)
    foreground = np.isin(clean_mask, (1, 2))
    attachment = np.asarray(attachment_roi, dtype=bool)
    if translated.shape != clean_mask.shape or attachment.shape != clean_mask.shape:
        raise ValueError("occluder, clean mask, and attachment ROI shapes must match")
    attachment_overlap = int(np.count_nonzero(translated & attachment))
    foreground_overlap = int(np.count_nonzero(translated & foreground))
    if region in {"calyx_tip", "flesh_far"} and attachment_overlap != 0:
        raise ValueError(
            f"{region} occluder footprint must have zero attachment-ROI overlap"
        )
    if region == "background" and foreground_overlap != 0:
        raise ValueError(
            "background occluder footprint must have zero foreground overlap"
        )
    return {
        "occluder_attachment_roi_overlap_pixels": attachment_overlap,
        "occluder_foreground_overlap_pixels": foreground_overlap,
    }


def _geometry_diagnostics(result: GeometryResult) -> dict[str, Any]:
    common = {
        "predicted_calyx_component_count": result.calyx_component_count,
        "attachment_contact_support_pixel_count": result.contact_pixel_count,
    }
    if isinstance(result, FixedAxisSearchCutlineResult):
        return {
            **common,
            "support_pixel_count": int(np.count_nonzero(result.v2a_support_mask)),
            "candidate_count": result.candidate_count,
            "feasible_candidate_count": result.feasible_candidate_count,
            "feasible_block_count": result.feasible_block_count,
            "selected_block_id": result.selected_block_id,
            "selected_block_start": result.selected_block_start,
            "selected_block_end": result.selected_block_end,
            "selected_block_candidate_count": result.selected_block_candidate_count,
            "selected_block_is_singleton": result.selected_block_is_singleton,
            "selected_local_contact_count": result.selected_local_contact_count,
            "selected_local_flesh_count": result.selected_local_flesh_count,
            "selected_local_calyx_count": result.selected_local_calyx_count,
        }
    return {
        **common,
        "support_pixel_count": result.support_pixel_count,
        "candidate_count": None,
        "feasible_candidate_count": None,
        "feasible_block_count": None,
        "selected_block_id": None,
        "selected_block_start": None,
        "selected_block_end": None,
        "selected_block_candidate_count": None,
        "selected_block_is_singleton": None,
        "selected_local_contact_count": None,
        "selected_local_flesh_count": None,
        "selected_local_calyx_count": None,
    }


def _incomplete_row(
    sample: DevelopmentSample,
    matched: MatchedOccluderSet,
    *,
    matched_set_id: str,
    severity: float,
    seed: int,
) -> dict[str, Any]:
    reasons = matched.unavailable_reasons
    return {
        "sample_id": sample.sample_id,
        "group_id": sample.group_id,
        "session_id": sample.session_id,
        "profile": PROFILE_NAME,
        "matched_set_id": matched_set_id,
        "severity_fraction": severity,
        "seed": seed,
        "occluder_id": matched.occluder.occluder_id,
        "occluder_template_sha256": matched.occluder.template_sha256,
        "occluder_area_pixels": matched.occluder.area_pixels,
        "missing_regions": ";".join(
            region for region in MATCHED_REGIONS if region in reasons
        ),
        **{f"{region}_reason": reasons.get(region) for region in MATCHED_REGIONS},
    }


def _curve_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    unit: str,
) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row[unit]), str(row["method"]), str(row["region"]))].append(row)
    output = []
    for (identifier, method, region), values in sorted(grouped.items()):
        first = values[0]
        prefix = (
            {
                "sample_id": identifier,
                "group_id": str(first["group_id"]),
                "session_id": str(first["session_id"]),
            }
            if unit == "sample_id"
            else {"group_id": identifier}
        )
        output.append(
            {
                **prefix,
                "method": method,
                "region": region,
                **_analysis_counts(values),
                **_severity_auc_values(values),
            }
        )
    return output


def _severity_auc_values(rows: Sequence[Mapping[str, Any]]) -> dict[str, float | None]:
    output: dict[str, float | None] = {}
    for outcome in SUMMARY_OUTCOMES:
        severity_values = []
        for severity in PROFILE_SEVERITIES:
            selected = [
                _outcome_value(row, outcome)
                for row in rows
                if _finite_or_none(row.get("severity_fraction")) == severity
            ]
            finite = [value for value in selected if value is not None]
            severity_values.append(mean(finite) if finite else None)
        output[f"{outcome}_auc"] = _normalized_trapezoid_auc(severity_values)
    return output


def _outcome_value(row: Mapping[str, Any], outcome: str) -> float | None:
    if outcome == "structured_failure_rate":
        return float(str(row.get("method_status")) != "ok")
    match = re.fullmatch(r"silent_drift_(2|5|10)_rate", outcome)
    if match:
        return float(_as_bool(row.get(f"silent_drift_{match.group(1)}_pixels")))
    return _finite_or_none(row.get(outcome))


def _normalized_trapezoid_auc(values: Sequence[float | None]) -> float | None:
    if len(values) != len(PROFILE_SEVERITIES) or any(value is None for value in values):
        return None
    x = np.asarray(PROFILE_SEVERITIES, dtype=np.float64)
    x = (x - x[0]) / (x[-1] - x[0])
    y = np.asarray(values, dtype=np.float64)
    return float(np.trapezoid(y, x))


def _condition_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by(rows, ("method", "region", "severity_fraction"))
    output = []
    for key, values in grouped:
        group_means = _group_means(
            values,
            (
                "stability_abs",
                "gt_reference_abs",
                "excess_gt_reference_error",
                "structured_failure_rate",
                "silent_drift_2_rate",
                "silent_drift_5_rate",
                "silent_drift_10_rate",
            ),
        )
        output.append(
            {
                "method": key[0],
                "region": key[1],
                "severity_fraction": _finite_or_none(key[2]),
                **_analysis_counts(values),
                "mean_stability_abs": group_means["stability_abs"],
                "mean_gt_reference_abs": group_means["gt_reference_abs"],
                "mean_excess_gt_reference_error": group_means[
                    "excess_gt_reference_error"
                ],
                "structured_failure_rate": group_means["structured_failure_rate"],
                "silent_drift_2_rate": group_means["silent_drift_2_rate"],
                "silent_drift_5_rate": group_means["silent_drift_5_rate"],
                "silent_drift_10_rate": group_means["silent_drift_10_rate"],
            }
        )
    return output


def _paired_region_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    group_ids = sorted({str(row["group_id"]) for row in rows})
    for group_id in group_ids:
        for method in METHODS:
            method_rows = [
                row
                for row in rows
                if str(row["group_id"]) == group_id and row.get("method") == method
            ]
            attachment = [
                row for row in method_rows if row.get("region") == "attachment"
            ]
            for control in COMPARISON_REGIONS:
                control_rows = [
                    row for row in method_rows if row.get("region") == control
                ]
                common_sets = {str(row["matched_set_id"]) for row in attachment} & {
                    str(row["matched_set_id"]) for row in control_rows
                }
                paired_attachment = [
                    row
                    for row in attachment
                    if str(row["matched_set_id"]) in common_sets
                ]
                paired_control = [
                    row
                    for row in control_rows
                    if str(row["matched_set_id"]) in common_sets
                ]
                if not common_sets:
                    continue
                attachment_auc = _severity_auc_values(paired_attachment)
                control_auc = _severity_auc_values(paired_control)
                combined = [*paired_attachment, *paired_control]
                output.append(
                    {
                        "group_id": group_id,
                        "method": method,
                        "comparison": f"attachment_minus_{control}",
                        "comparison_priority": (
                            "primary" if control == "background" else "secondary"
                        ),
                        **_analysis_counts(combined),
                        **{
                            f"{outcome}_auc_difference_attachment_minus_control": _difference(
                                attachment_auc[f"{outcome}_auc"],
                                control_auc[f"{outcome}_auc"],
                            )
                            for outcome in SUMMARY_OUTCOMES
                        },
                    }
                )
    return output


def _failure_summary_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    grouped = _group_by(rows, ("method", "region", "severity_fraction"))
    output = []
    for key, values in grouped:
        group_means = _group_means(
            values,
            (
                "structured_failure_rate",
                "finite_cut_rate",
                "silent_drift_2_rate",
                "silent_drift_5_rate",
                "silent_drift_10_rate",
            ),
        )
        output.append(
            {
                "method": key[0],
                "region": key[1],
                "severity_fraction": _finite_or_none(key[2]),
                **_analysis_counts(values),
                **group_means,
            }
        )
    return output


def _segmentation_summary_rows(
    rows: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    unique = {}
    for row in rows:
        unique.setdefault(str(row["perturbation_id"]), row)
    grouped = _group_by(list(unique.values()), ("region", "severity_fraction"))
    metrics = (
        "iou_background",
        "iou_flesh",
        "iou_calyx",
        "dice_background",
        "dice_flesh",
        "dice_calyx",
        "mean_foreground_iou",
        "mean_foreground_dice",
    )
    output = []
    for key, values in grouped:
        group_means = _group_means(values, metrics)
        output.append(
            {
                "region": key[0],
                "severity_fraction": _finite_or_none(key[1]),
                **_analysis_counts(values),
                **{
                    (metric if metric.startswith("mean_") else f"mean_{metric}"): value
                    for metric, value in group_means.items()
                },
            }
        )
    return output


def _group_means(
    rows: Sequence[Mapping[str, Any]],
    outcomes: Sequence[str],
) -> dict[str, float | None]:
    per_group: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        per_group[str(row["group_id"])].append(row)
    output = {}
    for outcome in outcomes:
        values = []
        for group_rows in per_group.values():
            if outcome == "finite_cut_rate":
                group_values = [
                    float(_as_bool(row.get("finite_cut_returned")))
                    for row in group_rows
                ]
            else:
                group_values = [
                    value
                    for row in group_rows
                    if (value := _outcome_value(row, outcome)) is not None
                ]
            if group_values:
                values.append(mean(group_values))
        output[outcome] = mean(values) if values else None
    return output


def _analysis_counts(
    rows: Sequence[Mapping[str, Any]],
    *,
    metadata_rows: Sequence[Mapping[str, Any]] = (),
) -> dict[str, int]:
    source = list(rows)
    groups = {
        str(row["group_id"]) for row in source if row.get("group_id") not in (None, "")
    }
    images = {
        str(row["sample_id"])
        for row in source
        if row.get("sample_id") not in (None, "")
    }
    sessions = {
        str(row["session_id"])
        for row in source
        if row.get("session_id") not in (None, "")
    }
    perturbations = {
        str(row["perturbation_id"])
        for row in source
        if row.get("perturbation_id") not in (None, "")
    }
    if metadata_rows and not source:
        groups = {str(row["group_id"]) for row in metadata_rows if row.get("group_id")}
        images = {
            str(row["sample_id"]) for row in metadata_rows if row.get("sample_id")
        }
        sessions = {
            str(row["session_id"]) for row in metadata_rows if row.get("session_id")
        }
    return {
        "n_groups": len(groups),
        "n_images": len(images),
        "n_sessions": len(sessions),
        "perturbation_evaluations": len(perturbations),
    }


def _group_by(
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> list[tuple[tuple[str, ...], list[Mapping[str, Any]]]]:
    grouped: dict[tuple[str, ...], list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(str(row.get(field, "")) for field in fields)].append(row)
    return sorted(grouped.items())


def _result_coordinate(result: GeometryResult | None) -> float | None:
    if result is None or result.status != "ok":
        return None
    value = (
        result.selected_cut_coordinate
        if isinstance(result, FixedAxisSearchCutlineResult)
        else result.final_cut_coordinate
    )
    return _finite_or_none(value)


def _masked_mean(values: torch.Tensor, mask: np.ndarray) -> float | None:
    selected = values[torch.from_numpy(np.array(mask, dtype=bool, copy=True))]
    return float(selected.mean().item()) if selected.numel() else None


def _validated_prediction(value: Any, *, shape: tuple[int, int]) -> torch.Tensor:
    tensor = torch.as_tensor(value).cpu()
    if tensor.shape != shape:
        raise ValueError(
            "inference prediction dimensions must match the evaluation mask"
        )
    if tensor.dtype not in (
        torch.uint8,
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
    ):
        raise TypeError("inference prediction must use an integer dtype")
    if torch.any((tensor < 0) | (tensor > 2)):
        raise ValueError("inference prediction must contain only class IDs 0, 1, and 2")
    return tensor.long()


def _validated_entropy(value: Any, *, shape: tuple[int, int]) -> torch.Tensor:
    tensor = torch.as_tensor(value, dtype=torch.float32).cpu()
    if tensor.shape != shape or not torch.all(torch.isfinite(tensor)):
        raise ValueError("inference entropy must be a finite evaluation-shape tensor")
    if torch.any((tensor < 0.0) | (tensor > 1.0)):
        raise ValueError("inference entropy values must be in [0, 1]")
    return tensor


def _write_analysis_artifacts(
    root: Path, analysis: Mapping[str, Sequence[Mapping[str, Any]]]
) -> None:
    definitions = (
        ("per_image_summary", PER_IMAGE_SUMMARY_FIELDS),
        ("per_group_summary", PER_GROUP_SUMMARY_FIELDS),
        ("condition_summary", CONDITION_SUMMARY_FIELDS),
        ("paired_region_differences", PAIRED_FIELDS),
        ("bootstrap_intervals", BOOTSTRAP_FIELDS),
        ("failure_awareness_summary", FAILURE_SUMMARY_FIELDS),
        ("segmentation_summary", SEGMENTATION_SUMMARY_FIELDS),
        ("matched_set_completeness", MATCHED_SET_COMPLETENESS_FIELDS),
    )
    for name, fields in definitions:
        _write_csv(root / f"{name}.csv", fields, analysis[name])


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return format(numeric, ".12g") if math.isfinite(numeric) else ""
    if isinstance(value, np.integer):
        return int(value)
    return value


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _validate_output_destination(path: Path, *, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(f"Output already exists: {path}")
        if path.is_symlink() or not path.is_dir():
            raise ValueError("Existing output must be a non-symlink directory")


def _install_staged_directory(
    staging: Path, destination: Path, *, overwrite: bool
) -> None:
    backup: Path | None = None
    if destination.exists():
        if not overwrite:
            raise FileExistsError(f"Output appeared during generation: {destination}")
        backup = destination.with_name(f".{destination.name}.backup-{uuid.uuid4().hex}")
        destination.replace(backup)
    try:
        staging.replace(destination)
    except Exception:
        if backup is not None and backup.exists():
            backup.replace(destination)
        raise
    if backup is not None:
        shutil.rmtree(backup)


def _validated_relative_path(value: str, *, context: str) -> Path:
    if not value:
        raise ValueError(f"{context} must be non-empty")
    if _looks_like_absolute_path(value):
        raise ValueError(f"{context} must be relative")
    normalized = PurePosixPath(value.replace("\\", "/"))
    if any(part in ("", ".", "..") for part in normalized.parts):
        raise ValueError(f"{context} must not contain empty, dot, or parent components")
    return Path(*normalized.parts)


def _looks_like_absolute_path(value: str) -> bool:
    return (
        PureWindowsPath(value).is_absolute()
        or PurePosixPath(value).is_absolute()
        or re.match(r"^[A-Za-z]:", value) is not None
        or value.startswith(("//", "\\\\"))
    )


def _assert_no_absolute_paths(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_no_absolute_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_absolute_paths(item)
    elif isinstance(value, str) and _looks_like_absolute_path(value):
        raise ValueError("Persisted metadata must not contain absolute paths")


def _validate_identifier(value: str, *, name: str) -> None:
    if not value or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", value):
        raise ValueError(f"{name} must be a non-empty path-free identifier")


def _parse_strict_boolean(value: str, *, name: str) -> bool:
    normalized = value.strip().lower()
    if normalized not in {"true", "false"}:
        raise ValueError(f"{name} must be exactly true or false")
    return normalized == "true"


def _validate_visualization_ids(
    values: Sequence[str],
    samples: Sequence[DevelopmentSample],
) -> frozenset[str]:
    if isinstance(values, (str, bytes)):
        raise TypeError("visualization_sample_ids must be a sequence")
    known = {sample.sample_id for sample in samples if sample.fixed_axis_eligible}
    selected = set(values)
    unknown = sorted(selected - known)
    if unknown:
        raise ValueError(
            "Visualization sample IDs must be listed fixed-axis-eligible samples: "
            + ", ".join(unknown)
        )
    return frozenset(selected)


def _stable_id(prefix: str, *parts: str) -> str:
    digest = hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()
    return f"{prefix}_{digest[:20]}"


def _derived_bootstrap_seed(
    benchmark_seed: int,
    *,
    method: str,
    comparison: str,
    outcome: str,
) -> int:
    canonical = (
        f"{PROFILE_NAME}|bootstrap|{benchmark_seed}|{method}|{comparison}|{outcome}"
    )
    digest = hashlib.sha256(canonical.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="big", signed=False)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _finite_or_none(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None
    return numeric if math.isfinite(numeric) else None


def _difference(first: Any, second: Any) -> float | None:
    first_value = _finite_or_none(first)
    second_value = _finite_or_none(second)
    if first_value is None or second_value is None:
        return None
    return first_value - second_value


def _as_bool(value: Any) -> bool:
    if isinstance(value, (bool, np.bool_)):
        return bool(value)
    return str(value).strip().lower() == "true"


def _require_profile(profile: str) -> None:
    if profile != PROFILE_NAME:
        raise ValueError(f"Only frozen profile {PROFILE_NAME!r} is supported")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen development-only Milestone 4 benchmark."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate-manifest")
    validate.add_argument("--profile", choices=(PROFILE_NAME,), required=True)
    validate.add_argument("--development-manifest", type=Path, required=True)
    validate.add_argument("--safe-data-root", type=Path, required=True)

    def add_run_arguments(command: argparse.ArgumentParser) -> None:
        command.add_argument("--profile", choices=(PROFILE_NAME,), required=True)
        command.add_argument("--development-manifest", type=Path, required=True)
        command.add_argument("--safe-data-root", type=Path, required=True)
        command.add_argument("--checkpoint", type=Path, required=True)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--device", required=True)
        command.add_argument("--overwrite", action="store_true")

    run = subparsers.add_parser("run")
    add_run_arguments(run)
    run.add_argument("--visualize-sample-id", action="append", default=[])

    summarize = subparsers.add_parser("summarize")
    summarize.add_argument("--profile", choices=(PROFILE_NAME,), required=True)
    summarize.add_argument("--output-root", type=Path, required=True)

    visualize = subparsers.add_parser("visualize")
    add_run_arguments(visualize)
    visualize.add_argument("--sample-id", action="append", required=True)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the Milestone 4 command-line workflow."""

    arguments = _build_argument_parser().parse_args(argv)
    if arguments.command == "validate-manifest":
        rows = validate_development_manifest(
            arguments.development_manifest,
            arguments.safe_data_root,
        )
        counts = _analysis_counts([row.snapshot_row() for row in rows])
        print(
            f"Validated {counts['n_images']} development images in "
            f"{counts['n_groups']} independent groups."
        )
        return 0
    if arguments.command == "summarize":
        analysis = summarize_existing_output(
            arguments.output_root,
            profile=arguments.profile,
        )
        print(
            "Regenerated group-level summaries; perturbation rows were not used "
            "as statistical n."
        )
        print(f"Wrote {len(analysis['bootstrap_intervals'])} descriptive intervals.")
        return 0
    selected = (
        arguments.visualize_sample_id
        if arguments.command == "run"
        else arguments.sample_id
    )
    result = run_occlusion_robustness(
        arguments.development_manifest,
        arguments.safe_data_root,
        arguments.checkpoint,
        arguments.output_root,
        device=arguments.device,
        profile=arguments.profile,
        overwrite=arguments.overwrite,
        visualization_sample_ids=selected,
    )
    manifest = result["manifest"]
    print(
        f"Completed {PROFILE_NAME}: {manifest['n_groups']} groups, "
        f"{manifest['n_images']} images, and "
        f"{manifest['perturbation_evaluations']} perturbation evaluations."
    )
    print("Independent analysis n is n_groups, never perturbation evaluations.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
