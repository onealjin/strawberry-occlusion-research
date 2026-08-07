"""Reproducible preliminary new-batch evaluation for a pilot holdout.

This module prepares a seven-image VisionTrain export, audits it against an
optional v1 reference, creates a manually editable metadata template, and runs
segmentation plus fixed-axis mask-geometry agreement. It never trains or
fine-tunes a model and exposes no geometry-tuning arguments.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import re
import shutil
import tempfile
import uuid
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path, PurePosixPath, PureWindowsPath
from statistics import mean, median
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader

from strawberry_occlusion.data import SegmentationDataset
from strawberry_occlusion.data.vt_export import (
    CLASS_MAPPING,
    OVERLAP_RULE,
    _category_id,
    _load_json,
    _required_string,
    _required_value,
    _validate_category_mapping,
    _validate_manifest_images,
    _validate_saved_mask,
    construct_mask,
    convert_vt_export,
)
from strawberry_occlusion.evaluation.fixed_axis_cutline import (
    V2A_ALGORITHM_NAME,
    V2A_ALGORITHM_VERSION,
    V2B_ALGORITHM_NAME,
    V2B_ALGORITHM_VERSION,
    _validate_v2a_result,
    _validate_v2b_result,
    calculate_fixed_axis_line_side_proxies,
    calculate_fixed_axis_search_line_side_proxies,
)
from strawberry_occlusion.evaluation.fixed_axis_sensitivity import (
    COMMITTED_BASELINE_CONFIGURATION_ID,
)
from strawberry_occlusion.evaluation.segmentation import (
    ModelFactory,
    _image_tensor_to_rgb,
    _model_logits,
    _validate_logits,
    confidence_and_entropy,
    load_segmentation_checkpoint,
    metrics_from_confusion_matrix,
    per_image_metrics,
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
from strawberry_occlusion.visualization.pilot_holdout import (
    create_pilot_holdout_visualization,
)


PathLike = str | Path
CategoryId = int | str
V2aEstimator = Callable[..., FixedAxisCutlineResult]
V2bEstimator = Callable[..., FixedAxisSearchCutlineResult]
PilotVisualizationFunction = Callable[..., Image.Image]

DEFAULT_EXPECTED_SAMPLE_COUNT = 7
EXPECTED_CANONICAL_CHECKPOINT_EPOCH = 92
PILOT_SPLIT = "pilot"
PILOT_DATASET_KIND = "pilot_holdout"
PILOT_DATASET_VERSION = 1
PILOT_DATASET_IDENTIFIER = "pilot_holdout_normalized_v1"
STUDY_NAME = "preliminary new-batch evaluation"
PILOT_SUBSET_NAME = "seven-image pilot subset"
INPUT_FORMATS = ("auto", "full-export", "extracted-pairs")
EXTRACTED_IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
EXTRACTED_IMAGE_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
}
HARMLESS_PLATFORM_FILENAMES = frozenset({"desktop.ini", ".ds_store", "thumbs.db"})

SUPPORTED_ORIENTATION_CATEGORIES = (
    "right",
    "upper_right",
    "lower_right",
    "left",
    "up",
    "down",
    "diagonal_left",
    "ambiguous",
    "unknown",
)
FIXED_AXIS_ELIGIBILITY_STATUSES = ("", "unknown", "false", "true")
METADATA_FIELDS = (
    "sample_id",
    "orientation_category",
    "canonical_fixed_axis_pose",
    "fixed_axis_eligible",
    "occlusion_category",
    "annotation_qa_status",
    "notes",
)
EXCLUSION_FIELDS = (
    "sample_id",
    "orientation_category",
    "fixed_axis_eligible",
    "exclusion_reason",
)
SEGMENTATION_FIELDS = (
    "sample_id",
    "iou_background",
    "iou_flesh",
    "iou_calyx",
    "dice_background",
    "dice_flesh",
    "dice_calyx",
    "flesh_precision",
    "flesh_recall",
    "calyx_precision",
    "calyx_recall",
    "mean_foreground_iou",
    "mean_foreground_dice",
    "flesh_tp",
    "flesh_fp",
    "flesh_fn",
    "calyx_tp",
    "calyx_fp",
    "calyx_fn",
    "mean_maximum_softmax_confidence",
    "mean_normalized_entropy",
    "percent_confidence_below_0_50",
    "percent_confidence_below_0_75",
)
CUTLINE_FIELDS = (
    "sample_id",
    "method",
    "ground_truth_status",
    "ground_truth_failure_code",
    "ground_truth_structured_status",
    "predicted_status",
    "predicted_failure_code",
    "predicted_structured_status",
    "status_agreement",
    "structured_status_agreement",
    "ground_truth_coordinate",
    "predicted_coordinate",
    "signed_coordinate_difference_predicted_minus_ground_truth",
    "absolute_coordinate_difference",
    "within_2_pixels",
    "within_5_pixels",
    "within_10_pixels",
    "ground_truth_flesh_component_count",
    "ground_truth_calyx_component_count",
    "ground_truth_selected_flesh_component",
    "ground_truth_selected_calyx_component",
    "ground_truth_contact_pixel_count",
    "predicted_flesh_component_count",
    "predicted_calyx_component_count",
    "predicted_selected_flesh_component",
    "predicted_selected_calyx_component",
    "predicted_contact_pixel_count",
    "ground_truth_candidate_count",
    "ground_truth_feasible_candidate_count",
    "ground_truth_feasible_block_count",
    "ground_truth_selected_block_id",
    "ground_truth_selected_block_start",
    "ground_truth_selected_block_end",
    "ground_truth_selected_block_candidate_count",
    "ground_truth_selected_block_is_singleton",
    "predicted_candidate_count",
    "predicted_feasible_candidate_count",
    "predicted_feasible_block_count",
    "predicted_selected_block_id",
    "predicted_selected_block_start",
    "predicted_selected_block_end",
    "predicted_selected_block_candidate_count",
    "predicted_selected_block_is_singleton",
    "ground_truth_flesh_loss_proxy_pixel_count",
    "ground_truth_flesh_loss_proxy_ratio",
    "ground_truth_calyx_retention_proxy_pixel_count",
    "ground_truth_calyx_retention_proxy_ratio",
    "predicted_flesh_loss_proxy_pixel_count",
    "predicted_flesh_loss_proxy_ratio",
    "predicted_calyx_retention_proxy_pixel_count",
    "predicted_calyx_retention_proxy_ratio",
    "flesh_loss_proxy_pixel_count_change_predicted_minus_ground_truth",
    "flesh_loss_proxy_ratio_change_predicted_minus_ground_truth",
    "calyx_retention_proxy_pixel_count_change_predicted_minus_ground_truth",
    "calyx_retention_proxy_ratio_change_predicted_minus_ground_truth",
)

# Frozen e050_w064 configuration. These values are deliberately not CLI options.
FROZEN_FIXED_AXIS_CONFIGURATION: dict[str, Any] = {
    "configuration_id": COMMITTED_BASELINE_CONFIGURATION_ID,
    "removal_axis": [1.0, 0.0],
    "projection_quantile": 0.95,
    "support_band_width_pixels": 5.0,
    "calyx_dilation_radius": 1,
    "component_connectivity": 8,
    "signed_offset_pixels": 0.0,
    "candidate_step_pixels": 1.0,
    "inward_search_margin_pixels": 10.0,
    "outward_search_margin_pixels": 0.0,
    "blade_band_half_width_pixels": 1.0,
    "lateral_window_half_width_pixels": 64.0,
    "minimum_attachment_evidence_fraction": 0.50,
    "minimum_flesh_band_pixels": 1,
    "minimum_calyx_band_pixels": 1,
}

PILOT_LIMITATIONS = (
    "This is a preliminary new-batch evaluation on a seven-image pilot subset.",
    "Seven samples are too few for statistical proof of generalization.",
    "The segmentation checkpoint is evaluated without retraining or fine-tuning.",
    "Fixed-axis results measure agreement between geometry derived from predicted "
    "and manually labelled visible semantic masks; they are not physical cutting "
    "measurements.",
    "Only manually designated fixed-axis-eligible samples contribute to coordinate "
    "agreement aggregates.",
    "Visible-mask annotations cannot establish performance on hidden anatomy or "
    "production operating conditions.",
)


class PilotOverlapError(ValueError):
    """Raised when a pilot sample overlaps the reference and no override is set."""

    def __init__(self, audit: Mapping[str, Any]) -> None:
        super().__init__(
            "Pilot overlap audit failed: a sample ID or image SHA-256 matches the "
            "v1 reference. Pass allow_overlap=True or --allow-overlap only for an "
            "explicit research investigation."
        )
        self.audit = dict(audit)


def prepare_pilot_holdout_dataset(
    raw_labelled_dir: PathLike,
    normalized_output_dir: PathLike,
    *,
    expected_sample_count: int = DEFAULT_EXPECTED_SAMPLE_COUNT,
    reference_v1_root: PathLike | None = None,
    overwrite: bool = False,
    metadata_manifest_output: PathLike | None = None,
    allow_overlap: bool = False,
    hash_images: bool = True,
    hash_masks: bool = True,
    input_format: str = "auto",
    category_mapping: Mapping[CategoryId, str] | None = None,
    converter: Callable[..., dict[str, Any]] = convert_vt_export,
) -> dict[str, Any]:
    """Prepare a strict VisionTrain batch as a normalized local pilot dataset.

    Full exports are delegated unchanged to :func:`convert_vt_export`. Extracted
    stem-matched pairs reuse :func:`construct_mask`, including its VisionTrain
    brush decoding, class mapping, and Calyx-over-Flesh overlap behavior. This
    wrapper adds exact-pair validation, a dedicated ``pilot`` split, optional v1
    overlap auditing, sanitized metadata, and staged finalization. Source files
    are only read and are never modified.
    """

    expected = _positive_integer(expected_sample_count, name="expected_sample_count")
    source_root = Path(raw_labelled_dir)
    destination_root = Path(normalized_output_dir)
    reference_root = Path(reference_v1_root) if reference_v1_root is not None else None
    if not source_root.is_dir():
        raise FileNotFoundError(
            f"Raw labelled-pair directory does not exist: {source_root}"
        )
    _validate_directory_destination(
        destination_root,
        overwrite=overwrite,
        operation="preparation",
    )
    if _paths_overlap(source_root.resolve(), destination_root.resolve()):
        raise ValueError("Raw input and normalized output directories must not overlap")
    if reference_root is not None:
        if not reference_root.is_dir():
            raise FileNotFoundError(
                f"Reference v1 root does not exist: {reference_root}"
            )
        if _paths_overlap(reference_root.resolve(), destination_root.resolve()):
            raise ValueError(
                "Reference v1 and normalized output directories must not overlap"
            )

    selected_input_format, extracted_pairs, extracted_categories = _resolve_pilot_input(
        source_root,
        input_format=input_format,
        expected_sample_count=expected,
        category_mapping=category_mapping,
    )
    source_fingerprints = _source_fingerprints(source_root)
    if metadata_manifest_output is not None:
        metadata_output = Path(metadata_manifest_output)
        if _paths_overlap(source_root.resolve(), metadata_output.resolve()):
            raise ValueError("The metadata template must not be written into raw input")
        _validate_file_destination(
            metadata_output,
            overwrite=overwrite,
            operation="metadata template generation",
        )

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.staging-",
            dir=destination_root.parent,
        )
    )
    try:
        converted_root = staging_root / "_converted"
        if selected_input_format == "full-export":
            converted_manifest = converter(
                source_root,
                converted_root,
                overwrite=False,
            )
        else:
            assert extracted_pairs is not None
            assert extracted_categories is not None
            converted_manifest = _convert_extracted_pairs(
                extracted_pairs,
                converted_root,
                categories=extracted_categories,
            )
        converted_samples = converted_manifest.get("samples")
        if (
            not isinstance(converted_samples, list)
            or len(converted_samples) != expected
        ):
            raise ValueError(
                "Existing converter did not return the expected sample count"
            )

        image_output = staging_root / PILOT_SPLIT / "images"
        mask_output = staging_root / PILOT_SPLIT / "masks"
        image_output.mkdir(parents=True)
        mask_output.mkdir(parents=True)
        pilot_samples: list[dict[str, Any]] = []
        used_artifact_ids: set[str] = set()
        used_output_names: set[str] = set()
        for sample in sorted(
            converted_samples, key=lambda item: str(item["sample_id"])
        ):
            source_sample_id = sample.get("sample_id")
            sample_id = _sample_id_string(source_sample_id)
            source_image = _safe_manifest_file(converted_root, sample.get("image_path"))
            source_mask = _safe_manifest_file(converted_root, sample.get("mask_path"))
            image_name = source_image.name
            mask_name = source_mask.name
            output_key = Path(image_name).stem.casefold()
            if output_key in used_output_names:
                raise ValueError(
                    f"Duplicate pilot output stem is ambiguous: {image_name!r}"
                )
            used_output_names.add(output_key)
            artifact_id = _unique_artifact_id(sample_id, used_artifact_ids)
            target_image = image_output / image_name
            target_mask = mask_output / mask_name
            shutil.move(str(source_image), target_image)
            shutil.move(str(source_mask), target_mask)
            _validate_image_mask_shape(target_image, target_mask)
            pilot_samples.append(
                {
                    "sample_id": source_sample_id,
                    "artifact_id": artifact_id,
                    "source_image_filename": image_name,
                    "image_path": target_image.relative_to(staging_root).as_posix(),
                    "mask_path": target_mask.relative_to(staging_root).as_posix(),
                    "image_width": int(sample["image_width"]),
                    "image_height": int(sample["image_height"]),
                    "source_annotation_count": int(sample["source_annotation_count"]),
                    "flesh_annotation_count": int(sample["flesh_annotation_count"]),
                    "calyx_annotation_count": int(sample["calyx_annotation_count"]),
                    "background_pixel_count": int(sample["background_pixel_count"]),
                    "flesh_pixel_count": int(sample["flesh_pixel_count"]),
                    "calyx_pixel_count": int(sample["calyx_pixel_count"]),
                    "cross_class_overlap_pixel_count": int(
                        sample["cross_class_overlap_pixel_count"]
                    ),
                }
            )
        shutil.rmtree(converted_root)

        manifest: dict[str, Any] = {
            "dataset_name": PILOT_DATASET_IDENTIFIER,
            "dataset_kind": PILOT_DATASET_KIND,
            "dataset_version": PILOT_DATASET_VERSION,
            "evaluation_role": STUDY_NAME,
            "pilot_subset_name": PILOT_SUBSET_NAME,
            "is_pilot_dataset": True,
            "is_v1_or_v2_dataset": False,
            "normalized_split": PILOT_SPLIT,
            "source_input_format": selected_input_format,
            "class_mapping": dict(CLASS_MAPPING),
            "overlap_rule": OVERLAP_RULE,
            "expected_sample_count": expected,
            "sample_count": len(pilot_samples),
            "samples": pilot_samples,
        }
        _write_json(staging_root / "manifest.json", manifest)
        audit = audit_pilot_overlap(
            staging_root,
            reference_root,
            hash_images=hash_images,
            hash_masks=hash_masks,
            allow_overlap=allow_overlap,
        )
        _write_json(staging_root / "overlap_audit.json", audit)
        summary = {
            "study": STUDY_NAME,
            "pilot_sample_count": len(pilot_samples),
            "expected_sample_count": expected,
            "class_mapping": dict(CLASS_MAPPING),
            "overlap_rule": OVERLAP_RULE,
            "reference_v1_supplied": reference_root is not None,
            "overlap_override_used": audit["overlap_override_used"],
            "source_files_unchanged": True,
            "normalized_split": PILOT_SPLIT,
            "source_input_format": selected_input_format,
        }
        manifest["overlap_audit_path"] = "overlap_audit.json"
        manifest["preparation_summary"] = summary
        _assert_no_absolute_paths(manifest)
        _assert_no_absolute_paths(audit)
        _write_json(staging_root / "manifest.json", manifest)
        _install_staged_directory(staging_root, destination_root, overwrite=overwrite)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    if _source_fingerprints(source_root) != source_fingerprints:
        raise RuntimeError("Source files changed during pilot preparation")
    if metadata_manifest_output is not None:
        write_pilot_metadata_template(
            destination_root,
            metadata_manifest_output,
            overwrite=overwrite,
        )
    return {"manifest": manifest, "overlap_audit": audit, "summary": summary}


def audit_pilot_overlap(
    pilot_root: PathLike,
    reference_v1_root: PathLike | None,
    *,
    hash_images: bool = True,
    hash_masks: bool = True,
    allow_overlap: bool = False,
) -> dict[str, Any]:
    """Audit pilot/reference identity and content overlap without exposing images."""

    pilot_path = Path(pilot_root)
    pilot_manifest = _load_pilot_manifest(pilot_path)
    pilot_samples = _manifest_samples(pilot_manifest, context="pilot manifest")
    audit: dict[str, Any] = {
        "audit_role": "pilot holdout overlap audit",
        "pilot_sample_count": len(pilot_samples),
        "reference_v1_supplied": reference_v1_root is not None,
        "v1_reference_count": None,
        "image_sha256_enabled": bool(hash_images),
        "mask_sha256_enabled": bool(hash_masks),
        "matching_sample_ids": [],
        "matching_original_filenames": [],
        "matching_normalized_relative_filenames": [],
        "identical_image_hashes": [],
        "identical_mask_hashes": [],
        "missing_comparisons": [],
        "ambiguous_comparisons": [],
        "overlap_override_used": False,
        "overlap_override_warning": None,
        "hard_fail_reasons": [],
    }
    if reference_v1_root is None:
        audit["missing_comparisons"].append("reference_v1_root_not_supplied")
        audit["hard_fail_triggered"] = False
        return audit

    reference_path = Path(reference_v1_root)
    if not reference_path.is_dir():
        raise FileNotFoundError(f"Reference v1 root does not exist: {reference_path}")
    reference_manifest = _load_json(reference_path / "manifest.json")
    if not isinstance(reference_manifest, Mapping):
        raise ValueError("Reference v1 manifest.json must contain a JSON object")
    reference_samples = _manifest_samples(
        reference_manifest, context="reference manifest"
    )
    audit["v1_reference_count"] = len(reference_samples)

    pilot_descriptors = _overlap_descriptors(
        pilot_path,
        pilot_samples,
        hash_images=hash_images,
        hash_masks=hash_masks,
        missing=audit["missing_comparisons"],
    )
    reference_descriptors = _overlap_descriptors(
        reference_path,
        reference_samples,
        hash_images=hash_images,
        hash_masks=hash_masks,
        missing=audit["missing_comparisons"],
    )
    audit["matching_sample_ids"] = _matching_values(
        pilot_descriptors,
        reference_descriptors,
        key="sample_id",
        casefold=False,
    )
    audit["matching_original_filenames"] = _matching_values(
        pilot_descriptors,
        reference_descriptors,
        key="original_filename",
        casefold=True,
    )
    audit["matching_normalized_relative_filenames"] = _matching_values(
        pilot_descriptors,
        reference_descriptors,
        key="normalized_relative_filename",
        casefold=True,
    )
    if hash_images:
        audit["identical_image_hashes"], ambiguous = _matching_hashes(
            pilot_descriptors,
            reference_descriptors,
            key="image_sha256",
        )
        audit["ambiguous_comparisons"].extend(ambiguous)
    if hash_masks:
        audit["identical_mask_hashes"], ambiguous = _matching_hashes(
            pilot_descriptors,
            reference_descriptors,
            key="mask_sha256",
        )
        audit["ambiguous_comparisons"].extend(ambiguous)

    if audit["matching_sample_ids"]:
        audit["hard_fail_reasons"].append("matching_sample_id")
    if audit["identical_image_hashes"]:
        audit["hard_fail_reasons"].append("identical_image_sha256")
    should_fail = bool(audit["hard_fail_reasons"])
    audit["hard_fail_triggered"] = should_fail and not allow_overlap
    if should_fail and allow_overlap:
        audit["overlap_override_used"] = True
        audit["overlap_override_warning"] = (
            "EXPLICIT RESEARCH OVERRIDE: pilot/reference overlap was detected; "
            "results must not be interpreted as independent holdout evidence."
        )
    _assert_no_absolute_paths(audit)
    if should_fail and not allow_overlap:
        raise PilotOverlapError(audit)
    return audit


def write_pilot_metadata_template(
    dataset_root: PathLike,
    output_path: PathLike,
    *,
    overwrite: bool = False,
) -> Path:
    """Write an overwrite-safe CSV template containing every pilot sample."""

    root = Path(dataset_root)
    manifest = _load_pilot_manifest(root)
    samples = _manifest_samples(manifest, context="pilot manifest")
    destination = Path(output_path)
    _validate_file_destination(
        destination,
        overwrite=overwrite,
        operation="metadata template generation",
    )
    rows = [
        {
            "sample_id": _sample_id_string(sample.get("sample_id")),
            "orientation_category": "unknown",
            "canonical_fixed_axis_pose": "",
            "fixed_axis_eligible": "",
            "occlusion_category": "unknown",
            "annotation_qa_status": "unknown",
            "notes": "",
        }
        for sample in samples
    ]
    _write_csv_atomic(destination, METADATA_FIELDS, rows, overwrite=overwrite)
    return destination


def read_pilot_metadata(
    metadata_manifest: PathLike,
    *,
    expected_sample_ids: Sequence[str],
) -> list[dict[str, str]]:
    """Read, sanitize, and validate the manually reviewed pilot metadata CSV."""

    path = Path(metadata_manifest)
    if not path.is_file():
        raise FileNotFoundError(f"Pilot metadata manifest does not exist: {path}")
    with path.open("r", encoding="utf-8-sig", newline="") as input_file:
        reader = csv.DictReader(input_file)
        if tuple(reader.fieldnames or ()) != METADATA_FIELDS:
            raise ValueError(
                "Pilot metadata columns must exactly match: "
                + ", ".join(METADATA_FIELDS)
            )
        raw_rows = list(reader)

    rows: list[dict[str, str]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows, start=2):
        row = {field: (raw.get(field) or "").strip() for field in METADATA_FIELDS}
        sample_id = row["sample_id"]
        _validate_sample_identifier(sample_id, context=f"metadata row {index}")
        if sample_id in seen:
            raise ValueError(f"Duplicate metadata sample_id {sample_id!r}")
        seen.add(sample_id)
        orientation = row["orientation_category"].lower()
        if not orientation:
            orientation = "unknown"
        if orientation not in SUPPORTED_ORIENTATION_CATEGORIES:
            raise ValueError(
                f"metadata row {index} orientation_category must be one of "
                f"{', '.join(SUPPORTED_ORIENTATION_CATEGORIES)}"
            )
        row["orientation_category"] = orientation
        eligibility = row["fixed_axis_eligible"].lower()
        if eligibility not in FIXED_AXIS_ELIGIBILITY_STATUSES:
            raise ValueError(
                "fixed_axis_eligible must be blank, unknown, false, or true"
            )
        row["fixed_axis_eligible"] = eligibility
        qa_status = row["annotation_qa_status"].lower()
        if qa_status and not re.fullmatch(r"[a-z0-9_-]+", qa_status):
            raise ValueError(
                "annotation_qa_status must be blank or a path-free status token"
            )
        row["annotation_qa_status"] = qa_status
        for field, value in row.items():
            if _looks_like_absolute_path(value):
                raise ValueError(
                    f"metadata row {index} field {field!r} must not contain an absolute path"
                )
        rows.append(row)

    expected = list(expected_sample_ids)
    if len(set(expected)) != len(expected):
        raise ValueError("Expected pilot sample IDs must be unique")
    missing = sorted(set(expected) - seen)
    unexpected = sorted(seen - set(expected))
    if missing or unexpected:
        parts = []
        if missing:
            parts.append("missing sample IDs: " + ", ".join(missing))
        if unexpected:
            parts.append("unexpected sample IDs: " + ", ".join(unexpected))
        raise ValueError("Pilot metadata sample mismatch; " + "; ".join(parts))
    by_id = {row["sample_id"]: row for row in rows}
    return [by_id[sample_id] for sample_id in expected]


def fixed_axis_eligibility(row: Mapping[str, str]) -> tuple[bool, str | None]:
    """Resolve explicit pilot eligibility without inferring from names or position."""

    status = (row.get("fixed_axis_eligible") or "").strip().lower()
    orientation = (row.get("orientation_category") or "unknown").strip().lower()
    reasons: list[str] = []
    if status != "true":
        reasons.append(
            {
                "": "fixed_axis_eligibility_blank",
                "unknown": "fixed_axis_eligibility_unknown",
                "false": "fixed_axis_eligibility_false",
            }.get(status, "fixed_axis_eligibility_not_explicit_true")
        )
    if orientation in {"", "unknown"}:
        reasons.append("orientation_unknown")
    if reasons:
        return False, ";".join(reasons)
    return True, None


def evaluate_pilot_holdout(
    dataset_root: PathLike,
    checkpoint_path: PathLike,
    output_root: PathLike,
    metadata_manifest: PathLike,
    *,
    expected_sample_count: int = DEFAULT_EXPECTED_SAMPLE_COUNT,
    device: str | torch.device = "auto",
    batch_size: int = 1,
    num_workers: int = 0,
    overwrite: bool = False,
    model_factory: ModelFactory | None = None,
    v2a_estimator: V2aEstimator = estimate_fixed_axis_cutline,
    v2b_estimator: V2bEstimator = estimate_fixed_axis_search_cutline,
    visualization_function: PilotVisualizationFunction = (
        create_pilot_holdout_visualization
    ),
) -> dict[str, Any]:
    """Evaluate segmentation and fixed-axis mask-geometry agreement for the pilot."""

    expected = _positive_integer(expected_sample_count, name="expected_sample_count")
    batch_size = _positive_integer(batch_size, name="batch_size")
    if (
        isinstance(num_workers, bool)
        or not isinstance(num_workers, int)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")
    source_root = Path(dataset_root)
    checkpoint_file = Path(checkpoint_path)
    destination_root = Path(output_root)
    metadata_file = Path(metadata_manifest)
    _validate_evaluation_paths(
        source_root,
        checkpoint_file,
        destination_root,
        metadata_file,
        overwrite=overwrite,
    )
    manifest = _load_pilot_manifest(source_root)
    samples = _manifest_samples(manifest, context="pilot manifest")
    if len(samples) != expected or manifest.get("sample_count") != expected:
        raise ValueError(
            f"Pilot dataset must contain exactly {expected} samples, got {len(samples)}"
        )
    expected_ids = [_sample_id_string(sample.get("sample_id")) for sample in samples]
    metadata_rows = read_pilot_metadata(
        metadata_file,
        expected_sample_ids=expected_ids,
    )
    metadata_by_id = {row["sample_id"]: row for row in metadata_rows}
    sample_by_image = _sample_lookup(source_root, samples)

    resolved_device = resolve_device(device)
    model, checkpoint = load_segmentation_checkpoint(
        checkpoint_file,
        device=resolved_device,
        model_factory=model_factory,
    )
    if checkpoint["completed_epoch"] != EXPECTED_CANONICAL_CHECKPOINT_EPOCH:
        raise ValueError(
            "Pilot evaluation requires the canonical epoch-92 checkpoint; "
            f"received completed_epoch={checkpoint['completed_epoch']}"
        )
    height = int(checkpoint["input_height"])
    width = int(checkpoint["input_width"])
    dataset = SegmentationDataset(
        source_root / PILOT_SPLIT / "images",
        source_root / PILOT_SPLIT / "masks",
        image_transform=partial(preprocess_rgb_image, height=height, width=width),
        mask_transform=partial(preprocess_class_mask, height=height, width=width),
    )
    if len(dataset) != expected:
        raise ValueError(
            f"Pilot normalized folders contain {len(dataset)} pairs, expected {expected}"
        )
    for index in range(len(dataset)):
        dataset[index]
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.staging-",
            dir=destination_root.parent,
        )
    )
    try:
        visualization_root = staging_root / "visualizations"
        visualization_root.mkdir()
        global_matrix = torch.zeros((3, 3), dtype=torch.long)
        segmentation_rows: list[dict[str, Any]] = []
        cutline_rows: list[dict[str, Any]] = []
        excluded_rows: list[dict[str, str]] = []
        output_samples: list[dict[str, Any]] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(resolved_device)
                targets = batch["mask"].to(resolved_device)
                logits = _model_logits(model(images))
                _validate_logits(logits, images)
                _, confidence, entropy = confidence_and_entropy(logits)
                predictions = logits.argmax(dim=1)
                global_matrix += confusion_matrix(predictions, targets, 3).cpu()

                for index, image_path_text in enumerate(batch["image_path"]):
                    image_path = Path(image_path_text).resolve()
                    if image_path not in sample_by_image:
                        raise ValueError(
                            "Normalized image is not represented in pilot manifest"
                        )
                    sample = sample_by_image[image_path]
                    sample_id = _sample_id_string(sample["sample_id"])
                    artifact_id = _artifact_id(sample)
                    metadata = metadata_by_id[sample_id]
                    target = targets[index].cpu()
                    prediction = predictions[index].cpu()
                    sample_confidence = confidence[index].cpu()
                    sample_entropy = entropy[index].cpu()
                    sample_metrics = per_image_metrics(
                        prediction,
                        target,
                        sample_confidence,
                        sample_entropy,
                    )
                    segmentation_rows.append({"sample_id": sample_id, **sample_metrics})

                    eligible, exclusion_reason = fixed_axis_eligibility(metadata)
                    ground_truth_v2a = None
                    ground_truth_v2b = None
                    predicted_v2a = None
                    predicted_v2b = None
                    if eligible:
                        ground_truth_v2a, ground_truth_v2b = _run_frozen_geometry(
                            target.numpy(),
                            v2a_estimator=v2a_estimator,
                            v2b_estimator=v2b_estimator,
                        )
                        predicted_v2a, predicted_v2b = _run_frozen_geometry(
                            prediction.numpy(),
                            v2a_estimator=v2a_estimator,
                            v2b_estimator=v2b_estimator,
                        )
                        cutline_rows.extend(
                            _cutline_comparison_rows(
                                sample_id,
                                target.numpy(),
                                prediction.numpy(),
                                ground_truth_v2a,
                                ground_truth_v2b,
                                predicted_v2a,
                                predicted_v2b,
                            )
                        )
                    else:
                        assert exclusion_reason is not None
                        excluded_rows.append(
                            {
                                "sample_id": sample_id,
                                "orientation_category": metadata[
                                    "orientation_category"
                                ],
                                "fixed_axis_eligible": metadata["fixed_axis_eligible"],
                                "exclusion_reason": exclusion_reason,
                            }
                        )

                    visualization = visualization_function(
                        _image_tensor_to_rgb(images[index].cpu()),
                        target.numpy(),
                        prediction.numpy(),
                        sample_confidence.numpy(),
                        sample_entropy.numpy(),
                        sample_id=sample_id,
                        metadata=metadata,
                        segmentation_metrics=sample_metrics,
                        fixed_axis_eligible=eligible,
                        ground_truth_v2a=ground_truth_v2a,
                        ground_truth_v2b=ground_truth_v2b,
                        predicted_v2a=predicted_v2a,
                        predicted_v2b=predicted_v2b,
                    )
                    if not isinstance(visualization, Image.Image):
                        raise TypeError(
                            "visualization_function must return a PIL image"
                        )
                    visualization_path = Path("visualizations") / f"{artifact_id}.png"
                    visualization.save(staging_root / visualization_path, format="PNG")
                    output_samples.append(
                        {
                            "sample_id": sample_id,
                            "fixed_axis_eligible": eligible,
                            "visualization_path": visualization_path.as_posix(),
                        }
                    )

        if len(segmentation_rows) != expected:
            raise RuntimeError("Not every pilot sample received segmentation metrics")
        eligible_count = sum(sample["fixed_axis_eligible"] for sample in output_samples)
        if len(cutline_rows) != eligible_count * 2:
            raise RuntimeError("Eligible samples must receive one v2a and one v2b row")

        global_metrics = metrics_from_confusion_matrix(global_matrix)
        agreement = {
            method: coordinate_agreement_summary(cutline_rows, method=method)
            for method in ("v2a", "v2b")
        }
        structured_failures = _structured_failure_counts(cutline_rows)
        orientation_distribution = Counter(
            row["orientation_category"] or "unknown" for row in metadata_rows
        )
        annotation_qa_distribution = Counter(
            row["annotation_qa_status"] or "unknown" for row in metadata_rows
        )
        summary = {
            "study": STUDY_NAME,
            "pilot_subset_name": PILOT_SUBSET_NAME,
            "total_sample_count": expected,
            "segmentation_evaluated_count": len(segmentation_rows),
            "fixed_axis_eligible_count": eligible_count,
            "fixed_axis_excluded_count": len(excluded_rows),
            "orientation_distribution": dict(sorted(orientation_distribution.items())),
            "annotation_qa_distribution": dict(
                sorted(annotation_qa_distribution.items())
            ),
            "global_segmentation_metrics": global_metrics,
            "v2a_agreement_metrics": agreement["v2a"],
            "v2b_agreement_metrics": agreement["v2b"],
            "structured_failure_counts": structured_failures,
            "fixed_axis_coordinate_difference_convention": (
                "predicted-mask coordinate minus ground-truth-mask coordinate"
            ),
            "fixed_axis_coordinate_scope": (
                "agreement between geometry derived from predicted and manually "
                "labelled visible masks at checkpoint evaluation resolution"
            ),
            "pilot_study_limitations": list(PILOT_LIMITATIONS),
        }
        overlap_audit = _load_overlap_audit(source_root)
        checkpoint_identity = {
            "identifier": checkpoint_file.name,
            "sha256": _sha256(checkpoint_file),
            "model_name": checkpoint["model_name"],
            "completed_epoch": checkpoint["completed_epoch"],
            "stored_best_mean_foreground_iou": checkpoint["best_mean_foreground_iou"],
        }
        output_manifest = {
            "study": STUDY_NAME,
            "pilot_subset_name": PILOT_SUBSET_NAME,
            "dataset_identifier": manifest["dataset_name"],
            "dataset_kind": PILOT_DATASET_KIND,
            "sample_count": expected,
            "class_mapping": dict(CLASS_MAPPING),
            "checkpoint": checkpoint_identity,
            "evaluation_configuration": {
                "device_type": resolved_device.type,
                "batch_size": batch_size,
                "num_workers": num_workers,
                "input_height": height,
                "input_width": width,
                "rgb_resize": "bilinear with antialiasing",
                "class_mask_resize": "nearest-neighbour",
                "automatic_orientation_normalization": False,
                "fixed_axis_configuration": dict(FROZEN_FIXED_AXIS_CONFIGURATION),
                "retraining_or_fine_tuning": False,
            },
            "algorithms": {
                "v2a": {
                    "name": V2A_ALGORITHM_NAME,
                    "version": V2A_ALGORITHM_VERSION,
                },
                "v2b": {
                    "name": V2B_ALGORITHM_NAME,
                    "version": V2B_ALGORITHM_VERSION,
                },
            },
            "artifacts": {
                "manifest": "manifest.json",
                "summary": "summary.json",
                "overlap_audit": "overlap_audit.json",
                "pilot_metadata_snapshot": "pilot_metadata_snapshot.csv",
                "segmentation_per_image": "segmentation_per_image.csv",
                "segmentation_confusion_matrix": ("segmentation_confusion_matrix.csv"),
                "cutline_per_image": "cutline_per_image.csv",
                "excluded_from_fixed_axis": "excluded_from_fixed_axis.csv",
                "visualizations": "visualizations",
            },
            "samples": output_samples,
        }
        _assert_no_absolute_paths(output_manifest)
        _assert_no_absolute_paths(summary)
        _assert_no_absolute_paths(overlap_audit)
        _write_json(staging_root / "manifest.json", output_manifest)
        _write_json(staging_root / "summary.json", summary)
        _write_json(staging_root / "overlap_audit.json", overlap_audit)
        _write_csv(
            staging_root / "pilot_metadata_snapshot.csv", METADATA_FIELDS, metadata_rows
        )
        _write_csv(
            staging_root / "segmentation_per_image.csv",
            SEGMENTATION_FIELDS,
            segmentation_rows,
        )
        _write_confusion_matrix(
            staging_root / "segmentation_confusion_matrix.csv",
            global_matrix,
        )
        _write_csv(staging_root / "cutline_per_image.csv", CUTLINE_FIELDS, cutline_rows)
        _write_csv(
            staging_root / "excluded_from_fixed_axis.csv",
            EXCLUSION_FIELDS,
            excluded_rows,
        )
        _install_staged_directory(staging_root, destination_root, overwrite=overwrite)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    return {
        "manifest": _json_ready(output_manifest),
        "summary": _json_ready(summary),
        "segmentation_rows": _json_ready(segmentation_rows),
        "cutline_rows": _json_ready(cutline_rows),
        "excluded_rows": excluded_rows,
    }


def coordinate_agreement_summary(
    rows: Sequence[Mapping[str, Any]],
    *,
    method: str,
) -> dict[str, Any]:
    """Summarize successful finite predicted-minus-ground-truth coordinates."""

    if method not in {"v2a", "v2b"}:
        raise ValueError("method must be 'v2a' or 'v2b'")
    method_rows = [row for row in rows if row.get("method") == method]
    signed = [
        float(row["signed_coordinate_difference_predicted_minus_ground_truth"])
        for row in method_rows
        if _finite_number(
            row.get("signed_coordinate_difference_predicted_minus_ground_truth")
        )
    ]
    absolute = [abs(value) for value in signed]
    denominator = len(absolute)
    result: dict[str, Any] = {
        "eligible_sample_count": len(method_rows),
        "successful_coordinate_pair_count": denominator,
        "failed_coordinate_pair_count": len(method_rows) - denominator,
        "successful_coordinate_pair_proportion": (
            denominator / len(method_rows) if method_rows else None
        ),
        "status_agreement_count": sum(
            row.get("status_agreement") is True for row in method_rows
        ),
        "structured_status_agreement_count": sum(
            row.get("structured_status_agreement") is True for row in method_rows
        ),
        "mean_signed_coordinate_difference": mean(signed) if signed else None,
        "median_signed_coordinate_difference": median(signed) if signed else None,
        "mean_absolute_coordinate_difference": mean(absolute) if absolute else None,
        "median_absolute_coordinate_difference": median(absolute) if absolute else None,
        "maximum_absolute_coordinate_difference": max(absolute) if absolute else None,
        "finite_error_denominator_definition": (
            "successful finite ground-truth/predicted coordinate pairs only"
        ),
    }
    result["status_agreement_proportion"] = (
        result["status_agreement_count"] / len(method_rows) if method_rows else None
    )
    result["structured_status_agreement_proportion"] = (
        result["structured_status_agreement_count"] / len(method_rows)
        if method_rows
        else None
    )
    for threshold in (2, 5, 10):
        count = sum(value <= threshold for value in absolute)
        result[f"within_{threshold}_pixels"] = {
            "count": count,
            "denominator": denominator,
            "proportion": count / denominator if denominator else None,
        }
    return result


def _resolve_pilot_input(
    source_root: Path,
    *,
    input_format: str,
    expected_sample_count: int,
    category_mapping: Mapping[CategoryId, str] | None,
) -> tuple[
    str,
    list[dict[str, Any]] | None,
    dict[CategoryId, str] | None,
]:
    if input_format not in INPUT_FORMATS:
        raise ValueError(
            f"input_format must be one of {', '.join(INPUT_FORMATS)}, "
            f"got {input_format!r}"
        )
    has_dataset_manifest = (source_root / "dataset.json").is_file()
    if input_format == "auto":
        selected = "full-export" if has_dataset_manifest else "extracted-pairs"
    else:
        selected = input_format

    if selected == "full-export":
        if not has_dataset_manifest:
            raise FileNotFoundError(
                "Full-export input requires <raw-labelled-dir>/dataset.json"
            )
        if category_mapping is not None:
            raise ValueError(
                "category_mapping/--category-map applies only to extracted-pairs input"
            )
        _validate_full_export_input(
            source_root,
            expected_sample_count=expected_sample_count,
        )
        return selected, None, None

    if has_dataset_manifest:
        raise ValueError(
            "Extracted-pairs input must not contain dataset.json; use full-export "
            "or auto input format"
        )
    pairs, categories = _validate_extracted_pairs(
        source_root,
        expected_sample_count=expected_sample_count,
        category_mapping=category_mapping,
    )
    return selected, pairs, categories


def _validate_full_export_input(
    source_root: Path,
    *,
    expected_sample_count: int,
) -> list[dict[str, Any]]:
    dataset_path = source_root / "dataset.json"
    if not dataset_path.is_file():
        raise FileNotFoundError(
            "Full-export input requires <raw-labelled-dir>/dataset.json"
        )
    dataset = _load_json(dataset_path)
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset.json must contain a JSON object")
    raw_images = dataset.get("images")
    if not isinstance(raw_images, list):
        raise ValueError("dataset.json images must be a JSON list")
    records = _validate_manifest_images(raw_images)
    if len(records) != len(raw_images):
        raise ValueError("Pilot raw directory must contain labelled samples only")
    if len(records) != expected_sample_count:
        raise ValueError(
            f"Expected exactly {expected_sample_count} labelled samples, got {len(records)}"
        )

    image_names = [record["file_name"] for record in records]
    annotation_names = [f"{Path(name).stem}.json" for name in image_names]
    if len({name.casefold() for name in annotation_names}) != len(annotation_names):
        raise ValueError("Duplicate image stems produce ambiguous annotation pairs")
    expected_names = {"dataset.json", *image_names, *annotation_names}
    entries = list(source_root.iterdir())
    actual_names = {entry.name for entry in entries}
    directories = sorted(entry.name for entry in entries if not entry.is_file())
    if directories:
        raise ValueError("Unexpected non-file entries: " + ", ".join(directories))
    missing_images = sorted(name for name in image_names if name not in actual_names)
    missing_annotations = sorted(
        name for name in annotation_names if name not in actual_names
    )
    if missing_images:
        raise ValueError("Missing image file(s): " + ", ".join(missing_images))
    if missing_annotations:
        raise ValueError(
            "Missing annotation file(s): " + ", ".join(missing_annotations)
        )
    unexpected = sorted(actual_names - expected_names)
    if unexpected:
        raise ValueError("Unexpected extra file(s): " + ", ".join(unexpected))
    return records


def _validate_extracted_pairs(
    source_root: Path,
    *,
    expected_sample_count: int,
    category_mapping: Mapping[CategoryId, str] | None,
) -> tuple[list[dict[str, Any]], dict[CategoryId, str]]:
    entries = sorted(source_root.iterdir(), key=lambda path: path.name.casefold())
    non_files = [entry.name for entry in entries if not entry.is_file()]
    if non_files:
        raise ValueError(
            "Unexpected non-file entries in extracted-pairs input: "
            + ", ".join(non_files)
        )
    content_entries = [
        path
        for path in entries
        if path.name.casefold() not in HARMLESS_PLATFORM_FILENAMES
    ]
    image_paths = [
        path
        for path in content_entries
        if path.suffix.lower() in EXTRACTED_IMAGE_SUFFIXES
    ]
    annotation_paths = [
        path for path in content_entries if path.suffix.lower() == ".json"
    ]
    recognized = set(image_paths) | set(annotation_paths)
    unexpected = [path.name for path in content_entries if path not in recognized]
    if unexpected:
        raise ValueError(
            "Unexpected unrelated file(s) in extracted-pairs input: "
            + ", ".join(unexpected)
        )

    images_by_stem = _unique_paths_by_stem(image_paths, kind="extracted image")
    annotations_by_stem = _unique_paths_by_stem(
        annotation_paths,
        kind="extracted annotation JSON",
    )
    image_stems = set(images_by_stem)
    annotation_stems = set(annotations_by_stem)
    missing_annotations = sorted(image_stems - annotation_stems)
    missing_images = sorted(annotation_stems - image_stems)
    pairing_errors = []
    if missing_annotations:
        pairing_errors.append(
            "Missing extracted annotation JSON for image stem(s): "
            + ", ".join(missing_annotations)
        )
    if missing_images:
        pairing_errors.append(
            "Missing extracted image for annotation JSON stem(s), or unrelated "
            "unexpected JSON: " + ", ".join(missing_images)
        )
    if pairing_errors:
        raise ValueError("; ".join(pairing_errors))
    if len(image_stems) != expected_sample_count:
        raise ValueError(
            f"Expected exactly {expected_sample_count} extracted image/annotation "
            f"pairs, got {len(image_stems)}"
        )

    loaded_pairs: list[dict[str, Any]] = []
    for stem in sorted(image_stems, key=lambda value: (value.casefold(), value)):
        image_path = images_by_stem[stem]
        annotation_path = annotations_by_stem[stem]
        annotations = _load_json(annotation_path)
        if not isinstance(annotations, list):
            raise ValueError(
                f"Extracted annotation {annotation_path.name!r} must contain a JSON list"
            )
        loaded_pairs.append(
            {
                "sample_id": image_path.stem,
                "image_path": image_path,
                "annotation_path": annotation_path,
                "annotations": annotations,
            }
        )

    categories = _extracted_category_mapping(
        loaded_pairs,
        explicit_mapping=category_mapping,
    )
    for pair in loaded_pairs:
        pair["annotations"] = _normalize_extracted_annotations(
            pair["annotations"],
            image_name=pair["image_path"].name,
            categories=categories,
            explicit_mapping_supplied=category_mapping is not None,
        )
    return loaded_pairs, categories


def _unique_paths_by_stem(paths: Sequence[Path], *, kind: str) -> dict[str, Path]:
    grouped: dict[str, list[Path]] = {}
    for path in paths:
        grouped.setdefault(path.stem.casefold(), []).append(path)
    duplicates = sorted(stem for stem, matches in grouped.items() if len(matches) > 1)
    if duplicates:
        raise ValueError(
            f"Duplicate sample stem(s) are ambiguous for {kind}: "
            + ", ".join(duplicates)
        )
    return {matches[0].stem: matches[0] for matches in grouped.values()}


def _extracted_category_mapping(
    pairs: Sequence[Mapping[str, Any]],
    *,
    explicit_mapping: Mapping[CategoryId, str] | None,
) -> dict[CategoryId, str]:
    if explicit_mapping is not None:
        try:
            return _validate_category_mapping(explicit_mapping)
        except ValueError as error:
            raise ValueError(
                f"Invalid explicit extracted category mapping: {error}"
            ) from error

    derived: dict[CategoryId, str] = {}
    for pair in pairs:
        annotations = pair["annotations"]
        assert isinstance(annotations, list)
        for index, annotation in enumerate(annotations):
            if not isinstance(annotation, Mapping):
                raise ValueError(
                    f"Annotation {index} in {pair['annotation_path'].name!r} "
                    "must be a JSON object"
                )
            if "category_id" not in annotation:
                raise ValueError(
                    f"Annotation {index} in {pair['annotation_path'].name!r} is "
                    "missing category_id; category IDs cannot be guessed"
                )
            category_id = _category_id(
                annotation["category_id"],
                field=f"annotation {index} category_id",
            )
            if "labelname" not in annotation:
                raise ValueError(
                    f"Annotation {index} in {pair['annotation_path'].name!r} is "
                    "missing labelname; supply explicit --category-map SOURCE_ID=LABEL"
                )
            label = _required_string(
                annotation,
                "labelname",
                context=f"annotation {index}",
            )
            previous = derived.get(category_id)
            if previous is not None and previous != label:
                raise ValueError(
                    f"Category ID {category_id!r} maps to both {previous!r} and "
                    f"{label!r} in extracted annotations"
                )
            derived[category_id] = label
    if not derived:
        raise ValueError(
            "Extracted annotations contain no category metadata; supply explicit "
            "--category-map SOURCE_ID=LABEL values"
        )
    try:
        return _validate_category_mapping(derived)
    except ValueError as error:
        raise ValueError(
            f"Inconsistent category metadata in extracted annotations: {error}"
        ) from error


def _normalize_extracted_annotations(
    annotations: Sequence[Any],
    *,
    image_name: str,
    categories: Mapping[CategoryId, str],
    explicit_mapping_supplied: bool,
) -> list[dict[str, Any]]:
    normalized = []
    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise ValueError(f"Annotation {index} must be a JSON object")
        copied = dict(annotation)
        category_id = _category_id(
            _required_value(copied, "category_id", context=f"annotation {index}"),
            field=f"annotation {index} category_id",
        )
        if "labelname" not in copied:
            if not explicit_mapping_supplied or category_id not in categories:
                raise ValueError(
                    f"Annotation {index} is missing labelname; supply an explicit "
                    "--category-map entry for its category_id"
                )
            copied["labelname"] = categories[category_id]
        if "image_name" not in copied:
            copied["image_name"] = image_name
        else:
            annotation_image_name = _required_string(
                copied,
                "image_name",
                context=f"annotation {index}",
            )
            if annotation_image_name != image_name:
                raise ValueError(
                    "Ambiguous extracted pairing: annotation image_name "
                    f"{annotation_image_name!r} does not match stem-paired image "
                    f"{image_name!r}"
                )
        normalized.append(copied)
    return normalized


def _convert_extracted_pairs(
    pairs: Sequence[Mapping[str, Any]],
    output_root: Path,
    *,
    categories: Mapping[CategoryId, str],
) -> dict[str, Any]:
    image_output = output_root / "val" / "images"
    mask_output = output_root / "val" / "masks"
    image_output.mkdir(parents=True)
    mask_output.mkdir(parents=True)
    samples = []
    for pair in pairs:
        image_path = pair["image_path"]
        annotations = pair["annotations"]
        assert isinstance(image_path, Path)
        assert isinstance(annotations, list)
        try:
            with Image.open(image_path) as image:
                width, height = image.size
                source_format = image.format
                image.verify()
        except (OSError, ValueError) as error:
            raise ValueError(
                f"Extracted source image is invalid: {image_path.name}"
            ) from error
        expected_format = EXTRACTED_IMAGE_FORMATS[image_path.suffix.lower()]
        if source_format != expected_format:
            raise ValueError(
                f"Extracted source image {image_path.name!r} has suffix "
                f"{image_path.suffix!r} but is not a valid {expected_format} file"
            )
        mask, summary = construct_mask(
            annotations,
            image_width=width,
            image_height=height,
            categories=categories,
            image_name=image_path.name,
        )
        relative_image = Path("val") / "images" / image_path.name
        relative_mask = Path("val") / "masks" / f"{image_path.stem}.png"
        output_image = output_root / relative_image
        output_mask = output_root / relative_mask
        shutil.copyfile(image_path, output_image)
        Image.fromarray(mask).save(output_mask, format="PNG")
        _validate_saved_mask(
            output_mask,
            expected_width=width,
            expected_height=height,
        )
        samples.append(
            {
                "sample_id": pair["sample_id"],
                "split": "val",
                "image_path": relative_image.as_posix(),
                "mask_path": relative_mask.as_posix(),
                "image_width": width,
                "image_height": height,
                **summary,
            }
        )
    source_categories = [
        {
            "source_category_id": category_id,
            "source_category_name": label,
            "output_class_id": CLASS_MAPPING[label],
        }
        for category_id, label in categories.items()
    ]
    manifest = {
        "dataset_name": "extracted_pairs_staging",
        "class_mapping": dict(CLASS_MAPPING),
        "overlap_rule": OVERLAP_RULE,
        "source_categories": source_categories,
        "sample_counts": {"train": 0, "val": len(samples)},
        "samples": samples,
    }
    _write_json(output_root / "manifest.json", manifest)
    return manifest


def _source_fingerprints(root: Path) -> dict[str, str]:
    return {
        path.name: _sha256(path)
        for path in sorted(root.iterdir(), key=lambda item: item.name.casefold())
        if path.is_file()
    }


def _overlap_descriptors(
    root: Path,
    samples: Sequence[Mapping[str, Any]],
    *,
    hash_images: bool,
    hash_masks: bool,
    missing: list[Any],
) -> list[dict[str, Any]]:
    descriptors = []
    for sample in samples:
        sample_id = _sample_id_string(sample.get("sample_id"))
        image_value = sample.get("image_path")
        mask_value = sample.get("mask_path")
        image_path = _safe_manifest_file(root, image_value, must_exist=False)
        mask_path = _safe_manifest_file(root, mask_value, must_exist=False)
        original_filename = sample.get("source_image_filename")
        if not isinstance(original_filename, str) or not original_filename:
            original_filename = image_path.name
        descriptor: dict[str, Any] = {
            "sample_id": sample_id,
            "original_filename": original_filename,
            "normalized_relative_filename": _normalized_relative_filename(
                str(image_value)
            ),
            "image_sha256": None,
            "mask_sha256": None,
        }
        for kind, enabled, path in (
            ("image", hash_images, image_path),
            ("mask", hash_masks, mask_path),
        ):
            if not enabled:
                continue
            if not path.is_file():
                missing.append({"sample_id": sample_id, "comparison": f"{kind}_sha256"})
                continue
            descriptor[f"{kind}_sha256"] = _sha256(path)
        descriptors.append(descriptor)
    return descriptors


def _matching_values(
    pilot: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    key: str,
    casefold: bool,
) -> list[dict[str, Any]]:
    def normalized(value: Any) -> str:
        text = str(value)
        return text.casefold() if casefold else text

    reference_by_value: dict[str, list[str]] = {}
    display: dict[str, str] = {}
    for item in reference:
        value = normalized(item[key])
        display.setdefault(value, str(item[key]))
        reference_by_value.setdefault(value, []).append(str(item["sample_id"]))
    matches = []
    for item in pilot:
        value = normalized(item[key])
        if value in reference_by_value:
            matches.append(
                {
                    "value": display[value],
                    "pilot_sample_id": str(item["sample_id"]),
                    "reference_sample_ids": sorted(reference_by_value[value]),
                }
            )
    return sorted(matches, key=lambda item: (item["pilot_sample_id"], item["value"]))


def _matching_hashes(
    pilot: Sequence[Mapping[str, Any]],
    reference: Sequence[Mapping[str, Any]],
    *,
    key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    reference_by_hash: dict[str, list[str]] = {}
    for item in reference:
        digest = item.get(key)
        if digest:
            reference_by_hash.setdefault(str(digest), []).append(str(item["sample_id"]))
    pilot_by_hash: dict[str, list[str]] = {}
    for item in pilot:
        digest = item.get(key)
        if digest:
            pilot_by_hash.setdefault(str(digest), []).append(str(item["sample_id"]))
    matches = []
    ambiguous = []
    for digest in sorted(set(pilot_by_hash) & set(reference_by_hash)):
        entry = {
            "sha256": digest,
            "pilot_sample_ids": sorted(pilot_by_hash[digest]),
            "reference_sample_ids": sorted(reference_by_hash[digest]),
        }
        matches.append(entry)
        if len(entry["pilot_sample_ids"]) > 1 or len(entry["reference_sample_ids"]) > 1:
            ambiguous.append({"comparison": key, **entry})
    return matches, ambiguous


def _normalized_relative_filename(value: str) -> str:
    path = PurePosixPath(value.replace("\\", "/"))
    parts = path.parts
    if "images" in parts:
        index = parts.index("images")
        return PurePosixPath(*parts[index:]).as_posix()
    return path.as_posix()


def _sample_lookup(
    root: Path,
    samples: Sequence[Mapping[str, Any]],
) -> dict[Path, Mapping[str, Any]]:
    lookup: dict[Path, Mapping[str, Any]] = {}
    for sample in samples:
        image = _safe_manifest_file(root, sample.get("image_path")).resolve()
        if image in lookup:
            raise ValueError(
                "Pilot manifest contains a duplicate normalized image path"
            )
        lookup[image] = sample
    return lookup


def _run_frozen_geometry(
    mask: np.ndarray,
    *,
    v2a_estimator: V2aEstimator,
    v2b_estimator: V2bEstimator,
) -> tuple[FixedAxisCutlineResult, FixedAxisSearchCutlineResult]:
    configuration = FROZEN_FIXED_AXIS_CONFIGURATION
    v2a = v2a_estimator(
        mask,
        removal_axis=tuple(configuration["removal_axis"]),
        projection_quantile=configuration["projection_quantile"],
        support_band_width_pixels=configuration["support_band_width_pixels"],
        calyx_dilation_radius=configuration["calyx_dilation_radius"],
        component_connectivity=configuration["component_connectivity"],
        signed_offset_pixels=configuration["signed_offset_pixels"],
    )
    _validate_v2a_result(v2a, mask_shape=mask.shape)
    v2b = v2b_estimator(
        mask,
        removal_axis=tuple(configuration["removal_axis"]),
        projection_quantile=configuration["projection_quantile"],
        support_band_width_pixels=configuration["support_band_width_pixels"],
        calyx_dilation_radius=configuration["calyx_dilation_radius"],
        component_connectivity=configuration["component_connectivity"],
        signed_offset_pixels=configuration["signed_offset_pixels"],
        candidate_step_pixels=configuration["candidate_step_pixels"],
        inward_search_margin_pixels=configuration["inward_search_margin_pixels"],
        outward_search_margin_pixels=configuration["outward_search_margin_pixels"],
        blade_band_half_width_pixels=configuration["blade_band_half_width_pixels"],
        lateral_window_half_width_pixels=configuration[
            "lateral_window_half_width_pixels"
        ],
        minimum_attachment_evidence_fraction=configuration[
            "minimum_attachment_evidence_fraction"
        ],
        minimum_flesh_band_pixels=configuration["minimum_flesh_band_pixels"],
        minimum_calyx_band_pixels=configuration["minimum_calyx_band_pixels"],
        v2a_result=v2a,
    )
    _validate_v2b_result(v2b, mask_shape=mask.shape)
    return v2a, v2b


def _cutline_comparison_rows(
    sample_id: str,
    ground_truth_mask: np.ndarray,
    predicted_mask: np.ndarray,
    ground_truth_v2a: FixedAxisCutlineResult,
    ground_truth_v2b: FixedAxisSearchCutlineResult,
    predicted_v2a: FixedAxisCutlineResult,
    predicted_v2b: FixedAxisSearchCutlineResult,
) -> list[dict[str, Any]]:
    return [
        _cutline_row(
            sample_id,
            "v2a",
            ground_truth_v2a,
            predicted_v2a,
            calculate_fixed_axis_line_side_proxies(ground_truth_mask, ground_truth_v2a),
            calculate_fixed_axis_line_side_proxies(predicted_mask, predicted_v2a),
        ),
        _cutline_row(
            sample_id,
            "v2b",
            ground_truth_v2b,
            predicted_v2b,
            calculate_fixed_axis_search_line_side_proxies(
                ground_truth_mask, ground_truth_v2b
            ),
            calculate_fixed_axis_search_line_side_proxies(
                predicted_mask, predicted_v2b
            ),
        ),
    ]


def _cutline_row(
    sample_id: str,
    method: str,
    ground_truth: FixedAxisCutlineResult | FixedAxisSearchCutlineResult,
    predicted: FixedAxisCutlineResult | FixedAxisSearchCutlineResult,
    ground_truth_proxies: Mapping[str, int | float | None],
    predicted_proxies: Mapping[str, int | float | None],
) -> dict[str, Any]:
    ground_truth_coordinate = _result_coordinate(ground_truth)
    predicted_coordinate = _result_coordinate(predicted)
    paired = (
        ground_truth.status == "ok"
        and predicted.status == "ok"
        and _finite_number(ground_truth_coordinate)
        and _finite_number(predicted_coordinate)
    )
    signed_difference = (
        float(predicted_coordinate) - float(ground_truth_coordinate) if paired else None
    )
    ground_truth_search = _search_diagnostics(ground_truth)
    predicted_search = _search_diagnostics(predicted)
    ground_truth_structured = _structured_status(ground_truth)
    predicted_structured = _structured_status(predicted)
    ground_truth_flesh_loss = ground_truth_proxies["flesh_loss_proxy_pixel_count"]
    predicted_flesh_loss = predicted_proxies["flesh_loss_proxy_pixel_count"]
    ground_truth_flesh_ratio = ground_truth_proxies["flesh_loss_proxy_ratio"]
    predicted_flesh_ratio = predicted_proxies["flesh_loss_proxy_ratio"]
    ground_truth_calyx = ground_truth_proxies["calyx_retention_proxy_pixel_count"]
    predicted_calyx = predicted_proxies["calyx_retention_proxy_pixel_count"]
    ground_truth_calyx_ratio = ground_truth_proxies["calyx_retention_proxy_ratio"]
    predicted_calyx_ratio = predicted_proxies["calyx_retention_proxy_ratio"]
    return {
        "sample_id": sample_id,
        "method": method,
        "ground_truth_status": ground_truth.status,
        "ground_truth_failure_code": ground_truth.failure_code,
        "ground_truth_structured_status": ground_truth_structured,
        "predicted_status": predicted.status,
        "predicted_failure_code": predicted.failure_code,
        "predicted_structured_status": predicted_structured,
        "status_agreement": ground_truth.status == predicted.status,
        "structured_status_agreement": ground_truth_structured == predicted_structured,
        "ground_truth_coordinate": ground_truth_coordinate,
        "predicted_coordinate": predicted_coordinate,
        "signed_coordinate_difference_predicted_minus_ground_truth": (
            signed_difference
        ),
        "absolute_coordinate_difference": (
            abs(signed_difference) if signed_difference is not None else None
        ),
        "within_2_pixels": (
            abs(signed_difference) <= 2.0 if signed_difference is not None else None
        ),
        "within_5_pixels": (
            abs(signed_difference) <= 5.0 if signed_difference is not None else None
        ),
        "within_10_pixels": (
            abs(signed_difference) <= 10.0 if signed_difference is not None else None
        ),
        **_component_diagnostics("ground_truth", ground_truth),
        **_component_diagnostics("predicted", predicted),
        **{f"ground_truth_{key}": value for key, value in ground_truth_search.items()},
        **{f"predicted_{key}": value for key, value in predicted_search.items()},
        "ground_truth_flesh_loss_proxy_pixel_count": ground_truth_flesh_loss,
        "ground_truth_flesh_loss_proxy_ratio": ground_truth_flesh_ratio,
        "ground_truth_calyx_retention_proxy_pixel_count": ground_truth_calyx,
        "ground_truth_calyx_retention_proxy_ratio": ground_truth_calyx_ratio,
        "predicted_flesh_loss_proxy_pixel_count": predicted_flesh_loss,
        "predicted_flesh_loss_proxy_ratio": predicted_flesh_ratio,
        "predicted_calyx_retention_proxy_pixel_count": predicted_calyx,
        "predicted_calyx_retention_proxy_ratio": predicted_calyx_ratio,
        "flesh_loss_proxy_pixel_count_change_predicted_minus_ground_truth": (
            _difference(predicted_flesh_loss, ground_truth_flesh_loss)
        ),
        "flesh_loss_proxy_ratio_change_predicted_minus_ground_truth": (
            _difference(predicted_flesh_ratio, ground_truth_flesh_ratio)
        ),
        "calyx_retention_proxy_pixel_count_change_predicted_minus_ground_truth": (
            _difference(predicted_calyx, ground_truth_calyx)
        ),
        "calyx_retention_proxy_ratio_change_predicted_minus_ground_truth": (
            _difference(predicted_calyx_ratio, ground_truth_calyx_ratio)
        ),
    }


def _component_diagnostics(
    prefix: str,
    result: FixedAxisCutlineResult | FixedAxisSearchCutlineResult,
) -> dict[str, Any]:
    return {
        f"{prefix}_flesh_component_count": result.flesh_component_count,
        f"{prefix}_calyx_component_count": result.calyx_component_count,
        f"{prefix}_selected_flesh_component": result.selected_flesh_component,
        f"{prefix}_selected_calyx_component": result.selected_calyx_component,
        f"{prefix}_contact_pixel_count": result.contact_pixel_count,
    }


def _search_diagnostics(
    result: FixedAxisCutlineResult | FixedAxisSearchCutlineResult,
) -> dict[str, Any]:
    if not isinstance(result, FixedAxisSearchCutlineResult):
        return {
            "candidate_count": None,
            "feasible_candidate_count": None,
            "feasible_block_count": None,
            "selected_block_id": None,
            "selected_block_start": None,
            "selected_block_end": None,
            "selected_block_candidate_count": None,
            "selected_block_is_singleton": None,
        }
    return {
        "candidate_count": result.candidate_count,
        "feasible_candidate_count": result.feasible_candidate_count,
        "feasible_block_count": result.feasible_block_count,
        "selected_block_id": result.selected_block_id,
        "selected_block_start": result.selected_block_start,
        "selected_block_end": result.selected_block_end,
        "selected_block_candidate_count": result.selected_block_candidate_count,
        "selected_block_is_singleton": result.selected_block_is_singleton,
    }


def _result_coordinate(
    result: FixedAxisCutlineResult | FixedAxisSearchCutlineResult,
) -> float | None:
    if isinstance(result, FixedAxisSearchCutlineResult):
        return result.selected_cut_coordinate
    return result.final_cut_coordinate


def _structured_status(
    result: FixedAxisCutlineResult | FixedAxisSearchCutlineResult,
) -> str:
    return (
        "ok" if result.status == "ok" else f"failed:{result.failure_code or 'unknown'}"
    )


def _structured_failure_counts(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, int]]]:
    output: dict[str, dict[str, dict[str, int]]] = {}
    for method in ("v2a", "v2b"):
        method_rows = [row for row in rows if row.get("method") == method]
        output[method] = {}
        for side in ("ground_truth", "predicted"):
            failures = Counter(
                str(row[f"{side}_failure_code"] or "unknown_failure")
                for row in method_rows
                if row[f"{side}_status"] != "ok"
            )
            output[method][side] = dict(sorted(failures.items()))
    return output


def _load_pilot_manifest(root: Path) -> dict[str, Any]:
    path = root / "manifest.json"
    if not path.is_file():
        raise FileNotFoundError(f"Pilot dataset manifest does not exist: {path}")
    manifest = _load_json(path)
    if not isinstance(manifest, dict):
        raise ValueError("Pilot manifest.json must contain a JSON object")
    if manifest.get("dataset_kind") != PILOT_DATASET_KIND:
        raise ValueError("Dataset is not marked as a pilot holdout")
    if manifest.get("dataset_name") != PILOT_DATASET_IDENTIFIER:
        raise ValueError("Pilot dataset identifier is missing or incompatible")
    if manifest.get("is_v1_or_v2_dataset") is not False:
        raise ValueError(
            "Pilot manifest must explicitly record that it is not v1 or v2"
        )
    if manifest.get("class_mapping") != CLASS_MAPPING:
        raise ValueError("Pilot class mapping must be background=0, Flesh=1, Calyx=2")
    if manifest.get("overlap_rule") != OVERLAP_RULE:
        raise ValueError(
            "Pilot overlap semantics do not match the VisionTrain converter"
        )
    return manifest


def _load_overlap_audit(root: Path) -> dict[str, Any]:
    path = root / "overlap_audit.json"
    if not path.is_file():
        raise FileNotFoundError("Prepared pilot dataset is missing overlap_audit.json")
    value = _load_json(path)
    if not isinstance(value, dict):
        raise ValueError("overlap_audit.json must contain a JSON object")
    return value


def _manifest_samples(
    manifest: Mapping[str, Any],
    *,
    context: str,
) -> list[Mapping[str, Any]]:
    samples = manifest.get("samples")
    if not isinstance(samples, list) or not all(
        isinstance(sample, Mapping) for sample in samples
    ):
        raise ValueError(f"{context} samples must be a JSON list of objects")
    ids = [_sample_id_string(sample.get("sample_id")) for sample in samples]
    if len(ids) != len(set(ids)):
        raise ValueError(f"{context} contains duplicate sample IDs")
    return samples


def _safe_manifest_file(
    root: Path,
    value: Any,
    *,
    must_exist: bool = True,
) -> Path:
    if not isinstance(value, str) or not value:
        raise ValueError("Manifest file paths must be non-empty relative strings")
    relative = PurePosixPath(value.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Manifest file paths must be safe and relative")
    path = root.joinpath(*relative.parts)
    resolved_root = root.resolve()
    resolved = path.resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise ValueError("Manifest path escapes its dataset root")
    if must_exist and not path.is_file():
        raise FileNotFoundError(f"Manifest-referenced file does not exist: {value}")
    return path


def _validate_image_mask_shape(image_path: Path, mask_path: Path) -> None:
    with Image.open(image_path) as image, Image.open(mask_path) as mask:
        if image.size != mask.size:
            raise ValueError(
                f"Image/mask shape mismatch for {image_path.name}: "
                f"image={image.size}, mask={mask.size}"
            )
        mask_array = np.asarray(mask)
    if mask_array.ndim != 2 or not np.all(np.isin(mask_array, (0, 1, 2))):
        raise ValueError(
            "Normalized pilot mask must contain only class IDs 0, 1, and 2"
        )


def _artifact_id(sample: Mapping[str, Any]) -> str:
    value = sample.get("artifact_id")
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9._-]+", value):
        raise ValueError(
            "Pilot artifact_id must contain only letters, digits, dot, dash, or underscore"
        )
    return value


def _unique_artifact_id(sample_id: str, used: set[str]) -> str:
    base = re.sub(r"[^A-Za-z0-9._-]+", "_", sample_id).strip("._") or "sample"
    candidate = base
    suffix = 2
    while candidate.casefold() in used:
        candidate = f"{base}_{suffix}"
        suffix += 1
    used.add(candidate.casefold())
    return candidate


def _sample_id_string(value: Any) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError("sample_id must be an integer or path-free string")
    text = str(value)
    _validate_sample_identifier(text, context="sample_id")
    return text


def _validate_sample_identifier(value: str, *, context: str) -> None:
    if (
        not value
        or value in {".", ".."}
        or any(character in value for character in "/\\:")
        or _looks_like_absolute_path(value)
    ):
        raise ValueError(f"{context} must be a path-free identifier")


def _validate_directory_destination(
    path: Path,
    *,
    overwrite: bool,
    operation: str,
) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Pass overwrite=True or --overwrite."
            )
        if path.is_symlink() or not path.is_dir():
            raise ValueError(
                f"Existing {operation} output must be a non-symlink directory"
            )


def _validate_file_destination(
    path: Path,
    *,
    overwrite: bool,
    operation: str,
) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output already exists: {path}. Pass overwrite=True or --overwrite."
            )
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"Existing {operation} output must be a non-symlink file")


def _validate_evaluation_paths(
    dataset_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    metadata_manifest: Path,
    *,
    overwrite: bool,
) -> None:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Pilot dataset root does not exist: {dataset_root}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    if not metadata_manifest.is_file():
        raise FileNotFoundError(
            f"Pilot metadata manifest does not exist: {metadata_manifest}"
        )
    _validate_directory_destination(
        output_root, overwrite=overwrite, operation="evaluation"
    )
    dataset_resolved = dataset_root.resolve()
    output_resolved = output_root.resolve()
    if _paths_overlap(dataset_resolved, output_resolved):
        raise ValueError("Pilot dataset and evaluation output roots must not overlap")
    checkpoint_resolved = checkpoint_path.resolve()
    if (
        output_resolved == checkpoint_resolved
        or output_resolved in checkpoint_resolved.parents
    ):
        raise ValueError("Evaluation output root must not contain the checkpoint")
    metadata_resolved = metadata_manifest.resolve()
    if (
        output_resolved == metadata_resolved
        or output_resolved in metadata_resolved.parents
    ):
        raise ValueError(
            "Evaluation output root must not contain the metadata manifest"
        )


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _install_staged_directory(
    staging_root: Path,
    destination_root: Path,
    *,
    overwrite: bool,
) -> None:
    backup_root: Path | None = None
    if destination_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output appeared during generation: {destination_root}"
            )
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise ValueError("Existing output must be a non-symlink directory")
        backup_root = destination_root.with_name(
            f".{destination_root.name}.backup-{uuid.uuid4().hex}"
        )
        destination_root.replace(backup_root)
    try:
        staging_root.replace(destination_root)
    except Exception:
        if backup_root is not None and backup_root.exists():
            backup_root.replace(destination_root)
        raise
    if backup_root is not None:
        shutil.rmtree(backup_root)


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(_json_ready(value), indent=2, ensure_ascii=False, allow_nan=False)
        + "\n",
        encoding="utf-8",
    )


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row.get(field)) for field in fields})


def _write_csv_atomic(
    destination: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
    *,
    overwrite: bool,
) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.staging-",
        dir=destination.parent,
    )
    os.close(handle)
    temporary = Path(temporary_name)
    try:
        _write_csv(temporary, fields, rows)
        if destination.exists() and not overwrite:
            raise FileExistsError(f"Output appeared during generation: {destination}")
        temporary.replace(destination)
    except Exception:
        if temporary.exists():
            temporary.unlink()
        raise


def _write_confusion_matrix(path: Path, matrix: torch.Tensor) -> None:
    class_names = tuple(CLASS_MAPPING)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(["target/prediction", *class_names])
        for index, name in enumerate(class_names):
            writer.writerow([name, *(int(value) for value in matrix[index].tolist())])


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        return "" if not math.isfinite(value) else format(value, ".12g")
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


def _assert_no_absolute_paths(value: Any) -> None:
    if isinstance(value, Mapping):
        for item in value.values():
            _assert_no_absolute_paths(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            _assert_no_absolute_paths(item)
    elif isinstance(value, str) and _looks_like_absolute_path(value):
        raise ValueError("Persisted metadata must not contain absolute paths")


def _looks_like_absolute_path(value: str) -> bool:
    if not value:
        return False
    candidates = [item.strip(" \t\"'") for item in re.split(r"[\r\n]+", value)]
    return any(
        PureWindowsPath(item).is_absolute()
        or PurePosixPath(item).is_absolute()
        or re.search(r"(?:^|\s)[A-Za-z]:[\\/]", item) is not None
        or re.search(r"(?:^|\s)[\\/]{2}[^\s\\/]+[\\/]", item) is not None
        or re.search(r"(?:^|\s)/(?:[^\s/]+/)+", item) is not None
        for item in candidates
        if item
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as input_file:
        for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _difference(
    current: int | float | None,
    reference: int | float | None,
) -> int | float | None:
    if current is None or reference is None:
        return None
    return current - reference


def _parse_category_map_arguments(
    values: Sequence[str] | None,
) -> dict[CategoryId, str] | None:
    if not values:
        return None
    mapping: dict[CategoryId, str] = {}
    for value in values:
        source_id_text, separator, label = value.partition("=")
        if not separator or not source_id_text or not label:
            raise ValueError("--category-map values must use SOURCE_ID=LABEL syntax")
        source_id: CategoryId = (
            int(source_id_text)
            if re.fullmatch(r"-?[0-9]+", source_id_text)
            else source_id_text
        )
        source_id = _category_id(source_id, field="--category-map source ID")
        if label not in CLASS_MAPPING:
            raise ValueError("--category-map LABEL must be background, Flesh, or Calyx")
        if source_id in mapping:
            raise ValueError(f"Duplicate --category-map source ID {source_id!r}")
        mapping[source_id] = label
    return mapping


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Prepare or evaluate the seven-image pilot holdout."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="prepare strict VisionTrain pairs as a normalized pilot dataset",
    )
    prepare.add_argument("--raw-labelled-dir", type=Path, required=True)
    prepare.add_argument("--normalized-output-dir", type=Path, required=True)
    prepare.add_argument(
        "--expected-sample-count", type=int, default=DEFAULT_EXPECTED_SAMPLE_COUNT
    )
    prepare.add_argument(
        "--input-format",
        choices=INPUT_FORMATS,
        default="auto",
        help=(
            "auto selects full-export when dataset.json is present and otherwise "
            "requires strict stem-matched extracted pairs"
        ),
    )
    prepare.add_argument(
        "--category-map",
        action="append",
        metavar="SOURCE_ID=LABEL",
        help=(
            "repeatable explicit category mapping for extracted annotations that "
            "lack labelname; category IDs are never guessed"
        ),
    )
    prepare.add_argument("--reference-v1-root", type=Path)
    prepare.add_argument(
        "--manifest-output",
        type=Path,
        help="optional path for the manually editable pilot metadata CSV",
    )
    prepare.add_argument(
        "--allow-overlap",
        action="store_true",
        help="explicit research override for an unexpected v1 overlap",
    )
    prepare.add_argument("--no-image-hash", action="store_true")
    prepare.add_argument("--no-mask-hash", action="store_true")
    prepare.add_argument("--overwrite", action="store_true")

    template = subparsers.add_parser(
        "template",
        aliases=["manifest-template"],
        help="generate a manually editable pilot metadata CSV",
    )
    template.add_argument("--dataset-root", type=Path, required=True)
    template.add_argument("--output", type=Path, required=True)
    template.add_argument("--overwrite", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate",
        help="run segmentation and fixed-axis mask-geometry agreement",
    )
    evaluate.add_argument("--dataset-root", type=Path, required=True)
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output-root", type=Path, required=True)
    evaluate.add_argument("--metadata-manifest", type=Path, required=True)
    evaluate.add_argument(
        "--expected-sample-count", type=int, default=DEFAULT_EXPECTED_SAMPLE_COUNT
    )
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument("--batch-size", type=int, default=1)
    evaluate.add_argument("--num-workers", type=int, default=0)
    evaluate.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the pilot holdout command-line workflow."""

    arguments = _build_argument_parser().parse_args(argv)
    if arguments.command == "prepare":
        try:
            category_mapping = _parse_category_map_arguments(arguments.category_map)
        except ValueError as error:
            raise SystemExit(str(error)) from error
        output = prepare_pilot_holdout_dataset(
            arguments.raw_labelled_dir,
            arguments.normalized_output_dir,
            expected_sample_count=arguments.expected_sample_count,
            reference_v1_root=arguments.reference_v1_root,
            overwrite=arguments.overwrite,
            metadata_manifest_output=arguments.manifest_output,
            allow_overlap=arguments.allow_overlap,
            hash_images=not arguments.no_image_hash,
            hash_masks=not arguments.no_mask_hash,
            input_format=arguments.input_format,
            category_mapping=category_mapping,
        )
        print(
            f"Prepared {output['summary']['pilot_sample_count']} samples for the "
            f"{STUDY_NAME} from {output['summary']['source_input_format']} input."
        )
        if arguments.manifest_output is None:
            print("Run the template command before fixed-axis evaluation.")
        return 0
    if arguments.command in {"template", "manifest-template"}:
        path = write_pilot_metadata_template(
            arguments.dataset_root,
            arguments.output,
            overwrite=arguments.overwrite,
        )
        print(f"Wrote pilot metadata template to {path}")
        return 0
    output = evaluate_pilot_holdout(
        arguments.dataset_root,
        arguments.checkpoint,
        arguments.output_root,
        arguments.metadata_manifest,
        expected_sample_count=arguments.expected_sample_count,
        device=arguments.device,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        overwrite=arguments.overwrite,
    )
    summary = output["summary"]
    print(
        f"Completed {STUDY_NAME}: "
        f"{summary['segmentation_evaluated_count']} segmentation samples and "
        f"{summary['fixed_axis_eligible_count']} fixed-axis-eligible samples."
    )
    print(
        "Coordinate agreement is visible-mask geometry agreement, not physical cut accuracy."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
