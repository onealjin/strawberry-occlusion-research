from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from strawberry_occlusion.augmentation.synthetic_occluder import (
    MATCHED_REGIONS,
    SyntheticOccluder,
    apply_occluder,
    build_matched_occluder_set,
    generate_synthetic_occluder,
)
from strawberry_occlusion.evaluation.occlusion_robustness import (
    BOOTSTRAP_SEED,
    M4_V1_PLACEMENT_PARAMETERS,
    PROFILE_SEEDS,
    PROFILE_SEVERITIES,
    REQUIRED_MANIFEST_FIELDS,
    SUMMARY_OUTCOMES,
    TRIAL_FIELDS,
    SegmentationInferenceOutput,
    _build_argument_parser,
    _install_staged_directory,
    _validated_inference_output,
    _validate_output_destination,
    benchmark_profile,
    bootstrap_group_intervals,
    matched_set_completeness_rows,
    masked_segmentation_metrics,
    reference_error_metrics,
    run_occlusion_robustness,
    silent_drift_flags,
    summarize_existing_output,
    summarize_robustness_trials,
    validate_development_manifest,
)
from strawberry_occlusion.evaluation.pilot_holdout import (
    FROZEN_FIXED_AXIS_CONFIGURATION,
)
from strawberry_occlusion.geometry import (
    estimate_fixed_axis_cutline,
    estimate_fixed_axis_search_cutline,
)
from strawberry_occlusion.visualization.occlusion_robustness import (
    PLOT_FILENAMES,
    create_matched_quartet_visualization,
)


HEIGHT = 128
WIDTH = 160
PLACEMENT_KWARGS = {
    "attachment_roi_dilation_pixels": int(
        M4_V1_PLACEMENT_PARAMETERS["attachment_roi_dilation_pixels"]
    ),
    "background_separation_pixels": int(
        M4_V1_PLACEMENT_PARAMETERS["background_foreground_separation_pixels"]
    ),
}


class TinyM4Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(3, 3, kernel_size=1)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))
            self.classifier.weight[1, 0, 0, 0] = 5.0
            self.classifier.weight[1, 1, 0, 0] = -1.0
            self.classifier.weight[2, 1, 0, 0] = 5.0
            self.classifier.weight[2, 0, 0, 0] = -1.0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(image)


class AlwaysBackgroundM4Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(3, 3, kernel_size=1)
        with torch.no_grad():
            self.classifier.weight.zero_()
            self.classifier.bias.copy_(torch.tensor([1.0, 0.0, 0.0]))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(image)


def _semantic_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[32:100, 25:105] = 1
    mask[50:82, 105:130] = 2
    return mask


def _rgb(mask: np.ndarray) -> np.ndarray:
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask == 1] = (220, 35, 35)
    image[mask == 2] = (35, 220, 35)
    return image


def _write_pair(root: Path, sample_id: str) -> tuple[str, str]:
    image_relative = Path("images") / f"{sample_id}.png"
    mask_relative = Path("masks") / f"{sample_id}.png"
    (root / image_relative).parent.mkdir(parents=True, exist_ok=True)
    (root / mask_relative).parent.mkdir(parents=True, exist_ok=True)
    mask = _semantic_mask()
    Image.fromarray(_rgb(mask), mode="RGB").save(root / image_relative)
    Image.fromarray(mask, mode="L").save(root / mask_relative)
    return image_relative.as_posix(), mask_relative.as_posix()


def _manifest_row(
    root: Path,
    sample_id: str = "fruit_01",
    *,
    group_id: str = "group_01",
    session_id: str = "session_01",
    role: str = "development",
    eligible: str = "true",
) -> dict[str, str]:
    image_path, mask_path = _write_pair(root, sample_id)
    return {
        "sample_id": sample_id,
        "group_id": group_id,
        "session_id": session_id,
        "role": role,
        "image_path": image_path,
        "mask_path": mask_path,
        "orientation_category": "right",
        "fixed_axis_eligible": eligible,
        "annotation_qa_status": "reviewed",
    }


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=REQUIRED_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)


def _write_checkpoint(
    path: Path,
    *,
    completed_epoch: int = 92,
    model_class: type[nn.Module] = TinyM4Model,
) -> None:
    model = model_class()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "completed_epoch": completed_epoch,
            "best_mean_foreground_iou": 0.5,
            "class_mapping": {"background": 0, "Flesh": 1, "Calyx": 2},
            "input_height": HEIGHT,
            "input_width": WIDTH,
            "model_name": model_class.__name__,
        },
        path,
    )


def _geometry() -> tuple[Any, Any, np.ndarray]:
    mask = _semantic_mask()
    configuration = FROZEN_FIXED_AXIS_CONFIGURATION
    v2a = estimate_fixed_axis_cutline(
        mask,
        removal_axis=tuple(configuration["removal_axis"]),
        projection_quantile=configuration["projection_quantile"],
        support_band_width_pixels=configuration["support_band_width_pixels"],
        signed_offset_pixels=configuration["signed_offset_pixels"],
        calyx_dilation_radius=configuration["calyx_dilation_radius"],
        component_connectivity=configuration["component_connectivity"],
    )
    v2b = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=tuple(configuration["removal_axis"]),
        projection_quantile=configuration["projection_quantile"],
        support_band_width_pixels=configuration["support_band_width_pixels"],
        signed_offset_pixels=configuration["signed_offset_pixels"],
        calyx_dilation_radius=configuration["calyx_dilation_radius"],
        component_connectivity=configuration["component_connectivity"],
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
    return v2a, v2b, mask


def _synthetic_trial_rows(
    *,
    sample_ids: tuple[str, ...] = ("image_a", "image_b"),
    group_id: str = "group_a",
) -> list[dict[str, Any]]:
    rows = []
    for sample_index, sample_id in enumerate(sample_ids):
        for method in ("v2a", "v2b"):
            for region_index, region in enumerate(MATCHED_REGIONS):
                for severity in PROFILE_SEVERITIES:
                    for seed in PROFILE_SEEDS:
                        matched_set = f"{sample_id}-{severity}-{seed}"
                        stability = sample_index + region_index + severity * 10
                        rows.append(
                            {
                                "sample_id": sample_id,
                                "group_id": group_id,
                                "session_id": f"session_{sample_index}",
                                "method": method,
                                "region": region,
                                "severity_fraction": severity,
                                "seed": seed,
                                "matched_set_id": matched_set,
                                "perturbation_id": f"{matched_set}-{region}",
                                "matched_set_complete": True,
                                "method_status": "ok",
                                "finite_cut_returned": True,
                                "stability_abs": stability,
                                "gt_reference_abs": stability + 1,
                                "excess_gt_reference_error": stability - 0.5,
                                "silent_drift_2_pixels": stability > 2,
                                "silent_drift_5_pixels": stability > 5,
                                "silent_drift_10_pixels": stability > 10,
                                "iou_background": 0.9,
                                "iou_flesh": 0.8,
                                "iou_calyx": 0.7,
                                "dice_background": 0.95,
                                "dice_flesh": 0.88,
                                "dice_calyx": 0.82,
                                "mean_foreground_iou": 0.75,
                                "mean_foreground_dice": 0.85,
                            }
                        )
    return rows


def _paired_group_rows(
    values: list[float],
    *,
    comparison: str = "attachment_minus_background",
    group_prefix: str = "group",
) -> list[dict[str, Any]]:
    rows = []
    for index, value in enumerate(values):
        row = {
            "group_id": f"{group_prefix}_{index}",
            "method": "v2a",
            "comparison": comparison,
            "comparison_priority": (
                "primary"
                if comparison == "attachment_minus_background"
                else "secondary"
            ),
            "n_images": 1,
            "n_sessions": 1,
            "perturbation_evaluations": 24,
        }
        for outcome in SUMMARY_OUTCOMES:
            row[f"{outcome}_auc_difference_attachment_minus_control"] = value
        rows.append(row)
    return rows


def _bootstrap_result(
    rows: list[dict[str, Any]],
    *,
    comparison: str = "attachment_minus_background",
    outcome: str = "stability_abs",
) -> dict[str, Any]:
    results = bootstrap_group_intervals(
        rows,
        seed=BOOTSTRAP_SEED,
        replicates=1_000,
    )
    return next(
        row
        for row in results
        if row["method"] == "v2a"
        and row["comparison"] == comparison
        and row["outcome"] == outcome
    )


def test_development_only_manifest_succeeds_and_retains_segmentation_rows(
    tmp_path: Path,
) -> None:
    safe_root = tmp_path / "safe"
    rows = [
        _manifest_row(safe_root, "eligible"),
        _manifest_row(safe_root, "segmentation_only", eligible="false"),
    ]
    manifest = tmp_path / "development.csv"
    _write_manifest(manifest, rows)

    validated = validate_development_manifest(manifest, safe_root)

    assert [sample.sample_id for sample in validated] == [
        "eligible",
        "segmentation_only",
    ]
    assert [sample.fixed_axis_eligible for sample in validated] == [True, False]


@pytest.mark.parametrize("role", ["test", "final_test", "locked_test"])
def test_test_roles_hard_fail_before_missing_file_checks(
    tmp_path: Path,
    role: str,
) -> None:
    safe_root = tmp_path / "safe"
    row = _manifest_row(safe_root, role=role)
    (safe_root / row["image_path"]).unlink()
    manifest = tmp_path / "development.csv"
    _write_manifest(manifest, [row])

    with pytest.raises(ValueError, match="forbidden"):
        validate_development_manifest(manifest, safe_root)


def test_absolute_and_escaping_manifest_paths_fail(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    absolute = _manifest_row(safe_root, "absolute")
    absolute["image_path"] = str((safe_root / absolute["image_path"]).resolve())
    manifest = tmp_path / "absolute.csv"
    _write_manifest(manifest, [absolute])
    with pytest.raises(ValueError, match="relative"):
        validate_development_manifest(manifest, safe_root)

    escaping = _manifest_row(safe_root, "escaping")
    escaping["mask_path"] = "../outside.png"
    manifest = tmp_path / "escaping.csv"
    _write_manifest(manifest, [escaping])
    with pytest.raises(ValueError, match="parent components"):
        validate_development_manifest(manifest, safe_root)

    drive_relative = _manifest_row(safe_root, "drive_relative")
    drive_relative["image_path"] = "C:a.png"
    manifest = tmp_path / "drive-relative.csv"
    _write_manifest(manifest, [drive_relative])
    with pytest.raises(ValueError, match="relative"):
        validate_development_manifest(manifest, safe_root)


@pytest.mark.parametrize("component", ["locked_test", "Locked_Test", "FINAL_TEST"])
def test_locked_directory_component_fails_even_with_development_role(
    tmp_path: Path, component: str
) -> None:
    safe_root = tmp_path / "safe"
    row = _manifest_row(safe_root)
    locked = safe_root / component / "image.png"
    locked.parent.mkdir()
    locked.write_bytes((safe_root / row["image_path"]).read_bytes())
    row["image_path"] = f"{component}/image.png"
    manifest = tmp_path / "development.csv"
    _write_manifest(manifest, [row])
    with pytest.raises(ValueError, match="forbidden directory component"):
        validate_development_manifest(manifest, safe_root)


def test_non_development_role_is_rejected(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    row = _manifest_row(safe_root, role="val")
    manifest = tmp_path / "development.csv"
    _write_manifest(manifest, [row])

    with pytest.raises(ValueError, match="role must equal 'development'"):
        validate_development_manifest(manifest, safe_root)


def test_duplicate_sample_missing_group_and_unreviewed_qa_fail(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    row = _manifest_row(safe_root)
    manifest = tmp_path / "duplicate.csv"
    _write_manifest(manifest, [row, row])
    with pytest.raises(ValueError, match="Duplicate sample_id"):
        validate_development_manifest(manifest, safe_root)

    missing_group = dict(row, sample_id="other", group_id="")
    manifest = tmp_path / "group.csv"
    _write_manifest(manifest, [missing_group])
    with pytest.raises(ValueError, match="group_id must be non-empty"):
        validate_development_manifest(manifest, safe_root)

    unreviewed = dict(row, annotation_qa_status="pending")
    manifest = tmp_path / "qa.csv"
    _write_manifest(manifest, [unreviewed])
    with pytest.raises(ValueError, match="not reviewed"):
        validate_development_manifest(manifest, safe_root)


def test_matched_occluder_identity_area_location_and_determinism() -> None:
    v2a, _, mask = _geometry()
    foreground = int(np.count_nonzero(mask))
    first = generate_synthetic_occluder(
        sample_id="fruit",
        severity_fraction=0.02,
        seed=1,
        foreground_area_pixels=foreground,
    )
    second = generate_synthetic_occluder(
        sample_id="fruit",
        severity_fraction=0.02,
        seed=1,
        foreground_area_pixels=foreground,
    )
    assert first.occluder_id == second.occluder_id
    assert first.template_sha256 == second.template_sha256
    assert np.array_equal(first.mask, second.mask)
    assert np.array_equal(first.rgb, second.rgb)

    matched = build_matched_occluder_set(mask, v2a, first, **PLACEMENT_KWARGS)
    assert matched.complete
    assert tuple(matched.placements) == MATCHED_REGIONS
    assert {item.area_pixels for item in matched.placements.values()} == {
        first.area_pixels
    }
    assert {item.template_sha256 for item in matched.placements.values()} == {
        first.template_sha256
    }
    assert len({(item.top, item.left) for item in matched.placements.values()}) == 4

    image = _rgb(mask)
    appearances = []
    for region, placement in matched.placements.items():
        output, translated = apply_occluder(image, first, placement)
        assert int(translated.sum()) == first.area_pixels
        if region in {"calyx_tip", "flesh_far"}:
            assert not np.any(translated & matched.attachment_roi)
        if region == "background":
            assert not np.any(translated & (mask > 0))
        appearances.append(output[translated])
    assert all(np.array_equal(appearances[0], values) for values in appearances[1:])


def test_all_four_region_placements_and_structured_incomplete_set() -> None:
    v2a, _, mask = _geometry()
    occluder = generate_synthetic_occluder(
        sample_id="fruit",
        severity_fraction=0.04,
        seed=0,
        foreground_area_pixels=int(np.count_nonzero(mask)),
    )
    matched = build_matched_occluder_set(mask, v2a, occluder, **PLACEMENT_KWARGS)
    assert matched.complete
    assert set(matched.placements) == set(MATCHED_REGIONS)

    no_background = np.ones_like(mask)
    failed_geometry = estimate_fixed_axis_cutline(no_background, removal_axis=(1, 0))
    incomplete = build_matched_occluder_set(
        no_background,
        failed_geometry,
        occluder,
        **PLACEMENT_KWARGS,
    )
    assert not incomplete.complete
    assert "background" in incomplete.unavailable_reasons
    assert "attachment" in incomplete.unavailable_reasons


def test_full_footprint_separation_can_make_calyx_tip_unavailable() -> None:
    v2a, _, mask = _geometry()
    template_mask = np.ones((7, 43), dtype=bool)
    occluder = SyntheticOccluder(
        occluder_id="wide_test_occluder",
        template_sha256="wide_test_template",
        target_area_pixels=int(template_mask.sum()),
        rotation_degrees=0.0,
        scale_pixels=1.0,
        opacity=1.0,
        mask=template_mask,
        rgb=np.full((*template_mask.shape, 3), 120, dtype=np.uint8),
    )

    matched = build_matched_occluder_set(
        mask,
        v2a,
        occluder,
        **PLACEMENT_KWARGS,
    )

    assert not matched.complete
    assert matched.unavailable_reasons["calyx_tip"] == (
        "distal_calyx_support_unavailable"
    )
    if "flesh_far" in matched.placements:
        _, translated = apply_occluder(
            _rgb(mask), occluder, matched.placements["flesh_far"]
        )
        assert not np.any(translated & matched.attachment_roi)


def test_matched_set_completeness_reports_region_and_quartet_attrition() -> None:
    complete_trials = [
        {
            "sample_id": "complete_image",
            "group_id": "complete_group",
            "matched_set_id": "complete-low",
            "severity_fraction": 0.01,
        }
    ]
    incomplete = [
        {
            "sample_id": "tight_image",
            "group_id": "tight_group",
            "matched_set_id": "incomplete-high",
            "severity_fraction": 0.04,
            "calyx_tip_reason": "distal_calyx_support_unavailable",
        }
    ]

    summary = matched_set_completeness_rows(complete_trials, incomplete)
    high_calyx = next(
        row
        for row in summary
        if row["summary_level"] == "region"
        and row["region"] == "calyx_tip"
        and row["severity_fraction"] == 0.04
    )
    assert high_calyx["attempted_matched_sets"] == 1
    assert high_calyx["successfully_placed"] == 0
    assert high_calyx["unavailable"] == 1
    assert high_calyx["completion_proportion"] == 0.0
    assert high_calyx["n_groups_attempted"] == 1
    assert high_calyx["n_groups_represented_among_successful_placements"] == 0
    assert json.loads(high_calyx["structured_unavailable_reason_counts"]) == {
        "distal_calyx_support_unavailable": 1
    }

    low_quartet = next(
        row
        for row in summary
        if row["summary_level"] == "quartet" and row["severity_fraction"] == 0.01
    )
    high_quartet = next(
        row
        for row in summary
        if row["summary_level"] == "quartet" and row["severity_fraction"] == 0.04
    )
    assert low_quartet["attempted_matched_quartets"] == 1
    assert low_quartet["complete_matched_quartets"] == 1
    assert low_quartet["incomplete_matched_quartets"] == 0
    assert low_quartet["completion_fraction"] == 1.0
    assert low_quartet["n_groups_with_complete_quartet"] == 1
    assert high_quartet["attempted_matched_quartets"] == 1
    assert high_quartet["complete_matched_quartets"] == 0
    assert high_quartet["incomplete_matched_quartets"] == 1
    assert high_quartet["completion_fraction"] == 0.0
    assert high_quartet["n_groups_with_complete_quartet"] == 0


def test_occluder_application_refuses_clipping() -> None:
    _, _, mask = _geometry()
    occluder = generate_synthetic_occluder(
        sample_id="fruit",
        severity_fraction=0.04,
        seed=0,
        foreground_area_pixels=int(np.count_nonzero(mask)),
    )
    from strawberry_occlusion.augmentation.synthetic_occluder import OccluderPlacement

    clipped = OccluderPlacement(
        region="attachment",
        top=-1,
        left=0,
        area_pixels=occluder.area_pixels,
        template_sha256=occluder.template_sha256,
    )
    with pytest.raises(ValueError, match="clip"):
        apply_occluder(_rgb(mask), occluder, clipped)


def test_occluder_pixels_are_excluded_and_visible_metrics_keep_semantics() -> None:
    target = torch.tensor([[0, 1], [2, 1]], dtype=torch.long)
    prediction = torch.tensor([[2, 1], [2, 0]], dtype=torch.long)
    occluder = torch.tensor([[True, False], [False, False]])

    metrics = masked_segmentation_metrics(prediction, target, occluder)

    assert metrics["visible_evaluated_pixel_count"] == 3
    assert metrics["iou_calyx"] == 1.0
    assert metrics["recall_flesh"] == 0.5
    assert metrics["precision_flesh"] == 1.0


def test_reference_sign_conventions_excess_error_and_silent_drift() -> None:
    values = reference_error_metrics(c_gt=10.0, c_clean=12.0, c_occ=18.0)
    assert values == {
        "stability_signed": 6.0,
        "stability_abs": 6.0,
        "gt_reference_signed": 8.0,
        "gt_reference_abs": 8.0,
        "clean_gt_reference_abs": 2.0,
        "excess_gt_reference_error": 6.0,
    }
    assert silent_drift_flags(
        method_status="ok", finite_cut_returned=True, stability_abs=6.0
    ) == {
        "silent_drift_2_pixels": True,
        "silent_drift_5_pixels": True,
        "silent_drift_10_pixels": False,
    }
    assert not any(
        silent_drift_flags(
            method_status="failed", finite_cut_returned=False, stability_abs=99.0
        ).values()
    )


def test_probability_consistency_accepts_softmax_rounding_tie_at_predicted_class() -> (
    None
):
    logits = torch.tensor(
        [0.0, torch.nextafter(torch.tensor(0.0), torch.tensor(1.0)), -100.0]
    )
    probabilities = torch.softmax(logits, dim=0).reshape(3, 1, 1)
    assert probabilities[0, 0, 0] == probabilities[1, 0, 0]
    assert logits.argmax().item() == 1
    assert probabilities.argmax().item() == 0

    output = _validated_inference_output(
        SegmentationInferenceOutput(
            prediction=torch.tensor([[1]], dtype=torch.long),
            normalized_entropy=torch.tensor([[0.5]]),
            probabilities=probabilities,
        ),
        shape=(1, 1),
    )

    assert output.prediction.item() == 1


def test_probability_consistency_rejects_prediction_below_probability_maximum() -> None:
    with pytest.raises(
        ValueError,
        match="probabilities must attain their maximum at the predicted class",
    ):
        _validated_inference_output(
            SegmentationInferenceOutput(
                prediction=torch.tensor([[1]], dtype=torch.long),
                normalized_entropy=torch.tensor([[0.5]]),
                probabilities=torch.tensor([[[0.7]], [[0.2]], [[0.1]]]),
            ),
            shape=(1, 1),
        )


def test_two_tensor_inference_output_remains_supported_without_probabilities() -> None:
    output = _validated_inference_output(
        (
            torch.tensor([[2]], dtype=torch.long),
            torch.tensor([[0.25]], dtype=torch.float32),
        ),
        shape=(1, 1),
    )

    assert output.prediction.item() == 2
    assert output.probabilities is None


def test_frozen_profile_and_cli_expose_no_geometry_or_severity_tuning() -> None:
    profile = benchmark_profile()
    assert profile["severity_fractions"] == [0.01, 0.02, 0.04]
    assert profile["seeds"] == [0, 1]
    assert profile["geometry"] == FROZEN_FIXED_AXIS_CONFIGURATION
    assert profile["geometry"]["configuration_id"] == "e050_w064"
    assert profile["training_or_optimization"] is False
    profile["geometry"]["removal_axis"][0] = 99.0
    profile["placement"]["attachment_roi_dilation_pixels"] = 99
    fresh_profile = benchmark_profile()
    assert fresh_profile["geometry"] == FROZEN_FIXED_AXIS_CONFIGURATION
    assert {
        key: fresh_profile["placement"][key] for key in M4_V1_PLACEMENT_PARAMETERS
    } == dict(M4_V1_PLACEMENT_PARAMETERS)

    parser = _build_argument_parser()
    help_text = parser.format_help()
    assert "projection-quantile" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--profile",
                "m4_v1",
                "--development-manifest",
                "manifest.csv",
                "--safe-data-root",
                "safe",
                "--checkpoint",
                "checkpoint.pt",
                "--output-root",
                "output",
                "--device",
                "cpu",
                "--severity",
                "0.5",
            ]
        )


def test_repeated_images_share_one_group_and_counts_do_not_explode() -> None:
    rows = _synthetic_trial_rows()
    metadata = [
        {"sample_id": "image_a", "group_id": "group_a", "session_id": "session_0"},
        {"sample_id": "image_b", "group_id": "group_a", "session_id": "session_1"},
    ]
    analysis = summarize_robustness_trials(rows, metadata)

    group_row = analysis["per_group_summary"][0]
    assert group_row["n_groups"] == 1
    assert group_row["n_images"] == 2
    assert group_row["n_sessions"] == 2
    assert group_row["perturbation_evaluations"] > group_row["n_groups"]
    assert all(row["n_groups"] == 1 for row in analysis["condition_summary"])

    exploded = rows * 20
    exploded_analysis = summarize_robustness_trials(exploded, metadata)
    assert exploded_analysis["condition_summary"][0]["n_groups"] == 1


def test_auc_is_none_when_required_severity_coverage_is_incomplete() -> None:
    rows = [
        row
        for row in _synthetic_trial_rows(sample_ids=("image_a",))
        if row["severity_fraction"] != PROFILE_SEVERITIES[-1]
    ]

    analysis = summarize_robustness_trials(rows, ())
    group_row = next(
        row
        for row in analysis["per_group_summary"]
        if row["method"] == "v2a" and row["region"] == "attachment"
    )

    assert group_row["stability_abs_auc"] is None
    assert group_row["structured_failure_rate_auc"] is None


def test_paired_comparisons_use_common_matched_sets_only() -> None:
    rows = _synthetic_trial_rows(sample_ids=("image_a",))
    removed_set = "image_a-0.04-1"
    rows = [
        row
        for row in rows
        if not (row["region"] == "background" and row["matched_set_id"] == removed_set)
    ]
    analysis = summarize_robustness_trials(rows, ())
    primary = next(
        row
        for row in analysis["paired_region_differences"]
        if row["method"] == "v2a" and row["comparison"] == "attachment_minus_background"
    )
    assert primary["comparison_priority"] == "primary"
    assert primary["perturbation_evaluations"] == 10


def test_bootstrap_is_group_resampled_deterministic_and_reports_small_n() -> None:
    rows = _synthetic_trial_rows(sample_ids=("image_a",))
    analysis = summarize_robustness_trials(rows, ())
    paired = analysis["paired_region_differences"]
    first = bootstrap_group_intervals(paired, seed=BOOTSTRAP_SEED, replicates=100)
    second = bootstrap_group_intervals(paired, seed=BOOTSTRAP_SEED, replicates=100)
    assert first == second
    assert all(row["n_groups"] == 1 for row in first)
    assert all(row["stable_interval"] is False for row in first)
    assert all("too few independent groups" in row["interval_note"] for row in first)
    assert not any("p_value" in row for row in first)


def test_bootstrap_with_six_groups_is_non_degenerate() -> None:
    result = _bootstrap_result(_paired_group_rows([-8, -3, 0, 4, 9, 14]))

    assert result["n_groups"] == 6
    assert result["stable_interval"] is True
    assert result["lower_bound"] < result["upper_bound"]


def test_group_bootstrap_is_unchanged_by_duplicate_perturbation_rows() -> None:
    rows = _paired_group_rows([-8, -3, 0, 4, 9, 14])
    original = _bootstrap_result(rows)
    duplicated = _bootstrap_result(rows * 10)

    assert duplicated["n_groups"] == original["n_groups"] == 6
    assert duplicated["mean_paired_difference"] == original["mean_paired_difference"]
    assert duplicated["lower_bound"] == original["lower_bound"]
    assert duplicated["upper_bound"] == original["upper_bound"]


def test_adding_independent_groups_can_narrow_group_bootstrap_interval() -> None:
    original_rows = _paired_group_rows([-12, -6, 0, 6, 12, 18])
    expanded_rows = original_rows + _paired_group_rows(
        [3.0] * 30,
        group_prefix="additional_group",
    )
    original = _bootstrap_result(original_rows)
    expanded = _bootstrap_result(expanded_rows)

    original_width = original["upper_bound"] - original["lower_bound"]
    expanded_width = expanded["upper_bound"] - expanded["lower_bound"]
    assert expanded["n_groups"] == 36
    assert expanded_width < original_width


def test_bootstrap_seed_is_independent_of_unrelated_comparisons_and_order() -> None:
    target_rows = _paired_group_rows([-8, -3, 0, 4, 9, 14])
    unrelated = _paired_group_rows(
        [100, 101, 102, 103, 104, 105],
        comparison="attachment_minus_flesh_far",
        group_prefix="unrelated_group",
    )
    target_only = _bootstrap_result(target_rows)
    with_unrelated = _bootstrap_result(list(reversed(unrelated + target_rows)))

    assert with_unrelated["bootstrap_seed"] == target_only["bootstrap_seed"]
    assert (
        with_unrelated["mean_paired_difference"]
        == target_only["mean_paired_difference"]
    )
    assert with_unrelated["lower_bound"] == target_only["lower_bound"]
    assert with_unrelated["upper_bound"] == target_only["upper_bound"]


def test_matched_quartet_visualization_is_headless_and_deterministic() -> None:
    v2a, v2b, mask = _geometry()
    image = _rgb(mask)
    conditions = {
        region: {
            "image": image,
            "clean_results": {"v2a": v2a, "v2b": v2b},
            "occluded_results": {"v2a": v2a, "v2b": v2b},
            "metrics": {
                "v2a": {
                    "stability_signed": 0.0,
                    "gt_reference_abs": 0.0,
                    "method_status": "ok",
                },
                "v2b": {
                    "stability_signed": 0.0,
                    "gt_reference_abs": 0.0,
                    "method_status": "ok",
                },
            },
        }
        for region in MATCHED_REGIONS
    }
    first = create_matched_quartet_visualization(
        image,
        conditions,
        sample_id="fruit",
        severity_fraction=0.01,
        seed=0,
        occluder_id="occ_public",
    )
    second = create_matched_quartet_visualization(
        image,
        conditions,
        sample_id="fruit",
        severity_fraction=0.01,
        seed=0,
        occluder_id="occ_public",
    )
    assert first.mode == "RGB"
    assert first.tobytes() == second.tobytes()


def test_checkpoint_epoch_other_than_92_is_rejected(tmp_path: Path) -> None:
    safe_root = tmp_path / "safe"
    manifest_path = tmp_path / "development.csv"
    _write_manifest(manifest_path, [_manifest_row(safe_root)])
    checkpoint = tmp_path / "epoch91.pt"
    _write_checkpoint(checkpoint, completed_epoch=91)

    with pytest.raises(ValueError, match="epoch-92"):
        run_occlusion_robustness(
            manifest_path,
            safe_root,
            checkpoint,
            tmp_path / "output",
            device="cpu",
            model_factory=TinyM4Model,
        )


def test_overwrite_replaces_only_an_existing_output_directory(tmp_path: Path) -> None:
    destination = tmp_path / "output"
    destination.mkdir()
    (destination / "old.txt").write_text("old", encoding="utf-8")
    with pytest.raises(FileExistsError, match="already exists"):
        _validate_output_destination(destination, overwrite=False)
    _validate_output_destination(destination, overwrite=True)

    staging = tmp_path / "staging"
    staging.mkdir()
    (staging / "new.txt").write_text("new", encoding="utf-8")
    _install_staged_directory(staging, destination, overwrite=True)

    assert not staging.exists()
    assert not (destination / "old.txt").exists()
    assert (destination / "new.txt").read_text(encoding="utf-8") == "new"


def test_failed_geometry_persists_empty_coordinate_and_stability_fields(
    tmp_path: Path,
) -> None:
    safe_root = tmp_path / "safe"
    manifest_path = tmp_path / "development.csv"
    _write_manifest(manifest_path, [_manifest_row(safe_root)])
    checkpoint = tmp_path / "epoch92-background.pt"
    _write_checkpoint(checkpoint, model_class=AlwaysBackgroundM4Model)
    output = tmp_path / "output"

    run_occlusion_robustness(
        manifest_path,
        safe_root,
        checkpoint,
        output,
        device="cpu",
        model_factory=AlwaysBackgroundM4Model,
    )

    with (output / "perturbation_trials.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        trials = list(csv.DictReader(input_file))
    assert trials
    for row in trials:
        assert row["method_status"] != "ok"
        assert row["finite_cut_returned"] == "false"
        for field in (
            "c_occ",
            "stability_signed",
            "stability_abs",
            "gt_reference_signed",
            "gt_reference_abs",
            "excess_gt_reference_error",
        ):
            assert row[field] == ""


def test_full_synthetic_run_uses_rgb_only_and_writes_sanitized_artifacts(
    tmp_path: Path,
) -> None:
    safe_root = tmp_path / "safe"
    row = _manifest_row(safe_root)
    manifest_path = tmp_path / "development.csv"
    _write_manifest(manifest_path, [row])
    checkpoint = tmp_path / "epoch92.pt"
    _write_checkpoint(checkpoint)
    output = tmp_path / "output"
    calls: list[tuple[int, ...]] = []

    def inference_only(
        model: nn.Module,
        image: torch.Tensor,
        *,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        calls.append(tuple(image.shape))
        batch = image.unsqueeze(0).to(device)
        logits = model(batch)
        probabilities = torch.softmax(logits, dim=1)
        entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
        entropy = entropy / np.log(3.0)
        return logits.argmax(dim=1)[0].cpu(), entropy[0].cpu()

    result = run_occlusion_robustness(
        manifest_path,
        safe_root,
        checkpoint,
        output,
        device="cpu",
        model_factory=TinyM4Model,
        inference_function=inference_only,
        visualization_sample_ids=("fruit_01",),
    )

    assert calls and all(shape == (3, HEIGHT, WIDTH) for shape in calls)
    assert result["manifest"]["n_groups"] == 1
    assert result["manifest"]["n_images"] == 1
    assert result["manifest"]["n_sessions"] == 1
    assert result["manifest"]["perturbation_evaluations"] == 24
    assert result["manifest"]["checkpoint"]["epoch"] == 92
    assert result["manifest"]["checkpoint"]["basename"] == "epoch92.pt"
    assert result["manifest"]["placement_parameters"] == dict(
        M4_V1_PLACEMENT_PARAMETERS
    )
    assert (
        "conditional on successful finite cut estimates"
        in result["manifest"]["analysis_conditioning"]["coordinate_stability"]
    )
    assert (
        "missing perturbations are not imputed"
        in result["manifest"]["analysis_conditioning"]["severity_coverage"]
    )
    assert "absolute" not in json.dumps(result["manifest"]).lower()
    assert all(
        (output / name).is_file()
        for name in (
            "manifest.json",
            "benchmark_profile.json",
            "development_metadata_snapshot.csv",
            "clean_reference_per_image.csv",
            "perturbation_trials.csv",
            "incomplete_matched_sets.csv",
            "matched_set_completeness.csv",
            "per_image_summary.csv",
            "per_group_summary.csv",
            "condition_summary.csv",
            "paired_region_differences.csv",
            "bootstrap_intervals.csv",
            "failure_awareness_summary.csv",
            "segmentation_summary.csv",
        )
    )
    assert all((output / "plots" / filename).is_file() for filename in PLOT_FILENAMES)
    with (output / "perturbation_trials.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        trials = list(csv.DictReader(input_file))
    assert len(trials) == 48
    assert tuple(trials[0]) == TRIAL_FIELDS
    for trial in trials:
        if trial["region"] in {"calyx_tip", "flesh_far"}:
            assert int(trial["occluder_attachment_roi_overlap_pixels"]) == 0
        if trial["region"] == "background":
            assert int(trial["occluder_foreground_overlap_pixels"]) == 0
    for matched_set in {row["matched_set_id"] for row in trials}:
        selected = [row for row in trials if row["matched_set_id"] == matched_set]
        assert len({row["occluder_template_sha256"] for row in selected}) == 1
        assert len({row["occluder_area_pixels"] for row in selected}) == 1
        assert {row["method"] for row in selected} == {"v2a", "v2b"}
    with (output / "matched_set_completeness.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        completeness = list(csv.DictReader(input_file))
    assert len(completeness) == len(PROFILE_SEVERITIES) * (len(MATCHED_REGIONS) + 1)
    assert {row["summary_level"] for row in completeness} == {"region", "quartet"}
    assert all(
        row["complete_matched_quartets"] == "2"
        for row in completeness
        if row["summary_level"] == "quartet"
    )

    deterministic_csvs = (
        "per_image_summary.csv",
        "per_group_summary.csv",
        "condition_summary.csv",
        "paired_region_differences.csv",
        "bootstrap_intervals.csv",
        "failure_awareness_summary.csv",
        "segmentation_summary.csv",
        "matched_set_completeness.csv",
    )
    first_summary = summarize_existing_output(output)
    first_bytes = {name: (output / name).read_bytes() for name in deterministic_csvs}
    second_summary = summarize_existing_output(output)
    assert first_summary == second_summary
    assert first_bytes == {
        name: (output / name).read_bytes() for name in deterministic_csvs
    }
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in output.iterdir()
        if path.is_file()
    )
    assert str(tmp_path.resolve()) not in persisted
    assert "physical danger" not in persisted.lower()
    assert "optimizer" not in persisted.lower()
