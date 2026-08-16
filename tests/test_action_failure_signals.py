from __future__ import annotations

import csv
import inspect
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

from strawberry_occlusion.evaluation.action_failure_signals import (
    ATTACHMENT_ROI_DILATION_PIXELS,
    PROFILE_NAME,
    _build_argument_parser,
    _spearman_from_group_rows,
    aggregate_probability_signals,
    analyze_action_failure_observations,
    derive_predicted_attachment_roi,
    extract_inference_time_signals,
    frozen_m5_configuration,
    geometry_support_signals,
    predictive_entropy,
    probability_margin,
    run_action_failure_signals,
    summarize_existing_action_failure_output,
)
from strawberry_occlusion.evaluation.occlusion_robustness import (
    REQUIRED_MANIFEST_FIELDS,
    TRIAL_FIELDS,
)
from strawberry_occlusion.evaluation.pilot_holdout import _run_frozen_geometry
from strawberry_occlusion.geometry import (
    estimate_fixed_axis_cutline,
    estimate_fixed_axis_search_cutline,
)
from strawberry_occlusion.visualization.action_failure_signals import (
    PLOT_FILENAMES,
)


HEIGHT = 128
WIDTH = 160


class TinyM5Model(nn.Module):
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


def _semantic_mask() -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[32:100, 25:105] = 1
    mask[50:82, 105:130] = 2
    return mask


def _geometry(mask: np.ndarray | None = None) -> tuple[Any, Any]:
    return _run_frozen_geometry(
        _semantic_mask() if mask is None else mask,
        v2a_estimator=estimate_fixed_axis_cutline,
        v2b_estimator=estimate_fixed_axis_search_cutline,
    )


def _probabilities(shape: tuple[int, int] = (HEIGHT, WIDTH)) -> np.ndarray:
    values = np.zeros((3, *shape), dtype=np.float64)
    values[0] = 0.8
    values[1] = 0.15
    values[2] = 0.05
    return values


def test_predictive_entropy_uses_natural_log_categorical_definition() -> None:
    probabilities = np.array(
        [
            [[0.5, 1.0 / 3.0]],
            [[0.5, 1.0 / 3.0]],
            [[0.0, 1.0 / 3.0]],
        ]
    )

    entropy = predictive_entropy(probabilities)

    assert entropy.shape == (1, 2)
    assert entropy[0, 0] == pytest.approx(np.log(2.0))
    assert entropy[0, 1] == pytest.approx(np.log(3.0))
    assert entropy.flags.writeable is False


def test_probability_margin_is_top_one_minus_top_two() -> None:
    probabilities = np.array(
        [
            [[0.7, 0.2]],
            [[0.2, 0.5]],
            [[0.1, 0.3]],
        ]
    )

    margin = probability_margin(probabilities)

    assert np.allclose(margin, [[0.5, 0.2]])
    assert margin.flags.writeable is False


def test_predicted_attachment_roi_and_aggregation_are_deterministic() -> None:
    v2a, _ = _geometry()
    first = derive_predicted_attachment_roi(v2a)
    second = derive_predicted_attachment_roi(v2a)
    probabilities = _probabilities()
    probabilities[:, first] = np.array([[1 / 3], [1 / 3], [1 / 3]])

    signals = aggregate_probability_signals(probabilities, first)

    assert np.array_equal(first, second)
    assert first.flags.writeable is False
    assert np.count_nonzero(first) > np.count_nonzero(v2a.contact_mask)
    assert signals["predictive_entropy_attachment"] == pytest.approx(np.log(3.0))
    assert signals["max_probability_attachment"] == pytest.approx(1.0 / 3.0)
    assert signals["probability_margin_attachment"] == pytest.approx(0.0)


def test_empty_predicted_attachment_roi_propagates_missing_local_signals() -> None:
    empty = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    v2a, v2b = _geometry(empty)

    signals = extract_inference_time_signals(_probabilities(), v2a, v2b)

    assert signals["attachment_roi_pixel_count"] == 0
    assert signals["attachment_roi_defined"] is False
    assert signals["attachment_roi_status"] == "empty_predicted_attachment_contact_roi"
    assert signals["predictive_entropy_attachment"] is None
    assert signals["max_probability_attachment"] is None
    assert signals["probability_margin_attachment"] is None


def test_primary_signal_constructor_exposes_no_ground_truth_or_clean_input() -> None:
    parameter_names = set(inspect.signature(extract_inference_time_signals).parameters)
    forbidden_fragments = ("ground", "target", "manual", "clean", "future")

    assert not any(
        fragment in parameter.lower()
        for parameter in parameter_names
        for fragment in forbidden_fragments
    )
    assert parameter_names == {
        "probabilities",
        "v2a_result",
        "v2b_result",
        "attachment_roi_dilation_pixels",
    }


def test_inference_time_signals_ignore_changed_manual_attachment_geometry() -> None:
    predicted_v2a, predicted_v2b = _geometry()
    manual_mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    manual_mask[32:100, 55:135] = 1
    manual_mask[50:82, 30:55] = 2
    manual_v2a, _ = _geometry(manual_mask)
    predicted_roi = derive_predicted_attachment_roi(predicted_v2a)
    manual_roi = derive_predicted_attachment_roi(manual_v2a)
    assert not np.array_equal(predicted_roi, manual_roi)

    probabilities = _probabilities()
    probabilities[:, predicted_roi] = np.array([[1 / 3], [1 / 3], [1 / 3]])
    first = extract_inference_time_signals(
        probabilities,
        predicted_v2a,
        predicted_v2b,
    )
    changed_manual_information = {
        "ground_truth_mask": manual_mask,
        "manual_v2a": manual_v2a,
        "manual_attachment_roi": manual_roi,
    }
    assert changed_manual_information["manual_attachment_roi"].any()
    second = extract_inference_time_signals(
        probabilities.copy(),
        predicted_v2a,
        predicted_v2b,
    )

    assert first == second
    expected = aggregate_probability_signals(probabilities, predicted_roi)
    manual_based = aggregate_probability_signals(probabilities, manual_roi)
    assert first["attachment_roi_pixel_count"] == int(np.count_nonzero(predicted_roi))
    assert first["predictive_entropy_attachment"] == pytest.approx(
        expected["predictive_entropy_attachment"]
    )
    assert first["predictive_entropy_attachment"] != pytest.approx(
        manual_based["predictive_entropy_attachment"]
    )


def test_geometry_support_signals_are_exact_reuses_or_deterministic_ratios() -> None:
    _, v2b = _geometry()

    signals = geometry_support_signals(v2b)

    assert signals["v2b_candidate_count"] == v2b.candidate_count
    assert signals["v2b_feasible_candidate_count"] == v2b.feasible_candidate_count
    assert signals["v2b_feasible_block_count"] == v2b.feasible_block_count
    assert signals["v2b_selected_block_is_singleton"] is v2b.selected_block_is_singleton
    assert signals["v2b_selected_block_relative_size"] == pytest.approx(
        v2b.selected_block_candidate_count / v2b.feasible_candidate_count
    )
    assert signals["v2b_selected_contact_evidence_fraction"] == pytest.approx(
        v2b.selected_local_contact_count / v2b.maximum_local_contact_evidence
    )


def _association_fixture() -> list[dict[str, Any]]:
    rows = []
    for index in range(4):
        rows.append(
            {
                "group_id": f"group_{index}",
                "method": "v2b",
                "region": "attachment",
                "predictive_entropy_global": float(index + 1),
                "stability_abs": float((index + 1) * 2),
                "method_status": "ok",
            }
        )
    rows.append(
        {
            "group_id": "group_0",
            "method": "v2b",
            "region": "attachment",
            "predictive_entropy_global": 9.0,
            "stability_abs": None,
            "method_status": "failed",
        }
    )
    rows.append(
        {
            "group_id": "group_1",
            "method": "v2b",
            "region": "attachment",
            "predictive_entropy_global": None,
            "stability_abs": 3.0,
            "method_status": "ok",
        }
    )
    return rows


def test_group_level_association_missing_propagation_and_reproducibility() -> None:
    rows = _association_fixture()

    first = analyze_action_failure_observations(rows)
    second = analyze_action_failure_observations(rows)
    association = next(
        row
        for row in first["association_table"]
        if row["method"] == "v2b"
        and row["region_scope"] == "all"
        and row["signal"] == "predictive_entropy_global"
    )

    assert first == second
    assert association["n_groups_total"] == 4
    assert association["n_groups_pairwise_complete"] == 4
    assert association["pairwise_observation_count"] == 4
    assert association["missing_signal_observation_count"] == 1
    assert association["missing_outcome_observation_count"] == 1
    assert association["spearman_rho"] == pytest.approx(1.0)
    assert association["undefined_reason"] is None
    group_rows = [
        row
        for row in first["per_group_signal_summary"]
        if row["method"] == "v2b"
        and row["region_scope"] == "all"
        and row["signal"] == "predictive_entropy_global"
    ]
    assert len(group_rows) == 4
    assert all(row["paired_observation_count"] == 1 for row in group_rows)


def test_association_is_group_first_when_row_pooling_would_reverse_the_sign() -> None:
    rows = []
    for group_index in range(3):
        for within_group_signal in np.linspace(0.0, 10.0, 5):
            rows.append(
                {
                    "group_id": f"group_{group_index}",
                    "method": "v2b",
                    "region": "attachment",
                    "predictive_entropy_global": (within_group_signal + group_index),
                    "stability_abs": (9.0 + group_index - within_group_signal),
                    "method_status": "ok",
                }
            )

    analysis = analyze_action_failure_observations(rows)
    association = next(
        row
        for row in analysis["association_table"]
        if row["method"] == "v2b"
        and row["region_scope"] == "all"
        and row["signal"] == "predictive_entropy_global"
    )
    group_rows = [
        row
        for row in analysis["per_group_signal_summary"]
        if row["method"] == "v2b"
        and row["region_scope"] == "all"
        and row["signal"] == "predictive_entropy_global"
    ]
    pooled_rho, _ = _spearman_from_group_rows(
        [
            {
                "mean_signal": row["predictive_entropy_global"],
                "mean_stability_abs": row["stability_abs"],
            }
            for row in rows
        ]
    )

    assert association["n_groups_pairwise_complete"] == 3
    assert association["pairwise_observation_count"] == 15
    assert association["spearman_rho"] == pytest.approx(1.0)
    assert pooled_rho is not None and pooled_rho < -0.9
    assert [row["paired_observation_count"] for row in group_rows] == [5, 5, 5]
    assert [row["mean_signal"] for row in group_rows] == [5.0, 6.0, 7.0]
    assert [row["mean_stability_abs"] for row in group_rows] == [4.0, 5.0, 6.0]


def test_spearman_tied_ranks_have_pinned_average_rank_result() -> None:
    rho, reason = _spearman_from_group_rows(
        [
            {"mean_signal": 1.0, "mean_stability_abs": 1.0},
            {"mean_signal": 1.0, "mean_stability_abs": 2.0},
            {"mean_signal": 2.0, "mean_stability_abs": 2.0},
            {"mean_signal": 3.0, "mean_stability_abs": 3.0},
        ]
    )

    assert rho == pytest.approx(5.0 / 6.0)
    assert reason is None


@pytest.mark.parametrize(
    ("rows", "expected_reason"),
    (
        (
            [
                {"mean_signal": 1.0, "mean_stability_abs": 1.0},
                {"mean_signal": 2.0, "mean_stability_abs": 2.0},
            ],
            "fewer_than_3_pairwise_complete_groups",
        ),
        (
            [
                {"mean_signal": 1.0, "mean_stability_abs": 1.0},
                {"mean_signal": 1.0, "mean_stability_abs": 2.0},
                {"mean_signal": 1.0, "mean_stability_abs": 3.0},
            ],
            "constant_group_mean_signal",
        ),
        (
            [
                {"mean_signal": 1.0, "mean_stability_abs": 2.0},
                {"mean_signal": 2.0, "mean_stability_abs": 2.0},
                {"mean_signal": 3.0, "mean_stability_abs": 2.0},
            ],
            "constant_group_mean_outcome",
        ),
    ),
)
def test_spearman_undefined_edge_cases(
    rows: list[dict[str, float]],
    expected_reason: str,
) -> None:
    rho, reason = _spearman_from_group_rows(rows)

    assert rho is None
    assert reason == expected_reason


def test_frozen_configuration_and_cli_expose_no_signal_or_geometry_tuning() -> None:
    configuration = frozen_m5_configuration()
    assert configuration["profile"] == PROFILE_NAME
    assert configuration["source_benchmark"]["profile"] == "m4_v1"
    assert configuration["attachment_roi"]["dilation_radius_pixels"] == 5
    assert configuration["statistics"]["independent_unit"] == "group_id"
    assert configuration["statistics"]["n_association_coefficients"] is None
    assert configuration["statistics"]["multiplicity_correction"] is None
    assert configuration["selective_action_analysis"] == "deferred_to_m5_v1"
    assert configuration["training_or_optimization"] is False

    parser = _build_argument_parser()
    help_text = parser.format_help()
    assert "roi-dilation" not in help_text
    assert "severity" not in help_text
    assert "projection-quantile" not in help_text
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "run",
                "--profile",
                PROFILE_NAME,
                "--development-manifest",
                "development.csv",
                "--safe-data-root",
                "safe",
                "--checkpoint",
                "checkpoint.pt",
                "--output-root",
                "output",
                "--device",
                "cpu",
                "--roi-dilation",
                "9",
            ]
        )


def _write_synthetic_inputs(root: Path) -> tuple[Path, Path, Path]:
    safe_root = root / "safe"
    image_path = safe_root / "images" / "fruit_01.png"
    mask_path = safe_root / "masks" / "fruit_01.png"
    image_path.parent.mkdir(parents=True)
    mask_path.parent.mkdir(parents=True)
    mask = _semantic_mask()
    image = np.zeros((*mask.shape, 3), dtype=np.uint8)
    image[mask == 1] = (220, 35, 35)
    image[mask == 2] = (35, 220, 35)
    Image.fromarray(image, mode="RGB").save(image_path)
    Image.fromarray(mask, mode="L").save(mask_path)
    manifest = root / "development.csv"
    with manifest.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=REQUIRED_MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": "fruit_01",
                "group_id": "group_01",
                "session_id": "session_01",
                "role": "development",
                "image_path": "images/fruit_01.png",
                "mask_path": "masks/fruit_01.png",
                "orientation_category": "right",
                "fixed_axis_eligible": "true",
                "annotation_qa_status": "reviewed",
            }
        )
    checkpoint = root / "epoch92.pt"
    model = TinyM5Model()
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "completed_epoch": 92,
            "best_mean_foreground_iou": 0.5,
            "class_mapping": {"background": 0, "Flesh": 1, "Calyx": 2},
            "input_height": HEIGHT,
            "input_width": WIDTH,
            "model_name": "TinyM5Model",
        },
        checkpoint,
    )
    return manifest, safe_root, checkpoint


def test_full_synthetic_m5_run_writes_sanitized_reproducible_outputs(
    tmp_path: Path,
) -> None:
    manifest, safe_root, checkpoint = _write_synthetic_inputs(tmp_path)
    output = tmp_path / "m5_output"

    result = run_action_failure_signals(
        manifest,
        safe_root,
        checkpoint,
        output,
        device="cpu",
        model_factory=TinyM5Model,
    )

    assert result["manifest"]["profile"] == PROFILE_NAME
    assert result["summary"]["counts"]["n_groups_represented"] == 1
    assert result["summary"]["counts"]["n_perturbations_represented"] == 24
    assert result["summary"]["n_association_coefficients"] == len(
        result["analysis"]["association_table"]
    )
    assert all((output / "plots" / name).is_file() for name in PLOT_FILENAMES)
    with (output / "per_observation_signals.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        observations = list(csv.DictReader(input_file))
    assert len(observations) == 48
    assert all(
        row["attachment_roi_definition"]
        == "square_dilation_of_current_predicted_v2a_contact_mask"
        for row in observations
    )
    assert all(
        row["v2b_candidate_count"] == ""
        for row in observations
        if row["method"] == "v2a"
    )
    with (output / "m4_reference" / "perturbation_trials.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        m4_reader = csv.DictReader(input_file)
        assert tuple(m4_reader.fieldnames or ()) == TRIAL_FIELDS
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in output.iterdir()
        if path.is_file()
    )
    assert str(tmp_path.resolve()) not in persisted
    deterministic = (
        "per_group_signal_summary.csv",
        "signal_descriptive_statistics.csv",
        "association_table.csv",
        "summary.json",
        "configuration.json",
    )
    observation_path = output / "per_observation_signals.csv"
    observation_bytes = observation_path.read_bytes()
    observation_mtime = observation_path.stat().st_mtime_ns
    run_derived_bytes = {name: (output / name).read_bytes() for name in deterministic}
    summarize_existing_action_failure_output(output)
    assert observation_path.read_bytes() == observation_bytes
    assert observation_path.stat().st_mtime_ns == observation_mtime
    assert run_derived_bytes == {
        name: (output / name).read_bytes() for name in deterministic
    }
    first_summary_bytes = {name: (output / name).read_bytes() for name in deterministic}
    summarize_existing_action_failure_output(output)
    assert observation_path.read_bytes() == observation_bytes
    assert observation_path.stat().st_mtime_ns == observation_mtime
    assert first_summary_bytes == {
        name: (output / name).read_bytes() for name in deterministic
    }
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    configuration = json.loads(
        (output / "configuration.json").read_text(encoding="utf-8")
    )
    assert summary["claims"]["calibrated_confidence"] is False
    assert summary["n_association_coefficients"] == len(
        result["analysis"]["association_table"]
    )
    assert configuration["statistics"]["n_association_coefficients"] == len(
        result["analysis"]["association_table"]
    )
    assert ATTACHMENT_ROI_DILATION_PIXELS == 5
