import csv
import json
import subprocess
import sys
from dataclasses import replace
from itertools import count
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

import strawberry_occlusion.evaluation.fixed_axis_sensitivity as sensitivity
import strawberry_occlusion.visualization.fixed_axis_sensitivity as sensitivity_plots
from strawberry_occlusion.evaluation.fixed_axis_sensitivity import (
    COMMITTED_BASELINE_CONFIGURATION_ID,
    CONFIGURATION_AGGREGATE_FIELDS,
    CONFIGURATION_BASELINE_CHANGE_FIELDS,
    CONFIGURATION_SUMMARY_FIELDS,
    DEFAULT_COORDINATE_TOLERANCE_PIXELS,
    DEFAULT_FEASIBLE_BLOCK_COUNT_FLAG_THRESHOLD,
    DEFAULT_LATERAL_WINDOW_HALF_WIDTH_PIXELS,
    DEFAULT_MINIMUM_ATTACHMENT_EVIDENCE_FRACTIONS,
    FLAGGED_SAMPLE_FIELDS,
    PARAMETER_GRID_FIELDS,
    PER_SAMPLE_CONFIGURATION_FIELDS,
    PER_SAMPLE_STABILITY_FIELDS,
    _build_argument_parser,
    _configuration_summary_rows,
    _parse_repeated_float_values,
    _sample_flag_rows,
    _sample_stability_row,
    build_parameter_grid,
    run_fixed_axis_sensitivity_dataset,
)
from strawberry_occlusion.geometry import estimate_fixed_axis_search_cutline
from strawberry_occlusion.visualization.fixed_axis_sensitivity import (
    _distinguishing_decimal_places,
    _draw_heatmap,
    _finite_panel_limits,
    _plot_configuration_tradeoffs,
    _plot_coordinate_stability,
)


PLOT_FILENAMES = {
    "aggregate_metric_heatmaps": "aggregate_metric_heatmaps.png",
    "status_and_fragmentation_heatmaps": "status_and_fragmentation_heatmaps.png",
    "coordinate_stability": "coordinate_stability.png",
    "configuration_tradeoffs": "configuration_tradeoffs.png",
}


def _horizontal_mask() -> np.ndarray:
    mask = np.zeros((16, 20), dtype=np.uint8)
    mask[4:12, 2:12] = 1
    mask[7:9, 12:16] = 2
    return mask


def _draped_mask() -> np.ndarray:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:20, 3:17] = 1
    mask[8:11, 8:19] = 2
    return mask


def _make_split(root: Path, split: str) -> None:
    (root / split / "images").mkdir(parents=True, exist_ok=True)
    (root / split / "masks").mkdir(parents=True, exist_ok=True)


def _write_pair(
    root: Path,
    split: str,
    sample_id: str,
    mask: np.ndarray,
    *,
    image_size: tuple[int, int] | None = None,
) -> None:
    _make_split(root, split)
    height, width = mask.shape[:2]
    Image.new(
        "RGB",
        image_size or (width, height),
        color=(30, 60, 90),
    ).save(root / split / "images" / f"{sample_id}.png")
    Image.fromarray(mask).save(root / split / "masks" / f"{sample_id}.png")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _stub_plots(
    configuration_rows: Any,
    stability_rows: Any,
    output_root: str | Path,
) -> dict[str, str]:
    assert configuration_rows
    assert stability_rows
    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    for filename in PLOT_FILENAMES.values():
        Image.new("RGB", (2, 2), color=(10, 20, 30)).save(root / filename)
    return dict(PLOT_FILENAMES)


def _run(
    dataset_root: Path,
    output_root: Path,
    **arguments: Any,
) -> dict[str, Any]:
    arguments.setdefault("plot_function", _stub_plots)
    return run_fixed_axis_sensitivity_dataset(
        dataset_root,
        output_root,
        split=arguments.pop("split", "train"),
        **arguments,
    )


def _analysis_row(
    configuration_id: str,
    *,
    coordinate: float | None = 10.0,
    status: str = "ok",
    v2a_coordinate: float = 12.0,
    flesh_proxy: float | None = 0.2,
    calyx_proxy: float | None = 0.3,
    block_count: int = 1,
    feasible_count: int = 2,
    singleton: bool | None = False,
    largest: bool | None = True,
) -> dict[str, Any]:
    shift = coordinate - v2a_coordinate if coordinate is not None else None
    success = status == "ok" and coordinate is not None
    return {
        "sample_id": "sample",
        "split": "train",
        "configuration_id": configuration_id,
        "is_committed_baseline": (
            configuration_id == COMMITTED_BASELINE_CONFIGURATION_ID
        ),
        "v2b_status": status,
        "structured_status": "ok" if success else "failed:synthetic_failure",
        "v2b_cut_coordinate": coordinate,
        "v2b_equals_v2a_within_tolerance": (
            success and abs(shift) <= DEFAULT_COORDINATE_TOLERANCE_PIXELS
        ),
        "v2b_moved_inward": (success and shift < -DEFAULT_COORDINATE_TOLERANCE_PIXELS),
        "inward_shift_pixels": max(0.0, -shift) if success else None,
        "feasible_candidate_count": feasible_count,
        "feasible_block_count": block_count,
        "selected_block_is_singleton": singleton,
        "selected_block_is_largest": largest,
        "selected_block_width_pixels": 0.0 if singleton else 2.0,
        "feasible_hull_width_pixels": 4.0,
        "whole_mask_flesh_loss_ratio": flesh_proxy,
        "whole_mask_calyx_retention_ratio": calyx_proxy,
        "selected_pair_flesh_loss_ratio": flesh_proxy,
        "selected_pair_calyx_retention_ratio": calyx_proxy,
        "selected_pair_retained_contact_ratio": 0.4,
    }


def test_exact_default_grid_ids_order_and_explicit_baseline() -> None:
    grid = build_parameter_grid()

    assert DEFAULT_MINIMUM_ATTACHMENT_EVIDENCE_FRACTIONS == (0.30, 0.50, 0.70)
    assert DEFAULT_LATERAL_WINDOW_HALF_WIDTH_PIXELS == (48.0, 64.0, 96.0)
    assert [configuration.configuration_id for configuration in grid] == [
        "e030_w048",
        "e030_w064",
        "e030_w096",
        "e050_w048",
        "e050_w064",
        "e050_w096",
        "e070_w048",
        "e070_w064",
        "e070_w096",
    ]
    assert len(grid) == 9
    baseline = [
        configuration for configuration in grid if configuration.is_committed_baseline
    ]
    assert len(baseline) == 1
    assert baseline[0].configuration_id == COMMITTED_BASELINE_CONFIGURATION_ID
    assert baseline[0].minimum_attachment_evidence_fraction == 0.50
    assert baseline[0].lateral_window_half_width_pixels == 64.0


@pytest.mark.parametrize(
    ("evidence", "width", "error"),
    [
        ([], [64.0], ValueError),
        ([0.5], [], ValueError),
        ([0.5, float("nan")], [64.0], ValueError),
        ([0.5], [64.0, float("inf")], ValueError),
        ([0.0, 0.5], [64.0], ValueError),
        ([0.5], [0.0, 64.0], ValueError),
        ([-0.1, 0.5], [64.0], ValueError),
        ([0.5, 1.1], [64.0], ValueError),
        ([0.5, 0.5], [64.0], ValueError),
        (["bad", 0.5], [64.0], TypeError),
    ],
)
def test_invalid_empty_nonfinite_zero_and_out_of_range_grid_values(
    evidence: Any,
    width: Any,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        build_parameter_grid(evidence, width)


def test_custom_grid_must_retain_explicit_committed_baseline() -> None:
    with pytest.raises(ValueError, match="committed baseline"):
        build_parameter_grid([0.3, 0.7], [48.0, 96.0])


def test_runner_writes_one_row_per_sample_configuration_and_exact_outputs(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "synthetic-dataset"
    output_root = tmp_path / "sensitivity"
    _write_pair(dataset_root, "train", "zeta", _horizontal_mask())
    _write_pair(dataset_root, "train", "alpha", _draped_mask())

    output = _run(dataset_root, output_root)

    assert len(output["result_rows"]) == 18
    assert [row["sample_id"] for row in output["result_rows"][:9]] == ["alpha"] * 9
    assert [row["configuration_id"] for row in output["result_rows"][:9]] == [
        configuration.configuration_id for configuration in build_parameter_grid()
    ]
    expected_files = {
        "manifest.json",
        "summary.json",
        "parameter_grid.csv",
        "configuration_summary.csv",
        "per_sample_configuration_results.csv",
        "per_sample_stability.csv",
        "flagged_samples.csv",
    }
    assert expected_files <= {path.name for path in output_root.iterdir()}
    assert sorted(path.name for path in (output_root / "plots").iterdir()) == sorted(
        PLOT_FILENAMES.values()
    )
    assert not (output_root / "visualizations").exists()
    assert tuple(_read_csv(output_root / "parameter_grid.csv")[0]) == (
        PARAMETER_GRID_FIELDS
    )
    assert tuple(
        _read_csv(output_root / "per_sample_configuration_results.csv")[0]
    ) == (PER_SAMPLE_CONFIGURATION_FIELDS)
    assert tuple(_read_csv(output_root / "per_sample_stability.csv")[0]) == (
        PER_SAMPLE_STABILITY_FIELDS
    )
    assert tuple(_read_csv(output_root / "flagged_samples.csv")[0]) == (
        FLAGGED_SAMPLE_FIELDS
    )
    configuration_fields = tuple(
        _read_csv(output_root / "configuration_summary.csv")[0]
    )
    assert configuration_fields == CONFIGURATION_SUMMARY_FIELDS
    assert "aggregate_runtime_seconds" in configuration_fields
    assert "aggregate_runtime_seconds_change_from_baseline" not in (
        configuration_fields
    )
    assert output["manifest"]["runtime"]["total_runtime_seconds"] >= 0.0
    assert set(
        output["manifest"]["runtime"]["aggregate_runtime_seconds_by_configuration"]
    ) == {configuration.configuration_id for configuration in build_parameter_grid()}
    assert DEFAULT_COORDINATE_TOLERANCE_PIXELS == 1e-6
    assert output["manifest"]["thresholds"]["coordinate_tolerance_pixels"] == 1e-6
    assert all(
        row["coordinate_tolerance_pixels"] == 1e-6 for row in output["result_rows"]
    )


def test_parameter_grid_records_every_fixed_default_and_baseline(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())

    _run(dataset_root, output_root)
    rows = _read_csv(output_root / "parameter_grid.csv")
    baseline = next(row for row in rows if row["is_committed_baseline"] == "true")

    assert len(rows) == 9
    assert baseline["configuration_id"] == "e050_w064"
    assert baseline["removal_axis_x"] == "1"
    assert baseline["removal_axis_y"] == "0"
    assert baseline["projection_quantile"] == "0.95"
    assert baseline["support_band_width_pixels"] == "5"
    assert baseline["calyx_dilation_radius"] == "1"
    assert baseline["component_connectivity"] == "8"
    assert baseline["signed_offset_pixels"] == "0"
    assert baseline["candidate_step_pixels"] == "1"
    assert baseline["inward_search_margin_pixels"] == "10"
    assert baseline["outward_search_margin_pixels"] == "0"
    assert baseline["blade_band_half_width_pixels"] == "1"
    assert baseline["minimum_flesh_band_pixels"] == "1"
    assert baseline["minimum_calyx_band_pixels"] == "1"


def test_structured_v2b_failures_continue_and_aggregate_counts_match_rows(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _draped_mask())

    def selective_failure(mask: np.ndarray, **parameters: Any) -> Any:
        result = estimate_fixed_axis_search_cutline(mask, **parameters)
        if parameters["minimum_attachment_evidence_fraction"] == 0.70:
            return replace(
                result,
                status="failed",
                failure_code="synthetic_structured_failure",
                failure_reason="Synthetic structured failure for runner testing.",
                selected_cut_coordinate=None,
                candidate_cutline=None,
                final_cutline=None,
                diagnostic_reference_point=None,
            )
        return result

    output = _run(dataset_root, output_root, v2b_estimator=selective_failure)

    assert len(output["result_rows"]) == 9
    assert sum(row["v2b_status"] == "failed" for row in output["result_rows"]) == 3
    for configuration in output["configuration_rows"]:
        matching = [
            row
            for row in output["result_rows"]
            if row["configuration_id"] == configuration["configuration_id"]
        ]
        assert configuration["sample_count"] == len(matching)
        assert configuration["success_count"] == sum(
            row["v2b_status"] == "ok" for row in matching
        )
        assert configuration["structured_failure_count"] == sum(
            row["v2b_status"] != "ok" for row in matching
        )
    assert output["stability_rows"][0]["failure_count"] == 3
    assert any(
        row["flag_reason"] == "structured_failure" for row in output["flagged_rows"]
    )


def test_fragmentation_singleton_and_selected_not_largest_summaries() -> None:
    grid = build_parameter_grid()
    rows = []
    for configuration in grid:
        fragmented = configuration.configuration_id == "e030_w048"
        rows.append(
            _analysis_row(
                configuration.configuration_id,
                block_count=2 if fragmented else 1,
                feasible_count=4,
                singleton=fragmented,
                largest=not fragmented,
            )
        )
    summaries = _configuration_summary_rows(
        rows,
        grid,
        configuration_runtime_seconds={
            configuration.configuration_id: 0.0 for configuration in grid
        },
    )
    fragmented = summaries[0]

    assert fragmented["fragmented_set_count"] == 1
    assert fragmented["fragmented_set_rate"] == 1.0
    assert fragmented["singleton_selected_block_count"] == 1
    assert fragmented["singleton_selected_block_rate"] == 1.0
    assert fragmented["selected_block_not_largest_count"] == 1
    assert fragmented["selected_block_not_largest_rate"] == 1.0
    assert fragmented["median_selected_block_width_pixels"] == 0.0
    assert fragmented["median_feasible_hull_width_pixels"] == 4.0


def test_stability_mixed_success_failure_uses_only_successful_coordinates() -> None:
    rows = [
        _analysis_row("a", coordinate=2.0, v2a_coordinate=4.0),
        _analysis_row(
            "b",
            coordinate=100.0,
            status="failed",
            block_count=0,
            feasible_count=0,
            singleton=None,
            largest=None,
        ),
        _analysis_row("c", coordinate=6.0, v2a_coordinate=4.0),
    ]

    stability = _sample_stability_row(
        rows,
        coordinate_tolerance_pixels=DEFAULT_COORDINATE_TOLERANCE_PIXELS,
    )

    assert stability["configuration_count"] == 3
    assert stability["success_count"] == 2
    assert stability["failure_count"] == 1
    assert stability["coordinate_minimum"] == 2.0
    assert stability["coordinate_maximum"] == 6.0
    assert stability["coordinate_median"] == 4.0
    assert stability["coordinate_range"] == 4.0
    assert stability["coordinate_standard_deviation"] == 2.0
    assert stability["maximum_inward_shift_pixels"] == 2.0
    assert (
        stability["all_successful_configurations_same_coordinate_within_tolerance"]
        is False
    )
    assert stability["unique_structured_status_count"] == 2


def test_stability_no_success_is_explicitly_undefined() -> None:
    rows = [
        _analysis_row(
            "a",
            coordinate=None,
            status="failed",
            block_count=0,
            feasible_count=0,
            singleton=None,
            largest=None,
        )
    ]

    stability = _sample_stability_row(
        rows,
        coordinate_tolerance_pixels=DEFAULT_COORDINATE_TOLERANCE_PIXELS,
    )

    for field in (
        "coordinate_minimum",
        "coordinate_maximum",
        "coordinate_median",
        "coordinate_range",
        "coordinate_standard_deviation",
        "maximum_inward_shift_pixels",
        "all_successful_configurations_same_coordinate_within_tolerance",
    ):
        assert stability[field] is None


def test_default_tolerance_treats_submicro_pixel_range_as_equal() -> None:
    rows = [
        _analysis_row("a", coordinate=1000.0, v2a_coordinate=1000.0),
        _analysis_row("b", coordinate=1000.0000005, v2a_coordinate=1000.0),
    ]

    stability = _sample_stability_row(
        rows,
        coordinate_tolerance_pixels=DEFAULT_COORDINATE_TOLERANCE_PIXELS,
    )

    assert stability["coordinate_range"] == pytest.approx(5e-7)
    assert (
        stability["all_successful_configurations_same_coordinate_within_tolerance"]
        is True
    )


def test_baseline_relative_change_sign_is_current_minus_baseline() -> None:
    grid = build_parameter_grid()
    rows = []
    for configuration in grid:
        flesh = 0.5
        if configuration.configuration_id == "e030_w048":
            flesh = 0.25
        rows.append(
            _analysis_row(
                configuration.configuration_id,
                flesh_proxy=flesh,
            )
        )
    summaries = _configuration_summary_rows(
        rows,
        grid,
        configuration_runtime_seconds={
            configuration.configuration_id: 0.0 for configuration in grid
        },
    )
    current = summaries[0]
    baseline = next(row for row in summaries if row["is_committed_baseline"])

    assert current[
        "median_whole_mask_flesh_loss_ratio_change_from_baseline"
    ] == pytest.approx(-0.25)
    assert baseline["median_whole_mask_flesh_loss_ratio_change_from_baseline"] == 0.0
    assert set(CONFIGURATION_BASELINE_CHANGE_FIELDS) == {
        field.removesuffix("_change_from_baseline")
        for field in current
        if field.endswith("_change_from_baseline")
    }
    assert set(CONFIGURATION_BASELINE_CHANGE_FIELDS) == set(
        CONFIGURATION_AGGREGATE_FIELDS
    ) - {"aggregate_runtime_seconds"}
    assert "aggregate_runtime_seconds_change_from_baseline" not in current


def test_flagged_sample_reasons_are_separate_and_thresholds_are_strict() -> None:
    stability = {
        "sample_id": "sample",
        "split": "train",
        "failure_count": 1,
        "coordinate_range": 6.0,
        "singleton_selection_count": 1,
        "selected_not_largest_count": 1,
        "maximum_feasible_block_count": 6,
        "maximum_inward_shift_pixels": 11.0,
    }

    flags = _sample_flag_rows(
        stability,
        stability_threshold_pixels=5.0,
        maximum_inward_shift_threshold_pixels=10.0,
        feasible_block_count_flag_threshold=(
            DEFAULT_FEASIBLE_BLOCK_COUNT_FLAG_THRESHOLD
        ),
    )

    assert [row["flag_reason"] for row in flags] == [
        "structured_failure",
        "coordinate_range_exceeds_threshold",
        "singleton_selected_block",
        "selected_block_not_largest",
        "feasible_block_count_exceeds_threshold",
        "maximum_inward_shift_exceeds_threshold",
    ]
    boundary = dict(stability)
    boundary.update(
        failure_count=0,
        coordinate_range=5.0,
        singleton_selection_count=0,
        selected_not_largest_count=0,
        maximum_feasible_block_count=5,
        maximum_inward_shift_pixels=10.0,
    )
    assert not _sample_flag_rows(
        boundary,
        stability_threshold_pixels=5.0,
        maximum_inward_shift_threshold_pixels=10.0,
        feasible_block_count_flag_threshold=(
            DEFAULT_FEASIBLE_BLOCK_COUNT_FLAG_THRESHOLD
        ),
    )


def test_nondefault_feasible_block_flag_threshold_is_used_everywhere(
    tmp_path: Path,
) -> None:
    stability = {
        "sample_id": "sample",
        "split": "train",
        "failure_count": 0,
        "coordinate_range": 0.0,
        "singleton_selection_count": 0,
        "selected_not_largest_count": 0,
        "maximum_feasible_block_count": 4,
        "maximum_inward_shift_pixels": 0.0,
    }

    flags = _sample_flag_rows(
        stability,
        stability_threshold_pixels=5.0,
        maximum_inward_shift_threshold_pixels=10.0,
        feasible_block_count_flag_threshold=3,
    )

    assert flags == [
        {
            "sample_id": "sample",
            "split": "train",
            "flag_reason": "feasible_block_count_exceeds_threshold",
            "observed_value": 4,
            "comparison": ">",
            "threshold": 3,
        }
    ]
    assert not _sample_flag_rows(
        stability,
        stability_threshold_pixels=5.0,
        maximum_inward_shift_threshold_pixels=10.0,
        feasible_block_count_flag_threshold=4,
    )
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())
    output = _run(
        dataset_root,
        output_root,
        feasible_block_count_flag_threshold=3,
    )
    assert output["manifest"]["thresholds"]["feasible_block_count_flag_threshold"] == 3
    assert output["summary"]["thresholds"]["feasible_block_count_flag_threshold"] == 3


def test_metadata_is_sanitized_inputs_unchanged_and_no_data_path_is_used(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "synthetic-local-dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())
    before = _snapshot(dataset_root)

    _run(dataset_root, output_root)

    assert _snapshot(dataset_root) == before
    assert "data" not in dataset_root.relative_to(tmp_path).parts
    for filename in ("manifest.json", "summary.json"):
        serialized = (output_root / filename).read_text(encoding="utf-8")
        assert str(tmp_path.resolve()) not in serialized
        assert str(dataset_root.resolve()) not in serialized
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_basename"] == dataset_root.name
    assert all(
        not Path(path).is_absolute() for path in manifest["artifacts"]["plots"].values()
    )


def test_overwrite_protection_and_atomic_cleanup(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())
    _run(dataset_root, output_root)
    marker = output_root / "user-marker.txt"
    marker.write_text("preserve until successful replacement", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        _run(dataset_root, output_root)
    assert marker.is_file()

    def broken_plots(*arguments: Any, **keywords: Any) -> dict[str, str]:
        raise RuntimeError("synthetic plot failure")

    with pytest.raises(RuntimeError, match="synthetic plot failure"):
        _run(
            dataset_root,
            output_root,
            overwrite=True,
            plot_function=broken_plots,
        )
    assert marker.is_file()
    assert not list(tmp_path.glob(".output.staging-*"))

    _run(dataset_root, output_root, overwrite=True)
    assert not marker.exists()


def test_empty_split_and_image_mask_size_mismatch_are_rejected(tmp_path: Path) -> None:
    empty_root = tmp_path / "empty"
    _make_split(empty_root, "train")
    with pytest.raises(ValueError, match="No supported image/mask pairs"):
        _run(empty_root, tmp_path / "empty-output")

    mismatch_root = tmp_path / "mismatch"
    _write_pair(
        mismatch_root,
        "train",
        "sample",
        _horizontal_mask(),
        image_size=(21, 16),
    )
    with pytest.raises(ValueError, match="dimensions differ"):
        _run(mismatch_root, tmp_path / "mismatch-output")


def test_invalid_mask_class_mapping_and_mask_values_are_rejected(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    mask = _horizontal_mask()
    mask[0, 0] = 3
    _write_pair(dataset_root, "train", "sample", mask)

    with pytest.raises(ValueError, match="class_mapping"):
        _run(
            dataset_root,
            tmp_path / "mapping-output",
            class_mapping={"background": 0, "Flesh": 2, "Calyx": 1},
        )
    with pytest.raises(ValueError, match="only class IDs"):
        _run(dataset_root, tmp_path / "mask-output")


def test_proxy_heatmaps_use_adaptive_limits_precision_and_explicit_na(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    rows = []
    for index, configuration in enumerate(build_parameter_grid()):
        rows.append(
            {
                "configuration_id": configuration.configuration_id,
                "minimum_attachment_evidence_fraction": (
                    configuration.minimum_attachment_evidence_fraction
                ),
                "lateral_window_half_width_pixels": (
                    configuration.lateral_window_half_width_pixels
                ),
                "success_rate": 0.8 + index * 0.01,
                "median_whole_mask_flesh_loss_ratio": (
                    None if index == 8 else 0.12340 + index * 0.00001
                ),
                "median_whole_mask_calyx_retention_ratio": 0.7500,
            }
        )
    path = tmp_path / "aggregate_metric_heatmaps.png"
    original_close = sensitivity_plots.plt.close
    monkeypatch.setattr(sensitivity_plots.plt, "close", lambda figure: None)
    try:
        sensitivity_plots._plot_aggregate_metric_heatmaps(rows, path)
        figure = sensitivity_plots.plt.gcf()
        success_axis, flesh_axis, calyx_axis = figure.axes[:3]

        assert success_axis.images[0].norm.vmin == 0.0
        assert success_axis.images[0].norm.vmax == 1.0
        assert 0.12 < flesh_axis.images[0].norm.vmin < 0.13
        assert 0.12 < flesh_axis.images[0].norm.vmax < 0.13
        flesh_labels = [text.get_text() for text in flesh_axis.texts]
        assert "n/a" in flesh_labels
        finite_flesh_labels = [label for label in flesh_labels if label != "n/a"]
        assert len(set(finite_flesh_labels)) == len(finite_flesh_labels)
        assert all(len(label.partition(".")[2]) >= 4 for label in finite_flesh_labels)
        assert calyx_axis.images[0].norm.vmin < 0.75
        assert calyx_axis.images[0].norm.vmax > 0.75
        assert path.is_file()
    finally:
        original_close(sensitivity_plots.plt.gcf())

    matrix = np.asarray([[0.12340, 0.12341, np.nan]], dtype=np.float64)
    assert _finite_panel_limits(matrix) == pytest.approx((0.1233995, 0.1234105))
    assert _distinguishing_decimal_places(matrix, 4) >= 5
    equal_limits = _finite_panel_limits(np.full((2, 2), 0.25))
    assert equal_limits[0] < 0.25 < equal_limits[1]


def test_heatmap_annotation_contrast_handles_dark_light_and_missing_cells() -> None:
    figure, axis = sensitivity_plots.plt.subplots()
    try:
        _draw_heatmap(
            axis,
            np.asarray([[0.0, 1.0, np.nan]], dtype=np.float64),
            evidence_values=[0.5],
            width_values=[48.0, 64.0, 96.0],
            title="Contrast test",
            value_format=".4f",
            color_map="viridis",
            minimum=0.0,
            maximum=1.0,
        )

        assert [text.get_text() for text in axis.texts] == [
            "0.0000",
            "1.0000",
            "n/a",
        ]
        assert [text.get_color() for text in axis.texts] == [
            "white",
            "black",
            "black",
        ]
    finally:
        sensitivity_plots.plt.close(figure)


def test_configuration_tradeoff_caution_is_layout_safe_and_headless(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "configuration_tradeoffs.png"
    rows = [
        {
            "configuration_id": "e050_w064",
            "is_committed_baseline": True,
            "median_whole_mask_flesh_loss_ratio": 0.1234,
            "median_whole_mask_calyx_retention_ratio": 0.5678,
        }
    ]
    original_close = sensitivity_plots.plt.close
    monkeypatch.setattr(sensitivity_plots.plt, "close", lambda figure: None)
    try:
        _plot_configuration_tradeoffs(rows, path)
        figure = sensitivity_plots.plt.gcf()
        axis = figure.axes[0]

        assert figure.get_constrained_layout() is True
        assert "Geometric visible-mask proxies" in axis.get_xlabel()
        assert "physical outcome" in axis.get_xlabel()
        assert not any(
            "Geometric visible-mask proxies" in text.get_text() for text in figure.texts
        )
        assert path.is_file()
        with Image.open(path) as rendered:
            assert rendered.width >= 1000
            assert rendered.height >= 900
    finally:
        original_close(sensitivity_plots.plt.gcf())


def test_all_failed_sample_does_not_distort_coordinate_axis_and_legends_exist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    path = tmp_path / "coordinate_stability.png"
    rows = [
        {
            "sample_id": "successful-a",
            "coordinate_minimum": 1000.0,
            "coordinate_maximum": 1004.0,
            "coordinate_median": 1002.0,
            "committed_baseline_coordinate": 1001.0,
            "failure_count": 0,
        },
        {
            "sample_id": "successful-b",
            "coordinate_minimum": 1010.0,
            "coordinate_maximum": 1014.0,
            "coordinate_median": 1012.0,
            "committed_baseline_coordinate": 1011.0,
            "failure_count": 1,
        },
        {
            "sample_id": "all-failed",
            "coordinate_minimum": None,
            "coordinate_maximum": None,
            "coordinate_median": None,
            "committed_baseline_coordinate": None,
            "failure_count": 9,
        },
    ]
    original_close = sensitivity_plots.plt.close
    monkeypatch.setattr(sensitivity_plots.plt, "close", lambda figure: None)
    try:
        _plot_coordinate_stability(rows, path)
        figure = sensitivity_plots.plt.gcf()
        coordinate_axis, failure_axis = figure.axes
        minimum, maximum = coordinate_axis.get_xlim()

        assert minimum > 900.0
        assert maximum < 1100.0
        coordinate_labels = {
            text.get_text() for text in coordinate_axis.get_legend().get_texts()
        }
        assert "Successful coordinate range and median" in coordinate_labels
        assert "Committed baseline e050_w064" in coordinate_labels
        failure_labels = {
            text.get_text() for text in failure_axis.get_legend().get_texts()
        }
        assert failure_labels == {"Structured failures"}
        assert path.is_file()
    finally:
        original_close(sensitivity_plots.plt.gcf())


def test_headless_plots_are_generated_with_readable_dimensions(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _draped_mask())

    run_fixed_axis_sensitivity_dataset(
        dataset_root,
        output_root,
        split="train",
    )

    for filename in PLOT_FILENAMES.values():
        with Image.open(output_root / "plots" / filename) as plot:
            assert plot.width >= 900
            assert plot.height >= 600


def test_runner_module_import_does_not_require_torch(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['torch'] = None; "
                "import strawberry_occlusion.evaluation.fixed_axis_sensitivity "
                "as module; print(module.COMMITTED_BASELINE_CONFIGURATION_ID)"
            ),
        ],
        capture_output=True,
        text=True,
        cwd=Path(__file__).parents[1],
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "e050_w064"


def test_two_identical_synthetic_runs_have_stable_results_and_artifacts(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    _write_pair(dataset_root, "train", "sample", _draped_mask())
    clock = count()
    monkeypatch.setattr(sensitivity.time, "perf_counter", lambda: next(clock))

    first = _run(dataset_root, first_root)
    second = _run(dataset_root, second_root)

    assert first["result_rows"] == second["result_rows"]
    assert first["stability_rows"] == second["stability_rows"]
    assert first["flagged_rows"] == second["flagged_rows"]
    assert first["configuration_rows"] == second["configuration_rows"]
    assert _snapshot(first_root) == _snapshot(second_root)


def test_non_training_split_prints_and_records_explicit_warning(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    split = "val"
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, split, "sample", _horizontal_mask())

    output = _run(dataset_root, output_root, split=split)
    captured = capsys.readouterr()

    assert "WARNING" in captured.err
    assert "not TRAIN" in captured.err
    assert output["manifest"]["non_training_split"] is True
    assert "not TRAIN" in output["manifest"]["non_training_split_warning"]
    assert output["summary"]["non_training_split"] is True


def test_held_out_test_split_is_rejected_by_runner_and_cli(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_pair(dataset_root, "test", "sample", _horizontal_mask())

    with pytest.raises(ValueError, match="train, val"):
        _run(dataset_root, tmp_path / "output", split="test")

    parser = _build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "--dataset-root",
                "input",
                "--output-root",
                "output",
                "--split",
                "test",
            ]
        )
    assert "{train,val}" in " ".join(parser.format_help().split())
    assert "{train,val,test}" not in " ".join(parser.format_help().split())


def test_cli_requires_split_and_parses_repeated_or_comma_separated_values() -> None:
    parser = _build_argument_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(["--dataset-root", "input", "--output-root", "output"])

    arguments = parser.parse_args(
        [
            "--dataset-root",
            "input",
            "--output-root",
            "output",
            "--split",
            "train",
            "--minimum-attachment-evidence-fraction",
            "0.3,0.5",
            "--minimum-attachment-evidence-fraction",
            "0.7",
            "--lateral-window-half-width-pixels",
            "48,64,96",
        ]
    )
    assert _parse_repeated_float_values(
        arguments.minimum_attachment_evidence_fractions
    ) == (0.3, 0.5, 0.7)
    assert _parse_repeated_float_values(arguments.lateral_window_half_width_pixels) == (
        48.0,
        64.0,
        96.0,
    )


def test_optional_visualizations_are_explicit_and_compact(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())

    output = _run(
        dataset_root,
        output_root,
        visualization_configuration_ids=("e050_w064",),
    )

    paths = output["manifest"]["samples"][0]["visualization_paths"]
    assert paths == ["visualizations/e050_w064/sample.png"]
    assert (output_root / paths[0]).is_file()
    assert len(list((output_root / "visualizations").rglob("*.png"))) == 1


def test_summary_contains_definitions_but_no_selection_or_score(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())

    output = _run(dataset_root, output_root)
    summary = output["summary"]

    assert summary["automatic_configuration_selection"] is False
    assert summary["difference_convention"].startswith(
        "Aggregate change = current configuration - e050_w064"
    )
    assert (
        "negative Flesh-loss proxy change means a lower estimated"
        in summary["difference_convention"]
    )
    assert (
        "negative Calyx-retention proxy change means a lower estimated"
        in summary["difference_convention"]
    )
    assert (
        "Neither change is a confirmed physical improvement"
        in summary["difference_convention"]
    )
    assert (
        "population standard deviation"
        in summary["output_definitions"]["stability_definition"]
    )
    aggregate_definition = summary["output_definitions"]["aggregate_definition"]
    assert "Median and maximum feasible-block count use every sample row" in (
        aggregate_definition
    )
    assert "failed row with a recorded feasible-block count of zero" in (
        aggregate_definition
    )
    assert "Median feasible-hull width uses only rows with a finite hull width" in (
        aggregate_definition
    )
    assert "rows without a finite hull do not contribute" in aggregate_definition
    assert "not final generalization evidence" in summary["study_limitation"]
    assert any("physical" in caution for caution in summary["proxy_cautions"])
    assert sensitivity.WHOLE_MASK_PROXY_SCOPE in summary["proxy_cautions"]
    assert sensitivity.SELECTED_PAIR_PROXY_SCOPE in summary["proxy_cautions"]
    keys: list[str] = []

    def collect_keys(value: Any) -> None:
        if isinstance(value, dict):
            keys.extend(str(key).lower() for key in value)
            for nested in value.values():
                collect_keys(nested)
        elif isinstance(value, list):
            for nested in value:
                collect_keys(nested)

    collect_keys(summary)
    assert not any("best" in key or "score" in key for key in keys)
