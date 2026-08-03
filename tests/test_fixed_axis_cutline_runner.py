import csv
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from strawberry_occlusion.evaluation.cutline import calculate_line_side_proxies
from strawberry_occlusion.evaluation.fixed_axis_cutline import (
    CSV_FIELDS,
    CHANGE_CONVENTION,
    V1_ALGORITHM_NAME,
    V1_ALGORITHM_VERSION,
    V2A_ALGORITHM_NAME,
    V2A_ALGORITHM_VERSION,
    V2B_ALGORITHM_NAME,
    V2B_ALGORITHM_VERSION,
    _build_argument_parser,
    _tradeoff_counts,
    calculate_fixed_axis_line_side_proxies,
    calculate_fixed_axis_search_line_side_proxies,
    run_fixed_axis_cutline_comparison_dataset,
)
from strawberry_occlusion.geometry import (
    LineSegment,
    Point,
    calculate_fixed_axis_candidate_curve,
    estimate_fixed_axis_cutline,
    estimate_fixed_axis_search_cutline,
    estimate_visible_mask_cutline,
)
from strawberry_occlusion.visualization.fixed_axis_comparison import (
    V1_ANCHOR_COLOR,
    V1_LINE_COLOR,
    V2A_AXIS_COLOR,
    V2A_CANDIDATE_COLOR,
    V2A_FINAL_COLOR,
    V2A_ATTACHMENT_REFERENCE_COLOR,
    V2A_FLESH_REFERENCE_COLOR,
    V2A_REFERENCE_COLOR,
    V2B_LINE_COLOR,
    V2B_REFERENCE_COLOR,
    create_fixed_axis_comparison_visualization,
)


def _vertical_mask() -> np.ndarray:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[4:10, 3:9] = 1
    mask[2:4, 5:7] = 2
    return mask


def _horizontal_mask() -> np.ndarray:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 2:7] = 1
    mask[5:7, 7:9] = 2
    return mask


def _draped_mask() -> np.ndarray:
    mask = np.zeros((24, 24), dtype=np.uint8)
    mask[4:20, 3:17] = 1
    mask[8:11, 8:19] = 2
    return mask


def _degenerate_v1_mask() -> np.ndarray:
    mask = np.zeros((5, 5), dtype=np.uint8)
    mask[2, 2] = 1
    mask[1, 2] = 2
    return mask


def _offset_mask(*, outer: bool) -> np.ndarray:
    mask = np.zeros((12, 12), dtype=np.uint8)
    if outer:
        mask[3:9, 4:10] = 1
        mask[5:7, 10:12] = 2
    else:
        mask[3:9, 1:6] = 1
        mask[5:7, 6:8] = 2
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
) -> tuple[Path, Path]:
    _make_split(root, split)
    height, width = mask.shape[:2]
    if image_size is None:
        image_size = (width, height)
    image_path = root / split / "images" / f"{sample_id}.png"
    mask_path = root / split / "masks" / f"{sample_id}.png"
    Image.new("RGB", image_size, color=(40, 80, 120)).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return image_path, mask_path


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _run(
    dataset_root: Path,
    output_root: Path,
    **arguments: Any,
) -> dict[str, Any]:
    return run_fixed_axis_cutline_comparison_dataset(
        dataset_root,
        output_root,
        removal_axis=arguments.pop("removal_axis", (1.0, 0.0)),
        **arguments,
    )


def test_successful_multisample_train_run_is_sorted_and_complete(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "synthetic-dataset"
    output_root = tmp_path / "comparison"
    _write_pair(dataset_root, "train", "zeta", _horizontal_mask())
    _write_pair(dataset_root, "train", "alpha", _draped_mask())

    output = _run(dataset_root, output_root)

    assert output["summary"]["total_sample_count"] == 2
    assert output["summary"]["successful_v1_count"] == 2
    assert output["summary"]["successful_v2a_count"] == 2
    assert output["summary"]["successful_v2b_count"] == 2
    assert [row["sample_id"] for row in output["rows"]] == ["alpha", "zeta"]
    assert [sample["sample_id"] for sample in output["manifest"]["samples"]] == [
        "alpha",
        "zeta",
    ]
    for artifact in ("manifest.json", "summary.json", "per_image_comparison.csv"):
        assert (output_root / artifact).is_file()
    assert sorted(path.name for path in (output_root / "visualizations").iterdir()) == [
        "alpha.png",
        "zeta.png",
    ]
    assert sorted(
        path.name for path in (output_root / "candidate_curves").iterdir()
    ) == [
        "alpha.csv",
        "zeta.csv",
    ]
    rows = _read_rows(output_root / "per_image_comparison.csv")
    assert len(rows) == 2
    assert tuple(rows[0]) == CSV_FIELDS
    assert all(row["v1_status"] == "ok" for row in rows)
    assert all(row["v2a_status"] == "ok" for row in rows)
    assert all(row["v2b_status"] == "ok" for row in rows)


def test_manifest_records_algorithms_configuration_and_relative_artifacts(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())

    _run(dataset_root, output_root, removal_axis=(2.0, 0.0))
    manifest = json.loads((output_root / "manifest.json").read_text(encoding="utf-8"))

    assert manifest["v1_algorithm_name"] == V1_ALGORITHM_NAME
    assert manifest["v1_algorithm_version"] == V1_ALGORITHM_VERSION
    assert manifest["v2a_algorithm_name"] == V2A_ALGORITHM_NAME
    assert manifest["v2a_algorithm_version"] == V2A_ALGORITHM_VERSION
    assert manifest["v2b_algorithm_name"] == V2B_ALGORITHM_NAME
    assert manifest["v2b_algorithm_version"] == V2B_ALGORITHM_VERSION
    assert manifest["class_mapping"] == {"background": 0, "Flesh": 1, "Calyx": 2}
    assert manifest["parameter_configuration"]["removal_axis"] == [2.0, 0.0]
    assert manifest["parameter_configuration"]["normalized_removal_axis"] == [
        1.0,
        0.0,
    ]
    assert manifest["parameter_configuration"]["v1_signed_offset_pixels"] == 0.0
    assert manifest["parameter_configuration"]["outward_search_margin_pixels"] == 0.0
    assert manifest["artifacts"] == {
        "manifest": "manifest.json",
        "per_image_comparison": "per_image_comparison.csv",
        "summary": "summary.json",
        "visualizations": "visualizations",
        "candidate_curves": "candidate_curves",
    }
    assert manifest["samples"][0]["visualization_path"] == ("visualizations/sample.png")
    assert manifest["samples"][0]["candidate_curve_path"] == (
        "candidate_curves/sample.csv"
    )


def test_train_and_synthetic_val_split_selection(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_pair(dataset_root, "train", "train-sample", _horizontal_mask())
    _write_pair(dataset_root, "val", "val-sample", _vertical_mask())

    train_output = _run(dataset_root, tmp_path / "train-output", split="train")
    val_output = _run(
        dataset_root,
        tmp_path / "val-output",
        split="val",
        removal_axis=(0.0, -1.0),
    )

    assert [row["sample_id"] for row in train_output["rows"]] == ["train-sample"]
    assert [row["sample_id"] for row in val_output["rows"]] == ["val-sample"]
    assert train_output["summary"]["evaluated_split"] == "train"
    assert val_output["summary"]["evaluated_split"] == "val"


@pytest.mark.parametrize(
    ("missing_kind", "message"),
    (("image", "Missing images"), ("mask", "Missing masks")),
)
def test_missing_image_or_mask_is_rejected(
    tmp_path: Path,
    missing_kind: str,
    message: str,
) -> None:
    dataset_root = tmp_path / "dataset"
    image_path, mask_path = _write_pair(
        dataset_root,
        "train",
        "sample",
        _horizontal_mask(),
    )
    (image_path if missing_kind == "image" else mask_path).unlink()
    output_root = tmp_path / "output"

    with pytest.raises(ValueError, match=message):
        _run(dataset_root, output_root)

    assert not output_root.exists()


def test_duplicate_stems_are_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    image_path, _ = _write_pair(
        dataset_root,
        "train",
        "sample",
        _horizontal_mask(),
    )
    Image.open(image_path).save(image_path.with_suffix(".jpg"))

    with pytest.raises(ValueError, match="Duplicate image stem"):
        _run(dataset_root, tmp_path / "output")


def test_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _write_pair(
        dataset_root,
        "train",
        "sample",
        _horizontal_mask(),
        image_size=(13, 12),
    )

    with pytest.raises(ValueError, match="dimensions differ"):
        _run(dataset_root, tmp_path / "output")


def test_invalid_mask_classes_are_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    mask = _horizontal_mask()
    mask[0, 0] = 3
    _write_pair(dataset_root, "train", "sample", mask)

    with pytest.raises(ValueError, match="only class IDs 0, 1, and 2"):
        _run(dataset_root, tmp_path / "output")


def test_multichannel_mask_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _, mask_path = _write_pair(
        dataset_root,
        "train",
        "sample",
        _horizontal_mask(),
    )
    Image.new("RGB", (12, 12), color=(0, 1, 2)).save(mask_path)

    with pytest.raises(ValueError, match="single-channel"):
        _run(dataset_root, tmp_path / "output")


def test_empty_split_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _make_split(dataset_root, "train")

    with pytest.raises(ValueError, match="No supported image/mask pairs"):
        _run(dataset_root, tmp_path / "output")


def test_v1_structured_failure_continues_while_v2a_succeeds(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "a-degenerate-v1", _degenerate_v1_mask())
    _write_pair(dataset_root, "train", "b-success", _vertical_mask())

    output = _run(
        dataset_root,
        output_root,
        removal_axis=(0.0, -1.0),
    )

    assert len(output["rows"]) == 2
    assert output["rows"][0]["v1_failure_code"] == "degenerate_direction"
    assert output["rows"][0]["v2a_status"] == "ok"
    assert output["summary"]["successful_v1_count"] == 1
    assert output["summary"]["successful_v2a_count"] == 2
    assert output["summary"]["failure_counts_by_method_and_reason"]["v1"] == {
        "degenerate_direction": 1
    }
    assert (output_root / "visualizations/a-degenerate-v1.png").is_file()


def test_v2a_structured_failure_continues_while_v1_succeeds(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "a-in-bounds", _offset_mask(outer=False))
    _write_pair(dataset_root, "train", "b-outside", _offset_mask(outer=True))

    output = _run(
        dataset_root,
        output_root,
        signed_offset_pixels=4.0,
    )

    assert len(output["rows"]) == 2
    assert all(row["v1_status"] == "ok" for row in output["rows"])
    assert output["rows"][0]["v2a_status"] == "ok"
    assert output["rows"][1]["v2a_failure_code"] == "offset_line_outside_image"
    assert output["summary"]["successful_v2a_count"] == 1
    assert output["summary"]["failure_counts_by_method_and_reason"]["v2a"] == {
        "offset_line_outside_image": 1
    }
    assert (output_root / "visualizations/b-outside.png").is_file()


def test_fixed_axis_line_side_proxy_sign_and_on_line_retention() -> None:
    mask = _horizontal_mask()
    mask[0, 3] = 2
    result = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    custom = replace(
        result,
        normalized_removal_axis=(1.0, 0.0),
        diagnostic_reference_point=Point(4.0, 5.5),
        final_cut_coordinate=4.0,
        final_cutline=LineSegment(Point(4.0, 0.0), Point(4.0, 11.0)),
    )

    proxies = calculate_fixed_axis_line_side_proxies(mask, custom)

    assert proxies["flesh_loss_proxy_pixel_count"] == 12
    assert proxies["flesh_loss_proxy_ratio"] == pytest.approx(0.4)
    assert proxies["calyx_retention_proxy_pixel_count"] == 1
    assert proxies["calyx_retention_proxy_ratio"] == pytest.approx(0.2)


def test_v2b_whole_mask_proxy_sign_and_on_line_retention() -> None:
    mask = _horizontal_mask()
    mask[0, 3] = 2
    result = estimate_fixed_axis_search_cutline(mask, removal_axis=(1.0, 0.0))
    custom = replace(
        result,
        normalized_removal_axis=(1.0, 0.0),
        diagnostic_reference_point=Point(4.0, 5.5),
        selected_cut_coordinate=4.0,
        final_cutline=LineSegment(Point(4.0, 0.0), Point(4.0, 11.0)),
    )

    proxies = calculate_fixed_axis_search_line_side_proxies(mask, custom)

    assert proxies["flesh_loss_proxy_pixel_count"] == 12
    assert proxies["flesh_loss_proxy_ratio"] == pytest.approx(0.4)
    assert proxies["calyx_retention_proxy_pixel_count"] == 1
    assert proxies["calyx_retention_proxy_ratio"] == pytest.approx(0.2)


def test_candidate_curve_csv_matches_selected_candidate(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _draped_mask())

    output = _run(dataset_root, output_root)
    candidate_rows = _read_rows(output_root / "candidate_curves/sample.csv")
    selected_index = output["rows"][0]["v2b_selected_candidate_index"]

    assert candidate_rows
    assert tuple(candidate_rows[0]) == (
        "coordinate",
        "local_contact_count",
        "local_flesh_count",
        "local_calyx_count",
        "local_v2a_support_count",
        "selected_flesh_loss_ratio",
        "selected_calyx_retention_ratio",
        "retained_contact_ratio",
        "feasible",
        "feasible_block_id",
    )
    assert candidate_rows[selected_index]["feasible"] == "true"
    assert candidate_rows[selected_index]["feasible_block_id"] == str(
        output["rows"][0]["v2b_selected_block_id"]
    )
    assert all(
        (row["feasible_block_id"] != "") == (row["feasible"] == "true")
        for row in candidate_rows
    )
    assert float(candidate_rows[selected_index]["coordinate"]) == pytest.approx(
        output["rows"][0]["v2b_cut_coordinate"]
    )


def test_fragmented_feasible_set_csv_and_summary_diagnostics(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _draped_mask())

    def fragmented_v2b(mask: np.ndarray, **parameters: Any) -> Any:
        def candidate_curve(v2a_result: Any, **curve_parameters: Any) -> Any:
            table = calculate_fixed_axis_candidate_curve(
                v2a_result,
                **curve_parameters,
            )
            feasible = np.zeros(len(table), dtype=bool)
            feasible[[1, 2, 3, 5]] = True
            feasible.setflags(write=False)
            return replace(table, feasible=feasible)

        return estimate_fixed_axis_search_cutline(
            mask,
            **parameters,
            candidate_curve_function=candidate_curve,
        )

    output = _run(
        dataset_root,
        output_root,
        v2b_estimator=fragmented_v2b,
    )
    row = output["rows"][0]
    comparison_row = _read_rows(output_root / "per_image_comparison.csv")[0]
    candidate_rows = _read_rows(output_root / "candidate_curves/sample.csv")
    diagnostics = output["summary"]["feasible_set_diagnostics"]

    assert row["v2b_feasible_candidate_count"] == 4
    assert row["v2b_feasible_block_count"] == 2
    assert row["v2b_feasible_hull_start"] == pytest.approx(
        float(candidate_rows[1]["coordinate"])
    )
    assert row["v2b_feasible_hull_end"] == pytest.approx(
        float(candidate_rows[5]["coordinate"])
    )
    assert row["v2b_selected_block_id"] == 1
    assert row["v2b_selected_block_start"] == row["v2b_selected_block_end"]
    assert row["v2b_selected_block_candidate_count"] == 1
    assert row["v2b_selected_block_is_singleton"] is True
    assert comparison_row["v2b_feasible_candidate_count"] == "4"
    assert comparison_row["v2b_feasible_block_count"] == "2"
    assert comparison_row["v2b_selected_block_is_singleton"] == "true"
    assert [candidate_rows[index]["feasible_block_id"] for index in (1, 2, 3)] == [
        "0",
        "0",
        "0",
    ]
    assert candidate_rows[4]["feasible_block_id"] == ""
    assert candidate_rows[5]["feasible_block_id"] == "1"
    assert diagnostics["fragmented_sample_count"] == 1
    assert diagnostics["selected_block_not_largest_sample_count"] == 1
    assert diagnostics["singleton_selected_block_sample_count"] == 1
    assert diagnostics["feasible_block_count_statistics"] == {
        "sample_count": 1,
        "minimum": 2.0,
        "maximum": 2.0,
        "mean": 2.0,
        "median": 2.0,
    }
    assert diagnostics["selected_block_width_pixels_statistics"] == {
        "sample_count": 1,
        "minimum": 0.0,
        "maximum": 0.0,
        "mean": 0.0,
        "median": 0.0,
    }


def test_v2b_structured_failure_continues_and_writes_candidate_curve(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "a-fail", _horizontal_mask())
    _write_pair(dataset_root, "train", "b-success", _draped_mask())

    def selective_v2b(mask: np.ndarray, **parameters: Any) -> Any:
        if mask.shape == (12, 12):
            parameters["minimum_calyx_band_pixels"] = 100
        return estimate_fixed_axis_search_cutline(mask, **parameters)

    output = _run(dataset_root, output_root, v2b_estimator=selective_v2b)

    assert len(output["rows"]) == 2
    assert output["rows"][0]["v2b_failure_code"] == ("no_feasible_fixed_axis_cut")
    assert output["rows"][1]["v2b_status"] == "ok"
    assert output["summary"]["successful_v2b_count"] == 1
    assert (output_root / "candidate_curves/a-fail.csv").is_file()
    assert (output_root / "visualizations/a-fail.png").is_file()


def test_csv_proxies_differences_and_coordinate_shift_use_v2a_minus_v1(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    mask = _draped_mask()
    _write_pair(dataset_root, "train", "sample", mask)

    output = _run(dataset_root, output_root)
    row = output["rows"][0]
    v1_result = estimate_visible_mask_cutline(mask)
    v2a_result = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    v2b_result = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        v2a_result=v2a_result,
    )
    expected_v1 = calculate_line_side_proxies(mask, v1_result)
    expected_v2a = calculate_fixed_axis_line_side_proxies(mask, v2a_result)
    expected_v2b = calculate_fixed_axis_search_line_side_proxies(mask, v2b_result)

    assert row["v1_flesh_loss_pixels"] == expected_v1["flesh_loss_proxy_pixel_count"]
    assert row["v2a_flesh_loss_pixels"] == expected_v2a["flesh_loss_proxy_pixel_count"]
    assert row["flesh_loss_pixel_change"] == (
        row["v2a_flesh_loss_pixels"] - row["v1_flesh_loss_pixels"]
    )
    assert row["flesh_loss_ratio_change"] == pytest.approx(
        row["v2a_flesh_loss_ratio"] - row["v1_flesh_loss_ratio"]
    )
    assert row["calyx_retention_pixel_change"] == (
        row["v2a_calyx_retention_pixels"] - row["v1_calyx_retention_pixels"]
    )
    assert row["calyx_retention_ratio_change"] == pytest.approx(
        row["v2a_calyx_retention_ratio"] - row["v1_calyx_retention_ratio"]
    )
    assert row["cut_coordinate_shift_from_v1"] == pytest.approx(
        row["v2a_final_cut_coordinate"] - row["v1_anchor_projection_on_removal_axis"]
    )
    assert "v2a minus v1" in CHANGE_CONVENTION
    assert (
        row["v2b_whole_mask_flesh_loss_pixels"]
        == expected_v2b["flesh_loss_proxy_pixel_count"]
    )
    assert row["v2b_minus_v1_flesh_loss_pixel_change"] == (
        row["v2b_whole_mask_flesh_loss_pixels"] - row["v1_flesh_loss_pixels"]
    )
    assert row["v2b_minus_v2a_calyx_retention_ratio_change"] == pytest.approx(
        row["v2b_whole_mask_calyx_retention_ratio"] - row["v2a_calyx_retention_ratio"]
    )
    assert row["v2b_coordinate_shift_from_v1"] == pytest.approx(
        row["v2b_cut_coordinate"] - row["v1_anchor_projection_on_removal_axis"]
    )
    assert row["v2b_coordinate_shift_from_v2a"] == pytest.approx(
        row["v2b_cut_coordinate"] - row["v2a_final_cut_coordinate"]
    )


def test_summary_contains_aggregates_definitions_and_comparison_counts(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "a", _horizontal_mask())
    _write_pair(dataset_root, "train", "b", _draped_mask())

    _run(dataset_root, output_root)
    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))

    assert summary["total_sample_count"] == 2
    expected_aggregates = {
        "contact_pixel_count",
        "cut_coordinate_shift_from_v1",
        "support_pixel_count",
        "v1_calyx_retention_ratio",
        "v1_flesh_loss_ratio",
        "v2a_axis_disagreement_degrees",
        "v2a_calyx_retention_ratio",
        "v2a_cut_margin_to_flesh_extent",
        "v2a_flesh_loss_ratio",
        "v2b_whole_mask_flesh_loss_ratio",
        "v2b_whole_mask_calyx_retention_ratio",
        "v2b_coordinate_shift_from_v1",
        "v2b_coordinate_shift_from_v2a",
        "v2b_cut_margin_to_flesh_extent",
        "v2b_selected_local_contact_count",
    }
    assert set(summary["aggregate_statistics"]) == expected_aggregates
    assert summary["aggregate_statistics"]["contact_pixel_count"]["sample_count"] == 2
    assert set(summary["comparison_counts"]) == {
        "both_proxies_improve",
        "one_improves_while_other_worsens",
        "v2a_lowers_calyx_retention",
        "v2a_lowers_flesh_loss",
        "v2a_raises_calyx_retention",
        "v2a_raises_flesh_loss",
        "v2b_vs_v1_lowers_flesh_loss",
        "v2b_vs_v1_raises_flesh_loss",
        "v2b_vs_v1_lowers_calyx_retention",
        "v2b_vs_v1_raises_calyx_retention",
        "v2b_vs_v1_both_proxies_improve",
        "v2b_vs_v1_one_improves_while_other_worsens",
        "v2b_vs_v2a_lowers_flesh_loss",
        "v2b_vs_v2a_raises_flesh_loss",
        "v2b_vs_v2a_lowers_calyx_retention",
        "v2b_vs_v2a_raises_calyx_retention",
        "v2b_vs_v2a_both_proxies_improve",
        "v2b_vs_v2a_one_improves_while_other_worsens",
    }
    assert "all Flesh and Calyx pixels" in summary["proxy_definition"]
    assert "> 0 is the Calyx/removal side" in summary["v1_line_side_sign_convention"]
    assert "> 0 is the Calyx/removal side" in summary["v2a_line_side_sign_convention"]
    assert "> 0 is the Calyx/removal side" in summary["v2b_line_side_sign_convention"]
    assert "all Flesh and Calyx pixels" in summary["whole_mask_proxy_scope"]
    assert "selected Flesh/Calyx component pair" in summary["selected_pair_proxy_scope"]
    assert "maximal run" in summary["feasible_block_definition"]
    assert "does not imply" in summary["feasible_hull_definition"]
    assert "outermost-feasible" in summary["selected_block_definition"]
    assert "not proof of reliable cutting" in summary["feasible_set_caution"]
    assert set(summary["feasible_set_diagnostics"]) == {
        "feasible_block_count_statistics",
        "fragmented_sample_count",
        "selected_block_not_largest_sample_count",
        "selected_block_width_pixels_statistics",
        "singleton_selected_block_sample_count",
    }
    assert (
        summary["feasible_set_diagnostics"]["feasible_block_count_statistics"][
            "sample_count"
        ]
        == 2
    )
    assert summary["parameter_configuration"]["outward_search_margin_pixels"] == 0.0
    assert "not prove" in summary["status_definitions"]["v1"]
    assert "not prove" in summary["status_definitions"]["v2a"]
    assert "not prove" in summary["status_definitions"]["v2b"]


def test_tradeoff_count_signs() -> None:
    rows: list[dict[str, float | None]] = [
        {"flesh_loss_ratio_change": -0.1, "calyx_retention_ratio_change": -0.2},
        {"flesh_loss_ratio_change": -0.1, "calyx_retention_ratio_change": 0.2},
        {"flesh_loss_ratio_change": 0.1, "calyx_retention_ratio_change": -0.2},
        {"flesh_loss_ratio_change": 0.1, "calyx_retention_ratio_change": 0.2},
        {"flesh_loss_ratio_change": None, "calyx_retention_ratio_change": None},
    ]
    for row in rows:
        row["v2b_minus_v1_flesh_loss_ratio_change"] = row["flesh_loss_ratio_change"]
        row["v2b_minus_v1_calyx_retention_ratio_change"] = row[
            "calyx_retention_ratio_change"
        ]
        row["v2b_minus_v2a_flesh_loss_ratio_change"] = (
            -row["flesh_loss_ratio_change"]
            if row["flesh_loss_ratio_change"] is not None
            else None
        )
        row["v2b_minus_v2a_calyx_retention_ratio_change"] = (
            -row["calyx_retention_ratio_change"]
            if row["calyx_retention_ratio_change"] is not None
            else None
        )

    assert _tradeoff_counts(rows) == {
        "v2a_lowers_flesh_loss": 2,
        "v2a_raises_flesh_loss": 2,
        "v2a_lowers_calyx_retention": 2,
        "v2a_raises_calyx_retention": 2,
        "both_proxies_improve": 1,
        "one_improves_while_other_worsens": 2,
        "v2b_vs_v1_lowers_flesh_loss": 2,
        "v2b_vs_v1_raises_flesh_loss": 2,
        "v2b_vs_v1_lowers_calyx_retention": 2,
        "v2b_vs_v1_raises_calyx_retention": 2,
        "v2b_vs_v1_both_proxies_improve": 1,
        "v2b_vs_v1_one_improves_while_other_worsens": 2,
        "v2b_vs_v2a_lowers_flesh_loss": 2,
        "v2b_vs_v2a_raises_flesh_loss": 2,
        "v2b_vs_v2a_lowers_calyx_retention": 2,
        "v2b_vs_v2a_raises_calyx_retention": 2,
        "v2b_vs_v2a_both_proxies_improve": 1,
        "v2b_vs_v2a_one_improves_while_other_worsens": 2,
    }


def test_source_coordinates_are_preserved_in_csv(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())

    _run(dataset_root, output_root)
    row = _read_rows(output_root / "per_image_comparison.csv")[0]

    assert row["source_width"] == "12"
    assert row["source_height"] == "12"
    assert row["v1_attachment_anchor_x"] == "6"
    assert row["v2a_final_cut_coordinate"] == "6"
    assert row["v2a_final_start_x"] == "6"
    assert row["v2a_final_end_x"] == "6"


def test_metadata_is_sanitized_deterministic_and_sources_are_unchanged(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "synthetic-local-dataset"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    _write_pair(dataset_root, "train", "sample", _draped_mask())
    before = _snapshot(dataset_root)

    _run(dataset_root, first_output)
    _run(dataset_root, second_output)

    assert _snapshot(dataset_root) == before
    assert _snapshot(first_output) == _snapshot(second_output)
    for metadata_name in ("manifest.json", "summary.json"):
        serialized = (first_output / metadata_name).read_text(encoding="utf-8")
        assert str(tmp_path.resolve()) not in serialized
        assert str(dataset_root.resolve()) not in serialized
        assert "base64" not in serialized.lower()
    manifest = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_basename"] == dataset_root.name
    assert all(
        not Path(sample["visualization_path"]).is_absolute()
        for sample in manifest["samples"]
    )
    assert all(
        not Path(sample["candidate_curve_path"]).is_absolute()
        for sample in manifest["samples"]
    )


def test_visualization_is_lossless_deterministic_and_has_distinct_geometry(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _draped_mask())

    _run(dataset_root, output_root)
    visualization_path = output_root / "visualizations/sample.png"
    with Image.open(visualization_path) as visualization:
        assert visualization.format == "PNG"
        assert visualization.mode == "RGB"
        rendered = np.asarray(visualization)
    rendered_colors = {
        tuple(color) for color in np.unique(rendered.reshape(-1, 3), axis=0)
    }
    for color in (
        V1_LINE_COLOR,
        V1_ANCHOR_COLOR,
        V2A_AXIS_COLOR,
        V2A_CANDIDATE_COLOR,
        V2A_FINAL_COLOR,
        V2A_REFERENCE_COLOR,
        V2A_FLESH_REFERENCE_COLOR,
        V2A_ATTACHMENT_REFERENCE_COLOR,
        V2B_LINE_COLOR,
        V2B_REFERENCE_COLOR,
    ):
        assert color in rendered_colors


def test_zero_offset_visualization_states_candidate_and_final_coincide() -> None:
    mask = _draped_mask()
    v1_result = estimate_visible_mask_cutline(mask)
    v2a_result = estimate_fixed_axis_cutline(mask, removal_axis=(1.0, 0.0))
    v2b_result = estimate_fixed_axis_search_cutline(
        mask,
        removal_axis=(1.0, 0.0),
        v2a_result=v2a_result,
    )
    image = Image.new("RGB", (24, 24), color=(40, 80, 120))

    first = create_fixed_axis_comparison_visualization(
        mask,
        v1_result,
        v2a_result,
        v2b_result,
        image=image,
        metrics={},
    )
    second = create_fixed_axis_comparison_visualization(
        mask,
        v1_result,
        v2a_result,
        v2b_result,
        image=image,
        metrics={},
    )

    assert first.tobytes() == second.tobytes()
    assert v2a_result.candidate_final_lines_coincide is True


def test_overwrite_refusal_and_explicit_overwrite(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        _run(dataset_root, output_root)
    assert marker.read_text(encoding="utf-8") == "keep"

    _run(dataset_root, output_root, overwrite=True)
    assert not marker.exists()
    assert (output_root / "manifest.json").is_file()


def test_staging_is_cleaned_after_generation_failure(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())

    def failing_visualization(*args: Any, **kwargs: Any) -> Image.Image:
        raise RuntimeError("synthetic visualization failure")

    with pytest.raises(RuntimeError, match="synthetic visualization failure"):
        _run(
            dataset_root,
            output_root,
            visualization_function=failing_visualization,
        )

    assert not output_root.exists()
    assert not list(tmp_path.glob(".output.staging-*"))


def test_estimators_and_visualization_are_injectable(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _horizontal_mask())
    v1_calls: list[dict[str, Any]] = []
    v2a_calls: list[dict[str, Any]] = []
    v2b_calls: list[dict[str, Any]] = []
    visualization_calls: list[dict[str, Any]] = []

    def v1_estimator(mask: np.ndarray, **parameters: Any) -> Any:
        v1_calls.append(parameters)
        return estimate_visible_mask_cutline(mask, **parameters)

    def v2a_estimator(mask: np.ndarray, **parameters: Any) -> Any:
        v2a_calls.append(parameters)
        return estimate_fixed_axis_cutline(mask, **parameters)

    def v2b_estimator(mask: np.ndarray, **parameters: Any) -> Any:
        v2b_calls.append(parameters)
        return estimate_fixed_axis_search_cutline(mask, **parameters)

    def visualization(
        mask: np.ndarray,
        v1_result: Any,
        v2a_result: Any,
        v2b_result: Any,
        **arguments: Any,
    ) -> Image.Image:
        visualization_calls.append(arguments)
        return create_fixed_axis_comparison_visualization(
            mask,
            v1_result,
            v2a_result,
            v2b_result,
            **arguments,
        )

    _run(
        dataset_root,
        output_root,
        projection_quantile=0.8,
        support_band_width_pixels=3.0,
        signed_offset_pixels=1.0,
        v1_estimator=v1_estimator,
        v2a_estimator=v2a_estimator,
        v2b_estimator=v2b_estimator,
        visualization_function=visualization,
    )

    assert v1_calls == [
        {
            "calyx_dilation_radius": 1,
            "component_connectivity": 8,
            "signed_offset": 0.0,
        }
    ]
    assert v2a_calls == [
        {
            "removal_axis": (1.0, 0.0),
            "projection_quantile": 0.8,
            "support_band_width_pixels": 3.0,
            "calyx_dilation_radius": 1,
            "component_connectivity": 8,
            "signed_offset_pixels": 1.0,
        }
    ]
    assert len(v2b_calls) == 1
    assert v2b_calls[0]["v2a_result"].final_cut_coordinate == 7.0
    assert v2b_calls[0]["candidate_step_pixels"] == 1.0
    assert v2b_calls[0]["outward_search_margin_pixels"] == 0.0
    assert v2b_calls[0]["minimum_attachment_evidence_fraction"] == 0.5
    assert len(visualization_calls) == 1
    assert set(visualization_calls[0]) == {"image", "metrics"}


def test_cli_requires_axis_and_uses_requested_defaults() -> None:
    parser = _build_argument_parser()
    arguments = parser.parse_args(
        (
            "--dataset-root",
            "dataset",
            "--output-root",
            "output",
            "--removal-axis-x",
            "1",
            "--removal-axis-y",
            "0",
        )
    )

    assert arguments.split == "train"
    assert arguments.projection_quantile == 0.95
    assert arguments.support_band_width_pixels == 5.0
    assert arguments.calyx_dilation_radius == 1
    assert arguments.component_connectivity == 8
    assert arguments.signed_offset_pixels == 0.0
    assert arguments.candidate_step_pixels == 1.0
    assert arguments.inward_search_margin_pixels == 10.0
    assert arguments.outward_search_margin_pixels == 0.0
    assert arguments.blade_band_half_width_pixels == 1.0
    assert arguments.lateral_window_half_width_pixels == 64.0
    assert arguments.minimum_attachment_evidence_fraction == 0.5
    assert arguments.minimum_flesh_band_pixels == 1
    assert arguments.minimum_calyx_band_pixels == 1
    normalized_help = " ".join(parser.format_help().split())
    assert "explicit positive values remain research overrides" in normalized_help

    overridden = parser.parse_args(
        (
            "--dataset-root",
            "dataset",
            "--output-root",
            "output",
            "--removal-axis-x",
            "1",
            "--removal-axis-y",
            "0",
            "--outward-search-margin-pixels",
            "3.5",
        )
    )
    assert overridden.outward_search_margin_pixels == 3.5


def test_runner_module_import_does_not_require_torch() -> None:
    completed = subprocess.run(
        (
            sys.executable,
            "-c",
            (
                "import sys; sys.modules['torch'] = None; "
                "import strawberry_occlusion.evaluation.fixed_axis_cutline"
            ),
        ),
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
