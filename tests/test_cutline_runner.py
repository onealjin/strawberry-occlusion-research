import csv
import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from strawberry_occlusion.evaluation.cutline import (
    CSV_FIELDS,
    calculate_line_side_proxies,
    run_visible_mask_cutline_dataset,
)
from strawberry_occlusion.geometry import (
    LineSegment,
    Point,
    estimate_visible_mask_cutline,
)
from strawberry_occlusion.visualization.cutline import (
    create_mask_cutline_visualization,
)


def _success_mask() -> np.ndarray:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[4:10, 3:9] = 1
    mask[2:4, 5:7] = 2
    return mask


def _horizontal_mask() -> np.ndarray:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[3:9, 2:7] = 1
    mask[5:7, 7:9] = 2
    return mask


def _failure_mask() -> np.ndarray:
    mask = np.zeros((12, 12), dtype=np.uint8)
    mask[4:10, 3:9] = 1
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
    image = Image.new("RGB", image_size, color=(40, 80, 120))
    image_path = root / split / "images" / f"{sample_id}.png"
    mask_path = root / split / "masks" / f"{sample_id}.png"
    image.save(image_path)
    Image.fromarray(mask).save(mask_path)
    return image_path, mask_path


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def test_complete_successful_dataset_run_writes_geometry_and_visualization(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "synthetic-dataset"
    output_root = tmp_path / "run"
    _write_pair(dataset_root, "train", "vertical", _success_mask())
    _write_pair(dataset_root, "train", "horizontal", _horizontal_mask())

    output = run_visible_mask_cutline_dataset(
        dataset_root,
        output_root,
        split="train",
    )

    assert output["summary"]["total_sample_count"] == 2
    assert output["summary"]["success_count"] == 2
    assert output["summary"]["failure_count"] == 0
    assert output["manifest"]["sample_count"] == 2
    assert output["manifest"]["algorithm_version"] == "1"
    for artifact in ("manifest.json", "summary.json", "per_image_geometry.csv"):
        assert (output_root / artifact).is_file()
    assert sorted(path.name for path in (output_root / "visualizations").iterdir()) == [
        "horizontal.png",
        "vertical.png",
    ]


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
    output_root = tmp_path / "output"
    image_path, mask_path = _write_pair(
        dataset_root,
        "train",
        "sample",
        _success_mask(),
    )
    (image_path if missing_kind == "image" else mask_path).unlink()

    with pytest.raises(ValueError, match=message):
        run_visible_mask_cutline_dataset(dataset_root, output_root)

    assert not output_root.exists()


def test_image_mask_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(
        dataset_root,
        "train",
        "sample",
        _success_mask(),
        image_size=(13, 12),
    )

    with pytest.raises(ValueError, match="dimensions differ"):
        run_visible_mask_cutline_dataset(dataset_root, output_root)

    assert not output_root.exists()


def test_invalid_mask_values_are_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    mask = _success_mask()
    mask[0, 0] = 3
    _write_pair(dataset_root, "train", "sample", mask)

    with pytest.raises(ValueError, match="only class IDs 0, 1, and 2"):
        run_visible_mask_cutline_dataset(dataset_root, output_root)

    assert not output_root.exists()


def test_multichannel_mask_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _, mask_path = _write_pair(
        dataset_root,
        "train",
        "sample",
        _success_mask(),
    )
    Image.new("RGB", (12, 12), color=(0, 1, 2)).save(mask_path)

    with pytest.raises(ValueError, match="single-channel"):
        run_visible_mask_cutline_dataset(dataset_root, output_root)

    assert not output_root.exists()


def test_mixed_success_and_structured_failure_continue_and_summarize(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "a-success", _success_mask())
    _write_pair(dataset_root, "train", "b-no-calyx", _failure_mask())

    output = run_visible_mask_cutline_dataset(dataset_root, output_root)

    summary = output["summary"]
    assert summary["total_sample_count"] == 2
    assert summary["success_count"] == 1
    assert summary["failure_count"] == 1
    assert summary["failure_counts_by_reason"] == {"no_calyx": 1}
    assert "does not classify" in summary["status_definition"]
    rows = _read_rows(output_root / "per_image_geometry.csv")
    assert [row["sample_id"] for row in rows] == ["a-success", "b-no-calyx"]
    assert rows[0]["status"] == "ok"
    assert rows[1]["status"] == "failed"
    assert "no Calyx" in rows[1]["failure_reason"]
    assert rows[1]["final_start_x"] == ""
    assert rows[1]["final_offset_anchor_x"] == ""
    assert rows[1]["final_offset_anchor_y"] == ""
    assert (output_root / "visualizations/a-success.png").is_file()
    assert (output_root / "visualizations/b-no-calyx.png").is_file()


def test_csv_geometry_and_contact_diagnostics_preserve_source_coordinates(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _success_mask())

    run_visible_mask_cutline_dataset(dataset_root, output_root)

    row = _read_rows(output_root / "per_image_geometry.csv")[0]
    assert tuple(row) == CSV_FIELDS
    assert row["source_width"] == "12"
    assert row["source_height"] == "12"
    assert row["flesh_centroid_x"] == "5.5"
    assert row["flesh_centroid_y"] == "6.5"
    assert row["attachment_anchor_x"] == "5.5"
    assert row["attachment_anchor_y"] == "4"
    assert row["final_offset_anchor_x"] == "5.5"
    assert row["final_offset_anchor_y"] == "4"
    assert row["candidate_start_x"] == "0"
    assert row["candidate_end_x"] == "11"
    assert row["candidate_angle_degrees"] == "0"
    assert row["contact_bbox_width"] == "4"
    assert row["contact_bbox_height"] == "1"
    assert row["contact_spatial_extent_pixels"] == "3"
    assert row["anchor_to_nearest_flesh_boundary_pixels"] == "0.5"
    assert row["anchor_to_flesh_centroid_pixels"] == "2.5"
    assert row["candidate_final_lines_coincide"] == "true"


def test_line_side_proxy_sign_convention_and_ratios() -> None:
    mask = _horizontal_mask()
    mask[0, 3] = 2
    result = estimate_visible_mask_cutline(mask)
    custom_result = replace(
        result,
        fruit_to_attachment_direction=(1.0, 0.0),
        offset_anchor=Point(4.0, 5.5),
        final_cutline=LineSegment(
            start=Point(4.0, 0.0),
            end=Point(4.0, 11.0),
        ),
    )

    proxies = calculate_line_side_proxies(mask, custom_result)

    assert proxies["flesh_loss_proxy_pixel_count"] == 12
    assert proxies["flesh_loss_proxy_ratio"] == pytest.approx(0.4)
    assert proxies["calyx_retention_proxy_pixel_count"] == 1
    assert proxies["calyx_retention_proxy_ratio"] == pytest.approx(0.2)


def test_summary_contains_requested_aggregate_statistics(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _success_mask())

    run_visible_mask_cutline_dataset(dataset_root, output_root)

    summary = json.loads((output_root / "summary.json").read_text(encoding="utf-8"))
    assert summary["aggregate_contact_size_statistics"]["sample_count"] == 1
    assert summary["aggregate_contact_size_statistics"]["mean"] == 4.0
    assert summary["aggregate_contact_component_statistics"]["mean"] == 1.0
    assert summary["aggregate_flesh_loss_proxy_statistics"]["ratio"]["mean"] == 0.0
    assert summary["aggregate_calyx_retention_proxy_statistics"]["ratio"]["mean"] == 0.0
    assert "all Flesh and Calyx pixels" in summary["proxy_definition"]
    assert "Detached fragments" in summary["proxy_definition"]
    assert "> 0 is the Calyx/removal side" in summary["line_side_sign_convention"]
    assert (
        "complete Flesh-mask"
        in summary["diagnostic_definitions"]["anchor_to_nearest_flesh_boundary_pixels"]
    )
    assert "square root" in summary["diagnostic_definitions"]["normalized_anchor_depth"]
    assert summary["parameter_configuration"] == {
        "calyx_dilation_radius": 1,
        "component_connectivity": 8,
        "signed_offset_pixels": 0.0,
    }


def test_overwrite_refusal_and_explicit_overwrite(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _success_mask())
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        run_visible_mask_cutline_dataset(dataset_root, output_root)
    assert marker.read_text(encoding="utf-8") == "keep"

    run_visible_mask_cutline_dataset(
        dataset_root,
        output_root,
        overwrite=True,
    )
    assert not marker.exists()
    assert (output_root / "manifest.json").is_file()


def test_deterministic_sanitized_output_and_unchanged_sources(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "private-looking-dataset-name"
    first_output = tmp_path / "first"
    second_output = tmp_path / "second"
    _write_pair(dataset_root, "train", "sample", _success_mask())
    before = _snapshot(dataset_root)

    run_visible_mask_cutline_dataset(dataset_root, first_output)
    run_visible_mask_cutline_dataset(dataset_root, second_output)

    assert _snapshot(dataset_root) == before
    first_files = _snapshot(first_output)
    second_files = _snapshot(second_output)
    assert first_files == second_files
    for metadata_name in ("manifest.json", "summary.json"):
        serialized = (first_output / metadata_name).read_text(encoding="utf-8")
        assert str(tmp_path.resolve()) not in serialized
        assert str(dataset_root.resolve()) not in serialized
    manifest = json.loads((first_output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["dataset_basename"] == dataset_root.name
    assert manifest["artifacts"] == {
        "per_image_geometry": "per_image_geometry.csv",
        "summary": "summary.json",
        "visualizations": "visualizations",
    }
    assert manifest["samples"][0]["visualization_path"] == ("visualizations/sample.png")


def test_train_and_val_split_selection(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "val-output"
    _write_pair(dataset_root, "train", "train-sample", _success_mask())
    _write_pair(dataset_root, "val", "val-sample", _horizontal_mask())

    output = run_visible_mask_cutline_dataset(
        dataset_root,
        output_root,
        split="val",
    )

    assert output["summary"]["evaluated_split"] == "val"
    assert [sample["sample_id"] for sample in output["manifest"]["samples"]] == [
        "val-sample"
    ]
    assert not (output_root / "visualizations/train-sample.png").exists()


def test_estimator_and_visualization_are_injectable(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    _write_pair(dataset_root, "train", "sample", _success_mask())
    estimator_calls: list[dict[str, Any]] = []
    visualization_calls: list[dict[str, Any]] = []

    def injected_estimator(mask: np.ndarray, **parameters: Any) -> Any:
        estimator_calls.append(parameters)
        return estimate_visible_mask_cutline(mask, **parameters)

    def injected_visualization(
        mask: np.ndarray,
        result: Any,
        **arguments: Any,
    ) -> Image.Image:
        visualization_calls.append(arguments)
        return create_mask_cutline_visualization(mask, result, **arguments)

    run_visible_mask_cutline_dataset(
        dataset_root,
        output_root,
        signed_offset=1.0,
        estimator=injected_estimator,
        visualization_function=injected_visualization,
    )

    assert estimator_calls == [
        {
            "calyx_dilation_radius": 1,
            "component_connectivity": 8,
            "signed_offset": 1.0,
        }
    ]
    assert len(visualization_calls) == 1
    assert set(visualization_calls[0]) == {"image", "diagnostics"}
