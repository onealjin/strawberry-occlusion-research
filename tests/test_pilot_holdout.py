import base64
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import strawberry_occlusion.evaluation.pilot_holdout as pilot_holdout_module
from strawberry_occlusion.data.vt_export import CLASS_MAPPING, convert_vt_export
from strawberry_occlusion.evaluation.pilot_holdout import (
    CUTLINE_FIELDS,
    DEFAULT_EXPECTED_SAMPLE_COUNT,
    FIXED_AXIS_ELIGIBILITY_STATUSES,
    FROZEN_FIXED_AXIS_CONFIGURATION,
    METADATA_FIELDS,
    PILOT_LIMITATIONS,
    SEGMENTATION_FIELDS,
    SUPPORTED_ORIENTATION_CATEGORIES,
    PilotOverlapError,
    _build_argument_parser,
    _cutline_comparison_rows,
    _parse_category_map_arguments,
    _run_frozen_geometry,
    audit_pilot_overlap,
    coordinate_agreement_summary,
    evaluate_pilot_holdout,
    fixed_axis_eligibility,
    prepare_pilot_holdout_dataset,
    read_pilot_metadata,
    write_pilot_metadata_template,
)
from strawberry_occlusion.geometry import (
    estimate_fixed_axis_cutline,
    estimate_fixed_axis_search_cutline,
)
from strawberry_occlusion.visualization.pilot_holdout import (
    INELIGIBLE_MESSAGE,
    create_pilot_holdout_visualization,
)


WIDTH = 32
HEIGHT = 24
CATEGORIES = [
    {"id": 17, "name": "Flesh", "color": "#880000"},
    {"id": 29, "name": "Calyx", "color": "#008800"},
]


class TinyPilotModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(12.0))
        self.was_training_during_forward: bool | None = None
        self.grad_enabled_during_forward: bool | None = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.was_training_during_forward = self.training
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        return image * self.scale


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _encode_mask(mask: np.ndarray) -> str:
    packed = np.packbits(mask.astype(np.uint8).reshape(-1), bitorder="big")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _semantic_mask(*, shift: int = 0, include_calyx: bool = True) -> np.ndarray:
    mask = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    mask[5:19, 3 + shift : 21 + shift] = 1
    if include_calyx:
        mask[10:14, 20 + shift : 28 + shift] = 2
    return mask


def _annotations(image_name: str, *, shift: int = 0) -> list[dict[str, Any]]:
    flesh = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    calyx = np.zeros((HEIGHT, WIDTH), dtype=np.uint8)
    flesh[5:19, 3 + shift : 23 + shift] = 1
    calyx[10:14, 20 + shift : 28 + shift] = 1
    return [
        {
            "category_id": 17,
            "labelname": "Flesh",
            "bbox": [0, 0, WIDTH, HEIGHT],
            "pixel_image": _encode_mask(flesh),
            "image_name": image_name,
            "type": "pencil",
        },
        {
            "category_id": 29,
            "labelname": "Calyx",
            "bbox": [0, 0, WIDTH, HEIGHT],
            "pixel_image": _encode_mask(calyx),
            "image_name": image_name,
            "type": "pencil",
        },
    ]


def _rgb_for_mask(mask: np.ndarray, *, token: int = 0) -> np.ndarray:
    image = np.full((*mask.shape, 3), 3, dtype=np.uint8)
    for class_id in range(3):
        image[..., class_id][mask == class_id] = 245
    image[0, token % WIDTH, :] = np.asarray([20 + token, 40, 60], dtype=np.uint8)
    return image


def _make_raw_export(
    root: Path,
    *,
    count: int = DEFAULT_EXPECTED_SAMPLE_COUNT,
    id_prefix: str = "pilot",
) -> Path:
    root.mkdir(parents=True)
    images = []
    for index in range(count):
        sample_id = f"{id_prefix}-{index}"
        image_name = f"sample-{index}.jpg"
        images.append(
            {
                "id": sample_id,
                "file_name": image_name,
                "width": WIDTH,
                "height": HEIGHT,
                "image_type": "val",
                "labeled": True,
            }
        )
        mask = _semantic_mask()
        Image.fromarray(_rgb_for_mask(mask, token=index)).save(
            root / image_name,
            quality=100,
            subsampling=0,
        )
        _write_json(root / f"sample-{index}.json", _annotations(image_name))
    _write_json(root / "dataset.json", {"categories": CATEGORIES, "images": images})
    return root


def _make_extracted_export(
    root: Path,
    *,
    count: int = DEFAULT_EXPECTED_SAMPLE_COUNT,
    id_prefix: str = "pilot",
) -> Path:
    _make_raw_export(root, count=count, id_prefix=id_prefix)
    (root / "dataset.json").unlink()
    return root


def _make_png_extracted_export(
    root: Path,
    *,
    count: int = DEFAULT_EXPECTED_SAMPLE_COUNT,
) -> Path:
    _make_extracted_export(root, count=count)
    for jpeg_path in sorted(root.glob("*.jpg")):
        png_path = jpeg_path.with_suffix(".png")
        with Image.open(jpeg_path) as image:
            image.save(png_path, format="PNG")
        jpeg_path.unlink()

        annotation_path = jpeg_path.with_suffix(".json")
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        for annotation in annotations:
            annotation["image_name"] = png_path.name
        _write_json(annotation_path, annotations)
    return root


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _prepare(tmp_path: Path, *, name: str = "pilot") -> tuple[Path, Path]:
    raw = _make_raw_export(tmp_path / f"{name}-raw")
    normalized = tmp_path / f"{name}-normalized"
    prepare_pilot_holdout_dataset(raw, normalized)
    return raw, normalized


def _write_checkpoint(
    path: Path,
    model: nn.Module | None = None,
    *,
    completed_epoch: int = 92,
) -> nn.Module:
    model = model or TinyPilotModel()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {"must_not_be_used": True},
            "completed_epoch": completed_epoch,
            "best_mean_foreground_iou": 0.75,
            "class_mapping": dict(CLASS_MAPPING),
            "input_height": HEIGHT,
            "input_width": WIDTH,
            "seed": 42,
            "model_name": model.__class__.__name__,
        },
        path,
    )
    return model


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as input_file:
        return list(csv.DictReader(input_file))


def _edit_metadata(path: Path, eligible_ids: set[str]) -> None:
    rows = _read_csv(path)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=METADATA_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            row["orientation_category"] = "right"
            row["fixed_axis_eligible"] = (
                "true" if row["sample_id"] in eligible_ids else "false"
            )
            row["annotation_qa_status"] = "approved"
            writer.writerow(row)


def _evaluate(
    tmp_path: Path,
    *,
    output_name: str = "evaluation",
    eligible_count: int = 2,
) -> tuple[Path, Path, Path, Path, dict[str, Any], TinyPilotModel]:
    _, normalized = _prepare(tmp_path)
    metadata = tmp_path / "pilot_metadata.csv"
    write_pilot_metadata_template(normalized, metadata)
    eligible = {f"pilot-{index}" for index in range(eligible_count)}
    _edit_metadata(metadata, eligible)
    checkpoint = tmp_path / "checkpoint" / "best_checkpoint.pt"
    restored = TinyPilotModel()
    _write_checkpoint(checkpoint)
    output = tmp_path / output_name
    result = evaluate_pilot_holdout(
        normalized,
        checkpoint,
        output,
        metadata,
        device="cpu",
        model_factory=lambda: restored,
    )
    return normalized, metadata, checkpoint, output, result, restored


def _make_reference(
    root: Path,
    *,
    sample_id: str,
    image: Image.Image | np.ndarray,
    mask: np.ndarray | None = None,
) -> Path:
    image_root = root / "train" / "images"
    mask_root = root / "train" / "masks"
    image_root.mkdir(parents=True)
    mask_root.mkdir(parents=True)
    image_path = image_root / "reference.jpg"
    if isinstance(image, Image.Image):
        image.save(image_path, format="JPEG", quality=100, subsampling=0)
    else:
        Image.fromarray(image).save(image_path, quality=100, subsampling=0)
    Image.fromarray(mask if mask is not None else _semantic_mask()).save(
        mask_root / "reference.png"
    )
    _write_json(
        root / "manifest.json",
        {
            "samples": [
                {
                    "sample_id": sample_id,
                    "image_path": "train/images/reference.jpg",
                    "mask_path": "train/masks/reference.png",
                }
            ]
        },
    )
    return root


def test_prepare_seven_valid_pairs_reuses_converter_mapping_and_preserves_sources(
    tmp_path: Path,
) -> None:
    raw = _make_raw_export(tmp_path / "raw")
    before = _snapshot(raw)
    normalized = tmp_path / "normalized"
    calls = []

    def converter(*args: Any, **kwargs: Any) -> dict[str, Any]:
        calls.append((args, kwargs))
        return convert_vt_export(*args, **kwargs)

    output = prepare_pilot_holdout_dataset(
        raw,
        normalized,
        input_format="full-export",
        converter=converter,
    )

    assert len(calls) == 1
    assert before == _snapshot(raw)
    assert output["summary"]["pilot_sample_count"] == 7
    assert output["manifest"]["class_mapping"] == CLASS_MAPPING
    assert output["manifest"]["overlap_rule"] == "Calyx overrides Flesh"
    assert output["manifest"]["dataset_kind"] == "pilot_holdout"
    assert output["manifest"]["is_v1_or_v2_dataset"] is False
    assert output["manifest"]["source_input_format"] == "full-export"
    assert {sample["sample_id"] for sample in output["manifest"]["samples"]} == {
        f"pilot-{index}" for index in range(7)
    }
    assert len(list((normalized / "pilot" / "images").glob("*.jpg"))) == 7
    assert len(list((normalized / "pilot" / "masks").glob("*.png"))) == 7
    with Image.open(normalized / "pilot" / "masks" / "sample-0.png") as mask_file:
        mask = np.asarray(mask_file)
    assert mask[11, 21] == CLASS_MAPPING["Calyx"]


def test_prepare_seven_valid_extracted_pairs_without_dataset_manifest(
    tmp_path: Path,
) -> None:
    raw = _make_extracted_export(tmp_path / "extracted")
    before = _snapshot(raw)
    normalized = tmp_path / "normalized"

    output = prepare_pilot_holdout_dataset(raw, normalized)

    assert output["manifest"]["source_input_format"] == "extracted-pairs"
    assert output["summary"]["pilot_sample_count"] == 7
    assert {sample["sample_id"] for sample in output["manifest"]["samples"]} == {
        f"sample-{index}" for index in range(7)
    }
    assert output["manifest"]["class_mapping"] == CLASS_MAPPING
    assert output["manifest"]["overlap_rule"] == "Calyx overrides Flesh"
    assert not (raw / "dataset.json").exists()
    assert not (normalized / "dataset.json").exists()
    assert before == _snapshot(raw)


def test_prepare_seven_exact_png_json_pairs_without_dataset_manifest(
    tmp_path: Path,
) -> None:
    raw = _make_png_extracted_export(tmp_path / "extracted")
    before = _snapshot(raw)
    normalized = tmp_path / "normalized"

    output = prepare_pilot_holdout_dataset(raw, normalized)

    assert output["manifest"]["source_input_format"] == "extracted-pairs"
    assert {sample["sample_id"] for sample in output["manifest"]["samples"]} == {
        f"sample-{index}" for index in range(7)
    }
    normalized_images = normalized / "pilot" / "images"
    assert len(list(normalized_images.glob("*.png"))) == 7
    for source_image in sorted(raw.glob("*.png")):
        assert (
            normalized_images / source_image.name
        ).read_bytes() == source_image.read_bytes()
    assert before == _snapshot(raw)


def test_extracted_pairs_ignore_only_harmless_platform_files(tmp_path: Path) -> None:
    raw = _make_png_extracted_export(tmp_path / "extracted")
    for filename in ("desktop.ini", ".DS_Store", "Thumbs.db"):
        (raw / filename).write_text("platform metadata", encoding="utf-8")
    before = _snapshot(raw)

    output = prepare_pilot_holdout_dataset(
        raw,
        tmp_path / "normalized",
        input_format="extracted-pairs",
    )

    assert output["summary"]["pilot_sample_count"] == 7
    assert before == _snapshot(raw)


def test_extracted_pairs_reject_arbitrary_unknown_file(tmp_path: Path) -> None:
    raw = _make_png_extracted_export(tmp_path / "extracted")
    (raw / "notes.txt").write_text("unexpected", encoding="utf-8")

    with pytest.raises(ValueError, match="Unexpected unrelated file.*notes.txt"):
        prepare_pilot_holdout_dataset(
            raw,
            tmp_path / "normalized",
            input_format="extracted-pairs",
        )


def test_extracted_pairs_require_exact_case_sensitive_basename(tmp_path: Path) -> None:
    raw = _make_png_extracted_export(tmp_path / "extracted")
    (raw / "sample-0.json").rename(raw / "Sample-0.json")

    with pytest.raises(ValueError, match="Missing extracted annotation JSON"):
        prepare_pilot_holdout_dataset(
            raw,
            tmp_path / "normalized",
            input_format="extracted-pairs",
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_annotation", "Missing extracted annotation JSON"),
        ("missing_image", "Missing extracted image"),
        ("duplicate_stem", "Duplicate sample stem"),
        ("ambiguous_pairing", "Ambiguous extracted pairing"),
        ("wrong_count", "Expected exactly 7 extracted"),
        ("unexpected_json", "unrelated unexpected JSON"),
    ),
)
def test_extracted_pair_validation_failures(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    count = 6 if mutation == "wrong_count" else 7
    raw = _make_extracted_export(tmp_path / "extracted", count=count)
    if mutation == "missing_annotation":
        (raw / "sample-0.json").unlink()
    elif mutation == "missing_image":
        (raw / "sample-0.jpg").unlink()
    elif mutation == "duplicate_stem":
        (raw / "sample-0.jpeg").write_bytes((raw / "sample-0.jpg").read_bytes())
    elif mutation == "ambiguous_pairing":
        annotations = json.loads((raw / "sample-0.json").read_text(encoding="utf-8"))
        for annotation in annotations:
            annotation["image_name"] = "sample-1.jpg"
        _write_json(raw / "sample-0.json", annotations)
    elif mutation == "unexpected_json":
        _write_json(raw / "notes.json", {"not": "an annotation"})

    output = tmp_path / "normalized"
    with pytest.raises(ValueError, match=message):
        prepare_pilot_holdout_dataset(
            raw,
            output,
            input_format="extracted-pairs",
        )
    assert not output.exists()
    assert not list(tmp_path.glob(".normalized.staging-*"))


def test_extracted_auto_detection_is_unambiguous(tmp_path: Path) -> None:
    raw = _make_extracted_export(tmp_path / "extracted")
    output = prepare_pilot_holdout_dataset(raw, tmp_path / "normalized")

    assert output["manifest"]["source_input_format"] == "extracted-pairs"


def test_extracted_mode_reuses_construct_mask_and_explicit_category_mapping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    raw = _make_extracted_export(tmp_path / "extracted")
    for annotation_path in sorted(raw.glob("*.json")):
        annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
        for annotation in annotations:
            annotation.pop("labelname")
        _write_json(annotation_path, annotations)

    with pytest.raises(ValueError, match="--category-map"):
        prepare_pilot_holdout_dataset(
            raw,
            tmp_path / "missing-mapping",
            input_format="extracted-pairs",
        )

    original_construct_mask = pilot_holdout_module.construct_mask
    calls = []

    def recording_construct_mask(*args: Any, **kwargs: Any) -> Any:
        calls.append((args, kwargs))
        return original_construct_mask(*args, **kwargs)

    monkeypatch.setattr(
        pilot_holdout_module,
        "construct_mask",
        recording_construct_mask,
    )
    output = prepare_pilot_holdout_dataset(
        raw,
        tmp_path / "normalized",
        input_format="extracted-pairs",
        category_mapping={17: "Flesh", 29: "Calyx"},
    )

    assert len(calls) == 7
    assert output["manifest"]["class_mapping"] == CLASS_MAPPING
    with Image.open(
        tmp_path / "normalized" / "pilot" / "masks" / "sample-0.png"
    ) as mask_file:
        assert np.asarray(mask_file)[11, 21] == CLASS_MAPPING["Calyx"]


def test_full_export_and_extracted_pairs_normalize_equivalently(
    tmp_path: Path,
) -> None:
    full_raw = _make_raw_export(tmp_path / "full", id_prefix="sample")
    extracted_raw = _make_extracted_export(
        tmp_path / "extracted",
        id_prefix="sample",
    )
    full_output = tmp_path / "full-normalized"
    extracted_output = tmp_path / "extracted-normalized"

    full = prepare_pilot_holdout_dataset(
        full_raw,
        full_output,
        input_format="full-export",
    )
    extracted = prepare_pilot_holdout_dataset(
        extracted_raw,
        extracted_output,
        input_format="extracted-pairs",
    )

    assert _snapshot(full_output / "pilot") == _snapshot(extracted_output / "pilot")
    assert full["manifest"]["samples"] == extracted["manifest"]["samples"]


def test_extracted_pilot_output_retains_overlap_audit(tmp_path: Path) -> None:
    raw = _make_extracted_export(tmp_path / "extracted")
    normalized = tmp_path / "normalized"
    prepare_pilot_holdout_dataset(
        raw,
        normalized,
        input_format="extracted-pairs",
    )
    reference = _make_reference(
        tmp_path / "reference",
        sample_id="sample-0",
        image=np.full((HEIGHT, WIDTH, 3), 99, dtype=np.uint8),
    )

    with pytest.raises(PilotOverlapError):
        audit_pilot_overlap(normalized, reference, hash_images=False)


@pytest.mark.parametrize(
    ("mutation", "message"),
    (
        ("missing_annotation", "Missing annotation"),
        ("missing_image", "Missing image"),
        ("unexpected_extra", "Unexpected extra"),
        ("wrong_count", "Expected exactly 7"),
        ("duplicate_pair", "Duplicate dataset image"),
        ("shape_mismatch", "do not match dataset.json"),
    ),
)
def test_prepare_rejects_invalid_raw_batches(
    tmp_path: Path,
    mutation: str,
    message: str,
) -> None:
    raw = _make_raw_export(
        tmp_path / "raw", count=6 if mutation == "wrong_count" else 7
    )
    if mutation == "missing_annotation":
        (raw / "sample-0.json").unlink()
    elif mutation == "missing_image":
        (raw / "sample-0.jpg").unlink()
    elif mutation == "unexpected_extra":
        (raw / "extra.txt").write_text("unexpected", encoding="utf-8")
    elif mutation == "duplicate_pair":
        dataset = json.loads((raw / "dataset.json").read_text(encoding="utf-8"))
        duplicate = dict(dataset["images"][0])
        duplicate["id"] = "duplicate-id"
        dataset["images"].append(duplicate)
        _write_json(raw / "dataset.json", dataset)
    elif mutation == "shape_mismatch":
        dataset = json.loads((raw / "dataset.json").read_text(encoding="utf-8"))
        dataset["images"][0]["width"] = WIDTH - 1
        _write_json(raw / "dataset.json", dataset)

    output = tmp_path / "normalized"
    with pytest.raises((ValueError, FileNotFoundError), match=message):
        prepare_pilot_holdout_dataset(raw, output)
    assert not output.exists()
    assert not list(tmp_path.glob(".normalized.staging-*"))


def test_prepare_is_overwrite_safe_and_atomic_on_converter_failure(
    tmp_path: Path,
) -> None:
    raw = _make_raw_export(tmp_path / "raw")
    output = tmp_path / "normalized"
    output.mkdir()
    marker = output / "keep.txt"
    marker.write_text("keep", encoding="utf-8")
    with pytest.raises(FileExistsError, match="--overwrite"):
        prepare_pilot_holdout_dataset(raw, output)
    assert marker.read_text(encoding="utf-8") == "keep"

    output.rmdir() if not any(output.iterdir()) else None
    marker.unlink()
    output.rmdir()

    def failing_converter(*args: Any, **kwargs: Any) -> dict[str, Any]:
        convert_vt_export(*args, **kwargs)
        raise RuntimeError("synthetic conversion failure")

    with pytest.raises(RuntimeError, match="synthetic conversion failure"):
        prepare_pilot_holdout_dataset(raw, output, converter=failing_converter)
    assert not output.exists()
    assert not list(tmp_path.glob(".normalized.staging-*"))


def test_sample_id_overlap_hard_fails_and_override_is_prominent(tmp_path: Path) -> None:
    _, pilot = _prepare(tmp_path)
    reference = _make_reference(
        tmp_path / "reference",
        sample_id="pilot-0",
        image=np.full((HEIGHT, WIDTH, 3), 100, dtype=np.uint8),
    )
    with pytest.raises(PilotOverlapError) as error:
        audit_pilot_overlap(pilot, reference, hash_images=False)
    assert error.value.audit["matching_sample_ids"]
    assert error.value.audit["hard_fail_reasons"] == ["matching_sample_id"]

    audit = audit_pilot_overlap(
        pilot,
        reference,
        hash_images=False,
        allow_overlap=True,
    )
    assert audit["overlap_override_used"] is True
    assert audit["overlap_override_warning"].startswith("EXPLICIT RESEARCH OVERRIDE")


def test_image_hash_overlap_detected_with_different_ids(tmp_path: Path) -> None:
    _, pilot = _prepare(tmp_path)
    source_image = Image.open(pilot / "pilot" / "images" / "sample-0.jpg")
    try:
        reference = _make_reference(
            tmp_path / "reference",
            sample_id="different-reference-id",
            image=source_image,
        )
    finally:
        source_image.close()
    # Re-saving a JPEG is not byte-identical, so copy the prepared image exactly.
    (reference / "train" / "images" / "reference.jpg").write_bytes(
        (pilot / "pilot" / "images" / "sample-0.jpg").read_bytes()
    )

    with pytest.raises(PilotOverlapError) as error:
        audit_pilot_overlap(pilot, reference)
    assert error.value.audit["identical_image_hashes"]
    assert not error.value.audit["matching_sample_ids"]


def test_no_overlap_audit_succeeds_and_reports_counts(tmp_path: Path) -> None:
    _, pilot = _prepare(tmp_path)
    reference = _make_reference(
        tmp_path / "reference",
        sample_id="reference-only",
        image=np.full((HEIGHT, WIDTH, 3), 77, dtype=np.uint8),
        mask=np.zeros((HEIGHT, WIDTH), dtype=np.uint8),
    )
    audit = audit_pilot_overlap(pilot, reference)
    assert audit["pilot_sample_count"] == 7
    assert audit["v1_reference_count"] == 1
    assert audit["hard_fail_triggered"] is False
    assert audit["matching_sample_ids"] == []
    assert audit["identical_image_hashes"] == []


def test_metadata_template_contains_every_sample_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    _, pilot = _prepare(tmp_path)
    metadata = tmp_path / "metadata.csv"
    write_pilot_metadata_template(pilot, metadata)
    rows = _read_csv(metadata)
    assert tuple(rows[0]) == METADATA_FIELDS
    assert len(rows) == 7
    assert {row["sample_id"] for row in rows} == {f"pilot-{i}" for i in range(7)}
    assert all(row["orientation_category"] == "unknown" for row in rows)
    assert all(row["fixed_axis_eligible"] == "" for row in rows)
    with pytest.raises(FileExistsError, match="--overwrite"):
        write_pilot_metadata_template(pilot, metadata)


def test_fixed_axis_eligibility_is_explicit_and_unknown_orientation_is_excluded() -> (
    None
):
    assert FIXED_AXIS_ELIGIBILITY_STATUSES == ("", "unknown", "false", "true")
    eligible, reason = fixed_axis_eligibility(
        {"fixed_axis_eligible": "true", "orientation_category": "right"}
    )
    assert eligible is True
    assert reason is None
    for status in ("", "unknown", "false"):
        eligible, reason = fixed_axis_eligibility(
            {"fixed_axis_eligible": status, "orientation_category": "right"}
        )
        assert eligible is False
        assert reason is not None
    eligible, reason = fixed_axis_eligibility(
        {"fixed_axis_eligible": "true", "orientation_category": "unknown"}
    )
    assert eligible is False
    assert reason == "orientation_unknown"


def test_metadata_validation_rejects_inference_by_filename_and_absolute_notes(
    tmp_path: Path,
) -> None:
    _, pilot = _prepare(tmp_path)
    metadata = tmp_path / "metadata.csv"
    write_pilot_metadata_template(pilot, metadata)
    rows = _read_csv(metadata)
    rows[0]["orientation_category"] = "right"
    rows[0]["fixed_axis_eligible"] = ""
    rows[0]["notes"] = "C:\\private\\image.jpg"
    with metadata.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=METADATA_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(ValueError, match="absolute path"):
        read_pilot_metadata(
            metadata,
            expected_sample_ids=[f"pilot-{index}" for index in range(7)],
        )


def test_evaluation_metrics_eligibility_outputs_and_inference_only(
    tmp_path: Path,
) -> None:
    normalized, _, _, output, result, restored = _evaluate(tmp_path)
    summary = result["summary"]
    assert summary["total_sample_count"] == 7
    assert summary["segmentation_evaluated_count"] == 7
    assert summary["fixed_axis_eligible_count"] == 2
    assert summary["fixed_axis_excluded_count"] == 5
    assert len(result["segmentation_rows"]) == 7
    assert len(result["cutline_rows"]) == 4
    assert len(result["excluded_rows"]) == 5
    assert restored.was_training_during_forward is False
    assert restored.grad_enabled_during_forward is False
    assert restored.training is False
    assert normalized.is_dir()

    expected_files = {
        "manifest.json",
        "summary.json",
        "overlap_audit.json",
        "pilot_metadata_snapshot.csv",
        "segmentation_per_image.csv",
        "segmentation_confusion_matrix.csv",
        "cutline_per_image.csv",
        "excluded_from_fixed_axis.csv",
    }
    assert expected_files <= {path.name for path in output.iterdir() if path.is_file()}
    assert len(list((output / "visualizations").glob("*.png"))) == 7
    assert (
        tuple(_read_csv(output / "segmentation_per_image.csv")[0])
        == SEGMENTATION_FIELDS
    )
    assert tuple(_read_csv(output / "cutline_per_image.csv")[0]) == CUTLINE_FIELDS
    assert all(
        row["sample_id"] in {"pilot-0", "pilot-1"} for row in result["cutline_rows"]
    )


def test_global_segmentation_metrics_and_counts_are_complete(tmp_path: Path) -> None:
    _, _, _, output, result, _ = _evaluate(tmp_path, eligible_count=0)
    metrics = result["summary"]["global_segmentation_metrics"]
    required = {
        "iou_background",
        "iou_flesh",
        "iou_calyx",
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
    }
    assert required <= set(metrics)
    for row in result["segmentation_rows"]:
        assert {
            "calyx_precision",
            "calyx_recall",
            "flesh_fp",
            "flesh_fn",
            "calyx_fp",
            "calyx_fn",
        } <= set(row)
    confusion = list(
        csv.reader(
            (output / "segmentation_confusion_matrix.csv").open(encoding="utf-8")
        )
    )
    assert confusion[0] == ["target/prediction", "background", "Flesh", "Calyx"]


def test_v2a_and_v2b_coordinate_difference_sign_convention() -> None:
    ground_truth_mask = _semantic_mask(shift=0)
    predicted_mask = _semantic_mask(shift=2)
    ground_truth_v2a, ground_truth_v2b = _run_frozen_geometry(
        ground_truth_mask,
        v2a_estimator=estimate_fixed_axis_cutline,
        v2b_estimator=estimate_fixed_axis_search_cutline,
    )
    predicted_v2a, predicted_v2b = _run_frozen_geometry(
        predicted_mask,
        v2a_estimator=estimate_fixed_axis_cutline,
        v2b_estimator=estimate_fixed_axis_search_cutline,
    )
    rows = _cutline_comparison_rows(
        "sample",
        ground_truth_mask,
        predicted_mask,
        ground_truth_v2a,
        ground_truth_v2b,
        predicted_v2a,
        predicted_v2b,
    )
    for row in rows:
        assert row[
            "signed_coordinate_difference_predicted_minus_ground_truth"
        ] == pytest.approx(row["predicted_coordinate"] - row["ground_truth_coordinate"])
        assert row["absolute_coordinate_difference"] == pytest.approx(
            abs(row["signed_coordinate_difference_predicted_minus_ground_truth"])
        )


def test_within_threshold_calculations_and_failed_pairs_are_excluded() -> None:
    rows = [
        {
            "method": "v2a",
            "signed_coordinate_difference_predicted_minus_ground_truth": 2.0,
            "status_agreement": True,
            "structured_status_agreement": True,
        },
        {
            "method": "v2a",
            "signed_coordinate_difference_predicted_minus_ground_truth": -4.0,
            "status_agreement": True,
            "structured_status_agreement": True,
        },
        {
            "method": "v2a",
            "signed_coordinate_difference_predicted_minus_ground_truth": 8.0,
            "status_agreement": False,
            "structured_status_agreement": False,
        },
        {
            "method": "v2a",
            "signed_coordinate_difference_predicted_minus_ground_truth": None,
            "status_agreement": False,
            "structured_status_agreement": False,
        },
    ]
    summary = coordinate_agreement_summary(rows, method="v2a")
    assert summary["eligible_sample_count"] == 4
    assert summary["successful_coordinate_pair_count"] == 3
    assert summary["failed_coordinate_pair_count"] == 1
    assert summary["within_2_pixels"] == {
        "count": 1,
        "denominator": 3,
        "proportion": pytest.approx(1 / 3),
    }
    assert summary["within_5_pixels"] == {
        "count": 2,
        "denominator": 3,
        "proportion": pytest.approx(2 / 3),
    }
    assert summary["within_10_pixels"] == {
        "count": 3,
        "denominator": 3,
        "proportion": 1.0,
    }


def test_structured_ground_truth_and_prediction_failures_have_no_finite_error() -> None:
    successful_mask = _semantic_mask()
    failed_mask = _semantic_mask(include_calyx=False)
    ground_truth_v2a, ground_truth_v2b = _run_frozen_geometry(
        successful_mask,
        v2a_estimator=estimate_fixed_axis_cutline,
        v2b_estimator=estimate_fixed_axis_search_cutline,
    )
    predicted_v2a, predicted_v2b = _run_frozen_geometry(
        failed_mask,
        v2a_estimator=estimate_fixed_axis_cutline,
        v2b_estimator=estimate_fixed_axis_search_cutline,
    )
    rows = _cutline_comparison_rows(
        "failure",
        successful_mask,
        failed_mask,
        ground_truth_v2a,
        ground_truth_v2b,
        predicted_v2a,
        predicted_v2b,
    )
    assert all(row["predicted_status"] == "failed" for row in rows)
    assert all(row["predicted_failure_code"] for row in rows)
    assert all(row["absolute_coordinate_difference"] is None for row in rows)
    for method in ("v2a", "v2b"):
        summary = coordinate_agreement_summary(rows, method=method)
        assert summary["successful_coordinate_pair_count"] == 0
        assert summary["within_10_pixels"]["denominator"] == 0


def test_frozen_e050_w064_parameters_are_exact_and_not_cli_tunable() -> None:
    assert FROZEN_FIXED_AXIS_CONFIGURATION == {
        "configuration_id": "e050_w064",
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
    evaluate_parser = _build_argument_parser()
    with pytest.raises(SystemExit):
        evaluate_parser.parse_args(
            [
                "evaluate",
                "--dataset-root",
                "dataset",
                "--checkpoint",
                "checkpoint.pt",
                "--output-root",
                "output",
                "--metadata-manifest",
                "metadata.csv",
                "--projection-quantile",
                "0.5",
            ]
        )


def test_frozen_parameters_are_forwarded_without_tuning() -> None:
    calls: dict[str, dict[str, Any]] = {}

    def v2a(mask: np.ndarray, **kwargs: Any) -> Any:
        calls["v2a"] = dict(kwargs)
        return estimate_fixed_axis_cutline(mask, **kwargs)

    def v2b(mask: np.ndarray, **kwargs: Any) -> Any:
        calls["v2b"] = dict(kwargs)
        return estimate_fixed_axis_search_cutline(mask, **kwargs)

    _run_frozen_geometry(_semantic_mask(), v2a_estimator=v2a, v2b_estimator=v2b)
    assert calls["v2a"] == {
        "removal_axis": (1.0, 0.0),
        "projection_quantile": 0.95,
        "support_band_width_pixels": 5.0,
        "calyx_dilation_radius": 1,
        "component_connectivity": 8,
        "signed_offset_pixels": 0.0,
    }
    assert calls["v2b"]["candidate_step_pixels"] == 1.0
    assert calls["v2b"]["lateral_window_half_width_pixels"] == 64.0
    assert calls["v2b"]["minimum_attachment_evidence_fraction"] == 0.50
    assert calls["v2b"]["v2a_result"] is not None


def test_visualization_is_headless_lossless_and_ineligible_has_no_geometry() -> None:
    mask = _semantic_mask()
    image = _rgb_for_mask(mask)
    confidence = np.full(mask.shape, 0.8, dtype=np.float32)
    entropy = np.full(mask.shape, 0.2, dtype=np.float32)
    visualization = create_pilot_holdout_visualization(
        image,
        mask,
        mask,
        confidence,
        entropy,
        sample_id="sample",
        metadata={
            "orientation_category": "unknown",
            "fixed_axis_eligible": "",
            "occlusion_category": "unknown",
            "annotation_qa_status": "approved",
        },
        segmentation_metrics={},
        fixed_axis_eligible=False,
    )
    assert isinstance(visualization, Image.Image)
    assert visualization.mode == "RGB"
    assert visualization.width >= WIDTH * 3
    assert INELIGIBLE_MESSAGE.startswith("segmentation-only")


def test_metadata_and_checkpoint_identity_are_sanitized(tmp_path: Path) -> None:
    normalized, _, checkpoint, output, result, _ = _evaluate(tmp_path)
    serialized = "\n".join(
        (output / name).read_text(encoding="utf-8")
        for name in ("manifest.json", "summary.json", "overlap_audit.json")
    )
    assert str(tmp_path.resolve()) not in serialized
    assert str(normalized.resolve()) not in serialized
    assert str(checkpoint.resolve()) not in serialized
    checkpoint_identity = result["manifest"]["checkpoint"]
    assert checkpoint_identity["identifier"] == "best_checkpoint.pt"
    assert len(checkpoint_identity["sha256"]) == 64
    assert set(checkpoint_identity) == {
        "identifier",
        "sha256",
        "model_name",
        "completed_epoch",
        "stored_best_mean_foreground_iou",
    }


def test_noncanonical_checkpoint_epoch_is_rejected(tmp_path: Path) -> None:
    _, normalized = _prepare(tmp_path)
    metadata = tmp_path / "metadata.csv"
    write_pilot_metadata_template(normalized, metadata)
    _edit_metadata(metadata, set())
    checkpoint = tmp_path / "checkpoint" / "wrong_epoch.pt"
    _write_checkpoint(checkpoint, completed_epoch=91)

    with pytest.raises(ValueError, match="canonical epoch-92"):
        evaluate_pilot_holdout(
            normalized,
            checkpoint,
            tmp_path / "output",
            metadata,
            device="cpu",
            model_factory=TinyPilotModel,
        )


def test_evaluation_is_deterministic_and_overwrite_safe(tmp_path: Path) -> None:
    normalized, metadata, checkpoint, first, _, _ = _evaluate(
        tmp_path, output_name="first"
    )
    second = tmp_path / "second"
    evaluate_pilot_holdout(
        normalized,
        checkpoint,
        second,
        metadata,
        device="cpu",
        model_factory=TinyPilotModel,
    )
    assert _snapshot(first) == _snapshot(second)
    with pytest.raises(FileExistsError, match="--overwrite"):
        evaluate_pilot_holdout(
            normalized,
            checkpoint,
            first,
            metadata,
            device="cpu",
            model_factory=TinyPilotModel,
        )


def test_exclusion_reasons_and_limitations_use_pilot_terminology(
    tmp_path: Path,
) -> None:
    _, _, _, output, result, _ = _evaluate(tmp_path, eligible_count=0)
    excluded = _read_csv(output / "excluded_from_fixed_axis.csv")
    assert len(excluded) == 7
    assert all(
        row["exclusion_reason"] == "fixed_axis_eligibility_false" for row in excluded
    )
    limitations = result["summary"]["pilot_study_limitations"]
    assert tuple(limitations) == PILOT_LIMITATIONS
    assert "preliminary new-batch evaluation" in limitations[0]
    assert "seven-image pilot subset" in limitations[0]
    assert all("physical cut accuracy" not in text.lower() for text in limitations)


def test_cli_commands_parse_without_private_or_absolute_defaults() -> None:
    parser = _build_argument_parser()
    prepare = parser.parse_args(
        [
            "prepare",
            "--raw-labelled-dir",
            "raw",
            "--normalized-output-dir",
            "normalized",
        ]
    )
    template = parser.parse_args(
        ["template", "--dataset-root", "normalized", "--output", "metadata.csv"]
    )
    evaluate = parser.parse_args(
        [
            "evaluate",
            "--dataset-root",
            "normalized",
            "--checkpoint",
            "best_checkpoint.pt",
            "--output-root",
            "evaluation",
            "--metadata-manifest",
            "metadata.csv",
        ]
    )
    assert prepare.expected_sample_count == 7
    assert prepare.input_format == "auto"
    assert prepare.category_map is None
    assert template.command == "template"
    assert evaluate.expected_sample_count == 7
    assert not hasattr(evaluate, "projection_quantile")
    assert SUPPORTED_ORIENTATION_CATEGORIES[:3] == (
        "right",
        "upper_right",
        "lower_right",
    )
    assert _parse_category_map_arguments(["17=Flesh", "29=Calyx"]) == {
        17: "Flesh",
        29: "Calyx",
    }
