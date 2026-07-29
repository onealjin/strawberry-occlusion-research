import base64
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from strawberry_occlusion.data.vt_export import (
    CLASS_MAPPING,
    _validate_bbox,
    construct_mask,
    convert_vt_export,
)


IMAGE_WIDTH = 16
IMAGE_HEIGHT = 4
CATEGORIES = {17: "Flesh", 29: "Calyx"}


def _encode_mask(rows: Sequence[Sequence[int]]) -> str:
    mask = np.asarray(rows, dtype=np.uint8)
    packed = np.packbits(mask.reshape(-1), bitorder="big")
    return base64.b64encode(packed.tobytes()).decode("ascii")


def _annotation(
    label: str,
    bbox: list[int | float],
    rows: Sequence[Sequence[int]],
    *,
    image_name: str = "train-sample.jpg",
    category_id: int | None = None,
) -> dict[str, Any]:
    if category_id is None:
        category_id = 17 if label == "Flesh" else 29
    return {
        "category_id": category_id,
        "labelname": label,
        "bbox": bbox,
        "pixel_image": _encode_mask(rows),
        "image_name": image_name,
        "type": "pencil",
    }


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value), encoding="utf-8")


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _default_train_annotations() -> list[dict[str, Any]]:
    return [
        _annotation(
            "Flesh",
            [2, 1, 8, 2],
            [
                [1, 0, 1, 0, 0, 0, 0, 0],
                [0, 1, 0, 1, 0, 0, 0, 0],
            ],
        ),
        _annotation(
            "Calyx",
            [2, 1, 8, 2],
            [
                [1, 1, 0, 0, 0, 0, 0, 0],
                [0, 0, 0, 0, 1, 0, 0, 0],
            ],
        ),
    ]


def _create_export(tmp_path: Path) -> tuple[Path, Path]:
    input_root = tmp_path / "vt-input"
    output_root = tmp_path / "normalized-output"
    input_root.mkdir()

    categories = [
        {"id": 17, "name": "Flesh", "color": "#8A1C1C"},
        {"id": 29, "name": "Calyx", "color": "#1C8A2A"},
    ]
    images = [
        {
            "id": "train-sample",
            "file_name": "train-sample.jpg",
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "image_type": "train",
            "labeled": True,
        },
        {
            "id": "val-sample",
            "file_name": "val-sample.jpg",
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "image_type": "val",
            "labeled": True,
        },
        {
            "id": "unlabeled-sample",
            "file_name": "unlabeled-sample.jpg",
            "width": IMAGE_WIDTH,
            "height": IMAGE_HEIGHT,
            "image_type": "train",
            "labeled": False,
        },
    ]
    _write_json(
        input_root / "dataset.json", {"categories": categories, "images": images}
    )

    Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(110, 20, 30)).save(
        input_root / "train-sample.jpg",
        quality=91,
    )
    Image.new("RGB", (IMAGE_WIDTH, IMAGE_HEIGHT), color=(30, 110, 20)).save(
        input_root / "val-sample.jpg",
        quality=91,
    )
    _write_json(input_root / "train-sample.json", _default_train_annotations())
    _write_json(
        input_root / "val-sample.json",
        [
            _annotation(
                "Calyx",
                [8, 0, 8, 1],
                [[0, 0, 0, 1, 1, 0, 0, 0]],
                image_name="val-sample.jpg",
            )
        ],
    )
    return input_root, output_root


def _replace_train_annotations(
    input_root: Path,
    annotations: list[dict[str, Any]],
) -> None:
    _write_json(input_root / "train-sample.json", annotations)


def test_bit_packed_big_endian_decoding_and_exact_bbox_placement() -> None:
    annotations = [
        _annotation(
            "Flesh",
            [3, 1, 8, 2],
            [
                [1, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 1, 0],
            ],
        )
    ]

    mask, summary = construct_mask(
        annotations,
        image_width=16,
        image_height=4,
        categories=CATEGORIES,
        image_name="train-sample.jpg",
    )

    expected = np.zeros((4, 16), dtype=np.uint8)
    expected[1, [3, 10]] = 1
    expected[2, [4, 9]] = 1
    np.testing.assert_array_equal(mask, expected)
    assert summary["flesh_pixel_count"] == 4
    assert summary["background_pixel_count"] == 60


def test_integer_valued_float_bbox_has_exact_placement_and_integer_coordinates() -> (
    None
):
    bbox = [2.0, 1.0, 8.0, 2.0]
    annotations = [
        _annotation(
            "Flesh",
            bbox,
            [
                [1, 0, 0, 0, 0, 0, 0, 1],
                [0, 1, 0, 0, 0, 0, 1, 0],
            ],
        )
    ]

    coordinates = _validate_bbox(
        bbox,
        image_width=16,
        image_height=4,
        annotation_index=0,
    )
    mask, _ = construct_mask(
        annotations,
        image_width=16,
        image_height=4,
        categories=CATEGORIES,
        image_name="train-sample.jpg",
    )

    expected = np.zeros((4, 16), dtype=np.uint8)
    expected[1, [2, 9]] = 1
    expected[2, [3, 8]] = 1
    assert coordinates == (2, 1, 8, 2)
    assert all(isinstance(value, int) for value in coordinates)
    np.testing.assert_array_equal(mask, expected)


@pytest.mark.parametrize(
    "invalid_value",
    [True, "2", float("nan"), float("inf"), float("-inf"), 2.5],
)
def test_bbox_rejects_non_integer_values(invalid_value: Any) -> None:
    bbox = [invalid_value, 1, 8, 2]

    with pytest.raises(ValueError, match="finite integer-valued numbers"):
        _validate_bbox(
            bbox,
            image_width=16,
            image_height=4,
            annotation_index=0,
        )


def test_multiple_same_class_regions_are_unioned() -> None:
    annotations = [
        _annotation(
            "Flesh",
            [0, 0, 8, 1],
            [[1, 1, 0, 0, 0, 0, 0, 0]],
        ),
        _annotation(
            "Flesh",
            [0, 0, 8, 1],
            [[0, 1, 1, 0, 0, 0, 0, 0]],
        ),
        _annotation(
            "Flesh",
            [8, 1, 8, 1],
            [[0, 0, 0, 0, 0, 0, 0, 1]],
        ),
    ]

    mask, summary = construct_mask(
        annotations,
        image_width=16,
        image_height=2,
        categories=CATEGORIES,
        image_name="train-sample.jpg",
    )

    expected = np.zeros((2, 16), dtype=np.uint8)
    expected[0, :3] = 1
    expected[1, 15] = 1
    np.testing.assert_array_equal(mask, expected)
    assert summary["flesh_annotation_count"] == 3
    assert summary["flesh_pixel_count"] == 4


def test_calyx_priority_overlap_reporting_background_and_final_values() -> None:
    annotations = [
        _annotation(
            "Flesh",
            [0, 0, 8, 1],
            [[1, 1, 1, 0, 0, 0, 0, 0]],
        ),
        _annotation(
            "Calyx",
            [0, 0, 8, 1],
            [[0, 1, 1, 1, 0, 0, 0, 0]],
        ),
    ]

    mask, summary = construct_mask(
        annotations,
        image_width=8,
        image_height=2,
        categories=CATEGORIES,
        image_name="train-sample.jpg",
    )

    np.testing.assert_array_equal(
        mask[0],
        np.asarray([1, 2, 2, 2, 0, 0, 0, 0], dtype=np.uint8),
    )
    assert summary == {
        "source_annotation_count": 2,
        "flesh_annotation_count": 1,
        "calyx_annotation_count": 1,
        "background_pixel_count": 12,
        "flesh_pixel_count": 1,
        "calyx_pixel_count": 3,
        "cross_class_overlap_pixel_count": 2,
    }
    assert set(np.unique(mask)) == {0, 1, 2}
    assert (
        sum(
            summary[key]
            for key in (
                "background_pixel_count",
                "flesh_pixel_count",
                "calyx_pixel_count",
            )
        )
        == mask.size
    )


def test_conversion_preserves_train_val_and_writes_sanitized_outputs(
    tmp_path: Path,
) -> None:
    input_root, output_root = _create_export(tmp_path)
    train_source_bytes = (input_root / "train-sample.jpg").read_bytes()
    val_source_bytes = (input_root / "val-sample.jpg").read_bytes()

    manifest = convert_vt_export(input_root, output_root)

    assert manifest["dataset_name"] == output_root.name
    assert manifest["class_mapping"] == CLASS_MAPPING
    assert manifest["overlap_rule"] == "Calyx overrides Flesh"
    assert manifest["sample_counts"] == {"train": 1, "val": 1}
    assert {sample["sample_id"] for sample in manifest["samples"]} == {
        "train-sample",
        "val-sample",
    }
    assert (output_root / "train/images/train-sample.jpg").read_bytes() == (
        train_source_bytes
    )
    assert (output_root / "val/images/val-sample.jpg").read_bytes() == val_source_bytes
    assert not (output_root / "train/images/unlabeled-sample.jpg").exists()

    train_sample = next(
        sample
        for sample in manifest["samples"]
        if sample["sample_id"] == "train-sample"
    )
    assert train_sample["split"] == "train"
    assert train_sample["image_path"] == "train/images/train-sample.jpg"
    assert train_sample["mask_path"] == "train/masks/train-sample.png"
    assert train_sample["image_width"] == IMAGE_WIDTH
    assert train_sample["image_height"] == IMAGE_HEIGHT
    assert train_sample["cross_class_overlap_pixel_count"] == 1

    with Image.open(output_root / train_sample["mask_path"]) as output_mask:
        assert output_mask.mode == "L"
        assert output_mask.size == (IMAGE_WIDTH, IMAGE_HEIGHT)
        mask_array = np.array(output_mask)
    assert mask_array.ndim == 2
    assert set(np.unique(mask_array)) <= {0, 1, 2}

    on_disk_manifest = _read_json(output_root / "manifest.json")
    assert on_disk_manifest == manifest
    serialized_manifest = json.dumps(on_disk_manifest)
    assert str(tmp_path.resolve()) not in serialized_manifest
    assert str(input_root.resolve()) not in serialized_manifest
    assert "color" not in serialized_manifest
    for sample in on_disk_manifest["samples"]:
        assert not Path(sample["image_path"]).is_absolute()
        assert not Path(sample["mask_path"]).is_absolute()


def test_input_files_remain_byte_for_byte_unchanged(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    before = {
        path.name: path.read_bytes() for path in input_root.iterdir() if path.is_file()
    }

    convert_vt_export(input_root, output_root)

    after = {
        path.name: path.read_bytes() for path in input_root.iterdir() if path.is_file()
    }
    assert after == before


def test_missing_per_image_json_is_rejected(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    (input_root / "train-sample.json").unlink()

    with pytest.raises(FileNotFoundError, match="annotation JSON"):
        convert_vt_export(input_root, output_root)

    assert not output_root.exists()


def test_missing_image_is_rejected(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    (input_root / "train-sample.jpg").unlink()

    with pytest.raises(FileNotFoundError, match="source image"):
        convert_vt_export(input_root, output_root)

    assert not output_root.exists()


def test_image_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    dataset = _read_json(input_root / "dataset.json")
    dataset["images"][0]["width"] = IMAGE_WIDTH + 1
    _write_json(input_root / "dataset.json", dataset)

    with pytest.raises(ValueError, match="dimensions.*do not match"):
        convert_vt_export(input_root, output_root)


def test_unknown_annotation_category_is_rejected(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    annotation = _default_train_annotations()[0]
    annotation["category_id"] = 999
    _replace_train_annotations(input_root, [annotation])

    with pytest.raises(ValueError, match="unknown category ID"):
        convert_vt_export(input_root, output_root)


def test_unsupported_source_semantic_label_is_rejected(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    dataset = _read_json(input_root / "dataset.json")
    dataset["categories"].append({"id": 999, "name": "Stem", "color": "#000000"})
    _write_json(input_root / "dataset.json", dataset)

    with pytest.raises(ValueError, match="Unsupported source category"):
        convert_vt_export(input_root, output_root)


def test_invalid_base64_is_rejected(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    annotation = _default_train_annotations()[0]
    annotation["pixel_image"] = "not valid base64!!!"
    _replace_train_annotations(input_root, [annotation])

    with pytest.raises(ValueError, match="invalid base64"):
        convert_vt_export(input_root, output_root)


def test_decoded_byte_length_mismatch_is_rejected(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    annotation = _default_train_annotations()[0]
    annotation["pixel_image"] = base64.b64encode(b"\x80").decode("ascii")
    _replace_train_annotations(input_root, [annotation])

    with pytest.raises(ValueError, match="byte-length mismatch"):
        convert_vt_export(input_root, output_root)


def test_bbox_outside_image_is_rejected(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    annotation = _annotation(
        "Flesh",
        [9, 0, 8, 1],
        [[1, 0, 0, 0, 0, 0, 0, 0]],
    )
    _replace_train_annotations(input_root, [annotation])

    with pytest.raises(ValueError, match="outside the source image"):
        convert_vt_export(input_root, output_root)


def test_annotation_image_name_mismatch_is_rejected(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    annotation = _default_train_annotations()[0]
    annotation["image_name"] = "different.jpg"
    _replace_train_annotations(input_root, [annotation])

    with pytest.raises(ValueError, match="image_name does not match"):
        convert_vt_export(input_root, output_root)


def test_malformed_json_is_rejected(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    (input_root / "train-sample.json").write_text("{malformed", encoding="utf-8")

    with pytest.raises(ValueError, match="Malformed JSON"):
        convert_vt_export(input_root, output_root)


def test_overwrite_requires_explicit_flag(tmp_path: Path) -> None:
    input_root, output_root = _create_export(tmp_path)
    output_root.mkdir()
    marker_path = output_root / "keep.txt"
    marker_path.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        convert_vt_export(input_root, output_root)

    assert marker_path.read_text(encoding="utf-8") == "unchanged"

    manifest = convert_vt_export(input_root, output_root, overwrite=True)

    assert manifest["sample_counts"] == {"train": 1, "val": 1}
    assert not marker_path.exists()
    assert (output_root / "manifest.json").is_file()
