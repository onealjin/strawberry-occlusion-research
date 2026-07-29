import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from strawberry_occlusion.visualization.segmentation import (
    BOUNDARY_COLOR,
    TITLE_HEIGHT,
    colorize_mask,
    create_qa_image,
    find_class_boundaries,
    generate_segmentation_qa,
    overlay_boundaries,
    overlay_mask,
)


def _make_dataset_dirs(root: Path) -> None:
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)


def _write_pair(
    root: Path,
    split: str,
    stem: str,
    *,
    image: np.ndarray | None = None,
    mask: np.ndarray | None = None,
) -> tuple[Path, Path]:
    if image is None:
        image = np.full((2, 3, 3), (40, 80, 120), dtype=np.uint8)
    if mask is None:
        mask = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)

    image_path = root / split / "images" / f"{stem}.png"
    mask_path = root / split / "masks" / f"{stem}.png"
    Image.fromarray(image).save(image_path)
    Image.fromarray(mask).save(mask_path)
    return image_path, mask_path


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def test_colorize_mask_uses_deterministic_palette() -> None:
    mask = np.asarray([[0, 1, 2]], dtype=np.uint8)

    colorized = colorize_mask(mask)

    expected = np.asarray(
        [[[0, 0, 0], [255, 0, 0], [0, 255, 0]]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(colorized, expected)


def test_overlay_preserves_background_and_blends_flesh_and_calyx() -> None:
    image = np.full((1, 3, 3), 100, dtype=np.uint8)
    mask = np.asarray([[0, 1, 2]], dtype=np.uint8)

    overlay = overlay_mask(image, mask, alpha=0.4)

    expected = np.asarray(
        [[[100, 100, 100], [162, 60, 60], [60, 162, 60]]],
        dtype=np.uint8,
    )
    np.testing.assert_array_equal(overlay, expected)


def test_boundaries_use_exact_four_neighbour_coordinates() -> None:
    mask = np.zeros((3, 3), dtype=np.uint8)
    mask[1, 1] = 1

    boundaries = find_class_boundaries(mask)

    expected = np.asarray(
        [
            [False, True, False],
            [True, True, True],
            [False, True, False],
        ]
    )
    np.testing.assert_array_equal(boundaries, expected)

    image = np.zeros((3, 3, 3), dtype=np.uint8)
    boundary_overlay = overlay_boundaries(image, mask)
    np.testing.assert_array_equal(
        boundary_overlay[boundaries],
        np.tile(np.asarray(BOUNDARY_COLOR), (5, 1)),
    )
    assert np.all(boundary_overlay[~boundaries] == 0)


def test_qa_image_has_four_exact_resolution_panels() -> None:
    image = np.full((2, 3, 3), (10, 20, 30), dtype=np.uint8)
    mask = np.asarray([[0, 1, 2], [2, 1, 0]], dtype=np.uint8)

    qa_image = create_qa_image(image, mask, alpha=0.4)
    qa_array = np.array(qa_image)

    assert qa_image.size == (12, 2 + TITLE_HEIGHT)
    panel_pixels = qa_array[TITLE_HEIGHT:]
    np.testing.assert_array_equal(panel_pixels[:, 0:3], image)
    np.testing.assert_array_equal(panel_pixels[:, 3:6], colorize_mask(mask))
    np.testing.assert_array_equal(
        panel_pixels[:, 6:9],
        overlay_mask(image, mask, alpha=0.4),
    )
    np.testing.assert_array_equal(
        panel_pixels[:, 9:12],
        overlay_boundaries(image, mask),
    )


def test_invalid_mask_values_are_rejected() -> None:
    mask = np.asarray([[0, 1, 3]], dtype=np.uint8)

    with pytest.raises(ValueError, match="invalid class values"):
        colorize_mask(mask)


def test_multichannel_mask_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "qa"
    _make_dataset_dirs(dataset_root)
    _write_pair(dataset_root, "val", "valid")
    _, mask_path = _write_pair(dataset_root, "train", "multichannel")
    Image.new("RGB", (3, 2), color=(0, 1, 2)).save(mask_path)

    with pytest.raises(ValueError, match="single-channel"):
        generate_segmentation_qa(dataset_root, output_root)

    assert not output_root.exists()


def test_image_mask_dimension_mismatch_is_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "qa"
    _make_dataset_dirs(dataset_root)
    _write_pair(dataset_root, "val", "valid")
    _, mask_path = _write_pair(dataset_root, "train", "mismatch")
    Image.new("L", (4, 2), color=0).save(mask_path)

    with pytest.raises(ValueError, match="dimensions differ"):
        generate_segmentation_qa(dataset_root, output_root)


def test_train_val_generation_metadata_input_safety_and_headless_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("DISPLAY", raising=False)
    dataset_root = tmp_path / "normalized"
    output_root = tmp_path / "qa"
    _make_dataset_dirs(dataset_root)
    _write_pair(dataset_root, "train", "train-sample")
    _write_pair(
        dataset_root,
        "val",
        "val-sample",
        mask=np.asarray([[0, 0, 2], [1, 1, 2]], dtype=np.uint8),
    )
    input_before = _snapshot_files(dataset_root)

    manifest = generate_segmentation_qa(dataset_root, output_root, alpha=0.4)

    assert manifest["sample_counts"] == {"train": 1, "val": 1}
    assert manifest["alpha"] == 0.4
    assert manifest["class_mapping"] == {
        "background": 0,
        "Flesh": 1,
        "Calyx": 2,
    }
    assert manifest["palette"] == {
        "background": [0, 0, 0],
        "Flesh": [255, 0, 0],
        "Calyx": [0, 255, 0],
        "boundary": [255, 255, 0],
    }
    assert (output_root / "train/train-sample.png").is_file()
    assert (output_root / "val/val-sample.png").is_file()
    assert _snapshot_files(dataset_root) == input_before

    samples_by_id = {sample["sample_id"]: sample for sample in manifest["samples"]}
    assert samples_by_id["train-sample"]["output_path"] == ("train/train-sample.png")
    assert samples_by_id["train-sample"]["image_width"] == 3
    assert samples_by_id["train-sample"]["image_height"] == 2
    assert samples_by_id["train-sample"]["class_pixel_counts"] == {
        "background": 2,
        "Flesh": 2,
        "Calyx": 2,
    }

    on_disk_manifest: dict[str, Any] = json.loads(
        (output_root / "qa_manifest.json").read_text(encoding="utf-8")
    )
    assert on_disk_manifest == manifest
    serialized_manifest = json.dumps(on_disk_manifest)
    assert str(tmp_path.resolve()) not in serialized_manifest
    for sample in on_disk_manifest["samples"]:
        assert not Path(sample["output_path"]).is_absolute()

    with Image.open(output_root / "train/train-sample.png") as qa_image:
        assert qa_image.format == "PNG"
        assert qa_image.mode == "RGB"
        assert qa_image.size == (12, 2 + TITLE_HEIGHT)


def test_overwrite_requires_explicit_flag(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "qa"
    _make_dataset_dirs(dataset_root)
    _write_pair(dataset_root, "train", "train-sample")
    _write_pair(dataset_root, "val", "val-sample")
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        generate_segmentation_qa(dataset_root, output_root)

    assert marker.read_text(encoding="utf-8") == "unchanged"

    manifest = generate_segmentation_qa(
        dataset_root,
        output_root,
        overwrite=True,
    )
    assert manifest["sample_counts"] == {"train": 1, "val": 1}
    assert not marker.exists()


def test_unmatched_stems_are_rejected(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "qa"
    _make_dataset_dirs(dataset_root)
    _write_pair(dataset_root, "val", "valid")
    image_path, mask_path = _write_pair(dataset_root, "train", "missing-mask")
    assert image_path.is_file()
    mask_path.unlink()

    with pytest.raises(ValueError, match="Missing masks"):
        generate_segmentation_qa(dataset_root, output_root)
