from pathlib import Path

import pytest
import torch
from PIL import Image

from strawberry_occlusion.data import SegmentationDataset


def test_segmentation_dataset_loads_rgb_image_and_class_index_mask(
    tmp_path: Path,
) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    image_path = image_dir / "toy.png"
    mask_path = mask_dir / "toy.png"

    image = Image.new("RGB", (3, 2))
    image.putdata(
        [
            (255, 0, 0),
            (0, 255, 0),
            (0, 0, 255),
            (64, 128, 192),
            (32, 16, 8),
            (255, 255, 255),
        ]
    )
    image.save(image_path)

    mask = Image.new("L", (3, 2))
    mask.putdata([0, 1, 2, 3, 4, 5])
    mask.save(mask_path)

    dataset = SegmentationDataset(image_dir, mask_dir)
    sample = dataset[0]

    assert len(dataset) == 1
    assert set(sample) == {"image", "mask", "image_path", "mask_path"}
    assert sample["image"].shape == (3, 2, 3)
    assert sample["image"].dtype == torch.float32
    assert torch.all((sample["image"] >= 0.0) & (sample["image"] <= 1.0))
    assert sample["mask"].shape == (2, 3)
    assert sample["mask"].dtype == torch.long
    assert sample["mask"].tolist() == [[0, 1, 2], [3, 4, 5]]
    assert sample["image_path"] == str(image_path)
    assert sample["mask_path"] == str(mask_path)


def test_segmentation_dataset_preserves_palette_mask_indices(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    Image.new("RGB", (2, 2), color=(10, 20, 30)).save(image_dir / "palette.png")

    mask = Image.new("P", (2, 2))
    mask.putpalette([0, 0, 0, 255, 0, 0, 0, 255, 0] + [0, 0, 0] * 253)
    mask.putdata([0, 1, 2, 1])
    mask.save(mask_dir / "palette.png")

    sample = SegmentationDataset(image_dir, mask_dir)[0]

    assert sample["mask"].tolist() == [[0, 1], [2, 1]]


def test_segmentation_dataset_rejects_unpaired_files(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    Image.new("RGB", (2, 2)).save(image_dir / "image.png")
    Image.new("L", (2, 2)).save(mask_dir / "different.png")

    with pytest.raises(ValueError, match="Missing masks"):
        SegmentationDataset(image_dir, mask_dir)


def test_segmentation_dataset_rejects_multichannel_masks(tmp_path: Path) -> None:
    image_dir = tmp_path / "images"
    mask_dir = tmp_path / "masks"
    image_dir.mkdir()
    mask_dir.mkdir()

    Image.new("RGB", (2, 2)).save(image_dir / "toy.png")
    Image.new("RGB", (2, 2)).save(mask_dir / "toy.png")

    dataset = SegmentationDataset(image_dir, mask_dir)

    with pytest.raises(ValueError, match="single-channel"):
        dataset[0]
