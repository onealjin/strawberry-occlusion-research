import csv
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
import torch.nn.functional as F
from PIL import Image
from torch import nn

import strawberry_occlusion.training.segmentation as training_module
from strawberry_occlusion.training.losses import (
    CombinedSegmentationLoss,
    soft_foreground_dice_loss,
)
from strawberry_occlusion.training.segmentation import (
    create_training_datasets,
    preprocess_class_mask,
    preprocess_rgb_image,
    train_segmentation_baseline,
    validate_segmentation_epoch,
)


class TinySegmentationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(3, 3, kernel_size=1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.classifier(image)


class InputLogitsModel(nn.Module):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return image


def _tiny_model_factory() -> nn.Module:
    return TinySegmentationModel()


def _make_dataset(root: Path) -> None:
    for split in ("train", "val"):
        (root / split / "images").mkdir(parents=True)
        (root / split / "masks").mkdir(parents=True)

    masks = {
        "train": (
            np.asarray(
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [2, 2, 0, 0],
                    [2, 2, 0, 0],
                ],
                dtype=np.uint8,
            ),
            np.asarray(
                [
                    [2, 2, 1, 1],
                    [2, 2, 1, 1],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
        ),
        "val": (
            np.asarray(
                [
                    [0, 0, 1, 1],
                    [0, 0, 1, 1],
                    [2, 2, 2, 2],
                    [0, 0, 0, 0],
                ],
                dtype=np.uint8,
            ),
            np.asarray(
                [
                    [1, 1, 0, 0],
                    [1, 1, 0, 0],
                    [2, 2, 0, 0],
                    [2, 2, 0, 0],
                ],
                dtype=np.uint8,
            ),
        ),
    }

    for split, split_masks in masks.items():
        for index, mask in enumerate(split_masks):
            red = np.where(mask == 1, 220, 20)
            green = np.where(mask == 2, 220, 20)
            blue = np.where(mask == 0, 100, 20)
            image = np.stack((red, green, blue), axis=-1).astype(np.uint8)
            Image.fromarray(image).save(root / split / "images" / f"sample-{index}.png")
            Image.fromarray(mask).save(root / split / "masks" / f"sample-{index}.png")


def _snapshot_files(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _read_history(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as history_file:
        return list(csv.DictReader(history_file))


def _load_checkpoint(path: Path) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    assert isinstance(checkpoint, dict)
    return checkpoint


def _maximum_optimizer_step(checkpoint: dict[str, Any]) -> float:
    steps = [
        float(state["step"])
        for state in checkpoint["optimizer_state_dict"]["state"].values()
        if "step" in state
    ]
    return max(steps)


def _train_once(
    dataset_root: Path,
    output_root: Path,
    *,
    epochs: int = 1,
    seed: int = 7,
    amp: bool = False,
    resume: bool | Path | None = None,
) -> dict[str, Any]:
    return train_segmentation_baseline(
        dataset_root,
        output_root,
        epochs=epochs,
        batch_size=1,
        height=4,
        width=4,
        learning_rate=0.01,
        weight_decay=0.0,
        seed=seed,
        num_workers=0,
        device="cpu",
        amp=amp,
        resume=resume,
        model_factory=_tiny_model_factory,
    )


def test_combined_loss_is_finite_and_rewards_nearly_perfect_logits() -> None:
    target = torch.tensor([[[0, 1], [2, 1]]], dtype=torch.long)
    perfect_logits = torch.full((1, 3, 2, 2), -8.0)
    perfect_logits.scatter_(1, target.unsqueeze(1), 8.0)
    incorrect_target = (target + 1) % 3
    incorrect_logits = torch.full((1, 3, 2, 2), -8.0)
    incorrect_logits.scatter_(1, incorrect_target.unsqueeze(1), 8.0)
    loss_function = CombinedSegmentationLoss()

    perfect_loss = loss_function(perfect_logits, target)
    incorrect_loss = loss_function(incorrect_logits, target)

    assert torch.isfinite(perfect_loss)
    assert torch.isfinite(incorrect_loss)
    assert perfect_loss < incorrect_loss


def test_soft_dice_excludes_background() -> None:
    logits = torch.tensor(
        [
            [
                [[2.0, -1.0], [0.5, 1.0]],
                [[0.0, 2.0], [-1.0, 0.0]],
                [[-1.0, 0.0], [2.0, -0.5]],
            ]
        ]
    )
    target = torch.tensor([[[0, 1], [2, 0]]], dtype=torch.long)

    actual = soft_foreground_dice_loss(logits, target)

    probabilities = torch.softmax(logits, dim=1)
    one_hot = F.one_hot(target, num_classes=3).permute(0, 3, 1, 2).float()
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum((0, 2, 3))
    denominator = probabilities[:, 1:].sum((0, 2, 3)) + one_hot[:, 1:].sum((0, 2, 3))
    expected = 1.0 - ((2.0 * intersection + 1e-6) / (denominator + 1e-6)).mean()

    all_intersection = (probabilities * one_hot).sum((0, 2, 3))
    all_denominator = probabilities.sum((0, 2, 3)) + one_hot.sum((0, 2, 3))
    including_background = (
        1.0 - ((2.0 * all_intersection + 1e-6) / (all_denominator + 1e-6)).mean()
    )

    torch.testing.assert_close(actual, expected)
    assert not torch.isclose(actual, including_background)


def test_absent_foreground_classes_do_not_produce_nan_loss() -> None:
    logits = torch.randn(2, 3, 3, 4)
    target = torch.zeros(2, 3, 4, dtype=torch.long)

    loss = soft_foreground_dice_loss(logits, target)

    assert torch.isfinite(loss)


def test_preprocessing_uses_bilinear_images_and_nearest_long_masks() -> None:
    checker = np.asarray(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[255, 255, 255], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    image = Image.fromarray(checker)
    mask = Image.fromarray(np.asarray([[0, 1], [2, 0]], dtype=np.uint8))

    resized_image = preprocess_rgb_image(image, height=3, width=3)
    resized_mask = preprocess_class_mask(mask, height=4, width=4)

    assert resized_image.shape == (3, 3, 3)
    torch.testing.assert_close(
        resized_image[:, 1, 1],
        torch.full((3,), 0.5),
        atol=1e-6,
        rtol=0,
    )
    assert resized_mask.dtype == torch.long
    assert resized_mask.tolist() == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 0, 0],
        [2, 2, 0, 0],
    ]


def test_invalid_mask_values_are_rejected_before_training(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    invalid_mask = np.full((4, 4), 3, dtype=np.uint8)
    Image.fromarray(invalid_mask).save(
        dataset_root / "train" / "masks" / "sample-0.png"
    )

    with pytest.raises(ValueError, match="only class IDs 0, 1, and 2"):
        create_training_datasets(dataset_root, height=4, width=4)


def test_multichannel_masks_are_rejected_before_training(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    Image.new("RGB", (4, 4), color=(0, 1, 2)).save(
        dataset_root / "train" / "masks" / "sample-0.png"
    )

    with pytest.raises(ValueError, match="single-channel"):
        create_training_datasets(dataset_root, height=4, width=4)


def test_image_mask_dimension_mismatch_is_rejected_before_training(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    Image.new("L", (5, 4), color=0).save(
        dataset_root / "train" / "masks" / "sample-0.png"
    )

    with pytest.raises(ValueError, match="sizes differ"):
        create_training_datasets(dataset_root, height=4, width=4)


def test_fixed_train_and_val_directories_are_used(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    (dataset_root / "unused").mkdir()

    train_dataset, val_dataset = create_training_datasets(
        dataset_root,
        height=4,
        width=4,
    )

    assert len(train_dataset) == 2
    assert len(val_dataset) == 2
    assert train_dataset.image_dir == dataset_root / "train" / "images"
    assert train_dataset.mask_dir == dataset_root / "train" / "masks"
    assert val_dataset.image_dir == dataset_root / "val" / "images"
    assert val_dataset.mask_dir == dataset_root / "val" / "masks"


def test_validation_confusion_matrix_accumulates_across_batches() -> None:
    first_logits = torch.tensor([[[[5.0, 0.0]], [[0.0, 5.0]], [[0.0, 0.0]]]])
    second_logits = torch.tensor([[[[0.0, 0.0]], [[5.0, 0.0]], [[0.0, 5.0]]]])
    batches = [
        {
            "image": first_logits,
            "mask": torch.tensor([[[0, 1]]], dtype=torch.long),
        },
        {
            "image": second_logits,
            "mask": torch.tensor([[[2, 2]]], dtype=torch.long),
        },
    ]

    metrics, matrix = validate_segmentation_epoch(
        InputLogitsModel(),
        batches,
        CombinedSegmentationLoss(),
        torch.device("cpu"),
    )

    assert torch.equal(
        matrix,
        torch.tensor(
            [
                [1, 0, 0],
                [0, 1, 0],
                [0, 1, 1],
            ]
        ),
    )
    assert metrics["val_iou_flesh"] == pytest.approx(0.5)
    assert metrics["val_iou_calyx"] == pytest.approx(0.5)
    assert metrics["val_mean_foreground_iou"] == pytest.approx(0.5)


def test_one_cpu_epoch_creates_artifacts_sanitized_metadata_and_preserves_input(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "synthetic-dataset"
    output_root = tmp_path / "run"
    _make_dataset(dataset_root)
    input_before = _snapshot_files(dataset_root)

    summary = _train_once(dataset_root, output_root, amp=True)

    assert summary["completed_epochs"] == 1
    assert summary["amp_enabled"] is False
    for artifact in (
        "config.json",
        "history.csv",
        "best_checkpoint.pt",
        "last_checkpoint.pt",
        "summary.json",
        "best_validation_metrics.json",
    ):
        assert (output_root / artifact).is_file()
    history = _read_history(output_root / "history.csv")
    assert len(history) == 1
    assert set(history[0]) == set(training_module.HISTORY_FIELDS)

    config = json.loads((output_root / "config.json").read_text(encoding="utf-8"))
    assert config["fixed_split_directories"] == {
        "train_images": "train/images",
        "train_masks": "train/masks",
        "val_images": "val/images",
        "val_masks": "val/masks",
    }
    assert config["seed"] == 7
    assert config["amp_requested"] is True
    assert config["amp_enabled"] is False
    assert config["train_sample_count"] == 2
    assert config["val_sample_count"] == 2
    assert _snapshot_files(dataset_root) == input_before

    for metadata_name in (
        "config.json",
        "summary.json",
        "best_validation_metrics.json",
    ):
        serialized = (output_root / metadata_name).read_text(encoding="utf-8")
        assert str(tmp_path.resolve()) not in serialized
        assert str(dataset_root.resolve()) not in serialized

    checkpoint = _load_checkpoint(output_root / "last_checkpoint.pt")
    assert checkpoint["completed_epoch"] == 1
    assert checkpoint["class_mapping"] == {
        "background": 0,
        "Flesh": 1,
        "Calyx": 2,
    }
    assert checkpoint["input_height"] == 4
    assert checkpoint["input_width"] == 4
    assert checkpoint["seed"] == 7
    assert checkpoint["model_name"] == "TinySegmentationModel"
    assert checkpoint["loss_configuration"] == {
        "cross_entropy_weight": 0.5,
        "dice_weight": 0.5,
        "foreground_classes": [1, 2],
    }


def test_best_checkpoint_uses_mean_foreground_iou(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "run"
    _make_dataset(dataset_root)
    mean_ious = iter((0.2, 0.8, 0.5))

    def fake_train_epoch(*args: Any, **kwargs: Any) -> dict[str, float]:
        return {
            "train_total_loss": 1.0,
            "train_cross_entropy_loss": 1.0,
            "train_dice_loss": 1.0,
        }

    def fake_validation_epoch(
        *args: Any,
        **kwargs: Any,
    ) -> tuple[dict[str, float], torch.Tensor]:
        mean_iou = next(mean_ious)
        return (
            {
                "val_total_loss": 1.0,
                "val_cross_entropy_loss": 1.0,
                "val_dice_loss": 1.0,
                "val_iou_background": 0.9,
                "val_iou_flesh": mean_iou,
                "val_iou_calyx": mean_iou,
                "val_dice_background": 0.9,
                "val_dice_flesh": mean_iou,
                "val_dice_calyx": mean_iou,
                "val_mean_foreground_iou": mean_iou,
                "val_mean_foreground_dice": mean_iou,
            },
            torch.zeros((3, 3), dtype=torch.long),
        )

    monkeypatch.setattr(
        training_module,
        "train_segmentation_epoch",
        fake_train_epoch,
    )
    monkeypatch.setattr(
        training_module,
        "validate_segmentation_epoch",
        fake_validation_epoch,
    )

    _train_once(dataset_root, output_root, epochs=3)

    best_checkpoint = _load_checkpoint(output_root / "best_checkpoint.pt")
    best_metrics = json.loads(
        (output_root / "best_validation_metrics.json").read_text(encoding="utf-8")
    )
    assert best_checkpoint["completed_epoch"] == 2
    assert best_checkpoint["best_mean_foreground_iou"] == pytest.approx(0.8)
    assert best_metrics["epoch"] == 2
    assert best_metrics["val_mean_foreground_iou"] == pytest.approx(0.8)


def test_resume_restores_epoch_optimizer_state_and_validates_configuration(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "run"
    _make_dataset(dataset_root)

    _train_once(dataset_root, output_root, epochs=1)
    first_checkpoint = _load_checkpoint(output_root / "last_checkpoint.pt")
    first_step = _maximum_optimizer_step(first_checkpoint)

    summary = _train_once(
        dataset_root,
        output_root,
        epochs=2,
        resume=True,
    )
    resumed_checkpoint = _load_checkpoint(output_root / "last_checkpoint.pt")

    assert summary["completed_epochs"] == 2
    assert resumed_checkpoint["completed_epoch"] == 2
    assert _maximum_optimizer_step(resumed_checkpoint) > first_step
    assert len(_read_history(output_root / "history.csv")) == 2

    with pytest.raises(ValueError, match="Incompatible checkpoint input_height"):
        train_segmentation_baseline(
            dataset_root,
            output_root,
            epochs=3,
            batch_size=1,
            height=5,
            width=4,
            learning_rate=0.01,
            weight_decay=0.0,
            seed=7,
            device="cpu",
            resume=True,
            model_factory=_tiny_model_factory,
        )


def test_overwrite_protection_and_resume_conflict(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "run"
    _make_dataset(dataset_root)
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("unchanged", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        _train_once(dataset_root, output_root)

    assert marker.read_text(encoding="utf-8") == "unchanged"

    with pytest.raises(ValueError, match="cannot be used together"):
        train_segmentation_baseline(
            dataset_root,
            output_root,
            overwrite=True,
            resume=True,
            model_factory=_tiny_model_factory,
        )


def test_same_seed_produces_matching_training_history(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    _make_dataset(dataset_root)
    first_output = tmp_path / "first-run"
    second_output = tmp_path / "second-run"

    _train_once(dataset_root, first_output, seed=23)
    _train_once(dataset_root, second_output, seed=23)

    assert (first_output / "history.csv").read_bytes() == (
        second_output / "history.csv"
    ).read_bytes()
    first_checkpoint = _load_checkpoint(first_output / "last_checkpoint.pt")
    second_checkpoint = _load_checkpoint(second_output / "last_checkpoint.pt")
    for key, first_tensor in first_checkpoint["model_state_dict"].items():
        torch.testing.assert_close(
            first_tensor,
            second_checkpoint["model_state_dict"][key],
            rtol=0,
            atol=0,
        )
