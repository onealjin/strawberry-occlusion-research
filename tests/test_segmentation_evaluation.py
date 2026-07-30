import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn

import strawberry_occlusion.evaluation as evaluation_package
from strawberry_occlusion.evaluation.segmentation import (
    CLASS_MAPPING,
    confidence_and_entropy,
    create_evaluation_dataset,
    diagnostic_masks,
    evaluate_segmentation_checkpoint,
    load_segmentation_checkpoint,
    metrics_from_confusion_matrix,
    per_image_metrics,
)


def test_evaluation_package_exports_segmentation_module() -> None:
    assert evaluation_package.__all__ == ["segmentation"]
    assert (
        evaluation_package.segmentation.evaluate_segmentation_checkpoint
        is evaluate_segmentation_checkpoint
    )


class TinyEvaluationModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(12.0))
        self.was_training_during_forward: bool | None = None
        self.grad_enabled_during_forward: bool | None = None

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        self.was_training_during_forward = self.training
        self.grad_enabled_during_forward = torch.is_grad_enabled()
        return image * self.scale


class WrongShapeModel(TinyEvaluationModel):
    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return super().forward(image)[:, :, :-1, :]


def _tiny_model_factory() -> nn.Module:
    return TinyEvaluationModel()


def _write_sample(root: Path, sample_id: str, mask: np.ndarray) -> None:
    image_dir = root / "val" / "images"
    mask_dir = root / "val" / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    image = np.full((*mask.shape, 3), 5, dtype=np.uint8)
    for class_id in range(3):
        image[..., class_id][mask == class_id] = 255
    Image.fromarray(image).save(image_dir / f"{sample_id}.png")
    Image.fromarray(mask.astype(np.uint8)).save(mask_dir / f"{sample_id}.png")


def _make_dataset(root: Path) -> None:
    _write_sample(
        root,
        "a-sample",
        np.asarray([[0, 1, 2], [0, 1, 2]], dtype=np.uint8),
    )
    _write_sample(
        root,
        "b-sample",
        np.asarray([[2, 2, 1], [0, 0, 1]], dtype=np.uint8),
    )


def _write_checkpoint(
    path: Path,
    *,
    model: nn.Module | None = None,
    height: int = 2,
    width: int = 3,
    class_mapping: dict[str, int] | None = None,
) -> None:
    model = model or TinyEvaluationModel()
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": {},
            "completed_epoch": 4,
            "best_mean_foreground_iou": 0.75,
            "class_mapping": class_mapping or dict(CLASS_MAPPING),
            "input_height": height,
            "input_width": width,
            "seed": 42,
            "model_name": model.__class__.__name__,
            "loss_configuration": {
                "cross_entropy_weight": 0.5,
                "dice_weight": 0.5,
                "foreground_classes": [1, 2],
            },
        },
        path,
    )


def _write_history(path: Path) -> None:
    path.write_text(
        "epoch,train_total_loss,val_total_loss,val_iou_flesh,"
        "val_iou_calyx,val_mean_foreground_iou\n"
        "1,1.0,1.2,0.3,0.4,0.35\n"
        "4,0.5,0.7,0.7,0.8,0.75\n",
        encoding="utf-8",
    )


def _snapshot(root: Path) -> dict[str, bytes]:
    return {
        path.relative_to(root).as_posix(): path.read_bytes()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def _evaluate(
    tmp_path: Path,
    *,
    output_name: str = "evaluation",
    write_history: bool = False,
    model_factory: Any = _tiny_model_factory,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    dataset_root = tmp_path / "synthetic-dataset"
    checkpoint = tmp_path / "training-run" / "best_checkpoint.pt"
    output_root = tmp_path / output_name
    _make_dataset(dataset_root)
    _write_checkpoint(checkpoint)
    if write_history:
        _write_history(checkpoint.parent / "history.csv")
    manifest = evaluate_segmentation_checkpoint(
        dataset_root,
        checkpoint,
        output_root,
        split="val",
        device="cpu",
        batch_size=1,
        model_factory=model_factory,
    )
    return dataset_root, checkpoint, output_root, manifest


def test_aggregate_metrics_and_global_confusion_accumulate_across_batches(
    tmp_path: Path,
) -> None:
    _, _, output_root, manifest = _evaluate(tmp_path)

    with (output_root / "confusion_matrix.csv").open(
        encoding="utf-8", newline=""
    ) as matrix_file:
        rows = list(csv.reader(matrix_file))

    assert rows == [
        ["target/prediction", "background", "Flesh", "Calyx"],
        ["background", "4", "0", "0"],
        ["Flesh", "0", "4", "0"],
        ["Calyx", "0", "0", "4"],
    ]
    aggregate = manifest["aggregate_metrics"]
    assert aggregate["iou_background"] == pytest.approx(1.0)
    assert aggregate["iou_flesh"] == pytest.approx(1.0)
    assert aggregate["iou_calyx"] == pytest.approx(1.0)
    assert aggregate["dice_background"] == pytest.approx(1.0)
    assert aggregate["dice_flesh"] == pytest.approx(1.0)
    assert aggregate["dice_calyx"] == pytest.approx(1.0)
    assert aggregate["pixel_accuracy"] == pytest.approx(1.0)


def test_precision_recall_tp_fp_fn_and_per_image_metrics() -> None:
    target = torch.tensor([[0, 1, 1], [2, 2, 0]], dtype=torch.long)
    predicted = torch.tensor([[0, 1, 2], [2, 0, 1]], dtype=torch.long)
    confidence = torch.full((2, 3), 0.8)
    entropy = torch.full((2, 3), 0.2)

    row = per_image_metrics(predicted, target, confidence, entropy)

    assert row["flesh_tp"] == 1
    assert row["flesh_fp"] == 1
    assert row["flesh_fn"] == 1
    assert row["calyx_tp"] == 1
    assert row["calyx_fp"] == 1
    assert row["calyx_fn"] == 1
    assert row["flesh_precision"] == pytest.approx(0.5)
    assert row["flesh_recall"] == pytest.approx(0.5)
    assert row["calyx_precision"] == pytest.approx(0.5)
    assert row["calyx_recall"] == pytest.approx(0.5)
    assert row["iou_flesh"] == pytest.approx(1 / 3)
    assert row["dice_flesh"] == pytest.approx(0.5)


def test_per_image_metrics_are_not_substituted_for_aggregate_metrics() -> None:
    confidence = torch.ones((1, 2))
    entropy = torch.zeros((1, 2))
    first = per_image_metrics(
        torch.tensor([[1, 1]]),
        torch.tensor([[1, 1]]),
        confidence,
        entropy,
    )
    second = per_image_metrics(
        torch.tensor([[0, 0]]),
        torch.tensor([[1, 0]]),
        confidence,
        entropy,
    )
    aggregate = metrics_from_confusion_matrix(
        torch.tensor(
            [
                [1, 0, 0],
                [1, 2, 0],
                [0, 0, 0],
            ]
        )
    )

    per_image_average = (first["iou_flesh"] + second["iou_flesh"]) / 2
    assert per_image_average == pytest.approx(0.5)
    assert aggregate["iou_flesh"] == pytest.approx(2 / 3)
    assert aggregate["iou_flesh"] != pytest.approx(per_image_average)


def test_false_positive_and_false_negative_masks() -> None:
    target = torch.tensor([[0, 1, 2], [1, 2, 0]])
    predicted = torch.tensor([[1, 0, 2], [2, 0, 2]])

    masks = diagnostic_masks(predicted, target)

    assert masks["correct"].tolist() == [
        [False, False, True],
        [False, False, False],
    ]
    assert masks["flesh_false_positive"].tolist() == [
        [True, False, False],
        [False, False, False],
    ]
    assert masks["flesh_false_negative"].tolist() == [
        [False, True, False],
        [True, False, False],
    ]
    assert masks["calyx_false_positive"].tolist() == [
        [False, False, False],
        [True, False, True],
    ]
    assert masks["calyx_false_negative"].tolist() == [
        [False, False, False],
        [False, True, False],
    ]


def test_confidence_is_max_softmax_and_entropy_is_finite_unit_interval() -> None:
    logits = torch.tensor(
        [[[[2.0]], [[1.0]], [[-1.0]]]],
        dtype=torch.float32,
    )

    probabilities, confidence, entropy = confidence_and_entropy(logits)

    expected = torch.softmax(logits, dim=1)
    torch.testing.assert_close(probabilities, expected)
    torch.testing.assert_close(confidence, expected.max(dim=1).values)
    assert torch.all(torch.isfinite(entropy))
    assert torch.all((entropy >= 0) & (entropy <= 1))


def test_absent_classes_produce_nan_and_means_ignore_only_nan() -> None:
    matrix = torch.tensor([[4, 0, 0], [0, 0, 0], [0, 0, 0]])

    metrics = metrics_from_confusion_matrix(matrix)

    assert metrics["iou_background"] == pytest.approx(1.0)
    assert math.isnan(metrics["iou_flesh"])
    assert math.isnan(metrics["iou_calyx"])
    assert math.isnan(metrics["precision_flesh"])
    assert math.isnan(metrics["recall_calyx"])
    assert math.isnan(metrics["mean_foreground_iou"])

    one_present = torch.tensor([[1, 0, 0], [0, 2, 0], [0, 0, 0]])
    present_metrics = metrics_from_confusion_matrix(one_present)
    assert present_metrics["mean_foreground_iou"] == pytest.approx(1.0)
    assert present_metrics["mean_foreground_dice"] == pytest.approx(1.0)


def test_prediction_png_is_single_channel_and_contains_only_class_ids(
    tmp_path: Path,
) -> None:
    _, _, output_root, _ = _evaluate(tmp_path)

    for prediction_path in sorted((output_root / "predictions").glob("*.png")):
        with Image.open(prediction_path) as prediction:
            values = set(np.unique(np.asarray(prediction)).tolist())
            assert prediction.mode == "L"
            assert values <= {0, 1, 2}


def test_evaluation_preprocessing_uses_bilinear_image_and_nearest_mask(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    image = np.asarray(
        [
            [[0, 0, 0], [255, 255, 255]],
            [[255, 255, 255], [0, 0, 0]],
        ],
        dtype=np.uint8,
    )
    mask = np.asarray([[0, 1], [2, 0]], dtype=np.uint8)
    (dataset_root / "val" / "images").mkdir(parents=True)
    (dataset_root / "val" / "masks").mkdir(parents=True)
    Image.fromarray(image).save(dataset_root / "val" / "images" / "sample.png")
    Image.fromarray(mask).save(dataset_root / "val" / "masks" / "sample.png")

    dataset = create_evaluation_dataset(
        dataset_root,
        split="val",
        height=4,
        width=4,
    )
    sample = dataset[0]

    assert sample["image"].shape == (3, 4, 4)
    assert sample["image"][:, 1, 1].tolist() == pytest.approx([0.375] * 3)
    assert sample["mask"].dtype == torch.long
    assert sample["mask"].tolist() == [
        [0, 0, 1, 1],
        [0, 0, 1, 1],
        [2, 2, 0, 0],
        [2, 2, 0, 0],
    ]


def test_model_eval_and_inference_mode_are_used(tmp_path: Path) -> None:
    dataset_root = tmp_path / "dataset"
    checkpoint = tmp_path / "run" / "best_checkpoint.pt"
    output_root = tmp_path / "output"
    _make_dataset(dataset_root)
    model = TinyEvaluationModel()
    _write_checkpoint(checkpoint, model=model)
    restored = TinyEvaluationModel()

    evaluate_segmentation_checkpoint(
        dataset_root,
        checkpoint,
        output_root,
        device="cpu",
        model_factory=lambda: restored,
    )

    assert restored.was_training_during_forward is False
    assert restored.grad_enabled_during_forward is False
    assert restored.training is False


def test_checkpoint_class_mismatch_and_strict_state_loading_are_rejected(
    tmp_path: Path,
) -> None:
    mismatch = tmp_path / "mismatch.pt"
    _write_checkpoint(
        mismatch,
        class_mapping={"background": 0, "Flesh": 2, "Calyx": 1},
    )
    with pytest.raises(ValueError, match="class_mapping is incompatible"):
        load_segmentation_checkpoint(
            mismatch,
            model_factory=_tiny_model_factory,
        )

    invalid_state = tmp_path / "invalid-state.pt"
    _write_checkpoint(invalid_state)
    checkpoint = torch.load(invalid_state, weights_only=True)
    checkpoint["model_state_dict"] = {}
    torch.save(checkpoint, invalid_state)
    with pytest.raises(ValueError, match="Strict model-state loading failed"):
        load_segmentation_checkpoint(
            invalid_state,
            model_factory=_tiny_model_factory,
        )


def test_checkpoint_input_size_controls_evaluation_and_shape_mismatch_is_clear(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    checkpoint = tmp_path / "run" / "best_checkpoint.pt"
    output_root = tmp_path / "output"
    _make_dataset(dataset_root)
    _write_checkpoint(checkpoint, height=4, width=6)

    manifest = evaluate_segmentation_checkpoint(
        dataset_root,
        checkpoint,
        output_root,
        device="cpu",
        model_factory=_tiny_model_factory,
    )

    assert manifest["evaluation_height"] == 4
    assert manifest["evaluation_width"] == 6
    with Image.open(output_root / "predictions" / "a-sample.png") as prediction:
        assert prediction.size == (6, 4)

    wrong_checkpoint = tmp_path / "wrong" / "best_checkpoint.pt"
    _write_checkpoint(wrong_checkpoint, model=WrongShapeModel())
    with pytest.raises(ValueError, match="checkpoint evaluation size"):
        evaluate_segmentation_checkpoint(
            dataset_root,
            wrong_checkpoint,
            tmp_path / "wrong-output",
            device="cpu",
            model_factory=WrongShapeModel,
        )


def test_overwrite_protection_and_sanitized_relative_metadata(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    checkpoint = tmp_path / "run" / "best_checkpoint.pt"
    output_root = tmp_path / "output"
    _make_dataset(dataset_root)
    _write_checkpoint(checkpoint)
    output_root.mkdir()
    marker = output_root / "keep.txt"
    marker.write_text("keep", encoding="utf-8")

    with pytest.raises(FileExistsError, match="--overwrite"):
        evaluate_segmentation_checkpoint(
            dataset_root,
            checkpoint,
            output_root,
            model_factory=_tiny_model_factory,
        )
    assert marker.read_text(encoding="utf-8") == "keep"

    evaluate_segmentation_checkpoint(
        dataset_root,
        checkpoint,
        output_root,
        overwrite=True,
        model_factory=_tiny_model_factory,
    )
    serialized = (output_root / "evaluation_manifest.json").read_text(encoding="utf-8")
    assert str(tmp_path.resolve()) not in serialized
    manifest = json.loads(serialized)
    assert manifest["artifacts"]["aggregate_metrics"] == "aggregate_metrics.json"
    assert manifest["samples"][0]["prediction_path"].startswith("predictions/")


def test_inputs_unchanged_csv_deterministic_and_visualizations_headless(
    tmp_path: Path,
) -> None:
    dataset_root = tmp_path / "dataset"
    checkpoint = tmp_path / "run" / "best_checkpoint.pt"
    _make_dataset(dataset_root)
    _write_checkpoint(checkpoint)
    before_dataset = _snapshot(dataset_root)
    before_checkpoint = checkpoint.read_bytes()

    evaluate_segmentation_checkpoint(
        dataset_root,
        checkpoint,
        tmp_path / "first",
        model_factory=_tiny_model_factory,
    )
    evaluate_segmentation_checkpoint(
        dataset_root,
        checkpoint,
        tmp_path / "second",
        model_factory=_tiny_model_factory,
    )

    assert _snapshot(dataset_root) == before_dataset
    assert checkpoint.read_bytes() == before_checkpoint
    assert (tmp_path / "first" / "per_image_metrics.csv").read_bytes() == (
        tmp_path / "second" / "per_image_metrics.csv"
    ).read_bytes()
    visualization_paths = sorted((tmp_path / "first" / "visualizations").glob("*.png"))
    assert len(visualization_paths) == 2
    with Image.open(visualization_paths[0]) as visualization:
        assert visualization.format == "PNG"
        assert visualization.mode == "RGB"


def test_history_plots_created_when_present_and_absence_recorded(
    tmp_path: Path,
) -> None:
    _, _, with_history, manifest = _evaluate(
        tmp_path / "present",
        write_history=True,
    )
    assert manifest["training_history_available"] is True
    assert (with_history / "loss_curves.png").is_file()
    assert (with_history / "iou_curves.png").is_file()

    _, _, without_history, no_history_manifest = _evaluate(tmp_path / "absent")
    assert no_history_manifest["training_history_available"] is False
    assert no_history_manifest["artifacts"]["loss_curves"] is None
    assert not (without_history / "loss_curves.png").exists()


@pytest.mark.parametrize(
    "history_contents",
    (
        "",
        "epoch,train_total_loss,val_total_loss,val_iou_flesh,"
        "val_iou_calyx,val_mean_foreground_iou\n",
    ),
    ids=("empty", "header-only"),
)
def test_empty_or_header_only_history_is_unavailable(
    tmp_path: Path,
    history_contents: str,
) -> None:
    dataset_root = tmp_path / "dataset"
    checkpoint = tmp_path / "training-run" / "best_checkpoint.pt"
    output_root = tmp_path / "evaluation"
    _make_dataset(dataset_root)
    _write_checkpoint(checkpoint)
    (checkpoint.parent / "history.csv").write_text(
        history_contents,
        encoding="utf-8",
    )

    manifest = evaluate_segmentation_checkpoint(
        dataset_root,
        checkpoint,
        output_root,
        device="cpu",
        model_factory=_tiny_model_factory,
    )

    assert manifest["training_history_available"] is False
    assert manifest["artifacts"]["loss_curves"] is None
    assert manifest["artifacts"]["iou_curves"] is None
    assert (output_root / "aggregate_metrics.json").is_file()
    assert list((output_root / "predictions").glob("*.png"))
    assert not (output_root / "loss_curves.png").exists()
    assert not (output_root / "iou_curves.png").exists()
