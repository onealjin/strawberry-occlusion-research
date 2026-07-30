"""Evaluate a trained three-class segmentation checkpoint."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn
from torch.utils.data import DataLoader

from strawberry_occlusion.data import SegmentationDataset
from strawberry_occlusion.metrics import (
    confusion_matrix,
    dice_score,
    intersection_over_union,
    mean_ignore_nan,
)
from strawberry_occlusion.models import ResNet34UNet
from strawberry_occlusion.training.segmentation import (
    CLASS_MAPPING,
    MODEL_NAME,
    NUM_CLASSES,
    preprocess_class_mask,
    preprocess_rgb_image,
    resolve_device,
)
from strawberry_occlusion.visualization.segmentation import (
    DEFAULT_ALPHA,
    PALETTE,
    colorize_mask,
    overlay_mask,
)


PathLike = str | Path
ModelFactory = Callable[[], nn.Module]

SUPPORTED_SPLITS = ("train", "val")
TITLE_HEIGHT = 24
LEGEND_HEIGHT = 44
PANEL_TITLES = (
    "Evaluation RGB",
    "Ground truth",
    "Prediction",
    "Prediction overlay",
    "Combined errors",
    "Maximum confidence",
    "Normalized entropy",
    "Calyx errors",
)
ERROR_PALETTE = {
    "correct": (0, 0, 0),
    "flesh_false_positive": (255, 0, 255),
    "flesh_false_negative": (0, 255, 255),
    "calyx_false_positive": (255, 255, 0),
    "calyx_false_negative": (0, 96, 255),
}
PER_IMAGE_FIELDS = (
    "sample_id",
    "iou_background",
    "iou_flesh",
    "iou_calyx",
    "dice_background",
    "dice_flesh",
    "dice_calyx",
    "flesh_precision",
    "flesh_recall",
    "calyx_precision",
    "calyx_recall",
    "mean_foreground_iou",
    "mean_foreground_dice",
    "flesh_tp",
    "flesh_fp",
    "flesh_fn",
    "calyx_tp",
    "calyx_fp",
    "calyx_fn",
    "mean_maximum_softmax_confidence",
    "mean_normalized_entropy",
    "percent_confidence_below_0_50",
    "percent_confidence_below_0_75",
    "prediction_path",
    "visualization_path",
)


def create_evaluation_dataset(
    dataset_root: PathLike,
    *,
    split: str = "val",
    height: int,
    width: int,
) -> SegmentationDataset:
    """Build and eagerly validate one fixed dataset split."""

    if split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"split must be one of {', '.join(SUPPORTED_SPLITS)}, got {split!r}"
        )
    root = Path(dataset_root)
    image_transform = partial(preprocess_rgb_image, height=height, width=width)
    mask_transform = partial(preprocess_class_mask, height=height, width=width)
    dataset = SegmentationDataset(
        root / split / "images",
        root / split / "masks",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    for index in range(len(dataset)):
        dataset[index]
    return dataset


def load_segmentation_checkpoint(
    checkpoint_path: PathLike,
    *,
    device: str | torch.device = "cpu",
    model_factory: ModelFactory | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    """Safely validate a checkpoint and strictly restore its model."""

    path = Path(checkpoint_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {path}")
    resolved_device = resolve_device(device)
    checkpoint = torch.load(path, map_location=resolved_device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")

    required = {
        "model_state_dict",
        "completed_epoch",
        "best_mean_foreground_iou",
        "class_mapping",
        "input_height",
        "input_width",
        "model_name",
    }
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {', '.join(missing)}")
    if checkpoint["class_mapping"] != CLASS_MAPPING:
        raise ValueError(
            "Checkpoint class_mapping is incompatible: "
            f"expected {CLASS_MAPPING!r}, got {checkpoint['class_mapping']!r}"
        )

    height = _positive_integer(checkpoint["input_height"], name="input_height")
    width = _positive_integer(checkpoint["input_width"], name="input_width")
    completed_epoch = checkpoint["completed_epoch"]
    if (
        isinstance(completed_epoch, bool)
        or not isinstance(completed_epoch, int)
        or completed_epoch < 0
    ):
        raise ValueError("Checkpoint completed_epoch must be a non-negative integer")
    best_metric = checkpoint["best_mean_foreground_iou"]
    if best_metric is not None and (
        isinstance(best_metric, bool)
        or not isinstance(best_metric, (int, float))
        or not math.isfinite(best_metric)
    ):
        raise ValueError("Checkpoint best_mean_foreground_iou must be finite or None")
    model_name = checkpoint["model_name"]
    if not isinstance(model_name, str) or not model_name:
        raise ValueError("Checkpoint model_name must be a non-empty string")
    if not isinstance(checkpoint["model_state_dict"], Mapping):
        raise ValueError("Checkpoint model_state_dict must be a mapping")

    if model_factory is None:
        if model_name != MODEL_NAME:
            raise ValueError(
                f"Unsupported checkpoint model_name {model_name!r}; "
                f"expected {MODEL_NAME!r}"
            )
        model = ResNet34UNet(num_classes=NUM_CLASSES, encoder_weights=None)
    else:
        model = model_factory()
        if not isinstance(model, nn.Module):
            raise TypeError("model_factory must return a torch.nn.Module")
        if model.__class__.__name__ != model_name:
            raise ValueError(
                "Checkpoint model_name does not match injected model: "
                f"checkpoint={model_name!r}, model={model.__class__.__name__!r}"
            )

    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    except (RuntimeError, TypeError, ValueError) as error:
        raise ValueError("Strict model-state loading failed") from error

    checkpoint["input_height"] = height
    checkpoint["input_width"] = width
    model.to(resolved_device)
    model.eval()
    return model, checkpoint


def class_precision_recall(
    matrix: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return per-class precision and recall with NaN for zero denominators."""

    matrix = matrix.to(torch.float64)
    if matrix.ndim != 2 or matrix.shape != (NUM_CLASSES, NUM_CLASSES):
        raise ValueError(
            f"confusion matrix must have shape [{NUM_CLASSES}, {NUM_CLASSES}], "
            f"got {list(matrix.shape)}"
        )
    if not torch.all(torch.isfinite(matrix)) or torch.any(matrix < 0):
        raise ValueError("confusion matrix values must be finite and non-negative")

    true_positives = matrix.diagonal()
    predicted_counts = matrix.sum(dim=0)
    target_counts = matrix.sum(dim=1)
    precision = torch.full_like(true_positives, torch.nan)
    recall = torch.full_like(true_positives, torch.nan)
    precision[predicted_counts > 0] = (
        true_positives[predicted_counts > 0] / predicted_counts[predicted_counts > 0]
    )
    recall[target_counts > 0] = (
        true_positives[target_counts > 0] / target_counts[target_counts > 0]
    )
    return precision, recall


def metrics_from_confusion_matrix(matrix: torch.Tensor) -> dict[str, float]:
    """Calculate aggregate metrics from one target-row confusion matrix."""

    iou = intersection_over_union(matrix)
    dice = dice_score(matrix)
    precision, recall = class_precision_recall(matrix)
    total = matrix.sum()
    pixel_accuracy = (
        float(matrix.diagonal().sum().item() / total.item())
        if total.item() > 0
        else math.nan
    )
    return {
        "iou_background": _as_float(iou[0]),
        "iou_flesh": _as_float(iou[1]),
        "iou_calyx": _as_float(iou[2]),
        "dice_background": _as_float(dice[0]),
        "dice_flesh": _as_float(dice[1]),
        "dice_calyx": _as_float(dice[2]),
        "precision_background": _as_float(precision[0]),
        "precision_flesh": _as_float(precision[1]),
        "precision_calyx": _as_float(precision[2]),
        "recall_background": _as_float(recall[0]),
        "recall_flesh": _as_float(recall[1]),
        "recall_calyx": _as_float(recall[2]),
        "mean_foreground_iou": _as_float(mean_ignore_nan(iou[1:])),
        "mean_foreground_dice": _as_float(mean_ignore_nan(dice[1:])),
        "pixel_accuracy": pixel_accuracy,
    }


def confidence_and_entropy(
    logits: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return softmax probabilities, maximum confidence, and normalized entropy."""

    if not isinstance(logits, torch.Tensor) or logits.ndim != 4:
        raise ValueError("logits must be a tensor with shape [B, 3, H, W]")
    if logits.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"model must output exactly {NUM_CLASSES} classes, got {logits.shape[1]}"
        )
    probabilities = torch.softmax(logits.float(), dim=1)
    confidence = probabilities.max(dim=1).values
    safe_probabilities = probabilities.clamp_min(torch.finfo(probabilities.dtype).tiny)
    entropy = -(probabilities * safe_probabilities.log()).sum(dim=1)
    normalized_entropy = (entropy / math.log(NUM_CLASSES)).clamp(0.0, 1.0)
    for name, values in (
        ("confidence", confidence),
        ("normalized entropy", normalized_entropy),
    ):
        if not torch.all(torch.isfinite(values)):
            raise ValueError(f"{name} values must be finite")
        if torch.any(values < 0) or torch.any(values > 1):
            raise ValueError(f"{name} values must be in [0, 1]")
    return probabilities, confidence, normalized_entropy


def diagnostic_masks(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Derive correct, false-positive, and false-negative pixel masks."""

    if predicted.shape != target.shape or predicted.ndim != 2:
        raise ValueError("predicted and target must have the same [H, W] shape")
    return {
        "correct": predicted == target,
        "flesh_false_positive": (predicted == 1) & (target != 1),
        "flesh_false_negative": (predicted != 1) & (target == 1),
        "calyx_false_positive": (predicted == 2) & (target != 2),
        "calyx_false_negative": (predicted != 2) & (target == 2),
    }


def per_image_metrics(
    predicted: torch.Tensor,
    target: torch.Tensor,
    confidence: torch.Tensor,
    normalized_entropy: torch.Tensor,
) -> dict[str, float | int]:
    """Calculate metrics and uncertainty summaries for one image."""

    if (
        confidence.shape != predicted.shape
        or normalized_entropy.shape != predicted.shape
    ):
        raise ValueError("confidence and entropy must match the prediction shape")
    matrix = confusion_matrix(predicted, target, NUM_CLASSES).cpu()
    metrics = metrics_from_confusion_matrix(matrix)
    result: dict[str, float | int] = {
        "iou_background": metrics["iou_background"],
        "iou_flesh": metrics["iou_flesh"],
        "iou_calyx": metrics["iou_calyx"],
        "dice_background": metrics["dice_background"],
        "dice_flesh": metrics["dice_flesh"],
        "dice_calyx": metrics["dice_calyx"],
        "flesh_precision": metrics["precision_flesh"],
        "flesh_recall": metrics["recall_flesh"],
        "calyx_precision": metrics["precision_calyx"],
        "calyx_recall": metrics["recall_calyx"],
        "mean_foreground_iou": metrics["mean_foreground_iou"],
        "mean_foreground_dice": metrics["mean_foreground_dice"],
        "mean_maximum_softmax_confidence": float(confidence.mean().item()),
        "mean_normalized_entropy": float(normalized_entropy.mean().item()),
        "percent_confidence_below_0_50": float(
            (confidence < 0.50).to(torch.float64).mean().item() * 100.0
        ),
        "percent_confidence_below_0_75": float(
            (confidence < 0.75).to(torch.float64).mean().item() * 100.0
        ),
    }
    for class_name, class_id in (("flesh", 1), ("calyx", 2)):
        true_positive = int(matrix[class_id, class_id].item())
        result[f"{class_name}_tp"] = true_positive
        result[f"{class_name}_fp"] = int(
            matrix[:, class_id].sum().item() - true_positive
        )
        result[f"{class_name}_fn"] = int(
            matrix[class_id, :].sum().item() - true_positive
        )
    return result


def create_diagnostic_visualization(
    image: Image.Image | np.ndarray,
    target: np.ndarray,
    predicted: np.ndarray,
    confidence: np.ndarray,
    normalized_entropy: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> Image.Image:
    """Create a deterministic titled two-by-four diagnostic image."""

    image_array = _rgb_array(image)
    target_array = _class_mask_array(target, name="target")
    predicted_array = _class_mask_array(predicted, name="prediction")
    if target_array.shape != predicted_array.shape:
        raise ValueError("target and prediction dimensions must match")
    if image_array.shape[:2] != target_array.shape:
        raise ValueError("image and mask dimensions must match")
    confidence_array = _unit_interval_array(confidence, name="confidence")
    entropy_array = _unit_interval_array(normalized_entropy, name="normalized entropy")
    if confidence_array.shape != target_array.shape:
        raise ValueError("confidence dimensions must match the masks")
    if entropy_array.shape != target_array.shape:
        raise ValueError("entropy dimensions must match the masks")

    masks = diagnostic_masks(
        torch.from_numpy(predicted_array.astype(np.int64)),
        torch.from_numpy(target_array.astype(np.int64)),
    )
    error_map = np.zeros((*target_array.shape, 3), dtype=np.uint8)
    for key in (
        "flesh_false_positive",
        "flesh_false_negative",
        "calyx_false_positive",
        "calyx_false_negative",
    ):
        error_map[masks[key].numpy()] = ERROR_PALETTE[key]

    calyx_errors = np.zeros_like(error_map)
    correct_calyx = (target_array == 2) & (predicted_array == 2)
    calyx_errors[correct_calyx] = PALETTE[2]
    calyx_errors[masks["calyx_false_positive"].numpy()] = ERROR_PALETTE[
        "calyx_false_positive"
    ]
    calyx_errors[masks["calyx_false_negative"].numpy()] = ERROR_PALETTE[
        "calyx_false_negative"
    ]
    confidence_map = _grayscale_rgb(confidence_array)
    entropy_map = _grayscale_rgb(entropy_array)
    panels = (
        image_array,
        colorize_mask(target_array),
        colorize_mask(predicted_array),
        overlay_mask(image_array, predicted_array, alpha=alpha),
        error_map,
        confidence_map,
        entropy_map,
        calyx_errors,
    )

    height, width = target_array.shape
    canvas = Image.new(
        "RGB",
        (4 * width, 2 * (height + TITLE_HEIGHT) + LEGEND_HEIGHT),
        color=(0, 0, 0),
    )
    for index, (title, panel_array) in enumerate(
        zip(PANEL_TITLES, panels, strict=True)
    ):
        panel = Image.new("RGB", (width, height + TITLE_HEIGHT), color=(0, 0, 0))
        ImageDraw.Draw(panel).text((4, 5), title, fill=(255, 255, 255))
        panel.paste(Image.fromarray(panel_array), (0, TITLE_HEIGHT))
        column = index % 4
        row = index // 4
        canvas.paste(panel, (column * width, row * (height + TITLE_HEIGHT)))

    _draw_error_legend(
        canvas,
        top=2 * (height + TITLE_HEIGHT),
        width=4 * width,
    )
    return canvas


def evaluate_segmentation_checkpoint(
    dataset_root: PathLike,
    checkpoint_path: PathLike,
    output_root: PathLike,
    *,
    split: str = "val",
    device: str | torch.device = "auto",
    batch_size: int = 1,
    num_workers: int = 0,
    overwrite: bool = False,
    model_factory: ModelFactory | None = None,
    alpha: float = DEFAULT_ALPHA,
) -> dict[str, Any]:
    """Evaluate one checkpoint and write sanitized diagnostic artifacts."""

    batch_size = _positive_integer(batch_size, name="batch_size")
    if isinstance(num_workers, bool) or not isinstance(num_workers, int):
        raise ValueError("num_workers must be a non-negative integer")
    if num_workers < 0:
        raise ValueError("num_workers must be a non-negative integer")
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise ValueError("alpha must be a finite number in [0, 1]")
    alpha = float(alpha)
    if not math.isfinite(alpha) or not 0 <= alpha <= 1:
        raise ValueError("alpha must be a finite number in [0, 1]")

    source_root = Path(dataset_root)
    checkpoint_file = Path(checkpoint_path)
    destination_root = Path(output_root)
    _validate_output_destination(
        source_root,
        checkpoint_file,
        destination_root,
        overwrite=overwrite,
    )
    resolved_device = resolve_device(device)
    model, checkpoint = load_segmentation_checkpoint(
        checkpoint_file,
        device=resolved_device,
        model_factory=model_factory,
    )
    height = checkpoint["input_height"]
    width = checkpoint["input_width"]
    dataset = create_evaluation_dataset(
        source_root,
        split=split,
        height=height,
        width=width,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.staging-",
            dir=destination_root.parent,
        )
    )
    try:
        (staging_root / "predictions").mkdir()
        (staging_root / "visualizations").mkdir()
        global_matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)
        rows: list[dict[str, Any]] = []
        model.eval()
        with torch.inference_mode():
            for batch in loader:
                images = batch["image"].to(resolved_device)
                targets = batch["mask"].to(resolved_device)
                logits = _model_logits(model(images))
                _validate_logits(logits, images)
                _, confidence, entropy = confidence_and_entropy(logits)
                predictions = logits.argmax(dim=1)
                global_matrix += confusion_matrix(
                    predictions, targets, NUM_CLASSES
                ).cpu()

                for index, image_path in enumerate(batch["image_path"]):
                    sample_id = Path(image_path).stem
                    prediction = predictions[index].cpu()
                    target = targets[index].cpu()
                    sample_confidence = confidence[index].cpu()
                    sample_entropy = entropy[index].cpu()
                    prediction_path = Path("predictions") / f"{sample_id}.png"
                    visualization_path = Path("visualizations") / f"{sample_id}.png"
                    _save_prediction(
                        prediction,
                        staging_root / prediction_path,
                    )
                    sample_metrics = per_image_metrics(
                        prediction,
                        target,
                        sample_confidence,
                        sample_entropy,
                    )
                    rgb = _image_tensor_to_rgb(images[index].cpu())
                    visualization = create_diagnostic_visualization(
                        rgb,
                        target.numpy(),
                        prediction.numpy(),
                        sample_confidence.numpy(),
                        sample_entropy.numpy(),
                        alpha=alpha,
                    )
                    visualization.save(
                        staging_root / visualization_path,
                        format="PNG",
                    )
                    rows.append(
                        {
                            "sample_id": sample_id,
                            **sample_metrics,
                            "prediction_path": prediction_path.as_posix(),
                            "visualization_path": visualization_path.as_posix(),
                        }
                    )

        aggregate_metrics = metrics_from_confusion_matrix(global_matrix)
        _write_json(
            staging_root / "aggregate_metrics.json",
            aggregate_metrics,
        )
        _write_per_image_csv(staging_root / "per_image_metrics.csv", rows)
        _write_confusion_matrix_csv(
            staging_root / "confusion_matrix.csv",
            global_matrix,
        )
        history_available = _write_history_plots(
            checkpoint_file.parent / "history.csv",
            staging_root,
            checkpoint_epoch=checkpoint["completed_epoch"],
        )
        artifacts: dict[str, str | None] = {
            "aggregate_metrics": "aggregate_metrics.json",
            "per_image_metrics": "per_image_metrics.csv",
            "confusion_matrix": "confusion_matrix.csv",
            "predictions": "predictions",
            "visualizations": "visualizations",
            "loss_curves": "loss_curves.png" if history_available else None,
            "iou_curves": "iou_curves.png" if history_available else None,
        }
        manifest: dict[str, Any] = {
            "dataset_name": source_root.name,
            "evaluated_split": split,
            "sample_count": len(dataset),
            "class_mapping": dict(CLASS_MAPPING),
            "class_palette": {
                name: list(PALETTE[class_id])
                for name, class_id in CLASS_MAPPING.items()
            },
            "error_palette": {
                name: list(color) for name, color in ERROR_PALETTE.items()
            },
            "overlay_alpha": alpha,
            "checkpoint_model_name": checkpoint["model_name"],
            "checkpoint_epoch": checkpoint["completed_epoch"],
            "checkpoint_stored_best_metric": checkpoint["best_mean_foreground_iou"],
            "evaluation_height": height,
            "evaluation_width": width,
            "device_type": resolved_device.type,
            "training_history_available": history_available,
            "artifacts": artifacts,
            "aggregate_metrics": aggregate_metrics,
            "samples": [
                {
                    "sample_id": row["sample_id"],
                    "prediction_path": row["prediction_path"],
                    "visualization_path": row["visualization_path"],
                }
                for row in rows
            ],
        }
        _write_json(staging_root / "evaluation_manifest.json", manifest)
        _install_staged_output(
            staging_root,
            destination_root,
            overwrite=overwrite,
        )
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    return _json_ready(manifest)


def _model_logits(output: Any) -> torch.Tensor:
    if isinstance(output, dict):
        if "logits" not in output:
            raise KeyError("model output dictionary must contain a 'logits' key")
        output = output["logits"]
    if not isinstance(output, torch.Tensor):
        raise TypeError("model output must be a logits tensor")
    return output


def _validate_logits(logits: torch.Tensor, images: torch.Tensor) -> None:
    expected = (images.shape[0], NUM_CLASSES, images.shape[2], images.shape[3])
    if tuple(logits.shape) != expected:
        raise ValueError(
            "model output must have shape "
            f"{list(expected)} at checkpoint evaluation size, "
            f"got {list(logits.shape)}"
        )


def _save_prediction(prediction: torch.Tensor, path: Path) -> None:
    array = prediction.numpy()
    array = _class_mask_array(array, name="prediction")
    output = Image.fromarray(array, mode="L")
    output.save(path, format="PNG")


def _validate_output_destination(
    dataset_root: Path,
    checkpoint_path: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> None:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint does not exist: {checkpoint_path}")
    dataset_resolved = dataset_root.resolve()
    checkpoint_resolved = checkpoint_path.resolve()
    output_resolved = output_root.resolve()
    if _paths_overlap(dataset_resolved, output_resolved):
        raise ValueError("Dataset and output roots must not overlap")
    if (
        output_resolved == checkpoint_resolved
        or output_resolved in checkpoint_resolved.parents
    ):
        raise ValueError("Output root must not contain the checkpoint")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. "
                "Pass overwrite=True or --overwrite to replace it."
            )
        if output_root.is_symlink() or not output_root.is_dir():
            raise ValueError("Existing output root must be a non-symlink directory")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _install_staged_output(
    staging_root: Path,
    destination_root: Path,
    *,
    overwrite: bool,
) -> None:
    if destination_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root appeared during evaluation: {destination_root}"
            )
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise ValueError("Existing output root must be a non-symlink directory")
        shutil.rmtree(destination_root)
    staging_root.replace(destination_root)


def _write_per_image_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=PER_IMAGE_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _csv_value(row[field]) for field in PER_IMAGE_FIELDS}
            )


def _write_confusion_matrix_csv(path: Path, matrix: torch.Tensor) -> None:
    class_names = tuple(CLASS_MAPPING)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.writer(output_file, lineterminator="\n")
        writer.writerow(["target/prediction", *class_names])
        for index, class_name in enumerate(class_names):
            writer.writerow(
                [class_name, *(int(value) for value in matrix[index].tolist())]
            )


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            _json_ready(value),
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _json_ready(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return "nan"
        return format(value, ".12g")
    return value


def _write_history_plots(
    history_path: Path,
    output_root: Path,
    *,
    checkpoint_epoch: int,
) -> bool:
    if not history_path.is_file():
        return False
    with history_path.open("r", encoding="utf-8", newline="") as history_file:
        reader = csv.DictReader(history_file)
        rows = list(reader)
    if not rows:
        return False
    required = {
        "epoch",
        "train_total_loss",
        "val_total_loss",
        "val_iou_flesh",
        "val_iou_calyx",
        "val_mean_foreground_iou",
    }
    fieldnames = set(reader.fieldnames or ())
    if not required.issubset(fieldnames):
        missing = sorted(required - fieldnames)
        raise ValueError(
            f"history.csv is missing required columns: {', '.join(missing)}"
        )

    epochs = [_history_number(row, "epoch") for row in rows]
    _draw_curve_plot(
        output_root / "loss_curves.png",
        title="Training and validation total loss",
        epochs=epochs,
        series={
            "Train loss": [_history_number(row, "train_total_loss") for row in rows],
            "Validation loss": [_history_number(row, "val_total_loss") for row in rows],
        },
        checkpoint_epoch=checkpoint_epoch,
    )
    _draw_curve_plot(
        output_root / "iou_curves.png",
        title="Validation foreground IoU",
        epochs=epochs,
        series={
            "Flesh IoU": [_history_number(row, "val_iou_flesh") for row in rows],
            "Calyx IoU": [_history_number(row, "val_iou_calyx") for row in rows],
            "Mean foreground IoU": [
                _history_number(row, "val_mean_foreground_iou") for row in rows
            ],
        },
        checkpoint_epoch=checkpoint_epoch,
        fixed_range=(0.0, 1.0),
    )
    return True


def _history_number(row: Mapping[str, str], field: str) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"history.csv contains an invalid {field!r} value") from error
    return value


def _draw_curve_plot(
    path: Path,
    *,
    title: str,
    epochs: list[float],
    series: Mapping[str, list[float]],
    checkpoint_epoch: int,
    fixed_range: tuple[float, float] | None = None,
) -> None:
    canvas_width, canvas_height = 800, 480
    left, top, right, bottom = 72, 54, 30, 60
    plot_width = canvas_width - left - right
    plot_height = canvas_height - top - bottom
    image = Image.new("RGB", (canvas_width, canvas_height), color=(255, 255, 255))
    draw = ImageDraw.Draw(image)
    draw.text((left, 16), title, fill=(0, 0, 0))
    draw.line((left, top, left, top + plot_height), fill=(0, 0, 0), width=1)
    draw.line(
        (left, top + plot_height, left + plot_width, top + plot_height),
        fill=(0, 0, 0),
        width=1,
    )
    finite_epochs = [epoch for epoch in epochs if math.isfinite(epoch)]
    all_values = [
        value for values in series.values() for value in values if math.isfinite(value)
    ]
    if finite_epochs:
        x_min, x_max = min(finite_epochs), max(finite_epochs)
    else:
        x_min, x_max = 0.0, 1.0
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if fixed_range is not None:
        y_min, y_max = fixed_range
    elif all_values:
        y_min, y_max = min(all_values), max(all_values)
        if y_min == y_max:
            padding = max(abs(y_min) * 0.05, 0.05)
            y_min -= padding
            y_max += padding
    else:
        y_min, y_max = 0.0, 1.0

    def point(epoch: float, value: float) -> tuple[int, int]:
        x = left + round((epoch - x_min) / (x_max - x_min) * plot_width)
        y = top + plot_height - round((value - y_min) / (y_max - y_min) * plot_height)
        return x, y

    checkpoint_x = point(float(checkpoint_epoch), y_min)[0]
    if left <= checkpoint_x <= left + plot_width:
        draw.line(
            (checkpoint_x, top, checkpoint_x, top + plot_height),
            fill=(128, 128, 128),
            width=1,
        )
        draw.text(
            (min(checkpoint_x + 3, canvas_width - 100), top + 3),
            f"checkpoint {checkpoint_epoch}",
            fill=(80, 80, 80),
        )

    colors = ((210, 30, 30), (30, 90, 210), (20, 150, 70))
    for series_index, (label, values) in enumerate(series.items()):
        color = colors[series_index % len(colors)]
        points = [
            point(epoch, value)
            for epoch, value in zip(epochs, values, strict=True)
            if math.isfinite(epoch) and math.isfinite(value)
        ]
        if len(points) > 1:
            draw.line(points, fill=color, width=2)
        for x, y in points:
            draw.ellipse((x - 2, y - 2, x + 2, y + 2), fill=color)
        legend_x = left + series_index * 190
        draw.line(
            (legend_x, canvas_height - 25, legend_x + 20, canvas_height - 25),
            fill=color,
            width=2,
        )
        draw.text((legend_x + 25, canvas_height - 32), label, fill=(0, 0, 0))

    draw.text((8, top - 6), format(y_max, ".3g"), fill=(0, 0, 0))
    draw.text((8, top + plot_height - 6), format(y_min, ".3g"), fill=(0, 0, 0))
    draw.text((left, top + plot_height + 8), format(x_min, ".3g"), fill=(0, 0, 0))
    draw.text(
        (left + plot_width - 20, top + plot_height + 8),
        format(x_max, ".3g"),
        fill=(0, 0, 0),
    )
    image.save(path, format="PNG")


def _draw_error_legend(image: Image.Image, *, top: int, width: int) -> None:
    draw = ImageDraw.Draw(image)
    entries = (
        ("Flesh FP", ERROR_PALETTE["flesh_false_positive"]),
        ("Flesh FN", ERROR_PALETTE["flesh_false_negative"]),
        ("Calyx FP", ERROR_PALETTE["calyx_false_positive"]),
        ("Calyx FN", ERROR_PALETTE["calyx_false_negative"]),
    )
    x = 6
    for label, color in entries:
        draw.rectangle((x, top + 8, x + 14, top + 22), fill=color)
        draw.text((x + 19, top + 8), label, fill=(255, 255, 255))
        x += max(100, width // len(entries))


def _rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.asarray(image.convert("RGB"), dtype=np.uint8)
    array = np.asarray(image)
    if array.ndim != 3 or array.shape[2] != 3:
        raise ValueError("image must have shape [H, W, 3]")
    if not np.issubdtype(array.dtype, np.integer):
        raise ValueError("image values must use an integer dtype")
    if np.any(array < 0) or np.any(array > 255):
        raise ValueError("image values must be in [0, 255]")
    return array.astype(np.uint8, copy=True)


def _class_mask_array(mask: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"{name} must be a single-channel [H, W] mask")
    valid = np.isin(array, tuple(CLASS_MAPPING.values()))
    if not np.all(valid):
        invalid = np.unique(array[~valid]).tolist()
        raise ValueError(
            f"{name} must contain only class IDs 0, 1, and 2, found {invalid}"
        )
    return array.astype(np.uint8, copy=False)


def _unit_interval_array(values: np.ndarray, *, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError(f"{name} must have shape [H, W]")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} values must be finite")
    if np.any(array < 0) or np.any(array > 1):
        raise ValueError(f"{name} values must be in [0, 1]")
    return array


def _grayscale_rgb(values: np.ndarray) -> np.ndarray:
    gray = np.rint(values * 255.0).clip(0, 255).astype(np.uint8)
    return np.repeat(gray[..., None], 3, axis=2)


def _image_tensor_to_rgb(image: torch.Tensor) -> np.ndarray:
    array = (
        image.clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .permute(1, 2, 0)
        .numpy()
    )
    return array


def _positive_integer(value: Any, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _as_float(value: torch.Tensor) -> float:
    return float(value.item())


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate a trained three-class segmentation checkpoint."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="checkpoint to evaluate; canonical baseline evaluation uses best_checkpoint.pt",
    )
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="val")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory after successful evaluation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run checkpoint evaluation from the command line."""

    arguments = _build_argument_parser().parse_args(argv)
    manifest = evaluate_segmentation_checkpoint(
        arguments.dataset_root,
        arguments.checkpoint,
        arguments.output_root,
        split=arguments.split,
        device=arguments.device,
        batch_size=arguments.batch_size,
        num_workers=arguments.num_workers,
        overwrite=arguments.overwrite,
        alpha=arguments.alpha,
    )
    aggregate = manifest["aggregate_metrics"]
    print(
        f"Evaluated {manifest['sample_count']} {manifest['evaluated_split']} samples "
        f"at {manifest['evaluation_width']}x{manifest['evaluation_height']}"
    )
    print(
        "Mean foreground IoU "
        f"{_display_metric(aggregate['mean_foreground_iou'])}; "
        "mean foreground Dice "
        f"{_display_metric(aggregate['mean_foreground_dice'])}"
    )
    print(f"Artifacts written to {arguments.output_root}")
    return 0


def _display_metric(value: float | None) -> str:
    return "NaN" if value is None else f"{value:.6f}"


if __name__ == "__main__":
    raise SystemExit(main())
