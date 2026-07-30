"""Reproducible three-class semantic segmentation training baseline."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from collections.abc import Callable, Iterable, Mapping, Sequence
from contextlib import nullcontext
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as TF

from strawberry_occlusion.data import SegmentationDataset
from strawberry_occlusion.metrics import (
    confusion_matrix,
    dice_score,
    intersection_over_union,
    mean_ignore_nan,
)
from strawberry_occlusion.models import ResNet34UNet
from strawberry_occlusion.training.losses import CombinedSegmentationLoss


PathLike = str | Path
ModelFactory = Callable[[], nn.Module]

CLASS_MAPPING = {"background": 0, "Flesh": 1, "Calyx": 2}
MODEL_NAME = "ResNet34UNet"
NUM_CLASSES = 3
DEFAULT_HEIGHT = 640
DEFAULT_WIDTH = 768
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_WEIGHT_DECAY = 1e-2
DEFAULT_CROSS_ENTROPY_WEIGHT = 0.5
DEFAULT_DICE_WEIGHT = 0.5
LAST_CHECKPOINT_SENTINEL = "__last_checkpoint__"

HISTORY_FIELDS = (
    "epoch",
    "train_total_loss",
    "train_cross_entropy_loss",
    "train_dice_loss",
    "val_total_loss",
    "val_cross_entropy_loss",
    "val_dice_loss",
    "val_iou_background",
    "val_iou_flesh",
    "val_iou_calyx",
    "val_dice_background",
    "val_dice_flesh",
    "val_dice_calyx",
    "val_mean_foreground_iou",
    "val_mean_foreground_dice",
)


def preprocess_rgb_image(
    image: Image.Image,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Convert an RGB image to float and resize it with bilinear interpolation."""

    height, width = _validate_size(height, width)
    image_tensor = TF.pil_to_tensor(image.convert("RGB")).float().div(255.0)
    return TF.resize(
        image_tensor,
        size=[height, width],
        interpolation=InterpolationMode.BILINEAR,
        antialias=True,
    )


def preprocess_class_mask(
    mask: Image.Image,
    *,
    height: int,
    width: int,
) -> torch.Tensor:
    """Validate a class-index mask and resize it with nearest neighbour."""

    height, width = _validate_size(height, width)
    mask_array = np.array(mask, copy=True)
    if mask_array.ndim != 2:
        raise ValueError(
            f"Mask must be single-channel, got mode={mask.mode!r} "
            f"and shape={mask_array.shape}"
        )
    if not np.issubdtype(mask_array.dtype, np.integer):
        raise ValueError(
            f"Mask class IDs must use an integer dtype, got {mask_array.dtype}"
        )

    valid_pixels = np.isin(mask_array, tuple(CLASS_MAPPING.values()))
    if not np.all(valid_pixels):
        invalid_values = np.unique(mask_array[~valid_pixels]).tolist()
        raise ValueError(
            f"Mask must contain only class IDs 0, 1, and 2, found {invalid_values}"
        )

    mask_tensor = torch.from_numpy(mask_array.astype(np.int64, copy=False)).unsqueeze(0)
    resized = TF.resize(
        mask_tensor,
        size=[height, width],
        interpolation=InterpolationMode.NEAREST,
    )
    return resized.squeeze(0).long()


def create_training_datasets(
    dataset_root: PathLike,
    *,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
) -> tuple[SegmentationDataset, SegmentationDataset]:
    """Build and eagerly validate the fixed train and val datasets."""

    root = Path(dataset_root)
    height, width = _validate_size(height, width)
    image_transform = partial(preprocess_rgb_image, height=height, width=width)
    mask_transform = partial(preprocess_class_mask, height=height, width=width)

    train_dataset = SegmentationDataset(
        root / "train" / "images",
        root / "train" / "masks",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )
    val_dataset = SegmentationDataset(
        root / "val" / "images",
        root / "val" / "masks",
        image_transform=image_transform,
        mask_transform=mask_transform,
    )

    for dataset in (train_dataset, val_dataset):
        for index in range(len(dataset)):
            dataset[index]
    return train_dataset, val_dataset


def resolve_device(requested: str | torch.device = "auto") -> torch.device:
    """Resolve an explicit device or choose CUDA, MPS, then CPU."""

    if str(requested) == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        mps = getattr(torch.backends, "mps", None)
        if mps is not None and mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is not available")
    if device.type == "mps":
        mps = getattr(torch.backends, "mps", None)
        if mps is None or not mps.is_available():
            raise ValueError("MPS was requested but is not available")
    if device.type not in {"cpu", "cuda", "mps"}:
        raise ValueError(f"Unsupported device type {device.type!r}")
    return device


def seed_everything(seed: int) -> None:
    """Seed Python, NumPy, PyTorch, and available CUDA devices."""

    _validate_seed(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python and NumPy from the deterministic PyTorch worker seed."""

    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def train_segmentation_epoch(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    optimizer: torch.optim.Optimizer,
    loss_function: CombinedSegmentationLoss,
    device: torch.device,
    *,
    amp_enabled: bool = False,
    scaler: torch.amp.GradScaler | None = None,
) -> dict[str, float]:
    """Train one epoch and return sample-weighted component losses."""

    model.train()
    scaler = scaler or torch.amp.GradScaler("cuda", enabled=amp_enabled)
    totals = {"total": 0.0, "cross_entropy": 0.0, "dice": 0.0}
    sample_count = 0

    for batch in batches:
        images, targets = _batch_tensors(batch, device=device)
        optimizer.zero_grad(set_to_none=True)
        autocast_context = (
            torch.autocast(device_type="cuda") if amp_enabled else nullcontext()
        )
        with autocast_context:
            logits = _model_logits(model, images)
            total_loss, cross_entropy_loss, dice_loss = loss_function.components(
                logits,
                targets,
            )

        scaler.scale(total_loss).backward()
        scaler.step(optimizer)
        scaler.update()

        batch_size = images.shape[0]
        totals["total"] += float(total_loss.detach().cpu()) * batch_size
        totals["cross_entropy"] += float(cross_entropy_loss.detach().cpu()) * batch_size
        totals["dice"] += float(dice_loss.detach().cpu()) * batch_size
        sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Training loader must contain at least one sample")
    return {
        "train_total_loss": totals["total"] / sample_count,
        "train_cross_entropy_loss": totals["cross_entropy"] / sample_count,
        "train_dice_loss": totals["dice"] / sample_count,
    }


def validate_segmentation_epoch(
    model: nn.Module,
    batches: Iterable[Mapping[str, Any]],
    loss_function: CombinedSegmentationLoss,
    device: torch.device,
    *,
    amp_enabled: bool = False,
) -> tuple[dict[str, float], torch.Tensor]:
    """Validate one epoch using one confusion matrix for the complete split."""

    model.eval()
    totals = {"total": 0.0, "cross_entropy": 0.0, "dice": 0.0}
    sample_count = 0
    matrix = torch.zeros((NUM_CLASSES, NUM_CLASSES), dtype=torch.long)

    with torch.inference_mode():
        for batch in batches:
            images, targets = _batch_tensors(batch, device=device)
            autocast_context = (
                torch.autocast(device_type="cuda") if amp_enabled else nullcontext()
            )
            with autocast_context:
                logits = _model_logits(model, images)
                total_loss, cross_entropy_loss, dice_loss = loss_function.components(
                    logits,
                    targets,
                )

            predictions = logits.argmax(dim=1)
            matrix += confusion_matrix(
                predictions.cpu(),
                targets.cpu(),
                num_classes=NUM_CLASSES,
            )
            batch_size = images.shape[0]
            totals["total"] += float(total_loss.cpu()) * batch_size
            totals["cross_entropy"] += float(cross_entropy_loss.cpu()) * batch_size
            totals["dice"] += float(dice_loss.cpu()) * batch_size
            sample_count += batch_size

    if sample_count == 0:
        raise ValueError("Validation loader must contain at least one sample")

    iou = intersection_over_union(matrix)
    dice = dice_score(matrix)
    mean_foreground_iou = mean_ignore_nan(iou[1:])
    mean_foreground_dice = mean_ignore_nan(dice[1:])
    metrics = {
        "val_total_loss": totals["total"] / sample_count,
        "val_cross_entropy_loss": totals["cross_entropy"] / sample_count,
        "val_dice_loss": totals["dice"] / sample_count,
        "val_iou_background": float(iou[0]),
        "val_iou_flesh": float(iou[1]),
        "val_iou_calyx": float(iou[2]),
        "val_dice_background": float(dice[0]),
        "val_dice_flesh": float(dice[1]),
        "val_dice_calyx": float(dice[2]),
        "val_mean_foreground_iou": float(mean_foreground_iou),
        "val_mean_foreground_dice": float(mean_foreground_dice),
    }
    return metrics, matrix


def train_segmentation_baseline(
    dataset_root: PathLike,
    output_root: PathLike,
    *,
    epochs: int = 100,
    batch_size: int = 2,
    height: int = DEFAULT_HEIGHT,
    width: int = DEFAULT_WIDTH,
    learning_rate: float = DEFAULT_LEARNING_RATE,
    weight_decay: float = DEFAULT_WEIGHT_DECAY,
    cross_entropy_weight: float = DEFAULT_CROSS_ENTROPY_WEIGHT,
    dice_weight: float = DEFAULT_DICE_WEIGHT,
    seed: int = 42,
    num_workers: int = 0,
    device: str | torch.device = "auto",
    amp: bool = False,
    overwrite: bool = False,
    resume: bool | PathLike | None = None,
    model_factory: ModelFactory | None = None,
) -> dict[str, Any]:
    """Train or resume the reproducible real-only segmentation baseline."""

    _validate_training_arguments(
        epochs=epochs,
        batch_size=batch_size,
        height=height,
        width=width,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        seed=seed,
        num_workers=num_workers,
    )
    if overwrite and resume is not None and resume is not False:
        raise ValueError("--overwrite and --resume cannot be used together")

    dataset_path = Path(dataset_root)
    run_path = Path(output_root)
    if not dataset_path.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_path}")
    if _paths_overlap(dataset_path.resolve(), run_path.resolve()):
        raise ValueError("Dataset and output roots must not overlap")

    resume_path = _resolve_resume_path(run_path, resume)
    seed_everything(seed)
    resolved_device = resolve_device(device)
    amp_enabled = bool(amp and resolved_device.type == "cuda")
    train_dataset, val_dataset = create_training_datasets(
        dataset_path,
        height=height,
        width=width,
    )
    _prepare_output_root(
        run_path,
        overwrite=overwrite,
        resuming=resume_path is not None,
    )

    train_generator = torch.Generator()
    train_generator.manual_seed(seed)
    val_generator = torch.Generator()
    val_generator.manual_seed(seed + 1)

    model = (
        model_factory()
        if model_factory is not None
        else ResNet34UNet(num_classes=NUM_CLASSES, encoder_weights=None)
    )
    model_name = model.__class__.__name__
    model.to(resolved_device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loss_function = CombinedSegmentationLoss(
        cross_entropy_weight=cross_entropy_weight,
        dice_weight=dice_weight,
    )
    loss_configuration = {
        "cross_entropy_weight": float(cross_entropy_weight),
        "dice_weight": float(dice_weight),
        "foreground_classes": [1, 2],
    }
    optimizer_configuration = {
        "name": "AdamW",
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
    }
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)

    completed_epoch = 0
    best_mean_foreground_iou: float | None = None
    if resume_path is not None:
        checkpoint = _load_checkpoint(resume_path, device=resolved_device)
        _validate_checkpoint_compatibility(
            checkpoint,
            model_name=model_name,
            height=height,
            width=width,
            seed=seed,
            loss_configuration=loss_configuration,
            optimizer_configuration=optimizer_configuration,
        )
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        completed_epoch = checkpoint["completed_epoch"]
        best_mean_foreground_iou = checkpoint["best_mean_foreground_iou"]
        if "train_generator_state" in checkpoint:
            train_generator.set_state(checkpoint["train_generator_state"])
        if "torch_rng_state" in checkpoint:
            torch.set_rng_state(checkpoint["torch_rng_state"].cpu())
        if resolved_device.type == "cuda" and checkpoint.get("cuda_rng_state_all"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
        if amp_enabled and checkpoint.get("amp_scaler_state_dict"):
            scaler.load_state_dict(checkpoint["amp_scaler_state_dict"])
        if completed_epoch >= epochs:
            raise ValueError(
                f"Checkpoint completed epoch {completed_epoch}, "
                f"but requested epochs is {epochs}"
            )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=train_generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        worker_init_fn=seed_dataloader_worker,
        generator=val_generator,
    )

    config = {
        "dataset_name": dataset_path.name,
        "run_name": run_path.name,
        "class_mapping": dict(CLASS_MAPPING),
        "model_name": model_name,
        "num_classes": NUM_CLASSES,
        "input_height": height,
        "input_width": width,
        "epochs": epochs,
        "batch_size": batch_size,
        "optimizer": optimizer_configuration,
        "loss": loss_configuration,
        "seed": seed,
        "num_workers": num_workers,
        "device_requested": str(device),
        "device_resolved": str(resolved_device),
        "amp_requested": bool(amp),
        "amp_enabled": amp_enabled,
        "fixed_split_directories": {
            "train_images": "train/images",
            "train_masks": "train/masks",
            "val_images": "val/images",
            "val_masks": "val/masks",
        },
        "train_sample_count": len(train_dataset),
        "val_sample_count": len(val_dataset),
        "resumed": resume_path is not None,
    }
    _write_json(run_path / "config.json", config)

    history_path = run_path / "history.csv"
    _prepare_history(
        history_path,
        completed_epoch=completed_epoch,
        resuming=resume_path is not None,
    )
    best_checkpoint_path = run_path / "best_checkpoint.pt"
    last_checkpoint_path = run_path / "last_checkpoint.pt"
    best_metrics_path = run_path / "best_validation_metrics.json"
    best_epoch = _read_best_epoch(best_metrics_path)
    last_validation_metrics: dict[str, float] | None = None

    for epoch in range(completed_epoch + 1, epochs + 1):
        train_metrics = train_segmentation_epoch(
            model,
            train_loader,
            optimizer,
            loss_function,
            resolved_device,
            amp_enabled=amp_enabled,
            scaler=scaler,
        )
        val_metrics, _ = validate_segmentation_epoch(
            model,
            val_loader,
            loss_function,
            resolved_device,
            amp_enabled=amp_enabled,
        )
        last_validation_metrics = val_metrics
        history_row = {"epoch": epoch, **train_metrics, **val_metrics}
        _append_history_row(history_path, history_row)

        current_mean_iou = val_metrics["val_mean_foreground_iou"]
        current_is_finite = math.isfinite(current_mean_iou)
        is_best = current_is_finite and (
            best_mean_foreground_iou is None
            or current_mean_iou > best_mean_foreground_iou
        )
        if not best_checkpoint_path.exists():
            is_best = True
        if is_best:
            if current_is_finite:
                best_mean_foreground_iou = current_mean_iou
            best_epoch = epoch

        checkpoint = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            completed_epoch=epoch,
            best_mean_foreground_iou=best_mean_foreground_iou,
            height=height,
            width=width,
            seed=seed,
            model_name=model_name,
            loss_configuration=loss_configuration,
            optimizer_configuration=optimizer_configuration,
            train_generator=train_generator,
            scaler=scaler,
        )
        if is_best:
            _save_checkpoint(best_checkpoint_path, checkpoint)
            _write_json(
                best_metrics_path,
                {
                    "epoch": epoch,
                    **_json_ready_metrics(val_metrics),
                },
            )
        _save_checkpoint(last_checkpoint_path, checkpoint)
        _print_epoch_summary(
            epoch,
            epochs,
            train_metrics=train_metrics,
            val_metrics=val_metrics,
            best_mean_foreground_iou=best_mean_foreground_iou,
        )

    summary = {
        "dataset_name": dataset_path.name,
        "run_name": run_path.name,
        "model_name": model_name,
        "completed_epochs": epochs,
        "best_epoch": best_epoch,
        "best_mean_foreground_iou": best_mean_foreground_iou,
        "last_validation_metrics": (
            _json_ready_metrics(last_validation_metrics)
            if last_validation_metrics is not None
            else None
        ),
        "seed": seed,
        "amp_enabled": amp_enabled,
        "artifacts": {
            "config": "config.json",
            "history": "history.csv",
            "best_checkpoint": "best_checkpoint.pt",
            "last_checkpoint": "last_checkpoint.pt",
            "best_validation_metrics": "best_validation_metrics.json",
        },
    }
    _write_json(run_path / "summary.json", summary)
    return summary


def _batch_tensors(
    batch: Mapping[str, Any],
    *,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    images = batch["image"]
    targets = batch["mask"]
    if not isinstance(images, torch.Tensor) or not isinstance(targets, torch.Tensor):
        raise TypeError("Training batches must contain image and mask tensors")
    images = images.to(device)
    targets = targets.to(device)
    if images.dtype != torch.float32:
        raise TypeError(f"Images must use torch.float32, got {images.dtype}")
    if targets.dtype != torch.long:
        raise TypeError(f"Masks must use torch.long, got {targets.dtype}")
    return images, targets


def _model_logits(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    output = model(images)
    if isinstance(output, Mapping):
        if "logits" not in output:
            raise KeyError("Model output mapping must contain 'logits'")
        output = output["logits"]
    if not isinstance(output, torch.Tensor):
        raise TypeError("Model must return logits as a torch.Tensor")
    if (
        output.ndim != 4
        or output.shape[0] != images.shape[0]
        or output.shape[1] != NUM_CLASSES
        or output.shape[2:] != images.shape[2:]
    ):
        raise ValueError(
            "Model logits must have shape [B, 3, H, W] matching the input, "
            f"got {list(output.shape)}"
        )
    return output


def _validate_checkpoint_compatibility(
    checkpoint: Mapping[str, Any],
    *,
    model_name: str,
    height: int,
    width: int,
    seed: int,
    loss_configuration: Mapping[str, Any],
    optimizer_configuration: Mapping[str, Any],
) -> None:
    expected_values = {
        "class_mapping": CLASS_MAPPING,
        "input_height": height,
        "input_width": width,
        "seed": seed,
        "model_name": model_name,
        "loss_configuration": dict(loss_configuration),
        "optimizer_configuration": dict(optimizer_configuration),
    }
    for key, expected in expected_values.items():
        if checkpoint.get(key) != expected:
            raise ValueError(
                f"Incompatible checkpoint {key}: "
                f"expected {expected!r}, got {checkpoint.get(key)!r}"
            )

    required_keys = {
        "model_state_dict",
        "optimizer_state_dict",
        "completed_epoch",
        "best_mean_foreground_iou",
    }
    missing = sorted(required_keys - set(checkpoint))
    if missing:
        raise ValueError(f"Checkpoint is missing required keys: {', '.join(missing)}")
    if (
        not isinstance(checkpoint["completed_epoch"], int)
        or checkpoint["completed_epoch"] < 0
    ):
        raise ValueError("Checkpoint completed_epoch must be a non-negative integer")


def _checkpoint_payload(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    completed_epoch: int,
    best_mean_foreground_iou: float | None,
    height: int,
    width: int,
    seed: int,
    model_name: str,
    loss_configuration: Mapping[str, Any],
    optimizer_configuration: Mapping[str, Any],
    train_generator: torch.Generator,
    scaler: torch.amp.GradScaler,
) -> dict[str, Any]:
    return {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "completed_epoch": completed_epoch,
        "best_mean_foreground_iou": best_mean_foreground_iou,
        "class_mapping": dict(CLASS_MAPPING),
        "input_height": height,
        "input_width": width,
        "seed": seed,
        "model_name": model_name,
        "loss_configuration": dict(loss_configuration),
        "optimizer_configuration": dict(optimizer_configuration),
        "train_generator_state": train_generator.get_state(),
        "amp_scaler_state_dict": scaler.state_dict(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": (
            torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []
        ),
    }


def _load_checkpoint(path: Path, *, device: torch.device) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Resume checkpoint does not exist: {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=True)
    if not isinstance(checkpoint, dict):
        raise ValueError("Checkpoint must contain a dictionary")
    return checkpoint


def _save_checkpoint(path: Path, checkpoint: Mapping[str, Any]) -> None:
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(dict(checkpoint), temporary_path)
    temporary_path.replace(path)


def _resolve_resume_path(
    output_root: Path,
    resume: bool | PathLike | None,
) -> Path | None:
    if resume is None or resume is False:
        return None
    if resume is True or str(resume) == LAST_CHECKPOINT_SENTINEL:
        return output_root / "last_checkpoint.pt"
    return Path(resume)


def _prepare_output_root(
    output_root: Path,
    *,
    overwrite: bool,
    resuming: bool,
) -> None:
    if resuming:
        if not output_root.is_dir():
            raise FileNotFoundError(
                f"Output root must already exist when resuming: {output_root}"
            )
        return

    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. "
                "Pass overwrite=True or --overwrite to replace it."
            )
        if output_root.is_symlink() or not output_root.is_dir():
            raise ValueError("Existing output root must be a non-symlink directory")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True)


def _prepare_history(
    path: Path,
    *,
    completed_epoch: int,
    resuming: bool,
) -> None:
    existing_rows: list[dict[str, str]] = []
    if resuming and path.is_file():
        with path.open("r", encoding="utf-8", newline="") as history_file:
            reader = csv.DictReader(history_file)
            if tuple(reader.fieldnames or ()) != HISTORY_FIELDS:
                raise ValueError("Existing history.csv columns are incompatible")
            for row in reader:
                if int(row["epoch"]) <= completed_epoch:
                    existing_rows.append(row)

    with path.open("w", encoding="utf-8", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        writer.writeheader()
        writer.writerows(existing_rows)


def _append_history_row(path: Path, row: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8", newline="") as history_file:
        writer = csv.DictWriter(history_file, fieldnames=HISTORY_FIELDS)
        writer.writerow({field: row[field] for field in HISTORY_FIELDS})


def _read_best_epoch(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8")).get("epoch")
    except (json.JSONDecodeError, OSError, AttributeError):
        return None
    return value if isinstance(value, int) else None


def _json_ready_metrics(metrics: Mapping[str, float]) -> dict[str, float | None]:
    return {
        key: value if math.isfinite(value) else None for key, value in metrics.items()
    }


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _print_epoch_summary(
    epoch: int,
    epochs: int,
    *,
    train_metrics: Mapping[str, float],
    val_metrics: Mapping[str, float],
    best_mean_foreground_iou: float | None,
) -> None:
    print(f"Epoch {epoch:03d}/{epochs:03d}")
    print(
        f"train loss {train_metrics['train_total_loss']:.6f} | "
        f"val loss {val_metrics['val_total_loss']:.6f}"
    )
    print(
        f"Flesh IoU {_format_metric(val_metrics['val_iou_flesh'])} | "
        f"Calyx IoU {_format_metric(val_metrics['val_iou_calyx'])} | "
        "mean foreground IoU "
        f"{_format_metric(val_metrics['val_mean_foreground_iou'])} | "
        f"best {_format_metric(best_mean_foreground_iou)}"
    )


def _format_metric(value: float | None) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def _validate_size(height: int, width: int) -> tuple[int, int]:
    for name, value in (("height", height), ("width", width)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    return height, width


def _validate_seed(seed: int) -> None:
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        raise ValueError("seed must be a non-negative integer")


def _validate_training_arguments(
    *,
    epochs: int,
    batch_size: int,
    height: int,
    width: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
    num_workers: int,
) -> None:
    _validate_size(height, width)
    _validate_seed(seed)
    for name, value in (("epochs", epochs), ("batch_size", batch_size)):
        if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
            raise ValueError(f"{name} must be a positive integer")
    if (
        not isinstance(num_workers, int)
        or isinstance(num_workers, bool)
        or num_workers < 0
    ):
        raise ValueError("num_workers must be a non-negative integer")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be positive and finite")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be non-negative and finite")


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the reproducible real-only segmentation baseline."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--height", type=int, default=DEFAULT_HEIGHT)
    parser.add_argument("--width", type=int, default=DEFAULT_WIDTH)
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LEARNING_RATE)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--cross-entropy-weight",
        type=float,
        default=DEFAULT_CROSS_ENTROPY_WEIGHT,
    )
    parser.add_argument("--dice-weight", type=float, default=DEFAULT_DICE_WEIGHT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--resume",
        nargs="?",
        const=LAST_CHECKPOINT_SENTINEL,
        default=None,
        metavar="CHECKPOINT",
        help="resume from output-root/last_checkpoint.pt or an explicit checkpoint",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the segmentation baseline CLI."""

    arguments = _build_argument_parser().parse_args(argv)
    train_segmentation_baseline(
        arguments.dataset_root,
        arguments.output_root,
        epochs=arguments.epochs,
        batch_size=arguments.batch_size,
        height=arguments.height,
        width=arguments.width,
        learning_rate=arguments.learning_rate,
        weight_decay=arguments.weight_decay,
        cross_entropy_weight=arguments.cross_entropy_weight,
        dice_weight=arguments.dice_weight,
        seed=arguments.seed,
        num_workers=arguments.num_workers,
        device=arguments.device,
        amp=arguments.amp,
        overwrite=arguments.overwrite,
        resume=arguments.resume,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
