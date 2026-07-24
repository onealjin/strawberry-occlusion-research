"""Stateless metrics for class-index segmentation masks."""

import torch

_INTEGER_DTYPES = {
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
}


def confusion_matrix(
    predicted: torch.Tensor,
    target: torch.Tensor,
    num_classes: int,
    *,
    ignore_index: int | None = None,
) -> torch.Tensor:
    """Return target-row, prediction-column counts for segmentation masks."""

    if not isinstance(predicted, torch.Tensor):
        raise TypeError(
            f"predicted must be a torch.Tensor, got {type(predicted).__name__}"
        )
    if not isinstance(target, torch.Tensor):
        raise TypeError(f"target must be a torch.Tensor, got {type(target).__name__}")
    if not isinstance(num_classes, int) or isinstance(num_classes, bool):
        raise TypeError(
            f"num_classes must be an integer, got {type(num_classes).__name__}"
        )
    if num_classes < 1:
        raise ValueError(f"num_classes must be at least 1, got {num_classes}")
    if ignore_index is not None and (
        not isinstance(ignore_index, int) or isinstance(ignore_index, bool)
    ):
        raise TypeError(
            f"ignore_index must be an integer or None, got {type(ignore_index).__name__}"
        )
    if predicted.shape != target.shape:
        raise ValueError(
            "predicted and target shapes must match, "
            f"got {list(predicted.shape)} and {list(target.shape)}"
        )
    if predicted.ndim not in (2, 3):
        raise ValueError(
            "predicted and target must have shape [H, W] or [B, H, W], "
            f"got {list(predicted.shape)}"
        )
    if predicted.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"predicted must use an integer dtype, got {predicted.dtype}")
    if target.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"target must use an integer dtype, got {target.dtype}")
    if predicted.device != target.device:
        raise ValueError(
            "predicted and target must be on the same device, "
            f"got {predicted.device} and {target.device}"
        )

    predicted_labels = predicted.reshape(-1)
    target_labels = target.reshape(-1)
    if ignore_index is not None:
        included = target_labels != ignore_index
        predicted_labels = predicted_labels[included]
        target_labels = target_labels[included]

    for name, labels in (
        ("predicted", predicted_labels),
        ("target", target_labels),
    ):
        invalid = (labels < 0) | (labels >= num_classes)
        if torch.any(invalid):
            invalid_label = int(labels[invalid][0].item())
            raise ValueError(
                f"{name} labels must be in [0, {num_classes - 1}], "
                f"found {invalid_label}"
            )

    flat_indices = target_labels.to(torch.long) * num_classes
    flat_indices += predicted_labels.to(torch.long)
    return torch.bincount(
        flat_indices,
        minlength=num_classes * num_classes,
    ).reshape(num_classes, num_classes)


def intersection_over_union(matrix: torch.Tensor) -> torch.Tensor:
    """Return per-class IoU, with NaN for classes whose union is zero."""

    matrix = _validate_confusion_matrix(matrix)
    true_positives = matrix.diagonal()
    union = matrix.sum(dim=1) + matrix.sum(dim=0) - true_positives
    scores = torch.full_like(true_positives, torch.nan)
    present = union > 0
    scores[present] = true_positives[present] / union[present]
    return scores


def dice_score(matrix: torch.Tensor) -> torch.Tensor:
    """Return per-class Dice, with NaN for classes absent from both masks."""

    matrix = _validate_confusion_matrix(matrix)
    true_positives = matrix.diagonal()
    denominator = matrix.sum(dim=1) + matrix.sum(dim=0)
    scores = torch.full_like(true_positives, torch.nan)
    present = denominator > 0
    scores[present] = 2 * true_positives[present] / denominator[present]
    return scores


def mean_ignore_nan(values: torch.Tensor) -> torch.Tensor:
    """Return the scalar mean of metric values while ignoring NaNs."""

    if not isinstance(values, torch.Tensor):
        raise TypeError(f"values must be a torch.Tensor, got {type(values).__name__}")
    if not values.dtype.is_floating_point and values.dtype not in _INTEGER_DTYPES:
        raise TypeError(f"values must use a real numeric dtype, got {values.dtype}")
    if not values.dtype.is_floating_point:
        values = values.to(torch.float32)
    return torch.nanmean(values)


def _validate_confusion_matrix(matrix: torch.Tensor) -> torch.Tensor:
    if not isinstance(matrix, torch.Tensor):
        raise TypeError(
            f"confusion matrix must be a torch.Tensor, got {type(matrix).__name__}"
        )
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] == 0:
        raise ValueError(
            "confusion matrix must be a non-empty square tensor, "
            f"got shape {list(matrix.shape)}"
        )
    if not matrix.dtype.is_floating_point and matrix.dtype not in _INTEGER_DTYPES:
        raise TypeError(
            f"confusion matrix must use a real numeric dtype, got {matrix.dtype}"
        )

    float_matrix = matrix.to(torch.float64)
    if not torch.all(torch.isfinite(float_matrix)):
        raise ValueError("confusion matrix values must be finite")
    if torch.any(float_matrix < 0):
        raise ValueError("confusion matrix values must be non-negative")
    return float_matrix
