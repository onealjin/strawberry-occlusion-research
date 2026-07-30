"""Losses for three-class semantic segmentation training."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn


FOREGROUND_CLASSES = (1, 2)


def soft_foreground_dice_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    *,
    smooth: float = 1e-6,
) -> torch.Tensor:
    """Return soft Dice loss averaged over Flesh and Calyx only."""

    _validate_logits_and_target(logits, target)
    if smooth <= 0:
        raise ValueError("smooth must be positive")

    probabilities = torch.softmax(logits, dim=1)
    one_hot_target = F.one_hot(target, num_classes=logits.shape[1])
    one_hot_target = one_hot_target.permute(0, 3, 1, 2).to(probabilities.dtype)

    foreground_probabilities = probabilities[:, FOREGROUND_CLASSES]
    foreground_target = one_hot_target[:, FOREGROUND_CLASSES]
    reduce_dimensions = (0, 2, 3)
    intersection = (foreground_probabilities * foreground_target).sum(
        dim=reduce_dimensions
    )
    denominator = foreground_probabilities.sum(
        dim=reduce_dimensions
    ) + foreground_target.sum(dim=reduce_dimensions)
    dice = (2.0 * intersection + smooth) / (denominator + smooth)
    return 1.0 - dice.mean()


class CombinedSegmentationLoss(nn.Module):
    """Weighted unweighted-cross-entropy and foreground soft-Dice loss."""

    def __init__(
        self,
        *,
        cross_entropy_weight: float = 0.5,
        dice_weight: float = 0.5,
    ) -> None:
        super().__init__()
        for name, value in (
            ("cross_entropy_weight", cross_entropy_weight),
            ("dice_weight", dice_weight),
        ):
            if (
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(f"{name} must be finite and non-negative")
        if cross_entropy_weight == 0 and dice_weight == 0:
            raise ValueError("At least one loss weight must be positive")
        self.cross_entropy_weight = float(cross_entropy_weight)
        self.dice_weight = float(dice_weight)

    def components(
        self,
        logits: torch.Tensor,
        target: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return total, cross-entropy, and foreground Dice losses."""

        _validate_logits_and_target(logits, target)
        cross_entropy = F.cross_entropy(logits, target)
        dice = soft_foreground_dice_loss(logits, target)
        total = self.cross_entropy_weight * cross_entropy + self.dice_weight * dice
        return total, cross_entropy, dice

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        total, _, _ = self.components(logits, target)
        return total


def _validate_logits_and_target(
    logits: torch.Tensor,
    target: torch.Tensor,
) -> None:
    if not isinstance(logits, torch.Tensor):
        raise TypeError("logits must be a torch.Tensor")
    if not isinstance(target, torch.Tensor):
        raise TypeError("target must be a torch.Tensor")
    if logits.ndim != 4 or logits.shape[1] != 3:
        raise ValueError(
            f"logits must have shape [B, 3, H, W], got {list(logits.shape)}"
        )
    if target.ndim != 3:
        raise ValueError(f"target must have shape [B, H, W], got {list(target.shape)}")
    if logits.shape[0] != target.shape[0] or logits.shape[2:] != target.shape[1:]:
        raise ValueError("logits and target batch/spatial dimensions must match")
    if target.dtype != torch.long:
        raise TypeError(f"target must use torch.long, got {target.dtype}")
    if torch.any((target < 0) | (target > 2)):
        raise ValueError("target must contain only class IDs 0, 1, and 2")
