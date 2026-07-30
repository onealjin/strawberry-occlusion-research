"""Training entry points and small training-loop helpers."""

from strawberry_occlusion.training.losses import (
    CombinedSegmentationLoss,
    soft_foreground_dice_loss,
)
from strawberry_occlusion.training.runner import train_one_epoch

__all__ = [
    "CombinedSegmentationLoss",
    "soft_foreground_dice_loss",
    "train_one_epoch",
]
