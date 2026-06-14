"""Minimal training loop utilities for prototype experiments."""

from collections.abc import Callable, Iterable

import torch
from torch import nn


def train_one_epoch(
    model: nn.Module,
    batches: Iterable[tuple[torch.Tensor, torch.Tensor]],
    optimizer: torch.optim.Optimizer,
    loss_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    device: torch.device,
) -> float:
    """Train for one epoch and return the mean loss."""

    model.train()
    total_loss = 0.0
    batch_count = 0

    for images, targets in batches:
        images = images.to(device)
        targets = targets.to(device)

        optimizer.zero_grad(set_to_none=True)
        predictions = model(images)
        loss = loss_fn(predictions, targets)
        loss.backward()
        optimizer.step()

        total_loss += float(loss.detach().cpu())
        batch_count += 1

    if batch_count == 0:
        return 0.0
    return total_loss / batch_count
