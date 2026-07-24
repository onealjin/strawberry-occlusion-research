"""Minimal inference helper for semantic segmentation models."""

import torch
from torch import nn


def predict_segmentation_masks(
    model: nn.Module,
    image: torch.Tensor,
    *,
    device: str | torch.device = "cpu",
) -> torch.Tensor:
    """Run segmentation inference and return batched class-index masks.

    The model may return logits directly or in a dictionary under ``"logits"``.
    Logit height and width must match the input because this helper does not resize
    outputs. The returned masks remain on the requested ``device``.
    """

    if not isinstance(model, nn.Module):
        raise TypeError(f"model must be a torch.nn.Module, got {type(model).__name__}")
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"image must be a torch.Tensor, got {type(image).__name__}")
    if image.ndim not in (3, 4):
        raise ValueError(
            f"image must have shape [3, H, W] or [B, 3, H, W], got {list(image.shape)}"
        )

    channel_dimension = 0 if image.ndim == 3 else 1
    if image.shape[channel_dimension] != 3:
        raise ValueError(
            "image must have exactly 3 channels in shape [3, H, W] or "
            f"[B, 3, H, W], got {list(image.shape)}"
        )

    batched_image = image.unsqueeze(0) if image.ndim == 3 else image
    if batched_image.shape[0] == 0:
        raise ValueError("image batch dimension must be greater than zero")
    if batched_image.shape[2] == 0 or batched_image.shape[3] == 0:
        raise ValueError("image height and width must be greater than zero")

    original_device = next((parameter.device for parameter in model.parameters()), None)
    if original_device is None:
        original_device = next((buffer.device for buffer in model.buffers()), None)

    target_device = torch.device(device)
    batched_image = batched_image.to(target_device)
    was_training = model.training
    try:
        model.to(target_device)
        model.eval()
        with torch.inference_mode():
            output = model(batched_image)

        if isinstance(output, dict):
            if "logits" not in output:
                raise KeyError("model output dictionary must contain a 'logits' key")
            logits = output["logits"]
            if not isinstance(logits, torch.Tensor):
                raise TypeError(
                    "model output dictionary value for 'logits' must be a "
                    f"torch.Tensor, got {type(logits).__name__}"
                )
        else:
            logits = output
        if not isinstance(logits, torch.Tensor):
            raise TypeError(
                "model output must be a logits torch.Tensor or a dictionary "
                "containing a tensor under 'logits', "
                f"got {type(logits).__name__}"
            )
        if logits.ndim != 4:
            raise ValueError(
                f"model output must have shape [B, C, H, W], got {list(logits.shape)}"
            )
        if logits.shape[0] != batched_image.shape[0]:
            raise ValueError(
                "model output batch dimension must match the input batch: "
                f"expected {batched_image.shape[0]}, got {logits.shape[0]}"
            )
        if logits.shape[1] == 0:
            raise ValueError("model output class dimension must be greater than zero")
        if logits.shape[2:] != batched_image.shape[2:]:
            raise ValueError(
                "model output height and width must match the input; output resizing "
                "is not supported: "
                f"expected {list(batched_image.shape[2:])}, "
                f"got {list(logits.shape[2:])}"
            )

        return logits.argmax(dim=1)
    finally:
        model.train(was_training)
        if original_device is not None:
            model.to(original_device)
