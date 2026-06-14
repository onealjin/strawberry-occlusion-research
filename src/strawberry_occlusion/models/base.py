"""Base model interfaces for segmentation-style research models."""

from pathlib import Path

import torch
from torch import nn


class SegmentationModel(nn.Module):
    """Base class for models that predict visible or amodal masks."""

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the model and return named tensors suitable for JSON post-processing."""

        raise NotImplementedError


def export_onnx(
    model: nn.Module,
    sample_input: torch.Tensor,
    output_path: str | Path,
    *,
    opset_version: int = 17,
) -> None:
    """Export a model to ONNX with a single image tensor input."""

    model.eval()
    torch.onnx.export(
        model,
        sample_input,
        Path(output_path),
        input_names=["image"],
        output_names=["outputs"],
        opset_version=opset_version,
    )
