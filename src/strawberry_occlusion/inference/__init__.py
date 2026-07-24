"""Inference helpers and serializable prediction containers."""

from strawberry_occlusion.inference.predictor import InferenceResult
from strawberry_occlusion.inference.segmentation import predict_segmentation_masks

__all__ = ["InferenceResult", "predict_segmentation_masks"]
