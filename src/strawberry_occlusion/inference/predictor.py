"""Serializable inference result containers."""

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class InferenceResult:
    """JSON-ready output for one occlusion-aware cutline prediction."""

    visible_segmentation: dict[str, Any]
    occlusion_state: str
    attachment_zone: dict[str, Any]
    cutline_geometry: dict[str, Any]
    uncertainty: dict[str, float] = field(default_factory=dict)

    def to_json_dict(self) -> dict[str, Any]:
        """Return a dictionary that can be serialized by the standard json module."""

        return {
            "visible_segmentation": self.visible_segmentation,
            "occlusion_state": self.occlusion_state,
            "attachment_zone": self.attachment_zone,
            "cutline_geometry": self.cutline_geometry,
            "uncertainty": self.uncertainty,
        }
