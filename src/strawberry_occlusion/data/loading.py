"""Dataset discovery utilities for configured research data locations."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ImageRecord:
    """Reference to one image and an optional sanitized label file."""

    image_path: Path
    label_path: Path | None = None


def list_image_records(root: str | Path, pattern: str = "*.png") -> list[ImageRecord]:
    """Return image records under a caller-provided public or synthetic data root."""

    root_path = Path(root)
    return [ImageRecord(image_path=path) for path in sorted(root_path.glob(pattern))]
