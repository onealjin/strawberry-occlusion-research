"""Dataset for public, synthetic, or sanitized segmentation image/mask pairs."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from pathlib import Path
from typing import TypedDict

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF


PathLike = str | Path
TensorTransform = Callable[[Image.Image], torch.Tensor]

DEFAULT_IMAGE_SUFFIXES = frozenset(
    {".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"}
)
DEFAULT_MASK_SUFFIXES = frozenset({".bmp", ".png", ".tif", ".tiff"})


class SegmentationSample(TypedDict):
    """One segmentation sample returned by ``SegmentationDataset``."""

    image: torch.Tensor
    mask: torch.Tensor
    image_path: str
    mask_path: str


class SegmentationDataset(Dataset[SegmentationSample]):
    """Read paired RGB images and single-channel class-index masks from folders.

    Files are paired by matching filename stem. For example, ``images/a.jpg`` is
    paired with ``masks/a.png``. The folders are always caller-provided; this
    class intentionally contains no project-specific data locations.
    """

    def __init__(
        self,
        image_dir: PathLike,
        mask_dir: PathLike,
        *,
        image_suffixes: Iterable[str] = DEFAULT_IMAGE_SUFFIXES,
        mask_suffixes: Iterable[str] = DEFAULT_MASK_SUFFIXES,
        image_transform: TensorTransform | None = None,
        mask_transform: TensorTransform | None = None,
    ) -> None:
        self.image_dir = Path(image_dir)
        self.mask_dir = Path(mask_dir)
        self.image_transform = image_transform or _image_to_float_tensor
        self.mask_transform = mask_transform or _mask_to_long_tensor

        if not self.image_dir.is_dir():
            raise FileNotFoundError(f"Image folder does not exist: {self.image_dir}")
        if not self.mask_dir.is_dir():
            raise FileNotFoundError(f"Mask folder does not exist: {self.mask_dir}")

        image_paths = _list_files(
            self.image_dir, suffixes=_normalize_suffixes(image_suffixes)
        )
        mask_paths = _list_files(
            self.mask_dir, suffixes=_normalize_suffixes(mask_suffixes)
        )

        if not image_paths:
            raise ValueError(f"No supported image files found in {self.image_dir}")
        if not mask_paths:
            raise ValueError(f"No supported mask files found in {self.mask_dir}")

        _raise_for_duplicate_stems(image_paths, kind="image")
        _raise_for_duplicate_stems(mask_paths, kind="mask")

        masks_by_stem = {path.stem: path for path in mask_paths}
        image_stems = {path.stem for path in image_paths}
        mask_stems = set(masks_by_stem)

        missing_masks = sorted(image_stems - mask_stems)
        extra_masks = sorted(mask_stems - image_stems)
        if missing_masks:
            raise ValueError(
                f"Missing masks for image stem(s): {_format_stems(missing_masks)}"
            )
        if extra_masks:
            raise ValueError(
                f"Masks without matching images: {_format_stems(extra_masks)}"
            )

        self.pairs = tuple(
            (image_path, masks_by_stem[image_path.stem]) for image_path in image_paths
        )

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, index: int) -> SegmentationSample:
        image_path, mask_path = self.pairs[index]

        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            if image_file.size != mask_file.size:
                raise ValueError(
                    "Image and mask sizes differ for "
                    f"{image_path.name}: image={image_file.size}, mask={mask_file.size}"
                )

            image = self.image_transform(image_file.convert("RGB"))
            mask = self.mask_transform(mask_file)

        _validate_image_tensor(image, image_path=image_path)
        _validate_mask_tensor(mask, mask_path=mask_path)

        if image.shape[-2:] != mask.shape:
            raise ValueError(
                "Transformed image and mask sizes differ for "
                f"{image_path.name}: image={tuple(image.shape[-2:])}, "
                f"mask={tuple(mask.shape)}"
            )

        return {
            "image": image,
            "mask": mask,
            "image_path": str(image_path),
            "mask_path": str(mask_path),
        }


def _image_to_float_tensor(image: Image.Image) -> torch.Tensor:
    return TF.pil_to_tensor(image).float().div(255.0)


def _mask_to_long_tensor(mask: Image.Image) -> torch.Tensor:
    mask_tensor = TF.pil_to_tensor(mask)
    if mask_tensor.ndim != 3 or mask_tensor.shape[0] != 1:
        raise ValueError(
            f"Mask must be a single-channel class-index image, got mode={mask.mode!r}"
        )
    return mask_tensor.squeeze(0).long()


def _normalize_suffixes(suffixes: Iterable[str]) -> frozenset[str]:
    normalized = []
    for suffix in suffixes:
        suffix = suffix.lower()
        normalized.append(suffix if suffix.startswith(".") else f".{suffix}")
    return frozenset(normalized)


def _list_files(folder: Path, *, suffixes: frozenset[str]) -> list[Path]:
    return sorted(
        path
        for path in folder.iterdir()
        if path.is_file() and path.suffix.lower() in suffixes
    )


def _raise_for_duplicate_stems(paths: Iterable[Path], *, kind: str) -> None:
    stems: dict[str, list[Path]] = {}
    for path in paths:
        stems.setdefault(path.stem, []).append(path)

    duplicate_stems = sorted(
        stem for stem, matches in stems.items() if len(matches) > 1
    )
    if duplicate_stems:
        raise ValueError(
            f"Duplicate {kind} stem(s) are ambiguous: {_format_stems(duplicate_stems)}"
        )


def _format_stems(stems: list[str], *, limit: int = 5) -> str:
    shown = ", ".join(stems[:limit])
    remaining = len(stems) - limit
    if remaining > 0:
        return f"{shown}, ... ({remaining} more)"
    return shown


def _validate_image_tensor(image: torch.Tensor, *, image_path: Path) -> None:
    if not isinstance(image, torch.Tensor):
        raise TypeError(f"Image transform for {image_path.name} must return a tensor")
    if image.ndim != 3 or image.shape[0] != 3:
        raise ValueError(
            f"Image tensor for {image_path.name} must have shape [3, H, W], "
            f"got {tuple(image.shape)}"
        )
    if not image.is_floating_point():
        raise TypeError(f"Image tensor for {image_path.name} must use a float dtype")


def _validate_mask_tensor(mask: torch.Tensor, *, mask_path: Path) -> None:
    if not isinstance(mask, torch.Tensor):
        raise TypeError(f"Mask transform for {mask_path.name} must return a tensor")
    if mask.ndim != 2:
        raise ValueError(
            f"Mask tensor for {mask_path.name} must have shape [H, W], "
            f"got {tuple(mask.shape)}"
        )
    if mask.dtype != torch.long:
        raise TypeError(f"Mask tensor for {mask_path.name} must use torch.long dtype")
