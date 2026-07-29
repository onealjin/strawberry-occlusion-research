"""Create headless four-panel QA images for segmentation datasets."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw


PathLike = str | Path

CLASS_MAPPING = {"background": 0, "Flesh": 1, "Calyx": 2}
PALETTE = {
    0: (0, 0, 0),
    1: (255, 0, 0),
    2: (0, 255, 0),
}
BOUNDARY_COLOR = (255, 255, 0)
DEFAULT_ALPHA = 0.45
TITLE_HEIGHT = 24
PANEL_TITLES = ("Original RGB", "Segmentation", "Overlay", "Boundaries")
SUPPORTED_SPLITS = ("train", "val")
IMAGE_SUFFIXES = frozenset({".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})
MASK_SUFFIXES = frozenset({".bmp", ".png", ".tif", ".tiff"})


def colorize_mask(mask: np.ndarray) -> np.ndarray:
    """Map a class-index mask to the deterministic RGB QA palette."""

    mask_array = _validated_mask_array(mask)
    colorized = np.empty((*mask_array.shape, 3), dtype=np.uint8)
    for class_id, color in PALETTE.items():
        colorized[mask_array == class_id] = color
    return colorized


def overlay_mask(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> np.ndarray:
    """Blend foreground classes onto RGB pixels while preserving background."""

    alpha = _validate_alpha(alpha)
    image_array = _rgb_array(image)
    mask_array = _validated_mask_array(mask)
    _validate_matching_dimensions(image_array, mask_array)

    overlay = image_array.copy()
    colorized = colorize_mask(mask_array)
    foreground = mask_array != CLASS_MAPPING["background"]
    blended = (1.0 - alpha) * image_array[foreground] + alpha * colorized[foreground]
    overlay[foreground] = np.rint(blended).clip(0, 255).astype(np.uint8)
    return overlay


def find_class_boundaries(mask: np.ndarray) -> np.ndarray:
    """Return both pixels beside each differing four-neighbour class edge."""

    mask_array = _validated_mask_array(mask)
    boundaries = np.zeros(mask_array.shape, dtype=bool)

    horizontal_changes = mask_array[:, 1:] != mask_array[:, :-1]
    boundaries[:, 1:] |= horizontal_changes
    boundaries[:, :-1] |= horizontal_changes

    vertical_changes = mask_array[1:, :] != mask_array[:-1, :]
    boundaries[1:, :] |= vertical_changes
    boundaries[:-1, :] |= vertical_changes
    return boundaries


def overlay_boundaries(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Draw deterministic class boundaries over an RGB image."""

    image_array = _rgb_array(image)
    mask_array = _validated_mask_array(mask)
    _validate_matching_dimensions(image_array, mask_array)

    boundary_overlay = image_array.copy()
    boundary_overlay[find_class_boundaries(mask_array)] = BOUNDARY_COLOR
    return boundary_overlay


def create_qa_image(
    image: Image.Image | np.ndarray,
    mask: np.ndarray,
    *,
    alpha: float = DEFAULT_ALPHA,
) -> Image.Image:
    """Build a titled four-panel QA image without changing source geometry."""

    image_array = _rgb_array(image)
    mask_array = _validated_mask_array(mask)
    _validate_matching_dimensions(image_array, mask_array)
    alpha = _validate_alpha(alpha)

    panel_arrays = (
        image_array,
        colorize_mask(mask_array),
        overlay_mask(image_array, mask_array, alpha=alpha),
        overlay_boundaries(image_array, mask_array),
    )
    height, width = mask_array.shape
    qa_image = Image.new("RGB", (4 * width, height + TITLE_HEIGHT), color=(0, 0, 0))

    for index, (title, panel_array) in enumerate(
        zip(PANEL_TITLES, panel_arrays, strict=True)
    ):
        panel = Image.new("RGB", (width, height + TITLE_HEIGHT), color=(0, 0, 0))
        ImageDraw.Draw(panel).text((4, 5), title, fill=(255, 255, 255))
        panel.paste(Image.fromarray(panel_array), (0, TITLE_HEIGHT))
        qa_image.paste(panel, (index * width, 0))

    return qa_image


def generate_segmentation_qa(
    dataset_root: PathLike,
    output_root: PathLike,
    *,
    alpha: float = DEFAULT_ALPHA,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Generate QA images and sanitized metadata for train and val splits."""

    alpha = _validate_alpha(alpha)
    source_root = Path(dataset_root)
    destination_root = Path(output_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {source_root}")

    source_resolved = source_root.resolve()
    destination_resolved = destination_root.resolve()
    if _paths_overlap(source_resolved, destination_resolved):
        raise ValueError("Dataset and output roots must not overlap")

    if destination_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {destination_root}. "
                "Pass overwrite=True or --overwrite to replace it."
            )
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise ValueError("Existing output root must be a non-symlink directory")

    pairs_by_split = {
        split: _paired_paths(source_root, split=split) for split in SUPPORTED_SPLITS
    }
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.staging-",
            dir=destination_root.parent,
        )
    )

    try:
        samples: list[dict[str, Any]] = []
        sample_counts: dict[str, int] = {}
        for split, pairs in pairs_by_split.items():
            (staging_root / split).mkdir()
            sample_counts[split] = len(pairs)
            for image_path, mask_path in pairs:
                samples.append(
                    _write_sample_qa(
                        image_path,
                        mask_path,
                        split=split,
                        staging_root=staging_root,
                        alpha=alpha,
                    )
                )

        manifest: dict[str, Any] = {
            "class_mapping": dict(CLASS_MAPPING),
            "palette": {
                "background": list(PALETTE[0]),
                "Flesh": list(PALETTE[1]),
                "Calyx": list(PALETTE[2]),
                "boundary": list(BOUNDARY_COLOR),
            },
            "alpha": alpha,
            "sample_counts": sample_counts,
            "samples": samples,
        }
        (staging_root / "qa_manifest.json").write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        _install_staged_output(
            staging_root,
            destination_root,
            overwrite=overwrite,
        )
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    return manifest


def _paired_paths(dataset_root: Path, *, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / split / "images"
    mask_dir = dataset_root / split / "masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist for split {split!r}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist for split {split!r}")

    images = _files_by_stem(image_dir, suffixes=IMAGE_SUFFIXES, kind="image")
    masks = _files_by_stem(mask_dir, suffixes=MASK_SUFFIXES, kind="mask")
    image_stems = set(images)
    mask_stems = set(masks)

    missing_masks = sorted(image_stems - mask_stems)
    extra_masks = sorted(mask_stems - image_stems)
    if missing_masks:
        raise ValueError(f"Missing masks for image stem(s): {', '.join(missing_masks)}")
    if extra_masks:
        raise ValueError(f"Masks without matching images: {', '.join(extra_masks)}")
    return [(images[stem], masks[stem]) for stem in sorted(image_stems)]


def _files_by_stem(
    folder: Path,
    *,
    suffixes: frozenset[str],
    kind: str,
) -> dict[str, Path]:
    paths_by_stem: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in paths_by_stem:
            raise ValueError(f"Duplicate {kind} stem is ambiguous: {path.stem}")
        paths_by_stem[path.stem] = path
    return paths_by_stem


def _write_sample_qa(
    image_path: Path,
    mask_path: Path,
    *,
    split: str,
    staging_root: Path,
    alpha: float,
) -> dict[str, Any]:
    try:
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            if image_file.size != mask_file.size:
                raise ValueError(
                    "Image and mask dimensions differ for "
                    f"{image_path.name}: image={image_file.size}, "
                    f"mask={mask_file.size}"
                )
            image_array = np.array(image_file.convert("RGB"), dtype=np.uint8)
            mask_array = np.array(mask_file)
    except OSError as error:
        raise ValueError(f"Could not read sample {image_path.stem!r}") from error

    mask_array = _validated_mask_array(mask_array)
    height, width = mask_array.shape
    qa_image = create_qa_image(image_array, mask_array, alpha=alpha)
    relative_output_path = Path(split) / f"{image_path.stem}.png"
    qa_image.save(staging_root / relative_output_path, format="PNG")

    return {
        "sample_id": image_path.stem,
        "split": split,
        "output_path": relative_output_path.as_posix(),
        "image_width": width,
        "image_height": height,
        "class_pixel_counts": {
            "background": int(np.count_nonzero(mask_array == 0)),
            "Flesh": int(np.count_nonzero(mask_array == 1)),
            "Calyx": int(np.count_nonzero(mask_array == 2)),
        },
    }


def _validated_mask_array(mask: np.ndarray) -> np.ndarray:
    mask_array = np.asarray(mask)
    if mask_array.ndim != 2:
        raise ValueError(
            f"Mask must be single-channel with shape [H, W], got {mask_array.shape}"
        )

    valid_pixels = np.isin(mask_array, tuple(PALETTE))
    if not np.all(valid_pixels):
        invalid_values = np.unique(mask_array[~valid_pixels]).tolist()
        raise ValueError(
            "Mask contains invalid class values; expected only 0, 1, and 2, "
            f"found {invalid_values}"
        )
    return mask_array.astype(np.uint8, copy=False)


def _rgb_array(image: Image.Image | np.ndarray) -> np.ndarray:
    if isinstance(image, Image.Image):
        return np.array(image.convert("RGB"), dtype=np.uint8)

    image_array = np.asarray(image)
    if image_array.ndim != 3 or image_array.shape[2] != 3:
        raise ValueError(
            f"Image must have RGB shape [H, W, 3], got {image_array.shape}"
        )
    if not np.issubdtype(image_array.dtype, np.integer):
        raise ValueError("RGB image arrays must use an integer dtype")
    if np.any(image_array < 0) or np.any(image_array > 255):
        raise ValueError("RGB image values must be in [0, 255]")
    return image_array.astype(np.uint8, copy=True)


def _validate_matching_dimensions(
    image: np.ndarray,
    mask: np.ndarray,
) -> None:
    if image.shape[:2] != mask.shape:
        raise ValueError(
            "Image and mask dimensions differ: "
            f"image={image.shape[:2]}, mask={mask.shape}"
        )


def _validate_alpha(alpha: float) -> float:
    if (
        isinstance(alpha, bool)
        or not isinstance(alpha, (int, float))
        or not math.isfinite(alpha)
        or not 0.0 <= alpha <= 1.0
    ):
        raise ValueError("alpha must be a finite number in [0, 1]")
    return float(alpha)


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _install_staged_output(
    staging_root: Path,
    destination_root: Path,
    *,
    overwrite: bool,
) -> None:
    if destination_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root appeared during generation: {destination_root}"
            )
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise ValueError("Existing output root must be a non-symlink directory")
        shutil.rmtree(destination_root)
    staging_root.replace(destination_root)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create headless four-panel QA images for segmentation datasets."
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--alpha",
        type=float,
        default=DEFAULT_ALPHA,
        help=f"foreground overlay opacity (default: {DEFAULT_ALPHA})",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory after successful generation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the segmentation QA CLI."""

    arguments = _build_argument_parser().parse_args(argv)
    manifest = generate_segmentation_qa(
        arguments.dataset_root,
        arguments.output_root,
        alpha=arguments.alpha,
        overwrite=arguments.overwrite,
    )
    counts = manifest["sample_counts"]
    print(
        "Generated QA images for "
        f"{counts['train']} train and {counts['val']} val samples "
        f"in {arguments.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
