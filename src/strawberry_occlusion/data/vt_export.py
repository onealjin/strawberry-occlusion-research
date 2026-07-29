"""Convert a user-cleared VisionTrain segmentation export to class-index masks."""

from __future__ import annotations

import argparse
import base64
import binascii
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


PathLike = str | Path
CategoryId = int | str

CLASS_MAPPING = {"background": 0, "Flesh": 1, "Calyx": 2}
OVERLAP_RULE = "Calyx overrides Flesh"
SUPPORTED_ANNOTATION_LABELS = frozenset({"Flesh", "Calyx"})
SUPPORTED_SOURCE_LABELS = frozenset(CLASS_MAPPING)
SUPPORTED_SPLITS = frozenset({"train", "val"})


def construct_mask(
    annotations: Sequence[Mapping[str, Any]],
    *,
    image_width: int,
    image_height: int,
    categories: Mapping[CategoryId, str],
    image_name: str,
) -> tuple[np.ndarray, dict[str, int]]:
    """Construct one class-index mask and return its sanitized pixel statistics.

    ``categories`` maps VisionTrain category IDs to their semantic names. Source
    IDs are used only to validate annotations; output pixels always use
    ``CLASS_MAPPING``.
    """

    width = _positive_integer(image_width, field="image_width")
    height = _positive_integer(image_height, field="image_height")
    expected_image_name = _safe_filename(image_name, field="image_name")
    category_names = _validate_category_mapping(categories)

    if not isinstance(annotations, Sequence) or isinstance(
        annotations, (str, bytes, bytearray)
    ):
        raise ValueError("Annotations must be a JSON-style list")

    flesh_regions = np.zeros((height, width), dtype=bool)
    calyx_regions = np.zeros((height, width), dtype=bool)
    annotation_counts = {"Flesh": 0, "Calyx": 0}

    for index, annotation in enumerate(annotations):
        if not isinstance(annotation, Mapping):
            raise ValueError(f"Annotation {index} must be a JSON object")

        annotation_image_name = _required_string(
            annotation, "image_name", context=f"annotation {index}"
        )
        if annotation_image_name != expected_image_name:
            raise ValueError(
                f"Annotation {index} image_name does not match {expected_image_name!r}"
            )

        category_id = _category_id(
            _required_value(annotation, "category_id", context=f"annotation {index}"),
            field=f"annotation {index} category_id",
        )
        if category_id not in category_names:
            raise ValueError(
                f"Annotation {index} uses unknown category ID {category_id!r}"
            )

        source_label = category_names[category_id]
        label_name = _required_string(
            annotation, "labelname", context=f"annotation {index}"
        )
        if label_name != source_label:
            raise ValueError(
                f"Annotation {index} labelname {label_name!r} does not match "
                f"dataset category {source_label!r}"
            )
        if label_name not in SUPPORTED_ANNOTATION_LABELS:
            raise ValueError(
                f"Annotation {index} uses unsupported semantic label {label_name!r}"
            )

        annotation_type = _required_string(
            annotation, "type", context=f"annotation {index}"
        )
        if not annotation_type:
            raise ValueError(f"Annotation {index} type must not be empty")

        x, y, roi_width, roi_height = _validate_bbox(
            _required_value(annotation, "bbox", context=f"annotation {index}"),
            image_width=width,
            image_height=height,
            annotation_index=index,
        )
        roi_mask = _decode_roi_mask(
            _required_value(annotation, "pixel_image", context=f"annotation {index}"),
            width=roi_width,
            height=roi_height,
            annotation_index=index,
        )

        target = flesh_regions if label_name == "Flesh" else calyx_regions
        target[y : y + roi_height, x : x + roi_width] |= roi_mask
        annotation_counts[label_name] += 1

    overlap_pixels = int(np.count_nonzero(flesh_regions & calyx_regions))
    mask = np.zeros((height, width), dtype=np.uint8)
    mask[flesh_regions] = CLASS_MAPPING["Flesh"]
    mask[calyx_regions] = CLASS_MAPPING["Calyx"]

    summary = {
        "source_annotation_count": len(annotations),
        "flesh_annotation_count": annotation_counts["Flesh"],
        "calyx_annotation_count": annotation_counts["Calyx"],
        "background_pixel_count": int(np.count_nonzero(mask == 0)),
        "flesh_pixel_count": int(np.count_nonzero(mask == 1)),
        "calyx_pixel_count": int(np.count_nonzero(mask == 2)),
        "cross_class_overlap_pixel_count": overlap_pixels,
    }
    return mask, summary


def convert_vt_export(
    input_root: PathLike,
    output_root: PathLike,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert every labeled manifest image in a VisionTrain export.

    Conversion is staged in a temporary sibling directory. The requested output
    is installed only after all source files and generated masks pass validation.
    """

    source_root = Path(input_root)
    destination_root = Path(output_root)
    if not source_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {source_root}")

    source_resolved = source_root.resolve()
    destination_resolved = destination_root.resolve()
    if _paths_overlap(source_resolved, destination_resolved):
        raise ValueError("Input and output roots must not overlap")

    if destination_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {destination_root}. "
                "Pass overwrite=True or --overwrite to replace it."
            )
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise ValueError("Existing output root must be a non-symlink directory")

    dataset_path = source_root / "dataset.json"
    if not dataset_path.is_file():
        raise FileNotFoundError(f"Dataset manifest does not exist: {dataset_path}")
    dataset = _load_json(dataset_path)
    if not isinstance(dataset, Mapping):
        raise ValueError("dataset.json must contain a JSON object")

    category_mapping, source_categories = _parse_source_categories(
        _required_value(dataset, "categories", context="dataset.json")
    )
    images = _required_value(dataset, "images", context="dataset.json")
    if not isinstance(images, list):
        raise ValueError("dataset.json images must be a JSON list")

    labeled_images = _validate_manifest_images(images)
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.staging-",
            dir=destination_root.parent,
        )
    )

    try:
        for split in sorted(SUPPORTED_SPLITS):
            (staging_root / split / "images").mkdir(parents=True)
            (staging_root / split / "masks").mkdir(parents=True)

        samples: list[dict[str, Any]] = []
        split_counts = {"train": 0, "val": 0}
        for image_record in labeled_images:
            sample = _convert_sample(
                image_record,
                source_root=source_root,
                staging_root=staging_root,
                categories=category_mapping,
            )
            samples.append(sample)
            split_counts[sample["split"]] += 1

        manifest: dict[str, Any] = {
            "dataset_name": destination_root.name,
            "class_mapping": dict(CLASS_MAPPING),
            "overlap_rule": OVERLAP_RULE,
            "source_categories": source_categories,
            "sample_counts": split_counts,
            "samples": samples,
        }
        manifest_path = staging_root / "manifest.json"
        manifest_path.write_text(
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


def _convert_sample(
    image_record: Mapping[str, Any],
    *,
    source_root: Path,
    staging_root: Path,
    categories: Mapping[CategoryId, str],
) -> dict[str, Any]:
    sample_id = image_record["sample_id"]
    image_name = image_record["file_name"]
    split = image_record["split"]
    width = image_record["width"]
    height = image_record["height"]

    image_path = source_root / image_name
    if not image_path.is_file():
        raise FileNotFoundError(f"Missing source image for sample {sample_id!r}")

    annotation_path = source_root / f"{Path(image_name).stem}.json"
    if not annotation_path.is_file():
        raise FileNotFoundError(
            f"Missing per-image annotation JSON for sample {sample_id!r}"
        )

    try:
        with Image.open(image_path) as image:
            actual_size = image.size
            image.verify()
    except (OSError, ValueError) as error:
        raise ValueError(f"Source image is not a valid JPG: {image_name}") from error
    if actual_size != (width, height):
        raise ValueError(
            f"Image dimensions for {image_name} do not match dataset.json: "
            f"expected {(width, height)}, got {actual_size}"
        )

    annotations = _load_json(annotation_path)
    if not isinstance(annotations, list):
        raise ValueError(f"Per-image JSON for {image_name} must contain a list")

    mask, summary = construct_mask(
        annotations,
        image_width=width,
        image_height=height,
        categories=categories,
        image_name=image_name,
    )

    relative_image_path = Path(split) / "images" / image_name
    relative_mask_path = Path(split) / "masks" / f"{Path(image_name).stem}.png"
    output_image_path = staging_root / relative_image_path
    output_mask_path = staging_root / relative_mask_path

    shutil.copyfile(image_path, output_image_path)
    Image.fromarray(mask).save(output_mask_path, format="PNG")
    _validate_saved_mask(
        output_mask_path,
        expected_width=width,
        expected_height=height,
    )

    return {
        "sample_id": sample_id,
        "split": split,
        "image_path": relative_image_path.as_posix(),
        "mask_path": relative_mask_path.as_posix(),
        "image_width": width,
        "image_height": height,
        **summary,
    }


def _parse_source_categories(
    categories: Any,
) -> tuple[dict[CategoryId, str], list[dict[str, Any]]]:
    if not isinstance(categories, list):
        raise ValueError("dataset.json categories must be a JSON list")

    mapping: dict[CategoryId, str] = {}
    seen_names: set[str] = set()
    sanitized_categories: list[dict[str, Any]] = []
    for index, category in enumerate(categories):
        if not isinstance(category, Mapping):
            raise ValueError(f"Dataset category {index} must be a JSON object")
        category_id = _category_id(
            _required_value(category, "id", context=f"dataset category {index}"),
            field=f"dataset category {index} id",
        )
        name = _required_string(category, "name", context=f"dataset category {index}")
        if name not in SUPPORTED_SOURCE_LABELS:
            raise ValueError(f"Unsupported source category name {name!r}")
        if category_id in mapping:
            raise ValueError(f"Duplicate dataset category ID {category_id!r}")
        if name in seen_names:
            raise ValueError(f"Duplicate dataset category name {name!r}")

        mapping[category_id] = name
        seen_names.add(name)
        sanitized_categories.append(
            {
                "source_category_id": category_id,
                "source_category_name": name,
                "output_class_id": CLASS_MAPPING[name],
            }
        )

    _validate_category_mapping(mapping)
    return mapping, sanitized_categories


def _validate_manifest_images(images: list[Any]) -> list[dict[str, Any]]:
    labeled_images: list[dict[str, Any]] = []
    seen_ids: set[CategoryId] = set()
    seen_names: set[str] = set()
    seen_output_paths: set[str] = set()

    for index, image in enumerate(images):
        if not isinstance(image, Mapping):
            raise ValueError(f"Dataset image {index} must be a JSON object")

        sample_id = _sample_id(
            _required_value(image, "id", context=f"dataset image {index}"),
            field=f"dataset image {index} id",
        )
        file_name = _safe_filename(
            _required_string(image, "file_name", context=f"dataset image {index}"),
            field=f"dataset image {index} file_name",
        )
        if Path(file_name).suffix.lower() != ".jpg":
            raise ValueError(f"Dataset image {file_name!r} must be a JPG file")

        width = _positive_integer(
            _required_value(image, "width", context=f"dataset image {index}"),
            field=f"dataset image {index} width",
        )
        height = _positive_integer(
            _required_value(image, "height", context=f"dataset image {index}"),
            field=f"dataset image {index} height",
        )
        split = _required_string(image, "image_type", context=f"dataset image {index}")
        if split not in SUPPORTED_SPLITS:
            raise ValueError(
                f"Dataset image {file_name!r} uses unsupported split {split!r}"
            )
        labeled = _required_value(image, "labeled", context=f"dataset image {index}")
        if not isinstance(labeled, bool):
            raise ValueError(f"Dataset image {file_name!r} labeled must be boolean")

        if sample_id in seen_ids:
            raise ValueError(f"Duplicate dataset image ID {sample_id!r}")
        if file_name.casefold() in seen_names:
            raise ValueError(f"Duplicate dataset image filename {file_name!r}")
        seen_ids.add(sample_id)
        seen_names.add(file_name.casefold())

        output_key = f"{split}/{Path(file_name).stem}".casefold()
        if output_key in seen_output_paths:
            raise ValueError(f"Duplicate output stem for dataset image {file_name!r}")
        seen_output_paths.add(output_key)

        if labeled:
            labeled_images.append(
                {
                    "sample_id": sample_id,
                    "file_name": file_name,
                    "width": width,
                    "height": height,
                    "split": split,
                }
            )

    return labeled_images


def _validate_category_mapping(
    categories: Mapping[CategoryId, str],
) -> dict[CategoryId, str]:
    if not isinstance(categories, Mapping):
        raise ValueError("categories must map source category IDs to names")

    normalized: dict[CategoryId, str] = {}
    seen_names: set[str] = set()
    for raw_id, raw_name in categories.items():
        category_id = _category_id(raw_id, field="category ID")
        if not isinstance(raw_name, str) or not raw_name:
            raise ValueError("Category names must be non-empty strings")
        if raw_name not in SUPPORTED_SOURCE_LABELS:
            raise ValueError(f"Unsupported source category name {raw_name!r}")
        if raw_name in seen_names:
            raise ValueError(f"Duplicate source category name {raw_name!r}")
        normalized[category_id] = raw_name
        seen_names.add(raw_name)

    if not normalized:
        raise ValueError("At least one source category is required")
    return normalized


def _validate_bbox(
    bbox: Any,
    *,
    image_width: int,
    image_height: int,
    annotation_index: int,
) -> tuple[int, int, int, int]:
    if not isinstance(bbox, list) or len(bbox) != 4:
        raise ValueError(
            f"Annotation {annotation_index} bbox must contain four finite "
            "integer-valued numbers"
        )

    integer_bbox = []
    for value in bbox:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(
                f"Annotation {annotation_index} bbox must contain four finite "
                "integer-valued numbers"
            )
        if isinstance(value, float) and (
            not math.isfinite(value) or not value.is_integer()
        ):
            raise ValueError(
                f"Annotation {annotation_index} bbox must contain four finite "
                "integer-valued numbers"
            )
        integer_bbox.append(int(value))

    x, y, width, height = integer_bbox
    if x < 0 or y < 0 or width <= 0 or height <= 0:
        raise ValueError(
            f"Annotation {annotation_index} bbox must have non-negative coordinates "
            "and positive dimensions"
        )
    if width % 8 != 0:
        raise ValueError(
            f"Annotation {annotation_index} bbox width must be divisible by 8"
        )
    if x + width > image_width or y + height > image_height:
        raise ValueError(
            f"Annotation {annotation_index} bbox lies outside the source image"
        )
    return x, y, width, height


def _decode_roi_mask(
    encoded_mask: Any,
    *,
    width: int,
    height: int,
    annotation_index: int,
) -> np.ndarray:
    if not isinstance(encoded_mask, str):
        raise ValueError(f"Annotation {annotation_index} pixel_image must be base64")
    try:
        decoded = base64.b64decode(encoded_mask, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(
            f"Annotation {annotation_index} pixel_image contains invalid base64"
        ) from error

    expected_bytes = width * height // 8
    if len(decoded) != expected_bytes:
        raise ValueError(
            f"Annotation {annotation_index} decoded byte-length mismatch: "
            f"expected {expected_bytes}, got {len(decoded)}"
        )

    unpacked = np.unpackbits(
        np.frombuffer(decoded, dtype=np.uint8),
        bitorder="big",
    )
    return unpacked.reshape(height, width).astype(bool, copy=False)


def _validate_saved_mask(
    mask_path: Path,
    *,
    expected_width: int,
    expected_height: int,
) -> None:
    with Image.open(mask_path) as saved_mask:
        if saved_mask.mode != "L":
            raise ValueError(
                f"Output mask must be single-channel, got mode {saved_mask.mode!r}"
            )
        if saved_mask.size != (expected_width, expected_height):
            raise ValueError("Output mask dimensions changed during PNG serialization")
        mask_array = np.asarray(saved_mask)

    if mask_array.ndim != 2:
        raise ValueError("Output mask must be a two-dimensional single-channel image")
    values = set(int(value) for value in np.unique(mask_array))
    if not values <= set(CLASS_MAPPING.values()):
        raise ValueError(f"Output mask contains invalid class values: {sorted(values)}")


def _install_staged_output(
    staging_root: Path,
    destination_root: Path,
    *,
    overwrite: bool,
) -> None:
    if destination_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root appeared during conversion: {destination_root}"
            )
        if destination_root.is_symlink() or not destination_root.is_dir():
            raise ValueError("Existing output root must be a non-symlink directory")
        shutil.rmtree(destination_root)
    staging_root.replace(destination_root)


def _load_json(path: Path) -> Any:
    try:
        contents = path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeError) as error:
        raise ValueError(f"Could not read JSON file {path.name!r}") from error
    try:
        return json.loads(
            contents,
            object_pairs_hook=_mapping_without_duplicate_keys,
            parse_constant=_reject_nonstandard_json_constant,
        )
    except (json.JSONDecodeError, ValueError) as error:
        raise ValueError(f"Malformed JSON in {path.name!r}") from error


def _mapping_without_duplicate_keys(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key {key!r}")
        result[key] = value
    return result


def _reject_nonstandard_json_constant(value: str) -> None:
    raise ValueError(f"Non-standard JSON constant {value!r}")


def _required_value(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> Any:
    if key not in mapping:
        raise ValueError(f"{context} is missing required field {key!r}")
    return mapping[key]


def _required_string(
    mapping: Mapping[str, Any],
    key: str,
    *,
    context: str,
) -> str:
    value = _required_value(mapping, key, context=context)
    if not isinstance(value, str):
        raise ValueError(f"{context} field {key!r} must be a string")
    return value


def _positive_integer(value: Any, *, field: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{field} must be a positive integer")
    return value


def _category_id(value: Any, *, field: str) -> CategoryId:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be an integer or string")
    if isinstance(value, str):
        _safe_identifier(value, field=field)
    return value


def _sample_id(value: Any, *, field: str) -> CategoryId:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{field} must be an integer or string")
    if isinstance(value, str):
        _safe_identifier(value, field=field)
    return value


def _safe_identifier(value: str, *, field: str) -> str:
    if (
        not value
        or value in {".", ".."}
        or "/" in value
        or "\\" in value
        or ":" in value
    ):
        raise ValueError(f"{field} must be a path-free identifier")
    return value


def _safe_filename(value: str, *, field: str) -> str:
    if not value or value in {".", ".."} or Path(value).name != value:
        raise ValueError(f"{field} must be a filename without directory components")
    return value


def _paths_overlap(first: Path, second: Path) -> bool:
    return first == second or first in second.parents or second in first.parents


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert a user-cleared VisionTrain export to normalized segmentation "
            "images, masks, and a sanitized manifest."
        )
    )
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory after successful conversion",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the VisionTrain export converter CLI."""

    arguments = _build_argument_parser().parse_args(argv)
    manifest = convert_vt_export(
        arguments.input_root,
        arguments.output_root,
        overwrite=arguments.overwrite,
    )
    counts = manifest["sample_counts"]
    print(
        "Converted "
        f"{counts['train']} train and {counts['val']} val samples "
        f"to {arguments.output_root}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
