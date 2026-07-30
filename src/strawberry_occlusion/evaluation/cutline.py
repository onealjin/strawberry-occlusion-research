"""Dataset runner for ground-truth visible-mask cutline baseline v1."""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from statistics import median
from typing import Any

import numpy as np
from PIL import Image

from strawberry_occlusion.geometry import (
    LineSegment,
    MaskCutlineResult,
    Point,
    estimate_visible_mask_cutline,
)
from strawberry_occlusion.visualization.cutline import (
    create_mask_cutline_visualization,
)


PathLike = str | Path
Estimator = Callable[..., MaskCutlineResult]
VisualizationFunction = Callable[..., Image.Image]

CLASS_MAPPING = {"background": 0, "Flesh": 1, "Calyx": 2}
ALGORITHM_NAME = "ground_truth_visible_mask_cutline"
ALGORITHM_VERSION = "1"
SUPPORTED_SPLITS = ("train", "val")
IMAGE_SUFFIXES = frozenset({".bmp", ".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp"})
MASK_SUFFIXES = frozenset({".bmp", ".png", ".tif", ".tiff"})

CSV_FIELDS = (
    "sample_id",
    "status",
    "failure_code",
    "failure_reason",
    "source_width",
    "source_height",
    "selected_flesh_component",
    "selected_calyx_component",
    "flesh_component_count",
    "calyx_component_count",
    "contact_pixel_count",
    "contact_component_count",
    "contact_bbox_width",
    "contact_bbox_height",
    "contact_spatial_extent_pixels",
    "flesh_centroid_x",
    "flesh_centroid_y",
    "attachment_anchor_x",
    "attachment_anchor_y",
    "final_offset_anchor_x",
    "final_offset_anchor_y",
    "anchor_to_nearest_flesh_boundary_pixels",
    "normalized_anchor_depth",
    "anchor_to_flesh_centroid_pixels",
    "direction_x",
    "direction_y",
    "candidate_start_x",
    "candidate_start_y",
    "candidate_end_x",
    "candidate_end_y",
    "final_start_x",
    "final_start_y",
    "final_end_x",
    "final_end_y",
    "candidate_angle_degrees",
    "final_angle_degrees",
    "candidate_final_lines_coincide",
    "flesh_loss_proxy_pixel_count",
    "flesh_loss_proxy_ratio",
    "calyx_retention_proxy_pixel_count",
    "calyx_retention_proxy_ratio",
    "calyx_dilation_radius",
    "component_connectivity",
    "signed_offset",
)

STATUS_DEFINITION = (
    "ok means finite baseline-v1 geometry was produced; it does not classify "
    "the cutline as geometrically good or physically accurate"
)
PROXY_DEFINITION = (
    "Mask-side measurements use all Flesh and Calyx pixels in the visible "
    "semantic mask, not only the selected connected-component pair. Detached "
    "fragments or additional fruit components may influence these whole-mask "
    "geometric proxies. They are not physical cutting results or true cutline "
    "accuracy"
)
LINE_SIDE_CONVENTION = (
    "For pixel centre p, dot(p - final_offset_anchor, "
    "fruit_to_attachment_direction) > 0 is the Calyx/removal side; "
    "<= 0 is the Flesh/retained side, so pixels on the line are retained"
)
DIAGNOSTIC_DEFINITIONS = {
    "contact_bbox": (
        "inclusive width and height of the contact-band pixel-centre bounding box"
    ),
    "contact_spatial_extent_pixels": (
        "Euclidean diagonal between opposite extrema of the contact-band "
        "pixel-centre bounding box"
    ),
    "anchor_to_nearest_flesh_boundary_pixels": (
        "Euclidean distance from the attachment anchor to the nearest complete "
        "Flesh-mask boundary pixel centre"
    ),
    "normalized_anchor_depth": (
        "anchor-to-boundary distance divided by the square root of the selected "
        "Flesh component pixel count"
    ),
    "anchor_to_flesh_centroid_pixels": (
        "Euclidean distance from the attachment anchor to the selected Flesh "
        "component centroid"
    ),
    "candidate_final_lines_coincide": (
        "endpoint-order-invariant equality of candidate and final clipped segments"
    ),
    "signed_offset_pixels": (
        "positive moves along the fruit-to-attachment direction; negative moves "
        "toward the selected Flesh centroid"
    ),
}


def run_visible_mask_cutline_dataset(
    dataset_root: PathLike,
    output_root: PathLike,
    *,
    split: str = "train",
    calyx_dilation_radius: int = 1,
    component_connectivity: int = 8,
    signed_offset: float = 0.0,
    overwrite: bool = False,
    estimator: Estimator = estimate_visible_mask_cutline,
    visualization_function: VisualizationFunction = (create_mask_cutline_visualization),
) -> dict[str, Any]:
    """Run baseline v1 at source resolution for one normalized dataset split."""

    parameters = _validate_runner_parameters(
        split=split,
        calyx_dilation_radius=calyx_dilation_radius,
        component_connectivity=component_connectivity,
        signed_offset=signed_offset,
    )
    source_root = Path(dataset_root)
    destination_root = Path(output_root)
    _validate_output_destination(
        source_root,
        destination_root,
        overwrite=overwrite,
    )
    pairs = _paired_paths(source_root, split=split)

    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.staging-",
            dir=destination_root.parent,
        )
    )
    try:
        visualization_root = staging_root / "visualizations"
        visualization_root.mkdir()
        rows: list[dict[str, Any]] = []
        manifest_samples: list[dict[str, Any]] = []
        for image_path, mask_path in pairs:
            image, mask = _read_source_pair(image_path, mask_path)
            result = estimator(
                mask,
                calyx_dilation_radius=calyx_dilation_radius,
                component_connectivity=component_connectivity,
                signed_offset=signed_offset,
            )
            _validate_estimator_result(result, mask_shape=mask.shape)
            diagnostics = calculate_cutline_diagnostics(mask, result)
            visualization = visualization_function(
                mask,
                result,
                image=image,
                diagnostics=diagnostics,
            )
            if not isinstance(visualization, Image.Image):
                raise TypeError("visualization_function must return a PIL.Image.Image")
            relative_visualization = Path("visualizations") / f"{image_path.stem}.png"
            visualization.save(
                staging_root / relative_visualization,
                format="PNG",
            )
            row = _geometry_row(
                image_path.stem,
                result,
                diagnostics,
            )
            rows.append(row)
            manifest_samples.append(
                {
                    "sample_id": image_path.stem,
                    "status": result.status,
                    "visualization_path": relative_visualization.as_posix(),
                }
            )

        summary = _build_summary(
            split=split,
            rows=rows,
            parameter_configuration=parameters,
        )
        manifest: dict[str, Any] = {
            "dataset_basename": source_root.name,
            "split": split,
            "sample_count": len(rows),
            "class_mapping": dict(CLASS_MAPPING),
            "algorithm_name": ALGORITHM_NAME,
            "algorithm_version": ALGORITHM_VERSION,
            "parameter_configuration": parameters,
            "artifacts": {
                "summary": "summary.json",
                "per_image_geometry": "per_image_geometry.csv",
                "visualizations": "visualizations",
            },
            "samples": manifest_samples,
        }
        _write_geometry_csv(staging_root / "per_image_geometry.csv", rows)
        _write_json(staging_root / "summary.json", summary)
        _write_json(staging_root / "manifest.json", manifest)
        _install_staged_output(
            staging_root,
            destination_root,
            overwrite=overwrite,
        )
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    return {"manifest": manifest, "summary": summary, "rows": rows}


def calculate_line_side_proxies(
    mask: np.ndarray,
    result: MaskCutlineResult,
) -> dict[str, int | float | None]:
    """Calculate visible-mask proxies using the documented line-side convention.

    For a pixel centre ``p``, positive signed side
    ``dot(p - final_offset_anchor, fruit_to_attachment_direction)`` is the
    Calyx/removal side. Zero and negative values are the Flesh/retained side.
    Counts and ratios use all Flesh and Calyx pixels in the semantic mask, not
    only the selected connected-component pair. Detached fragments or additional
    fruit components may therefore influence these whole-mask geometric proxies.
    They are not physical cutting results or true cutline accuracy.
    """

    mask_array = _validate_mask_array(mask)
    _validate_estimator_result(result, mask_shape=mask_array.shape)
    empty: dict[str, int | float | None] = {
        "flesh_loss_proxy_pixel_count": None,
        "flesh_loss_proxy_ratio": None,
        "calyx_retention_proxy_pixel_count": None,
        "calyx_retention_proxy_ratio": None,
    }
    if (
        not result.succeeded
        or result.final_cutline is None
        or result.offset_anchor is None
        or result.fruit_to_attachment_direction is None
    ):
        return empty

    direction_x, direction_y = result.fruit_to_attachment_direction
    pixel_y, pixel_x = np.indices(mask_array.shape, dtype=np.float64)
    signed_side = (pixel_x - result.offset_anchor.x) * direction_x + (
        pixel_y - result.offset_anchor.y
    ) * direction_y
    removal_side = signed_side > 0.0
    retained_side = ~removal_side
    flesh = mask_array == CLASS_MAPPING["Flesh"]
    calyx = mask_array == CLASS_MAPPING["Calyx"]
    flesh_total = int(np.count_nonzero(flesh))
    calyx_total = int(np.count_nonzero(calyx))
    flesh_loss = int(np.count_nonzero(flesh & removal_side))
    calyx_retention = int(np.count_nonzero(calyx & retained_side))
    return {
        "flesh_loss_proxy_pixel_count": flesh_loss,
        "flesh_loss_proxy_ratio": (flesh_loss / flesh_total if flesh_total else None),
        "calyx_retention_proxy_pixel_count": calyx_retention,
        "calyx_retention_proxy_ratio": (
            calyx_retention / calyx_total if calyx_total else None
        ),
    }


def calculate_cutline_diagnostics(
    mask: np.ndarray,
    result: MaskCutlineResult,
) -> dict[str, Any]:
    """Calculate runner diagnostics without changing baseline-v1 geometry."""

    mask_array = _validate_mask_array(mask)
    _validate_estimator_result(result, mask_shape=mask_array.shape)
    contact_coordinates = np.argwhere(result.contact_band)
    if contact_coordinates.size:
        min_y, min_x = contact_coordinates.min(axis=0)
        max_y, max_x = contact_coordinates.max(axis=0)
        contact_bbox_width: int | None = int(max_x - min_x + 1)
        contact_bbox_height: int | None = int(max_y - min_y + 1)
        contact_extent: float | None = math.hypot(
            float(max_x - min_x),
            float(max_y - min_y),
        )
    else:
        contact_bbox_width = None
        contact_bbox_height = None
        contact_extent = None

    boundary_distance: float | None = None
    normalized_anchor_depth: float | None = None
    anchor_centroid_distance: float | None = None
    if result.attachment_anchor is not None:
        boundary = flesh_boundary_mask(result.flesh_mask)
        boundary_coordinates = np.argwhere(boundary)
        if boundary_coordinates.size:
            delta_x = (
                boundary_coordinates[:, 1].astype(np.float64)
                - result.attachment_anchor.x
            )
            delta_y = (
                boundary_coordinates[:, 0].astype(np.float64)
                - result.attachment_anchor.y
            )
            boundary_distance = float(
                np.sqrt(delta_x * delta_x + delta_y * delta_y).min()
            )
            if result.selected_flesh_pixel_count > 0:
                normalized_anchor_depth = boundary_distance / math.sqrt(
                    result.selected_flesh_pixel_count
                )
        if result.flesh_centroid is not None:
            anchor_centroid_distance = math.hypot(
                result.attachment_anchor.x - result.flesh_centroid.x,
                result.attachment_anchor.y - result.flesh_centroid.y,
            )

    diagnostics: dict[str, Any] = {
        "contact_bbox_width": contact_bbox_width,
        "contact_bbox_height": contact_bbox_height,
        "contact_spatial_extent_pixels": contact_extent,
        "anchor_to_nearest_flesh_boundary_pixels": boundary_distance,
        "normalized_anchor_depth": normalized_anchor_depth,
        "anchor_to_flesh_centroid_pixels": anchor_centroid_distance,
        "candidate_angle_degrees": _line_angle(result.candidate_cutline),
        "final_angle_degrees": _line_angle(result.final_cutline),
        "candidate_final_lines_coincide": _segments_coincide(
            result.candidate_cutline,
            result.final_cutline,
        ),
        "signed_offset_pixels": result.parameters.signed_offset,
    }
    diagnostics.update(calculate_line_side_proxies(mask_array, result))
    return diagnostics


def flesh_boundary_mask(flesh_mask: np.ndarray) -> np.ndarray:
    """Return the four-neighbour boundary of a binary Flesh mask."""

    binary = np.asarray(flesh_mask)
    if binary.ndim != 2 or binary.dtype != np.bool_:
        raise ValueError("flesh_mask must be a two-dimensional boolean array")
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    interior = (
        binary
        & padded[:-2, 1:-1]
        & padded[2:, 1:-1]
        & padded[1:-1, :-2]
        & padded[1:-1, 2:]
    )
    return binary & ~interior


def _paired_paths(dataset_root: Path, *, split: str) -> list[tuple[Path, Path]]:
    image_dir = dataset_root / split / "images"
    mask_dir = dataset_root / split / "masks"
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist for split {split!r}")
    if not mask_dir.is_dir():
        raise FileNotFoundError(f"Mask directory does not exist for split {split!r}")

    images = _files_by_stem(image_dir, suffixes=IMAGE_SUFFIXES, kind="image")
    masks = _files_by_stem(mask_dir, suffixes=MASK_SUFFIXES, kind="mask")
    if not images and not masks:
        raise ValueError(f"No supported image/mask pairs found for split {split!r}")
    missing_masks = sorted(set(images) - set(masks))
    missing_images = sorted(set(masks) - set(images))
    if missing_masks:
        raise ValueError(f"Missing masks for image stem(s): {', '.join(missing_masks)}")
    if missing_images:
        raise ValueError(
            f"Missing images for mask stem(s): {', '.join(missing_images)}"
        )
    return [(images[stem], masks[stem]) for stem in sorted(images)]


def _files_by_stem(
    folder: Path,
    *,
    suffixes: frozenset[str],
    kind: str,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for path in sorted(folder.iterdir()):
        if not path.is_file() or path.suffix.lower() not in suffixes:
            continue
        if path.stem in paths:
            raise ValueError(f"Duplicate {kind} stem is ambiguous: {path.stem}")
        paths[path.stem] = path
    return paths


def _read_source_pair(
    image_path: Path,
    mask_path: Path,
) -> tuple[Image.Image, np.ndarray]:
    try:
        with Image.open(image_path) as image_file, Image.open(mask_path) as mask_file:
            if image_file.size != mask_file.size:
                raise ValueError(
                    "Image and mask dimensions differ for "
                    f"{image_path.name}: image={image_file.size}, "
                    f"mask={mask_file.size}"
                )
            image = image_file.convert("RGB").copy()
            mask = np.array(mask_file, copy=True)
    except OSError as error:
        raise ValueError(f"Could not read sample {image_path.stem!r}") from error
    return image, _validate_mask_array(mask)


def _validate_mask_array(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(
            f"Mask must be single-channel with shape [H, W], got {array.shape}"
        )
    if np.issubdtype(array.dtype, np.bool_) or not np.issubdtype(
        array.dtype, np.integer
    ):
        raise TypeError(f"Mask must use an integer dtype, got {array.dtype}")
    valid = np.isin(array, tuple(CLASS_MAPPING.values()))
    if not np.all(valid):
        invalid = np.unique(array[~valid]).tolist()
        raise ValueError(
            f"Mask must contain only class IDs 0, 1, and 2, found {invalid}"
        )
    return array


def _validate_estimator_result(
    result: MaskCutlineResult,
    *,
    mask_shape: tuple[int, int],
) -> None:
    if not isinstance(result, MaskCutlineResult):
        raise TypeError("estimator must return a MaskCutlineResult")
    if (result.image_height, result.image_width) != mask_shape:
        raise ValueError("estimator result dimensions must match the source mask")
    if result.status == "ok" and (
        result.final_cutline is None
        or result.offset_anchor is None
        or result.fruit_to_attachment_direction is None
    ):
        raise ValueError("successful estimator result is missing final geometry")


def _geometry_row(
    sample_id: str,
    result: MaskCutlineResult,
    diagnostics: Mapping[str, Any],
) -> dict[str, Any]:
    direction = result.fruit_to_attachment_direction
    final_offset_anchor = result.offset_anchor if result.succeeded else None
    return {
        "sample_id": sample_id,
        "status": result.status,
        "failure_code": result.failure_code,
        "failure_reason": result.failure_reason,
        "source_width": result.image_width,
        "source_height": result.image_height,
        "selected_flesh_component": result.selected_flesh_component,
        "selected_calyx_component": result.selected_calyx_component,
        "flesh_component_count": result.flesh_component_count,
        "calyx_component_count": result.calyx_component_count,
        "contact_pixel_count": result.contact_pixel_count,
        "contact_component_count": result.contact_component_count,
        "contact_bbox_width": diagnostics["contact_bbox_width"],
        "contact_bbox_height": diagnostics["contact_bbox_height"],
        "contact_spatial_extent_pixels": diagnostics["contact_spatial_extent_pixels"],
        "flesh_centroid_x": _point_value(result.flesh_centroid, "x"),
        "flesh_centroid_y": _point_value(result.flesh_centroid, "y"),
        "attachment_anchor_x": _point_value(result.attachment_anchor, "x"),
        "attachment_anchor_y": _point_value(result.attachment_anchor, "y"),
        "final_offset_anchor_x": _point_value(final_offset_anchor, "x"),
        "final_offset_anchor_y": _point_value(final_offset_anchor, "y"),
        "anchor_to_nearest_flesh_boundary_pixels": diagnostics[
            "anchor_to_nearest_flesh_boundary_pixels"
        ],
        "normalized_anchor_depth": diagnostics["normalized_anchor_depth"],
        "anchor_to_flesh_centroid_pixels": diagnostics[
            "anchor_to_flesh_centroid_pixels"
        ],
        "direction_x": direction[0] if direction is not None else None,
        "direction_y": direction[1] if direction is not None else None,
        "candidate_start_x": _segment_value(result.candidate_cutline, "start", "x"),
        "candidate_start_y": _segment_value(result.candidate_cutline, "start", "y"),
        "candidate_end_x": _segment_value(result.candidate_cutline, "end", "x"),
        "candidate_end_y": _segment_value(result.candidate_cutline, "end", "y"),
        "final_start_x": _segment_value(result.final_cutline, "start", "x"),
        "final_start_y": _segment_value(result.final_cutline, "start", "y"),
        "final_end_x": _segment_value(result.final_cutline, "end", "x"),
        "final_end_y": _segment_value(result.final_cutline, "end", "y"),
        "candidate_angle_degrees": diagnostics["candidate_angle_degrees"],
        "final_angle_degrees": diagnostics["final_angle_degrees"],
        "candidate_final_lines_coincide": diagnostics["candidate_final_lines_coincide"],
        "flesh_loss_proxy_pixel_count": diagnostics["flesh_loss_proxy_pixel_count"],
        "flesh_loss_proxy_ratio": diagnostics["flesh_loss_proxy_ratio"],
        "calyx_retention_proxy_pixel_count": diagnostics[
            "calyx_retention_proxy_pixel_count"
        ],
        "calyx_retention_proxy_ratio": diagnostics["calyx_retention_proxy_ratio"],
        "calyx_dilation_radius": result.parameters.calyx_dilation_radius,
        "component_connectivity": result.parameters.component_connectivity,
        "signed_offset": result.parameters.signed_offset,
    }


def _build_summary(
    *,
    split: str,
    rows: list[dict[str, Any]],
    parameter_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    success_rows = [row for row in rows if row["status"] == "ok"]
    failure_rows = [row for row in rows if row["status"] != "ok"]
    failure_counts = Counter(
        row["failure_code"] or "unknown_failure" for row in failure_rows
    )
    return {
        "evaluated_split": split,
        "total_sample_count": len(rows),
        "success_count": len(success_rows),
        "failure_count": len(failure_rows),
        "failure_counts_by_reason": dict(sorted(failure_counts.items())),
        "status_definition": STATUS_DEFINITION,
        "proxy_definition": PROXY_DEFINITION,
        "line_side_sign_convention": LINE_SIDE_CONVENTION,
        "diagnostic_definitions": dict(DIAGNOSTIC_DEFINITIONS),
        "aggregate_contact_size_statistics": _numeric_statistics(
            [row["contact_pixel_count"] for row in rows]
        ),
        "aggregate_contact_component_statistics": _numeric_statistics(
            [row["contact_component_count"] for row in rows]
        ),
        "aggregate_flesh_loss_proxy_statistics": {
            "pixel_count": _numeric_statistics(
                [row["flesh_loss_proxy_pixel_count"] for row in success_rows]
            ),
            "ratio": _numeric_statistics(
                [row["flesh_loss_proxy_ratio"] for row in success_rows]
            ),
        },
        "aggregate_calyx_retention_proxy_statistics": {
            "pixel_count": _numeric_statistics(
                [row["calyx_retention_proxy_pixel_count"] for row in success_rows]
            ),
            "ratio": _numeric_statistics(
                [row["calyx_retention_proxy_ratio"] for row in success_rows]
            ),
        },
        "parameter_configuration": dict(parameter_configuration),
    }


def _numeric_statistics(values: Sequence[int | float | None]) -> dict[str, Any]:
    finite = [
        float(value)
        for value in values
        if value is not None and math.isfinite(float(value))
    ]
    if not finite:
        return {
            "sample_count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }
    return {
        "sample_count": len(finite),
        "minimum": min(finite),
        "maximum": max(finite),
        "mean": sum(finite) / len(finite),
        "median": median(finite),
    }


def _line_angle(segment: LineSegment | None) -> float | None:
    if segment is None:
        return None
    delta_x = segment.end.x - segment.start.x
    delta_y = segment.end.y - segment.start.y
    angle = math.degrees(math.atan2(delta_y, delta_x)) % 180.0
    return 0.0 if math.isclose(angle, 180.0, abs_tol=1e-12) else angle


def _segments_coincide(
    first: LineSegment | None,
    second: LineSegment | None,
    *,
    tolerance: float = 1e-9,
) -> bool | None:
    if first is None or second is None:
        return None

    def close(left: Point, right: Point) -> bool:
        return math.hypot(left.x - right.x, left.y - right.y) <= tolerance

    return (close(first.start, second.start) and close(first.end, second.end)) or (
        close(first.start, second.end) and close(first.end, second.start)
    )


def _point_value(point: Point | None, coordinate: str) -> float | None:
    return getattr(point, coordinate) if point is not None else None


def _segment_value(
    segment: LineSegment | None,
    endpoint: str,
    coordinate: str,
) -> float | None:
    if segment is None:
        return None
    return getattr(getattr(segment, endpoint), coordinate)


def _write_geometry_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in CSV_FIELDS})


def _csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, float):
        return format(value, ".12g")
    return value


def _write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(
            value,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _validate_runner_parameters(
    *,
    split: str,
    calyx_dilation_radius: int,
    component_connectivity: int,
    signed_offset: float,
) -> dict[str, Any]:
    if split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"split must be one of {', '.join(SUPPORTED_SPLITS)}, got {split!r}"
        )
    if (
        isinstance(calyx_dilation_radius, bool)
        or not isinstance(calyx_dilation_radius, int)
        or calyx_dilation_radius < 1
    ):
        raise ValueError("calyx_dilation_radius must be an integer of at least 1")
    if (
        isinstance(component_connectivity, bool)
        or not isinstance(component_connectivity, int)
        or component_connectivity not in (4, 8)
    ):
        raise ValueError("component_connectivity must be 4 or 8")
    if (
        isinstance(signed_offset, bool)
        or not isinstance(signed_offset, (int, float))
        or not math.isfinite(signed_offset)
    ):
        raise ValueError("signed_offset must be a finite real number")
    return {
        "calyx_dilation_radius": calyx_dilation_radius,
        "component_connectivity": component_connectivity,
        "signed_offset_pixels": float(signed_offset),
    }


def _validate_output_destination(
    dataset_root: Path,
    output_root: Path,
    *,
    overwrite: bool,
) -> None:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")
    dataset_resolved = dataset_root.resolve()
    output_resolved = output_root.resolve()
    if _paths_overlap(dataset_resolved, output_resolved):
        raise ValueError("Dataset and output roots must not overlap")
    if output_root.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output root already exists: {output_root}. "
                "Pass overwrite=True or --overwrite to replace it."
            )
        if output_root.is_symlink() or not output_root.is_dir():
            raise ValueError("Existing output root must be a non-symlink directory")


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
        description=(
            "Run ground-truth visible-mask cutline baseline v1 at source resolution."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="train")
    parser.add_argument("--calyx-dilation-radius", type=int, default=1)
    parser.add_argument(
        "--component-connectivity",
        type=int,
        choices=(4, 8),
        default=8,
    )
    parser.add_argument("--signed-offset", type=float, default=0.0)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory after successful generation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the visible-mask cutline dataset CLI."""

    arguments = _build_argument_parser().parse_args(argv)
    output = run_visible_mask_cutline_dataset(
        arguments.dataset_root,
        arguments.output_root,
        split=arguments.split,
        calyx_dilation_radius=arguments.calyx_dilation_radius,
        component_connectivity=arguments.component_connectivity,
        signed_offset=arguments.signed_offset,
        overwrite=arguments.overwrite,
    )
    summary = output["summary"]
    print(
        f"Processed {summary['total_sample_count']} {summary['evaluated_split']} "
        f"samples: {summary['success_count']} finite geometries, "
        f"{summary['failure_count']} structured failures"
    )
    print(STATUS_DEFINITION)
    print(f"Artifacts written to {arguments.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
