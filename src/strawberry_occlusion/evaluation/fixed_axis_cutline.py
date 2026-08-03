"""Dataset runner comparing baseline-v1, fixed-axis-v2a, and v2b cutlines."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import tempfile
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from strawberry_occlusion.evaluation.cutline import (
    ALGORITHM_NAME as V1_ALGORITHM_NAME,
    ALGORITHM_VERSION as V1_ALGORITHM_VERSION,
    CLASS_MAPPING,
    LINE_SIDE_CONVENTION as V1_LINE_SIDE_CONVENTION,
    PROXY_DEFINITION,
    SUPPORTED_SPLITS,
    _csv_value,
    _install_staged_output,
    _numeric_statistics,
    _paired_paths,
    _read_source_pair,
    _validate_mask_array,
    _validate_output_destination,
    _write_json,
    calculate_line_side_proxies,
)
from strawberry_occlusion.geometry import (
    FixedAxisCutlineResult,
    FixedAxisSearchCutlineResult,
    LineSegment,
    MaskCutlineResult,
    Point,
    estimate_fixed_axis_cutline,
    estimate_fixed_axis_search_cutline,
    estimate_visible_mask_cutline,
)
from strawberry_occlusion.geometry.fixed_axis_cutline import (
    _validate_finite_real,
    _validate_removal_axis,
)
from strawberry_occlusion.geometry.fixed_axis_search_cutline import (
    _validate_search_parameters,
)
from strawberry_occlusion.visualization.fixed_axis_comparison import (
    create_fixed_axis_comparison_visualization,
)


PathLike = str | Path
V1Estimator = Callable[..., MaskCutlineResult]
V2aEstimator = Callable[..., FixedAxisCutlineResult]
V2bEstimator = Callable[..., FixedAxisSearchCutlineResult]
ComparisonVisualizationFunction = Callable[..., Image.Image]

V2A_ALGORITHM_NAME = "fixed_axis_robust_contact_cutline"
V2A_ALGORITHM_VERSION = "2a"
V2B_ALGORITHM_NAME = "fixed_axis_outermost_feasible_cutline"
V2B_ALGORITHM_VERSION = "2b"
V1_SIGNED_OFFSET_PIXELS = 0.0

CSV_FIELDS = (
    "sample_id",
    "source_width",
    "source_height",
    "v1_status",
    "v1_failure_code",
    "v1_failure_reason",
    "v2a_status",
    "v2a_failure_code",
    "v2a_failure_reason",
    "flesh_component_count",
    "calyx_component_count",
    "selected_flesh_component",
    "selected_calyx_component",
    "contact_pixel_count",
    "contact_component_count",
    "v1_attachment_anchor_x",
    "v1_attachment_anchor_y",
    "v1_direction_x",
    "v1_direction_y",
    "v1_line_angle_degrees",
    "v1_anchor_projection_on_removal_axis",
    "v1_flesh_loss_pixels",
    "v1_flesh_loss_ratio",
    "v1_calyx_retention_pixels",
    "v1_calyx_retention_ratio",
    "v1_cut_margin_to_flesh_extent",
    "removal_axis_x",
    "removal_axis_y",
    "projection_quantile",
    "support_band_width_pixels",
    "support_pixel_count",
    "support_lateral_extent_pixels",
    "v2a_unshifted_cut_coordinate",
    "v2a_final_cut_coordinate",
    "v2a_reference_point_x",
    "v2a_reference_point_y",
    "v2a_candidate_start_x",
    "v2a_candidate_start_y",
    "v2a_candidate_end_x",
    "v2a_candidate_end_y",
    "v2a_final_start_x",
    "v2a_final_start_y",
    "v2a_final_end_x",
    "v2a_final_end_y",
    "v2a_axis_disagreement_degrees",
    "v2a_cut_margin_to_flesh_extent",
    "v2a_flesh_loss_pixels",
    "v2a_flesh_loss_ratio",
    "v2a_calyx_retention_pixels",
    "v2a_calyx_retention_ratio",
    "v2a_candidate_final_lines_coincide",
    "signed_offset_pixels",
    "calyx_dilation_radius",
    "component_connectivity",
    "cut_coordinate_shift_from_v1",
    "flesh_loss_pixel_change",
    "flesh_loss_ratio_change",
    "calyx_retention_pixel_change",
    "calyx_retention_ratio_change",
    "v2a_flesh_centroid_projected_reference_x",
    "v2a_flesh_centroid_projected_reference_y",
    "v2a_flesh_centroid_reference_nearest_class",
    "v2a_flesh_centroid_reference_inside_selected_flesh",
    "v2a_flesh_centroid_reference_distance_to_selected_calyx",
    "v2a_flesh_centroid_reference_distance_to_selected_contact",
    "v2a_attachment_anchor_projected_reference_x",
    "v2a_attachment_anchor_projected_reference_y",
    "v2a_attachment_anchor_reference_nearest_class",
    "v2a_attachment_anchor_reference_inside_selected_flesh",
    "v2a_attachment_anchor_reference_distance_to_selected_calyx",
    "v2a_attachment_anchor_reference_distance_to_selected_contact",
    "v2b_status",
    "v2b_failure_code",
    "v2b_failure_reason",
    "v2b_selected_flesh_component",
    "v2b_selected_calyx_component",
    "v2b_selected_flesh_pixel_count",
    "v2b_selected_calyx_pixel_count",
    "v2b_contact_pixel_count",
    "v2b_blade_tangent_x",
    "v2b_blade_tangent_y",
    "v2b_v1_anchor_projection",
    "v2b_v2a_cut_coordinate",
    "v2b_lateral_reference_coordinate",
    "v2b_flesh_centroid_lateral_reference_coordinate",
    "v2b_search_interval_start",
    "v2b_search_interval_end",
    "v2b_candidate_count",
    "v2b_feasible_candidate_count",
    "v2b_feasible_block_count",
    "v2b_feasible_hull_start",
    "v2b_feasible_hull_end",
    "v2b_selected_candidate_index",
    "v2b_selected_candidate_feasible",
    "v2b_selected_block_id",
    "v2b_selected_block_start",
    "v2b_selected_block_end",
    "v2b_selected_block_candidate_count",
    "v2b_selected_block_is_singleton",
    "v2b_cut_coordinate",
    "v2b_candidate_start_x",
    "v2b_candidate_start_y",
    "v2b_candidate_end_x",
    "v2b_candidate_end_y",
    "v2b_final_start_x",
    "v2b_final_start_y",
    "v2b_final_end_x",
    "v2b_final_end_y",
    "v2b_reference_point_x",
    "v2b_reference_point_y",
    "v2b_maximum_local_contact_evidence",
    "v2b_selected_local_contact_count",
    "v2b_selected_local_flesh_count",
    "v2b_selected_local_calyx_count",
    "v2b_selected_local_v2a_support_count",
    "v2b_selected_pair_flesh_loss_pixels",
    "v2b_selected_pair_flesh_loss_ratio",
    "v2b_selected_pair_calyx_retention_pixels",
    "v2b_selected_pair_calyx_retention_ratio",
    "v2b_selected_pair_retained_contact_pixels",
    "v2b_selected_pair_retained_contact_ratio",
    "v2b_whole_mask_flesh_loss_pixels",
    "v2b_whole_mask_flesh_loss_ratio",
    "v2b_whole_mask_calyx_retention_pixels",
    "v2b_whole_mask_calyx_retention_ratio",
    "v2b_coordinate_shift_from_v1",
    "v2b_coordinate_shift_from_v2a",
    "v2b_lies_between_v1_and_v2a",
    "v2b_cut_margin_to_flesh_extent",
    "v2b_candidate_final_lines_coincide",
    "v2b_minus_v1_flesh_loss_pixel_change",
    "v2b_minus_v1_flesh_loss_ratio_change",
    "v2b_minus_v1_calyx_retention_pixel_change",
    "v2b_minus_v1_calyx_retention_ratio_change",
    "v2b_minus_v2a_flesh_loss_pixel_change",
    "v2b_minus_v2a_flesh_loss_ratio_change",
    "v2b_minus_v2a_calyx_retention_pixel_change",
    "v2b_minus_v2a_calyx_retention_ratio_change",
)

CANDIDATE_CSV_FIELDS = (
    "coordinate",
    "local_contact_count",
    "local_flesh_count",
    "local_calyx_count",
    "local_v2a_support_count",
    "selected_flesh_loss_ratio",
    "selected_calyx_retention_ratio",
    "retained_contact_ratio",
    "feasible",
    "feasible_block_id",
)

V2A_LINE_SIDE_CONVENTION = (
    "For pixel centre p, dot(p - diagnostic_reference_point, "
    "normalized_removal_axis) > 0 is the Calyx/removal side; <= 0 is the "
    "Flesh/retained side, so pixels on the line are retained"
)
V2B_LINE_SIDE_CONVENTION = (
    "For pixel centre p, dot(p - diagnostic_reference_point, "
    "normalized_removal_axis) > 0 is the Calyx/removal side; <= 0 is the "
    "Flesh/retained side, so pixels on the line are retained"
)
WHOLE_MASK_PROXY_SCOPE = (
    "Runner v1, v2a, and v2b line-side proxies use all Flesh and Calyx pixels "
    "in the semantic mask. They are geometric mask proxies, not physical "
    "cutting results or true cutline-accuracy measurements"
)
SELECTED_PAIR_PROXY_SCOPE = (
    "V2b decision proxies use only its selected Flesh/Calyx component pair "
    "and selected contact mask; they are separate from whole-mask proxies"
)
FEASIBLE_BLOCK_DEFINITION = (
    "A feasible block is a maximal run of feasible candidates that are adjacent "
    "in strictly increasing candidate order and separated by no more than the "
    "configured candidate step. Block IDs are zero-based in increasing-coordinate "
    "order"
)
FEASIBLE_HULL_DEFINITION = (
    "The feasible hull spans the minimum through maximum feasible coordinate; it "
    "does not imply that every intermediate candidate is feasible"
)
SELECTED_BLOCK_DEFINITION = (
    "The selected block is the feasible block containing the outermost-feasible "
    "selected candidate. Its width is selected_block_end minus "
    "selected_block_start"
)
FEASIBLE_SET_CAUTION = (
    "Fragmented or singleton feasible sets may indicate unstable local evidence "
    "and are not proof of reliable cutting"
)
CHANGE_CONVENTION = (
    "Existing v2a comparison fields are v2a minus v1; every named v2b change "
    "is v2b minus its stated comparison method. Negative Flesh-loss change "
    "and negative Calyx-retention change are improvements"
)
STATUS_DEFINITIONS = {
    "v1": (
        "ok means finite baseline-v1 geometry was produced; it does not prove "
        "that the cut is good, accurate, or physically suitable"
    ),
    "v2a": (
        "ok means finite fixed-axis-v2a geometry was produced; it does not prove "
        "that the cut is good, accurate, or physically suitable"
    ),
    "v2b": (
        "ok means finite fixed-axis-v2b research proxy geometry was produced; "
        "it does not prove attachment severance, cut quality, or accuracy"
    ),
}
COORDINATE_SHIFT_DEFINITION = (
    "v2a final cut coordinate minus the configured-axis projection of the "
    "baseline-v1 combined-contact attachment anchor"
)
CUT_MARGIN_DEFINITION = (
    "maximum configured-axis projection of the selected Flesh component minus "
    "the method's comparison coordinate; positive is inside the selected Flesh "
    "projection extent, zero is at its outer extent, and negative is beyond it"
)


def run_fixed_axis_cutline_comparison_dataset(
    dataset_root: PathLike,
    output_root: PathLike,
    *,
    removal_axis: Sequence[float],
    split: str = "train",
    projection_quantile: float = 0.95,
    support_band_width_pixels: float = 5.0,
    calyx_dilation_radius: int = 1,
    component_connectivity: int = 8,
    signed_offset_pixels: float = 0.0,
    candidate_step_pixels: float = 1.0,
    inward_search_margin_pixels: float = 10.0,
    outward_search_margin_pixels: float = 0.0,
    blade_band_half_width_pixels: float = 1.0,
    lateral_window_half_width_pixels: float = 64.0,
    minimum_attachment_evidence_fraction: float = 0.50,
    minimum_flesh_band_pixels: int = 1,
    minimum_calyx_band_pixels: int = 1,
    overwrite: bool = False,
    v1_estimator: V1Estimator = estimate_visible_mask_cutline,
    v2a_estimator: V2aEstimator = estimate_fixed_axis_cutline,
    v2b_estimator: V2bEstimator = estimate_fixed_axis_search_cutline,
    visualization_function: ComparisonVisualizationFunction = (
        create_fixed_axis_comparison_visualization
    ),
) -> dict[str, Any]:
    """Compare v1, v2a, and v2b without changing source coordinates.

    Baseline v1 is evaluated with its zero-offset configuration. The configured
    signed offset applies to v2a. Images and masks are read at source resolution
    and are never resized, cropped, rotated, or written back.
    """

    parameters = _validate_runner_parameters(
        split=split,
        removal_axis=removal_axis,
        projection_quantile=projection_quantile,
        support_band_width_pixels=support_band_width_pixels,
        calyx_dilation_radius=calyx_dilation_radius,
        component_connectivity=component_connectivity,
        signed_offset_pixels=signed_offset_pixels,
        candidate_step_pixels=candidate_step_pixels,
        inward_search_margin_pixels=inward_search_margin_pixels,
        outward_search_margin_pixels=outward_search_margin_pixels,
        blade_band_half_width_pixels=blade_band_half_width_pixels,
        lateral_window_half_width_pixels=lateral_window_half_width_pixels,
        minimum_attachment_evidence_fraction=(minimum_attachment_evidence_fraction),
        minimum_flesh_band_pixels=minimum_flesh_band_pixels,
        minimum_calyx_band_pixels=minimum_calyx_band_pixels,
    )
    configured_axis = tuple(parameters["removal_axis"])
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
        candidate_curve_root = staging_root / "candidate_curves"
        candidate_curve_root.mkdir()
        rows: list[dict[str, Any]] = []
        v2b_results: list[FixedAxisSearchCutlineResult] = []
        manifest_samples: list[dict[str, Any]] = []
        for image_path, mask_path in pairs:
            image, mask = _read_source_pair(image_path, mask_path)
            v1_result = v1_estimator(
                mask,
                calyx_dilation_radius=calyx_dilation_radius,
                component_connectivity=component_connectivity,
                signed_offset=V1_SIGNED_OFFSET_PIXELS,
            )
            v2a_result = v2a_estimator(
                mask,
                removal_axis=configured_axis,
                projection_quantile=projection_quantile,
                support_band_width_pixels=support_band_width_pixels,
                calyx_dilation_radius=calyx_dilation_radius,
                component_connectivity=component_connectivity,
                signed_offset_pixels=signed_offset_pixels,
            )
            _validate_v1_result(v1_result, mask_shape=mask.shape)
            _validate_v2a_result(v2a_result, mask_shape=mask.shape)
            v2b_result = v2b_estimator(
                mask,
                removal_axis=configured_axis,
                projection_quantile=projection_quantile,
                support_band_width_pixels=support_band_width_pixels,
                calyx_dilation_radius=calyx_dilation_radius,
                component_connectivity=component_connectivity,
                signed_offset_pixels=signed_offset_pixels,
                candidate_step_pixels=candidate_step_pixels,
                inward_search_margin_pixels=inward_search_margin_pixels,
                outward_search_margin_pixels=outward_search_margin_pixels,
                blade_band_half_width_pixels=blade_band_half_width_pixels,
                lateral_window_half_width_pixels=lateral_window_half_width_pixels,
                minimum_attachment_evidence_fraction=(
                    minimum_attachment_evidence_fraction
                ),
                minimum_flesh_band_pixels=minimum_flesh_band_pixels,
                minimum_calyx_band_pixels=minimum_calyx_band_pixels,
                v2a_result=v2a_result,
            )
            _validate_v2b_result(v2b_result, mask_shape=mask.shape)
            v1_proxies = calculate_line_side_proxies(mask, v1_result)
            v2a_proxies = calculate_fixed_axis_line_side_proxies(mask, v2a_result)
            v2b_proxies = calculate_fixed_axis_search_line_side_proxies(
                mask,
                v2b_result,
            )
            row = _comparison_row(
                image_path.stem,
                v1_result,
                v2a_result,
                v2b_result,
                v1_proxies=v1_proxies,
                v2a_proxies=v2a_proxies,
                v2b_proxies=v2b_proxies,
            )
            visualization = visualization_function(
                mask,
                v1_result,
                v2a_result,
                v2b_result,
                image=image,
                metrics=row,
            )
            if not isinstance(visualization, Image.Image):
                raise TypeError("visualization_function must return a PIL.Image.Image")
            relative_visualization = Path("visualizations") / f"{image_path.stem}.png"
            visualization.save(
                staging_root / relative_visualization,
                format="PNG",
            )
            relative_candidate_curve = (
                Path("candidate_curves") / f"{image_path.stem}.csv"
            )
            _write_candidate_curve_csv(
                staging_root / relative_candidate_curve,
                v2b_result,
            )
            rows.append(row)
            v2b_results.append(v2b_result)
            manifest_samples.append(
                {
                    "sample_id": image_path.stem,
                    "v1_status": v1_result.status,
                    "v2a_status": v2a_result.status,
                    "v2b_status": v2b_result.status,
                    "visualization_path": relative_visualization.as_posix(),
                    "candidate_curve_path": relative_candidate_curve.as_posix(),
                }
            )

        summary = _build_summary(
            split=split,
            rows=rows,
            v2b_results=v2b_results,
            parameter_configuration=parameters,
        )
        manifest: dict[str, Any] = {
            "dataset_basename": source_root.name,
            "split": split,
            "sample_count": len(rows),
            "class_mapping": dict(CLASS_MAPPING),
            "v1_algorithm_name": V1_ALGORITHM_NAME,
            "v1_algorithm_version": V1_ALGORITHM_VERSION,
            "v2a_algorithm_name": V2A_ALGORITHM_NAME,
            "v2a_algorithm_version": V2A_ALGORITHM_VERSION,
            "v2b_algorithm_name": V2B_ALGORITHM_NAME,
            "v2b_algorithm_version": V2B_ALGORITHM_VERSION,
            "parameter_configuration": parameters,
            "artifacts": {
                "manifest": "manifest.json",
                "summary": "summary.json",
                "per_image_comparison": "per_image_comparison.csv",
                "visualizations": "visualizations",
                "candidate_curves": "candidate_curves",
            },
            "samples": manifest_samples,
        }
        _write_comparison_csv(staging_root / "per_image_comparison.csv", rows)
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


def calculate_fixed_axis_line_side_proxies(
    mask: np.ndarray,
    result: FixedAxisCutlineResult,
) -> dict[str, int | float | None]:
    """Calculate whole-mask v2a proxies with on-line pixels retained."""

    mask_array = _validate_mask_array(mask)
    _validate_v2a_result(result, mask_shape=mask_array.shape)
    empty: dict[str, int | float | None] = {
        "flesh_loss_proxy_pixel_count": None,
        "flesh_loss_proxy_ratio": None,
        "calyx_retention_proxy_pixel_count": None,
        "calyx_retention_proxy_ratio": None,
    }
    if (
        not result.succeeded
        or result.final_cutline is None
        or result.diagnostic_reference_point is None
    ):
        return empty

    axis_x, axis_y = result.normalized_removal_axis
    reference = result.diagnostic_reference_point
    pixel_y, pixel_x = np.indices(mask_array.shape, dtype=np.float64)
    signed_side = (pixel_x - reference.x) * axis_x + (pixel_y - reference.y) * axis_y
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
        "flesh_loss_proxy_ratio": flesh_loss / flesh_total if flesh_total else None,
        "calyx_retention_proxy_pixel_count": calyx_retention,
        "calyx_retention_proxy_ratio": (
            calyx_retention / calyx_total if calyx_total else None
        ),
    }


def calculate_fixed_axis_search_line_side_proxies(
    mask: np.ndarray,
    result: FixedAxisSearchCutlineResult,
) -> dict[str, int | float | None]:
    """Calculate whole-mask v2b proxies with on-line pixels retained."""

    mask_array = _validate_mask_array(mask)
    _validate_v2b_result(result, mask_shape=mask_array.shape)
    empty: dict[str, int | float | None] = {
        "flesh_loss_proxy_pixel_count": None,
        "flesh_loss_proxy_ratio": None,
        "calyx_retention_proxy_pixel_count": None,
        "calyx_retention_proxy_ratio": None,
    }
    if (
        not result.succeeded
        or result.final_cutline is None
        or result.diagnostic_reference_point is None
    ):
        return empty

    axis_x, axis_y = result.normalized_removal_axis
    reference = result.diagnostic_reference_point
    pixel_y, pixel_x = np.indices(mask_array.shape, dtype=np.float64)
    signed_side = (pixel_x - reference.x) * axis_x + (pixel_y - reference.y) * axis_y
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
        "flesh_loss_proxy_ratio": flesh_loss / flesh_total if flesh_total else None,
        "calyx_retention_proxy_pixel_count": calyx_retention,
        "calyx_retention_proxy_ratio": (
            calyx_retention / calyx_total if calyx_total else None
        ),
    }


def _comparison_row(
    sample_id: str,
    v1_result: MaskCutlineResult,
    v2a_result: FixedAxisCutlineResult,
    v2b_result: FixedAxisSearchCutlineResult,
    *,
    v1_proxies: Mapping[str, int | float | None],
    v2a_proxies: Mapping[str, int | float | None],
    v2b_proxies: Mapping[str, int | float | None],
) -> dict[str, Any]:
    axis = v2a_result.normalized_removal_axis
    v1_anchor_projection = _point_projection(v1_result.attachment_anchor, axis)
    maximum_flesh_projection = _maximum_mask_projection(
        v2a_result.selected_flesh_mask,
        axis,
    )
    v1_cut_margin = (
        maximum_flesh_projection - v1_anchor_projection
        if maximum_flesh_projection is not None and v1_anchor_projection is not None
        else None
    )
    v1_flesh_loss = v1_proxies["flesh_loss_proxy_pixel_count"]
    v1_flesh_ratio = v1_proxies["flesh_loss_proxy_ratio"]
    v1_calyx_retention = v1_proxies["calyx_retention_proxy_pixel_count"]
    v1_calyx_ratio = v1_proxies["calyx_retention_proxy_ratio"]
    v2a_flesh_loss = v2a_proxies["flesh_loss_proxy_pixel_count"]
    v2a_flesh_ratio = v2a_proxies["flesh_loss_proxy_ratio"]
    v2a_calyx_retention = v2a_proxies["calyx_retention_proxy_pixel_count"]
    v2a_calyx_ratio = v2a_proxies["calyx_retention_proxy_ratio"]
    v2b_flesh_loss = v2b_proxies["flesh_loss_proxy_pixel_count"]
    v2b_flesh_ratio = v2b_proxies["flesh_loss_proxy_ratio"]
    v2b_calyx_retention = v2b_proxies["calyx_retention_proxy_pixel_count"]
    v2b_calyx_ratio = v2b_proxies["calyx_retention_proxy_ratio"]
    flesh_reference_diagnostics = (
        v2a_result.flesh_centroid_projected_reference_diagnostics
    )
    anchor_reference_diagnostics = (
        v2a_result.attachment_anchor_projected_reference_diagnostics
    )
    return {
        "sample_id": sample_id,
        "source_width": v1_result.image_width,
        "source_height": v1_result.image_height,
        "v1_status": v1_result.status,
        "v1_failure_code": v1_result.failure_code,
        "v1_failure_reason": v1_result.failure_reason,
        "v2a_status": v2a_result.status,
        "v2a_failure_code": v2a_result.failure_code,
        "v2a_failure_reason": v2a_result.failure_reason,
        "flesh_component_count": v1_result.flesh_component_count,
        "calyx_component_count": v1_result.calyx_component_count,
        "selected_flesh_component": v1_result.selected_flesh_component,
        "selected_calyx_component": v1_result.selected_calyx_component,
        "contact_pixel_count": v1_result.contact_pixel_count,
        "contact_component_count": v1_result.contact_component_count,
        "v1_attachment_anchor_x": _point_value(v1_result.attachment_anchor, "x"),
        "v1_attachment_anchor_y": _point_value(v1_result.attachment_anchor, "y"),
        "v1_direction_x": _tuple_value(
            v1_result.fruit_to_attachment_direction,
            0,
        ),
        "v1_direction_y": _tuple_value(
            v1_result.fruit_to_attachment_direction,
            1,
        ),
        "v1_line_angle_degrees": _line_angle(v1_result.final_cutline),
        "v1_anchor_projection_on_removal_axis": v1_anchor_projection,
        "v1_flesh_loss_pixels": v1_flesh_loss,
        "v1_flesh_loss_ratio": v1_flesh_ratio,
        "v1_calyx_retention_pixels": v1_calyx_retention,
        "v1_calyx_retention_ratio": v1_calyx_ratio,
        "v1_cut_margin_to_flesh_extent": v1_cut_margin,
        "removal_axis_x": axis[0],
        "removal_axis_y": axis[1],
        "projection_quantile": v2a_result.parameters.projection_quantile,
        "support_band_width_pixels": (v2a_result.parameters.support_band_width_pixels),
        "support_pixel_count": v2a_result.support_pixel_count,
        "support_lateral_extent_pixels": (v2a_result.support_lateral_extent_pixels),
        "v2a_unshifted_cut_coordinate": v2a_result.unshifted_cut_coordinate,
        "v2a_final_cut_coordinate": v2a_result.final_cut_coordinate,
        "v2a_reference_point_x": _point_value(
            v2a_result.diagnostic_reference_point,
            "x",
        ),
        "v2a_reference_point_y": _point_value(
            v2a_result.diagnostic_reference_point,
            "y",
        ),
        "v2a_candidate_start_x": _segment_value(
            v2a_result.candidate_cutline,
            "start",
            "x",
        ),
        "v2a_candidate_start_y": _segment_value(
            v2a_result.candidate_cutline,
            "start",
            "y",
        ),
        "v2a_candidate_end_x": _segment_value(
            v2a_result.candidate_cutline,
            "end",
            "x",
        ),
        "v2a_candidate_end_y": _segment_value(
            v2a_result.candidate_cutline,
            "end",
            "y",
        ),
        "v2a_final_start_x": _segment_value(v2a_result.final_cutline, "start", "x"),
        "v2a_final_start_y": _segment_value(v2a_result.final_cutline, "start", "y"),
        "v2a_final_end_x": _segment_value(v2a_result.final_cutline, "end", "x"),
        "v2a_final_end_y": _segment_value(v2a_result.final_cutline, "end", "y"),
        "v2a_axis_disagreement_degrees": (v2a_result.axis_disagreement_angle_degrees),
        "v2a_cut_margin_to_flesh_extent": (v2a_result.cut_margin_to_flesh_extent),
        "v2a_flesh_loss_pixels": v2a_flesh_loss,
        "v2a_flesh_loss_ratio": v2a_flesh_ratio,
        "v2a_calyx_retention_pixels": v2a_calyx_retention,
        "v2a_calyx_retention_ratio": v2a_calyx_ratio,
        "v2a_candidate_final_lines_coincide": (
            v2a_result.candidate_final_lines_coincide
        ),
        "signed_offset_pixels": v2a_result.signed_offset_pixels,
        "calyx_dilation_radius": v2a_result.parameters.calyx_dilation_radius,
        "component_connectivity": v2a_result.parameters.component_connectivity,
        "cut_coordinate_shift_from_v1": _difference(
            v2a_result.final_cut_coordinate,
            v1_anchor_projection,
        ),
        "flesh_loss_pixel_change": _difference(v2a_flesh_loss, v1_flesh_loss),
        "flesh_loss_ratio_change": _difference(v2a_flesh_ratio, v1_flesh_ratio),
        "calyx_retention_pixel_change": _difference(
            v2a_calyx_retention,
            v1_calyx_retention,
        ),
        "calyx_retention_ratio_change": _difference(
            v2a_calyx_ratio,
            v1_calyx_ratio,
        ),
        "v2a_flesh_centroid_projected_reference_x": _point_value(
            v2a_result.flesh_centroid_projected_reference_point,
            "x",
        ),
        "v2a_flesh_centroid_projected_reference_y": _point_value(
            v2a_result.flesh_centroid_projected_reference_point,
            "y",
        ),
        "v2a_flesh_centroid_reference_nearest_class": _diagnostic_value(
            flesh_reference_diagnostics,
            "nearest_in_bounds_semantic_class",
        ),
        "v2a_flesh_centroid_reference_inside_selected_flesh": (
            _diagnostic_value(
                flesh_reference_diagnostics,
                "inside_selected_flesh_component",
            )
        ),
        "v2a_flesh_centroid_reference_distance_to_selected_calyx": (
            _diagnostic_value(
                flesh_reference_diagnostics,
                "distance_to_selected_calyx_mask",
            )
        ),
        "v2a_flesh_centroid_reference_distance_to_selected_contact": (
            _diagnostic_value(
                flesh_reference_diagnostics,
                "distance_to_selected_contact_mask",
            )
        ),
        "v2a_attachment_anchor_projected_reference_x": _point_value(
            v2a_result.attachment_anchor_projected_reference_point,
            "x",
        ),
        "v2a_attachment_anchor_projected_reference_y": _point_value(
            v2a_result.attachment_anchor_projected_reference_point,
            "y",
        ),
        "v2a_attachment_anchor_reference_nearest_class": _diagnostic_value(
            anchor_reference_diagnostics,
            "nearest_in_bounds_semantic_class",
        ),
        "v2a_attachment_anchor_reference_inside_selected_flesh": (
            _diagnostic_value(
                anchor_reference_diagnostics,
                "inside_selected_flesh_component",
            )
        ),
        "v2a_attachment_anchor_reference_distance_to_selected_calyx": (
            _diagnostic_value(
                anchor_reference_diagnostics,
                "distance_to_selected_calyx_mask",
            )
        ),
        "v2a_attachment_anchor_reference_distance_to_selected_contact": (
            _diagnostic_value(
                anchor_reference_diagnostics,
                "distance_to_selected_contact_mask",
            )
        ),
        "v2b_status": v2b_result.status,
        "v2b_failure_code": v2b_result.failure_code,
        "v2b_failure_reason": v2b_result.failure_reason,
        "v2b_selected_flesh_component": v2b_result.selected_flesh_component,
        "v2b_selected_calyx_component": v2b_result.selected_calyx_component,
        "v2b_selected_flesh_pixel_count": v2b_result.selected_flesh_pixel_count,
        "v2b_selected_calyx_pixel_count": v2b_result.selected_calyx_pixel_count,
        "v2b_contact_pixel_count": v2b_result.contact_pixel_count,
        "v2b_blade_tangent_x": v2b_result.blade_tangent[0],
        "v2b_blade_tangent_y": v2b_result.blade_tangent[1],
        "v2b_v1_anchor_projection": v2b_result.v1_anchor_projection,
        "v2b_v2a_cut_coordinate": v2b_result.v2a_cut_coordinate,
        "v2b_lateral_reference_coordinate": (v2b_result.lateral_reference_coordinate),
        "v2b_flesh_centroid_lateral_reference_coordinate": (
            v2b_result.flesh_centroid_lateral_reference_coordinate
        ),
        "v2b_search_interval_start": _tuple_value(v2b_result.search_interval, 0),
        "v2b_search_interval_end": _tuple_value(v2b_result.search_interval, 1),
        "v2b_candidate_count": v2b_result.candidate_count,
        "v2b_feasible_candidate_count": v2b_result.feasible_candidate_count,
        "v2b_feasible_block_count": v2b_result.feasible_block_count,
        "v2b_feasible_hull_start": v2b_result.feasible_hull_start,
        "v2b_feasible_hull_end": v2b_result.feasible_hull_end,
        "v2b_selected_candidate_index": v2b_result.selected_candidate_index,
        "v2b_selected_candidate_feasible": (True if v2b_result.succeeded else None),
        "v2b_selected_block_id": v2b_result.selected_block_id,
        "v2b_selected_block_start": v2b_result.selected_block_start,
        "v2b_selected_block_end": v2b_result.selected_block_end,
        "v2b_selected_block_candidate_count": (
            v2b_result.selected_block_candidate_count
        ),
        "v2b_selected_block_is_singleton": (v2b_result.selected_block_is_singleton),
        "v2b_cut_coordinate": v2b_result.selected_cut_coordinate,
        "v2b_candidate_start_x": _segment_value(
            v2b_result.candidate_cutline,
            "start",
            "x",
        ),
        "v2b_candidate_start_y": _segment_value(
            v2b_result.candidate_cutline,
            "start",
            "y",
        ),
        "v2b_candidate_end_x": _segment_value(
            v2b_result.candidate_cutline,
            "end",
            "x",
        ),
        "v2b_candidate_end_y": _segment_value(
            v2b_result.candidate_cutline,
            "end",
            "y",
        ),
        "v2b_final_start_x": _segment_value(
            v2b_result.final_cutline,
            "start",
            "x",
        ),
        "v2b_final_start_y": _segment_value(
            v2b_result.final_cutline,
            "start",
            "y",
        ),
        "v2b_final_end_x": _segment_value(
            v2b_result.final_cutline,
            "end",
            "x",
        ),
        "v2b_final_end_y": _segment_value(
            v2b_result.final_cutline,
            "end",
            "y",
        ),
        "v2b_reference_point_x": _point_value(
            v2b_result.diagnostic_reference_point,
            "x",
        ),
        "v2b_reference_point_y": _point_value(
            v2b_result.diagnostic_reference_point,
            "y",
        ),
        "v2b_maximum_local_contact_evidence": (
            v2b_result.maximum_local_contact_evidence
        ),
        "v2b_selected_local_contact_count": (v2b_result.selected_local_contact_count),
        "v2b_selected_local_flesh_count": v2b_result.selected_local_flesh_count,
        "v2b_selected_local_calyx_count": v2b_result.selected_local_calyx_count,
        "v2b_selected_local_v2a_support_count": (
            v2b_result.selected_local_v2a_support_count
        ),
        "v2b_selected_pair_flesh_loss_pixels": (v2b_result.selected_flesh_loss_count),
        "v2b_selected_pair_flesh_loss_ratio": (v2b_result.selected_flesh_loss_ratio),
        "v2b_selected_pair_calyx_retention_pixels": (
            v2b_result.selected_calyx_retention_count
        ),
        "v2b_selected_pair_calyx_retention_ratio": (
            v2b_result.selected_calyx_retention_ratio
        ),
        "v2b_selected_pair_retained_contact_pixels": (
            v2b_result.retained_contact_count
        ),
        "v2b_selected_pair_retained_contact_ratio": (v2b_result.retained_contact_ratio),
        "v2b_whole_mask_flesh_loss_pixels": v2b_flesh_loss,
        "v2b_whole_mask_flesh_loss_ratio": v2b_flesh_ratio,
        "v2b_whole_mask_calyx_retention_pixels": v2b_calyx_retention,
        "v2b_whole_mask_calyx_retention_ratio": v2b_calyx_ratio,
        "v2b_coordinate_shift_from_v1": v2b_result.coordinate_shift_from_v1,
        "v2b_coordinate_shift_from_v2a": v2b_result.coordinate_shift_from_v2a,
        "v2b_lies_between_v1_and_v2a": _lies_between(
            v2b_result.selected_cut_coordinate,
            v1_anchor_projection,
            v2a_result.final_cut_coordinate,
        ),
        "v2b_cut_margin_to_flesh_extent": (v2b_result.cut_margin_to_flesh_extent),
        "v2b_candidate_final_lines_coincide": (
            v2b_result.candidate_final_lines_coincide
        ),
        "v2b_minus_v1_flesh_loss_pixel_change": _difference(
            v2b_flesh_loss,
            v1_flesh_loss,
        ),
        "v2b_minus_v1_flesh_loss_ratio_change": _difference(
            v2b_flesh_ratio,
            v1_flesh_ratio,
        ),
        "v2b_minus_v1_calyx_retention_pixel_change": _difference(
            v2b_calyx_retention,
            v1_calyx_retention,
        ),
        "v2b_minus_v1_calyx_retention_ratio_change": _difference(
            v2b_calyx_ratio,
            v1_calyx_ratio,
        ),
        "v2b_minus_v2a_flesh_loss_pixel_change": _difference(
            v2b_flesh_loss,
            v2a_flesh_loss,
        ),
        "v2b_minus_v2a_flesh_loss_ratio_change": _difference(
            v2b_flesh_ratio,
            v2a_flesh_ratio,
        ),
        "v2b_minus_v2a_calyx_retention_pixel_change": _difference(
            v2b_calyx_retention,
            v2a_calyx_retention,
        ),
        "v2b_minus_v2a_calyx_retention_ratio_change": _difference(
            v2b_calyx_ratio,
            v2a_calyx_ratio,
        ),
    }


def _build_summary(
    *,
    split: str,
    rows: list[dict[str, Any]],
    v2b_results: Sequence[FixedAxisSearchCutlineResult],
    parameter_configuration: Mapping[str, Any],
) -> dict[str, Any]:
    v1_failures = Counter(
        row["v1_failure_code"] or "unknown_failure"
        for row in rows
        if row["v1_status"] != "ok"
    )
    v2a_failures = Counter(
        row["v2a_failure_code"] or "unknown_failure"
        for row in rows
        if row["v2a_status"] != "ok"
    )
    v2b_failures = Counter(
        row["v2b_failure_code"] or "unknown_failure"
        for row in rows
        if row["v2b_status"] != "ok"
    )
    return {
        "evaluated_split": split,
        "total_sample_count": len(rows),
        "successful_v1_count": sum(row["v1_status"] == "ok" for row in rows),
        "successful_v2a_count": sum(row["v2a_status"] == "ok" for row in rows),
        "successful_v2b_count": sum(row["v2b_status"] == "ok" for row in rows),
        "failure_counts_by_method_and_reason": {
            "v1": dict(sorted(v1_failures.items())),
            "v2a": dict(sorted(v2a_failures.items())),
            "v2b": dict(sorted(v2b_failures.items())),
        },
        "status_definitions": dict(STATUS_DEFINITIONS),
        "proxy_definition": PROXY_DEFINITION,
        "v1_line_side_sign_convention": V1_LINE_SIDE_CONVENTION,
        "v2a_line_side_sign_convention": V2A_LINE_SIDE_CONVENTION,
        "v2b_line_side_sign_convention": V2B_LINE_SIDE_CONVENTION,
        "whole_mask_proxy_scope": WHOLE_MASK_PROXY_SCOPE,
        "selected_pair_proxy_scope": SELECTED_PAIR_PROXY_SCOPE,
        "feasible_block_definition": FEASIBLE_BLOCK_DEFINITION,
        "feasible_hull_definition": FEASIBLE_HULL_DEFINITION,
        "selected_block_definition": SELECTED_BLOCK_DEFINITION,
        "feasible_set_caution": FEASIBLE_SET_CAUTION,
        "change_convention": CHANGE_CONVENTION,
        "coordinate_shift_definition": COORDINATE_SHIFT_DEFINITION,
        "cut_margin_definition": CUT_MARGIN_DEFINITION,
        "aggregate_statistics": {
            "v1_flesh_loss_ratio": _numeric_statistics(
                [row["v1_flesh_loss_ratio"] for row in rows]
            ),
            "v2a_flesh_loss_ratio": _numeric_statistics(
                [row["v2a_flesh_loss_ratio"] for row in rows]
            ),
            "v2b_whole_mask_flesh_loss_ratio": _numeric_statistics(
                [row["v2b_whole_mask_flesh_loss_ratio"] for row in rows]
            ),
            "v1_calyx_retention_ratio": _numeric_statistics(
                [row["v1_calyx_retention_ratio"] for row in rows]
            ),
            "v2a_calyx_retention_ratio": _numeric_statistics(
                [row["v2a_calyx_retention_ratio"] for row in rows]
            ),
            "v2b_whole_mask_calyx_retention_ratio": _numeric_statistics(
                [row["v2b_whole_mask_calyx_retention_ratio"] for row in rows]
            ),
            "cut_coordinate_shift_from_v1": _numeric_statistics(
                [row["cut_coordinate_shift_from_v1"] for row in rows]
            ),
            "v2a_axis_disagreement_degrees": _numeric_statistics(
                [row["v2a_axis_disagreement_degrees"] for row in rows]
            ),
            "v2a_cut_margin_to_flesh_extent": _numeric_statistics(
                [row["v2a_cut_margin_to_flesh_extent"] for row in rows]
            ),
            "v2b_coordinate_shift_from_v1": _numeric_statistics(
                [row["v2b_coordinate_shift_from_v1"] for row in rows]
            ),
            "v2b_coordinate_shift_from_v2a": _numeric_statistics(
                [row["v2b_coordinate_shift_from_v2a"] for row in rows]
            ),
            "v2b_cut_margin_to_flesh_extent": _numeric_statistics(
                [row["v2b_cut_margin_to_flesh_extent"] for row in rows]
            ),
            "v2b_selected_local_contact_count": _numeric_statistics(
                [row["v2b_selected_local_contact_count"] for row in rows]
            ),
            "support_pixel_count": _numeric_statistics(
                [row["support_pixel_count"] for row in rows]
            ),
            "contact_pixel_count": _numeric_statistics(
                [row["contact_pixel_count"] for row in rows]
            ),
        },
        "feasible_set_diagnostics": _feasible_set_diagnostics(v2b_results),
        "comparison_counts": _tradeoff_counts(rows),
        "parameter_configuration": dict(parameter_configuration),
    }


def _feasible_set_diagnostics(
    results: Sequence[FixedAxisSearchCutlineResult],
) -> dict[str, Any]:
    selected_block_is_largest = [
        _selected_block_is_largest(result) for result in results
    ]
    selected_block_widths = [
        result.selected_block_end - result.selected_block_start
        for result in results
        if result.selected_block_start is not None
        and result.selected_block_end is not None
    ]
    return {
        "fragmented_sample_count": sum(
            result.feasible_block_count > 1 for result in results
        ),
        "selected_block_not_largest_sample_count": sum(
            value is False for value in selected_block_is_largest
        ),
        "singleton_selected_block_sample_count": sum(
            result.selected_block_is_singleton is True for result in results
        ),
        "feasible_block_count_statistics": _numeric_statistics(
            [result.feasible_block_count for result in results]
        ),
        "selected_block_width_pixels_statistics": _numeric_statistics(
            selected_block_widths
        ),
    }


def _selected_block_is_largest(
    result: FixedAxisSearchCutlineResult,
) -> bool | None:
    if result.selected_block_id is None:
        return None
    feasible_block_ids = result.candidate_table.feasible_block_id
    block_ids = feasible_block_ids[feasible_block_ids >= 0]
    if not len(block_ids):
        return None
    block_candidate_counts = np.bincount(block_ids)
    return bool(
        result.selected_block_candidate_count == int(block_candidate_counts.max())
    )


def _tradeoff_counts(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    counts = {
        "v2a_lowers_flesh_loss": 0,
        "v2a_raises_flesh_loss": 0,
        "v2a_lowers_calyx_retention": 0,
        "v2a_raises_calyx_retention": 0,
        "both_proxies_improve": 0,
        "one_improves_while_other_worsens": 0,
        "v2b_vs_v1_lowers_flesh_loss": 0,
        "v2b_vs_v1_raises_flesh_loss": 0,
        "v2b_vs_v1_lowers_calyx_retention": 0,
        "v2b_vs_v1_raises_calyx_retention": 0,
        "v2b_vs_v1_both_proxies_improve": 0,
        "v2b_vs_v1_one_improves_while_other_worsens": 0,
        "v2b_vs_v2a_lowers_flesh_loss": 0,
        "v2b_vs_v2a_raises_flesh_loss": 0,
        "v2b_vs_v2a_lowers_calyx_retention": 0,
        "v2b_vs_v2a_raises_calyx_retention": 0,
        "v2b_vs_v2a_both_proxies_improve": 0,
        "v2b_vs_v2a_one_improves_while_other_worsens": 0,
    }
    for row in rows:
        flesh_change = row["flesh_loss_ratio_change"]
        calyx_change = row["calyx_retention_ratio_change"]
        if flesh_change is not None:
            counts["v2a_lowers_flesh_loss"] += flesh_change < 0.0
            counts["v2a_raises_flesh_loss"] += flesh_change > 0.0
        if calyx_change is not None:
            counts["v2a_lowers_calyx_retention"] += calyx_change < 0.0
            counts["v2a_raises_calyx_retention"] += calyx_change > 0.0
        if flesh_change is not None and calyx_change is not None:
            counts["both_proxies_improve"] += flesh_change < 0.0 and calyx_change < 0.0
            counts["one_improves_while_other_worsens"] += (
                flesh_change < 0.0 < calyx_change or calyx_change < 0.0 < flesh_change
            )
        _update_pair_tradeoff_counts(
            counts,
            prefix="v2b_vs_v1",
            flesh_change=row.get("v2b_minus_v1_flesh_loss_ratio_change"),
            calyx_change=row.get("v2b_minus_v1_calyx_retention_ratio_change"),
        )
        _update_pair_tradeoff_counts(
            counts,
            prefix="v2b_vs_v2a",
            flesh_change=row.get("v2b_minus_v2a_flesh_loss_ratio_change"),
            calyx_change=row.get("v2b_minus_v2a_calyx_retention_ratio_change"),
        )
    return counts


def _update_pair_tradeoff_counts(
    counts: dict[str, int],
    *,
    prefix: str,
    flesh_change: int | float | None,
    calyx_change: int | float | None,
) -> None:
    if flesh_change is not None:
        counts[f"{prefix}_lowers_flesh_loss"] += flesh_change < 0.0
        counts[f"{prefix}_raises_flesh_loss"] += flesh_change > 0.0
    if calyx_change is not None:
        counts[f"{prefix}_lowers_calyx_retention"] += calyx_change < 0.0
        counts[f"{prefix}_raises_calyx_retention"] += calyx_change > 0.0
    if flesh_change is None or calyx_change is None:
        return
    counts[f"{prefix}_both_proxies_improve"] += (
        flesh_change < 0.0 and calyx_change < 0.0
    )
    counts[f"{prefix}_one_improves_while_other_worsens"] += (
        flesh_change < 0.0 < calyx_change or calyx_change < 0.0 < flesh_change
    )


def _validate_runner_parameters(
    *,
    split: str,
    removal_axis: Sequence[float],
    projection_quantile: float,
    support_band_width_pixels: float,
    calyx_dilation_radius: int,
    component_connectivity: int,
    signed_offset_pixels: float,
    candidate_step_pixels: float,
    inward_search_margin_pixels: float,
    outward_search_margin_pixels: float,
    blade_band_half_width_pixels: float,
    lateral_window_half_width_pixels: float,
    minimum_attachment_evidence_fraction: float,
    minimum_flesh_band_pixels: int,
    minimum_calyx_band_pixels: int,
) -> dict[str, Any]:
    if split not in SUPPORTED_SPLITS:
        raise ValueError(
            f"split must be one of {', '.join(SUPPORTED_SPLITS)}, got {split!r}"
        )
    original_axis, normalized_axis = _validate_removal_axis(removal_axis)
    quantile = _validate_finite_real(
        projection_quantile,
        name="projection_quantile",
    )
    if not 0.0 < quantile <= 1.0:
        raise ValueError("projection_quantile must satisfy 0 < value <= 1")
    support_width = _validate_finite_real(
        support_band_width_pixels,
        name="support_band_width_pixels",
    )
    if support_width < 0.0:
        raise ValueError("support_band_width_pixels must be non-negative")
    signed_offset = _validate_finite_real(
        signed_offset_pixels,
        name="signed_offset_pixels",
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
    search_parameters = _validate_search_parameters(
        candidate_step_pixels=candidate_step_pixels,
        inward_search_margin_pixels=inward_search_margin_pixels,
        outward_search_margin_pixels=outward_search_margin_pixels,
        blade_band_half_width_pixels=blade_band_half_width_pixels,
        lateral_window_half_width_pixels=lateral_window_half_width_pixels,
        minimum_attachment_evidence_fraction=(minimum_attachment_evidence_fraction),
        minimum_flesh_band_pixels=minimum_flesh_band_pixels,
        minimum_calyx_band_pixels=minimum_calyx_band_pixels,
    )
    return {
        "removal_axis": list(original_axis),
        "normalized_removal_axis": list(normalized_axis),
        "projection_quantile": quantile,
        "support_band_width_pixels": support_width,
        "calyx_dilation_radius": calyx_dilation_radius,
        "component_connectivity": component_connectivity,
        "v1_signed_offset_pixels": V1_SIGNED_OFFSET_PIXELS,
        "v2a_signed_offset_pixels": signed_offset,
        **search_parameters,
    }


def _validate_v1_result(
    result: MaskCutlineResult,
    *,
    mask_shape: tuple[int, int],
) -> None:
    if not isinstance(result, MaskCutlineResult):
        raise TypeError("v1_estimator must return a MaskCutlineResult")
    if (result.image_height, result.image_width) != mask_shape:
        raise ValueError("v1 estimator result dimensions must match the source mask")
    if result.succeeded and (
        result.final_cutline is None
        or result.offset_anchor is None
        or result.fruit_to_attachment_direction is None
    ):
        raise ValueError("successful v1 estimator result is missing final geometry")


def _validate_v2a_result(
    result: FixedAxisCutlineResult,
    *,
    mask_shape: tuple[int, int],
) -> None:
    if not isinstance(result, FixedAxisCutlineResult):
        raise TypeError("v2a_estimator must return a FixedAxisCutlineResult")
    if (result.image_height, result.image_width) != mask_shape:
        raise ValueError("v2a estimator result dimensions must match the source mask")
    if result.succeeded and (
        result.final_cutline is None or result.diagnostic_reference_point is None
    ):
        raise ValueError("successful v2a estimator result is missing final geometry")


def _validate_v2b_result(
    result: FixedAxisSearchCutlineResult,
    *,
    mask_shape: tuple[int, int],
) -> None:
    if not isinstance(result, FixedAxisSearchCutlineResult):
        raise TypeError("v2b_estimator must return a FixedAxisSearchCutlineResult")
    if (result.image_height, result.image_width) != mask_shape:
        raise ValueError("v2b estimator result dimensions must match the source mask")
    if result.succeeded and (
        result.final_cutline is None
        or result.diagnostic_reference_point is None
        or result.selected_cut_coordinate is None
    ):
        raise ValueError("successful v2b estimator result is missing final geometry")


def _maximum_mask_projection(
    binary_mask: np.ndarray,
    axis: tuple[float, float],
) -> float | None:
    coordinates_yx = np.argwhere(binary_mask)
    if not len(coordinates_yx):
        return None
    projections = (
        coordinates_yx[:, 1].astype(np.float64) * axis[0]
        + coordinates_yx[:, 0].astype(np.float64) * axis[1]
    )
    return float(projections.max())


def _point_projection(
    point: Point | None,
    axis: tuple[float, float],
) -> float | None:
    if point is None:
        return None
    return point.x * axis[0] + point.y * axis[1]


def _line_angle(segment: LineSegment | None) -> float | None:
    if segment is None:
        return None
    delta_x = segment.end.x - segment.start.x
    delta_y = segment.end.y - segment.start.y
    angle = math.degrees(math.atan2(delta_y, delta_x)) % 180.0
    return 0.0 if math.isclose(angle, 180.0, abs_tol=1e-12) else angle


def _difference(
    v2a_value: int | float | None,
    v1_value: int | float | None,
) -> int | float | None:
    if v2a_value is None or v1_value is None:
        return None
    return v2a_value - v1_value


def _point_value(point: Point | None, coordinate: str) -> float | None:
    return getattr(point, coordinate) if point is not None else None


def _diagnostic_value(diagnostics: Any, name: str) -> Any:
    return getattr(diagnostics, name) if diagnostics is not None else None


def _lies_between(
    value: float | None,
    endpoint_a: float | None,
    endpoint_b: float | None,
) -> bool | None:
    if value is None or endpoint_a is None or endpoint_b is None:
        return None
    return min(endpoint_a, endpoint_b) <= value <= max(endpoint_a, endpoint_b)


def _tuple_value(value: tuple[float, float] | None, index: int) -> float | None:
    return value[index] if value is not None else None


def _segment_value(
    segment: LineSegment | None,
    endpoint: str,
    coordinate: str,
) -> float | None:
    if segment is None:
        return None
    return getattr(getattr(segment, endpoint), coordinate)


def _write_comparison_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in CSV_FIELDS})


def _write_candidate_curve_csv(
    path: Path,
    result: FixedAxisSearchCutlineResult,
) -> None:
    table = result.candidate_table
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=CANDIDATE_CSV_FIELDS,
            lineterminator="\n",
        )
        writer.writeheader()
        for index in range(len(table)):
            row = {}
            for field in CANDIDATE_CSV_FIELDS:
                value = getattr(table, field)[index]
                if isinstance(value, np.generic):
                    value = value.item()
                if field == "feasible_block_id" and value < 0:
                    value = None
                row[field] = _csv_value(value)
            writer.writerow(row)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare baseline-v1 and fixed-axis-v2a/v2b cutlines at source resolution."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--split", choices=SUPPORTED_SPLITS, default="train")
    parser.add_argument("--removal-axis-x", type=float, required=True)
    parser.add_argument("--removal-axis-y", type=float, required=True)
    parser.add_argument("--projection-quantile", type=float, default=0.95)
    parser.add_argument("--support-band-width-pixels", type=float, default=5.0)
    parser.add_argument("--calyx-dilation-radius", type=int, default=1)
    parser.add_argument(
        "--component-connectivity",
        type=int,
        choices=(4, 8),
        default=8,
    )
    parser.add_argument("--signed-offset-pixels", type=float, default=0.0)
    parser.add_argument("--candidate-step-pixels", type=float, default=1.0)
    parser.add_argument(
        "--inward-search-margin-pixels",
        type=float,
        default=10.0,
    )
    parser.add_argument(
        "--outward-search-margin-pixels",
        type=float,
        default=0.0,
        help=(
            "additional outward search beyond the v2a coordinate; defaults to "
            "0.0, while explicit positive values remain research overrides"
        ),
    )
    parser.add_argument(
        "--blade-band-half-width-pixels",
        type=float,
        default=1.0,
    )
    parser.add_argument(
        "--lateral-window-half-width-pixels",
        type=float,
        default=64.0,
    )
    parser.add_argument(
        "--minimum-attachment-evidence-fraction",
        type=float,
        default=0.50,
    )
    parser.add_argument("--minimum-flesh-band-pixels", type=int, default=1)
    parser.add_argument("--minimum-calyx-band-pixels", type=int, default=1)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory after successful generation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the v1-versus-v2a-versus-v2b comparison CLI."""

    arguments = _build_argument_parser().parse_args(argv)
    output = run_fixed_axis_cutline_comparison_dataset(
        arguments.dataset_root,
        arguments.output_root,
        removal_axis=(arguments.removal_axis_x, arguments.removal_axis_y),
        split=arguments.split,
        projection_quantile=arguments.projection_quantile,
        support_band_width_pixels=arguments.support_band_width_pixels,
        calyx_dilation_radius=arguments.calyx_dilation_radius,
        component_connectivity=arguments.component_connectivity,
        signed_offset_pixels=arguments.signed_offset_pixels,
        candidate_step_pixels=arguments.candidate_step_pixels,
        inward_search_margin_pixels=arguments.inward_search_margin_pixels,
        outward_search_margin_pixels=arguments.outward_search_margin_pixels,
        blade_band_half_width_pixels=arguments.blade_band_half_width_pixels,
        lateral_window_half_width_pixels=(arguments.lateral_window_half_width_pixels),
        minimum_attachment_evidence_fraction=(
            arguments.minimum_attachment_evidence_fraction
        ),
        minimum_flesh_band_pixels=arguments.minimum_flesh_band_pixels,
        minimum_calyx_band_pixels=arguments.minimum_calyx_band_pixels,
        overwrite=arguments.overwrite,
    )
    summary = output["summary"]
    print(
        f"Processed {summary['total_sample_count']} {summary['evaluated_split']} "
        f"samples: v1={summary['successful_v1_count']} finite, "
        f"v2a={summary['successful_v2a_count']} finite, "
        f"v2b={summary['successful_v2b_count']} finite"
    )
    print("Successful finite geometry does not prove a good or accurate cut.")
    print(f"Artifacts written to {arguments.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
