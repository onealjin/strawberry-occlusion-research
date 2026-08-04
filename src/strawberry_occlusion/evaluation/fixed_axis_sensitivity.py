"""Deterministic parameter sensitivity for committed fixed-axis-v2b geometry."""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
import tempfile
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from numbers import Real
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

import numpy as np
from PIL import Image

from strawberry_occlusion.evaluation.cutline import (
    ALGORITHM_NAME as V1_ALGORITHM_NAME,
    ALGORITHM_VERSION as V1_ALGORITHM_VERSION,
    CLASS_MAPPING,
    PROXY_DEFINITION,
    _csv_value,
    _install_staged_output,
    _paired_paths,
    _read_source_pair,
    _validate_output_destination,
    _write_json,
    calculate_line_side_proxies,
)
from strawberry_occlusion.evaluation.fixed_axis_cutline import (
    FEASIBLE_BLOCK_DEFINITION,
    FEASIBLE_HULL_DEFINITION,
    FEASIBLE_SET_CAUTION,
    SELECTED_BLOCK_DEFINITION,
    SELECTED_PAIR_PROXY_SCOPE,
    V1_SIGNED_OFFSET_PIXELS,
    V2A_ALGORITHM_NAME,
    V2A_ALGORITHM_VERSION,
    V2B_ALGORITHM_NAME,
    V2B_ALGORITHM_VERSION,
    WHOLE_MASK_PROXY_SCOPE,
    _comparison_row,
    _selected_block_is_largest,
    _validate_runner_parameters,
    _validate_v1_result,
    _validate_v2a_result,
    _validate_v2b_result,
    calculate_fixed_axis_line_side_proxies,
    calculate_fixed_axis_search_line_side_proxies,
)
from strawberry_occlusion.geometry import (
    FixedAxisCutlineResult,
    FixedAxisSearchCutlineResult,
    MaskCutlineResult,
    estimate_fixed_axis_cutline,
    estimate_fixed_axis_search_cutline,
    estimate_visible_mask_cutline,
)
from strawberry_occlusion.visualization.fixed_axis_comparison import (
    create_fixed_axis_comparison_visualization,
)
from strawberry_occlusion.visualization.fixed_axis_sensitivity import (
    create_fixed_axis_sensitivity_plots,
)


PathLike = str | Path
V1Estimator = Callable[..., MaskCutlineResult]
V2aEstimator = Callable[..., FixedAxisCutlineResult]
V2bEstimator = Callable[..., FixedAxisSearchCutlineResult]
ComparisonVisualizationFunction = Callable[..., Image.Image]
PlotFunction = Callable[
    [Sequence[Mapping[str, Any]], Sequence[Mapping[str, Any]], PathLike],
    dict[str, str],
]

DEFAULT_MINIMUM_ATTACHMENT_EVIDENCE_FRACTIONS = (0.30, 0.50, 0.70)
DEFAULT_LATERAL_WINDOW_HALF_WIDTH_PIXELS = (48.0, 64.0, 96.0)
COMMITTED_BASELINE_EVIDENCE_FRACTION = 0.50
COMMITTED_BASELINE_LATERAL_WINDOW_HALF_WIDTH_PIXELS = 64.0
COMMITTED_BASELINE_CONFIGURATION_ID = "e050_w064"
DEFAULT_STABILITY_THRESHOLD_PIXELS = 5.0
DEFAULT_MAXIMUM_INWARD_SHIFT_THRESHOLD_PIXELS = 10.0
DEFAULT_FEASIBLE_BLOCK_COUNT_FLAG_THRESHOLD = 5
DEFAULT_COORDINATE_TOLERANCE_PIXELS = 1e-6
SENSITIVITY_SUPPORTED_SPLITS = ("train", "val")

DIFFERENCE_CONVENTION = (
    "Aggregate change = current configuration - e050_w064. A negative "
    "Flesh-loss proxy change means a lower estimated Flesh-loss proxy; a "
    "negative Calyx-retention proxy change means a lower estimated "
    "Calyx-retention proxy. Neither change is a confirmed physical "
    "improvement. Per-row coordinate shift is v2b minus v2a; negative "
    "coordinate shift is inward along the configured removal axis"
)
STABILITY_DEFINITION = (
    "Coordinate minimum, maximum, median, range, and population standard "
    "deviation (ddof=0) use only successful finite v2b coordinates. Failure "
    "counts remain separate. All-successful-coordinates-same is undefined "
    "when no configuration succeeds and otherwise means range no greater "
    "than the configured coordinate tolerance"
)
AGGREGATE_DEFINITION = (
    "Proxy means and medians use finite values from successful v2b rows only. "
    "Success and fragmentation rates use all samples; singleton and selected-"
    "not-largest rates use successful selections. Inward-shift medians and "
    "maxima use moved-inward successful rows. Median and maximum feasible-block "
    "count use every sample row for a configuration, including failed rows; a "
    "failed row with a recorded feasible-block count of zero contributes zero. "
    "Median feasible-hull width uses only rows with a finite hull width, so rows "
    "without a finite hull do not contribute to that median"
)
INWARD_DEFINITION = (
    "A successful v2b coordinate moved inward when v2b minus v2a is less than "
    "negative coordinate tolerance. Inward-shift magnitude is v2a minus v2b"
)
PROXY_CAUTIONS = (
    PROXY_DEFINITION,
    WHOLE_MASK_PROXY_SCOPE,
    SELECTED_PAIR_PROXY_SCOPE,
    "Proxy values are not physical cutting results, attachment-severance "
    "measurements, or final generalization evidence.",
)
STUDY_LIMITATION = (
    "This parameter-sensitivity study used one selected development split and "
    "is not final generalization evidence."
)


@dataclass(frozen=True)
class SensitivityConfiguration:
    """One deterministic pair in the v2b sensitivity grid."""

    configuration_id: str
    minimum_attachment_evidence_fraction: float
    lateral_window_half_width_pixels: float
    is_committed_baseline: bool


PARAMETER_GRID_FIELDS = (
    "configuration_id",
    "minimum_attachment_evidence_fraction",
    "lateral_window_half_width_pixels",
    "removal_axis_x",
    "removal_axis_y",
    "projection_quantile",
    "support_band_width_pixels",
    "calyx_dilation_radius",
    "component_connectivity",
    "signed_offset_pixels",
    "candidate_step_pixels",
    "inward_search_margin_pixels",
    "outward_search_margin_pixels",
    "blade_band_half_width_pixels",
    "minimum_flesh_band_pixels",
    "minimum_calyx_band_pixels",
    "is_committed_baseline",
)

PER_SAMPLE_CONFIGURATION_FIELDS = (
    "sample_id",
    "split",
    "configuration_id",
    "minimum_attachment_evidence_fraction",
    "lateral_window_half_width_pixels",
    "is_committed_baseline",
    "source_width",
    "source_height",
    "v1_status",
    "v1_failure_code",
    "v1_failure_reason",
    "v2a_status",
    "v2a_failure_code",
    "v2a_failure_reason",
    "v2b_status",
    "v2b_failure_code",
    "v2b_failure_reason",
    "structured_status",
    "structured_failure_reason",
    "v2a_cut_coordinate",
    "v2b_cut_coordinate",
    "v2b_minus_v2a_coordinate_shift",
    "v2b_equals_v2a_within_tolerance",
    "v2b_moved_inward",
    "inward_shift_pixels",
    "coordinate_tolerance_pixels",
    "v2b_candidate_count",
    "feasible_candidate_count",
    "feasible_block_count",
    "feasible_hull_width_pixels",
    "selected_block_width_pixels",
    "selected_block_candidate_count",
    "selected_block_is_singleton",
    "selected_block_is_largest",
    "selected_local_contact_count",
    "selected_local_flesh_count",
    "selected_local_calyx_count",
    "selected_pair_flesh_loss_ratio",
    "selected_pair_calyx_retention_ratio",
    "selected_pair_retained_contact_ratio",
    "whole_mask_flesh_loss_ratio",
    "whole_mask_calyx_retention_ratio",
)

CONFIGURATION_AGGREGATE_FIELDS = (
    "sample_count",
    "success_count",
    "success_rate",
    "structured_failure_count",
    "mean_whole_mask_flesh_loss_ratio",
    "median_whole_mask_flesh_loss_ratio",
    "mean_whole_mask_calyx_retention_ratio",
    "median_whole_mask_calyx_retention_ratio",
    "mean_selected_pair_flesh_loss_ratio",
    "median_selected_pair_flesh_loss_ratio",
    "mean_selected_pair_calyx_retention_ratio",
    "median_selected_pair_calyx_retention_ratio",
    "mean_selected_pair_retained_contact_ratio",
    "median_selected_pair_retained_contact_ratio",
    "v2b_equals_v2a_count",
    "moved_inward_count",
    "median_inward_shift_pixels",
    "maximum_inward_shift_pixels",
    "zero_feasible_count",
    "one_block_count",
    "fragmented_set_count",
    "fragmented_set_rate",
    "median_feasible_block_count",
    "maximum_feasible_block_count",
    "singleton_selected_block_count",
    "singleton_selected_block_rate",
    "selected_block_not_largest_count",
    "selected_block_not_largest_rate",
    "median_selected_block_width_pixels",
    "median_feasible_hull_width_pixels",
    "aggregate_runtime_seconds",
)

CONFIGURATION_BASELINE_CHANGE_FIELDS = tuple(
    field
    for field in CONFIGURATION_AGGREGATE_FIELDS
    if field != "aggregate_runtime_seconds"
)

CONFIGURATION_SUMMARY_FIELDS = (
    "configuration_id",
    "minimum_attachment_evidence_fraction",
    "lateral_window_half_width_pixels",
    "is_committed_baseline",
    *CONFIGURATION_AGGREGATE_FIELDS,
    *(
        f"{field}_change_from_baseline"
        for field in CONFIGURATION_BASELINE_CHANGE_FIELDS
    ),
)

PER_SAMPLE_STABILITY_FIELDS = (
    "sample_id",
    "split",
    "configuration_count",
    "success_count",
    "failure_count",
    "coordinate_minimum",
    "coordinate_maximum",
    "coordinate_median",
    "coordinate_range",
    "coordinate_standard_deviation",
    "configurations_equal_to_v2a_count",
    "configurations_moved_inward_count",
    "maximum_inward_shift_pixels",
    "minimum_feasible_block_count",
    "maximum_feasible_block_count",
    "singleton_selection_count",
    "selected_not_largest_count",
    "unique_structured_status_count",
    "all_successful_configurations_same_coordinate_within_tolerance",
)

FLAGGED_SAMPLE_FIELDS = (
    "sample_id",
    "split",
    "flag_reason",
    "observed_value",
    "comparison",
    "threshold",
)


def build_parameter_grid(
    minimum_attachment_evidence_fractions: Sequence[Real] = (
        DEFAULT_MINIMUM_ATTACHMENT_EVIDENCE_FRACTIONS
    ),
    lateral_window_half_width_pixels: Sequence[Real] = (
        DEFAULT_LATERAL_WINDOW_HALF_WIDTH_PIXELS
    ),
) -> tuple[SensitivityConfiguration, ...]:
    """Validate, sort, and cross-product the two sensitivity dimensions."""

    evidence_values = _validate_grid_values(
        minimum_attachment_evidence_fractions,
        name="minimum_attachment_evidence_fraction",
        lower_exclusive=0.0,
        upper_inclusive=1.0,
    )
    width_values = _validate_grid_values(
        lateral_window_half_width_pixels,
        name="lateral_window_half_width_pixels",
        lower_exclusive=0.0,
    )
    if COMMITTED_BASELINE_EVIDENCE_FRACTION not in evidence_values or (
        COMMITTED_BASELINE_LATERAL_WINDOW_HALF_WIDTH_PIXELS not in width_values
    ):
        raise ValueError(
            "The sensitivity grid must include committed baseline values "
            "minimum_attachment_evidence_fraction=0.50 and "
            "lateral_window_half_width_pixels=64.0"
        )

    configurations = []
    identifiers: set[str] = set()
    for evidence in evidence_values:
        for width in width_values:
            configuration_id = _configuration_id(evidence, width)
            if configuration_id in identifiers:
                raise ValueError(
                    "Grid values produce a duplicate configuration ID: "
                    f"{configuration_id}"
                )
            identifiers.add(configuration_id)
            baseline = (
                evidence == COMMITTED_BASELINE_EVIDENCE_FRACTION
                and width == COMMITTED_BASELINE_LATERAL_WINDOW_HALF_WIDTH_PIXELS
            )
            configurations.append(
                SensitivityConfiguration(
                    configuration_id=configuration_id,
                    minimum_attachment_evidence_fraction=evidence,
                    lateral_window_half_width_pixels=width,
                    is_committed_baseline=baseline,
                )
            )
    baseline_ids = [
        configuration.configuration_id
        for configuration in configurations
        if configuration.is_committed_baseline
    ]
    if baseline_ids != [COMMITTED_BASELINE_CONFIGURATION_ID]:
        raise ValueError(
            "Committed baseline configuration must have ID "
            f"{COMMITTED_BASELINE_CONFIGURATION_ID}"
        )
    return tuple(configurations)


def run_fixed_axis_sensitivity_dataset(
    dataset_root: PathLike,
    output_root: PathLike,
    *,
    split: str,
    minimum_attachment_evidence_fractions: Sequence[Real] | None = None,
    lateral_window_half_width_pixels: Sequence[Real] | None = None,
    removal_axis: Sequence[float] = (1.0, 0.0),
    projection_quantile: float = 0.95,
    support_band_width_pixels: float = 5.0,
    calyx_dilation_radius: int = 1,
    component_connectivity: int = 8,
    signed_offset_pixels: float = 0.0,
    candidate_step_pixels: float = 1.0,
    inward_search_margin_pixels: float = 10.0,
    outward_search_margin_pixels: float = 0.0,
    blade_band_half_width_pixels: float = 1.0,
    minimum_flesh_band_pixels: int = 1,
    minimum_calyx_band_pixels: int = 1,
    stability_threshold_pixels: float = DEFAULT_STABILITY_THRESHOLD_PIXELS,
    maximum_inward_shift_threshold_pixels: float = (
        DEFAULT_MAXIMUM_INWARD_SHIFT_THRESHOLD_PIXELS
    ),
    feasible_block_count_flag_threshold: int = (
        DEFAULT_FEASIBLE_BLOCK_COUNT_FLAG_THRESHOLD
    ),
    coordinate_tolerance_pixels: float = DEFAULT_COORDINATE_TOLERANCE_PIXELS,
    overwrite: bool = False,
    visualization_configuration_ids: Sequence[str] = (),
    visualize_flagged_samples: bool = False,
    class_mapping: Mapping[str, int] = CLASS_MAPPING,
    v1_estimator: V1Estimator = estimate_visible_mask_cutline,
    v2a_estimator: V2aEstimator = estimate_fixed_axis_cutline,
    v2b_estimator: V2bEstimator = estimate_fixed_axis_search_cutline,
    visualization_function: ComparisonVisualizationFunction = (
        create_fixed_axis_comparison_visualization
    ),
    plot_function: PlotFunction = create_fixed_axis_sensitivity_plots,
) -> dict[str, Any]:
    """Run a deterministic torch-free v2b sensitivity study on one split.

    Source images and masks stay at source resolution and are never resized,
    cropped, rotated, overwritten, or written into the output. Committed v1 and
    v2a estimators are evaluated once per sample; the committed v2b estimator is
    evaluated once for every ordered configuration with the supplied v2a result.
    """

    total_start = time.perf_counter()
    _validate_class_mapping(class_mapping)
    grid = build_parameter_grid(
        (
            DEFAULT_MINIMUM_ATTACHMENT_EVIDENCE_FRACTIONS
            if minimum_attachment_evidence_fractions is None
            else minimum_attachment_evidence_fractions
        ),
        (
            DEFAULT_LATERAL_WINDOW_HALF_WIDTH_PIXELS
            if lateral_window_half_width_pixels is None
            else lateral_window_half_width_pixels
        ),
    )
    _validate_split(split)
    validated = _validate_runner_parameters(
        split="train",
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
        lateral_window_half_width_pixels=(
            COMMITTED_BASELINE_LATERAL_WINDOW_HALF_WIDTH_PIXELS
        ),
        minimum_attachment_evidence_fraction=(COMMITTED_BASELINE_EVIDENCE_FRACTION),
        minimum_flesh_band_pixels=minimum_flesh_band_pixels,
        minimum_calyx_band_pixels=minimum_calyx_band_pixels,
    )
    stability_threshold = _validate_nonnegative_finite(
        stability_threshold_pixels,
        name="stability_threshold_pixels",
    )
    inward_threshold = _validate_nonnegative_finite(
        maximum_inward_shift_threshold_pixels,
        name="maximum_inward_shift_threshold_pixels",
    )
    feasible_block_threshold = _validate_nonnegative_integer(
        feasible_block_count_flag_threshold,
        name="feasible_block_count_flag_threshold",
    )
    coordinate_tolerance = _validate_nonnegative_finite(
        coordinate_tolerance_pixels,
        name="coordinate_tolerance_pixels",
    )
    explicit_visualization_ids = _validate_visualization_configuration_ids(
        visualization_configuration_ids,
        grid,
    )
    fixed_parameters = _fixed_parameters(validated)
    configured_axis = tuple(validated["removal_axis"])
    source_root = Path(dataset_root)
    destination_root = Path(output_root)
    _validate_output_destination(source_root, destination_root, overwrite=overwrite)
    pairs = _paired_paths(source_root, split=split)
    non_training_warning = _non_training_split_warning(split)
    if non_training_warning is not None:
        print(f"WARNING: {non_training_warning}", file=sys.stderr)

    parameter_rows = [
        _parameter_grid_row(configuration, fixed_parameters) for configuration in grid
    ]
    configuration_runtime = {
        configuration.configuration_id: 0.0 for configuration in grid
    }
    destination_root.parent.mkdir(parents=True, exist_ok=True)
    staging_root = Path(
        tempfile.mkdtemp(
            prefix=f".{destination_root.name}.staging-",
            dir=destination_root.parent,
        )
    )
    try:
        plots_root = staging_root / "plots"
        plots_root.mkdir()
        write_visualizations = bool(explicit_visualization_ids) or bool(
            visualize_flagged_samples
        )
        visualization_root = staging_root / "visualizations"
        if write_visualizations:
            visualization_root.mkdir()

        result_rows: list[dict[str, Any]] = []
        stability_rows: list[dict[str, Any]] = []
        flagged_rows: list[dict[str, Any]] = []
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
            v1_proxies = calculate_line_side_proxies(mask, v1_result)
            v2a_proxies = calculate_fixed_axis_line_side_proxies(mask, v2a_result)

            sample_rows: list[dict[str, Any]] = []
            results_by_configuration: dict[str, FixedAxisSearchCutlineResult] = {}
            comparison_rows_by_configuration: dict[str, dict[str, Any]] = {}
            for configuration in grid:
                configuration_start = time.perf_counter()
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
                    lateral_window_half_width_pixels=(
                        configuration.lateral_window_half_width_pixels
                    ),
                    minimum_attachment_evidence_fraction=(
                        configuration.minimum_attachment_evidence_fraction
                    ),
                    minimum_flesh_band_pixels=minimum_flesh_band_pixels,
                    minimum_calyx_band_pixels=minimum_calyx_band_pixels,
                    v2a_result=v2a_result,
                )
                _validate_v2b_result(v2b_result, mask_shape=mask.shape)
                _validate_v2b_configuration(v2b_result, configuration)
                v2b_proxies = calculate_fixed_axis_search_line_side_proxies(
                    mask,
                    v2b_result,
                )
                comparison_row = _comparison_row(
                    image_path.stem,
                    v1_result,
                    v2a_result,
                    v2b_result,
                    v1_proxies=v1_proxies,
                    v2a_proxies=v2a_proxies,
                    v2b_proxies=v2b_proxies,
                )
                row = _sensitivity_row(
                    image_path.stem,
                    split,
                    configuration,
                    comparison_row,
                    v2b_result,
                    coordinate_tolerance_pixels=coordinate_tolerance,
                )
                configuration_runtime[configuration.configuration_id] += (
                    time.perf_counter() - configuration_start
                )
                sample_rows.append(row)
                result_rows.append(row)
                results_by_configuration[configuration.configuration_id] = v2b_result
                comparison_rows_by_configuration[configuration.configuration_id] = (
                    comparison_row
                )

            stability_row = _sample_stability_row(
                sample_rows,
                coordinate_tolerance_pixels=coordinate_tolerance,
            )
            sample_flags = _sample_flag_rows(
                stability_row,
                stability_threshold_pixels=stability_threshold,
                maximum_inward_shift_threshold_pixels=inward_threshold,
                feasible_block_count_flag_threshold=feasible_block_threshold,
            )
            stability_rows.append(stability_row)
            flagged_rows.extend(sample_flags)
            selected_visualization_ids = set(explicit_visualization_ids)
            if visualize_flagged_samples and sample_flags:
                selected_visualization_ids.update(
                    configuration.configuration_id for configuration in grid
                )
            visualization_paths = _write_selected_visualizations(
                staging_root,
                mask=mask,
                image=image,
                sample_id=image_path.stem,
                grid=grid,
                selected_configuration_ids=selected_visualization_ids,
                v1_result=v1_result,
                v2a_result=v2a_result,
                results_by_configuration=results_by_configuration,
                comparison_rows_by_configuration=comparison_rows_by_configuration,
                visualization_function=visualization_function,
            )
            manifest_samples.append(
                {
                    "sample_id": image_path.stem,
                    "success_count": stability_row["success_count"],
                    "failure_count": stability_row["failure_count"],
                    "flag_reasons": [row["flag_reason"] for row in sample_flags],
                    "visualization_paths": visualization_paths,
                }
            )

        configuration_rows = _configuration_summary_rows(
            result_rows,
            grid,
            configuration_runtime_seconds=configuration_runtime,
        )
        plot_paths = plot_function(configuration_rows, stability_rows, plots_root)
        expected_plot_names = {
            "aggregate_metric_heatmaps": "aggregate_metric_heatmaps.png",
            "status_and_fragmentation_heatmaps": (
                "status_and_fragmentation_heatmaps.png"
            ),
            "coordinate_stability": "coordinate_stability.png",
            "configuration_tradeoffs": "configuration_tradeoffs.png",
        }
        if plot_paths != expected_plot_names:
            raise ValueError(
                "plot_function must return the four expected deterministic plot names"
            )
        total_runtime_seconds = time.perf_counter() - total_start
        thresholds = {
            "coordinate_tolerance_pixels": coordinate_tolerance,
            "stability_threshold_pixels": stability_threshold,
            "maximum_inward_shift_threshold_pixels": inward_threshold,
            "feasible_block_count_flag_threshold": feasible_block_threshold,
        }
        artifacts: dict[str, Any] = {
            "manifest": "manifest.json",
            "summary": "summary.json",
            "parameter_grid": "parameter_grid.csv",
            "configuration_summary": "configuration_summary.csv",
            "per_sample_configuration_results": (
                "per_sample_configuration_results.csv"
            ),
            "per_sample_stability": "per_sample_stability.csv",
            "flagged_samples": "flagged_samples.csv",
            "plots": {
                name: f"plots/{filename}" for name, filename in plot_paths.items()
            },
        }
        if write_visualizations:
            artifacts["visualizations"] = "visualizations"
        summary = _build_summary(
            dataset_basename=source_root.name,
            split=split,
            non_training_warning=non_training_warning,
            parameter_rows=parameter_rows,
            fixed_parameters=fixed_parameters,
            configuration_rows=configuration_rows,
            stability_rows=stability_rows,
            flagged_rows=flagged_rows,
            thresholds=thresholds,
            total_runtime_seconds=total_runtime_seconds,
            artifacts=artifacts,
        )
        manifest: dict[str, Any] = {
            "study": "fixed_axis_v2b_parameter_sensitivity",
            "dataset_basename": source_root.name,
            "split": split,
            "sample_count": len(pairs),
            "non_training_split": split != "train",
            "non_training_split_warning": non_training_warning,
            "class_mapping": dict(class_mapping),
            "torch_free": True,
            "v1_algorithm_name": V1_ALGORITHM_NAME,
            "v1_algorithm_version": V1_ALGORITHM_VERSION,
            "v2a_algorithm_name": V2A_ALGORITHM_NAME,
            "v2a_algorithm_version": V2A_ALGORITHM_VERSION,
            "v2b_algorithm_name": V2B_ALGORITHM_NAME,
            "v2b_algorithm_version": V2B_ALGORITHM_VERSION,
            "configuration_ids": [
                configuration.configuration_id for configuration in grid
            ],
            "committed_baseline_configuration": (_baseline_configuration_definition()),
            "fixed_parameters": fixed_parameters,
            "thresholds": thresholds,
            "runtime": {
                "total_runtime_seconds": total_runtime_seconds,
                "aggregate_runtime_seconds_by_configuration": {
                    row["configuration_id"]: row["aggregate_runtime_seconds"]
                    for row in configuration_rows
                },
            },
            "artifacts": artifacts,
            "samples": manifest_samples,
        }
        _write_csv(
            staging_root / "parameter_grid.csv", PARAMETER_GRID_FIELDS, parameter_rows
        )
        _write_csv(
            staging_root / "configuration_summary.csv",
            CONFIGURATION_SUMMARY_FIELDS,
            configuration_rows,
        )
        _write_csv(
            staging_root / "per_sample_configuration_results.csv",
            PER_SAMPLE_CONFIGURATION_FIELDS,
            result_rows,
        )
        _write_csv(
            staging_root / "per_sample_stability.csv",
            PER_SAMPLE_STABILITY_FIELDS,
            stability_rows,
        )
        _write_csv(
            staging_root / "flagged_samples.csv",
            FLAGGED_SAMPLE_FIELDS,
            flagged_rows,
        )
        _write_json(staging_root / "summary.json", summary)
        _write_json(staging_root / "manifest.json", manifest)
        _install_staged_output(staging_root, destination_root, overwrite=overwrite)
    except Exception:
        if staging_root.exists():
            shutil.rmtree(staging_root)
        raise

    return {
        "manifest": manifest,
        "summary": summary,
        "parameter_grid_rows": parameter_rows,
        "configuration_rows": configuration_rows,
        "result_rows": result_rows,
        "stability_rows": stability_rows,
        "flagged_rows": flagged_rows,
    }


def _sensitivity_row(
    sample_id: str,
    split: str,
    configuration: SensitivityConfiguration,
    comparison_row: Mapping[str, Any],
    v2b_result: FixedAxisSearchCutlineResult,
    *,
    coordinate_tolerance_pixels: float,
) -> dict[str, Any]:
    v2a_coordinate = _finite_or_none(comparison_row["v2a_final_cut_coordinate"])
    v2b_coordinate = _finite_or_none(comparison_row["v2b_cut_coordinate"])
    coordinate_shift = _difference(v2b_coordinate, v2a_coordinate)
    comparable = coordinate_shift is not None and v2b_result.status == "ok"
    equals_v2a = (
        abs(coordinate_shift) <= coordinate_tolerance_pixels if comparable else None
    )
    moved_inward = (
        coordinate_shift < -coordinate_tolerance_pixels if comparable else None
    )
    inward_shift = max(0.0, -coordinate_shift) if comparable else None
    feasible_hull_width = _width(
        v2b_result.feasible_hull_start,
        v2b_result.feasible_hull_end,
    )
    selected_block_width = _width(
        v2b_result.selected_block_start,
        v2b_result.selected_block_end,
    )
    structured_status = (
        "ok"
        if v2b_result.status == "ok"
        else f"failed:{v2b_result.failure_code or 'unknown_failure'}"
    )
    return {
        "sample_id": sample_id,
        "split": split,
        "configuration_id": configuration.configuration_id,
        "minimum_attachment_evidence_fraction": (
            configuration.minimum_attachment_evidence_fraction
        ),
        "lateral_window_half_width_pixels": (
            configuration.lateral_window_half_width_pixels
        ),
        "is_committed_baseline": configuration.is_committed_baseline,
        "source_width": comparison_row["source_width"],
        "source_height": comparison_row["source_height"],
        "v1_status": comparison_row["v1_status"],
        "v1_failure_code": comparison_row["v1_failure_code"],
        "v1_failure_reason": comparison_row["v1_failure_reason"],
        "v2a_status": comparison_row["v2a_status"],
        "v2a_failure_code": comparison_row["v2a_failure_code"],
        "v2a_failure_reason": comparison_row["v2a_failure_reason"],
        "v2b_status": v2b_result.status,
        "v2b_failure_code": v2b_result.failure_code,
        "v2b_failure_reason": v2b_result.failure_reason,
        "structured_status": structured_status,
        "structured_failure_reason": (
            v2b_result.failure_reason if v2b_result.status != "ok" else None
        ),
        "v2a_cut_coordinate": v2a_coordinate,
        "v2b_cut_coordinate": v2b_coordinate,
        "v2b_minus_v2a_coordinate_shift": coordinate_shift,
        "v2b_equals_v2a_within_tolerance": equals_v2a,
        "v2b_moved_inward": moved_inward,
        "inward_shift_pixels": inward_shift,
        "coordinate_tolerance_pixels": coordinate_tolerance_pixels,
        "v2b_candidate_count": v2b_result.candidate_count,
        "feasible_candidate_count": v2b_result.feasible_candidate_count,
        "feasible_block_count": v2b_result.feasible_block_count,
        "feasible_hull_width_pixels": feasible_hull_width,
        "selected_block_width_pixels": selected_block_width,
        "selected_block_candidate_count": (v2b_result.selected_block_candidate_count),
        "selected_block_is_singleton": v2b_result.selected_block_is_singleton,
        "selected_block_is_largest": _selected_block_is_largest(v2b_result),
        "selected_local_contact_count": v2b_result.selected_local_contact_count,
        "selected_local_flesh_count": v2b_result.selected_local_flesh_count,
        "selected_local_calyx_count": v2b_result.selected_local_calyx_count,
        "selected_pair_flesh_loss_ratio": v2b_result.selected_flesh_loss_ratio,
        "selected_pair_calyx_retention_ratio": (
            v2b_result.selected_calyx_retention_ratio
        ),
        "selected_pair_retained_contact_ratio": v2b_result.retained_contact_ratio,
        "whole_mask_flesh_loss_ratio": comparison_row[
            "v2b_whole_mask_flesh_loss_ratio"
        ],
        "whole_mask_calyx_retention_ratio": comparison_row[
            "v2b_whole_mask_calyx_retention_ratio"
        ],
    }


def _configuration_summary_rows(
    rows: Sequence[Mapping[str, Any]],
    grid: Sequence[SensitivityConfiguration],
    *,
    configuration_runtime_seconds: Mapping[str, float],
) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    for configuration in grid:
        configuration_rows = [
            row
            for row in rows
            if row["configuration_id"] == configuration.configuration_id
        ]
        successful_rows = [
            row
            for row in configuration_rows
            if row["v2b_status"] == "ok"
            and _finite_or_none(row["v2b_cut_coordinate"]) is not None
        ]
        sample_count = len(configuration_rows)
        success_count = len(successful_rows)
        moved_inward_shifts = _finite_values(
            [
                row["inward_shift_pixels"]
                for row in successful_rows
                if row["v2b_moved_inward"] is True
            ]
        )
        feasible_block_counts = _finite_values(
            [row["feasible_block_count"] for row in configuration_rows]
        )
        singleton_count = sum(
            row["selected_block_is_singleton"] is True for row in successful_rows
        )
        not_largest_count = sum(
            row["selected_block_is_largest"] is False for row in successful_rows
        )
        summary = {
            "configuration_id": configuration.configuration_id,
            "minimum_attachment_evidence_fraction": (
                configuration.minimum_attachment_evidence_fraction
            ),
            "lateral_window_half_width_pixels": (
                configuration.lateral_window_half_width_pixels
            ),
            "is_committed_baseline": configuration.is_committed_baseline,
            "sample_count": sample_count,
            "success_count": success_count,
            "success_rate": success_count / sample_count if sample_count else None,
            "structured_failure_count": sample_count - success_count,
            **_mean_median_fields(
                successful_rows,
                source_field="whole_mask_flesh_loss_ratio",
                output_stem="whole_mask_flesh_loss_ratio",
            ),
            **_mean_median_fields(
                successful_rows,
                source_field="whole_mask_calyx_retention_ratio",
                output_stem="whole_mask_calyx_retention_ratio",
            ),
            **_mean_median_fields(
                successful_rows,
                source_field="selected_pair_flesh_loss_ratio",
                output_stem="selected_pair_flesh_loss_ratio",
            ),
            **_mean_median_fields(
                successful_rows,
                source_field="selected_pair_calyx_retention_ratio",
                output_stem="selected_pair_calyx_retention_ratio",
            ),
            **_mean_median_fields(
                successful_rows,
                source_field="selected_pair_retained_contact_ratio",
                output_stem="selected_pair_retained_contact_ratio",
            ),
            "v2b_equals_v2a_count": sum(
                row["v2b_equals_v2a_within_tolerance"] is True
                for row in successful_rows
            ),
            "moved_inward_count": len(moved_inward_shifts),
            "median_inward_shift_pixels": _median_or_none(moved_inward_shifts),
            "maximum_inward_shift_pixels": (
                max(moved_inward_shifts) if moved_inward_shifts else None
            ),
            "zero_feasible_count": sum(
                int(row["feasible_candidate_count"]) == 0 for row in configuration_rows
            ),
            "one_block_count": sum(
                int(row["feasible_block_count"]) == 1 for row in configuration_rows
            ),
            "fragmented_set_count": sum(
                int(row["feasible_block_count"]) > 1 for row in configuration_rows
            ),
            "fragmented_set_rate": (
                sum(int(row["feasible_block_count"]) > 1 for row in configuration_rows)
                / sample_count
                if sample_count
                else None
            ),
            "median_feasible_block_count": _median_or_none(feasible_block_counts),
            "maximum_feasible_block_count": (
                max(feasible_block_counts) if feasible_block_counts else None
            ),
            "singleton_selected_block_count": singleton_count,
            "singleton_selected_block_rate": (
                singleton_count / success_count if success_count else None
            ),
            "selected_block_not_largest_count": not_largest_count,
            "selected_block_not_largest_rate": (
                not_largest_count / success_count if success_count else None
            ),
            "median_selected_block_width_pixels": _median_or_none(
                _finite_values(
                    [row["selected_block_width_pixels"] for row in successful_rows]
                )
            ),
            "median_feasible_hull_width_pixels": _median_or_none(
                _finite_values(
                    [row["feasible_hull_width_pixels"] for row in configuration_rows]
                )
            ),
            "aggregate_runtime_seconds": float(
                configuration_runtime_seconds[configuration.configuration_id]
            ),
        }
        summaries.append(summary)

    baseline_rows = [row for row in summaries if row["is_committed_baseline"]]
    if len(baseline_rows) != 1:
        raise ValueError("Exactly one committed baseline summary is required")
    baseline = baseline_rows[0]
    for summary in summaries:
        for field in CONFIGURATION_BASELINE_CHANGE_FIELDS:
            summary[f"{field}_change_from_baseline"] = _difference(
                summary[field],
                baseline[field],
            )
    return summaries


def _sample_stability_row(
    rows: Sequence[Mapping[str, Any]],
    *,
    coordinate_tolerance_pixels: float,
) -> dict[str, Any]:
    if not rows:
        raise ValueError("sample stability requires at least one configuration row")
    sample_ids = {str(row["sample_id"]) for row in rows}
    splits = {str(row["split"]) for row in rows}
    if len(sample_ids) != 1 or len(splits) != 1:
        raise ValueError("sample stability rows must describe one sample and split")
    successful_rows = [
        row
        for row in rows
        if row["v2b_status"] == "ok"
        and _finite_or_none(row["v2b_cut_coordinate"]) is not None
    ]
    coordinates = _finite_values([row["v2b_cut_coordinate"] for row in successful_rows])
    configuration_count = len(rows)
    success_count = len(coordinates)
    coordinate_minimum = min(coordinates) if coordinates else None
    coordinate_maximum = max(coordinates) if coordinates else None
    coordinate_range = coordinate_maximum - coordinate_minimum if coordinates else None
    inward_shifts = _finite_values(
        [row["inward_shift_pixels"] for row in successful_rows]
    )
    feasible_blocks = [int(row["feasible_block_count"]) for row in rows]
    baseline_rows = [row for row in rows if row.get("is_committed_baseline") is True]
    if len(baseline_rows) > 1:
        raise ValueError("sample stability rows contain multiple committed baselines")
    baseline_coordinate = (
        _finite_or_none(baseline_rows[0]["v2b_cut_coordinate"])
        if baseline_rows and baseline_rows[0]["v2b_status"] == "ok"
        else None
    )
    return {
        "sample_id": next(iter(sample_ids)),
        "split": next(iter(splits)),
        "configuration_count": configuration_count,
        "success_count": success_count,
        "failure_count": configuration_count - success_count,
        "coordinate_minimum": coordinate_minimum,
        "coordinate_maximum": coordinate_maximum,
        "coordinate_median": _median_or_none(coordinates),
        "coordinate_range": coordinate_range,
        "coordinate_standard_deviation": (pstdev(coordinates) if coordinates else None),
        "committed_baseline_coordinate": baseline_coordinate,
        "configurations_equal_to_v2a_count": sum(
            row["v2b_equals_v2a_within_tolerance"] is True for row in successful_rows
        ),
        "configurations_moved_inward_count": sum(
            row["v2b_moved_inward"] is True for row in successful_rows
        ),
        "maximum_inward_shift_pixels": (max(inward_shifts) if inward_shifts else None),
        "minimum_feasible_block_count": min(feasible_blocks),
        "maximum_feasible_block_count": max(feasible_blocks),
        "singleton_selection_count": sum(
            row["selected_block_is_singleton"] is True for row in rows
        ),
        "selected_not_largest_count": sum(
            row["selected_block_is_largest"] is False for row in rows
        ),
        "unique_structured_status_count": len(
            {str(row["structured_status"]) for row in rows}
        ),
        "all_successful_configurations_same_coordinate_within_tolerance": (
            coordinate_range <= coordinate_tolerance_pixels
            if coordinate_range is not None
            else None
        ),
    }


def _sample_flag_rows(
    stability_row: Mapping[str, Any],
    *,
    stability_threshold_pixels: float,
    maximum_inward_shift_threshold_pixels: float,
    feasible_block_count_flag_threshold: int,
) -> list[dict[str, Any]]:
    flags: list[dict[str, Any]] = []

    def add(reason: str, observed: Any, comparison: str, threshold: Any) -> None:
        flags.append(
            {
                "sample_id": stability_row["sample_id"],
                "split": stability_row["split"],
                "flag_reason": reason,
                "observed_value": observed,
                "comparison": comparison,
                "threshold": threshold,
            }
        )

    if int(stability_row["failure_count"]) > 0:
        add("structured_failure", stability_row["failure_count"], ">", 0)
    coordinate_range = _finite_or_none(stability_row["coordinate_range"])
    if coordinate_range is not None and coordinate_range > stability_threshold_pixels:
        add(
            "coordinate_range_exceeds_threshold",
            coordinate_range,
            ">",
            stability_threshold_pixels,
        )
    if int(stability_row["singleton_selection_count"]) > 0:
        add(
            "singleton_selected_block",
            stability_row["singleton_selection_count"],
            ">",
            0,
        )
    if int(stability_row["selected_not_largest_count"]) > 0:
        add(
            "selected_block_not_largest",
            stability_row["selected_not_largest_count"],
            ">",
            0,
        )
    if int(stability_row["maximum_feasible_block_count"]) > (
        feasible_block_count_flag_threshold
    ):
        add(
            "feasible_block_count_exceeds_threshold",
            stability_row["maximum_feasible_block_count"],
            ">",
            feasible_block_count_flag_threshold,
        )
    maximum_inward = _finite_or_none(stability_row["maximum_inward_shift_pixels"])
    if maximum_inward is not None and maximum_inward > (
        maximum_inward_shift_threshold_pixels
    ):
        add(
            "maximum_inward_shift_exceeds_threshold",
            maximum_inward,
            ">",
            maximum_inward_shift_threshold_pixels,
        )
    return flags


def _build_summary(
    *,
    dataset_basename: str,
    split: str,
    non_training_warning: str | None,
    parameter_rows: Sequence[Mapping[str, Any]],
    fixed_parameters: Mapping[str, Any],
    configuration_rows: Sequence[Mapping[str, Any]],
    stability_rows: Sequence[Mapping[str, Any]],
    flagged_rows: Sequence[Mapping[str, Any]],
    thresholds: Mapping[str, Any],
    total_runtime_seconds: float,
    artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    flag_counts = Counter(str(row["flag_reason"]) for row in flagged_rows)
    flagged_sample_ids = {str(row["sample_id"]) for row in flagged_rows}
    coordinate_ranges = _finite_values(
        [row["coordinate_range"] for row in stability_rows]
    )
    coordinate_standard_deviations = _finite_values(
        [row["coordinate_standard_deviation"] for row in stability_rows]
    )
    return {
        "dataset_basename": dataset_basename,
        "split": split,
        "sample_count": len(stability_rows),
        "non_training_split": split != "train",
        "non_training_split_warning": non_training_warning,
        "full_parameter_grid": [dict(row) for row in parameter_rows],
        "fixed_parameters": dict(fixed_parameters),
        "committed_baseline_configuration": _baseline_configuration_definition(),
        "output_definitions": {
            "artifacts": dict(artifacts),
            "aggregate_definition": AGGREGATE_DEFINITION,
            "stability_definition": STABILITY_DEFINITION,
            "inward_definition": INWARD_DEFINITION,
            "feasible_block_definition": FEASIBLE_BLOCK_DEFINITION,
            "feasible_hull_definition": FEASIBLE_HULL_DEFINITION,
            "selected_block_definition": SELECTED_BLOCK_DEFINITION,
            "feasible_set_caution": FEASIBLE_SET_CAUTION,
            "configuration_ordering": (
                "ascending evidence fraction, then ascending lateral-window half-width"
            ),
            "sample_ordering": "ascending sample ID from normalized split pairs",
        },
        "proxy_cautions": list(PROXY_CAUTIONS),
        "difference_convention": DIFFERENCE_CONVENTION,
        "thresholds": dict(thresholds),
        "aggregate_configuration_statistics": [dict(row) for row in configuration_rows],
        "per_sample_stability_statistics": [dict(row) for row in stability_rows],
        "per_sample_stability_aggregate": {
            "sample_count": len(stability_rows),
            "samples_with_any_failure_count": sum(
                int(row["failure_count"]) > 0 for row in stability_rows
            ),
            "samples_with_all_successful_coordinates_same_count": sum(
                row["all_successful_configurations_same_coordinate_within_tolerance"]
                is True
                for row in stability_rows
            ),
            "coordinate_range": _statistics(coordinate_ranges),
            "coordinate_standard_deviation": _statistics(
                coordinate_standard_deviations
            ),
        },
        "flagged_sample_counts": {
            "unique_sample_count": len(flagged_sample_ids),
            "flag_record_count": len(flagged_rows),
            "by_reason": dict(sorted(flag_counts.items())),
        },
        "total_runtime_seconds": total_runtime_seconds,
        "runtime_definition": (
            "Wall-clock seconds from runner entry through aggregate plot "
            "generation; final JSON/CSV metadata writes and atomic rename are "
            "excluded. Per-configuration runtime sums v2b estimation, proxy "
            "calculation, and row construction only"
        ),
        "automatic_configuration_selection": False,
        "study_limitation": STUDY_LIMITATION,
    }


def _write_selected_visualizations(
    staging_root: Path,
    *,
    mask: np.ndarray,
    image: Image.Image,
    sample_id: str,
    grid: Sequence[SensitivityConfiguration],
    selected_configuration_ids: set[str],
    v1_result: MaskCutlineResult,
    v2a_result: FixedAxisCutlineResult,
    results_by_configuration: Mapping[str, FixedAxisSearchCutlineResult],
    comparison_rows_by_configuration: Mapping[str, Mapping[str, Any]],
    visualization_function: ComparisonVisualizationFunction,
) -> list[str]:
    paths: list[str] = []
    for configuration in grid:
        configuration_id = configuration.configuration_id
        if configuration_id not in selected_configuration_ids:
            continue
        relative_path = Path("visualizations") / configuration_id / f"{sample_id}.png"
        output_path = staging_root / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)
        visualization = visualization_function(
            mask,
            v1_result,
            v2a_result,
            results_by_configuration[configuration_id],
            image=image,
            metrics=comparison_rows_by_configuration[configuration_id],
        )
        if not isinstance(visualization, Image.Image):
            raise TypeError("visualization_function must return a PIL.Image.Image")
        visualization.save(output_path, format="PNG")
        paths.append(relative_path.as_posix())
    return paths


def _parameter_grid_row(
    configuration: SensitivityConfiguration,
    fixed_parameters: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "configuration_id": configuration.configuration_id,
        "minimum_attachment_evidence_fraction": (
            configuration.minimum_attachment_evidence_fraction
        ),
        "lateral_window_half_width_pixels": (
            configuration.lateral_window_half_width_pixels
        ),
        **dict(fixed_parameters),
        "is_committed_baseline": configuration.is_committed_baseline,
    }


def _fixed_parameters(validated: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "removal_axis_x": validated["removal_axis"][0],
        "removal_axis_y": validated["removal_axis"][1],
        "projection_quantile": validated["projection_quantile"],
        "support_band_width_pixels": validated["support_band_width_pixels"],
        "calyx_dilation_radius": validated["calyx_dilation_radius"],
        "component_connectivity": validated["component_connectivity"],
        "signed_offset_pixels": validated["v2a_signed_offset_pixels"],
        "candidate_step_pixels": validated["candidate_step_pixels"],
        "inward_search_margin_pixels": validated["inward_search_margin_pixels"],
        "outward_search_margin_pixels": validated["outward_search_margin_pixels"],
        "blade_band_half_width_pixels": validated["blade_band_half_width_pixels"],
        "minimum_flesh_band_pixels": validated["minimum_flesh_band_pixels"],
        "minimum_calyx_band_pixels": validated["minimum_calyx_band_pixels"],
    }


def _baseline_configuration_definition() -> dict[str, Any]:
    return {
        "configuration_id": COMMITTED_BASELINE_CONFIGURATION_ID,
        "minimum_attachment_evidence_fraction": (COMMITTED_BASELINE_EVIDENCE_FRACTION),
        "lateral_window_half_width_pixels": (
            COMMITTED_BASELINE_LATERAL_WINDOW_HALF_WIDTH_PIXELS
        ),
    }


def _configuration_id(evidence: float, width: float) -> str:
    return f"e{_identifier_token(evidence, scale=100)}_w{_identifier_token(width)}"


def _identifier_token(value: float, *, scale: int = 1) -> str:
    scaled = Decimal(str(value)) * Decimal(scale)
    rendered = format(scaled.normalize(), "f")
    integer, separator, fraction = rendered.partition(".")
    integer = integer.zfill(3)
    return integer if not separator else f"{integer}p{fraction.rstrip('0')}"


def _validate_grid_values(
    values: Sequence[Real],
    *,
    name: str,
    lower_exclusive: float,
    upper_inclusive: float | None = None,
) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)) or not isinstance(values, Sequence):
        raise TypeError(f"{name} grid must be a sequence of real numbers")
    if not values:
        raise ValueError(f"{name} grid must not be empty")
    validated: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise TypeError(f"{name} grid values must be real numbers")
        numeric = float(value)
        if not math.isfinite(numeric):
            raise ValueError(f"{name} grid values must be finite")
        if numeric <= lower_exclusive:
            raise ValueError(f"{name} grid values must be greater than zero")
        if upper_inclusive is not None and numeric > upper_inclusive:
            raise ValueError(
                f"{name} grid values must be no greater than {upper_inclusive:g}"
            )
        validated.append(numeric)
    if len(set(validated)) != len(validated):
        raise ValueError(f"{name} grid values must be unique")
    return tuple(sorted(validated))


def _validate_class_mapping(class_mapping: Mapping[str, int]) -> None:
    if not isinstance(class_mapping, Mapping):
        raise TypeError("class_mapping must be a mapping")
    if any(
        isinstance(value, bool) or not isinstance(value, int)
        for value in class_mapping.values()
    ):
        raise ValueError("class_mapping values must be integer class IDs")
    if dict(class_mapping) != CLASS_MAPPING:
        raise ValueError(
            "class_mapping must exactly match background=0, Flesh=1, Calyx=2"
        )


def _validate_split(split: str) -> None:
    if split not in SENSITIVITY_SUPPORTED_SPLITS:
        raise ValueError(
            "split must be one of "
            f"{', '.join(SENSITIVITY_SUPPORTED_SPLITS)}, got {split!r}"
        )


def _validate_nonnegative_finite(value: Real, *, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise TypeError(f"{name} must be a real number")
    numeric = float(value)
    if not math.isfinite(numeric) or numeric < 0.0:
        raise ValueError(f"{name} must be finite and non-negative")
    return numeric


def _validate_nonnegative_integer(value: int, *, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _validate_visualization_configuration_ids(
    values: Sequence[str],
    grid: Sequence[SensitivityConfiguration],
) -> tuple[str, ...]:
    if isinstance(values, (str, bytes)):
        raise TypeError("visualization_configuration_ids must be a sequence")
    known = {configuration.configuration_id for configuration in grid}
    result: list[str] = []
    for value in values:
        if not isinstance(value, str) or not value:
            raise ValueError("visualization configuration IDs must be non-empty")
        if value not in known:
            raise ValueError(f"Unknown visualization configuration ID: {value}")
        if value not in result:
            result.append(value)
    return tuple(result)


def _validate_v2b_configuration(
    result: FixedAxisSearchCutlineResult,
    configuration: SensitivityConfiguration,
) -> None:
    if result.parameters.minimum_attachment_evidence_fraction != (
        configuration.minimum_attachment_evidence_fraction
    ):
        raise ValueError("v2b result evidence fraction must match its configuration")
    if result.parameters.lateral_window_half_width_pixels != (
        configuration.lateral_window_half_width_pixels
    ):
        raise ValueError("v2b result lateral-window width must match its configuration")
    if result.succeeded:
        required_finite = {
            "selected_cut_coordinate": result.selected_cut_coordinate,
            "selected_flesh_loss_ratio": result.selected_flesh_loss_ratio,
            "selected_calyx_retention_ratio": result.selected_calyx_retention_ratio,
            "retained_contact_ratio": result.retained_contact_ratio,
        }
        for name, value in required_finite.items():
            if value is None or not math.isfinite(float(value)):
                raise ValueError(f"successful v2b result has non-finite {name}")


def _non_training_split_warning(split: str) -> str | None:
    if split == "train":
        return None
    return (
        f"Split {split!r} is not TRAIN. Do not use this run to tune or freeze "
        "parameters unless that use was explicitly planned after configuration "
        "freezing."
    )


def _mean_median_fields(
    rows: Sequence[Mapping[str, Any]],
    *,
    source_field: str,
    output_stem: str,
) -> dict[str, float | None]:
    values = _finite_values([row[source_field] for row in rows])
    return {
        f"mean_{output_stem}": mean(values) if values else None,
        f"median_{output_stem}": median(values) if values else None,
    }


def _statistics(values: Sequence[float]) -> dict[str, float | int | None]:
    if not values:
        return {
            "finite_count": 0,
            "minimum": None,
            "maximum": None,
            "mean": None,
            "median": None,
        }
    return {
        "finite_count": len(values),
        "minimum": min(values),
        "maximum": max(values),
        "mean": mean(values),
        "median": median(values),
    }


def _finite_values(values: Sequence[Any]) -> list[float]:
    return [
        numeric for value in values if (numeric := _finite_or_none(value)) is not None
    ]


def _finite_or_none(value: Any) -> float | None:
    if value is None:
        return None
    numeric = float(value)
    return numeric if math.isfinite(numeric) else None


def _median_or_none(values: Sequence[float]) -> float | None:
    return median(values) if values else None


def _width(start: float | None, end: float | None) -> float | None:
    if start is None or end is None:
        return None
    return end - start


def _difference(current: Any, baseline: Any) -> int | float | None:
    if current is None or baseline is None:
        return None
    if isinstance(current, int) and isinstance(baseline, int):
        return current - baseline
    current_numeric = float(current)
    baseline_numeric = float(baseline)
    if not math.isfinite(current_numeric) or not math.isfinite(baseline_numeric):
        return None
    return current_numeric - baseline_numeric


def _write_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _csv_value(row[field]) for field in fields})


def _parse_repeated_float_values(
    values: Sequence[str] | None,
) -> tuple[float, ...] | None:
    if values is None:
        return None
    parsed: list[float] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part for part in parts):
            raise ValueError("Sweep values must not contain empty entries")
        parsed.extend(float(part) for part in parts)
    return tuple(parsed)


def _parse_repeated_string_values(values: Sequence[str] | None) -> tuple[str, ...]:
    if values is None:
        return ()
    parsed: list[str] = []
    for value in values:
        parts = [part.strip() for part in value.split(",")]
        if not parts or any(not part for part in parts):
            raise ValueError("Configuration IDs must not contain empty entries")
        parsed.extend(parts)
    return tuple(parsed)


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run deterministic fixed-axis-v2b parameter sensitivity on one "
            "explicitly selected normalized-dataset split."
        )
    )
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument(
        "--split",
        choices=SENSITIVITY_SUPPORTED_SPLITS,
        required=True,
    )
    parser.add_argument(
        "--minimum-attachment-evidence-fraction",
        action="append",
        dest="minimum_attachment_evidence_fractions",
        help=("repeat or pass comma-separated sweep values; default is 0.30,0.50,0.70"),
    )
    parser.add_argument(
        "--lateral-window-half-width-pixels",
        action="append",
        dest="lateral_window_half_width_pixels",
        help="repeat or pass comma-separated sweep values; default is 48,64,96",
    )
    parser.add_argument("--removal-axis-x", type=float, default=1.0)
    parser.add_argument("--removal-axis-y", type=float, default=0.0)
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
    parser.add_argument("--inward-search-margin-pixels", type=float, default=10.0)
    parser.add_argument("--outward-search-margin-pixels", type=float, default=0.0)
    parser.add_argument("--blade-band-half-width-pixels", type=float, default=1.0)
    parser.add_argument("--minimum-flesh-band-pixels", type=int, default=1)
    parser.add_argument("--minimum-calyx-band-pixels", type=int, default=1)
    parser.add_argument(
        "--stability-threshold-pixels",
        type=float,
        default=DEFAULT_STABILITY_THRESHOLD_PIXELS,
    )
    parser.add_argument(
        "--maximum-inward-shift-threshold-pixels",
        type=float,
        default=DEFAULT_MAXIMUM_INWARD_SHIFT_THRESHOLD_PIXELS,
    )
    parser.add_argument(
        "--coordinate-tolerance-pixels",
        type=float,
        default=DEFAULT_COORDINATE_TOLERANCE_PIXELS,
    )
    parser.add_argument(
        "--feasible-block-count-flag-threshold",
        type=int,
        default=DEFAULT_FEASIBLE_BLOCK_COUNT_FLAG_THRESHOLD,
    )
    parser.add_argument(
        "--visualize-configuration-id",
        action="append",
        help="repeat or pass comma-separated configuration IDs",
    )
    parser.add_argument("--visualize-flagged-samples", action="store_true")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing output directory after successful generation",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the fixed-axis-v2b sensitivity CLI."""

    parser = _build_argument_parser()
    arguments = parser.parse_args(argv)
    try:
        evidence_values = _parse_repeated_float_values(
            arguments.minimum_attachment_evidence_fractions
        )
        width_values = _parse_repeated_float_values(
            arguments.lateral_window_half_width_pixels
        )
        visualization_ids = _parse_repeated_string_values(
            arguments.visualize_configuration_id
        )
    except ValueError as error:
        parser.error(str(error))
    output = run_fixed_axis_sensitivity_dataset(
        arguments.dataset_root,
        arguments.output_root,
        split=arguments.split,
        minimum_attachment_evidence_fractions=evidence_values,
        lateral_window_half_width_pixels=width_values,
        removal_axis=(arguments.removal_axis_x, arguments.removal_axis_y),
        projection_quantile=arguments.projection_quantile,
        support_band_width_pixels=arguments.support_band_width_pixels,
        calyx_dilation_radius=arguments.calyx_dilation_radius,
        component_connectivity=arguments.component_connectivity,
        signed_offset_pixels=arguments.signed_offset_pixels,
        candidate_step_pixels=arguments.candidate_step_pixels,
        inward_search_margin_pixels=arguments.inward_search_margin_pixels,
        outward_search_margin_pixels=arguments.outward_search_margin_pixels,
        blade_band_half_width_pixels=arguments.blade_band_half_width_pixels,
        minimum_flesh_band_pixels=arguments.minimum_flesh_band_pixels,
        minimum_calyx_band_pixels=arguments.minimum_calyx_band_pixels,
        stability_threshold_pixels=arguments.stability_threshold_pixels,
        maximum_inward_shift_threshold_pixels=(
            arguments.maximum_inward_shift_threshold_pixels
        ),
        feasible_block_count_flag_threshold=(
            arguments.feasible_block_count_flag_threshold
        ),
        coordinate_tolerance_pixels=arguments.coordinate_tolerance_pixels,
        visualization_configuration_ids=visualization_ids,
        visualize_flagged_samples=arguments.visualize_flagged_samples,
        overwrite=arguments.overwrite,
    )
    summary = output["summary"]
    print(
        f"Processed {summary['sample_count']} {summary['split']} samples across "
        f"{len(summary['aggregate_configuration_statistics'])} configurations."
    )
    print("No best configuration or opaque combined score was generated.")
    print(STUDY_LIMITATION)
    print(f"Artifacts written to {arguments.output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
