"""Run a public-safe, torch-free demonstration of mask-to-action geometry.

The demo constructs procedural RGB images and semantic class-index masks. It
does not load a segmentation model, checkpoint, annotation export, or dataset.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

from strawberry_occlusion.augmentation.synthetic_occluder import (
    MATCHED_REGIONS,
    apply_occluder,
    build_matched_occluder_set,
    generate_synthetic_occluder,
)
from strawberry_occlusion.geometry import (
    estimate_fixed_axis_cutline,
    estimate_fixed_axis_search_cutline,
    estimate_visible_mask_cutline,
)
from strawberry_occlusion.visualization.fixed_axis_comparison import (
    create_fixed_axis_comparison_visualization,
)


CLASS_MAPPING = {"background": 0, "Flesh": 1, "Calyx": 2}
DEFAULT_OUTPUT_DIR = Path("toy_cutline_demo_output")
REMOVAL_AXIS = (1.0, 0.0)


def make_success_case() -> tuple[np.ndarray, np.ndarray]:
    """Return a procedural RGB image and mask with usable attachment evidence."""

    height, width = 112, 160
    pixel_y, pixel_x = np.indices((height, width))
    flesh = ((pixel_x - 72.0) / 49.0) ** 2 + ((pixel_y - 56.0) / 35.0) ** 2 <= 1.0

    mask = np.zeros((height, width), dtype=np.uint8)
    mask[flesh] = CLASS_MAPPING["Flesh"]
    mask[48:65, 116:146] = CLASS_MAPPING["Calyx"]

    image = _toy_rgb(mask)
    return image, mask


def make_insufficient_evidence_case() -> tuple[np.ndarray, np.ndarray]:
    """Return a procedural case with no visible Calyx attachment evidence."""

    image, mask = make_success_case()
    insufficient_mask = mask.copy()
    insufficient_mask[insufficient_mask == CLASS_MAPPING["Calyx"]] = CLASS_MAPPING[
        "background"
    ]
    return _toy_rgb(insufficient_mask), insufficient_mask


def run_demo(
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    *,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Run both toy cases and write deterministic visualizations plus JSON."""

    destination = _validated_output_dir(output_dir)
    if destination.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {Path(output_dir)}. "
                "Pass --overwrite to replace it."
            )
        if not destination.is_dir():
            raise ValueError(f"Output path is not a directory: {Path(output_dir)}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    cases = {
        "successful_coordinate": make_success_case(),
        "insufficient_evidence": make_insufficient_evidence_case(),
    }
    summaries: dict[str, dict[str, Any]] = {}
    for case_name, (image, mask) in cases.items():
        _validate_public_mask(mask)
        v1 = estimate_visible_mask_cutline(mask)
        v2a = estimate_fixed_axis_cutline(mask, removal_axis=REMOVAL_AXIS)
        v2b = estimate_fixed_axis_search_cutline(
            mask,
            removal_axis=REMOVAL_AXIS,
            v2a_result=v2a,
        )
        visualization = create_fixed_axis_comparison_visualization(
            mask,
            v1,
            v2a,
            v2b,
            image=image,
        )
        visualization_name = f"{case_name}.png"
        visualization.save(destination / visualization_name, format="PNG")
        summaries[case_name] = {
            "semantic_class_ids": sorted(int(value) for value in np.unique(mask)),
            "visualization": visualization_name,
            "v1": _result_summary(v1, coordinate_attribute=None),
            "v2a": _result_summary(
                v2a,
                coordinate_attribute="final_cut_coordinate",
            ),
            "v2b": _result_summary(
                v2b,
                coordinate_attribute="selected_cut_coordinate",
            ),
        }

    matched_figure_name = "matched_occlusion_regions.svg"
    matched_figure_details = write_matched_occlusion_figure(
        destination / matched_figure_name
    )
    summary: dict[str, Any] = {
        "demo": "procedural mask-to-action geometry",
        "input_provenance": "procedurally generated; no dataset or checkpoint",
        "class_mapping": CLASS_MAPPING,
        "coordinate_space": "toy image pixels",
        "machine_calibrated": False,
        "demonstrates_segmentation_model_accuracy": False,
        "removal_axis": list(REMOVAL_AXIS),
        "matched_occlusion_figure": {
            "visualization": matched_figure_name,
            **matched_figure_details,
        },
        "cases": summaries,
    }
    summary_path = destination / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def write_matched_occlusion_figure(output_path: Path) -> dict[str, Any]:
    """Render an SVG using the repository's actual matched-placement code."""

    image, mask = make_success_case()
    v2a = estimate_fixed_axis_cutline(mask, removal_axis=REMOVAL_AXIS)
    foreground_area = int(np.count_nonzero(np.isin(mask, (1, 2))))
    occluder = generate_synthetic_occluder(
        sample_id="public_toy",
        severity_fraction=0.02,
        seed=0,
        foreground_area_pixels=foreground_area,
    )
    matched = build_matched_occluder_set(
        mask,
        v2a,
        occluder,
        attachment_roi_dilation_pixels=5,
        background_separation_pixels=2,
    )
    if not matched.complete:
        raise RuntimeError(
            "Procedural README example did not produce all matched regions: "
            f"{matched.unavailable_reasons}"
        )
    for region in MATCHED_REGIONS:
        # Validate that every matched placement can be applied without clipping.
        apply_occluder(image, occluder, matched.placements[region])

    svg = _matched_occlusion_svg(mask, occluder.mask, matched.placements)
    output_path.write_text(svg, encoding="utf-8")
    return {
        "provenance": "procedural toy data and repository matched-placement code",
        "regions": list(MATCHED_REGIONS),
        "severity_fraction": 0.02,
        "seed": 0,
        "same_template_translated": True,
        "occluder_area_pixels": occluder.area_pixels,
    }


def _matched_occlusion_svg(
    mask: np.ndarray,
    occluder_mask: np.ndarray,
    placements: dict[str, Any],
) -> str:
    panel_width = mask.shape[1]
    panel_height = mask.shape[0]
    panel_x = (18, 207, 396, 585)
    panel_y = 36
    flesh_path = _mask_to_svg_path(mask == CLASS_MAPPING["Flesh"])
    calyx_path = _mask_to_svg_path(mask == CLASS_MAPPING["Calyx"])
    occluder_path = _mask_to_svg_path(occluder_mask)
    labels = {
        "attachment": "Attachment",
        "calyx_tip": "Calyx tip",
        "flesh_far": "Distant Flesh",
        "background": "Background",
    }
    panels = []
    for x, region in zip(panel_x, MATCHED_REGIONS, strict=True):
        placement = placements[region]
        panels.append(
            f'''  <g transform="translate({x} {panel_y})">
    <rect width="{panel_width}" height="{panel_height}" rx="5" fill="#15212b"/>
    <use href="#flesh"/>
    <use href="#calyx"/>
    <use href="#occluder" x="{placement.left}" y="{placement.top}"/>
    <rect width="{panel_width}" height="{panel_height}" rx="5" fill="none" stroke="#8fa5b5"/>
  </g>
  <text x="{x + panel_width / 2:g}" y="29" text-anchor="middle" class="label">{labels[region]}</text>'''
        )
    panel_markup = "\n".join(panels)
    return f'''<svg xmlns="http://www.w3.org/2000/svg" width="780" height="184" viewBox="0 0 780 184" role="img" aria-labelledby="title description">
  <title id="title">Matched four-region procedural occlusion example</title>
  <desc id="description">The same procedurally generated occluder is translated to attachment, calyx tip, distant Flesh, and background regions of a toy semantic fruit.</desc>
  <style>
    text {{ font-family: system-ui, sans-serif; fill: #203040; }}
    .label {{ font-size: 13px; font-weight: 650; }}
    .note {{ font-size: 12px; }}
  </style>
  <defs>
    <path id="flesh" d="{flesh_path}" fill="#cf4f50"/>
    <path id="calyx" d="{calyx_path}" fill="#56a96a"/>
    <path id="occluder" d="{occluder_path}" fill="#315f31" stroke="#d5e8a8" stroke-width="0.45"/>
  </defs>
  <rect width="780" height="184" fill="#f7fafb"/>
{panel_markup}
  <text x="390" y="165" text-anchor="middle" class="note">Same deterministic occluder; only integer translation changes between matched regions.</text>
  <text x="390" y="180" text-anchor="middle" class="note">Procedurally generated benchmark example — not a real fruit image.</text>
</svg>
'''


def _mask_to_svg_path(binary_mask: np.ndarray) -> str:
    commands: list[str] = []
    for y, row in enumerate(np.asarray(binary_mask, dtype=bool)):
        padded = np.pad(row, (1, 1), constant_values=False)
        changes = np.flatnonzero(padded[1:] != padded[:-1])
        for start, end in changes.reshape(-1, 2):
            commands.append(
                f"M{int(start)} {y}h{int(end - start)}v1h-{int(end - start)}z"
            )
    return "".join(commands)


def _toy_rgb(mask: np.ndarray) -> np.ndarray:
    height, width = mask.shape
    pixel_y, pixel_x = np.indices((height, width))
    image = np.empty((height, width, 3), dtype=np.uint8)
    image[..., 0] = np.clip(28 + pixel_x // 8, 0, 255)
    image[..., 1] = np.clip(38 + pixel_y // 7, 0, 255)
    image[..., 2] = 48
    image[mask == CLASS_MAPPING["Flesh"]] = (190, 50, 55)
    image[mask == CLASS_MAPPING["Calyx"]] = (55, 145, 65)
    return image


def _result_summary(
    result: Any,
    *,
    coordinate_attribute: str | None,
) -> dict[str, Any]:
    coordinate = (
        None if coordinate_attribute is None else getattr(result, coordinate_attribute)
    )
    return {
        "status": result.status,
        "failure_code": result.failure_code,
        "coordinate_pixels": None if coordinate is None else float(coordinate),
    }


def _validate_public_mask(mask: np.ndarray) -> None:
    if mask.ndim != 2 or mask.dtype != np.uint8:
        raise ValueError("Toy masks must be single-channel uint8 arrays")
    values = set(int(value) for value in np.unique(mask))
    if not values <= set(CLASS_MAPPING.values()):
        raise ValueError(f"Unexpected semantic class IDs: {sorted(values)}")


def _validated_output_dir(output_dir: str | Path) -> Path:
    path = Path(output_dir)
    if path.is_absolute():
        raise ValueError("--output-dir must be a relative path")
    workspace = Path.cwd().resolve()
    destination = (workspace / path).resolve()
    if not destination.is_relative_to(workspace) or destination == workspace:
        raise ValueError("--output-dir must stay inside the current working directory")
    return destination


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run a procedural, torch-free demo of semantic-mask-to-cutline "
            "geometry. This does not evaluate segmentation-model accuracy."
        )
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="replace an existing demo output directory",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the toy demo CLI."""

    arguments = _build_argument_parser().parse_args(argv)
    try:
        summary = run_demo(arguments.output_dir, overwrite=arguments.overwrite)
    except (FileExistsError, OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2

    success = summary["cases"]["successful_coordinate"]
    insufficient = summary["cases"]["insufficient_evidence"]
    print(
        "Successful toy case: "
        f"v2a={success['v2a']['coordinate_pixels']:.3f} px, "
        f"v2b={success['v2b']['coordinate_pixels']:.3f} px."
    )
    print(
        "Structured failure toy case: "
        f"v2a status={insufficient['v2a']['status']} "
        f"({insufficient['v2a']['failure_code']}), "
        f"v2b status={insufficient['v2b']['status']} "
        f"({insufficient['v2b']['failure_code']})."
    )
    print(
        f"Wrote sanitized toy artifacts to {arguments.output_dir}. "
        "Coordinates are image pixels, not machine-calibrated units."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
