# Cut-coordinate methods

This repository studies how a visible semantic mask is converted into a
one-dimensional action coordinate. The estimators operate on a single-channel
class-index mask:

| Class | ID |
|---|---:|
| Background | 0 |
| Flesh | 1 |
| Calyx | 2 |

Every coordinate remains in the input mask's image space. For model-based
evaluation, that is the checkpoint-preprocessed image space. The values are
pixels, not millimetres, and the code performs no camera-to-machine calibration.

## From visible mask to attachment evidence

The shared visible-mask stage validates the ontology, identifies connected
Flesh and Calyx components, and selects a component pair with visible contact
after a small Calyx dilation. Missing Flesh, missing Calyx, missing contact, or
degenerate geometry produces an explicit failure code instead of a fabricated
coordinate.

The three estimators then use that evidence differently.

### v1: visible-mask baseline

`estimate_visible_mask_cutline` derives a fruit-to-attachment direction from the
selected Flesh centroid and visible contact centroid. It places a perpendicular
line through the attachment anchor, optionally shifted along that direction.

Because both orientation and position come from visible mask geometry, this
baseline is intentionally sensitive to changes in the visible attachment shape.

### v2a: fixed-axis robust-contact coordinate

`estimate_fixed_axis_cutline` keeps a configured removal axis fixed. It projects
the visible contact pixels onto that axis and selects a deterministic,
non-interpolated projection quantile. The resulting cutline is perpendicular to
the configured axis.

The support band recorded by v2a is diagnostic: it describes nearby contact
support but does not change the selected coordinate. This separates cutline
orientation from potentially unstable visible-shape orientation.

### v2b: fixed-axis feasibility search

`estimate_fixed_axis_search_cutline` starts from v2a and evaluates an ordered
one-dimensional set of candidate coordinates. Each candidate records local
contact, Flesh, Calyx, and v2a-support evidence inside a fixed lateral window.
The estimator returns the outermost candidate in the outermost feasible block.

If no candidate satisfies the configured evidence requirements, v2b returns the
structured failure `no_feasible_fixed_axis_cut`. The selected coordinate and the
associated pixel-count ratios are research proxies; they do not establish true
attachment severance or physical cut quality.

## Estimator provenance

The intended experimental chain is:

1. v2b parameter sensitivity is examined on training masks;
2. the reference configuration is frozen;
3. that configuration is applied unchanged in later benchmark evaluation.

The frozen reference is `e050_w064`, including a minimum attachment-evidence
fraction of 0.50 and a lateral half-window of 64 pixels. Those values are
image-space research settings, not industrial production thresholds.

## Segmentation-to-action comparisons

The evaluation pipeline can derive coordinates independently from:

- a manually labelled visible semantic mask, producing a reference coordinate;
- a predicted visible semantic mask, producing the model-conditioned coordinate;
- an occluded prediction, producing a perturbed coordinate or structured failure.

This supports two different questions:

- **Agreement:** how far is a prediction-derived coordinate from the
  manual-mask-derived coordinate?
- **Stability:** how far does the coordinate move after a controlled visual
  perturbation relative to the clean prediction?

Agreement and stability are not physical measurements. A stable estimator can
be consistently wrong, and a low pixel difference is not a calibrated
tolerance.

## Failure awareness

The geometry APIs return status, failure code, diagnostics, and nullable
coordinates. This permits evaluation to keep two outcomes visible:

- structured failures, where the estimator declines to return a finite
  coordinate; and
- silent drift, where a finite coordinate is returned but moves beyond a
  descriptive pixel threshold.

The current code measures these behaviors. A learned or physically calibrated
abstention policy remains future work.

## Relevant implementation

- [`geometry/mask_cutline.py`](../src/strawberry_occlusion/geometry/mask_cutline.py)
- [`geometry/fixed_axis_cutline.py`](../src/strawberry_occlusion/geometry/fixed_axis_cutline.py)
- [`geometry/fixed_axis_search_cutline.py`](../src/strawberry_occlusion/geometry/fixed_axis_search_cutline.py)
- [`evaluation/fixed_axis_cutline.py`](../src/strawberry_occlusion/evaluation/fixed_axis_cutline.py)
- [`evaluation/fixed_axis_sensitivity.py`](../src/strawberry_occlusion/evaluation/fixed_axis_sensitivity.py)
