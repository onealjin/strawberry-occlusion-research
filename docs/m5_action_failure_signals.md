# M5: Action-Aware Failure Signals Under Occlusion

M5-v0 is a small, inference-only development analysis asking whether simple
perception-derived and geometry-derived signals rank occlusion-induced
cut-coordinate instability before an action is committed. It does not train a
model, change an estimator, or define a calibrated deployment policy.

## Frozen research question and hypotheses

**Research question:** Can perception-derived and geometry-derived confidence
signals identify occlusion-induced cut-coordinate instability before an action
is committed?

- **H1:** Attachment-localized predictive uncertainty will be more associated
  with downstream cut-coordinate instability than image-wide/global predictive
  uncertainty.
- **H2:** Geometry-support signals will contain information about action
  instability beyond conventional segmentation uncertainty.
- **H3:** Some substantial-drift cases may retain relatively strong
  conventional segmentation confidence.

H3 is deliberately empirical. M5-v0 does not begin from or support a claim that
softmax confidence necessarily fails.

## Eligible cohort and denominator

M5-v0 inherits the canonical matched, fixed-axis-eligible M4 development cohort
and does not redefine M4 eligibility, placement, severity, seed, checkpoint, or
geometry. At the current frozen reference state and time of writing, the M4
development manifest contains 18 fruit groups. This is recorded cohort state,
not a permanent code constant or a perturbation denominator.

Actual perturbation counts depend on fixed-axis eligibility and on whether the
exact synthetic template has a valid matched placement in all four regions.
Incomplete matched sets remain absent rather than being imputed. The M5 runner
reruns the frozen M4 workflow and uses its canonical completed perturbation rows
to determine the represented M5 cohort. Reports must state both the number of
development-manifest groups and the number represented in the analyzed rows;
they must not imply a naive `18 x placement x severity x seed` denominator.

## Frozen outcome and non-leakage boundary

The primary outcome is

```text
abs(c_occ - c_clean)
```

where both coordinates use the same frozen segmentation checkpoint and frozen
geometry method. The clean coordinate is used only to construct this evaluation
target. A row has an undefined primary outcome when either required coordinate
is not a successful finite estimate.

Every primary signal is constructed before the outcome is attached and uses
only the current occluded observation's softmax probabilities and geometry from
its current predicted mask. Ground-truth masks, manual-mask geometry, clean
counterparts, and future observations are prohibited signal inputs. The older
M4 field `mean_normalized_entropy_attachment_roi` uses the manual-mask placement
ROI and is therefore not an eligible primary M5 signal.

Manual-mask-derived geometry may be added later only as an explicitly labelled
secondary diagnostic. It is not implemented in M5-v0 and cannot replace the
primary M4 stability outcome.

## Perception signals

For the three-class softmax probabilities `p_c(x)`, pixel entropy is measured in
natural-log units:

```text
H(x) = -sum_c p_c(x) log(p_c(x))
```

The frozen global aggregation region is the entire checkpoint-preprocessed
image. The frozen attachment region is the current prediction's v2a selected
Flesh/Calyx contact mask after a five-pixel square (Chebyshev) dilation. It is
recomputed independently for each current observation. It is not the manual M4
placement region and does not use the clean prediction.

The frozen perception signals are:

- mean global predictive entropy;
- mean global maximum softmax probability;
- mean global top-one minus top-two probability margin;
- mean predicted-attachment predictive entropy;
- mean predicted-attachment maximum softmax probability;
- mean predicted-attachment top-one minus top-two probability margin.

If the predicted contact mask and its dilation are empty, the ROI has zero
pixels, `attachment_roi_defined` is false, and all three localized signals are
stored as missing. The implementation never substitutes the full image or a
manual attachment location.

## Geometry-support signals

M5-v0 reuses current v2b result fields and deterministic ratios of those fields.
It does not change the candidate curve, feasibility thresholds, selected block,
or selected coordinate. The implemented signals are:

- searched and feasible candidate counts, plus feasible candidate fraction;
- feasible block count;
- selected block width, candidate count relative to all feasible candidates,
  and singleton indicator;
- maximum local contact evidence, selected local contact count, and their ratio;
- selected local Flesh, Calyx, and v2a-support counts;
- total v2a-support pixel count exposed by the current v2b result.

These geometry signals are associated only with the v2b action outcome. Several
raw counts depend on image-scale support and are descriptive evidence-support
signals, not probabilities of success. A distance or gap to an alternative
feasible block is not a canonical v2b result field and is unavailable for
M5-v0; it is not reconstructed or used.

## Aggregation and association

Fruit group (`group_id`) is the independent statistical unit. Images, regions,
severities, seeds, and perturbation rows are repeated measurements. M5-v0
reports analyses for all placements combined and separately for each canonical
M4 placement.

For each signal, method, region scope, and group, the signal and absolute drift
are averaged over pairwise-complete repeated observations. Spearman rank
association is then computed across those group means. A coefficient is left
undefined when fewer than three pairwise-complete groups remain or when either
group-level variable is constant. No perturbation-row p-value is reported.
The emitted association coefficients are descriptive. M5-v0 applies no
multiplicity correction, and the largest absolute correlation must not be
selected and interpreted as a confirmatory result.

Descriptive signal summaries also aggregate observation values within group
before summarizing groups with equal weight. Every table records group counts,
observation counts, and missing signal/outcome counts. M5-v0 does not add a
bootstrap interval; any future interval must resample whole `group_id` values,
never individual perturbations.

`signal_descriptive_statistics.csv` summarizes all observations where a signal
is defined, whereas `per_group_signal_summary.csv` uses observations where both
the signal and primary drift outcome are defined. These tables intentionally
summarize different row sets, so their reported means should not be expected to
be numerically identical.

Structured geometry failures remain in the observation output with their
available inference-time signals. They have a missing coordinate-drift outcome
and are excluded only from pairwise-complete drift association. Missing signals
or perturbations are not imputed.

## Frozen configuration and outputs

The runner fixes:

| Component | Setting |
|---|---|
| M5 profile | `m5_v0` |
| Source benchmark | frozen `m4_v1` |
| Placements | `attachment`, `calyx_tip`, `flesh_far`, `background` |
| Severities | 0.01, 0.02, 0.04 |
| Seeds | 0, 1 |
| Geometry | frozen M4 v2a/v2b `e050_w064` configuration |
| Attachment ROI dilation | five-pixel square dilation |
| Independent unit | `group_id` |
| Association | Spearman correlation of group means |
| Training or tuning | none |

The output directory contains:

- `manifest.json` and `configuration.json`;
- `per_observation_signals.csv`;
- `per_group_signal_summary.csv`;
- `signal_descriptive_statistics.csv`;
- `association_table.csv` and `summary.json`;
- three deterministic group-level diagnostic figures;
- a nested `m4_reference/` output proving the exact frozen source cohort and
  run configuration.

The figures show group means, not independent perturbation observations. The
selective-action staircase remains outside the frozen M5-v0 scope and is
implemented separately in
[M5-v1: Descriptive Selective Action](m5_v1_selective_action.md), without implying
an act/abstain calibration.

The M5 observation CSV uses round-trip-safe float precision. Summarization
treats that inference output as read-only and regenerates only derived tables,
JSON, and figures. This M5-only precision handling preserves deterministic
run-to-summarize regeneration without changing the shared M4 CSV serializer or
any frozen M4 output format.

## Claim boundary

M5-v0 may describe preliminary development-cohort rank associations, evidence
support, action instability, formally valid coordinates, and candidate future
selective actions such as act, abstain, or acquire another observation.

It must not claim calibrated confidence, calibrated failure probability,
production safety, deployment readiness, that softmax necessarily fails, that
occlusion is solved, statistical independence of perturbation rows, or
generalization beyond the current development cohort. No threshold is a
physical or safety threshold.

## CLI

Run only against an explicitly approved development manifest, data root, and
canonical checkpoint:

```powershell
python -m strawberry_occlusion.evaluation.action_failure_signals run `
  --profile m5_v0 `
  --development-manifest <approved-development-manifest.csv> `
  --safe-data-root <approved-data-root> `
  --checkpoint <canonical-epoch92-checkpoint.pt> `
  --output-root data/evaluation_runs/m5_v0_development `
  --device cuda
```

The command refuses to replace an existing output unless `--overwrite` is
explicitly supplied. It exposes no geometry, placement, severity, seed, or ROI
tuning flags.
