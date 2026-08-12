# M4 matched-occlusion benchmark

M4 is an inference-only benchmark for measuring how the location of partial
occlusion affects semantic predictions, image-space cut-coordinate stability,
and structured failure behavior. It uses controlled synthetic occlusion for
evaluation; it is not a completed synthetic training-augmentation method.

## Frozen profile

The tracked `m4_v1` profile fixes the following design:

| Component | Frozen setting |
|---|---|
| Regions | `attachment`, `calyx_tip`, `flesh_far`, `background` |
| Severity fractions | 0.01, 0.02, 0.04 |
| Seeds | 0, 1 |
| Primary comparison | attachment versus background |
| Independent analysis unit | `group_id` / fruit group |
| Geometry | frozen v2a and v2b `e050_w064` configuration |
| Training during benchmark | none |

Severity is the exact opaque template area divided by clean visible
Flesh-plus-Calyx foreground area after checkpoint preprocessing.

## Matched intervention

For each eligible sample, severity, and seed, the benchmark generates one
deterministic opaque, tapered, leaf-like RGB template. The same discrete
template is translated between the four regions. Its pixel area, appearance,
opacity, severity, seed, and identity remain fixed; location is the intended
change.

Placements are derived from the clean manual-mask geometry:

- **attachment:** the clean contact region;
- **calyx tip:** distal selected Calyx support separated from the attachment;
- **flesh far:** selected Flesh support separated inward from the attachment;
- **background:** background separated from visible foreground.

A matched quartet is included only when the exact template can be placed in all
four regions without clipping and while satisfying the region constraints.
Placement attrition is recorded rather than imputed.

![Procedural matched four-region example](../assets/readme/matched_occlusion_regions.svg)

*Procedurally generated benchmark example—not a real fruit image. This tracked
figure is generated from the repository's matched-placement code using toy
semantic geometry.*

## Outcomes

For each method, the benchmark records:

- coordinate stability, `|c_occ - c_clean|`;
- agreement with the manual-mask-derived reference coordinate;
- excess reference error relative to the clean prediction;
- structured geometry failure and finite-coordinate rate;
- descriptive silent-drift indicators above 2, 5, and 10 pixels;
- visible-region segmentation metrics with opaque occluder pixels excluded;
- uncertainty and method-specific geometry diagnostics.

The pixel thresholds are descriptive analysis thresholds. They are not
calibrated physical tolerances or millimetre measurements.

## Analysis unit and conditioning

Fruit group (`group_id`) is the independent analysis unit. Repeated images,
seeds, severities, regions, and estimator rows are not treated as independent
fruit observations. The implementation first computes means within group and
then summarizes across groups.

Conditioning differs by outcome:

- mean stability uses successful finite coordinate estimates;
- structured-failure rates use all eligible benchmark perturbations in the
  relevant group/condition;
- silent-drift rates also use that perturbation denominator, with a row counted
  as silent drift only when the method returned a finite successful coordinate
  and its drift exceeded the descriptive threshold.

Mean stability, silent drift, and structured failure therefore need not share
the same effective conditioning. A public result table must state the relevant
`n_groups`, population, and completeness requirement for every statistic.

## Severity integration

The pre-specified primary summary is the **severity-integrated attachment −
background stability difference**. Within each group and region, the outcome is
aggregated at each of the three severities. Severity is then rescaled from 0 at
0.01 to 1 at 0.04, and normalized trapezoidal integration is applied. The
attachment integral minus the background integral is formed within group before
group-level summary and bootstrap.

Internal `*_auc` fields denote normalized trapezoidal integration of the outcome
over the three severity levels after severity is rescaled to `[0,1]`. This is
not a ROC AUC. Internal serialized field names are retained for benchmark-schema
compatibility.

Groups without complete coverage across all three required severity levels do
not contribute to severity-integrated comparisons. Missing perturbations are
not imputed.

## Descriptive group bootstrap

The implementation resamples independent `group_id` values with replacement,
using 2,000 deterministic replicates and percentile bounds. These are labelled
descriptive group-bootstrap intervals. The output records `n_groups` and marks
whether the implementation considers the interval stable; the current minimum
for that marker is five groups. No row-level hypothesis test is substituted for
group-level inference.

## Provenance and publication boundary

The benchmark code accepts only manifest-listed development samples with
reviewed annotation status. It rejects manifest roles named `test`,
`final_test`, or `locked_test`, and rejects paths containing final/locked-test
components. This documents an implementation guardrail; it does not by itself
establish the historical provenance of every external experiment.

No numerical M4 findings are published in this document because the tracked
public repository currently contains no approved canonical aggregate-result
summary. Any future public result document should contain aggregate values only,
distinguish the full benchmark population from complete-coverage analysis
subsets, and report the relevant denominator beside every interval or headline
statistic.

## Relevant implementation

- [`augmentation/synthetic_occluder.py`](../src/strawberry_occlusion/augmentation/synthetic_occluder.py)
- [`evaluation/occlusion_robustness.py`](../src/strawberry_occlusion/evaluation/occlusion_robustness.py)
- [`visualization/occlusion_robustness.py`](../src/strawberry_occlusion/visualization/occlusion_robustness.py)
