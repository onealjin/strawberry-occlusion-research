# M5-v1: Pre-Specified Selective Action Under Attachment Occlusion

M5-v1 is a frozen, inference-only analysis derived from the tracked M5-v0
`per_observation_signals.csv` contract. It asks how retained image-space
coordinate drift changes as one global inference-time risk threshold is swept
over a fixed requested-coverage grid. It does not train or calibrate a policy,
change v2b geometry, or establish a physical operating boundary.

The development-reuse analysis is complete and frozen. Its result is
descriptive, selection-biased, and non-confirmatory: the pre-specified
70%-requested geometry policy did not demonstrate improved retained action
stability. The full result and its claim boundary are recorded below without
changing the frozen analysis or promoting another coverage point.

## Frozen population and scores

The analysis targets pairwise-complete rows with:

- `method = v2b`;
- `region = attachment`;
- finite, internally consistent `c_clean`, `c_occ`, and `stability_abs`;
- both selected inference-time signals defined.

The tracked M5-v0 schema contains all three coordinate fields. M5-v1 validates
each targeted outcome by checking `stability_abs` against
`abs(c_occ - c_clean)` with relative and absolute tolerances of `1e-9`. It does
not join an M4 table or broaden the M5-v0 input contract.

The frozen risk scores are:

| Policy identity | Definition | Lower risk retained first |
|---|---|---|
| `geometry_risk` | `1 - v2b_selected_contact_evidence_fraction` | more selected contact evidence |
| `perception_risk` | `predictive_entropy_attachment` | lower attachment-localized entropy |

Risk scores use only current-observation signal information. Outcome values do
not enter threshold selection, tie handling, or common-group construction.

The mechanism-based scope was fixed before this analysis: M4 established
attachment occlusion as the action-critical region before M5 signal
exploration, and `v2b_selected_contact_evidence_fraction` is an internal v2b
action-support diagnostic. The M5-v1 target action is therefore v2b and its
frozen occlusion scope is attachment. This choice is not justified by selecting
whichever current-cohort v2a or v2b outcome looks more favorable.

## Primary group-first association

M5-v1 reports exactly two scientific association rows, in fixed order:
`geometry_risk` and `perception_risk`. For each score it first averages risk and
`stability_abs` arithmetically within each `group_id`, records the group's
targeted observation count, and then computes average-rank, tie-aware Spearman
rho across the group means. `group_id` is the independent unit; a pooled
perturbation-row coefficient is not reported as inferential evidence.

The observed association is undefined with fewer than three groups, constant
group-mean risk, or constant group-mean drift. No undefined result is replaced
with zero and no p-value is emitted.

The association interval resamples the pre-aggregated group-level risk/drift
pairs 10,000 times. Each score has an order-independent generator from
`SeedSequence([0, 101, score_code])`, where the score codes are 1 for geometry
and 2 for perception. Replicates with constant sampled risk or drift are
undefined and excluded from the percentile calculation. A 95% percentile
interval is emitted only when at least
`ceil(0.50 * association_bootstrap_replicates)` replicates are defined. The
`0.50` fraction is a deterministic degeneracy guard, not a significance or
power threshold; both the fraction and resolved replicate count are persisted.

For development-reuse, the geometry association is an integrity/re-expression
check rather than new empirical evidence. Algebraically,
`geometry_risk = 1 - evidence_fraction`; arithmetic group averaging commutes
with this affine transformation. On identical group rows its Spearman rho is
therefore exactly the negative of the raw evidence-fraction rho. The raw
fraction is not exposed as a third production association row.

For `cohort_role = independent_validation`, M5-v1 states only: "Cohort
independence was declared by the caller; image/content independence is not
verified by this module." The module does not audit image or content overlap.

## Global-threshold coverage rule

All policies are evaluated at requested coverages `1.00`, `0.90`, `0.80`,
`0.70`, `0.60`, and `0.50`. For `N` targeted observations and requested
coverage `q`:

```text
target_count = ceil(q * N)
risk_cutoff = target_count-th ascending risk order statistic
retain exactly those observations with risk <= risk_cutoff
```

The operational object is a single global risk threshold swept over requested
coverage, not an arbitrary choice of `K` rows. `target_count` determines the
threshold. Boundary ties are retained completely and are never split using an
outcome or random tie-break. Realized coverage is consequently at least
`ceil(q*N)/N` and can be higher. Requested coverage `1.00` always realizes
exactly `1.00`.

The primary outcome summary is the equal-weight mean across represented fruit
groups of each group's mean retained `stability_abs`. Every primary row also
reports the threshold, target and retained counts, realized coverage,
represented groups, maximum retained drift, and exact stored-zero drift count
and fraction. It additionally reports total observation/group denominators,
realized group coverage, and the observation-weighted retained mean without
replacing the group-balanced primary metric. Curve-bootstrap diagnostics retain
the median and interquartile realized observation coverage plus median
represented bootstrap-cluster count. A separate two-row risk summary reports
exact distinct-value counts, largest exact tie block and fraction, and finite
risk extrema for each frozen score.

## Bootstrap and headline contrast

Pointwise intervals and the headline interval use 10,000 deterministic
whole-`group_id` bootstrap replicates with seed `0`. Every replicate resamples
groups with replacement, reconstructs its observation table, and recomputes
each global threshold and tie-complete retained set before evaluating drift.
Intervals are 95% percentile intervals.

The association and selective-curve bootstraps intentionally use different
representations because their estimands differ:

- **Association bootstrap:** the statistic is defined on group-level means.
  Those means are deterministic after fixing the targeted observation table,
  so the bootstrap resamples pre-aggregated `(risk, drift)` group pairs.
- **Selective-curve cluster bootstrap:** selection occurs at the individual
  observation level and each replicate changes the empirical global risk
  distribution. Whole source groups are resampled, their observation rows are
  reconstructed, thresholds and tie blocks are recomputed, and duplicated
  source groups receive distinct bootstrap-copy cluster identities.

The curve bootstrap preserves its existing single stream from
`np.random.default_rng(0)`, with one whole-group position sample consumed per
replicate across the full curve analysis. Association uses separate per-score
`SeedSequence([0, 101, score_code])` streams. Both analyses use the same
scientific cohort; software-isolated Monte Carlo streams do not constitute
independent scientific evidence. Their different implementations are
intentional, not inconsistent statistical treatment.

There is exactly one headline selective-action contrast:

```text
headline_selective_improvement_px =
    geometry group-balanced mean drift at requested 1.00
    - geometry group-balanced mean drift at requested 0.70
```

A positive value means lower retained drift under the 0.70-requested geometry
policy; a negative value means higher retained drift. No p-value or other
coverage/score headline is reported. For `cohort_role = development_reuse`, the
contrast is explicitly labelled "descriptive, selection-biased development
reuse; non-confirmatory." Its interval measures group-resampling variability
within that development analysis and does not create independent validation.

## Descriptive random-withholding reference

For each score and requested coverage, M5-v1 observes the actual retained count
after ceil rounding and tie expansion. It then makes 10,000 uniform
without-replacement selections of exactly that count from the original targeted
observation table and computes the same primary metric. It reports the random
mean, 5th and 95th percentiles, and mean represented-group count.

This distribution is a descriptive random-withholding reference, not a
confidence interval. It is never nested inside the group bootstrap and has no
common-group variant. Random streams are order-independent. Each uses NumPy
`SeedSequence([base_seed, score_code, coverage_code])`, with base seed `0`, score
codes `1` and `2`, and `coverage_code = round(q * 10000)`.

## Common-group secondary curve

The common retention-stable set contains groups with at least one observation
retained at requested coverage `0.50` under both frozen policies. Its definition
uses scores and retained masks only.

The `0.50` reference is explicitly required to equal the minimum of the frozen
requested-coverage grid. The implementation fails loudly if a future grid adds
a lower coverage without an explicit common-group contract update.

For every curve point, M5-v1 first applies the primary policy to the complete
targeted table. It then restricts retained observations to the pre-defined
common set for evaluation. It never recomputes a threshold after that
restriction. The secondary metric is undefined when fewer than three common
groups exist. At requested coverage `1.00`, it must equal the unconditional
full-coverage metric recomputed over only the common groups.

## Frozen development-reuse result

### Cohort and pre-specified question

The frozen question was whether withholding higher-risk attachment observations
under a pre-specified geometry-support risk would reduce conditional action
drift. The completed analysis used:

| Field | Frozen value |
|---|---:|
| Cohort role | `development_reuse` |
| Method | `v2b` |
| Region | `attachment` |
| Targeted observations | 89 |
| Targeted groups | 17 |
| Independent unit | `group_id` |

The geometry policy used
`geometry_risk = 1 - v2b_selected_contact_evidence_fraction`. The
pre-specified perception comparator used
`perception_risk = predictive_entropy_attachment`. The one and only headline
operating point remained requested coverage `q = 0.70`.

### Primary headline and secondary aggregation check

The pre-specified primary metric was group-balanced mean absolute drift. Its
frozen headline was:

| Quantity | Frozen value (px) |
|---|---:|
| `D(1.00)` | 1.2861414285637016 |
| `D(0.70)` under the geometry policy | 1.8678841944248492 |
| `Delta_70 = D(1.00) - D(0.70)` | -0.5817427658611476 |
| 95% same-replicate group-cluster-bootstrap interval | [-1.0897819486262417, 0.788154289610585] |

The interval used 10,000 whole-group replicates and seed `0`. Positive
`Delta_70` denotes lower retained drift under the 0.70-requested geometry
policy; negative denotes higher retained drift.

At the pre-specified 70%-requested operating point, the frozen
geometry-support risk did not demonstrate improved action stability on the
development-reuse cohort. Group-balanced retained drift increased from
1.286 px at full coverage to 1.868 px, yielding `Delta_70 = -0.582 px`
(95% group-cluster-bootstrap interval [-1.090, 0.788]). The point estimate was
unfavorable, while the interval remained compatible with both worsening and
improvement. The interval crossing zero does not support a demonstrated-harm
claim.

As a secondary descriptive quantity, the observation-weighted mean absolute
drift was:

| Requested coverage | Observation-weighted mean (px) |
|---:|---:|
| 1.00 | 1.3914457907038222 |
| 0.70 | 1.4343298418444783 |
| Reference-minus-selective change | -0.0428840511406561 |

The rounded observation-weighted change was -0.043 px, compared with
-0.582 px for the group-balanced headline. The direction was unchanged, but
the substantially larger magnitude of the pre-specified group-balanced result
reflects equal weighting of fruit groups. The primary metric remains
group-balanced as pre-specified; this secondary check neither replaces nor
invalidates it.

### Association integrity check

| Score | Groups | Group-first Spearman rho | 95% group-bootstrap interval | Defined / undefined replicates |
|---|---:|---:|---:|---:|
| Geometry risk | 17 | 0.553857998636187 | [0.11455222155724581, 0.798470320351831] | 10,000 / 0 |
| Perception risk | 17 | -0.15767944304164477 | [-0.6888821574052458, 0.37450995045612806] | 10,000 / 0 |

On the development cohort, the frozen geometry risk showed a positive
group-first rank association with attachment-induced drift (`rho = 0.554`,
`n = 17`, 95% group-bootstrap interval [0.115, 0.798]). This is the exact
sign-reversed re-expression of the exploratory M5-v0 evidence-fraction
association on the same observations. It is reported as an integrity check,
not new empirical evidence or a confirmatory result. The percentile interval
does not change that claim boundary and is not a substitute for a p-value;
candidate selection followed exploratory screening on this cohort and did not
resolve multiplicity.

The pre-specified perception comparator showed little useful ranking
information on this development cohort (`rho = -0.158`, 95% interval
[-0.689, 0.375]), with the point estimate signed opposite to the intended risk
interpretation. These results do not establish that geometry generally
outperforms perception uncertainty.

### Risk distributions and finite-N coverage

| Score | Distinct values | Largest exact tie | Largest tie fraction | Observed minimum | Observed maximum |
|---|---:|---:|---:|---:|---:|
| Geometry risk | 46 | 33 | 0.3707865168539326 | 0 | 0.5 |
| Perception risk | 89 | 1 | 0.011235955056179775 | 0.0619243698248966 | 0.1677665960120653 |

Geometry risk was discrete and compressed. Its largest tie was the zero-risk
block: 33 of 89 observations (0.3707865168539326). Perception risk had no exact
ties in this cohort but occupied a comparatively narrow numerical range.
Neither observed range has a calibrated interpretation.

All frozen geometry coverage boundaries had zero tie expansion:

| Requested coverage | Target count | Retained | Realized coverage | Geometry cutoff |
|---:|---:|---:|---:|---:|
| 1.00 | 89 | 89 | 1.0000000000 | 0.5000000000 |
| 0.90 | 81 | 81 | 0.9101123596 | 0.4482758621 |
| 0.80 | 72 | 72 | 0.8089887640 | 0.3750000000 |
| 0.70 | 63 | 63 | 0.7078651685 | 0.3214285714 |
| 0.60 | 54 | 54 | 0.6067415730 | 0.2413793103 |
| 0.50 | 45 | 45 | 0.5056179775 | 0.1538461538 |

Requested-realized differences at these operating points came only from
`ceil(q*N)` rounding, not tie expansion. Future cohorts need not have equally
clean cutoff boundaries.

### Frozen selective-action curves

The full descriptive geometry curve was:

| Requested coverage | Retained observations | Represented groups | Group-balanced mean drift (px) | Observation-weighted mean drift (px) | Maximum retained drift (px) |
|---:|---:|---:|---:|---:|---:|
| 1.00 | 89 | 17 | 1.2861414285637016 | 1.3914457907038222 | 23.88326848249028 |
| 0.90 | 81 | 17 | 1.3634661283401508 | 1.3758868518807967 | 23.88326848249028 |
| 0.80 | 72 | 16 | 1.4589795502867657 | 1.3634146633096416 | 23.88326848249028 |
| 0.70 | 63 | 15 | 1.8678841944248492 | 1.4343298418444783 | 23.88326848249028 |
| 0.60 | 54 | 14 | 1.2321260405214589 | 0.7386695683821695 | 8.536082474226816 |
| 0.50 | 45 | 11 | 0.6345500513274266 | 0.5595598464086897 | 2.708453873352653 |

The geometry curve was non-monotonic. Maximum retained absolute drift remained
23.88 px through the pre-specified `q = 0.70` point, then fell to 8.54 px at
`q = 0.60` and 2.71 px at `q = 0.50`. Part of that reduction is whole-group
disappearance rather than within-group selection: fruit_003 retains no
observations at q = 0.60. The lower-coverage behavior remains part of the full
descriptive curve; it does not replace `q = 0.70` as the headline, identify a
preferred operating point, or establish a deployment threshold.

The perception comparator's group-balanced curve was:

| Requested coverage | Group-balanced mean drift (px) |
|---:|---:|
| 1.00 | 1.2861414285637016 |
| 0.90 | 1.3926826310057994 |
| 0.80 | 1.345590773516572 |
| 0.70 | 1.4595615430534699 |
| 0.60 | 1.723078327522598 |
| 0.50 | 1.7013844982806543 |

Every selective perception point was worse than full coverage, although the
perception curve itself was non-monotonic. Its maximum retained absolute drift
remained 23.88326848249028 px even at `q = 0.50`. This comparator result does
not establish general geometry superiority.

### Frozen descriptive random-withholding result

At the pre-specified `q = 0.70` point, the geometry-policy group-balanced mean
drift was 1.8678841944248492 px. The equal-size random-withholding reference had
a mean of 1.2936014772549305 px, a 5th percentile of 0.8976941804262282 px, a
95th percentile of 1.626638065506377 px, and an average of 16.7379 represented
groups. The geometry-policy mean lay above the descriptive reference's 95th
percentile. This is a descriptive comparison rather than an inferential test
and also reflects differing group representation: the geometry policy retained
15 groups, whereas random withholding retained approximately 16.7 groups on
average.

At `q = 0.50`, the geometry-policy mean was 0.6345500513274266 px. It lay below
the descriptive random 5th percentile of 0.754008458652847 px; the random mean
was 1.3206930855507186 px and its 95th percentile was 1.9563783226876501 px.
This lower-coverage point is reported only as part of the full curve and is not
a second headline.

### Outcome zero inflation

The outcome was strongly zero-inflated: 39 of 89 observations
(0.43820224719101125) had exactly zero drift. Five of 17 groups had zero drift
throughout: `fruit_005`, `fruit_007`, `fruit_009`, `fruit_010`, and
`fruit_011`. The cohort therefore has limited within-fruit dynamic range for an
observation-level selective policy in many groups.

### Common-retention sensitivity curve

The common-retention population contained six groups. Its frozen geometry
curve was:

| Requested coverage | Common-group balanced mean drift (px) |
|---:|---:|
| 1.00 | 0.6578351402331842 |
| 0.90 | 0.6300573624554064 |
| 0.80 | 0.6300573624554064 |
| 0.70 | 0.6300573624554064 |
| 0.60 | 0.6300573624554064 |
| 0.50 | 0.6183409965989711 |

The six-group common-retention sensitivity curve was comparatively flat, but
the common population was defined by survival under both policies at the most
aggressive 50%-requested coverage. This construction excludes difficult groups
such as `fruit_003` that retain no observations at that threshold. The common
curve is therefore a fixed-population sensitivity analysis among persistently
actionable groups, not a decomposition of the primary curve.

## POST-HOC / DESCRIPTIVE / NON-CONFIRMATORY characterizations

### Reproducibility disclosure

The all-group within-fruit Spearman characterization, the `fruit_003`
one-group-at-a-time counterfactual, and the `fruit_003` evidence-component
check were computed by ad-hoc read-only scripts against the frozen M5-v0
observation table during result interpretation. They are not part of the
tested M5-v1 pipeline and are not regenerable directly from the committed
M5-v1 analysis code. They therefore do not have the same software-audit status
as the frozen pipeline outputs. No new code is added for them here.
This documentation task transcribes the reviewed frozen values without
recomputing or extending them.

### Post-hoc within-fruit characterization

No within-fruit analysis was pre-specified. Across all 17 groups, 10 yielded a
defined within-fruit Spearman coefficient and seven were undefined. Among the
10 defined coefficients, four were positive, five negative, and one exactly
zero. The median was -0.166974, the mean was approximately -0.0676, and the
range was [-0.753702, +0.979796].

No consistent positive within-fruit monotone relationship was evident in this
post-hoc characterization. Seven groups were undefined for at least one reason
involving insufficient observations, constant risk, or constant outcome.
Because the diagnostic applies its guards sequentially, the reported undefined
reason is the first condition encountered rather than a mutually exclusive
partition. Most defined coefficients used only five or six observations.
These results are descriptive and do not establish a population-level
within-fruit effect or a systematic negative relationship.

### Post-hoc `fruit_003` mechanism example

`fruit_003` was an influential example, not evidence of a general mechanism.
It had six observations, mean geometry risk 0.34948693282026616, mean drift
approximately 9.6642601807 px, maximum drift 23.88326848249028 px, and an
exactly zero within-fruit Spearman coefficient. Its group-mean drift contributed
approximately `9.6643 / 21.8644 = 44.2%` of the sum of the 17 group-mean drifts
used by the full-coverage group-balanced statistic. This is not 44.2% of total
observation drift.

The relevant within-fruit rows were:

| Severity | Seed | Drift (px) | Geometry risk |
|---:|---:|---:|---:|
| 0.01 | 0 | 0 | 0.259259 |
| 0.01 | 1 | 0 | 0.333333 |
| 0.02 | 0 | 5.382716 | 0.428571 |
| 0.02 | 1 | 5.093117 | 0.500000 |
| 0.04 | 0 | 23.883268 | 0.303030 |
| 0.04 | 1 | 23.626459 | 0.272727 |

At the frozen `q = 0.70` cutoff of 0.3214285714, `fruit_003` retained the
zero-drift observation at risk 0.259259 and the two approximately 23.63-23.88
px observations. It rejected the zero-drift observation at risk 0.333333 and
the approximately 5.38 px and 5.09 px observations. Its retained mean rose from
approximately 9.664260 px at full coverage to approximately 15.836576 px at
`q = 0.70`.

The zero within-fruit coefficient was mid-distribution among the 10 defined
groups; five other groups had more strongly negative coefficients. Its
influence came from failure magnitude, not from a uniquely extreme rank
coefficient. The transferable observation is narrower: a score may show useful
aggregate or between-group association while still failing on individual
observations for which action error is most consequential.

### Post-hoc one-group-at-a-time counterfactual

This counterfactual replaced only `fruit_003`'s full-coverage group mean with
its `q = 0.70` retained mean while holding all other 16 full-coverage group
means and the 17-group denominator fixed. The full group-balanced mean was
1.2861414 px; the counterfactual mean was approximately 1.649219 px, an
increase of approximately 0.3631 px. The actual `q = 0.70` increase was
approximately 0.5817 px, so the one-group substitution accounted for
approximately 62.4% of the observed adverse change. The remainder reflects
residual policy and group-representation effects; it is not a formal causal
decomposition or a measure of pure group composition.

### Post-hoc evidence-component check

The frozen component values for `fruit_003` were:

| Severity | Seed | Selected evidence | Maximum evidence | Evidence fraction | Geometry risk |
|---:|---:|---:|---:|---:|---:|
| 0.01 | 0 | 20 | 27 | 20/27 = 0.740741 | 0.259259 |
| 0.01 | 1 | 18 | 27 | 18/27 = 0.666667 | 0.333333 |
| 0.02 | 0 | 20 | 35 | 20/35 = 0.571429 | 0.428571 |
| 0.02 | 1 | 17 | 34 | 17/34 = 0.500000 | 0.500000 |
| 0.04 | 0 | 23 | 33 | 23/33 = 0.696970 | 0.303030 |
| 0.04 | 1 | 24 | 33 | 24/33 = 0.727273 | 0.272727 |

| Severity | Mean drift (px) | Mean selected evidence | Mean maximum evidence | Mean geometry risk |
|---:|---:|---:|---:|---:|
| 0.01 | 0 | 19.0 | 27.0 | approximately 0.2963 |
| 0.02 | approximately 5.238 | 18.5 | 34.5 | approximately 0.4643 |
| 0.04 | approximately 23.755 | 23.5 | 33.0 | approximately 0.2879 |

A post-hoc hypothesis that normalized support concealed a collapse in absolute
contact evidence was directly checked and was not supported for `fruit_003`.
From severity 0.02 to 0.04, maximum available local contact evidence changed
only from approximately 34.5 to 33 pixels on average, while selected local
contact evidence increased from 18.5 to 23.5 pixels. Geometry risk therefore
decreased despite drift increasing from approximately 5.24 to 23.75 px.

One possible explanation is that strong local evidence can coexist with a
severely displaced downstream action. A well-supported-but-wrong geometric
structure remains a hypothesis-generating possibility only; it was not tested,
and these results do not identify a causal segmentation mechanism or establish
that an absolute-evidence measure would solve the problem.

## Synthetic-perturbation interpretation

The controlled benchmark establishes that perturbing the attachment region
disproportionately destabilizes the downstream cut coordinate. It does not
isolate the mechanism as removal of attachment evidence. The synthetic
perturbation may remove evidence, add competing or spurious segmentation
structure, displace boundaries, produce another segmentation response, or
combine several pathways. M4 and M5 do not distinguish which pathway generated
the drift. Independent real-occlusion or higher-fidelity synthetic validation
is therefore warranted later, without modifying the frozen M4 or M5 analyses.

## Frozen limitations

1. This is development-reuse analysis, not independent validation.
2. The cohort contains only 17 targeted groups and 89 targeted
   observations.
3. The geometry candidate emerged from exploratory M5-v0 analysis on this
   cohort.
4. The geometry association is an integrity/re-expression result, not
   confirmatory evidence.
5. Its association interval excluding zero does not license a confirmatory
   inferential claim.
6. The headline interval is wide and crosses zero.
7. Only `q = 0.70` was pre-specified as the headline operating point.
8. Geometry risk is discrete and compressed: 46 distinct values, 33 of 89
   observations at exact zero risk, and observed range [0, 0.5].
9. Perception risk has 89 distinct values, largest tie count 1, and observed
   range approximately [0.0619, 0.1678]; this does not give the score a
   calibrated interpretation.
10. The outcome is strongly zero-inflated: 39 of 89 exact-zero-drift
    observations and five of 17 groups with zero drift throughout.
11. Global observation thresholds also alter group representation.
12. The descriptive random-withholding comparison therefore reflects both
    ranking behavior and group-representation differences.
13. The secondary observation-weighted direction check has the same
    unfavorable sign but a much smaller magnitude than the group-balanced
    primary result, showing that aggregation materially affects reported
    magnitude.
14. The common-group analysis is conditioned on survival under both policies
    at `q = 0.50`.
15. The within-fruit characterizations are post-hoc.
16. Most defined within-fruit coefficients use only five or six observations.
17. Seven groups cannot yield a defined within-fruit coefficient under the
    descriptive diagnostic.
18. `fruit_003` is disproportionately influential in the group-balanced
    statistic.
19. The one-group counterfactual is post-hoc and is not a formal causal
    decomposition.
20. The denominator-collapse hypothesis was tested and not supported for
    `fruit_003`.
21. Synthetic attachment perturbations do not isolate the causal segmentation
    mechanism.
22. The post-hoc characterizations came from ad-hoc read-only scripts rather
    than the committed M5-v1 pipeline and have a reproducibility and
    software-audit limitation.
23. Favorable behavior at lower coverage is descriptive and does not replace
    `q = 0.70` as the headline.

## Frozen scientific takeaway

Aggregate or group-level association is not sufficient evidence of useful
observation-level selective action. A risk score can retain useful aggregate
association while failing on individual observations for which action error is
especially consequential.

M5-v1 is a clean, pre-specified development-reuse negative result with
mechanistic post-hoc characterization. It does not establish that geometry is
useless, that perception uncertainty is generally inferior, that abstention
works, or that the system is ready for physical operation.

## Next milestone boundary

M5-v1 motivates a prospectively specified next milestone focused on whether
action-readiness representations can improve over a single global
observation-level scalar risk. The specific M6 hypothesis and signals must be
frozen separately before evaluation and must not be selected by further
screening of this development cohort.

## Claim boundary and outputs

No AURC, integrated drift, or other scalar curve comparison is reported.
Integration over the requested grid would be mathematically definable, but a
single scalar could invite an unsupported confirmatory comparison favoring
geometry on the development cohort that generated the geometry hypothesis.
M5-v1 instead keeps the full curves, the single pre-specified geometry
contrast, and the descriptive random reference.

The output directory contains:

- `manifest.json`, `configuration.json`, `input_provenance.json`, and
  `summary.json`;
- `group_primary_summary.csv`, `association_summary.csv`, and
  `risk_distribution_summary.csv`;
- `selective_action_curve.csv`;
- `headline_selective_contrast.json`;
- `primary_geometry_risk_vs_drift.png`, `selective_action_curve.png`, and the
  common-group figure under `plots/`.

Every figure states method = v2b, region = attachment, the total targeted group
count, and cohort role. The common-group figure additionally states the common
retention group count. Development-reuse figures visibly carry DESCRIPTIVE —
DEVELOPMENT REUSE — NON-CONFIRMATORY. The primary selective-action x-axis is
realized observation coverage and explicitly notes that policies are indexed
by the fixed requested-coverage grid.

Under development-reuse, the association and curve have different claim
boundaries. The geometry association is only an arithmetic sign-reexpression /
integrity check on the already-selected signal. The selective-action curve is a
genuinely new operational descriptive quantity, but remains selection-biased
because this cohort contributed to selecting the geometry signal and is
therefore non-confirmatory. Selective figures visibly note that the entropy
comparator is shown for context and geometry was selected on the development
cohort.

M5-v1 does not copy source observation rows into its output. Public tests use
only synthetic tables and temporary artifacts; no frozen development-cohort
coefficients or expected comparator behavior are encoded in source or tests.
This document records the approved aggregate development-reuse result without
copying generated result artifacts into the tracked repository.

## CLI

Run against a separately stored, explicitly approved M5-v0 result:

```powershell
python -m strawberry_occlusion.evaluation.selective_action `
  --profile m5_v1 `
  --m5-v0-root <approved-m5-v0-output> `
  --output-root data/evaluation_runs/m5_v1_development `
  --cohort-role development_reuse
```

The input and output roots must be separate. Existing output is protected unless
`--overwrite` is supplied. The CLI exposes no coverage, score, bootstrap,
random-reference, geometry, or threshold tuning options.
