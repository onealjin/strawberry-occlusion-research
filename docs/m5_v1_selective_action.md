# M5-v1: Descriptive Selective Action

M5-v1 is a frozen, inference-only analysis derived from the tracked M5-v0
`per_observation_signals.csv` contract. It asks how retained image-space
coordinate drift changes as one global inference-time risk threshold is swept
over a fixed requested-coverage grid. It does not train or calibrate a policy,
change v2b geometry, or establish a physical safety threshold.

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

For development reuse, the geometry association is an integrity/re-expression
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

## Descriptive random reference

For each score and requested coverage, M5-v1 observes the actual retained count
after ceil rounding and tie expansion. It then makes 10,000 uniform
without-replacement selections of exactly that count from the original targeted
observation table and computes the same primary metric. It reports the random
mean, 5th and 95th percentiles, and mean represented-group count.

This distribution is a descriptive random-policy reference, not a confidence
interval. It is never nested inside the group bootstrap and has no common-group
variant. Random streams are order-independent. Each uses NumPy
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

## Claim boundary and outputs

No AURC, integrated drift, or other scalar curve comparison is reported.
Integration over the requested grid would be mathematically definable, but a
single scalar could invite a confirmatory "geometry beats entropy" claim from
the development cohort that generated the geometry hypothesis. M5-v1 instead
keeps the full curves, the single pre-specified geometry contrast, and the
descriptive random reference.

The output directory contains:

- `manifest.json`, `configuration.json`, `input_provenance.json`, and
  `summary.json`;
- `group_primary_summary.csv`, `association_summary.csv`, and
  `risk_distribution_summary.csv`;
- `selective_action_curve.csv`;
- `headline_selective_contrast.json`;
- `primary_geometry_risk_vs_drift.png`, `selective_action_curve.png`, and the
  common-group figure under `plots/`.

Every figure states method = v2b, region = attachment, the total targeted group count, and cohort role. The common-group figure additionally states the common retention group count. Development-reuse figures visibly carry DESCRIPTIVE — DEVELOPMENT REUSE — NON-CONFIRMATORY. The primary selective-action x-axis is realized observation coverage and explicitly notes that policies are indexed by the fixed requested-coverage grid.

Under development reuse, the association and curve have different claim
boundaries. The geometry association is only an arithmetic sign-reexpression /
integrity check on the already-selected signal. The selective-action curve is a
genuinely new operational descriptive quantity, but remains selection-biased
because this cohort contributed to selecting the geometry signal and is
therefore non-confirmatory. Selective figures visibly note that the entropy
comparator is shown for context and geometry was selected on the development
cohort.

M5-v1 does not copy source observation rows into its output. Public tests use
only synthetic tables and temporary artifacts; no real-cohort coefficients or
expected comparator behavior are encoded in source, tests, or documentation.

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
