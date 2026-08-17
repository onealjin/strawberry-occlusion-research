"""Frozen M5-v1 selective-action analysis derived from M5-v0 signals.

The deployed conceptual policy in this module is a single global risk threshold
swept over a fixed requested-coverage grid.  Thresholds use inference-time risk
scores only.  Absolute coordinate drift is attached solely for evaluation.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from strawberry_occlusion.evaluation.action_failure_signals import (
    OBSERVATION_FIELDS as M5_V0_OBSERVATION_FIELDS,
)
from strawberry_occlusion.evaluation.action_failure_signals import (
    PROFILE_NAME as M5_V0_PROFILE_NAME,
)
from strawberry_occlusion.evaluation.action_failure_signals import (
    _spearman_from_group_rows,
)
from strawberry_occlusion.evaluation.occlusion_robustness import (
    _assert_no_absolute_paths,
    _install_staged_directory,
    _validate_output_destination,
    _write_json,
)


PathLike = str | Path
PROFILE_NAME = "m5_v1"
METHOD = "v2b"
REGION = "attachment"
COHORT_ROLES = ("development_reuse", "independent_validation")
REQUESTED_COVERAGES = (1.00, 0.90, 0.80, 0.70, 0.60, 0.50)
COMMON_GROUP_REFERENCE_COVERAGE = 0.50
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 0
RANDOM_REFERENCE_DRAWS = 10_000
RANDOM_REFERENCE_BASE_SEED = 0
ASSOCIATION_BOOTSTRAP_REPLICATES = 10_000
ASSOCIATION_BOOTSTRAP_SEED = 0
ASSOCIATION_ANALYSIS_CODE = 101
ASSOCIATION_CONFIDENCE_LEVEL = 0.95
MINIMUM_DEFINED_BOOTSTRAP_FRACTION = 0.50
MINIMUM_COMMON_GROUPS = 3
HEADLINE_SCORE = "geometry_risk"
HEADLINE_REFERENCE_COVERAGE = 1.00
HEADLINE_SELECTIVE_COVERAGE = 0.70
DEVELOPMENT_REUSE_INTERPRETATION = (
    "descriptive, selection-biased development reuse; non-confirmatory"
)


@dataclass(frozen=True)
class RiskScoreDefinition:
    """Frozen source and direction for one M5-v1 policy score."""

    name: str
    source_signal: str
    score_code: int
    transformation: str
    higher_value_interpretation: str


SCORE_DEFINITIONS = (
    RiskScoreDefinition(
        name="geometry_risk",
        source_signal="v2b_selected_contact_evidence_fraction",
        score_code=1,
        transformation="1 - source_signal",
        higher_value_interpretation="less selected contact evidence",
    ),
    RiskScoreDefinition(
        name="perception_risk",
        source_signal="predictive_entropy_attachment",
        score_code=2,
        transformation="identity",
        higher_value_interpretation="more attachment-localized predictive entropy",
    ),
)
SCORE_NAMES = tuple(definition.name for definition in SCORE_DEFINITIONS)
_SCORE_BY_NAME = {definition.name: definition for definition in SCORE_DEFINITIONS}

SOURCE_COORDINATE_FIELDS = ("c_clean", "c_occ", "stability_abs")
REQUIRED_SOURCE_FIELDS = (
    "group_id",
    "method",
    "region",
    *SOURCE_COORDINATE_FIELDS,
    *(definition.source_signal for definition in SCORE_DEFINITIONS),
)

CURVE_FIELDS = (
    "method",
    "region",
    "cohort_role",
    "score",
    "source_signal",
    "requested_coverage",
    "target_count",
    "risk_cutoff",
    "total_observation_count",
    "total_group_count",
    "retained_observation_count",
    "realized_observation_coverage",
    "represented_group_count",
    "realized_group_coverage",
    "group_balanced_mean_abs_drift",
    "observation_weighted_mean_abs_drift",
    "group_cluster_bootstrap_ci_low",
    "group_cluster_bootstrap_ci_high",
    "bootstrap_realized_coverage_median",
    "bootstrap_realized_coverage_q25",
    "bootstrap_realized_coverage_q75",
    "bootstrap_represented_groups_median",
    "maximum_retained_abs_drift",
    "retained_zero_drift_count",
    "retained_zero_drift_fraction",
    "random_reference_mean",
    "random_reference_p05",
    "random_reference_p95",
    "random_reference_represented_groups_mean",
    "common_retention_group_count",
    "common_group_balanced_mean_abs_drift",
)

GROUP_PRIMARY_FIELDS = (
    "score_name",
    "source_signal",
    "group_id",
    "observation_count",
    "group_mean_risk",
    "group_mean_stability_abs",
    "method",
    "region",
    "cohort_role",
)

ASSOCIATION_FIELDS = (
    "score_name",
    "source_signal",
    "n_groups",
    "spearman_rho",
    "association_bootstrap_replicates",
    "association_bootstrap_seed",
    "minimum_defined_bootstrap_fraction",
    "minimum_defined_bootstrap_replicates",
    "bootstrap_defined_replicates",
    "bootstrap_undefined_replicates",
    "ci_lower",
    "ci_upper",
    "undefined_reason",
    "ci_undefined_reason",
    "cohort_role",
    "interpretation",
)

RISK_DISTRIBUTION_FIELDS = (
    "score_name",
    "source_signal",
    "n_distinct_risk_values",
    "largest_tie_count",
    "largest_tie_fraction",
    "minimum_risk",
    "maximum_risk",
)


@dataclass(frozen=True)
class TargetedObservationTable:
    """Pairwise-complete targeted M5-v1 observations."""

    group_ids: np.ndarray
    stability_abs: np.ndarray
    scores: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        group_ids = np.asarray(self.group_ids, dtype=object)
        stability = np.asarray(self.stability_abs, dtype=np.float64)
        if group_ids.ndim != 1 or stability.ndim != 1:
            raise ValueError("targeted table arrays must be one-dimensional")
        if len(group_ids) == 0 or len(group_ids) != len(stability):
            raise ValueError(
                "targeted table must contain equally sized non-empty arrays"
            )
        if any(str(value) == "" for value in group_ids):
            raise ValueError("targeted group_id values must be non-empty")
        if not np.all(np.isfinite(stability)) or np.any(stability < 0.0):
            raise ValueError("stability_abs must contain finite non-negative values")
        normalized_scores: dict[str, np.ndarray] = {}
        if set(self.scores) != set(SCORE_NAMES):
            raise ValueError(f"scores must contain exactly {SCORE_NAMES!r}")
        for name in SCORE_NAMES:
            values = np.asarray(self.scores[name], dtype=np.float64)
            if values.ndim != 1 or len(values) != len(stability):
                raise ValueError(f"{name} must match the targeted table length")
            if not np.all(np.isfinite(values)):
                raise ValueError(f"{name} must contain only finite values")
            values.setflags(write=False)
            normalized_scores[name] = values
        group_ids.setflags(write=False)
        stability.setflags(write=False)
        object.__setattr__(self, "group_ids", group_ids)
        object.__setattr__(self, "stability_abs", stability)
        object.__setattr__(self, "scores", normalized_scores)

    def take(self, indices: np.ndarray) -> TargetedObservationTable:
        """Return an indexed table, preserving all frozen scores."""

        return TargetedObservationTable(
            group_ids=self.group_ids[indices],
            stability_abs=self.stability_abs[indices],
            scores={name: self.scores[name][indices] for name in SCORE_NAMES},
        )

    @property
    def observation_count(self) -> int:
        return len(self.stability_abs)


@dataclass(frozen=True)
class RiskSelection:
    """Tie-complete retained set produced by one global risk threshold."""

    requested_coverage: float
    target_count: int
    risk_cutoff: float
    retained_mask: np.ndarray
    retained_count: int
    realized_coverage: float


@dataclass(frozen=True)
class BootstrapResult:
    """Pointwise primary-curve intervals and the frozen headline interval."""

    curve_intervals: Mapping[tuple[str, float], tuple[float, float]]
    headline_values: np.ndarray
    headline_ci_low: float
    headline_ci_high: float
    _curve_values: Mapping[tuple[str, float], np.ndarray]
    curve_diagnostics: Mapping[tuple[str, float], Mapping[str, float]]
    _sampled_group_positions: tuple[np.ndarray, ...]


@dataclass(frozen=True)
class AssociationBootstrapResult:
    """Observed group-first association and deterministic bootstrap accounting."""

    spearman_rho: float | None
    bootstrap_defined_replicates: int
    bootstrap_undefined_replicates: int
    ci_lower: float | None
    ci_upper: float | None
    undefined_reason: str | None
    ci_undefined_reason: str | None
    minimum_defined_bootstrap_fraction: float
    minimum_defined_bootstrap_replicates: int
    _replicate_values: tuple[float | None, ...]
    _replicate_undefined_reasons: tuple[str | None, ...]


def _validate_common_group_reference_coverage() -> None:
    if (
        COMMON_GROUP_REFERENCE_COVERAGE != 0.50
        or COMMON_GROUP_REFERENCE_COVERAGE != min(REQUESTED_COVERAGES)
    ):
        raise RuntimeError(
            "COMMON_GROUP_REFERENCE_COVERAGE must remain the frozen 0.50 "
            "minimum of REQUESTED_COVERAGES"
        )


_validate_common_group_reference_coverage()


def tracked_m5_v0_schema_has_coordinate_fields() -> bool:
    """Return whether the tracked M5-v0 observation schema has all drift fields."""

    return set(SOURCE_COORDINATE_FIELDS).issubset(M5_V0_OBSERVATION_FIELDS)


def validate_m5_v0_source_schema(fieldnames: Sequence[str] | None) -> None:
    """Validate both the tracked contract and an input CSV header."""

    if not tracked_m5_v0_schema_has_coordinate_fields():
        raise RuntimeError(
            "Tracked M5-v0 per_observation_signals.csv schema does not contain "
            "c_clean, c_occ, and stability_abs; the M5-v1 input contract must not "
            "be broadened implicitly"
        )
    supplied = set(fieldnames or ())
    missing = [field for field in REQUIRED_SOURCE_FIELDS if field not in supplied]
    if missing:
        raise ValueError(
            "M5-v0 per_observation_signals.csv is missing required tracked fields: "
            + ", ".join(missing)
        )


def prepare_targeted_observations(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[TargetedObservationTable, dict[str, int]]:
    """Validate and select pairwise-complete v2b attachment observations.

    Risk construction is based only on the two frozen inference-time signals.
    The stored outcome is validated independently and then attached for metric
    evaluation.  Undefined outcomes or risks are not imputed.
    """

    targeted = [
        row
        for row in rows
        if str(row.get("method")) == METHOD and str(row.get("region")) == REGION
    ]
    group_ids: list[str] = []
    stability_values: list[float] = []
    geometry_values: list[float] = []
    perception_values: list[float] = []
    undefined_outcomes = 0
    incomplete_risks = 0
    for row_index, row in enumerate(targeted):
        clean = _optional_finite_value(
            row.get("c_clean"), field="c_clean", row_index=row_index
        )
        occluded = _optional_finite_value(
            row.get("c_occ"), field="c_occ", row_index=row_index
        )
        stability = _optional_finite_value(
            row.get("stability_abs"), field="stability_abs", row_index=row_index
        )
        coordinate_values = (clean, occluded, stability)
        if all(value is None for value in coordinate_values):
            undefined_outcomes += 1
            continue
        if any(value is None for value in coordinate_values):
            raise ValueError(
                "Targeted M5-v0 row has a partial c_clean/c_occ/stability_abs "
                f"triple at targeted row index {row_index}"
            )
        assert clean is not None and occluded is not None and stability is not None
        recomputed = abs(occluded - clean)
        if not math.isclose(recomputed, stability, rel_tol=1e-9, abs_tol=1e-9):
            raise ValueError(
                "Targeted M5-v0 stability_abs is inconsistent with "
                f"abs(c_occ - c_clean) at targeted row index {row_index}"
            )
        if stability < 0.0:
            raise ValueError("Targeted M5-v0 stability_abs must be non-negative")
        contact_fraction = _optional_finite_value(
            row.get("v2b_selected_contact_evidence_fraction"),
            field="v2b_selected_contact_evidence_fraction",
            row_index=row_index,
        )
        attachment_entropy = _optional_finite_value(
            row.get("predictive_entropy_attachment"),
            field="predictive_entropy_attachment",
            row_index=row_index,
        )
        if contact_fraction is None or attachment_entropy is None:
            incomplete_risks += 1
            continue
        if not 0.0 <= contact_fraction <= 1.0:
            raise ValueError("v2b_selected_contact_evidence_fraction must be in [0, 1]")
        if attachment_entropy < 0.0:
            raise ValueError("predictive_entropy_attachment must be non-negative")
        group_id = str(row.get("group_id", ""))
        if not group_id:
            raise ValueError("Targeted M5-v0 rows must have a non-empty group_id")
        group_ids.append(group_id)
        stability_values.append(stability)
        geometry_values.append(1.0 - contact_fraction)
        perception_values.append(attachment_entropy)
    if not group_ids:
        raise ValueError(
            "No pairwise-complete v2b attachment observations are available for M5-v1"
        )
    table = TargetedObservationTable(
        group_ids=np.asarray(group_ids, dtype=object),
        stability_abs=np.asarray(stability_values, dtype=np.float64),
        scores={
            "geometry_risk": np.asarray(geometry_values, dtype=np.float64),
            "perception_risk": np.asarray(perception_values, dtype=np.float64),
        },
    )
    return table, {
        "source_observation_rows": len(rows),
        "v2b_attachment_rows": len(targeted),
        "undefined_outcome_rows": undefined_outcomes,
        "incomplete_risk_rows": incomplete_risks,
        "targeted_observation_rows": table.observation_count,
        "targeted_group_count": len(set(group_ids)),
    }


def select_by_global_risk_threshold(
    risk_scores: Sequence[float] | np.ndarray,
    requested_coverage: float,
) -> RiskSelection:
    """Apply the frozen ceil/order-statistic rule without splitting ties."""

    risks = np.asarray(risk_scores, dtype=np.float64)
    if risks.ndim != 1 or len(risks) == 0:
        raise ValueError("risk_scores must be a non-empty one-dimensional array")
    if not np.all(np.isfinite(risks)):
        raise ValueError("risk_scores must contain only finite values")
    if (
        isinstance(requested_coverage, bool)
        or not math.isfinite(float(requested_coverage))
        or not 0.0 < float(requested_coverage) <= 1.0
    ):
        raise ValueError("requested_coverage must be finite and in (0, 1]")
    requested = float(requested_coverage)
    target_count = math.ceil(requested * len(risks))
    ordered = np.sort(risks, kind="mergesort")
    risk_cutoff = float(ordered[target_count - 1])
    retained = risks <= risk_cutoff
    retained.setflags(write=False)
    retained_count = int(np.count_nonzero(retained))
    realized = float(retained_count / len(risks))
    if requested == 1.0 and (retained_count != len(risks) or realized != 1.0):
        raise AssertionError("The 100%-coverage policy must retain every observation")
    return RiskSelection(
        requested_coverage=requested,
        target_count=target_count,
        risk_cutoff=risk_cutoff,
        retained_mask=retained,
        retained_count=retained_count,
        realized_coverage=realized,
    )


def common_retention_groups(table: TargetedObservationTable) -> frozenset[str]:
    """Groups retained at requested 0.50 by both frozen global policies."""

    _validate_common_group_reference_coverage()
    retained_group_sets = []
    for score_name in SCORE_NAMES:
        selection = select_by_global_risk_threshold(
            table.scores[score_name], COMMON_GROUP_REFERENCE_COVERAGE
        )
        retained_group_sets.append(
            {
                str(group_id)
                for group_id in table.group_ids[selection.retained_mask].tolist()
            }
        )
    return frozenset.intersection(*(frozenset(value) for value in retained_group_sets))


def group_primary_summary(
    table: TargetedObservationTable,
    *,
    cohort_role: str,
) -> list[dict[str, Any]]:
    """Aggregate each frozen score and drift within group before association."""

    _require_cohort_role(cohort_role)
    rows: list[dict[str, Any]] = []
    for score_name in SCORE_NAMES:
        definition = _SCORE_BY_NAME[score_name]
        for group_id in sorted({str(value) for value in table.group_ids.tolist()}):
            group_mask = table.group_ids == group_id
            rows.append(
                {
                    "score_name": score_name,
                    "source_signal": definition.source_signal,
                    "group_id": group_id,
                    "observation_count": int(np.count_nonzero(group_mask)),
                    "group_mean_risk": float(
                        np.mean(table.scores[score_name][group_mask])
                    ),
                    "group_mean_stability_abs": float(
                        np.mean(table.stability_abs[group_mask])
                    ),
                    "method": METHOD,
                    "region": REGION,
                    "cohort_role": cohort_role,
                }
            )
    return rows


def association_bootstrap_seed(score_name: str) -> np.random.SeedSequence:
    """Return the frozen, score-specific association-bootstrap seed sequence."""

    if score_name not in _SCORE_BY_NAME:
        raise ValueError(f"Unknown score name: {score_name}")
    return np.random.SeedSequence(
        [
            ASSOCIATION_BOOTSTRAP_SEED,
            ASSOCIATION_ANALYSIS_CODE,
            _SCORE_BY_NAME[score_name].score_code,
        ]
    )


def _association_bootstrap_from_group_pairs(
    group_mean_risk: Sequence[float] | np.ndarray,
    group_mean_stability_abs: Sequence[float] | np.ndarray,
    *,
    replicates: int,
    minimum_defined_bootstrap_fraction: float,
    confidence_level: float = ASSOCIATION_CONFIDENCE_LEVEL,
    seed_sequence: np.random.SeedSequence | None = None,
    resample_indices: np.ndarray | None = None,
) -> AssociationBootstrapResult:
    """Bootstrap a tie-aware Spearman statistic defined on group-level pairs."""

    risks = np.asarray(group_mean_risk, dtype=np.float64)
    outcomes = np.asarray(group_mean_stability_abs, dtype=np.float64)
    if risks.ndim != 1 or outcomes.ndim != 1 or len(risks) != len(outcomes):
        raise ValueError("group-level risk and outcome arrays must be equal-length 1D")
    if len(risks) == 0 or not np.all(np.isfinite(risks)):
        raise ValueError("group-level risks must be non-empty and finite")
    if not np.all(np.isfinite(outcomes)):
        raise ValueError("group-level outcomes must be finite")
    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
    ):
        raise ValueError("replicates must be a positive integer")
    if not 0.0 < minimum_defined_bootstrap_fraction <= 1.0:
        raise ValueError("minimum_defined_bootstrap_fraction must be in (0, 1]")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must be in (0, 1)")
    observed_rho, observed_reason = _tie_aware_spearman(risks, outcomes)
    if resample_indices is None:
        if seed_sequence is None:
            raise ValueError(
                "seed_sequence is required when resample_indices is absent"
            )
        generator = np.random.default_rng(seed_sequence)
        sampled = generator.integers(0, len(risks), size=(replicates, len(risks)))
    else:
        sampled = np.asarray(resample_indices, dtype=np.int64)
        if sampled.shape != (replicates, len(risks)):
            raise ValueError(
                "resample_indices must have shape (replicates, n_group_pairs)"
            )
        if np.any((sampled < 0) | (sampled >= len(risks))):
            raise ValueError("resample_indices contains an out-of-range group index")
    replicate_values: list[float | None] = []
    replicate_reasons: list[str | None] = []
    for indices in sampled:
        rho, reason = _tie_aware_spearman(risks[indices], outcomes[indices])
        replicate_values.append(rho)
        replicate_reasons.append(reason)
    defined_values = np.asarray(
        [value for value in replicate_values if value is not None],
        dtype=np.float64,
    )
    defined_count = len(defined_values)
    undefined_count = replicates - defined_count
    minimum_defined = math.ceil(minimum_defined_bootstrap_fraction * replicates)
    ci_lower = None
    ci_upper = None
    undefined_reason = observed_reason
    ci_undefined_reason = None
    if defined_count >= minimum_defined:
        alpha = (1.0 - confidence_level) / 2.0
        ci_lower, ci_upper = (
            float(value)
            for value in np.percentile(
                defined_values,
                [100.0 * alpha, 100.0 * (1.0 - alpha)],
            )
        )
    else:
        ci_undefined_reason = (
            "fewer than the minimum required fraction of defined bootstrap replicates"
        )
        if undefined_reason is None:
            undefined_reason = ci_undefined_reason
    return AssociationBootstrapResult(
        spearman_rho=observed_rho,
        bootstrap_defined_replicates=defined_count,
        bootstrap_undefined_replicates=undefined_count,
        ci_lower=ci_lower,
        ci_upper=ci_upper,
        undefined_reason=undefined_reason,
        ci_undefined_reason=ci_undefined_reason,
        minimum_defined_bootstrap_fraction=minimum_defined_bootstrap_fraction,
        minimum_defined_bootstrap_replicates=minimum_defined,
        _replicate_values=tuple(replicate_values),
        _replicate_undefined_reasons=tuple(replicate_reasons),
    )


def association_summary_rows(
    table: TargetedObservationTable,
    *,
    cohort_role: str,
    replicates: int = ASSOCIATION_BOOTSTRAP_REPLICATES,
    score_iteration: Sequence[str] = SCORE_NAMES,
) -> list[dict[str, Any]]:
    """Return exactly two fixed-order, group-first scientific associations."""

    _require_cohort_role(cohort_role)
    iteration = tuple(score_iteration)
    if len(iteration) != len(SCORE_NAMES) or set(iteration) != set(SCORE_NAMES):
        raise ValueError(f"association scores must contain exactly {SCORE_NAMES!r}")
    group_rows = group_primary_summary(table, cohort_role=cohort_role)
    by_score: dict[str, dict[str, Any]] = {}
    for score_name in iteration:
        selected = [row for row in group_rows if row["score_name"] == score_name]
        result = _association_bootstrap_from_group_pairs(
            [float(row["group_mean_risk"]) for row in selected],
            [float(row["group_mean_stability_abs"]) for row in selected],
            replicates=replicates,
            minimum_defined_bootstrap_fraction=(MINIMUM_DEFINED_BOOTSTRAP_FRACTION),
            seed_sequence=association_bootstrap_seed(score_name),
        )
        by_score[score_name] = {
            "score_name": score_name,
            "source_signal": _SCORE_BY_NAME[score_name].source_signal,
            "n_groups": len(selected),
            "spearman_rho": result.spearman_rho,
            "association_bootstrap_replicates": replicates,
            "association_bootstrap_seed": ASSOCIATION_BOOTSTRAP_SEED,
            "minimum_defined_bootstrap_fraction": (
                result.minimum_defined_bootstrap_fraction
            ),
            "minimum_defined_bootstrap_replicates": (
                result.minimum_defined_bootstrap_replicates
            ),
            "bootstrap_defined_replicates": result.bootstrap_defined_replicates,
            "bootstrap_undefined_replicates": (result.bootstrap_undefined_replicates),
            "ci_lower": result.ci_lower,
            "ci_upper": result.ci_upper,
            "undefined_reason": result.undefined_reason,
            "ci_undefined_reason": result.ci_undefined_reason,
            "cohort_role": cohort_role,
            "interpretation": _association_interpretation(
                cohort_role, score_name=score_name
            ),
        }
    return [by_score[score_name] for score_name in SCORE_NAMES]


def risk_distribution_summary(
    table: TargetedObservationTable,
) -> list[dict[str, Any]]:
    """Summarize exact risk values and tie blocks in frozen score order."""

    rows = []
    for score_name in SCORE_NAMES:
        values = table.scores[score_name]
        distinct, counts = np.unique(values, return_counts=True)
        largest_tie_count = int(np.max(counts))
        rows.append(
            {
                "score_name": score_name,
                "source_signal": _SCORE_BY_NAME[score_name].source_signal,
                "n_distinct_risk_values": len(distinct),
                "largest_tie_count": largest_tie_count,
                "largest_tie_fraction": float(
                    largest_tie_count / table.observation_count
                ),
                "minimum_risk": float(np.min(values)),
                "maximum_risk": float(np.max(values)),
            }
        )
    return rows


def _tie_aware_spearman(
    group_mean_risk: np.ndarray,
    group_mean_stability_abs: np.ndarray,
) -> tuple[float | None, str | None]:
    rows = [
        {
            "mean_signal": float(risk),
            "mean_stability_abs": float(outcome),
        }
        for risk, outcome in zip(
            group_mean_risk,
            group_mean_stability_abs,
            strict=True,
        )
    ]
    return _spearman_from_group_rows(rows)


def random_reference_seed(
    *, base_seed: int, score_name: str, requested_coverage: float
) -> np.random.SeedSequence:
    """Derive a stable order-independent random-reference seed."""

    if score_name not in _SCORE_BY_NAME:
        raise ValueError(f"Unknown score name: {score_name}")
    coverage_code = int(round(float(requested_coverage) * 10_000))
    return np.random.SeedSequence(
        [int(base_seed), _SCORE_BY_NAME[score_name].score_code, coverage_code]
    )


def random_reference_summaries(
    table: TargetedObservationTable,
    selections: Mapping[tuple[str, float], RiskSelection],
    *,
    draws: int,
    base_seed: int,
    score_names: Sequence[str] = SCORE_NAMES,
    requested_coverages: Sequence[float] = REQUESTED_COVERAGES,
) -> dict[tuple[str, float], dict[str, float]]:
    """Simulate descriptive random policies on the original table only."""

    if isinstance(draws, bool) or not isinstance(draws, int) or draws <= 0:
        raise ValueError("draws must be a positive integer")
    output: dict[tuple[str, float], dict[str, float]] = {}
    observation_indices = np.arange(table.observation_count)
    for score_name in score_names:
        if score_name not in _SCORE_BY_NAME:
            raise ValueError(f"Unknown score name: {score_name}")
        for requested_coverage in requested_coverages:
            key = (score_name, float(requested_coverage))
            if key not in selections:
                raise ValueError(f"Missing frozen-policy selection for {key!r}")
            retained_count = selections[key].retained_count
            generator = np.random.default_rng(
                random_reference_seed(
                    base_seed=base_seed,
                    score_name=score_name,
                    requested_coverage=float(requested_coverage),
                )
            )
            metrics = np.empty(draws, dtype=np.float64)
            represented_groups = np.empty(draws, dtype=np.float64)
            for draw_index in range(draws):
                retained_indices = generator.choice(
                    observation_indices,
                    size=retained_count,
                    replace=False,
                )
                mask = np.zeros(table.observation_count, dtype=bool)
                mask[retained_indices] = True
                metrics[draw_index], represented_count = _group_balanced_metric(
                    table.stability_abs,
                    table.group_ids,
                    mask,
                )
                represented_groups[draw_index] = represented_count
            p05, p95 = np.percentile(metrics, [5.0, 95.0])
            output[key] = {
                "random_reference_mean": float(np.mean(metrics)),
                "random_reference_p05": float(p05),
                "random_reference_p95": float(p95),
                "random_reference_represented_groups_mean": float(
                    np.mean(represented_groups)
                ),
            }
    return output


def group_cluster_bootstrap(
    table: TargetedObservationTable,
    *,
    replicates: int,
    seed: int,
    requested_coverages: Sequence[float] = REQUESTED_COVERAGES,
) -> BootstrapResult:
    """Recompute thresholds within every whole-group bootstrap replicate."""

    if (
        isinstance(replicates, bool)
        or not isinstance(replicates, int)
        or replicates <= 0
    ):
        raise ValueError("replicates must be a positive integer")
    coverages = tuple(float(value) for value in requested_coverages)
    required_headline_coverages = {
        HEADLINE_REFERENCE_COVERAGE,
        HEADLINE_SELECTIVE_COVERAGE,
    }
    if not required_headline_coverages.issubset(coverages):
        raise ValueError("bootstrap coverage grid must contain 1.00 and 0.70")
    unique_groups = sorted({str(value) for value in table.group_ids.tolist()})
    group_indices = [
        np.flatnonzero(table.group_ids == group_id) for group_id in unique_groups
    ]
    generator = np.random.default_rng(seed)
    values = {
        (score_name, coverage): np.empty(replicates, dtype=np.float64)
        for score_name in SCORE_NAMES
        for coverage in coverages
    }
    realized_coverages = {key: np.empty(replicates, dtype=np.float64) for key in values}
    represented_group_counts = {
        key: np.empty(replicates, dtype=np.float64) for key in values
    }
    sampled_positions_by_replicate: list[np.ndarray] = []
    headline_values = np.empty(replicates, dtype=np.float64)
    for replicate_index in range(replicates):
        sampled_group_positions = generator.integers(
            0,
            len(unique_groups),
            size=len(unique_groups),
        )
        recorded_positions = sampled_group_positions.copy()
        recorded_positions.setflags(write=False)
        sampled_positions_by_replicate.append(recorded_positions)
        sampled_indices_parts = [
            group_indices[int(position)] for position in sampled_group_positions
        ]
        sampled_indices = np.concatenate(sampled_indices_parts)
        bootstrap_cluster_ids = np.concatenate(
            [
                np.full(len(indices), cluster_index, dtype=np.int64)
                for cluster_index, indices in enumerate(sampled_indices_parts)
            ]
        )
        bootstrap_table = table.take(sampled_indices)
        geometry_metrics: dict[float, float] = {}
        for score_name in SCORE_NAMES:
            for coverage in coverages:
                selection = select_by_global_risk_threshold(
                    bootstrap_table.scores[score_name], coverage
                )
                metric, represented_count = _group_balanced_metric(
                    bootstrap_table.stability_abs,
                    bootstrap_cluster_ids,
                    selection.retained_mask,
                )
                key = (score_name, coverage)
                values[key][replicate_index] = metric
                realized_coverages[key][replicate_index] = selection.realized_coverage
                represented_group_counts[key][replicate_index] = represented_count
                if score_name == HEADLINE_SCORE:
                    geometry_metrics[coverage] = metric
        headline_values[replicate_index] = (
            geometry_metrics[HEADLINE_REFERENCE_COVERAGE]
            - geometry_metrics[HEADLINE_SELECTIVE_COVERAGE]
        )
    intervals = {
        key: tuple(float(value) for value in np.percentile(samples, [2.5, 97.5]))
        for key, samples in values.items()
    }
    diagnostics = {}
    for key in values:
        coverage_q25, coverage_median, coverage_q75 = np.percentile(
            realized_coverages[key], [25.0, 50.0, 75.0]
        )
        diagnostics[key] = {
            "bootstrap_realized_coverage_median": float(coverage_median),
            "bootstrap_realized_coverage_q25": float(coverage_q25),
            "bootstrap_realized_coverage_q75": float(coverage_q75),
            "bootstrap_represented_groups_median": float(
                np.median(represented_group_counts[key])
            ),
        }
        values[key].setflags(write=False)
    headline_low, headline_high = np.percentile(headline_values, [2.5, 97.5])
    headline_values.setflags(write=False)
    return BootstrapResult(
        curve_intervals=intervals,
        headline_values=headline_values,
        headline_ci_low=float(headline_low),
        headline_ci_high=float(headline_high),
        _curve_values=values,
        curve_diagnostics=diagnostics,
        _sampled_group_positions=tuple(sampled_positions_by_replicate),
    )


def analyze_selective_action_table(
    table: TargetedObservationTable,
    *,
    cohort_role: str,
) -> dict[str, Any]:
    """Run the complete frozen M5-v1 analysis."""

    return _analyze_selective_action_table(
        table,
        cohort_role=cohort_role,
        association_bootstrap_replicates=ASSOCIATION_BOOTSTRAP_REPLICATES,
        bootstrap_replicates=BOOTSTRAP_REPLICATES,
        random_reference_draws=RANDOM_REFERENCE_DRAWS,
    )


def _analyze_selective_action_table(
    table: TargetedObservationTable,
    *,
    cohort_role: str,
    association_bootstrap_replicates: int,
    bootstrap_replicates: int,
    random_reference_draws: int,
) -> dict[str, Any]:
    """Testable implementation behind the fixed public analysis entry point."""

    _require_cohort_role(cohort_role)
    group_rows = group_primary_summary(table, cohort_role=cohort_role)
    association_rows = association_summary_rows(
        table,
        cohort_role=cohort_role,
        replicates=association_bootstrap_replicates,
    )
    risk_rows = risk_distribution_summary(table)
    selections = {
        (score_name, coverage): select_by_global_risk_threshold(
            table.scores[score_name], coverage
        )
        for score_name in SCORE_NAMES
        for coverage in REQUESTED_COVERAGES
    }
    common_groups = common_retention_groups(table)
    bootstrap = group_cluster_bootstrap(
        table,
        replicates=bootstrap_replicates,
        seed=BOOTSTRAP_SEED,
    )
    random_references = random_reference_summaries(
        table,
        selections,
        draws=random_reference_draws,
        base_seed=RANDOM_REFERENCE_BASE_SEED,
    )
    curve_rows = []
    point_metrics: dict[tuple[str, float], float] = {}
    total_group_count = len(set(table.group_ids.tolist()))
    for score_name in SCORE_NAMES:
        definition = _SCORE_BY_NAME[score_name]
        for coverage in REQUESTED_COVERAGES:
            key = (score_name, coverage)
            selection = selections[key]
            metric, represented_groups = _group_balanced_metric(
                table.stability_abs,
                table.group_ids,
                selection.retained_mask,
            )
            point_metrics[key] = metric
            common_metric = None
            if len(common_groups) >= MINIMUM_COMMON_GROUPS:
                common_mask = selection.retained_mask & np.isin(
                    table.group_ids,
                    tuple(common_groups),
                )
                common_metric, _ = _group_balanced_metric(
                    table.stability_abs,
                    table.group_ids,
                    common_mask,
                )
            retained_drifts = table.stability_abs[selection.retained_mask]
            zero_count = int(np.count_nonzero(retained_drifts == 0.0))
            ci_low, ci_high = bootstrap.curve_intervals[key]
            curve_rows.append(
                {
                    "method": METHOD,
                    "region": REGION,
                    "cohort_role": cohort_role,
                    "score": score_name,
                    "source_signal": definition.source_signal,
                    "requested_coverage": coverage,
                    "target_count": selection.target_count,
                    "risk_cutoff": selection.risk_cutoff,
                    "total_observation_count": table.observation_count,
                    "total_group_count": total_group_count,
                    "retained_observation_count": selection.retained_count,
                    "realized_observation_coverage": selection.realized_coverage,
                    "represented_group_count": represented_groups,
                    "realized_group_coverage": float(
                        represented_groups / total_group_count
                    ),
                    "group_balanced_mean_abs_drift": metric,
                    "observation_weighted_mean_abs_drift": float(
                        np.mean(retained_drifts)
                    ),
                    "group_cluster_bootstrap_ci_low": ci_low,
                    "group_cluster_bootstrap_ci_high": ci_high,
                    **bootstrap.curve_diagnostics[key],
                    "maximum_retained_abs_drift": float(np.max(retained_drifts)),
                    "retained_zero_drift_count": zero_count,
                    "retained_zero_drift_fraction": float(
                        zero_count / selection.retained_count
                    ),
                    **random_references[key],
                    "common_retention_group_count": len(common_groups),
                    "common_group_balanced_mean_abs_drift": common_metric,
                }
            )
    if len(common_groups) >= MINIMUM_COMMON_GROUPS:
        expected_common_100 = float(
            np.mean(
                [
                    np.mean(table.stability_abs[table.group_ids == group_id])
                    for group_id in sorted(common_groups)
                ]
            )
        )
        for score_name in SCORE_NAMES:
            row = next(
                item
                for item in curve_rows
                if item["score"] == score_name and item["requested_coverage"] == 1.0
            )
            if not math.isclose(
                float(row["common_group_balanced_mean_abs_drift"]),
                expected_common_100,
                rel_tol=1e-12,
                abs_tol=1e-12,
            ):
                raise AssertionError("Common-group 100%-coverage invariant failed")
    headline_value = (
        point_metrics[(HEADLINE_SCORE, HEADLINE_REFERENCE_COVERAGE)]
        - point_metrics[(HEADLINE_SCORE, HEADLINE_SELECTIVE_COVERAGE)]
    )
    headline = {
        "score": HEADLINE_SCORE,
        "requested_coverage_reference": HEADLINE_REFERENCE_COVERAGE,
        "requested_coverage_selective": HEADLINE_SELECTIVE_COVERAGE,
        "headline_selective_improvement_px": headline_value,
        "group_cluster_bootstrap_ci_low": bootstrap.headline_ci_low,
        "group_cluster_bootstrap_ci_high": bootstrap.headline_ci_high,
        "bootstrap_replicates": bootstrap_replicates,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "positive_value_interpretation": (
            "lower retained drift under the 0.70-requested geometry policy"
        ),
        "negative_value_interpretation": (
            "higher retained drift under the 0.70-requested geometry policy"
        ),
        "cohort_role": cohort_role,
        "interpretation": _cohort_interpretation(cohort_role),
        "confidence_interval_scope": (
            "group-resampling variability within this cohort analysis"
        ),
    }
    return {
        "group_primary_summary": group_rows,
        "association_summary": association_rows,
        "risk_distribution_summary": risk_rows,
        "selective_action_curve": curve_rows,
        "headline_selective_contrast": headline,
        "common_retention_groups": common_groups,
    }


def frozen_m5_v1_configuration(*, cohort_role: str) -> dict[str, Any]:
    """Return the public, non-tunable M5-v1 configuration."""

    _require_cohort_role(cohort_role)
    _validate_common_group_reference_coverage()
    minimum_defined_association_bootstraps = math.ceil(
        MINIMUM_DEFINED_BOOTSTRAP_FRACTION * ASSOCIATION_BOOTSTRAP_REPLICATES
    )
    return {
        "profile": PROFILE_NAME,
        "source_profile": M5_V0_PROFILE_NAME,
        "method": METHOD,
        "region": REGION,
        "cohort_role": cohort_role,
        "cohort_interpretation": _cohort_interpretation(cohort_role),
        "target_population": (
            "pairwise-complete v2b attachment rows with both frozen risk scores"
        ),
        "risk_scores": [definition.__dict__ for definition in SCORE_DEFINITIONS],
        "requested_coverages": list(REQUESTED_COVERAGES),
        "policy": {
            "conceptual_object": (
                "a single global risk threshold swept over requested coverage"
            ),
            "target_count": "ceil(requested_coverage * targeted_observation_count)",
            "cutoff": "target_count-th ascending risk order statistic",
            "retention": "risk <= risk_cutoff; boundary ties retained completely",
            "outcome_used_for_selection": False,
        },
        "primary_metric": (
            "equal-weight mean across represented group_id-specific retained "
            "mean stability_abs values"
        ),
        "association": {
            "independent_unit": "group_id",
            "aggregation": (
                "arithmetic group mean risk paired with arithmetic group mean "
                "stability_abs"
            ),
            "statistic": "tie-aware Spearman rho across group means",
            "bootstrap_representation": "pre-aggregated group-level pairs",
            "bootstrap_replicates": ASSOCIATION_BOOTSTRAP_REPLICATES,
            "bootstrap_base_seed": ASSOCIATION_BOOTSTRAP_SEED,
            "bootstrap_analysis_code": ASSOCIATION_ANALYSIS_CODE,
            "bootstrap_seed_derivation": ("numpy SeedSequence([0, 101, score_code])"),
            "confidence_level": ASSOCIATION_CONFIDENCE_LEVEL,
            "minimum_defined_bootstrap_fraction": (MINIMUM_DEFINED_BOOTSTRAP_FRACTION),
            "minimum_defined_bootstrap_replicates": (
                minimum_defined_association_bootstraps
            ),
            "minimum_defined_fraction_role": (
                "deterministic degeneracy guard; not significance or power threshold"
            ),
        },
        "selective_curve_bootstrap": {
            "unit": "group_id",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
            "seed_derivation": "numpy default_rng(0) from the integer base seed",
            "stream_scope": (
                "one generator shared across the full curve bootstrap; one whole-"
                "group position sample is consumed per replicate"
            ),
            "interval": "95% percentile",
            "bootstrap_representation": (
                "whole source groups reconstructed as observation rows with distinct "
                "bootstrap-copy cluster identities"
            ),
            "thresholds_recomputed_within_each_replicate": True,
            "random_reference_nested": False,
        },
        "monte_carlo_stream_relationship": {
            "same_scientific_cohort": True,
            "separate_deterministic_streams": True,
            "separate_streams_imply_independent_scientific_evidence": False,
        },
        "headline_contrast": {
            "score": HEADLINE_SCORE,
            "definition": (
                "group_balanced_mean_abs_drift_at_requested_1.00 minus "
                "group_balanced_mean_abs_drift_at_requested_0.70"
            ),
            "other_headline_contrasts": None,
        },
        "random_reference": {
            "population": "original targeted observation table",
            "draws": RANDOM_REFERENCE_DRAWS,
            "base_seed": RANDOM_REFERENCE_BASE_SEED,
            "sampling": "uniform without replacement at actual retained count K",
            "seed_derivation": (
                "numpy SeedSequence([base_seed, score_code, "
                "round(requested_coverage * 10000)])"
            ),
            "interval_interpretation": "descriptive random-policy distribution",
            "common_group_variant": False,
        },
        "common_group_secondary_curve": {
            "definition": (
                "groups retaining at least one observation at requested 0.50 "
                "under both frozen policies"
            ),
            "reference_requested_coverage": COMMON_GROUP_REFERENCE_COVERAGE,
            "reference_equals_grid_minimum": True,
            "minimum_groups": MINIMUM_COMMON_GROUPS,
            "thresholds_recomputed_after_group_restriction": False,
        },
        "curve_summary_policy": {
            "integrated_scalar_reported": False,
            "rationale": (
                "deferred because a scalar comparison could invite a geometry-"
                "versus-entropy confirmatory claim from the development cohort "
                "that generated the geometry hypothesis"
            ),
        },
        "source_stability_validation": {
            "definition": "abs(c_occ - c_clean)",
            "rel_tol": 1e-9,
            "abs_tol": 1e-9,
        },
        "zero_drift_definition": "exact stored validated stability_abs == 0.0",
        "training_or_optimization": False,
        "calibrated_policy": False,
    }


def run_selective_action_analysis(
    m5_v0_root: PathLike,
    output_root: PathLike,
    *,
    cohort_role: str,
    profile: str = PROFILE_NAME,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Read one M5-v0 artifact set and write a separate M5-v1 analysis."""

    _require_profile(profile)
    _require_cohort_role(cohort_role)
    source_root = Path(m5_v0_root)
    destination = Path(output_root)
    _validate_separate_roots(source_root, destination)
    _validate_output_destination(destination, overwrite=overwrite)
    source_manifest_path = source_root / "manifest.json"
    source_observations_path = source_root / "per_observation_signals.csv"
    if not source_manifest_path.is_file():
        raise FileNotFoundError(
            f"Required M5-v0 manifest does not exist: {source_manifest_path}"
        )
    if not source_observations_path.is_file():
        raise FileNotFoundError(
            f"Required M5-v0 observation table does not exist: {source_observations_path}"
        )
    source_manifest = json.loads(source_manifest_path.read_text(encoding="utf-8"))
    if source_manifest.get("profile") != M5_V0_PROFILE_NAME:
        raise ValueError("Source root is not a frozen m5_v0 result")
    with source_observations_path.open(
        "r", encoding="utf-8-sig", newline=""
    ) as input_file:
        reader = csv.DictReader(input_file)
        validate_m5_v0_source_schema(reader.fieldnames)
        source_rows = list(reader)
    table, input_counts = prepare_targeted_observations(source_rows)
    analysis = analyze_selective_action_table(table, cohort_role=cohort_role)
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{destination.name}.staging-", dir=destination.parent)
    )
    try:
        manifest, summary = _write_selective_action_artifacts(
            staging,
            table,
            analysis,
            input_counts=input_counts,
            cohort_role=cohort_role,
        )
        _install_staged_directory(staging, destination, overwrite=overwrite)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise
    return {"manifest": manifest, "summary": summary, "analysis": analysis}


def _write_selective_action_artifacts(
    root: Path,
    table: TargetedObservationTable,
    analysis: Mapping[str, Any],
    *,
    input_counts: Mapping[str, int],
    cohort_role: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    from strawberry_occlusion.visualization.selective_action import (
        create_selective_action_plots,
    )

    curve_rows = analysis["selective_action_curve"]
    headline = analysis["headline_selective_contrast"]
    _write_curve_csv(root / "selective_action_curve.csv", curve_rows)
    _write_rows_csv(
        root / "group_primary_summary.csv",
        GROUP_PRIMARY_FIELDS,
        analysis["group_primary_summary"],
    )
    _write_rows_csv(
        root / "association_summary.csv",
        ASSOCIATION_FIELDS,
        analysis["association_summary"],
    )
    _write_rows_csv(
        root / "risk_distribution_summary.csv",
        RISK_DISTRIBUTION_FIELDS,
        analysis["risk_distribution_summary"],
    )
    _write_json(root / "headline_selective_contrast.json", headline)
    configuration = frozen_m5_v1_configuration(cohort_role=cohort_role)
    _assert_no_absolute_paths(configuration)
    _write_json(root / "configuration.json", configuration)
    input_provenance = {
        "profile": PROFILE_NAME,
        "source_profile": M5_V0_PROFILE_NAME,
        "source_artifacts_read_only": [
            "manifest.json",
            "per_observation_signals.csv",
        ],
        "required_source_fields": list(REQUIRED_SOURCE_FIELDS),
        "method": METHOD,
        "region": REGION,
        "counts": dict(input_counts),
        "source_observation_rows_copied_to_output": False,
        "absolute_source_path_persisted": False,
    }
    _assert_no_absolute_paths(input_provenance)
    _write_json(root / "input_provenance.json", input_provenance)
    summary = {
        "profile": PROFILE_NAME,
        "source_profile": M5_V0_PROFILE_NAME,
        "method": METHOD,
        "region": REGION,
        "cohort_role": cohort_role,
        "interpretation": _cohort_interpretation(cohort_role),
        "counts": dict(input_counts),
        "common_retention_group_count": len(analysis["common_retention_groups"]),
        "common_group_analysis_defined": (
            len(analysis["common_retention_groups"]) >= MINIMUM_COMMON_GROUPS
        ),
        "association_summary": analysis["association_summary"],
        "risk_distribution_summary": analysis["risk_distribution_summary"],
        "headline_selective_contrast": headline,
        "claim_boundary": _claim_boundary(cohort_role),
        "claims": {
            "calibrated_confidence": False,
            "calibrated_failure_probability": False,
            "deployment_readiness": False,
            "confirmatory_geometry_vs_entropy_comparison": False,
        },
    }
    _assert_no_absolute_paths(summary)
    _write_json(root / "summary.json", summary)
    plots = create_selective_action_plots(
        curve_rows,
        analysis["group_primary_summary"],
        analysis["association_summary"],
        root / "plots",
        method=METHOD,
        region=REGION,
        n_groups=len(set(table.group_ids.tolist())),
        cohort_role=cohort_role,
    )
    manifest = {
        "study": "M5-v1 Selective Action Under Attachment Occlusion",
        "profile": PROFILE_NAME,
        "source_profile": M5_V0_PROFILE_NAME,
        "method": METHOD,
        "region": REGION,
        "cohort_role": cohort_role,
        "statistical_unit": "group_id",
        "calibrated_deployment_policy": False,
        "training_or_optimization_performed": False,
        "artifacts": {
            "configuration": "configuration.json",
            "input_provenance": "input_provenance.json",
            "group_primary_summary": "group_primary_summary.csv",
            "association_summary": "association_summary.csv",
            "risk_distribution_summary": "risk_distribution_summary.csv",
            "selective_action_curve": "selective_action_curve.csv",
            "headline_selective_contrast": "headline_selective_contrast.json",
            "summary": "summary.json",
            "plots": plots,
        },
    }
    _assert_no_absolute_paths(manifest)
    _write_json(root / "manifest.json", manifest)
    return manifest, summary


def _group_balanced_metric(
    stability_abs: np.ndarray,
    group_ids: np.ndarray,
    retained_mask: np.ndarray,
) -> tuple[float, int]:
    retained = np.asarray(retained_mask, dtype=bool)
    if retained.shape != stability_abs.shape:
        raise ValueError("retained_mask must match the outcome array")
    if not np.any(retained):
        raise ValueError("At least one observation must be retained")
    represented = np.unique(group_ids[retained])
    group_means = [
        float(np.mean(stability_abs[retained & (group_ids == group_id)]))
        for group_id in represented
    ]
    return float(np.mean(group_means)), len(represented)


def _write_curve_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
) -> None:
    """Write policy thresholds and metrics with round-trip-safe float precision."""

    _write_rows_csv(path, CURVE_FIELDS, rows)


def _write_rows_csv(
    path: Path,
    fields: Sequence[str],
    rows: Sequence[Mapping[str, Any]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=fields,
            lineterminator="\n",
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: _curve_csv_value(row.get(field)) for field in fields}
            )


def _curve_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if value else "false"
    if isinstance(value, (float, np.floating)):
        numeric = float(value)
        return format(numeric, ".17g") if math.isfinite(numeric) else ""
    if isinstance(value, np.integer):
        return int(value)
    return value


def _optional_finite_value(
    value: Any,
    *,
    field: str,
    row_index: int,
) -> float | None:
    if value is None or (isinstance(value, str) and value.strip() == ""):
        return None
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(
            f"Targeted M5-v0 {field} must be numeric at targeted row index {row_index}"
        )
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f"Targeted M5-v0 {field} must be numeric at targeted row index {row_index}"
        ) from error
    if not math.isfinite(numeric):
        raise ValueError(
            f"Targeted M5-v0 {field} must be finite at targeted row index {row_index}"
        )
    return numeric


def _cohort_interpretation(cohort_role: str) -> str:
    if cohort_role == "development_reuse":
        return DEVELOPMENT_REUSE_INTERPRETATION
    return (
        "Cohort independence was declared by the caller; image/content "
        "independence is not verified by this module."
    )


def _association_interpretation(cohort_role: str, *, score_name: str) -> str:
    if cohort_role == "independent_validation":
        return _cohort_interpretation(cohort_role)
    if score_name == "geometry_risk":
        return (
            "integrity / re-expression check on development reuse; not new "
            "empirical evidence"
        )
    return "descriptive development-reuse comparator; non-confirmatory"


def _claim_boundary(cohort_role: str) -> dict[str, str]:
    if cohort_role == "development_reuse":
        return {
            "association": (
                "geometry association is an integrity / re-expression check, "
                "not new empirical evidence"
            ),
            "selective_curve": (
                "new operational descriptive quantity, but selection-biased and "
                "non-confirmatory"
            ),
        }
    return {
        "association": _cohort_interpretation(cohort_role),
        "selective_curve": (
            "pre-specified operational analysis on a caller-declared cohort; "
            "image/content independence is not verified by this module"
        ),
    }


def _validate_separate_roots(source_root: Path, destination: Path) -> None:
    source = source_root.resolve()
    output = destination.resolve()
    if source == output or source in output.parents or output in source.parents:
        raise ValueError("M5-v0 input and M5-v1 output roots must be separate")


def _require_profile(profile: str) -> None:
    if profile != PROFILE_NAME:
        raise ValueError(f"Only frozen profile {PROFILE_NAME!r} is supported")


def _require_cohort_role(cohort_role: str) -> None:
    if cohort_role not in COHORT_ROLES:
        raise ValueError(f"cohort_role must be one of {COHORT_ROLES!r}")


def _build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run the frozen M5-v1 selective-action analysis."
    )
    parser.add_argument("--profile", choices=(PROFILE_NAME,), required=True)
    parser.add_argument("--m5-v0-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cohort-role", choices=COHORT_ROLES, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run the M5-v1 command-line workflow."""

    arguments = _build_argument_parser().parse_args(argv)
    result = run_selective_action_analysis(
        arguments.m5_v0_root,
        arguments.output_root,
        cohort_role=arguments.cohort_role,
        profile=arguments.profile,
        overwrite=arguments.overwrite,
    )
    counts = result["summary"]["counts"]
    print(
        f"Completed {PROFILE_NAME}: {counts['targeted_observation_rows']} targeted "
        f"observations from {counts['targeted_group_count']} groups."
    )
    print(_cohort_interpretation(arguments.cohort_role))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
