from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from matplotlib.axes import Axes
from matplotlib.figure import Figure

import strawberry_occlusion.evaluation.selective_action as selective_action
from strawberry_occlusion.evaluation.action_failure_signals import (
    OBSERVATION_FIELDS as M5_V0_OBSERVATION_FIELDS,
)
from strawberry_occlusion.evaluation.selective_action import (
    ASSOCIATION_FIELDS,
    BOOTSTRAP_SEED,
    COMMON_GROUP_REFERENCE_COVERAGE,
    CURVE_FIELDS,
    DEVELOPMENT_REUSE_INTERPRETATION,
    GROUP_PRIMARY_FIELDS,
    METHOD,
    PROFILE_NAME,
    RANDOM_REFERENCE_BASE_SEED,
    REGION,
    REQUESTED_COVERAGES,
    RISK_DISTRIBUTION_FIELDS,
    SCORE_NAMES,
    TargetedObservationTable,
    _analyze_selective_action_table,
    _association_bootstrap_from_group_pairs,
    _build_argument_parser,
    _validate_common_group_reference_coverage,
    association_summary_rows,
    common_retention_groups,
    frozen_m5_v1_configuration,
    group_cluster_bootstrap,
    group_primary_summary,
    prepare_targeted_observations,
    random_reference_summaries,
    risk_distribution_summary,
    run_selective_action_analysis,
    select_by_global_risk_threshold,
    tracked_m5_v0_schema_has_coordinate_fields,
    validate_m5_v0_source_schema,
)
from strawberry_occlusion.visualization.selective_action import (
    DEVELOPMENT_ASSOCIATION_NOTE,
    DEVELOPMENT_BANNER,
    DEVELOPMENT_COMPARATOR_NOTE,
    PLOT_FILENAMES,
)


def _table(
    group_ids: list[str],
    drifts: list[float],
    geometry_risks: list[float],
    perception_risks: list[float] | None = None,
) -> TargetedObservationTable:
    perception = geometry_risks if perception_risks is None else perception_risks
    return TargetedObservationTable(
        group_ids=np.asarray(group_ids, dtype=object),
        stability_abs=np.asarray(drifts, dtype=np.float64),
        scores={
            "geometry_risk": np.asarray(geometry_risks, dtype=np.float64),
            "perception_risk": np.asarray(perception, dtype=np.float64),
        },
    )


def _source_row(
    group_id: str,
    drift: float,
    geometry_risk: float,
    perception_risk: float,
    *,
    method: str = METHOD,
    region: str = REGION,
) -> dict[str, Any]:
    clean = 100.0
    return {
        "group_id": group_id,
        "method": method,
        "region": region,
        "c_clean": clean,
        "c_occ": clean + drift,
        "stability_abs": drift,
        "v2b_selected_contact_evidence_fraction": 1.0 - geometry_risk,
        "predictive_entropy_attachment": perception_risk,
    }


def _small_analysis(table: TargetedObservationTable) -> dict[str, Any]:
    return _analyze_selective_action_table(
        table,
        cohort_role="development_reuse",
        association_bootstrap_replicates=80,
        bootstrap_replicates=80,
        random_reference_draws=80,
    )


@pytest.mark.parametrize("requested_coverage", (0.50, 0.60, 0.70, 0.80, 0.90))
def test_distinct_risks_use_ceil_finite_sample_count_and_realized_coverage(
    requested_coverage: float,
) -> None:
    risks = np.asarray([0.05, 0.15, 0.25, 0.35, 0.45, 0.55, 0.65])

    selection = select_by_global_risk_threshold(risks, requested_coverage)

    expected = math.ceil(requested_coverage * len(risks))
    assert selection.target_count == expected
    assert selection.retained_count == expected
    assert selection.realized_coverage == expected / len(risks)


def test_boundary_tie_expands_after_ceil_cutoff_without_splitting_ties() -> None:
    risks = np.asarray([0.0, 0.1, 0.1, 0.1, 0.8, 0.9])

    selection = select_by_global_risk_threshold(risks, 0.30)

    assert selection.target_count == math.ceil(0.30 * len(risks)) == 2
    assert selection.risk_cutoff == 0.1
    assert selection.retained_mask.tolist() == [True, True, True, True, False, False]
    assert selection.retained_count == 4


@pytest.mark.parametrize("requested_coverage", REQUESTED_COVERAGES)
def test_all_tied_risks_realize_full_coverage(requested_coverage: float) -> None:
    selection = select_by_global_risk_threshold(np.full(9, 0.25), requested_coverage)

    assert selection.realized_coverage == 1.0
    assert selection.retained_count == 9


def test_requested_100_percent_retains_every_observation_exactly() -> None:
    selection = select_by_global_risk_threshold(np.asarray([0.4, 0.1, 0.7, 0.2]), 1.0)

    assert selection.target_count == 4
    assert selection.retained_count == 4
    assert selection.realized_coverage == 1.0
    assert np.all(selection.retained_mask)


def test_tracked_source_schema_and_coordinate_consistency_are_validated() -> None:
    assert tracked_m5_v0_schema_has_coordinate_fields() is True
    assert {"c_clean", "c_occ", "stability_abs"}.issubset(M5_V0_OBSERVATION_FIELDS)
    validate_m5_v0_source_schema(M5_V0_OBSERVATION_FIELDS)
    rows = [
        _source_row("group_a", 1.5, 0.2, 0.3),
        _source_row("group_b", 0.0, 0.4, 0.5),
    ]

    table, counts = prepare_targeted_observations(rows)

    assert table.stability_abs.tolist() == [1.5, 0.0]
    assert table.scores["geometry_risk"].tolist() == pytest.approx([0.2, 0.4])
    assert counts["targeted_observation_rows"] == 2


def test_source_schema_and_stability_mismatch_fail_loudly() -> None:
    with pytest.raises(ValueError, match="missing required tracked fields"):
        validate_m5_v0_source_schema(
            [field for field in M5_V0_OBSERVATION_FIELDS if field != "c_occ"]
        )
    row = _source_row("group_a", 2.0, 0.2, 0.3)
    row["stability_abs"] = 2.01

    with pytest.raises(ValueError, match="inconsistent"):
        prepare_targeted_observations([row])

    malformed = _source_row("group_a", 2.0, 0.2, 0.3)
    malformed["c_occ"] = "not-a-number"
    with pytest.raises(ValueError, match="c_occ must be numeric"):
        prepare_targeted_observations([malformed])


def test_analysis_stops_if_tracked_m5_v0_coordinate_contract_disappears(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        selective_action,
        "M5_V0_OBSERVATION_FIELDS",
        tuple(
            field
            for field in M5_V0_OBSERVATION_FIELDS
            if field not in {"c_clean", "c_occ"}
        ),
    )

    with pytest.raises(RuntimeError, match="must not be broadened"):
        validate_m5_v0_source_schema(M5_V0_OBSERVATION_FIELDS)


def test_coordinate_consistency_check_is_targeted_to_v2b_attachment_rows() -> None:
    valid = _source_row("group_a", 2.0, 0.2, 0.3)
    out_of_scope = _source_row(
        "group_a", 2.0, 0.2, 0.3, method="v2a", region="background"
    )
    out_of_scope["stability_abs"] = 999.0

    table, _ = prepare_targeted_observations([valid, out_of_scope])

    assert table.observation_count == 1


@pytest.mark.parametrize("evidence_fraction", (-0.01, 1.01))
def test_out_of_range_evidence_fraction_is_rejected(
    evidence_fraction: float,
) -> None:
    row = _source_row("group_a", 1.0, 0.2, 0.3)
    row["v2b_selected_contact_evidence_fraction"] = evidence_fraction

    with pytest.raises(ValueError, match=r"must be in \[0, 1\]"):
        prepare_targeted_observations([row])


def test_group_primary_association_has_exact_two_rows_in_fixed_order() -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        [0.0, 2.0, 2.0, 4.0, 5.0, 7.0, 9.0, 11.0],
        [0.0, 0.2, 0.3, 0.5, 0.5, 0.5, 0.8, 1.0],
        [0.9, 0.7, 0.6, 0.4, 0.4, 0.4, 0.2, 0.0],
    )

    group_rows = group_primary_summary(table, cohort_role="development_reuse")
    associations = association_summary_rows(
        table,
        cohort_role="development_reuse",
        replicates=200,
    )

    assert [row["score_name"] for row in associations] == [
        "geometry_risk",
        "perception_risk",
    ]
    assert [row["source_signal"] for row in associations] == [
        "v2b_selected_contact_evidence_fraction",
        "predictive_entropy_attachment",
    ]
    assert len(associations) == 2
    assert [row["score_name"] for row in group_rows] == [
        *(["geometry_risk"] * 4),
        *(["perception_risk"] * 4),
    ]
    geometry_a = next(
        row
        for row in group_rows
        if row["score_name"] == "geometry_risk" and row["group_id"] == "a"
    )
    assert geometry_a["observation_count"] == 2
    assert geometry_a["group_mean_risk"] == pytest.approx(0.1)
    assert geometry_a["group_mean_stability_abs"] == pytest.approx(1.0)
    assert associations[0]["n_groups"] == 4
    assert associations[0]["minimum_defined_bootstrap_fraction"] == 0.50
    assert associations[0]["minimum_defined_bootstrap_replicates"] == 100
    assert (
        associations[0]["bootstrap_defined_replicates"]
        + associations[0]["bootstrap_undefined_replicates"]
        == 200
    )
    assert "p_value" not in associations[0]


def test_association_result_is_invariant_to_score_iteration_order() -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "c", "d", "d", "e", "e"],
        [0.0, 1.0, 2.0, 6.0, 3.0, 9.0, 5.0, 12.0, 4.0, 15.0],
        [0.0, 0.2, 0.1, 0.4, 0.3, 0.6, 0.5, 0.8, 0.7, 1.0],
        [1.0, 0.8, 0.9, 0.6, 0.7, 0.4, 0.5, 0.2, 0.3, 0.0],
    )

    forward = association_summary_rows(
        table,
        cohort_role="independent_validation",
        replicates=300,
        score_iteration=SCORE_NAMES,
    )
    reverse = association_summary_rows(
        table,
        cohort_role="independent_validation",
        replicates=300,
        score_iteration=tuple(reversed(SCORE_NAMES)),
    )

    assert forward == reverse
    geometry_seed = selective_action.association_bootstrap_seed(
        "geometry_risk"
    ).generate_state(4)
    perception_seed = selective_action.association_bootstrap_seed(
        "perception_risk"
    ).generate_state(4)
    assert not np.array_equal(geometry_seed, perception_seed)


def test_production_association_rejects_third_or_alternate_score() -> None:
    table = _table(
        ["a", "b", "c"],
        [1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3],
    )

    with pytest.raises(ValueError, match="must contain exactly"):
        association_summary_rows(
            table,
            cohort_role="development_reuse",
            replicates=10,
            score_iteration=(
                "geometry_risk",
                "perception_risk",
                "raw_evidence_fraction",
            ),
        )
    with pytest.raises(ValueError, match="scores must contain exactly"):
        TargetedObservationTable(
            group_ids=np.asarray(["a", "b", "c"], dtype=object),
            stability_abs=np.asarray([1.0, 2.0, 3.0]),
            scores={
                "geometry_risk": np.asarray([0.1, 0.2, 0.3]),
                "perception_risk": np.asarray([0.2, 0.3, 0.4]),
                "raw_evidence_fraction": np.asarray([0.9, 0.8, 0.7]),
            },
        )


def test_association_uses_group_first_result_not_unbalanced_row_level_result() -> None:
    group_ids: list[str] = []
    risks: list[float] = []
    drifts: list[float] = []
    for group_index, count in enumerate((20, 5, 5)):
        for within_group in np.linspace(0.0, 10.0, count):
            group_ids.append(f"group_{group_index}")
            risks.append(float(within_group + group_index))
            drifts.append(float(15.0 + group_index - within_group))
    table = _table(group_ids, drifts, risks)

    reported = association_summary_rows(
        table,
        cohort_role="development_reuse",
        replicates=100,
    )[0]["spearman_rho"]
    row_level = selective_action._tie_aware_spearman(
        np.asarray(risks), np.asarray(drifts)
    )[0]

    assert reported == pytest.approx(1.0)
    assert row_level is not None and row_level < -0.5
    assert reported * row_level < 0.0


@pytest.mark.parametrize(
    ("risks", "outcomes", "reason"),
    (
        ([0.0, 1.0], [0.0, 1.0], "fewer_than_3_pairwise_complete_groups"),
        ([1.0, 1.0, 1.0], [0.0, 1.0, 2.0], "constant_group_mean_signal"),
        ([0.0, 1.0, 2.0], [1.0, 1.0, 1.0], "constant_group_mean_outcome"),
    ),
)
def test_observed_association_undefined_reasons(
    risks: list[float], outcomes: list[float], reason: str
) -> None:
    result = _association_bootstrap_from_group_pairs(
        risks,
        outcomes,
        replicates=20,
        minimum_defined_bootstrap_fraction=0.50,
        seed_sequence=np.random.SeedSequence([7]),
    )

    assert result.spearman_rho is None
    assert result.undefined_reason == reason
    assert result.ci_lower is None
    assert result.ci_upper is None
    assert result.ci_undefined_reason == (
        "fewer than the minimum required fraction of defined bootstrap replicates"
    )


def test_association_bootstrap_counts_undefined_and_enforces_fraction_guard() -> None:
    risks = np.asarray([0.0, 0.0, 1.0, 2.0])
    outcomes = np.asarray([0.0, 1.0, 1.0, 1.0])
    sampled = np.asarray(
        [
            [0, 1, 0, 1],
            [1, 2, 3, 2],
            [0, 1, 2, 3],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )

    result = _association_bootstrap_from_group_pairs(
        risks,
        outcomes,
        replicates=4,
        minimum_defined_bootstrap_fraction=0.50,
        resample_indices=sampled,
    )

    assert result.bootstrap_defined_replicates == 1
    assert result.bootstrap_undefined_replicates == 3
    assert result.minimum_defined_bootstrap_replicates == 2
    assert result.ci_lower is None
    assert result.ci_upper is None
    assert result.undefined_reason == (
        "fewer than the minimum required fraction of defined bootstrap replicates"
    )
    assert "constant_group_mean_signal" in result._replicate_undefined_reasons
    assert "constant_group_mean_outcome" in result._replicate_undefined_reasons
    assert all(value is None or value != 0.0 for value in result._replicate_values)


def test_association_bootstrap_emits_ci_when_defined_fraction_is_adequate() -> None:
    sampled = np.asarray(
        [
            [0, 1, 2, 3],
            [3, 2, 1, 0],
            [0, 0, 1, 2],
            [0, 1, 1, 3],
        ],
        dtype=np.int64,
    )
    result = _association_bootstrap_from_group_pairs(
        [0.0, 1.0, 2.0, 3.0],
        [0.0, 1.0, 2.0, 3.0],
        replicates=4,
        minimum_defined_bootstrap_fraction=0.50,
        resample_indices=sampled,
    )

    assert result.bootstrap_defined_replicates == 4
    assert result.minimum_defined_bootstrap_replicates == 2
    assert result.ci_lower == pytest.approx(1.0)
    assert result.ci_upper == pytest.approx(1.0)
    assert result.undefined_reason is None


def test_affine_risk_percentile_negation_uses_shared_group_resamples() -> None:
    evidence = np.asarray([0.1, 0.1, 0.4, 0.7, 0.7, 0.9])
    risk = 1.0 - evidence
    drift = np.asarray([1.0, 2.0, 2.0, 4.0, 5.0, 6.0])
    shared_indices = np.random.default_rng(19).integers(
        0, len(evidence), size=(500, len(evidence))
    )

    evidence_result = _association_bootstrap_from_group_pairs(
        evidence,
        drift,
        replicates=500,
        minimum_defined_bootstrap_fraction=0.50,
        resample_indices=shared_indices,
    )
    risk_result = _association_bootstrap_from_group_pairs(
        risk,
        drift,
        replicates=500,
        minimum_defined_bootstrap_fraction=0.50,
        resample_indices=shared_indices,
    )

    assert risk_result.spearman_rho == pytest.approx(-evidence_result.spearman_rho)
    for evidence_rho, risk_rho in zip(
        evidence_result._replicate_values,
        risk_result._replicate_values,
        strict=True,
    ):
        assert (evidence_rho is None) == (risk_rho is None)
        if evidence_rho is not None and risk_rho is not None:
            assert risk_rho == pytest.approx(-evidence_rho)
    assert risk_result.ci_lower == pytest.approx(-evidence_result.ci_upper)
    assert risk_result.ci_upper == pytest.approx(-evidence_result.ci_lower)


def test_one_and_only_headline_is_geometry_100_minus_070() -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        [0.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 14.0],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0],
    )

    analysis = _small_analysis(table)
    headline = analysis["headline_selective_contrast"]
    geometry_rows = {
        row["requested_coverage"]: row
        for row in analysis["selective_action_curve"]
        if row["score"] == "geometry_risk"
    }
    expected = (
        geometry_rows[1.0]["group_balanced_mean_abs_drift"]
        - geometry_rows[0.7]["group_balanced_mean_abs_drift"]
    )

    assert headline["score"] == "geometry_risk"
    assert headline["requested_coverage_reference"] == 1.0
    assert headline["requested_coverage_selective"] == 0.7
    assert headline["headline_selective_improvement_px"] == pytest.approx(expected)
    assert headline["interpretation"] == DEVELOPMENT_REUSE_INTERPRETATION
    assert "p_value" not in headline
    assert len(analysis["selective_action_curve"]) == 12


def test_headline_group_cluster_bootstrap_ci_is_deterministic_percentile() -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        [0.0, 4.0, 2.0, 8.0, 5.0, 12.0, 1.0, 15.0],
        [0.0, 0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5],
        [0.2, 0.3, 0.8, 0.9, 0.1, 0.4, 0.5, 0.7],
    )

    first = group_cluster_bootstrap(table, replicates=250, seed=BOOTSTRAP_SEED)
    second = group_cluster_bootstrap(table, replicates=250, seed=BOOTSTRAP_SEED)
    expected_low, expected_high = np.percentile(first.headline_values, [2.5, 97.5])

    assert np.array_equal(first.headline_values, second.headline_values)
    assert first.headline_ci_low == pytest.approx(expected_low)
    assert first.headline_ci_high == pytest.approx(expected_high)
    assert first.headline_ci_low <= first.headline_ci_high


def test_bootstrap_recomputes_both_headline_thresholds_and_never_calls_random_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "c"],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    original_selector = selective_action.select_by_global_risk_threshold
    calls: list[float] = []

    def recording_selector(scores: np.ndarray, coverage: float) -> Any:
        calls.append(coverage)
        return original_selector(scores, coverage)

    def forbidden_random_reference(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("random reference must not be nested in bootstrap")

    monkeypatch.setattr(
        selective_action, "select_by_global_risk_threshold", recording_selector
    )
    monkeypatch.setattr(
        selective_action, "random_reference_summaries", forbidden_random_reference
    )

    group_cluster_bootstrap(
        table,
        replicates=7,
        seed=0,
        requested_coverages=(1.0, 0.7),
    )

    assert calls.count(1.0) == 7 * len(SCORE_NAMES)
    assert calls.count(0.7) == 7 * len(SCORE_NAMES)


def test_headline_is_elementwise_difference_from_same_bootstrap_replicate() -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        [0.0, 9.0, 2.0, 7.0, 1.0, 12.0, 4.0, 15.0],
        [0.0, 0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5],
    )

    result = group_cluster_bootstrap(table, replicates=300, seed=BOOTSTRAP_SEED)
    expected = (
        result._curve_values[("geometry_risk", 1.0)]
        - result._curve_values[("geometry_risk", 0.7)]
    )

    assert np.array_equal(result.headline_values, expected)


def test_duplicated_bootstrap_source_groups_remain_distinct_clusters() -> None:
    table = _table(
        ["a", "b", "b", "c", "c", "c"],
        [0.0, 10.0, 10.0, 100.0, 100.0, 100.0],
        [0.0, 0.2, 0.3, 0.5, 0.6, 0.7],
    )
    result = group_cluster_bootstrap(table, replicates=200, seed=BOOTSTRAP_SEED)
    source_group_means = np.asarray([0.0, 10.0, 100.0])

    demonstrated = False
    for replicate_index, positions in enumerate(result._sampled_group_positions):
        if (
            len(np.unique(positions)) == len(positions)
            or len(np.unique(positions)) == 1
        ):
            continue
        correct_distinct_copy_mean = float(np.mean(source_group_means[positions]))
        collapsed_bug_mean = float(np.mean(source_group_means[np.unique(positions)]))
        if math.isclose(correct_distinct_copy_mean, collapsed_bug_mean):
            continue
        production_value = result._curve_values[("geometry_risk", 1.0)][replicate_index]
        assert production_value == pytest.approx(correct_distinct_copy_mean)
        assert production_value != pytest.approx(collapsed_bug_mean)
        demonstrated = True
        break
    assert demonstrated, (
        "fixture must contain a duplicate-cluster adversarial replicate"
    )


def test_selective_curve_bootstrap_moves_complete_source_groups_together(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _table(
        ["a", "a", "b", "b", "b", "c"],
        [0.0, 1.0, 5.0, 6.0, 7.0, 20.0],
        [0.0, 0.1, 0.3, 0.4, 0.5, 0.9],
    )
    captured_indices: list[np.ndarray] = []
    original_take = TargetedObservationTable.take

    def recording_take(
        self: TargetedObservationTable, indices: np.ndarray
    ) -> TargetedObservationTable:
        captured_indices.append(np.asarray(indices).copy())
        return original_take(self, indices)

    monkeypatch.setattr(TargetedObservationTable, "take", recording_take)

    group_cluster_bootstrap(table, replicates=15, seed=BOOTSTRAP_SEED)

    assert len(captured_indices) == 15
    for sampled_indices in captured_indices:
        row_multiplicities = np.bincount(
            sampled_indices, minlength=table.observation_count
        )
        for group_id in ("a", "b", "c"):
            group_positions = np.flatnonzero(table.group_ids == group_id)
            assert len(set(row_multiplicities[group_positions].tolist())) == 1


def test_association_bootstrap_depends_only_on_whole_group_pairs() -> None:
    compact = _table(
        ["a", "b", "c", "d"],
        [1.0, 3.0, 6.0, 10.0],
        [0.1, 0.3, 0.6, 0.9],
    )
    expanded = _table(
        ["a", "a", "a", "b", "b", "c", "c", "c", "c", "d", "d"],
        [0.0, 1.0, 2.0, 2.0, 4.0, 3.0, 5.0, 7.0, 9.0, 8.0, 12.0],
        [0.0, 0.1, 0.2, 0.2, 0.4, 0.3, 0.5, 0.7, 0.9, 0.8, 1.0],
    )

    compact_result = association_summary_rows(
        compact,
        cohort_role="independent_validation",
        replicates=250,
    )
    expanded_result = association_summary_rows(
        expanded,
        cohort_role="independent_validation",
        replicates=250,
    )

    assert compact_result == expanded_result


def test_common_curve_reuses_primary_masks_and_100_percent_invariant() -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        [0.0, 2.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0],
        [0.0, 0.1, 0.2, 0.9, 0.3, 0.8, 0.7, 0.6],
        [0.0, 0.9, 0.1, 0.8, 0.2, 0.7, 0.6, 0.5],
    )

    analysis = _small_analysis(table)
    assert analysis["common_retention_groups"] == frozenset({"a", "b", "c"})
    geometry_rows = {
        row["requested_coverage"]: row
        for row in analysis["selective_action_curve"]
        if row["score"] == "geometry_risk"
    }

    primary_half_mask = select_by_global_risk_threshold(
        table.scores["geometry_risk"], 0.5
    ).retained_mask
    common_primary_mask = primary_half_mask & np.isin(table.group_ids, ("a", "b", "c"))
    expected_half = np.mean(
        [
            np.mean(
                table.stability_abs[common_primary_mask & (table.group_ids == group)]
            )
            for group in ("a", "b", "c")
        ]
    )
    expected_full = np.mean([1.0, 15.0, 35.0])

    assert geometry_rows[0.5]["common_group_balanced_mean_abs_drift"] == pytest.approx(
        expected_half
    )
    assert geometry_rows[1.0]["common_group_balanced_mean_abs_drift"] == pytest.approx(
        expected_full
    )
    assert geometry_rows[0.5]["common_retention_group_count"] == 3


def test_common_group_analysis_is_undefined_below_three_groups() -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "c"],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.1, 0.2, 0.8, 0.9, 1.0],
        [0.0, 0.9, 0.1, 0.8, 0.7, 0.6],
    )

    analysis = _small_analysis(table)

    assert len(analysis["common_retention_groups"]) < 3
    assert all(
        row["common_group_balanced_mean_abs_drift"] is None
        for row in analysis["selective_action_curve"]
    )


def test_common_group_reference_is_frozen_grid_minimum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    assert COMMON_GROUP_REFERENCE_COVERAGE == min(REQUESTED_COVERAGES) == 0.50
    _validate_common_group_reference_coverage()

    monkeypatch.setattr(
        selective_action,
        "REQUESTED_COVERAGES",
        (*REQUESTED_COVERAGES, 0.40),
    )
    with pytest.raises(RuntimeError, match="must remain the frozen 0.50 minimum"):
        _validate_common_group_reference_coverage()


def test_random_reference_reports_represented_groups_on_original_table() -> None:
    table = _table(
        ["a", "b", "c", "d", "e", "f"],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    selection = select_by_global_risk_threshold(table.scores["geometry_risk"], 0.5)
    key = ("geometry_risk", 0.5)

    summary = random_reference_summaries(
        table,
        {key: selection},
        draws=100,
        base_seed=RANDOM_REFERENCE_BASE_SEED,
        score_names=("geometry_risk",),
        requested_coverages=(0.5,),
    )[key]

    assert selection.retained_count == 3
    assert summary["random_reference_represented_groups_mean"] == 3.0
    assert summary["random_reference_p05"] <= summary["random_reference_mean"]
    assert summary["random_reference_mean"] <= summary["random_reference_p95"]


def test_random_reference_draws_exactly_k_unique_observations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "d"],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )
    selection = select_by_global_risk_threshold(table.scores["geometry_risk"], 0.5)
    key = ("geometry_risk", 0.5)
    original_default_rng = np.random.default_rng
    recorded: list[tuple[np.ndarray, int, bool]] = []

    class RecordingGenerator:
        def __init__(self, seed: Any) -> None:
            self._generator = original_default_rng(seed)

        def choice(self, values: np.ndarray, *, size: int, replace: bool) -> np.ndarray:
            selected = self._generator.choice(values, size=size, replace=replace)
            recorded.append((selected.copy(), size, replace))
            return selected

    monkeypatch.setattr(
        selective_action.np.random,
        "default_rng",
        lambda seed: RecordingGenerator(seed),
    )

    random_reference_summaries(
        table,
        {key: selection},
        draws=25,
        base_seed=0,
        score_names=("geometry_risk",),
        requested_coverages=(0.5,),
    )

    assert len(recorded) == 25
    assert all(size == selection.retained_count for _, size, _ in recorded)
    assert all(replace is False for _, _, replace in recorded)
    assert all(len(np.unique(indices)) == len(indices) for indices, _, _ in recorded)


def test_random_reference_is_invariant_to_score_and_coverage_iteration_order() -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "c", "d", "d"],
        [0.0, 1.0, 4.0, 5.0, 2.0, 8.0, 3.0, 10.0],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.7, 0.5, 0.3, 0.1, 0.6, 0.4, 0.2, 0.0],
    )
    selections = {
        (score, coverage): select_by_global_risk_threshold(
            table.scores[score], coverage
        )
        for score in SCORE_NAMES
        for coverage in REQUESTED_COVERAGES
    }

    forward = random_reference_summaries(
        table,
        selections,
        draws=120,
        base_seed=0,
    )
    reversed_order = random_reference_summaries(
        table,
        selections,
        draws=120,
        base_seed=0,
        score_names=tuple(reversed(SCORE_NAMES)),
        requested_coverages=tuple(reversed(REQUESTED_COVERAGES)),
    )

    assert forward == reversed_order


def test_exact_zero_drift_count_and_fraction_are_reported() -> None:
    table = _table(
        ["a", "b", "c", "d", "e", "f"],
        [0.0, 0.0, 1e-15, 2.0, 0.0, 3.0],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )

    analysis = _small_analysis(table)
    full = next(
        row
        for row in analysis["selective_action_curve"]
        if row["score"] == "geometry_risk" and row["requested_coverage"] == 1.0
    )

    assert full["retained_zero_drift_count"] == 3
    assert full["retained_zero_drift_fraction"] == 0.5
    assert full["maximum_retained_abs_drift"] == 3.0


def test_group_balanced_primary_differs_from_observation_weighted_metric() -> None:
    table = _table(
        ["a", "b", "b", "b"],
        [0.0, 10.0, 10.0, 10.0],
        [0.0, 0.2, 0.4, 0.6],
    )

    analysis = _small_analysis(table)
    full = next(
        row
        for row in analysis["selective_action_curve"]
        if row["score"] == "geometry_risk" and row["requested_coverage"] == 1.0
    )

    assert full["group_balanced_mean_abs_drift"] == pytest.approx(5.0)
    assert full["observation_weighted_mean_abs_drift"] == pytest.approx(7.5)
    assert (
        full["group_balanced_mean_abs_drift"]
        != full["observation_weighted_mean_abs_drift"]
    )
    assert full["total_observation_count"] == 4
    assert full["total_group_count"] == 2
    assert full["realized_group_coverage"] == 1.0


def test_group_drop_and_curve_bootstrap_diagnostics_are_reported() -> None:
    table = _table(
        ["a", "a", "b", "c"],
        [1.0, 2.0, 10.0, 20.0],
        [0.0, 0.1, 0.8, 0.9],
        [0.0, 0.1, 0.8, 0.9],
    )

    analysis = _small_analysis(table)
    half = next(
        row
        for row in analysis["selective_action_curve"]
        if row["score"] == "geometry_risk" and row["requested_coverage"] == 0.5
    )

    assert half["represented_group_count"] == 1
    assert half["total_group_count"] == 3
    assert half["realized_group_coverage"] == pytest.approx(1.0 / 3.0)
    for field in (
        "bootstrap_realized_coverage_median",
        "bootstrap_realized_coverage_q25",
        "bootstrap_realized_coverage_q75",
        "bootstrap_represented_groups_median",
    ):
        assert half[field] is not None
        assert math.isfinite(float(half[field]))
    assert (
        half["bootstrap_realized_coverage_q25"]
        <= half["bootstrap_realized_coverage_median"]
        <= half["bootstrap_realized_coverage_q75"]
    )


def test_risk_distribution_summary_has_fixed_rows_and_exact_tie_diagnostics() -> None:
    table = _table(
        ["a", "a", "b", "b", "c", "c"],
        [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
        [0.0, 0.0, 0.5, 0.5, 0.5, 1.0],
        [0.0, 0.1, 0.2, 0.3, 0.4, 0.5],
    )

    rows = risk_distribution_summary(table)

    assert [row["score_name"] for row in rows] == [
        "geometry_risk",
        "perception_risk",
    ]
    assert len(rows) == 2
    geometry = rows[0]
    assert geometry["n_distinct_risk_values"] == 3
    assert geometry["largest_tie_count"] == 3
    assert geometry["largest_tie_fraction"] == 0.5
    assert geometry["minimum_risk"] == 0.0
    assert geometry["maximum_risk"] == 1.0


def test_outcomes_do_not_change_selection_or_common_groups() -> None:
    groups = ["a", "a", "b", "b", "c", "c", "d", "d"]
    geometry = [0.0, 0.8, 0.1, 0.7, 0.2, 0.6, 0.3, 0.5]
    perception = [0.1, 0.9, 0.2, 0.8, 0.3, 0.7, 0.4, 0.6]
    first = _table(groups, [0.0] * 8, geometry, perception)
    second = _table(
        groups, [100.0, 1.0, 99.0, 2.0, 98.0, 3.0, 97.0, 4.0], geometry, perception
    )

    for score_name in SCORE_NAMES:
        for coverage in REQUESTED_COVERAGES:
            first_selection = select_by_global_risk_threshold(
                first.scores[score_name], coverage
            )
            second_selection = select_by_global_risk_threshold(
                second.scores[score_name], coverage
            )
            assert first_selection.risk_cutoff == second_selection.risk_cutoff
            assert np.array_equal(
                first_selection.retained_mask, second_selection.retained_mask
            )
    assert common_retention_groups(first) == common_retention_groups(second)


def test_frozen_configuration_has_no_tuning_or_integrated_scalar() -> None:
    configuration = frozen_m5_v1_configuration(cohort_role="development_reuse")
    parser_help = _build_argument_parser().format_help()

    assert configuration["requested_coverages"] == list(REQUESTED_COVERAGES)
    assert configuration["policy"]["target_count"].startswith("ceil(")
    assert configuration["selective_curve_bootstrap"]["replicates"] == 10_000
    assert configuration["selective_curve_bootstrap"]["seed"] == 0
    assert (
        configuration["selective_curve_bootstrap"]["random_reference_nested"] is False
    )
    assert configuration["association"]["bootstrap_replicates"] == 10_000
    assert configuration["association"]["minimum_defined_bootstrap_replicates"] == 5_000
    assert configuration["association"]["minimum_defined_bootstrap_fraction"] == 0.50
    assert configuration["association"]["bootstrap_seed_derivation"] == (
        "numpy SeedSequence([0, 101, score_code])"
    )
    assert configuration["selective_curve_bootstrap"]["seed_derivation"] == (
        "numpy default_rng(0) from the integer base seed"
    )
    assert (
        configuration["monte_carlo_stream_relationship"][
            "separate_deterministic_streams"
        ]
        is True
    )
    assert configuration["random_reference"]["draws"] == 10_000
    assert "SeedSequence" in configuration["random_reference"]["seed_derivation"]
    assert configuration["curve_summary_policy"]["integrated_scalar_reported"] is False
    assert configuration["cohort_interpretation"] == DEVELOPMENT_REUSE_INTERPRETATION
    assert "coverage" not in parser_help.replace("cohort", "")
    assert "bootstrap" not in parser_help
    assert "random-reference" not in parser_help
    assert "signal" not in parser_help


def test_independent_validation_wording_is_caller_declared_not_module_verified() -> (
    None
):
    table = _table(
        ["a", "b", "c"],
        [1.0, 2.0, 3.0],
        [0.1, 0.2, 0.3],
    )

    configuration = frozen_m5_v1_configuration(cohort_role="independent_validation")
    associations = association_summary_rows(
        table,
        cohort_role="independent_validation",
        replicates=30,
    )

    wording = configuration["cohort_interpretation"]
    assert "declared by the caller" in wording
    assert "not verified by this module" in wording
    assert all(row["interpretation"] == wording for row in associations)


def _write_synthetic_m5_v0(root: Path) -> Path:
    source = root / "m5_v0_source"
    source.mkdir()
    (source / "manifest.json").write_text(
        json.dumps({"profile": "m5_v0"}) + "\n", encoding="utf-8"
    )
    rows = []
    for group_index, group_id in enumerate(("a", "b", "c", "d")):
        for observation_index in range(3):
            rows.append(
                _source_row(
                    group_id,
                    float(group_index + observation_index),
                    geometry_risk=(group_index * 3 + observation_index) / 20.0,
                    perception_risk=(12 - group_index * 3 - observation_index) / 20.0,
                )
            )
    with (source / "per_observation_signals.csv").open(
        "w", encoding="utf-8", newline=""
    ) as output_file:
        writer = csv.DictWriter(
            output_file, fieldnames=M5_V0_OBSERVATION_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {field: row.get(field, "") for field in M5_V0_OBSERVATION_FIELDS}
            )
    return source


def test_full_synthetic_run_writes_sanitized_stamped_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_synthetic_m5_v0(tmp_path)
    output = tmp_path / "m5_v1_output"
    source_bytes = (source / "per_observation_signals.csv").read_bytes()
    monkeypatch.setattr(selective_action, "ASSOCIATION_BOOTSTRAP_REPLICATES", 40)
    monkeypatch.setattr(selective_action, "BOOTSTRAP_REPLICATES", 40)
    monkeypatch.setattr(selective_action, "RANDOM_REFERENCE_DRAWS", 40)
    figure_text: list[str] = []
    axis_text: list[str] = []
    scatter_point_counts: list[int] = []
    original_text = Figure.text
    original_axis_text = Axes.text
    original_scatter = Axes.scatter

    def recording_text(
        self: Figure, x: float, y: float, s: str, *args: Any, **kwargs: Any
    ) -> Any:
        figure_text.append(s)
        return original_text(self, x, y, s, *args, **kwargs)

    def recording_axis_text(
        self: Axes, x: float, y: float, s: str, *args: Any, **kwargs: Any
    ) -> Any:
        axis_text.append(s)
        return original_axis_text(self, x, y, s, *args, **kwargs)

    def recording_scatter(self: Axes, x: Any, y: Any, *args: Any, **kwargs: Any) -> Any:
        scatter_point_counts.append(len(x))
        return original_scatter(self, x, y, *args, **kwargs)

    monkeypatch.setattr(Figure, "text", recording_text)
    monkeypatch.setattr(Axes, "text", recording_axis_text)
    monkeypatch.setattr(Axes, "scatter", recording_scatter)

    result = run_selective_action_analysis(
        source,
        output,
        cohort_role="development_reuse",
    )

    assert result["manifest"]["profile"] == PROFILE_NAME
    assert result["summary"]["counts"]["targeted_observation_rows"] == 12
    assert (source / "per_observation_signals.csv").read_bytes() == source_bytes
    assert all((output / "plots" / name).is_file() for name in PLOT_FILENAMES)
    assert any(
        "method = v2b" in value
        and "region = attachment" in value
        and "n groups = 4" in value
        and "cohort role = development_reuse" in value
        for value in figure_text
    )
    assert figure_text.count(DEVELOPMENT_BANNER) == len(PLOT_FILENAMES)
    assert figure_text.count(DEVELOPMENT_COMPARATOR_NOTE) == 2
    assert figure_text.count(DEVELOPMENT_ASSOCIATION_NOTE) == 1
    assert any(
        "total targeted groups = 4" in value and "common retention groups" in value
        for value in figure_text
    )
    assert scatter_point_counts == [4]
    assert any("Group-first Spearman rho" in value for value in axis_text)
    with (output / "selective_action_curve.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        reader = csv.DictReader(input_file)
        curve_rows = list(reader)
    assert tuple(reader.fieldnames or ()) == CURVE_FIELDS
    assert len(curve_rows) == 12
    assert {float(row["requested_coverage"]) for row in curve_rows} == set(
        REQUESTED_COVERAGES
    )
    persisted_by_key = {
        (row["score"], float(row["requested_coverage"])): row for row in curve_rows
    }
    for row in result["analysis"]["selective_action_curve"]:
        persisted_row = persisted_by_key[(row["score"], row["requested_coverage"])]
        assert float(persisted_row["risk_cutoff"]) == row["risk_cutoff"]
    with (output / "association_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        association_reader = csv.DictReader(input_file)
        association_rows = list(association_reader)
    assert tuple(association_reader.fieldnames or ()) == ASSOCIATION_FIELDS
    assert [row["score_name"] for row in association_rows] == [
        "geometry_risk",
        "perception_risk",
    ]
    with (output / "risk_distribution_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        risk_reader = csv.DictReader(input_file)
        risk_rows = list(risk_reader)
    assert tuple(risk_reader.fieldnames or ()) == RISK_DISTRIBUTION_FIELDS
    assert [row["score_name"] for row in risk_rows] == [
        "geometry_risk",
        "perception_risk",
    ]
    with (output / "group_primary_summary.csv").open(
        "r", encoding="utf-8", newline=""
    ) as input_file:
        group_reader = csv.DictReader(input_file)
        group_rows = list(group_reader)
    assert tuple(group_reader.fieldnames or ()) == GROUP_PRIMARY_FIELDS
    assert len(group_rows) == 8
    assert (output / "input_provenance.json").is_file()
    persisted = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in output.iterdir()
        if path.is_file()
    )
    assert str(tmp_path.resolve()) not in persisted
    assert "p_value" not in persisted
    configuration = json.loads(
        (output / "configuration.json").read_text(encoding="utf-8")
    )
    assert (
        configuration["common_group_secondary_curve"][
            "thresholds_recomputed_after_group_restriction"
        ]
        is False
    )
    assert configuration["curve_summary_policy"]["integrated_scalar_reported"] is False
    with pytest.raises(FileExistsError):
        run_selective_action_analysis(
            source,
            output,
            cohort_role="development_reuse",
        )


def test_input_and_output_roots_must_be_separate(tmp_path: Path) -> None:
    source = _write_synthetic_m5_v0(tmp_path)

    with pytest.raises(ValueError, match="must be separate"):
        run_selective_action_analysis(
            source,
            source,
            cohort_role="development_reuse",
            overwrite=True,
        )


def test_identical_synthetic_outputs_and_atomic_overwrite_replacement(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _write_synthetic_m5_v0(tmp_path)
    first = tmp_path / "first_output"
    second = tmp_path / "second_output"
    monkeypatch.setattr(selective_action, "ASSOCIATION_BOOTSTRAP_REPLICATES", 30)
    monkeypatch.setattr(selective_action, "BOOTSTRAP_REPLICATES", 30)
    monkeypatch.setattr(selective_action, "RANDOM_REFERENCE_DRAWS", 30)

    run_selective_action_analysis(
        source,
        first,
        cohort_role="independent_validation",
    )
    run_selective_action_analysis(
        source,
        second,
        cohort_role="independent_validation",
    )

    def snapshot(root: Path) -> dict[str, bytes]:
        return {
            path.relative_to(root).as_posix(): path.read_bytes()
            for path in root.rglob("*")
            if path.is_file()
        }

    expected = snapshot(second)
    assert snapshot(first) == expected
    stale = first / "stale.txt"
    stale.write_text("must be removed by atomic replacement", encoding="utf-8")

    run_selective_action_analysis(
        source,
        first,
        cohort_role="independent_validation",
        overwrite=True,
    )

    assert not stale.exists()
    assert snapshot(first) == expected
