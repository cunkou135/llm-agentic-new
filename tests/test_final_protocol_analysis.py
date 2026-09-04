from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from pandas.testing import assert_frame_equal

from emergence_attribution.final_protocol import (
    DOSE_EFFECT_COLUMNS,
    DOSE_LABELS,
    DOSE_SUMMARY_COLUMNS,
    HOLDOUT_MECHANISM_COLUMNS,
    HOLDOUT_PATH_COLUMNS,
    HOLDOUT_PROSPECTIVE_COLUMNS,
    NEGATIVE_CONTROL_COLUMNS,
    PATH_ATTENUATION_COLUMNS,
    circular_shift_parent_series,
    dose_response_effects,
    dose_response_summary,
    five_point_dose_levels,
    holdout_mechanism_confirmation,
    holdout_path_confirmation,
    holdout_prospective_confirmation,
    path_mechanism_attenuation,
    temporal_negative_control,
)
from emergence_attribution.temporal import TemporalEdge


def _representation() -> dict:
    return {
        "indicators": [
            {"id": "micro_a", "scale": "micro"},
            {"id": "meso_b", "scale": "meso"},
            {"id": "macro_c", "scale": "macro"},
        ],
        "candidate_paths": [
            {
                "path_id": "path_toy_01", "parameter": "theta",
                "intervention_direction": "plus", "micro_indicator": "micro_a",
                "meso_indicator": "meso_b", "macro_indicator": "macro_c",
                "expected_micro_response": "increase",
                "expected_meso_response": "increase",
                "expected_macro_response": "increase",
            }
        ],
        "candidate_edges": [
            {
                "source": "micro_a",
                "target": "meso_b",
                "expected_direction": "increase",
            },
            {
                "source": "meso_b",
                "target": "macro_c",
                "expected_direction": "increase",
            },
        ],
    }


def _frozen_path() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "toy",
                "path_id": "theta:plus:micro_a>meso_b>macro_c",
                "parameter": "theta",
                "direction": "plus",
                "source": "micro_a",
                "meso": "meso_b",
                "macro": "macro_c",
                "node_id": node,
                "scale": scale,
            }
            for node, scale in (
                ("micro_a", "micro"),
                ("meso_b", "meso"),
                ("macro_c", "macro"),
            )
        ]
    )


def _edge(
    source: str,
    target: str,
    hypothesis_group_id: str = "macro_outcome_macro_c",
) -> TemporalEdge:
    return TemporalEdge(
        source=source,
        target=target,
        lag=1,
        beta=0.8,
        p_value=0.001,
        q_value=0.002,
        effect_direction="increase",
        support=0.9,
        lag_support=0.8,
        lag_std=0.0,
        hypothesis_group_id=hypothesis_group_id,
    )


def test_five_point_doses_preserve_primary_extremes_and_add_only_midpoints() -> None:
    levels = five_point_dose_levels([0.2, 0.5, 0.9])
    assert tuple(levels) == DOSE_LABELS
    assert levels["minus"] == 0.2
    assert levels["baseline"] == 0.5
    assert levels["plus"] == 0.9
    assert levels["mid_minus"] == 0.35
    assert levels["mid_plus"] == 0.7


def test_dose_response_effects_and_summary_are_secondary_and_paired() -> None:
    rows = []
    normalized = {
        "minus": -1.0,
        "mid_minus": -0.5,
        "baseline": 0.0,
        "mid_plus": 0.5,
        "plus": 1.0,
    }
    for seed in (1, 2, 3, 4):
        for label in DOSE_LABELS:
            condition = "baseline" if label == "baseline" else f"theta_{label}"
            for time in range(8):
                baseline = 2.0 + 0.1 * seed + 0.2 * time
                effect = normalized[label] * (1.0 + 0.05 * seed)
                rows.append(
                    {
                        "scenario": "toy",
                        "seed": seed,
                        "condition": condition,
                        "time": time,
                        "micro_a": baseline + effect,
                    }
                )
    dataset = pd.DataFrame(rows)
    representations = {
        "toy": {
            "indicators": [
                {"id": "micro_a", "scale": "micro"}
            ],
            "candidate_edges": [],
        }
    }
    effects = dose_response_effects(
        dataset,
        representations,
        {"toy": {"interventions": {"theta": [0.0, 1.0, 2.0]}}},
        {"evaluation_start": 2, "bootstrap_repetitions": 40, "confidence_level": 0.95},
        master_seed=77,
    )
    assert list(effects.columns) == DOSE_EFFECT_COLUMNS
    assert set(effects["dose_label"]) == set(DOSE_LABELS)
    assert not effects["primary_support_input"].any()
    assert effects.loc[
        effects["dose_label"].isin(["mid_minus", "mid_plus"]), "is_mid_dose"
    ].all()
    summary = dose_response_summary(
        effects,
        bootstrap_repetitions=60,
        confidence_level=0.95,
        master_seed=77,
    )
    assert list(summary.columns) == DOSE_SUMMARY_COLUMNS
    row = summary.iloc[0]
    assert row["dose_response_slope"] > 0
    assert np.isclose(row["spearman_rho"], 1.0)
    assert row["monotonic_direction_score"] == 1.0
    assert row["adjacent_dose_consistency"] == 1.0
    assert row["slope_bootstrap_unit"] == "paired_seeds"
    assert not bool(row["primary_classification_changed"])


def test_holdout_path_confirmation_uses_shared_stage3_path_result() -> None:
    frozen = _frozen_path()
    before = frozen.copy(deep=True)
    holdout_paths = pd.DataFrame(
        [{
            "scenario": "toy",
            "path_id": "theta:plus:micro_a>meso_b>macro_c",
            "hypothesis_group_id": "macro_outcome_macro_c",
            "parameter": "theta", "direction": "plus",
            "micro": "micro_a", "meso": "meso_b", "macro": "macro_c",
            "micro_meso_class": "supported", "meso_macro_class": "supported",
            "path_classification": "supported",
        }]
    )
    result = holdout_path_confirmation(frozen, holdout_paths)
    assert list(result.columns) == HOLDOUT_PATH_COLUMNS
    assert result.iloc[0]["classification"] == "confirmed"
    assert bool(result.iloc[0]["primary_result_unchanged"])
    assert_frame_equal(frozen, before)


def test_holdout_prospective_confirmation_reuses_frozen_prediction_and_graph() -> None:
    prediction = {
        "prediction_id": "pred_1",
        "candidate_path_id": "path_toy_01",
        "prospective_priority": 0,
        "scientific_rationale": "Frozen before any numerical data are generated.",
        "falsification_condition": "wrong direction or order",
    }
    frozen = {"scenarios": {"toy": [copy.deepcopy(prediction)]}}
    effects = pd.DataFrame(
        [
            {
                "scenario": "toy",
                "parameter": "theta",
                "direction": "plus",
                "node_id": node,
                "significant": True,
                "effect_sign": 1,
                "onset_time": onset,
            }
            for node, onset in (("micro_a", 1), ("meso_b", 3), ("macro_c", 5))
        ]
    )
    result = holdout_prospective_confirmation(
        frozen,
        {("toy", "full_method"): [_edge("micro_a", "meso_b"), _edge("meso_b", "macro_c")]},
        effects,
        {"toy": _representation()},
    )
    assert list(result.columns) == HOLDOUT_PROSPECTIVE_COLUMNS
    assert result.iloc[0]["classification"] == "confirmed"
    assert frozen["scenarios"]["toy"][0] == prediction


def test_holdout_mechanism_confirmation_is_separate_from_primary() -> None:
    primary = pd.DataFrame(
        [
            {
                "scenario": "toy",
                "method": "full_method",
                "hypothesis_group_id": "macro_outcome_macro_c",
                "root_source": "micro_a",
                "source": "micro_a",
                "target": "meso_b",
                "parameter": "theta",
                "direction": "plus",
                "primary_class": "supported",
            }
        ]
    )
    holdout = primary.drop(columns="method").copy()
    before = primary.copy(deep=True)
    result = holdout_mechanism_confirmation(
        primary,
        holdout,
        holdout_baseline_graphs={"toy": [_edge("micro_a", "meso_b")]},
        holdout_mechanism_disabled_graphs={"toy": []},
    )
    assert list(result.columns) == HOLDOUT_MECHANISM_COLUMNS
    assert result.iloc[0]["classification"] == "confirmed"
    assert bool(result.iloc[0]["retained_in_holdout_baseline"])
    assert not bool(result.iloc[0]["retained_in_holdout_mechanism_disabled"])
    assert_frame_equal(primary, before)


def test_temporal_shift_is_deterministic_exceeds_lag_and_does_not_mutate() -> None:
    frames = [
        pd.DataFrame(
            {
                "seed": np.repeat(seed, 80),
                "micro_a": np.arange(80, dtype=float) + seed,
                "meso_b": np.arange(80, dtype=float) * 2,
            }
        )
        for seed in (11, 12)
    ]
    original = [frame.copy(deep=True) for frame in frames]
    candidates = [{"source": "micro_a", "target": "meso_b"}]
    shifted_a, records_a = circular_shift_parent_series(
        frames,
        candidates,
        scenario="toy",
        repetition=0,
        maximum_lag=5,
        minimum_shift=20,
        maximum_shift=30,
        master_seed=99,
    )
    shifted_b, records_b = circular_shift_parent_series(
        frames,
        candidates,
        scenario="toy",
        repetition=0,
        maximum_lag=5,
        minimum_shift=20,
        maximum_shift=30,
        master_seed=99,
    )
    assert_frame_equal(records_a, records_b)
    assert (records_a["shift"] > records_a["maximum_lag"]).all()
    for untouched, expected in zip(frames, original):
        assert_frame_equal(untouched, expected)
    for left, right in zip(shifted_a, shifted_b):
        assert_frame_equal(left, right)


def test_temporal_negative_control_calls_stage2_core_and_is_deterministic() -> None:
    frames = []
    for seed in range(4):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=90)
        meso = np.roll(x, 1) + rng.normal(scale=0.05, size=90)
        macro = np.roll(meso, 1) + rng.normal(scale=0.05, size=90)
        frames.append(
            pd.DataFrame(
                {
                    "seed": np.repeat(seed, 90),
                    "micro_a": x,
                    "meso_b": meso,
                    "macro_c": macro,
                }
            )
        )
    temporal = {
        "maximum_lag": 3,
        "parent_alpha": 0.10,
        "fdr_alpha": 0.05,
        "bootstrap_repetitions": 4,
        "support_threshold": 0.50,
    }
    control = {"repetitions": 2, "minimum_shift": 20, "maximum_shift": 25}
    result_a = temporal_negative_control(
        "toy",
        frames,
        _representation(),
        temporal,
        control,
        master_seed=123,
        workers=1,
        primary_retained_edges=[_edge("micro_a", "meso_b")],
    )
    result_b = temporal_negative_control(
        "toy",
        frames,
        _representation(),
        temporal,
        control,
        master_seed=123,
        workers=1,
        primary_retained_edges=[_edge("micro_a", "meso_b")],
    )
    assert list(result_a.columns) == NEGATIVE_CONTROL_COLUMNS
    assert_frame_equal(result_a, result_b)
    assert (result_a["minimum_applied_shift"] > result_a["maximum_lag"]).all()
    assert result_a["control_only"].all()


def test_path_mechanism_attenuation_uses_only_public_frozen_path() -> None:
    frozen = _frozen_path()
    baseline = pd.DataFrame(
        {
            "scenario": ["toy"] * 3,
            "node_id": ["micro_a", "meso_b", "macro_c"],
            "effect": [1.0, 2.0, 4.0],
        }
    )
    disabled = pd.DataFrame(
        {
            "scenario": ["toy"] * 3,
            "node_id": ["micro_a", "meso_b", "macro_c"],
            "effect": [0.8, 1.0, 1.0],
        }
    )
    result = path_mechanism_attenuation(
        frozen,
        baseline,
        disabled,
        baseline_graphs={"toy": [_edge("micro_a", "meso_b"), _edge("meso_b", "macro_c")]},
        disabled_graphs={"toy": [_edge("micro_a", "meso_b")]},
        mechanism_variants={"toy": "disable_public_mechanism"},
    )
    assert list(result.columns) == PATH_ATTENUATION_COLUMNS
    assert len(result) == 3
    assert not result["uses_hidden_truth"].any()
    macro = result[result["scale"] == "macro"].iloc[0]
    assert macro["attenuation_ratio"] == 0.75
    assert bool(macro["temporal_edge_retained_baseline"])
    assert not bool(macro["temporal_edge_retained_disabled"])


def test_secondary_confirmations_do_not_borrow_edges_across_hypothesis_groups() -> None:
    prediction = {
        "prediction_id": "pred_c", "candidate_path_id": "path_toy_01",
        "scientific_rationale": "frozen", "falsification_condition": "missing edge",
    }
    effects = pd.DataFrame(
        [
            {
                "scenario": "toy", "parameter": "theta", "direction": "plus",
                "node_id": node, "significant": True, "effect_sign": 1,
                "onset_time": onset,
            }
            for node, onset in (("micro_a", 1), ("meso_b", 2), ("macro_c", 3))
        ]
    )
    cross_group_graph = {
        "toy": [
            _edge("micro_a", "meso_b", "macro_outcome_macro_d"),
            _edge("meso_b", "macro_c", "macro_outcome_macro_c"),
        ]
    }
    prospective = holdout_prospective_confirmation(
        {"scenarios": {"toy": [prediction]}}, cross_group_graph, effects,
        {"toy": _representation()},
    )
    assert not bool(prospective.iloc[0]["path_temporally_qualified"])
    assert prospective.iloc[0]["classification"] == "failed_confirmation"

    primary = pd.DataFrame(
        [{
            "scenario": "toy", "method": "full_method",
            "hypothesis_group_id": "macro_outcome_macro_c",
            "root_source": "micro_a", "source": "micro_a", "target": "meso_b",
            "parameter": "theta", "direction": "plus", "primary_class": "supported",
        }]
    )
    holdout = primary.drop(columns="method").copy()
    holdout["hypothesis_group_id"] = "macro_outcome_macro_d"
    mechanism = holdout_mechanism_confirmation(
        primary, holdout, holdout_baseline_graphs=cross_group_graph,
    )
    assert mechanism.iloc[0]["hypothesis_group_id"] == "macro_outcome_macro_c"
    assert mechanism.iloc[0]["classification"] == "inconclusive"
    assert not bool(mechanism.iloc[0]["retained_in_holdout_baseline"])

    metrics = pd.DataFrame(
        {
            "scenario": ["toy"] * 3,
            "node_id": ["micro_a", "meso_b", "macro_c"],
            "effect": [1.0, 1.0, 1.0],
        }
    )
    attenuation = path_mechanism_attenuation(
        _frozen_path(), metrics, metrics, baseline_graphs=cross_group_graph,
        disabled_graphs=cross_group_graph,
        mechanism_variants={"toy": "disable_public_mechanism"},
    )
    meso = attenuation[attenuation["scale"] == "meso"].iloc[0]
    assert meso["hypothesis_group_id"] == "macro_outcome_macro_c"
    assert not bool(meso["temporal_edge_retained_baseline"])
    assert not bool(meso["temporal_edge_retained_disabled"])


def test_empty_results_preserve_all_public_schemas() -> None:
    assert list(
        dose_response_summary(
            pd.DataFrame(),
            bootstrap_repetitions=2,
            confidence_level=0.95,
            master_seed=1,
        ).columns
    ) == DOSE_SUMMARY_COLUMNS
    assert list(holdout_path_confirmation(pd.DataFrame(), pd.DataFrame()).columns) == HOLDOUT_PATH_COLUMNS
    assert list(
        holdout_prospective_confirmation({}, {}, pd.DataFrame()).columns
    ) == HOLDOUT_PROSPECTIVE_COLUMNS
    assert list(
        holdout_mechanism_confirmation(pd.DataFrame(), pd.DataFrame()).columns
    ) == HOLDOUT_MECHANISM_COLUMNS
    assert list(
        path_mechanism_attenuation(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            baseline_graphs={},
            disabled_graphs={},
            mechanism_variants={},
        ).columns
    ) == PATH_ATTENUATION_COLUMNS
