from __future__ import annotations

import numpy as np
import pandas as pd

from emergence_attribution.interventions import (
    _effect_job,
    classify_edge_interventions,
    detect_onset,
)
from emergence_attribution.temporal import TemporalEdge


def _config() -> dict:
    return {
        "bootstrap_repetitions": 30,
        "confidence_level": 0.95,
        "onset_detection_start": 0,
        "minimum_standardised_effect": 0.10,
        "onset_consecutive_steps": 4,
        "evaluation_start": 15,
        "terminal_window": 10,
    }


def test_paired_intervention_preserves_raw_and_standardised_effects() -> None:
    baseline = np.tile(np.linspace(0, 1, 40)[None, :, None], (8, 1, 1))
    baseline += np.arange(8)[:, None, None] * 0.01
    intervention = baseline + 0.2
    result = _effect_job(
        {
            "scenario": "toy",
            "parameter": "strength",
            "direction": "plus",
            "node_ids": ["node"],
            "scales": {"node": "micro"},
            "baseline": baseline,
            "intervention": intervention,
            "config": _config(),
            "paired": True,
            "seed": 44,
        }
    )
    summary = result["summaries"][0]
    assert np.isclose(summary["cumulative_effect_raw"], 0.2)
    assert summary["cumulative_effect_standardised"] != summary["cumulative_effect_raw"]
    assert summary["paired"] is True


def test_zero_baseline_sd_keeps_raw_effect_but_standardised_is_not_estimable() -> None:
    baseline = np.zeros((8, 40, 2), dtype=float)
    baseline[:, :, 1] = np.linspace(0, 1, 40)[None, :]
    baseline[:, :, 1] += np.arange(8)[:, None] * 0.01
    intervention = baseline + np.asarray([0.5, 0.2])[None, None, :]
    result = _effect_job(
        {
            "scenario": "toy",
            "parameter": "strength",
            "direction": "plus",
            "node_ids": ["constant", "variable"],
            "scales": {"constant": "micro", "variable": "meso"},
            "baseline": baseline,
            "intervention": intervention,
            "config": _config(),
            "paired": True,
            "seed": 45,
        }
    )
    constant = result["summaries"][0]
    assert constant["baseline_sd"] == 0.0
    assert np.isclose(constant["cumulative_effect_raw"], 0.5)
    assert np.isnan(constant["cumulative_effect_standardised"])
    assert np.isnan(constant["cumulative_effect"])
    assert np.isnan(constant["cumulative_ci_low_standardised"])
    assert np.isnan(constant["cumulative_ci_high_standardised"])
    assert np.isnan(constant["terminal_effect_standardised"])
    assert constant["significant"] is False
    assert constant["onset_time"] == -1
    assert constant["peak_time"] == -1
    constant_curves = [
        row for row in result["curves"] if row["node_id"] == "constant"
    ]
    assert all(np.isnan(row["mean_standardised"]) for row in constant_curves)
    assert all(np.isclose(row["mean_raw"], 0.5) for row in constant_curves)
    variable = result["summaries"][1]
    assert variable["baseline_sd"] > 0
    assert np.isfinite(variable["cumulative_effect_standardised"])


def test_zero_baseline_sd_classification_is_inconclusive_without_crashing() -> None:
    baseline = np.zeros((8, 40, 2), dtype=float)
    intervention = baseline + np.asarray([0.5, 0.2])[None, None, :]
    result = _effect_job(
        {
            "scenario": "toy",
            "parameter": "strength",
            "direction": "plus",
            "node_ids": ["constant", "target"],
            "scales": {"constant": "micro", "target": "meso"},
            "baseline": baseline,
            "intervention": intervention,
            "config": _config(),
            "paired": True,
            "seed": 46,
        }
    )
    effects = pd.DataFrame(result["summaries"])
    representation = {
        "indicators": [
            {
                "id": "constant",
                "scale": "micro",
                "branch_id": "branch_a",
                "parameter_associations": [
                    {"parameter": "strength", "relationship": "direct"}
                ],
            },
            {
                "id": "target",
                "scale": "meso",
                "branch_id": "branch_a",
                "parameter_associations": [],
            },
        ],
        "candidate_edges": [
            {
                "source": "constant",
                "target": "target",
                "branch_id": "branch_a",
            }
        ],
    }
    edge = TemporalEdge(
        source="constant",
        target="target",
        lag=1,
        beta=0.7,
        p_value=0.001,
        q_value=0.002,
        effect_direction="increase",
        support=0.9,
        lag_support=0.8,
        lag_std=0.1,
        branch_id="branch_a",
    )

    frame = classify_edge_interventions(
        "toy", [edge], effects, representation, lag_tolerance=2
    )
    plus = frame[frame["direction"] == "plus"].iloc[0]
    assert np.isnan(plus["source_effect"])
    assert plus["primary_class"] == "inconclusive"


def test_onset_detection_respects_start_and_consecutive_rule() -> None:
    values = np.asarray([0.2, 0.2, 0.0, 0.3, 0.3, 0.3, 0.3])
    significant = np.abs(values) >= 0.1
    assert detect_onset(values, significant, 0, 4) == 3


def test_empty_graph_classification_has_stable_schema() -> None:
    frame = classify_edge_interventions(
        "toy", [], pd.DataFrame(), {"indicators": []}, lag_tolerance=2
    )
    assert frame.empty
    assert {"scenario", "source", "target", "primary_class"}.issubset(frame.columns)
