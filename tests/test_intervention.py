from __future__ import annotations

import numpy as np
import pandas as pd

from emergence_attribution.interventions import (
    _effect_job,
    classify_edge_interventions,
    detect_onset,
)


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
