from __future__ import annotations

import copy
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emergence_attribution.controlled import controlled_intervention_recovery_metrics
from emergence_attribution.dsl import compute_indicator, expression_fields
from emergence_attribution.interventions import aggregate_edge_intervention_evidence
from emergence_attribution.mock_semantic import mock_generation, mock_path_generation
from emergence_attribution.pipeline import load_experiment_config
from emergence_attribution.raw_schemas import public_raw_schema
from emergence_attribution.reference_truth import (
    mechanism_target_for_variant,
    reference_processes,
    reference_relations,
)
from emergence_attribution.rendering import figure_4
from emergence_attribution import robustness
from emergence_attribution.robustness import (
    _corrupt_candidates_and_frames,
    _execute_robustness_bootstrap_jobs,
    _merge_metric_fields,
    _metric_row,
    run_data_efficiency,
)
from emergence_attribution.schemas import PathGeneration, SemanticGeneration
from emergence_attribution.semantic import validate_generation
from emergence_attribution.simulators import run_scenario_with_hidden
from emergence_attribution.temporal import TemporalEdge, representation_candidates, stable_seed


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _edge(source: str, target: str) -> TemporalEdge:
    return TemporalEdge(
        source=source,
        target=target,
        lag=1,
        beta=0.5,
        p_value=0.01,
        q_value=0.02,
        effect_direction="increase",
        support=0.8,
        lag_support=0.7,
        lag_std=0.1,
        hypothesis_group_id="macro_outcome_b",
    )


def _edge_classes(*classes: str) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scenario": "toy",
                "method": "full_method",
                "parameter": "theta",
                "direction": "plus" if index % 2 == 0 else "minus",
                "source": "a",
                "target": "b",
                "primary_class": value,
            }
            for index, value in enumerate(classes)
        ]
    )


def _mechanism_contract(scenario: str, variant: str, source_field: str) -> None:
    target = mechanism_target_for_variant(scenario, variant)
    targeted = [edge for edge in reference_relations(scenario) if edge.mechanism == target]
    assert len(targeted) == 2
    assert targeted[0].target.endswith("meso_3" if scenario == "schelling" else "meso_2")
    process = next(
        item for item in reference_processes(scenario) if item.process_id == targeted[0].source
    )
    assert expression_fields(process.computation) == {source_field}


def test_schelling_public_and_hidden_disabled_mechanism_are_aligned() -> None:
    _mechanism_contract(
        "schelling", "disable_homophilic_relocation", "destination_similarity"
    )
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    spec = config["scenarios"]["schelling"]
    public_baseline, hidden_baseline = run_scenario_with_hidden(
        "schelling", 901, spec, spec["baseline"], "baseline"
    )
    public_disabled, hidden_disabled = run_scenario_with_hidden(
        "schelling", 901, spec, spec["baseline"], spec["mechanism_variant"]
    )
    assert not np.array_equal(
        public_baseline["destination_similarity"],
        public_disabled["destination_similarity"],
    )
    assert not np.array_equal(
        hidden_baseline["mechanism_channel"][:, 3],
        hidden_disabled["mechanism_channel"][:, 3],
    )


def test_deffuant_public_and_hidden_disabled_mechanism_are_aligned() -> None:
    _mechanism_contract("deffuant", "disable_backfire", "interaction_backfire")
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    spec = config["scenarios"]["deffuant"]
    public_baseline, hidden_baseline = run_scenario_with_hidden(
        "deffuant", 901, spec, spec["baseline"], "baseline"
    )
    public_disabled, hidden_disabled = run_scenario_with_hidden(
        "deffuant", 901, spec, spec["baseline"], spec["mechanism_variant"]
    )
    assert public_baseline["interaction_backfire"].any()
    assert not public_disabled["interaction_backfire"].any()
    assert not np.array_equal(
        hidden_baseline["mechanism_channel"][:, 2],
        hidden_disabled["mechanism_channel"][:, 2],
    )


def test_representation_robustness_uses_mutated_candidate_count() -> None:
    representation = mock_generation("schelling")["representation"]
    original_count = len(representation_candidates(representation))
    expected_direction = {
        "delete_candidate_relation": -1,
        "cross_hypothesis_group_relation": 1,
        "irrelevant_indicator": 1,
    }
    for index, (operator, direction) in enumerate(expected_direction.items()):
        candidates, _ = _corrupt_candidates_and_frames(
            representation, [], operator, 0.25, 71 + index
        )
        metrics = _metric_row(
            "schelling", operator, [], representation,
            candidate_count_override=len(candidates),
        )
        assert metrics["candidate_edge_count"] == len(candidates)
        assert np.sign(len(candidates) - original_count) == direction


def test_representation_robustness_qualification_denominator_is_actual() -> None:
    representation = mock_generation("schelling")["representation"]
    candidates, _ = _corrupt_candidates_and_frames(
        representation, [], "irrelevant_indicator", 0.25, 72
    )
    metrics = _metric_row(
        "schelling", "irrelevant_indicator", [_edge("a", "b")], representation,
        candidate_count_override=len(candidates),
    )
    assert metrics["temporal_qualification_rate"] == pytest.approx(1 / len(candidates))


def test_mechanism_checks_do_not_export_hidden_reference_edges(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / "analysis").mkdir()
    monkeypatch.setattr(
        robustness,
        "trajectories",
        lambda *_args, **_kwargs: {1: pd.DataFrame({"placeholder": [0.0]})},
    )
    monkeypatch.setattr(
        robustness, "prepare_target_blocks", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(
        robustness,
        "discover_point_graph_from_blocks",
        lambda *_args, **_kwargs: [],
    )
    config = {
        "temporal": {"maximum_lag": 1, "parent_alpha": 0.1, "fdr_alpha": 0.05},
        "scenarios": {
            "schelling": {"mechanism_variant": "disable_homophilic_relocation"}
        },
    }

    result = robustness.run_mechanism_checks(
        config,
        tmp_path,
        {"schelling": mock_generation("schelling")["representation"]},
        pd.DataFrame(),
    )

    saved = pd.read_csv(tmp_path / "analysis" / "mechanism_disabled_checks.csv")
    assert "targeted_reference_edges" not in result.columns
    assert "targeted_reference_edges" not in saved.columns


def test_data_efficiency_n1_stability_is_not_estimable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    representation = {
        "scenario": "toy",
        "indicators": [
            {"id": "a", "scale": "micro"},
            {"id": "b", "scale": "meso"},
        ],
        "candidate_edges": [
            {"source": "a", "target": "b", "hypothesis_group_ids": ["macro_outcome_b"], "expected_direction": "unknown"}
        ],
    }
    baseline = pd.DataFrame(
        [
            {
                "scenario": "toy", "condition": "baseline", "seed": 1,
                "time": time, "intervention_parameter": "",
                "intervention_direction": "baseline", "mechanism_variant": "baseline",
                "a": float(time), "b": float(time + 1),
            }
            for time in range(8)
        ]
    )
    config = {
        "master_seed": 11,
        "evaluation": {
            "trajectory_counts": [1],
            "repeated_subsampling_repetitions": 1,
            "data_efficiency_bootstrap_repetitions": 5,
        },
        "temporal": {
            "maximum_lag": 2, "parent_alpha": 0.1, "fdr_alpha": 0.05,
            "support_threshold": 0.65, "vote_threshold": 0.5,
        },
    }

    def fake_execute(payloads, executor, progress_callback=None):
        assert all(item["point_only"] for item in payloads)
        return [
            {"job_index": item["job_index"], "graph": [], "summary": None, "vote": []}
            for item in payloads
        ]

    monkeypatch.setattr(
        "emergence_attribution.robustness._execute_robustness_bootstrap_jobs",
        fake_execute,
    )
    (tmp_path / "analysis").mkdir()
    frame = run_data_efficiency(
        config, tmp_path, {"toy": representation}, baseline, workers=1
    )
    assert not frame["stability_estimable"].any()
    for column in ("stability", "lag_support", "lag_std", "stability_ci_low", "stability_ci_high"):
        assert frame[column].isna().all()


def test_figure4_handles_n1_missing_stability(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    pd.DataFrame(
        [
            {
                "scenario": "toy", "method": method, "trajectory_count": 1,
                "repetition": 0, "temporal_qualification_rate": 0.25,
                "stability": np.nan,
                "temporal_qualification_rate_ci_low": np.nan,
                "temporal_qualification_rate_ci_high": np.nan,
                "stability_ci_low": np.nan, "stability_ci_high": np.nan,
            }
            for method in ("full_method", "trajectory_vote")
        ]
    ).to_csv(analysis / "data_efficiency_repeated_subsampling.csv", index=False)
    paths = figure_4(
        tmp_path, tmp_path / "figures", ["png"],
        {"png_dpi": 40, "paper_background": "#ffffff"},
    )
    assert len(paths) == 1 and paths[0].is_file() and paths[0].stat().st_size > 0


def _assortativity(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    expression = {
        "op": "network_assortativity",
        "values": {"op": "field", "name": "state_opinion"},
        "edges": {"op": "field", "name": "network_edges"},
    }
    return compute_indicator(
        expression, {"op": "identity"},
        {"state_opinion": values, "network_edges": edges},
        public_raw_schema("deffuant"),
    )


def test_undirected_network_assortativity_is_node_relabeling_invariant() -> None:
    values = np.asarray([[0.1, 0.5, -0.4, 0.9], [0.2, -0.2, 0.8, 0.4]])
    edges = np.asarray([[0, 1], [0, 3], [1, 2]], dtype=int)
    permutation = np.asarray([2, 0, 3, 1])  # old id -> new id
    relabelled_values = np.empty_like(values)
    relabelled_values[:, permutation] = values
    relabelled_edges = permutation[edges]
    assert np.allclose(
        _assortativity(values, edges),
        _assortativity(relabelled_values, relabelled_edges),
    )


def test_network_assortativity_constant_values_are_safe() -> None:
    result = _assortativity(
        np.ones((3, 4)), np.asarray([[0, 1], [1, 2], [2, 3]], dtype=int)
    )
    assert np.array_equal(result, np.zeros(3))
    assert np.isfinite(result).all()


def _generation_validation(payload: dict) -> dict:
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    value = SemanticGeneration.model_validate(payload)
    return validate_generation(
        value, "schelling", config["scenarios"]["schelling"], config["representation"]
    )


def test_prospective_prediction_must_reference_a_frozen_path() -> None:
    payload = mock_path_generation("schelling")
    payload["prospective_predictions"][0]["candidate_path_id"] = "missing_path"
    with pytest.raises(Exception, match="unknown path"):
        PathGeneration.model_validate(payload)


def test_prospective_prediction_cannot_embed_an_alternative_edge_list() -> None:
    payload = mock_path_generation("schelling")
    payload["prospective_predictions"][0]["required_candidate_edges"] = []
    with pytest.raises(Exception, match="Extra inputs are not permitted"):
        PathGeneration.model_validate(payload)


def _candidate_signature(candidates: list[dict[str, str]]) -> tuple[tuple[str, ...], ...]:
    return tuple(
        sorted((item["source"], item["target"], item["hypothesis_group_id"]) for item in candidates)
    )


def test_cross_group_robustness_repetitions_are_distinct_and_reproducible() -> None:
    representation = mock_generation("schelling")["representation"]
    seed0 = stable_seed(99, "cross_hypothesis_group_relation", 0)
    seed1 = stable_seed(99, "cross_hypothesis_group_relation", 1)
    first, _ = _corrupt_candidates_and_frames(
        representation, [], "cross_hypothesis_group_relation", 0.2, seed0
    )
    replay, _ = _corrupt_candidates_and_frames(
        representation, [], "cross_hypothesis_group_relation", 0.2, seed0
    )
    second, _ = _corrupt_candidates_and_frames(
        representation, [], "cross_hypothesis_group_relation", 0.2, seed1
    )
    assert _candidate_signature(first) == _candidate_signature(replay)
    assert _candidate_signature(first) != _candidate_signature(second)


@pytest.mark.parametrize(
    "operator",
    [
        "irrelevant_indicator",
        "redundant_semantic_indicator",
        "delete_candidate_relation",
        "wrong_hypothesis_group_assignment",
    ],
)
def test_other_robustness_operators_use_repetition_seed(operator: str) -> None:
    representation = mock_generation("schelling")["representation"]
    seed0 = stable_seed(101, operator, 0)
    seed1 = stable_seed(101, operator, 1)
    first, _ = _corrupt_candidates_and_frames(
        representation, [], operator, 0.35, seed0
    )
    replay, _ = _corrupt_candidates_and_frames(
        representation, [], operator, 0.35, seed0
    )
    second, _ = _corrupt_candidates_and_frames(
        representation, [], operator, 0.35, seed1
    )
    assert _candidate_signature(first) == _candidate_signature(replay)
    assert _candidate_signature(first) != _candidate_signature(second)


def test_edge_level_supported_plus_contradiction_is_contradicted() -> None:
    result = aggregate_edge_intervention_evidence(
        _edge_classes("supported", "directionally_contradicted")
    )
    assert result.iloc[0]["edge_class"] == "directionally_contradicted"


def test_edge_level_supported_plus_manipulation_failure_is_supported() -> None:
    result = aggregate_edge_intervention_evidence(
        _edge_classes("supported", "manipulation_failure")
    )
    assert result.iloc[0]["edge_class"] == "supported"


@pytest.mark.parametrize(
    ("classes", "expected"),
    [
        (("manipulation_failure", "manipulation_failure"), "manipulation_failure"),
        (("supported",), "supported"),
    ],
)
def test_edge_level_remaining_frozen_precedence_cases(
    classes: tuple[str, ...], expected: str
) -> None:
    result = aggregate_edge_intervention_evidence(_edge_classes(*classes))
    assert result.iloc[0]["edge_class"] == expected


def test_controlled_intervention_f1_uses_edge_level_aggregation() -> None:
    aggregated = aggregate_edge_intervention_evidence(
        _edge_classes("supported", "directionally_contradicted")
    )
    supported = {
        (str(row.source), str(row.target))
        for row in aggregated.itertuples() if row.edge_class == "supported"
    }
    metrics = controlled_intervention_recovery_metrics(
        {("a", "b")}, {("a", "b")}, supported
    )
    assert metrics["intervention_f1"] == 0.0


def _normalise_job_results(results: list[dict]) -> list[dict]:
    return [
        {
            "job_index": item["job_index"],
            "graph": [edge.__dict__ for edge in item["graph"]],
            "summary": item["summary"],
        }
        for item in results
    ]


def test_workers_1_equals_workers_n_after_pool_refactor() -> None:
    rng = np.random.default_rng(14)
    frames = []
    for _ in range(3):
        source = rng.normal(size=50)
        target = np.roll(source, 1) + rng.normal(0, 0.05, size=50)
        frames.append(pd.DataFrame({"a": source, "b": target}))
    base = {
        "frames": frames,
        "candidates": [
            {"source": "a", "target": "b", "hypothesis_group_id": "macro_outcome_b", "expected_direction": "unknown"}
        ],
        "maximum_lag": 2, "parent_alpha": 0.1, "fdr_alpha": 0.05,
        "bootstrap_repetitions": 4, "support_threshold": 0.5,
        "master_seed": 55, "seed_label": "pool-equivalence",
        "point_only": False, "include_vote": False,
    }
    payloads = [{**base, "job_index": index} for index in range(2)]
    sequential = _execute_robustness_bootstrap_jobs(payloads, None)
    with ProcessPoolExecutor(max_workers=2) as executor:
        parallel = _execute_robustness_bootstrap_jobs(payloads, executor)
    assert _normalise_job_results(sequential) == _normalise_job_results(parallel)


def test_no_duplicate_dict_key_metric_override() -> None:
    assert _merge_metric_fields({"operator": "x"}, {"candidate_edge_count": 3}) == {
        "operator": "x", "candidate_edge_count": 3,
    }
    with pytest.raises(ValueError, match="silently overwritten"):
        _merge_metric_fields(
            {"candidate_edge_count": 2}, {"candidate_edge_count": 3}
        )
