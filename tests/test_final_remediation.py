from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from emergence_attribution.dsl import DSLValidationError, expression_fields
from emergence_attribution.evaluation import GRAPH_METHODS
from emergence_attribution.mock_semantic import mock_generation
from emergence_attribution.pipeline import STAGE_ORDER, load_experiment_config
from emergence_attribution.predefined import predefined_representation
from emergence_attribution.provenance import RunContractError, source_manifest
from emergence_attribution.raw_schemas import (
    HIDDEN_REFERENCE_FIELD_NAMES,
    hidden_reference_schema,
    public_raw_schema,
)
from emergence_attribution.schemas import SemanticGeneration
from emergence_attribution.semantic import build_prompt, validate_generation
from emergence_attribution.simulation import (
    IndicatorCompilationTask,
    SimulationTask,
    build_simulation_tasks,
    execute_simulation_task,
    compile_indicator_task,
)
from emergence_attribution.simulators import run_scenario, run_scenario_with_hidden
from emergence_attribution.temporal import discover_point_graph_from_blocks, prepare_target_blocks
from emergence_attribution import rendering, temporal


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _small_schelling() -> dict:
    return {
        "num_steps": 5,
        "num_agents": 12,
        "grid_width": 4,
        "grid_height": 4,
    }


def test_public_schema_has_no_reference_derived_fields() -> None:
    expected = {
            "schelling": {
                "state_grid", "agent_id", "agent_group", "agent_position", "district_id", "local_similarity",
            "neighbour_count", "unhappy", "unhappy_count", "moved",
            "move_distance", "destination_similarity", "boundary_agent",
            "num_steps", "agent_count",
        },
            "deffuant": {
                "state_opinion", "network_edges", "partner_id", "interaction_distance",
                "interaction_accepted", "interaction_backfire", "interaction_rejected",
                "edge_rewired", "agent_shift", "sign_flip", "extreme_agent_count", "num_steps", "agent_count",
        },
    }
    for scenario, names in expected.items():
        actual = {item["field_name"] for item in public_raw_schema(scenario)}
        assert actual == names
        assert not (actual & HIDDEN_REFERENCE_FIELD_NAMES)


def test_hidden_schema_is_separate_and_minimal() -> None:
    assert {item["field_name"] for item in hidden_reference_schema("schelling")} == {
        "mechanism_channel"
    }


def test_hidden_reference_never_reaches_semantic_prompt() -> None:
    config = load_experiment_config(PROJECT_ROOT / "config" / "experiment.json")
    template = (PROJECT_ROOT / "config" / "semantic_prompt.txt").read_text(encoding="utf-8")
    system, user = build_prompt(
        "schelling", config["scenarios"]["schelling"], config["representation"], template
    )
    assert not any(name in system + user for name in HIDDEN_REFERENCE_FIELD_NAMES)


def test_simulator_returns_disjoint_public_and_hidden_payloads() -> None:
    parameters = {
        "tolerance": 0.55,
        "move_probability": 0.1,
        "destination_preference": 0.8,
    }
    public = run_scenario("schelling", 7, _small_schelling(), parameters)
    public2, hidden = run_scenario_with_hidden(
        "schelling", 7, _small_schelling(), parameters
    )
    assert set(public) == set(public2)
    assert not (set(public) & set(hidden))
    assert set(hidden) == {"mechanism_channel"}


def test_public_npz_never_contains_hidden_field(tmp_path: Path) -> None:
    raw = tmp_path / "data" / "raw.npz"
    hidden = tmp_path / "data" / "hidden.npz"
    task = SimulationTask(
        task_id="schelling:baseline:1", scenario="schelling", seed=1,
        condition="baseline", intervention_parameter="",
        intervention_direction="baseline", mechanism_variant="baseline",
        scenario_spec=_small_schelling(),
        parameters={"tolerance": 0.55, "move_probability": 0.1, "destination_preference": 0.8},
        raw_path=str(raw), hidden_path=str(hidden), phase="baseline", formal_run=False,
    )
    result = execute_simulation_task(task)
    assert result.status == "completed"
    with np.load(raw, allow_pickle=False) as archive:
        assert not (set(archive.files) & HIDDEN_REFERENCE_FIELD_NAMES)
    with np.load(hidden, allow_pickle=False) as archive:
        assert set(archive.files) == {"mechanism_channel"}


def test_formal_npz_without_checkpoint_fails_closed(tmp_path: Path) -> None:
    raw = tmp_path / "raw.npz"
    hidden = tmp_path / "hidden.npz"
    np.savez_compressed(raw, x=np.ones(2))
    np.savez_compressed(hidden, mechanism_channel=np.ones((2, 8)))
    task = SimulationTask(
        task_id="x", scenario="schelling", seed=1, condition="baseline",
        intervention_parameter="", intervention_direction="baseline",
        mechanism_variant="baseline", scenario_spec=_small_schelling(),
        parameters={"tolerance": 0.55, "move_probability": 0.1, "destination_preference": 0.8},
        raw_path=str(raw), hidden_path=str(hidden), phase="baseline", formal_run=True,
    )
    result = execute_simulation_task(task)
    assert result.status == "failed"
    assert "sidecar" in (result.error or "")


def test_simulation_phases_are_disjoint() -> None:
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    baseline = build_simulation_tasks(config, Path("raw"), phase="baseline")
    intervention = build_simulation_tasks(config, Path("raw"), phase="intervention")
    assert {task.condition for task in baseline} == {"baseline"}
    assert "baseline" not in {task.condition for task in intervention}
    assert len(baseline) == 4
    assert len(intervention) == 28


@pytest.mark.parametrize("scenario", ["schelling", "deffuant"])
def test_predefined_comparator_uses_public_fields_only(scenario: str) -> None:
    representation = predefined_representation(scenario)
    assert len(representation["indicators"]) == 28
    assert len(representation["candidate_edges"]) == 28
    fields = set().union(
        *(expression_fields(item["computation"]) for item in representation["indicators"])
    )
    assert not (fields & HIDDEN_REFERENCE_FIELD_NAMES)
    incoming = {item["id"]: 0 for item in representation["indicators"]}
    outgoing = dict(incoming)
    for edge in representation["candidate_edges"]:
        outgoing[edge["source"]] += 1
        incoming[edge["target"]] += 1
    for item in representation["indicators"]:
        if item["scale"] == "micro":
            assert outgoing[item["id"]] > 0
        elif item["scale"] == "meso":
            assert incoming[item["id"]] > 0 and outgoing[item["id"]] > 0
        else:
            assert incoming[item["id"]] > 0


def test_mock_generation_satisfies_strict_semantic_contract() -> None:
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    value = SemanticGeneration.model_validate(mock_generation("schelling"))
    result = validate_generation(
        value, "schelling", config["scenarios"]["schelling"], config["representation"]
    )
    assert result["valid"], result["errors"]
    assert all(result["direct_micro_parameter_sources"].values())


def test_point_graph_has_nan_stability() -> None:
    rng = np.random.default_rng(3)
    source = rng.normal(size=120)
    target = np.roll(source, 1) + rng.normal(0, 0.01, 120)
    import pandas as pd

    blocks = prepare_target_blocks(
        [pd.DataFrame({"source": source, "target": target})],
        [{"source": "source", "target": "target", "branch_id": "b", "expected_direction": "unknown"}],
        3,
    )
    graph = discover_point_graph_from_blocks(blocks, 0.10, 0.05)
    assert graph
    assert all(np.isnan(edge.support) and np.isnan(edge.lag_support) and np.isnan(edge.lag_std) for edge in graph)


def test_prospective_prediction_frozen_before_simulation() -> None:
    assert STAGE_ORDER[:5] == [
        "semantic", "baseline_simulation", "temporal",
        "intervention_simulation", "intervention",
    ]
    assert STAGE_ORDER.index("prospective") < STAGE_ORDER.index("robustness")


def test_source_manifest_excludes_non_scientific_output_trees(tmp_path: Path) -> None:
    (tmp_path / "source.py").write_text("x=1\n", encoding="utf-8")
    for family in ("dev_runs", "smoke_runs", "runs"):
        path = tmp_path / family / "x.py"
        path.parent.mkdir(parents=True)
        path.write_text("x=2\n", encoding="utf-8")
    assert set(source_manifest(tmp_path)) == {"source.py"}


def test_non_scientific_run_cannot_target_formal_tree(tmp_path: Path) -> None:
    from emergence_attribution.provenance import RunManager

    config = {
        "formal_run": False, "random_seeds": [], "scenarios": {},
        "representation": {}, "temporal": {}, "intervention": {},
        "robustness": {}, "render": {},
    }
    llm = {"api_key": "", "model": "mock"}
    with pytest.raises(RunContractError, match="must not"):
        RunManager.initialise(tmp_path, "dev", config, llm, resume=False)


def test_generated_indicator_cannot_use_hidden_reference_field() -> None:
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    payload = mock_generation("schelling")
    indicator = payload["representation"]["indicators"][0]
    indicator["computation"] = {"op": "select", "input": {"op": "field", "name": "mechanism_channel"}, "axis": "channel", "index": 0}
    indicator["source_fields"] = ["mechanism_channel"]
    value = SemanticGeneration.model_validate(payload)
    result = validate_generation(
        value, "schelling", config["scenarios"]["schelling"], config["representation"]
    )
    assert not result["valid"]
    assert any("unknown raw field" in error for error in result["errors"])


def test_illegal_temporal_aggregation_rejected() -> None:
    from emergence_attribution.dsl import validate_temporal_aggregation

    with pytest.raises(DSLValidationError):
        validate_temporal_aggregation({"op": "rolling_mean", "window": -2})


def test_each_parameter_has_direct_micro_source() -> None:
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    value = SemanticGeneration.model_validate(mock_generation("deffuant"))
    result = validate_generation(
        value, "deffuant", config["scenarios"]["deffuant"], config["representation"]
    )
    assert set(result["direct_micro_parameter_sources"]) == set(
        config["scenarios"]["deffuant"]["interventions"]
    )
    assert all(result["direct_micro_parameter_sources"].values())


def test_prediction_source_matches_parameter() -> None:
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    payload = mock_generation("schelling")
    prediction = payload["prospective_predictions"][0]
    wrong = next(
        item["id"] for item in payload["representation"]["indicators"]
        if item["scale"] == "micro" and item["id"] != prediction["source_indicator"]
    )
    prediction["source_indicator"] = wrong
    prediction["expected_temporal_order"][0] = wrong
    value = SemanticGeneration.model_validate(payload)
    result = validate_generation(
        value, "schelling", config["scenarios"]["schelling"], config["representation"]
    )
    assert any("source must be a direct micro association" in error for error in result["errors"])


def test_prediction_path_is_real_candidate_path() -> None:
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    payload = mock_generation("schelling")
    prediction = payload["prospective_predictions"][0]
    pair = tuple(prediction["validation_criteria"]["required_candidate_edges"][0].values())
    payload["representation"]["candidate_edges"] = [
        edge for edge in payload["representation"]["candidate_edges"]
        if (edge["source"], edge["target"]) != pair
    ]
    value = SemanticGeneration.model_validate(payload)
    result = validate_generation(
        value, "schelling", config["scenarios"]["schelling"], config["representation"]
    )
    assert any("ordered path is absent" in error for error in result["errors"])


def test_all_generated_nodes_participate_in_graph() -> None:
    representation = mock_generation("deffuant")["representation"]
    touched = {
        node for edge in representation["candidate_edges"]
        for node in (edge["source"], edge["target"])
    }
    assert touched == {item["id"] for item in representation["indicators"]}


def test_candidate_edge_minimum_and_maximum() -> None:
    config = load_experiment_config(PROJECT_ROOT / "config" / "experiment.json")
    count = len(mock_generation("schelling")["representation"]["candidate_edges"])
    assert config["representation"]["minimum_candidate_edges"] <= count
    assert count <= config["representation"]["maximum_candidate_edges"]


def test_unrestricted_and_full_use_same_bootstrap_contract(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[int, float, float, int, float]] = []

    def fake_bootstrap(frames, candidates, maximum_lag, parent_alpha, fdr_alpha,
                       repetitions, support, master_seed, seed_label, workers,
                       progress_callback=None):
        calls.append((maximum_lag, parent_alpha, fdr_alpha, repetitions, support))
        return [], {"edge_sets": [], "bootstrap_repetitions": repetitions,
                    "trajectory_count": len(frames), "support_threshold": support}

    monkeypatch.setattr(temporal, "discover_bootstrap_graph", fake_bootstrap)
    monkeypatch.setattr(temporal, "discover_vote_graph", lambda *args, **kwargs: [])
    monkeypatch.setattr(temporal, "discover_point_graph_from_blocks", lambda *args, **kwargs: [])
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    representation = mock_generation("schelling")["representation"]
    rows = []
    for seed in (1, 2):
        for time_index in range(8):
            row = {"scenario": "schelling", "condition": "baseline", "seed": seed, "time": time_index,
                   "intervention_parameter": "", "intervention_direction": "baseline", "mechanism_variant": "baseline"}
            row.update({item["id"]: float(time_index + seed) for item in representation["indicators"]})
            rows.append(row)
    import pandas as pd

    temporal.run_temporal_stage(
        config, tmp_path, {"schelling": representation}, pd.DataFrame(rows), 1
    )
    assert len(calls) == 2
    assert calls[0] == calls[1]


def test_single_trajectory_has_no_fake_stability() -> None:
    rng = np.random.default_rng(44)
    import pandas as pd

    source = rng.normal(size=140)
    target = np.roll(source, 2) + rng.normal(0, 0.01, 140)
    graph = discover_point_graph_from_blocks(
        prepare_target_blocks(
            [pd.DataFrame({"source": source, "target": target})],
            [{"source": "source", "target": "target", "branch_id": "b", "expected_direction": "unknown"}], 5,
        ), 0.10, 0.05,
    )
    assert graph and all(np.isnan(edge.support) for edge in graph)


def test_reference_hidden_files_never_enter_indicator_compilation(tmp_path: Path) -> None:
    path = tmp_path / "public.npz"
    np.savez_compressed(path, x=np.ones((4, 2)), y=np.ones(4))
    task = IndicatorCompilationTask(
        scenario="toy", seed=1, condition="baseline", intervention_parameter="",
        intervention_direction="baseline", mechanism_variant="baseline",
        raw_path=str(path), indicators=[{
            "id": "hidden", "computation": {"op": "field", "name": "mechanism_channel"},
            "temporal_aggregation": {"op": "identity"},
        }],
    )
    with pytest.raises(DSLValidationError, match="unknown raw field"):
        compile_indicator_task(task)


def test_render_config_has_no_scientific_hardcodes() -> None:
    source = inspect.getsource(rendering.render_all_figures)
    assert "0.88" not in source
    assert 'render_config["graph_error_edge_alpha"]' in source
    config = load_experiment_config(PROJECT_ROOT / "config" / "experiment.json")
    assert config["render"]["effect_curve_centre"] == "mean"
    assert config["render"]["effect_matrix_colour_range"] == "full_data_range"


def test_method_runtime_export(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    test_unrestricted_and_full_use_same_bootstrap_contract(tmp_path, monkeypatch)
    import pandas as pd

    runtime = pd.read_csv(tmp_path / "analysis" / "method_runtime.csv")
    assert set(runtime["method"]) == {
        "llm_semantic_proposal", "unrestricted_temporal_search",
        "single_trajectory", "trajectory_vote", "full_method",
    }
    assert set(runtime["runtime_scope"]) == {"semantic_generation", "temporal_analysis"}


def test_intervention_metrics_for_all_applicable_methods() -> None:
    assert set(GRAPH_METHODS) == {
        "unrestricted_temporal_search", "single_trajectory",
        "trajectory_vote", "full_method",
    }
    source = inspect.getsource(__import__(
        "emergence_attribution.interventions", fromlist=["run_intervention_stage"]
    ).run_intervention_stage)
    assert all(method in source for method in GRAPH_METHODS)
