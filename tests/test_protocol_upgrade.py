from __future__ import annotations

import copy
import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emergence_attribution import interventions, semantic, temporal
from emergence_attribution.dsl import (
    DSLValidationError,
    canonical_computational_lineage,
    compute_indicator,
    execute_expression,
    is_trivial_cross_scale_transform,
)
from emergence_attribution.llm_client import load_llm_config
from emergence_attribution.interventions import PATH_TIMING_COLUMNS, path_timing_summary
from emergence_attribution.mock_semantic import mock_completion_provider, mock_generation
from emergence_attribution.pipeline import load_experiment_config, run_stage
from emergence_attribution.progress import ProgressReporter
from emergence_attribution.provenance import RunManager
from emergence_attribution.raw_schemas import (
    HIDDEN_REFERENCE_FIELD_NAMES,
    public_raw_schema,
    raw_schema,
)
from emergence_attribution.schemas import SemanticGeneration
from emergence_attribution.semantic import build_prompt, validate_generation
from emergence_attribution.simulators import (
    _neighbour_values,
    counter_rng,
    run_scenario_with_hidden,
    simulate_deffuant,
    simulate_schelling,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _dev_config() -> dict:
    return load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")


def _deffuant_spec() -> dict:
    return {
        "num_steps": 18,
        "num_agents": 20,
        "network_degree": 4,
        "network_rewire_probability": 0.08,
        "adaptive_rewiring_probability": 1.0,
        "rewiring_homophily_probability": 0.65,
    }


def _deffuant_parameters() -> dict[str, float]:
    return {
        "confidence_bound": 0.05,
        "assimilation_strength": 0.25,
        "backfire_threshold": 0.30,
        "backfire_strength": 0.02,
    }


def _payload_digest(payload: dict[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for name in sorted(payload):
        value = np.ascontiguousarray(payload[name])
        digest.update(name.encode("utf-8"))
        digest.update(value.dtype.str.encode("ascii"))
        digest.update(json.dumps(value.shape).encode("ascii"))
        digest.update(value.tobytes())
    return digest.hexdigest()


def _validation(payload: dict, scenario: str = "schelling") -> dict:
    config = _dev_config()
    return validate_generation(
        SemanticGeneration.model_validate(payload),
        scenario,
        config["scenarios"][scenario],
        config["representation"],
    )


def test_schelling_fixed_district_membership_matches_agent_positions() -> None:
    spec = {
        "num_steps": 6,
        "num_agents": 30,
        "grid_width": 7,
        "grid_height": 7,
        "district_rows": 3,
        "district_columns": 3,
    }
    raw = simulate_schelling(
        29,
        spec,
        {
            "tolerance": 0.55,
            "move_probability": 0.7,
            "destination_preference": 0.8,
        },
    )
    rows = raw["agent_position"][:, :, 0]
    columns = raw["agent_position"][:, :, 1]
    expected = (rows * 3 // 7) * 3 + columns * 3 // 7
    np.testing.assert_array_equal(raw["district_id"], expected)
    np.testing.assert_array_equal(raw["agent_id"], np.arange(spec["num_agents"]))
    assert np.any(raw["district_id"][1:] != raw["district_id"][:-1])


def test_schelling_periodic_moore_neighbourhood_is_preserved() -> None:
    grid = np.full((3, 3), -1, dtype=np.int8)
    grid[0, 0] = 1
    grid[2, 2] = 1
    similarity, same, total = _neighbour_values(grid)
    assert total[0, 0] == 1
    assert same[0, 0] == 1
    assert similarity[0, 0] == 1.0


def test_schelling_district_is_public_primitive_without_hidden_truth() -> None:
    fields = {item["field_name"] for item in public_raw_schema("schelling")}
    assert {"agent_id", "agent_position", "agent_group", "district_id"} <= fields
    assert not fields.intersection(HIDDEN_REFERENCE_FIELD_NAMES)


def test_deffuant_dynamic_network_invariants_and_rewiring() -> None:
    raw = simulate_deffuant(41, _deffuant_spec(), _deffuant_parameters())
    edges = raw["network_edges"]
    assert edges.ndim == 3 and edges.shape[0] == _deffuant_spec()["num_steps"]
    assert raw["edge_rewired"].any()
    assert np.any(edges[1:] != edges[:-1])
    edge_count = edges.shape[1]
    for time_slice in edges:
        canonical = np.sort(time_slice, axis=1)
        assert len(canonical) == edge_count
        assert np.all(canonical[:, 0] != canonical[:, 1])
        assert len({tuple(edge) for edge in canonical}) == edge_count
        degrees = np.bincount(canonical.ravel(), minlength=_deffuant_spec()["num_agents"])
        assert np.all(degrees >= 1)


def test_deffuant_rewiring_and_trajectory_are_bitwise_reproducible() -> None:
    first = simulate_deffuant(73, _deffuant_spec(), _deffuant_parameters())
    second = simulate_deffuant(73, _deffuant_spec(), _deffuant_parameters())
    assert set(first) == set(second)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_disable_backfire_preserves_rewiring_stream_and_only_disables_repulsion() -> None:
    baseline = simulate_deffuant(52, _deffuant_spec(), _deffuant_parameters())
    disabled = simulate_deffuant(
        52, _deffuant_spec(), _deffuant_parameters(), "disable_backfire"
    )
    assert baseline["interaction_backfire"].any()
    assert not disabled["interaction_backfire"].any()
    assert disabled["edge_rewired"].any()
    # Before different opinion updates can alter later eligibility, both
    # conditions consume the same partner, trigger, and homophily streams.
    for name in ("state_opinion", "network_edges", "partner_id", "edge_rewired"):
        np.testing.assert_array_equal(baseline[name][0], disabled[name][0])


def test_matched_parameter_conditions_share_initial_state_and_random_keys() -> None:
    spec = _deffuant_spec()
    baseline_parameters = _deffuant_parameters()
    minus_parameters = {**baseline_parameters, "confidence_bound": 0.01}
    plus_parameters = {**baseline_parameters, "confidence_bound": 0.60}
    baseline = simulate_deffuant(81, spec, baseline_parameters)
    minus = simulate_deffuant(81, spec, minus_parameters)
    plus = simulate_deffuant(81, spec, plus_parameters)
    for candidate in (minus, plus):
        np.testing.assert_array_equal(
            baseline["state_opinion"][0], candidate["state_opinion"][0]
        )
        np.testing.assert_array_equal(
            baseline["network_edges"][0], candidate["network_edges"][0]
        )
        np.testing.assert_array_equal(
            baseline["partner_id"][0], candidate["partner_id"][0]
        )
    np.testing.assert_array_equal(
        counter_rng(81, 0, 43).random(spec["num_agents"]),
        counter_rng(81, 0, 43).random(spec["num_agents"]),
    )


def test_dynamic_network_dsl_operators_execute_and_validate_edges() -> None:
    edges = np.asarray(
        [
            [[0, 1], [1, 2], [2, 3]],
            [[0, 1], [0, 2], [0, 3]],
        ],
        dtype=np.int32,
    )
    values = np.asarray([[0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 2.0, 3.0]])
    raw = {
        "network_edges": edges,
        "state_opinion": values,
        "agent_count": np.asarray([4], dtype=np.int32),
    }
    neighborhood = {
        "op": "network_neighborhood_reduce",
        "values": {"op": "field", "name": "state_opinion"},
        "edges": {"op": "field", "name": "network_edges"},
        "reducer": "mean",
    }
    result = execute_expression(neighborhood, raw, raw_schema("deffuant"))
    np.testing.assert_allclose(result[0], [1.0, 1.0, 2.0, 2.0])
    np.testing.assert_allclose(result[1], [2.0, 0.0, 0.0, 0.0])
    for op, expected in (
        ("network_component_count", [1.0, 1.0]),
        ("network_largest_component_fraction", [1.0, 1.0]),
    ):
        expression = {
            "op": op,
            "edges": {"op": "field", "name": "network_edges"},
            "node_count": {"op": "field", "name": "agent_count"},
        }
        np.testing.assert_allclose(
            compute_indicator(expression, {"op": "identity"}, raw, raw_schema("deffuant")),
            expected,
        )
    invalid = raw | {"network_edges": edges.astype(float)}
    invalid["network_edges"][0, 1] = [1.0, 0.0]
    with pytest.raises(DSLValidationError, match="duplicate undirected edge"):
        execute_expression(neighborhood, invalid, raw_schema("deffuant"))


def test_intervention_only_absence_preserves_nan_without_weakening_baseline_gate() -> None:
    expression = {
        "op": "mean",
        "input": {
            "op": "where",
            "condition": {"op": "field", "name": "interaction_backfire"},
            "input": {"op": "field", "name": "interaction_distance"},
        },
        "axis": "agent",
    }
    raw = {
        "interaction_backfire": np.zeros((3, 4), dtype=bool),
        "interaction_distance": np.ones((3, 4), dtype=float),
    }
    with pytest.raises(DSLValidationError, match="no finite values"):
        compute_indicator(expression, {"op": "identity"}, raw, raw_schema("deffuant"))
    result = compute_indicator(
        expression,
        {"op": "identity"},
        raw,
        raw_schema("deffuant"),
        allow_all_nan=True,
    )
    assert np.isnan(result).all()


def test_zero_validated_paths_remain_a_schema_valid_empty_result() -> None:
    representation = mock_generation("schelling")["representation"]
    result = path_timing_summary(
        "schelling", [], pd.DataFrame(), representation
    )
    assert result.empty
    assert list(result.columns) == PATH_TIMING_COLUMNS


def test_scale_entity_scopes_and_real_structural_mesos_are_accepted() -> None:
    for scenario in ("schelling", "deffuant"):
        result = _validation(mock_generation(scenario), scenario)
        assert result["valid"], result["errors"]
    schelling = mock_generation("schelling")["representation"]["indicators"]
    assert all(
        item["entity_scope"] in {
            "neighborhood", "district", "community", "cluster", "local_domain"
        }
        for item in schelling
        if item["scale"] == "meso"
    )
    assert all(
        item["entity_scope"] == "whole_system"
        for item in schelling
        if item["scale"] == "macro"
    )


def test_rolling_micro_signal_cannot_masquerade_as_meso() -> None:
    payload = mock_generation("schelling")
    representation = payload["representation"]
    edge = representation["candidate_edges"][0]
    lookup = {item["id"]: item for item in representation["indicators"]}
    source, target = lookup[edge["source"]], lookup[edge["target"]]
    target["computation"] = {
        "op": "rolling_mean",
        "input": copy.deepcopy(source["computation"]),
        "window": 5,
    }
    target["source_fields"] = list(source["source_fields"])
    result = _validation(payload)
    assert not result["valid"]
    assert any("trivial cross-scale transform" in error for error in result["errors"])


def test_nested_rolling_windows_cannot_form_three_scale_path() -> None:
    base = {"op": "mean", "input": {"op": "field", "name": "x"}, "axis": "agent"}
    meso = {"op": "rolling_mean", "input": copy.deepcopy(base), "window": 5}
    macro = {"op": "rolling_mean", "input": copy.deepcopy(meso), "window": 20}
    identity = {"op": "identity"}
    assert is_trivial_cross_scale_transform(base, identity, meso, identity)
    assert is_trivial_cross_scale_transform(meso, identity, macro, identity)
    assert canonical_computational_lineage(base, identity) == (
        canonical_computational_lineage(macro, identity)
    )


def test_constant_rescaling_is_trivial_but_group_structure_is_not() -> None:
    source = {"op": "mean", "input": {"op": "field", "name": "x"}, "axis": "agent"}
    scaled = {
        "op": "add",
        "left": {
            "op": "multiply",
            "left": copy.deepcopy(source),
            "right": {"op": "constant", "value": 2.0},
        },
        "right": {"op": "constant", "value": 1.0},
    }
    grouped = {
        "op": "variance",
        "input": {
            "op": "group_reduce",
            "values": {"op": "field", "name": "x"},
            "groups": {"op": "field", "name": "group_id"},
            "axis": "agent",
            "reducer": "mean",
        },
        "axis": "group",
    }
    identity = {"op": "identity"}
    assert is_trivial_cross_scale_transform(source, identity, scaled, identity)
    clipped = {
        "op": "clip",
        "input": copy.deepcopy(source),
        "minimum": 0.0,
        "maximum": 1.0,
    }
    assert is_trivial_cross_scale_transform(source, identity, clipped, identity)
    assert not is_trivial_cross_scale_transform(source, identity, grouped, identity)


def test_macro_scope_must_be_whole_system() -> None:
    payload = mock_generation("schelling")
    macro = next(
        item
        for item in payload["representation"]["indicators"]
        if item["scale"] == "macro"
    )
    macro["entity_scope"] = "community"
    result = _validation(payload)
    assert any("invalid macro entity scope" in error for error in result["errors"])


def test_hidden_truth_isolation_from_prompt_and_full_discovery_modules() -> None:
    config = _dev_config()
    template = (PROJECT_ROOT / "config" / "semantic_prompt.txt").read_text(
        encoding="utf-8"
    )
    for scenario in ("schelling", "deffuant"):
        system, user = build_prompt(
            scenario,
            config["scenarios"][scenario],
            config["representation"],
            template,
        )
        combined = system + user
        assert "mechanism_channel" not in combined
        assert not any(name in combined for name in HIDDEN_REFERENCE_FIELD_NAMES)
    for module in (semantic, temporal, interventions):
        source = inspect.getsource(module)
        assert "reference_truth" not in source
    # Semantic generation may name a hidden field only in its explicit denylist.
    # The two Full Discovery numerical stages must not read that payload at all.
    for module in (temporal, interventions):
        assert "mechanism_channel" not in inspect.getsource(module)


def test_public_and_hidden_payload_hashes_are_stable() -> None:
    config = _dev_config()
    spec = config["scenarios"]["deffuant"]
    parameters = spec["baseline"]
    first_public, first_hidden = run_scenario_with_hidden(
        "deffuant", 901, spec, parameters
    )
    second_public, second_hidden = run_scenario_with_hidden(
        "deffuant", 901, spec, parameters
    )
    assert _payload_digest(first_public) == _payload_digest(second_public)
    assert _payload_digest(first_hidden) == _payload_digest(second_hidden)
    assert not set(first_public).intersection(first_hidden)


def test_semantic_and_prediction_freeze_timestamps_precede_baseline(
    tmp_path: Path,
) -> None:
    config = _dev_config()
    llm_path = PROJECT_ROOT / "config" / "llm_api.mock.json"
    manager = RunManager.initialise(
        tmp_path,
        "freeze_order",
        config,
        load_llm_config(llm_path, require_key=False),
        resume=False,
        output_family="dev_runs",
    )
    with ProgressReporter(manager.run_root, workers=1) as reporter:
        for stage in (
            "indicator_generation", "indicator_freeze", "path_generation",
            "semantic_freeze",
        ):
            run_stage(
                stage,
                manager,
                workers=1,
                reporter=reporter,
                prompt_template_path=PROJECT_ROOT / "config" / "semantic_prompt.txt",
                llm_config_path=llm_path,
                plot_repo=None,
                completion_provider=mock_completion_provider,
            )
        run_stage(
            "baseline_simulation",
            manager,
            workers=1,
            reporter=reporter,
            prompt_template_path=PROJECT_ROOT / "config" / "semantic_prompt.txt",
            llm_config_path=llm_path,
            plot_repo=None,
            completion_provider=mock_completion_provider,
        )
    manifest = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
    events = manifest["event_timestamps"]
    assert events["semantic_freeze_unix_time"] <= events["baseline_simulation_start_unix_time"]
    assert events["prediction_freeze_unix_time"] <= events["baseline_simulation_start_unix_time"]
