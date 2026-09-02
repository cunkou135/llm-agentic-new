from __future__ import annotations

from pathlib import Path

import pytest

from emergence_attribution.dsl import (
    GENUINE_MESO_OPERATORS,
    GLOBAL_STRUCTURE_OPERATORS,
    canonical_source_family_lineage,
    expression_primitive_families,
    global_structure_operators,
    is_genuine_meso_expression,
)
from emergence_attribution.mock_semantic import mock_generation
from emergence_attribution.pipeline import load_experiment_config
from emergence_attribution.raw_schemas import (
    HIDDEN_REFERENCE_FIELD_NAMES,
    raw_schema,
)
from emergence_attribution.schemas import SemanticGeneration
from emergence_attribution.semantic import validate_generation


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG = load_experiment_config(PROJECT_ROOT / "config" / "experiment.json")


def _field(name: str) -> dict:
    return {"op": "field", "name": name}


def _validate(payload: dict, scenario: str) -> dict:
    return validate_generation(
        SemanticGeneration.model_validate(payload),
        scenario,
        CONFIG["scenarios"][scenario],
        CONFIG["representation"],
    )


def _indicator(payload: dict, scale: str, index: int = 0) -> dict:
    return [
        item
        for item in payload["representation"]["indicators"]
        if item["scale"] == scale
    ][index]


def _replace_computation(indicator: dict, expression: dict) -> None:
    indicator["computation"] = expression
    indicator["source_fields"] = sorted(
        name
        for name in {
            node.get("name")
            for node in _walk(expression)
            if node.get("op") == "field"
        }
        if isinstance(name, str)
    )


def _walk(value: object):
    if isinstance(value, dict):
        yield value
        for nested in value.values():
            yield from _walk(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from _walk(nested)


@pytest.mark.parametrize(
    "expression",
    [
        {
            "op": "network_assortativity",
            "values": _field("state_opinion"),
            "edges": _field("network_edges"),
        },
        {
            "op": "network_component_count",
            "edges": _field("network_edges"),
            "node_count": _field("agent_count"),
        },
    ],
)
def test_whole_network_operator_cannot_be_meso(expression: dict) -> None:
    payload = mock_generation("deffuant")
    _replace_computation(_indicator(payload, "meso"), expression)
    result = _validate(payload, "deffuant")
    assert not result["valid"]
    assert any(
        "global_structure_operator_invalid_for_meso" in error
        for error in result["errors"]
    )


@pytest.mark.parametrize(
    "op", ["largest_component_fraction", "spatial_neighbor_similarity"]
)
def test_whole_grid_operator_cannot_be_micro(op: str) -> None:
    payload = mock_generation("schelling")
    _replace_computation(
        _indicator(payload, "micro"),
        {"op": op, "input": _field("state_grid")},
    )
    result = _validate(payload, "schelling")
    assert not result["valid"]
    assert any(
        "global_structure_operator_invalid_for_micro" in error
        for error in result["errors"]
    )


def test_global_structure_operators_are_macro_legal() -> None:
    for scenario in ("schelling", "deffuant"):
        payload = mock_generation(scenario)
        result = _validate(payload, scenario)
        assert result["valid"], result["errors"]
        macros = [
            item
            for item in payload["representation"]["indicators"]
            if item["scale"] == "macro"
        ]
        assert any(global_structure_operators(item["computation"]) for item in macros)
        assert not any(
            "global_structure_operator_invalid_for_macro" in error
            for error in result["errors"]
        )
    assert "network_assortativity" in GLOBAL_STRUCTURE_OPERATORS
    assert "group_reduce" in GENUINE_MESO_OPERATORS


@pytest.mark.parametrize(
    ("scenario", "macro_index", "expression"),
    [
        (
            "schelling",
            0,
            {"op": "largest_component_fraction", "input": _field("state_grid")},
        ),
        (
            "schelling",
            2,
            {"op": "connected_component_count", "input": _field("state_grid")},
        ),
        (
            "schelling",
            3,
            {"op": "spatial_neighbor_similarity", "input": _field("state_grid")},
        ),
        (
            "deffuant",
            2,
            {
                "op": "network_assortativity",
                "values": _field("state_opinion"),
                "edges": _field("network_edges"),
            },
        ),
        (
            "deffuant",
            0,
            {
                "op": "network_density",
                "edges": _field("network_edges"),
                "node_count": _field("agent_count"),
            },
        ),
        (
            "deffuant",
            0,
            {
                "op": "network_component_count",
                "edges": _field("network_edges"),
                "node_count": _field("agent_count"),
            },
        ),
        (
            "deffuant",
            0,
            {
                "op": "network_largest_component_fraction",
                "edges": _field("network_edges"),
                "node_count": _field("agent_count"),
            },
        ),
    ],
)
def test_each_global_operator_is_accepted_at_macro_scope(
    scenario: str, macro_index: int, expression: dict
) -> None:
    payload = mock_generation(scenario)
    _replace_computation(_indicator(payload, "macro", macro_index), expression)
    result = _validate(payload, scenario)
    assert result["valid"], result["errors"]


def test_district_group_variance_is_genuine_meso() -> None:
    expression = {
        "op": "variance",
        "input": {
            "op": "group_reduce",
            "values": _field("unhappy"),
            "groups": _field("district_id"),
            "axis": "agent",
            "reducer": "fraction",
        },
        "axis": "group",
    }
    assert is_genuine_meso_expression(expression)


def test_neighborhood_variance_is_genuine_meso() -> None:
    expression = {
        "op": "variance",
        "input": {
            "op": "network_neighborhood_reduce",
            "values": _field("interaction_accepted"),
            "edges": _field("network_edges"),
            "reducer": "fraction",
        },
        "axis": "agent",
    }
    assert is_genuine_meso_expression(expression)


def test_pure_neighborhood_mean_is_rejected_as_complete_path_meso() -> None:
    payload = mock_generation("deffuant")
    expression = {
        "op": "mean",
        "input": {
            "op": "network_neighborhood_reduce",
            "values": _field("state_opinion"),
            "edges": _field("network_edges"),
            "reducer": "mean",
        },
        "axis": "agent",
    }
    assert not is_genuine_meso_expression(expression)
    _replace_computation(_indicator(payload, "meso"), expression)
    result = _validate(payload, "deffuant")
    assert any("pure outer mean/sum is insufficient" in e for e in result["errors"])


def test_group_mean_without_heterogeneity_is_rejected() -> None:
    payload = mock_generation("schelling")
    expression = {
        "op": "mean",
        "input": {
            "op": "group_reduce",
            "values": _field("unhappy"),
            "groups": _field("district_id"),
            "axis": "agent",
            "reducer": "fraction",
        },
        "axis": "group",
    }
    assert not is_genuine_meso_expression(expression)
    _replace_computation(_indicator(payload, "meso"), expression)
    result = _validate(payload, "schelling")
    assert any("pure outer mean/sum is insufficient" in e for e in result["errors"])


def test_equivalent_fraction_and_logged_count_path_is_rejected() -> None:
    payload = mock_generation("schelling")
    macro = _indicator(payload, "macro", 0)
    _replace_computation(
        macro,
        {
            "op": "divide",
            "left": _field("unhappy_count"),
            "right": _field("agent_count"),
        },
    )
    result = _validate(payload, "schelling")
    assert not result["valid"]
    assert any("trivial_micro_macro_lineage" in e for e in result["errors"])


def test_time_difference_of_micro_primitive_path_is_rejected() -> None:
    payload = mock_generation("schelling")
    macro = _indicator(payload, "macro", 0)
    _replace_computation(
        macro,
        {
            "op": "time_difference",
            "input": {
                "op": "fraction",
                "input": _field("moved"),
                "axis": "agent",
            },
        },
    )
    result = _validate(payload, "schelling")
    assert not result["valid"]
    assert any("trivial_micro_macro_lineage" in e for e in result["errors"])


def test_genuinely_independent_macro_lineage_is_accepted() -> None:
    for scenario in ("schelling", "deffuant"):
        result = _validate(mock_generation(scenario), scenario)
        assert result["valid"], result["errors"]
        assert result["trivial_micro_macro_lineage_count"] == 0


def test_public_alias_fields_share_primitive_family_without_hidden_truth() -> None:
    schema = raw_schema("schelling")
    metadata = {item["field_name"]: item for item in schema}
    assert metadata["unhappy"]["primitive_family"] == "dissatisfaction_event"
    assert metadata["unhappy_count"]["primitive_family"] == "dissatisfaction_event"
    assert metadata["unhappy"]["statistic_role"] == "elementary_event"
    assert metadata["unhappy_count"]["statistic_role"] == "aggregate_count"
    assert not ({item["field_name"] for item in schema} & HIDDEN_REFERENCE_FIELD_NAMES)
    assert all("truth" not in item["primitive_family"] for item in schema)


def test_count_fraction_aliases_have_same_canonical_public_lineage() -> None:
    schema = raw_schema("schelling")
    event_rate = {
        "op": "fraction",
        "input": _field("unhappy"),
        "axis": "agent",
    }
    logged_rate = {
        "op": "safe_ratio",
        "left": _field("unhappy_count"),
        "right": _field("agent_count"),
    }
    identity = {"op": "identity"}
    assert canonical_source_family_lineage(event_rate, identity, schema) == (
        canonical_source_family_lineage(logged_rate, identity, schema)
    )
    assert expression_primitive_families(event_rate, schema) == {
        "dissatisfaction_event"
    }
