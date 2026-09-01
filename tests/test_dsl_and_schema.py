from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from pydantic import ValidationError

from emergence_attribution.dsl import (
    DSLValidationError,
    compute_indicator,
    grammar_description,
    validate_temporal_aggregation,
    validate_indicator_expression,
)
from emergence_attribution.pipeline import load_experiment_config
from emergence_attribution.raw_schemas import raw_schema
from emergence_attribution.schemas import SemanticGeneration, StructuredRepresentation
from emergence_attribution.semantic import build_prompt, validate_generation


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _valid_toy_generation() -> dict:
    def indicator(identifier: str, scale: str, expression: dict) -> dict:
        return {
            "id": identifier,
            "semantic_name": identifier.replace("_", " "),
            "scientific_definition": "A sufficiently detailed deterministic toy definition.",
            "phenomenon": "toy organisation",
            "scale": scale,
            "branch_id": "branch_0",
            "entities": "toy entities",
            "source_fields": ["x"] if identifier != "toy_macro" else ["y"],
            "computation": expression,
            "temporal_aggregation": {"op": "identity"},
            "parameter_associations": (
                [
                    {
                        "parameter": "strength",
                        "relationship": "direct",
                        "expected_indicator_direction": "increase",
                        "rationale": "The configured strength directly changes this toy quantity.",
                    }
                ]
                if scale == "micro"
                else []
            ),
            "scientific_rationale": "This toy indicator checks the structured software contract.",
        }

    return {
        "representation": {
            "scenario": "toy",
            "phenomenon": "toy organisation",
            "indicators": [
                indicator(
                    "toy_micro",
                    "micro",
                    {"op": "mean", "input": {"op": "field", "name": "x"}, "axis": "agent"},
                ),
                indicator(
                    "toy_meso",
                    "meso",
                    {"op": "std", "input": {"op": "field", "name": "x"}, "axis": "agent"},
                ),
                indicator("toy_macro", "macro", {"op": "field", "name": "y"}),
            ],
            "candidate_edges": [
                {
                    "source": "toy_micro",
                    "target": "toy_meso",
                    "expected_direction": "increase",
                    "rationale": "Toy lower-level variation can precede intermediate organisation.",
                },
                {
                    "source": "toy_meso",
                    "target": "toy_macro",
                    "expected_direction": "increase",
                    "rationale": "Toy intermediate organisation can precede the aggregate.",
                },
            ],
            "interpretation_boundary": "Temporal qualification and intervention evidence are evaluated later and separately.",
        },
        "prospective_predictions": [
            {
                "prediction_id": "toy_prediction",
                "phenomenon": "toy organisation",
                "parameter": "strength",
                "intervention_direction": "plus",
                "source_indicator": "toy_micro",
                "expected_source_direction": "increase",
                "downstream_indicators": ["toy_meso", "toy_macro"],
                "expected_downstream_direction": ["increase", "increase"],
                "expected_temporal_order": ["toy_micro", "toy_meso", "toy_macro"],
                "validation_criteria": {
                    "required_source_response": True,
                    "required_downstream_response": [True, True],
                    "required_temporal_order": True,
                    "required_candidate_edges": [
                        {"source": "toy_micro", "target": "toy_meso"},
                        {"source": "toy_meso", "target": "toy_macro"}
                    ]
                },
                "scientific_rationale": "The toy prediction exercises prospective validation wiring.",
                "falsification_condition": "The source is changed but downstream direction or order fails.",
            }
        ],
    }


def test_structured_schema_accepts_complete_generation() -> None:
    value = SemanticGeneration.model_validate(_valid_toy_generation())
    validation = validate_generation(
        value,
        "toy",
        {"interventions": {"strength": [0.0, 1.0, 2.0]}},
        {
            "budget": {"micro": 1, "meso": 1, "macro": 1},
            "required_branch_count": 1,
            "require_all_parameters_associated": True,
            "minimum_candidate_edges": 2,
            "maximum_candidate_edges": 2,
        },
    )
    assert validation["valid"], validation["errors"]


def test_candidate_edge_scale_constraint() -> None:
    value = _valid_toy_generation()["representation"]
    value["candidate_edges"][0]["target"] = "toy_macro"
    with pytest.raises(ValidationError, match="adjacent-scale"):
        StructuredRepresentation.model_validate(value)


def test_indicator_dsl_executes_safe_mean() -> None:
    expression = {"op": "mean", "input": {"op": "field", "name": "x"}, "axis": "agent"}
    raw = {"x": np.arange(12, dtype=float).reshape(4, 3), "y": np.arange(4)}
    values = compute_indicator(expression, {"op": "identity"}, raw, raw_schema("toy"))
    np.testing.assert_allclose(values, [1.0, 4.0, 7.0, 10.0])


def test_illegal_field_rejected() -> None:
    expression = {"op": "mean", "input": {"op": "field", "name": "unknown"}, "axis": "agent"}
    with pytest.raises(DSLValidationError, match="unknown raw field"):
        validate_indicator_expression(expression, raw_schema("toy"))


def test_illegal_operator_rejected() -> None:
    expression = {"op": "python_eval", "input": {"op": "field", "name": "x"}}
    with pytest.raises(DSLValidationError, match="illegal computation operator"):
        validate_indicator_expression(expression, raw_schema("toy"))


def test_prompt_contains_no_evaluation_leakage() -> None:
    config = load_experiment_config(PROJECT_ROOT / "config" / "experiment.json")
    template = (PROJECT_ROOT / "config" / "semantic_prompt.txt").read_text(encoding="utf-8")
    system, user = build_prompt(
        "schelling", config["scenarios"]["schelling"], config["representation"], template
    )
    combined = (system + user).lower()
    forbidden = [
        "reference_" + "branch",
        "truth_" + "lag",
        "edge_" + "f1",
        "baseline_" + "numerical_summary",
    ]
    assert not any(value in combined for value in forbidden)


def test_temporal_aggregation_contract_is_strict() -> None:
    validate_temporal_aggregation({"op": "rolling_mean", "window": 3})
    with pytest.raises(DSLValidationError, match="positive integer"):
        validate_temporal_aggregation({"op": "rolling_mean", "window": 0})
    with pytest.raises(DSLValidationError, match="unexpected"):
        validate_temporal_aggregation({"op": "identity", "window": 2})


def test_continuous_entropy_requires_binning() -> None:
    expression = {
        "op": "entropy",
        "input": {"op": "field", "name": "x"},
        "axis": "agent",
    }
    with pytest.raises(DSLValidationError, match="binned_entropy"):
        validate_indicator_expression(expression, raw_schema("toy"))


def test_grammar_documents_each_operator_contract() -> None:
    operators = grammar_description()["operators"]
    assert operators
    for contract in operators.values():
        assert set(contract) == {
            "required", "optional", "types", "axis_semantics", "output", "example"
        }
