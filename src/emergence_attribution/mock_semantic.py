"""Deterministic, network-free semantic generator for non-scientific tests."""

from __future__ import annotations

import json
from typing import Any, Callable

from .llm_client import LLMResponse
from .predefined import predefined_representation


def mock_generation(scenario: str) -> dict[str, Any]:
    representation = predefined_representation(scenario)
    edges = {(item["source"], item["target"]) for item in representation["candidate_edges"]}
    lookup = {item["id"]: item for item in representation["indicators"]}
    predictions = []
    for indicator in representation["indicators"]:
        for association in indicator["parameter_associations"]:
            if association["relationship"] != "direct" or indicator["scale"] != "micro":
                continue
            meso = sorted(target for source, target in edges if source == indicator["id"])[0]
            macro = sorted(target for source, target in edges if source == meso)[0]
            path = [indicator["id"], meso, macro]
            predictions.append(
                {
                    "prediction_id": f"pred_{scenario}_{association['parameter']}",
                    "phenomenon": representation["phenomenon"],
                    "parameter": association["parameter"],
                    "intervention_direction": "plus",
                    "source_indicator": indicator["id"],
                    "expected_source_direction": "increase",
                    "downstream_indicators": [meso, macro],
                    "expected_downstream_direction": ["increase", "increase"],
                    "expected_temporal_order": path,
                    "validation_criteria": {
                        "required_source_response": True,
                        "required_downstream_response": [True, True],
                        "required_temporal_order": True,
                        "required_candidate_edges": [
                            {"source": path[0], "target": path[1]},
                            {"source": path[1], "target": path[2]},
                        ],
                    },
                    "scientific_rationale": "Deterministic mock prediction used only for pipeline validation.",
                    "falsification_condition": "The required signed responses or temporal order are not observed.",
                }
            )
    return {"representation": representation, "prospective_predictions": predictions}


def mock_completion_provider(
    scenario: str, generation_index: int
) -> Callable[[str, str], LLMResponse]:
    payload = json.dumps(mock_generation(scenario), ensure_ascii=False)

    def complete(_system: str, _user: str) -> LLMResponse:
        return LLMResponse(
            text=payload,
            input_tokens=0,
            output_tokens=0,
            model=f"deterministic-mock-{generation_index}",
        )

    return complete

