"""Deterministic network-free responses for both semantic phases."""

from __future__ import annotations

import json
from typing import Any, Callable

from .dsl import is_trivial_cross_scale_transform, is_trivial_micro_macro_lineage
from .llm_client import LLMResponse
from .predefined import predefined_representation
from .raw_schemas import raw_schema
from .schemas import IndicatorGeneration
from .semantic import derive_candidate_edges, sha256_json


def mock_indicator_generation(scenario: str) -> dict[str, Any]:
    source = predefined_representation(scenario)
    return {
        "scenario": scenario,
        "phenomenon": f"NON_SCIENTIFIC deterministic {scenario} observables",
        "indicators": source["indicators"],
        "interpretation_boundary": (
            "These observables are hypotheses only; temporal qualification and "
            "intervention evidence are evaluated later."
        ),
    }


def mock_path_generation(scenario: str) -> dict[str, Any]:
    frozen = mock_indicator_generation(scenario)
    frozen_canonical = IndicatorGeneration.model_validate(frozen).model_dump(
        mode="json", exclude_none=True
    )
    lookup = {item["id"]: item for item in frozen["indicators"]}
    micro_by_parameter = {}
    for item in frozen["indicators"]:
        if item["scale"] != "micro":
            continue
        for association in item["parameter_associations"]:
            if association["relationship"] == "direct":
                micro_by_parameter.setdefault(association["parameter"], item)
    mesos = [item for item in frozen["indicators"] if item["scale"] == "meso"]
    macros = [item for item in frozen["indicators"] if item["scale"] == "macro"]
    schema = raw_schema(scenario)
    candidates = []
    for parameter, micro in sorted(micro_by_parameter.items()):
        for meso in mesos:
            for macro in macros:
                if is_trivial_cross_scale_transform(
                    micro["computation"], micro["temporal_aggregation"],
                    meso["computation"], meso["temporal_aggregation"],
                ):
                    continue
                if is_trivial_cross_scale_transform(
                    meso["computation"], meso["temporal_aggregation"],
                    macro["computation"], macro["temporal_aggregation"],
                ):
                    continue
                if is_trivial_micro_macro_lineage(
                    micro["computation"], micro["temporal_aggregation"],
                    macro["computation"], macro["temporal_aggregation"], schema,
                ):
                    continue
                candidates.append((parameter, micro, meso, macro))
    # Round-robin over parameters and macro endpoints, data-blind and deterministic.
    selected = []
    parameter_counts = {name: 0 for name in micro_by_parameter}
    macro_counts = {item["id"]: 0 for item in macros}
    while len(selected) < 18:
        progress = False
        for item in candidates:
            parameter, _micro, _meso, macro = item
            if item in selected:
                continue
            minimum_parameter = min(parameter_counts.values())
            minimum_macro = min(macro_counts.values())
            if parameter_counts[parameter] > minimum_parameter or macro_counts[macro["id"]] > minimum_macro:
                continue
            selected.append(item)
            parameter_counts[parameter] += 1
            macro_counts[macro["id"]] += 1
            progress = True
            if len(selected) == 18:
                break
        if not progress:
            for item in candidates:
                if item not in selected:
                    selected.append(item)
                    parameter_counts[item[0]] += 1
                    macro_counts[item[3]["id"]] += 1
                    progress = True
                    break
        if not progress:
            raise RuntimeError("mock generator cannot construct 18 valid frozen paths")
    paths = []
    for index, (parameter, micro, meso, macro) in enumerate(selected):
        paths.append(
            {
                "path_id": f"path_{scenario}_{index:02d}",
                "parameter": parameter,
                "intervention_direction": "plus",
                "micro_indicator": micro["id"],
                "meso_indicator": meso["id"],
                "macro_indicator": macro["id"],
                "micro_to_meso_expected_direction": "increase",
                "meso_to_macro_expected_direction": "increase",
                "expected_micro_response": "increase",
                "expected_meso_response": "increase",
                "expected_macro_response": "increase",
                "scientific_rationale": "Deterministic mock hypothesis for pipeline contract validation.",
                "mechanistic_explanation": "A public parameter changes a Micro process that precedes Meso organization and a Macro outcome.",
                "falsification_condition": "The frozen directions, responses, or temporal ordering are not supported.",
            }
        )
    prediction_indices = []
    for parameter in sorted(micro_by_parameter):
        prediction_indices.extend(
            [
                index for index, path in enumerate(paths)
                if path["parameter"] == parameter
            ][:2]
        )
    predictions = [
        {
            "prediction_id": f"pred_{scenario}_{index:02d}",
            "candidate_path_id": paths[path_index]["path_id"],
            "prospective_priority": index,
            "scientific_rationale": "Frozen mock prediction selected without numerical data.",
            "falsification_condition": "The referenced candidate path fails its frozen prospective criteria.",
        }
        for index, path_index in enumerate(prediction_indices)
    ]
    return {
        "scenario": scenario,
        "indicator_set_sha256": sha256_json(frozen_canonical),
        "candidate_paths": paths,
        "prospective_predictions": predictions,
    }


def mock_generation(scenario: str) -> dict[str, Any]:
    """Compatibility bundle for older unit helpers; LLM calls never use it."""
    indicators = mock_indicator_generation(scenario)
    path_generation = mock_path_generation(scenario)
    representation = {
        **indicators,
        "candidate_paths": path_generation["candidate_paths"],
        "candidate_edges": derive_candidate_edges(path_generation["candidate_paths"]),
    }
    legacy_predictions = []
    path_by_id = {
        path["path_id"]: path for path in path_generation["candidate_paths"]
    }
    for prediction in path_generation["prospective_predictions"]:
        path = path_by_id[prediction["candidate_path_id"]]
        order = [path["micro_indicator"], path["meso_indicator"], path["macro_indicator"]]
        legacy_predictions.append(
            {
                **prediction,
                "phenomenon": indicators["phenomenon"],
                "parameter": path["parameter"],
                "intervention_direction": path["intervention_direction"],
                "source_indicator": order[0],
                "expected_source_direction": path["expected_micro_response"],
                "downstream_indicators": order[1:],
                "expected_downstream_direction": [
                    path["expected_meso_response"], path["expected_macro_response"]
                ],
                "expected_temporal_order": order,
                "validation_criteria": {
                    "required_source_response": True,
                    "required_downstream_response": [True, True],
                    "required_temporal_order": True,
                    "required_candidate_edges": [
                        {"source": order[0], "target": order[1]},
                        {"source": order[1], "target": order[2]},
                    ],
                },
            }
        )
    return {
        "representation": representation,
        "prospective_predictions": legacy_predictions,
    }


def mock_completion_provider(
    scenario: str, generation_index: int, phase: str = "indicator"
) -> Callable[[str, str], LLMResponse]:
    payload = (
        mock_path_generation(scenario)
        if phase == "path"
        else mock_indicator_generation(scenario)
    )
    encoded = json.dumps(payload, ensure_ascii=False)

    def complete(_system: str, _user: str) -> LLMResponse:
        return LLMResponse(
            text=encoded,
            input_tokens=0,
            output_tokens=0,
            model=f"deterministic-mock-{phase}-{generation_index}",
        )

    return complete
