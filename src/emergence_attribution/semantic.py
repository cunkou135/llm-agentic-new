"""Two-phase data-blind semantic generation and immutable freezing."""

from __future__ import annotations

import hashlib
import json
import os
import time
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd
from pydantic import BaseModel, ValidationError

from .dsl import (
    DSLValidationError,
    ELEMENTARY_OR_LOCAL_OPERATORS,
    GENUINE_MESO_OPERATORS,
    GLOBAL_STRUCTURE_OPERATORS,
    TRIVIAL_WRAPPERS,
    computation_signature,
    expression_fields,
    expression_primitive_families,
    global_structure_operators,
    grammar_description,
    is_genuine_meso_expression,
    is_trivial_cross_scale_transform,
    is_trivial_micro_macro_lineage,
    validate_indicator_expression,
    validate_temporal_aggregation,
)
from .llm_client import LLMResponse, OpenAICompatibleClient, load_llm_config
from .raw_schemas import prompt_scenario_contract, raw_schema
from .schemas import (
    CandidateEdge,
    IndicatorGeneration,
    PathGeneration,
    StructuredRepresentation,
)


FORBIDDEN_PROMPT_KEYS = {
    "truth_role", "truth_edge", "truth_lag", "truth_sign", "edge_f1", "shd",
    "temporal_results", "intervention_results", "baseline_numerical_summary",
    "mechanism_channel", "controlled latent", "reference process",
    "reference edge", "reference lag", "reference sign", "previous successful path",
    "res2", "res_f",
}
SCALE_ENTITY_SCOPES = {
    "micro": {"individual", "interaction", "elementary_event", "local_process"},
    "meso": {"neighborhood", "district", "community", "cluster", "local_domain"},
    "macro": {"whole_system"},
}


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    os.replace(temporary, path)


def _write_immutable(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to overwrite immutable semantic artifact: {path}")
        return
    path.write_text(content, encoding="utf-8")


def _write_immutable_json(path: Path, value: Any) -> None:
    _write_immutable(path, json.dumps(value, indent=2, ensure_ascii=False))


def _parse_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    return json.loads(candidate[start : end + 1])


def _check_prompt(system: str, user: str) -> None:
    combined = (system + "\n" + user).lower()
    leaked = sorted(key for key in FORBIDDEN_PROMPT_KEYS if key in combined)
    if leaked:
        raise RuntimeError(f"semantic prompt contains forbidden evidence keys: {leaked}")


def _scale_contract() -> dict[str, Any]:
    return {
        "micro": "individual, interaction, elementary event, or local primitive process",
        "meso": (
            "real district, neighborhood, community, cluster, or local-domain organization; "
            "an outer mean or sum alone is insufficient"
        ),
        "macro": "whole-system collective state or outcome",
        "operators": {
            "elementary_or_local": sorted(ELEMENTARY_OR_LOCAL_OPERATORS),
            "genuine_meso": sorted(GENUINE_MESO_OPERATORS),
            "global_structure": sorted(GLOBAL_STRUCTURE_OPERATORS),
            "trivial_wrappers": sorted(TRIVIAL_WRAPPERS),
        },
    }


def build_indicator_prompt(
    scenario: str,
    scenario_spec: dict[str, Any],
    representation_config: dict[str, Any],
    prompt_template: str,
) -> tuple[str, str]:
    contract = prompt_scenario_contract(scenario, scenario_spec)
    contract.update(
        {
            "phase": "indicator_generation",
            "generic_computation_grammar": grammar_description(),
            "indicator_budget": representation_config["budget"],
            "scale_semantics": _scale_contract(),
            "constraints": [
                "Return exactly 16 Micro, 8 Meso, and 4 Macro executable indicators.",
                "Return no candidate edges, candidate paths, or prospective predictions.",
                "Do not group indicators into arbitrary semantic groups.",
                "Every controllable parameter needs at least one direct Micro association.",
                "Use only public simulator semantics and the supplied raw-log schema.",
                "Do not infer any result from unprovided numerical data.",
            ],
        }
    )
    system = (
        prompt_template.strip()
        + "\nPHASE A ONLY: construct executable multiscale observables. "
        "Do not propose relationships, mechanism paths, or predictions."
    )
    user = (
        "Input contract:\n"
        + json.dumps(contract, indent=2, ensure_ascii=False)
        + "\n\nOutput JSON schema:\n"
        + json.dumps(IndicatorGeneration.model_json_schema(), indent=2, ensure_ascii=False)
    )
    _check_prompt(system, user)
    return system, user


def build_path_prompt(
    scenario: str,
    scenario_spec: dict[str, Any],
    representation_config: dict[str, Any],
    frozen_indicators: dict[str, Any],
    indicator_hash: str,
    prompt_template: str,
) -> tuple[str, str]:
    indicators = [
        {
            key: item[key]
            for key in (
                "id", "semantic_name", "scientific_definition", "scale", "entity_scope",
                "source_fields", "computation", "parameter_associations",
            )
        }
        for item in frozen_indicators["indicators"]
    ]
    contract = {
        "phase": "path_hypothesis_generation",
        "scenario": scenario,
        "public_simulator": prompt_scenario_contract(scenario, scenario_spec),
        "indicator_set_sha256": indicator_hash,
        "frozen_indicators": indicators,
        "candidate_path_bounds": {
            "minimum": int(representation_config["minimum_candidate_paths"]),
            "maximum": int(representation_config["maximum_candidate_paths"]),
        },
        "constraints": [
            "Use only the supplied frozen indicator IDs; never create or modify an observable.",
            "Every hypothesis is a complete parameter to Micro to Meso to Macro mechanism path.",
            "Cover every controllable parameter with at least four paths.",
            "Cover every frozen Macro endpoint with at least two paths.",
            "Do not repeat an identical Micro-Meso-Macro triple.",
            "Only the primary accepted generation enters Stage 2 and Stage 3.",
            "Return six prospective predictions bound to candidate_path_id values.",
            "Do not use or request simulation outcomes.",
        ],
    }
    system = (
        prompt_template.strip()
        + "\nPHASE B ONLY: construct complete testable mechanism hypotheses from frozen "
        "observables. No new observable is permitted."
    )
    user = (
        "Input contract:\n"
        + json.dumps(contract, indent=2, ensure_ascii=False)
        + "\n\nOutput JSON schema:\n"
        + json.dumps(PathGeneration.model_json_schema(), indent=2, ensure_ascii=False)
    )
    _check_prompt(system, user)
    return system, user


def build_prompt(
    scenario: str,
    scenario_spec: dict[str, Any],
    representation_config: dict[str, Any],
    prompt_template: str,
) -> tuple[str, str]:
    """Compatibility name now resolves exclusively to Phase A."""
    return build_indicator_prompt(
        scenario, scenario_spec, representation_config, prompt_template
    )


def validate_indicator_generation(
    value: IndicatorGeneration,
    scenario: str,
    scenario_spec: dict[str, Any],
    representation_config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if value.scenario != scenario:
        errors.append(f"scenario must be {scenario}")
    counts = {
        scale: sum(item.scale == scale for item in value.indicators)
        for scale in ("micro", "meso", "macro")
    }
    for scale, required in representation_config["budget"].items():
        if counts.get(scale, 0) != int(required):
            errors.append(f"scale {scale} requires exactly {required} indicators")
    schema = raw_schema(scenario)
    levels = {str(item["field_name"]): str(item.get("entity_level", "")) for item in schema}
    signatures: list[str] = []
    direct: dict[str, set[str]] = {
        name: set() for name in scenario_spec["interventions"]
    }
    families: dict[str, list[str]] = {}
    for indicator in value.indicators:
        try:
            validate_indicator_expression(indicator.computation, schema)
            validate_temporal_aggregation(
                indicator.temporal_aggregation.model_dump(mode="json", exclude_none=True)
            )
        except DSLValidationError as exc:
            errors.append(f"indicator {indicator.id}: {exc}")
        actual_fields = expression_fields(indicator.computation)
        if actual_fields != set(indicator.source_fields):
            errors.append(
                f"indicator {indicator.id}: source_fields must equal AST fields {sorted(actual_fields)}"
            )
        if indicator.entity_scope not in SCALE_ENTITY_SCOPES[indicator.scale]:
            errors.append(
                f"indicator {indicator.id}: invalid {indicator.scale} entity scope"
            )
        if indicator.scale == "micro" and not any(
            levels.get(field) in {"agent", "interaction", "cell", "edge"}
            for field in actual_fields
        ):
            errors.append(f"indicator {indicator.id}: Micro lacks an elementary primitive")
        global_ops = sorted(global_structure_operators(indicator.computation))
        if indicator.scale in {"micro", "meso"} and global_ops:
            errors.append(
                f"indicator {indicator.id}: global structure invalid for {indicator.scale}: {global_ops}"
            )
        if indicator.scale == "meso" and not is_genuine_meso_expression(indicator.computation):
            errors.append(f"indicator {indicator.id}: Meso lacks non-trivial organization")
        for association in indicator.parameter_associations:
            if association.parameter not in scenario_spec["interventions"]:
                errors.append(f"indicator {indicator.id}: unknown parameter {association.parameter}")
            elif association.relationship == "direct" and indicator.scale == "micro":
                direct[association.parameter].add(indicator.id)
        signatures.append(json.dumps(computation_signature(indicator.computation), sort_keys=True))
        families[indicator.id] = sorted(expression_primitive_families(indicator.computation, schema))
    if len(signatures) != len(set(signatures)):
        errors.append("indicator computations require unique canonical signatures")
    missing_direct = sorted(name for name, sources in direct.items() if not sources)
    if representation_config.get("require_all_parameters_associated", True) and missing_direct:
        errors.append(f"parameters require a direct Micro indicator: {missing_direct}")
    lower_boundary = value.interpretation_boundary.lower()
    if "temporal" not in lower_boundary or "intervention" not in lower_boundary:
        errors.append("interpretation_boundary must defer temporal and intervention evidence")
    return {
        "valid": not errors,
        "errors": errors,
        "scale_counts": counts,
        "indicator_count": len(value.indicators),
        "source_field_diversity": len(set().union(*(
            expression_fields(item.computation) for item in value.indicators
        ))),
        "direct_micro_parameter_sources": {
            name: sorted(sources) for name, sources in direct.items()
        },
        "source_families": families,
        "meso_structural_diversity": len({
            signatures[index] for index, item in enumerate(value.indicators)
            if item.scale == "meso"
        }),
        "macro_concept_diversity": len({
            item.semantic_name.strip().lower() for item in value.indicators
            if item.scale == "macro"
        }),
    }


def validate_generation(
    value: Any,
    scenario: str,
    scenario_spec: dict[str, Any],
    representation_config: dict[str, Any],
) -> dict[str, Any]:
    """Validate a legacy combined fixture through both new phase contracts.

    Production LLM calls never use this compatibility entry point.  Keeping it
    strict lets historical DSL tests exercise Phase A indicator validation and
    Phase B path-lineage validation without restoring the old combined schema.
    """
    if isinstance(value, IndicatorGeneration):
        indicator_value = value
        raw: dict[str, Any] = value.model_dump(mode="json", exclude_none=True)
    else:
        payload = value.model_dump(mode="json") if hasattr(value, "model_dump") else value
        raw = payload.get("representation", payload)
        indicator_payload = {
            key: raw[key]
            for key in ("scenario", "phenomenon", "indicators", "interpretation_boundary")
            if key in raw
        }
        indicator_value = IndicatorGeneration.model_validate(indicator_payload)
    result = validate_indicator_generation(
        indicator_value, scenario, scenario_spec, representation_config
    )
    result["trivial_micro_macro_lineage_count"] = 0
    paths = raw.get("candidate_paths", [])
    if not paths:
        return result
    prospective_source = (
        value.model_dump(mode="json").get("prospective_predictions", [])
        if hasattr(value, "model_dump")
        else []
    )
    prospective = [
        {
            key: item[key]
            for key in (
                "prediction_id", "candidate_path_id", "prospective_priority",
                "scientific_rationale", "falsification_condition",
            )
            if key in item
        }
        for item in prospective_source
    ]
    try:
        generation = PathGeneration.model_validate(
            {
                "scenario": scenario,
                "indicator_set_sha256": sha256_json(
                    indicator_value.model_dump(mode="json", exclude_none=True)
                ),
                "candidate_paths": paths,
                "prospective_predictions": prospective,
            }
        )
        path_result = validate_path_generation(
            generation,
            indicator_value,
            generation.indicator_set_sha256,
            scenario_spec,
            representation_config,
        )
        result["errors"].extend(path_result["errors"])
        result["candidate_path_count"] = path_result["candidate_path_count"]
        result["parameter_path_coverage"] = path_result["parameter_path_coverage"]
        result["macro_path_coverage"] = path_result["macro_path_coverage"]
        result["trivial_micro_macro_lineage_count"] = sum(
            "trivial Micro-Macro lineage" in error for error in path_result["errors"]
        )
    except ValidationError as exc:
        result["errors"].append(str(exc))
    result["valid"] = not result["errors"]
    return result


def validate_path_generation(
    value: PathGeneration,
    indicators: IndicatorGeneration,
    indicator_hash: str,
    scenario_spec: dict[str, Any],
    representation_config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    if value.scenario != indicators.scenario:
        errors.append("path scenario does not match frozen indicators")
    if value.indicator_set_sha256 != indicator_hash:
        errors.append("path indicator input hash does not match frozen indicator hash")
    lookup = {item.id: item for item in indicators.indicators}
    minimum = int(representation_config["minimum_candidate_paths"])
    maximum = int(representation_config["maximum_candidate_paths"])
    if not minimum <= len(value.candidate_paths) <= maximum:
        errors.append(f"candidate path count must be within [{minimum}, {maximum}]")
    parameter_counts = {name: 0 for name in scenario_spec["interventions"]}
    macro_counts = {
        item.id: 0 for item in indicators.indicators if item.scale == "macro"
    }
    schema = raw_schema(indicators.scenario)
    for path in value.candidate_paths:
        nodes = (path.micro_indicator, path.meso_indicator, path.macro_indicator)
        unknown = sorted(set(nodes) - set(lookup))
        if unknown:
            errors.append(f"path {path.path_id}: unknown frozen indicator IDs {unknown}")
            continue
        expected_scales = ("micro", "meso", "macro")
        actual_scales = tuple(lookup[node].scale for node in nodes)
        if actual_scales != expected_scales:
            errors.append(f"path {path.path_id}: expected Micro-Meso-Macro references")
        if path.parameter not in parameter_counts:
            errors.append(f"path {path.path_id}: unknown controllable parameter")
        else:
            parameter_counts[path.parameter] += 1
            direct = {
                item.parameter
                for item in lookup[path.micro_indicator].parameter_associations
                if item.relationship == "direct"
            }
            if path.parameter not in direct:
                errors.append(
                    f"path {path.path_id}: Micro source lacks direct parameter association"
                )
        macro_counts[path.macro_indicator] += 1
        micro, meso, macro = (lookup[node] for node in nodes)
        if is_trivial_cross_scale_transform(
            micro.computation,
            micro.temporal_aggregation.model_dump(mode="json", exclude_none=True),
            meso.computation,
            meso.temporal_aggregation.model_dump(mode="json", exclude_none=True),
        ) or is_trivial_cross_scale_transform(
            meso.computation,
            meso.temporal_aggregation.model_dump(mode="json", exclude_none=True),
            macro.computation,
            macro.temporal_aggregation.model_dump(mode="json", exclude_none=True),
        ):
            errors.append(f"path {path.path_id}: trivial cross-scale transform")
        if is_trivial_micro_macro_lineage(
            micro.computation,
            micro.temporal_aggregation.model_dump(mode="json", exclude_none=True),
            macro.computation,
            macro.temporal_aggregation.model_dump(mode="json", exclude_none=True),
            schema,
        ):
            errors.append(f"path {path.path_id}: trivial Micro-Macro lineage")
    minimum_per_parameter = int(
        representation_config.get("minimum_paths_per_parameter", 4)
    )
    sparse_parameters = sorted(
        name for name, count in parameter_counts.items()
        if count < minimum_per_parameter
    )
    if sparse_parameters:
        errors.append(
            f"parameters require at least {minimum_per_parameter} paths: {sparse_parameters}"
        )
    minimum_per_macro = int(representation_config.get("minimum_paths_per_macro", 2))
    sparse_macros = sorted(
        name for name, count in macro_counts.items() if count < minimum_per_macro
    )
    if sparse_macros:
        errors.append(
            f"Macro indicators require at least {minimum_per_macro} paths: {sparse_macros}"
        )
    expected_predictions = int(representation_config.get("prospective_prediction_count", 6))
    if len(value.prospective_predictions) != expected_predictions:
        errors.append(f"exactly {expected_predictions} prospective predictions are required")
    prediction_paths = {
        path.path_id: path for path in value.candidate_paths
    }
    prediction_parameters = {
        prediction_paths[item.candidate_path_id].parameter
        for item in value.prospective_predictions
        if item.candidate_path_id in prediction_paths
    }
    if prediction_parameters != set(parameter_counts):
        errors.append("prospective predictions must cover every controllable parameter")
    return {
        "valid": not errors,
        "errors": errors,
        "candidate_path_count": len(value.candidate_paths),
        "parameter_path_coverage": parameter_counts,
        "macro_path_coverage": macro_counts,
        "prospective_prediction_count": len(value.prospective_predictions),
    }


def derive_candidate_edges(paths: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    projected: dict[tuple[str, str], dict[str, Any]] = {}
    for path in paths:
        macro = str(path["macro_indicator"])
        group = f"macro_outcome_{macro}"
        for source, target, direction in (
            (
                str(path["micro_indicator"]), str(path["meso_indicator"]),
                str(path["micro_to_meso_expected_direction"]),
            ),
            (
                str(path["meso_indicator"]), macro,
                str(path["meso_to_macro_expected_direction"]),
            ),
        ):
            item = projected.setdefault(
                (source, target),
                {
                    "source": source,
                    "target": target,
                    "expected_direction": direction,
                    "hypothesis_group_ids": [],
                    "path_ids": [],
                },
            )
            if item["expected_direction"] != direction:
                item["expected_direction"] = "mixed"
            item["hypothesis_group_ids"].append(group)
            item["path_ids"].append(str(path["path_id"]))
    result = []
    for item in projected.values():
        item["hypothesis_group_ids"] = sorted(set(item["hypothesis_group_ids"]))
        item["path_ids"] = sorted(set(item["path_ids"]))
        result.append(CandidateEdge.model_validate(item).model_dump(mode="json"))
    return sorted(result, key=lambda item: (item["source"], item["target"]))


def _run_generation_call(
    *,
    root: Path,
    model: type[BaseModel],
    validator: Callable[[Any], dict[str, Any]],
    system: str,
    user: str,
    maximum_repairs: int,
    completion: Callable[[str, str], LLMResponse] | None,
    llm_config: dict[str, Any],
) -> dict[str, Any]:
    root.mkdir(parents=True, exist_ok=True)
    prompt_text = f"# System\n\n{system}\n\n# User\n\n{user}\n"
    _write_immutable(root / "prompt.md", prompt_text)
    completed_path = root / "accepted_payload.json"
    completed_hash_path = root / "accepted_payload.sha256"
    result_path = root / "generation_result.json"
    if completed_path.is_file() or completed_hash_path.is_file():
        if not completed_path.is_file() or not completed_hash_path.is_file():
            raise RuntimeError(f"incomplete immutable generation checkpoint: {root}")
        if _sha256_path(completed_path) != completed_hash_path.read_text(encoding="utf-8").strip():
            raise RuntimeError(f"accepted semantic payload hash mismatch: {root}")
        value = model.model_validate_json(completed_path.read_text(encoding="utf-8"))
        validation = validator(value)
        if not validation["valid"]:
            raise RuntimeError(f"frozen semantic payload no longer validates: {validation['errors']}")
        return json.loads(result_path.read_text(encoding="utf-8"))
    client = None if completion else OpenAICompatibleClient(llm_config)
    errors: list[str] = []
    prior = ""
    started = time.perf_counter()
    for repair_round in range(maximum_repairs + 1):
        repair = ""
        if repair_round:
            repair = (
                "\n\nThe previous response failed only these schema/executability checks:\n"
                + json.dumps(errors, indent=2, ensure_ascii=False)
                + "\nReturn a complete corrected object without using simulation outcomes.\n"
                + prior
            )
        request = user + repair
        _write_immutable(
            root / f"request_round_{repair_round:02d}.md",
            f"# System\n\n{system}\n\n# User\n\n{request}\n",
        )
        response_record_path = root / f"response_round_{repair_round:02d}.json"
        if response_record_path.is_file():
            record = json.loads(response_record_path.read_text(encoding="utf-8"))
            response = LLMResponse(
                text=str(record["raw_text"]),
                input_tokens=int(record["input_tokens"]),
                output_tokens=int(record["output_tokens"]),
                model=str(record["model"]),
            )
        else:
            response = completion(system, request) if completion else client.complete_json(system, request)
            record = {
                "raw_text": response.text,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "model": response.model,
            }
            _write_immutable(
                response_record_path,
                json.dumps(record, indent=2, ensure_ascii=False),
            )
        prior = response.text
        try:
            value = model.model_validate(_parse_json(response.text))
            validation = validator(value)
            errors = list(validation["errors"])
        except (ValueError, ValidationError, json.JSONDecodeError) as exc:
            value = None
            validation = {"valid": False, "errors": [str(exc)]}
            errors = list(validation["errors"])
        if value is not None and validation["valid"]:
            accepted = value.model_dump(mode="json", exclude_none=True)
            _write_immutable(
                completed_path, json.dumps(accepted, indent=2, ensure_ascii=False)
            )
            _write_immutable(completed_hash_path, _sha256_path(completed_path) + "\n")
            result = {
                "status": "accepted",
                "repair_rounds": repair_round,
                "validation": validation,
                "accepted_generation": accepted,
                "elapsed_seconds": time.perf_counter() - started,
            }
            _write_immutable(result_path, json.dumps(result, indent=2, ensure_ascii=False))
            return result
    result = {
        "status": "rejected",
        "repair_rounds": maximum_repairs,
        "validation": {"valid": False, "errors": errors},
        "accepted_generation": None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    _write_immutable(result_path, json.dumps(result, indent=2, ensure_ascii=False))
    return result


def _completion_for(
    provider: Callable[..., Callable[[str, str], LLMResponse]] | None,
    scenario: str,
    index: int,
    phase: str,
) -> Callable[[str, str], LLMResponse] | None:
    if provider is None:
        return None
    try:
        return provider(scenario, index, phase)
    except TypeError:
        return provider(scenario, index)


def _indicator_selection_key(item: dict[str, Any]) -> tuple[Any, ...]:
    validation = item["validation"]
    return (
        -int(validation["source_field_diversity"]),
        -sum(bool(value) for value in validation["direct_micro_parameter_sources"].values()),
        -int(validation["meso_structural_diversity"]),
        -int(validation["macro_concept_diversity"]),
        int(item["repair_rounds"]),
        sha256_json(item["accepted_generation"]),
    )


def _jaccard(first: set[Any], second: set[Any]) -> float:
    return 1.0 if not first and not second else len(first & second) / len(first | second)


def _write_indicator_replication(
    root: Path, generations: dict[str, list[dict[str, Any]]]
) -> None:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"schema_version": "2.0", "scenarios": {}}
    for scenario, items in sorted(generations.items()):
        pair_rows = []
        for left, right in combinations(items, 2):
            first = left["accepted_generation"]["indicators"]
            second = right["accepted_generation"]["indicators"]
            def signatures(values: list[dict[str, Any]]) -> set[str]:
                return {
                    json.dumps(computation_signature(item["computation"]), sort_keys=True)
                    for item in values
                }
            row = {
                "scenario": scenario,
                "generation_a": left["generation"],
                "generation_b": right["generation"],
                "computation_agreement": _jaccard(signatures(first), signatures(second)),
                "scale_agreement": _jaccard(
                    {(item["scale"], json.dumps(computation_signature(item["computation"]), sort_keys=True)) for item in first},
                    {(item["scale"], json.dumps(computation_signature(item["computation"]), sort_keys=True)) for item in second},
                ),
                "source_family_agreement": _jaccard(
                    {tuple(item["source_fields"]) for item in first},
                    {tuple(item["source_fields"]) for item in second},
                ),
                "parameter_source_agreement": _jaccard(
                    {(assoc["parameter"], item["id"]) for item in first for assoc in item["parameter_associations"] if assoc["relationship"] == "direct"},
                    {(assoc["parameter"], item["id"]) for item in second for assoc in item["parameter_associations"] if assoc["relationship"] == "direct"},
                ),
                "macro_concept_agreement": _jaccard(
                    {item["semantic_name"].lower() for item in first if item["scale"] == "macro"},
                    {item["semantic_name"].lower() for item in second if item["scale"] == "macro"},
                ),
            }
            rows.append(row)
            pair_rows.append(row)
        summary["scenarios"][scenario] = {
            key: (sum(float(row[key]) for row in pair_rows) / len(pair_rows) if pair_rows else None)
            for key in (
                "computation_agreement", "scale_agreement", "source_family_agreement",
                "parameter_source_agreement", "macro_concept_agreement",
            )
        }
    pd.DataFrame(rows).to_csv(root / "indicator_replication_pairwise.csv", index=False)
    _atomic_json(root / "representation_agreement.json", summary)


def _write_path_replication(
    root: Path, generations: dict[str, list[dict[str, Any]]]
) -> None:
    rows: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"schema_version": "1.0", "scenarios": {}}
    metrics = (
        "exact_path_jaccard", "micro_meso_edge_jaccard", "meso_macro_edge_jaccard",
        "parameter_path_coverage", "macro_path_coverage", "direction_agreement",
    )
    for scenario, items in sorted(generations.items()):
        pair_rows = []
        for left, right in combinations(items, 2):
            first = left["accepted_generation"]["candidate_paths"]
            second = right["accepted_generation"]["candidate_paths"]
            exact_a = {(p["micro_indicator"], p["meso_indicator"], p["macro_indicator"], p["parameter"]) for p in first}
            exact_b = {(p["micro_indicator"], p["meso_indicator"], p["macro_indicator"], p["parameter"]) for p in second}
            shared = exact_a & exact_b
            first_by = {(p["micro_indicator"], p["meso_indicator"], p["macro_indicator"], p["parameter"]): p for p in first}
            second_by = {(p["micro_indicator"], p["meso_indicator"], p["macro_indicator"], p["parameter"]): p for p in second}
            direction = (
                sum(
                    first_by[key]["micro_to_meso_expected_direction"] == second_by[key]["micro_to_meso_expected_direction"]
                    and first_by[key]["meso_to_macro_expected_direction"] == second_by[key]["meso_to_macro_expected_direction"]
                    for key in shared
                ) / len(shared)
                if shared else 0.0
            )
            row = {
                "scenario": scenario,
                "generation_a": left["generation"],
                "generation_b": right["generation"],
                "exact_path_jaccard": _jaccard(exact_a, exact_b),
                "micro_meso_edge_jaccard": _jaccard(
                    {(p["micro_indicator"], p["meso_indicator"]) for p in first},
                    {(p["micro_indicator"], p["meso_indicator"]) for p in second},
                ),
                "meso_macro_edge_jaccard": _jaccard(
                    {(p["meso_indicator"], p["macro_indicator"]) for p in first},
                    {(p["meso_indicator"], p["macro_indicator"]) for p in second},
                ),
                "parameter_path_coverage": _jaccard(
                    {p["parameter"] for p in first}, {p["parameter"] for p in second}
                ),
                "macro_path_coverage": _jaccard(
                    {p["macro_indicator"] for p in first}, {p["macro_indicator"] for p in second}
                ),
                "direction_agreement": direction,
            }
            rows.append(row)
            pair_rows.append(row)
        summary["scenarios"][scenario] = {
            key: (sum(float(row[key]) for row in pair_rows) / len(pair_rows) if pair_rows else None)
            for key in metrics
        }
    pd.DataFrame(rows).to_csv(root / "path_replication_pairwise.csv", index=False)
    _atomic_json(root / "path_replication_agreement.json", summary)


def _semantic_guard(run_root: Path) -> None:
    if any((run_root / "data").glob("*baseline_simulation_manifest.json")):
        raise RuntimeError("baseline already started; semantic stages are immutable")


def run_indicator_generation_stage(
    experiment_config: dict[str, Any],
    llm_config_path: Path,
    run_root: Path,
    prompt_template_path: Path,
    workers: int,
    progress_callback: Callable[[str, int, int], None] | None = None,
    completion_provider: Callable[..., Callable[[str, str], LLMResponse]] | None = None,
) -> dict[str, Any]:
    """Run Phase A only and save the data-blind selected indicator payload."""

    del workers
    _semantic_guard(run_root)
    llm_config = load_llm_config(llm_config_path, require_key=completion_provider is None)
    template = prompt_template_path.read_text(encoding="utf-8")
    rep_config = experiment_config["representation"]
    selection_count = int(experiment_config["semantic_replication"]["selection_generations"])
    indicator_replication_count = int(
        experiment_config["semantic_replication"]["replication_only_generations"]
    )
    path_replication_count = int(
        experiment_config.get("path_replication", {}).get("replication_only_generations", 2)
    )
    maximum_repairs = int(rep_config["maximum_repair_rounds"])
    histories = run_root / "llm"
    representation_root = run_root / "representation"
    representation_root.mkdir(parents=True, exist_ok=True)
    indicator_generations: dict[str, list[dict[str, Any]]] = {}
    selected_indicators: dict[str, dict[str, Any]] = {}
    validation_bundle: dict[str, Any] = {
        "schema_version": "4.0-two-phase-path-centered",
        "phase": "indicator_generation",
    }
    total_indicator_calls = len(experiment_config["scenarios"]) * (
        selection_count + indicator_replication_count
    )
    done = 0
    for scenario, scenario_spec in sorted(experiment_config["scenarios"].items()):
        system, user = build_indicator_prompt(scenario, scenario_spec, rep_config, template)
        results = []
        for index in range(selection_count + indicator_replication_count):
            result = _run_generation_call(
                root=histories / "indicator" / scenario / f"generation_{index:02d}",
                model=IndicatorGeneration,
                validator=lambda value, s=scenario, spec=scenario_spec: validate_indicator_generation(value, s, spec, rep_config),
                system=system,
                user=user,
                maximum_repairs=maximum_repairs,
                completion=_completion_for(completion_provider, scenario, index, "indicator"),
                llm_config=llm_config,
            )
            result["generation"] = index
            result["generation_role"] = "selection_eligible" if index < selection_count else "replication_only"
            results.append(result)
            done += 1
            if progress_callback:
                progress_callback("Indicator generations", done, total_indicator_calls)
        accepted = [item for item in results if item["status"] == "accepted"]
        eligible = [item for item in accepted if item["generation"] < selection_count]
        if not eligible:
            raise RuntimeError(f"no valid selection-eligible indicator generation for {scenario}")
        selected = min(eligible, key=_indicator_selection_key)
        selected_indicators[scenario] = selected["accepted_generation"]
        indicator_generations[scenario] = accepted
        validation_bundle[scenario] = {
            "selected_indicator_generation": selected["generation"],
            "indicator_selection_rule": rep_config["selection_rule"],
            "indicator_selection_key": list(_indicator_selection_key(selected)),
            "selection_used_simulation_data": False,
            "indicator_generations": [
                {
                    "generation": item["generation"],
                    "generation_role": item["generation_role"],
                    "status": item["status"],
                    "repair_rounds": item["repair_rounds"],
                    "validation": item["validation"],
                }
                for item in results
            ],
        }
    _write_indicator_replication(representation_root, indicator_generations)
    selection = {
        "schema_version": "1.0",
        "selection_used_simulation_data": False,
        "capacity_control": rep_config["budget"],
        "scenarios": selected_indicators,
    }
    _write_immutable_json(representation_root / "indicator_selection.json", selection)
    _write_immutable_json(
        representation_root / "indicator_generation_validation.json",
        validation_bundle,
    )
    return validation_bundle


def freeze_indicator_stage(
    experiment_config: dict[str, Any], run_root: Path
) -> dict[str, Any]:
    """Freeze the selected Phase A payload before any path call is allowed."""

    _semantic_guard(run_root)
    representation_root = run_root / "representation"
    selection_path = representation_root / "indicator_selection.json"
    if not selection_path.is_file():
        raise FileNotFoundError("indicator selection is missing")
    selection = json.loads(selection_path.read_text(encoding="utf-8"))
    expected = set(experiment_config["scenarios"])
    if set(selection.get("scenarios", {})) != expected:
        raise RuntimeError("indicator selection scenario set does not match configuration")
    for scenario, payload in selection["scenarios"].items():
        value = IndicatorGeneration.model_validate(payload)
        validation = validate_indicator_generation(
            value,
            scenario,
            experiment_config["scenarios"][scenario],
            experiment_config["representation"],
        )
        if not validation["valid"]:
            raise RuntimeError(f"selected indicators no longer validate: {validation['errors']}")
    indicators_bundle = {
        "schema_version": "1.0",
        "capacity_control": experiment_config["representation"]["budget"],
        "scenarios": selection["scenarios"],
    }
    indicators_path = representation_root / "indicators_frozen.json"
    _write_immutable_json(indicators_path, indicators_bundle)
    digest = _sha256_path(indicators_path)
    _write_immutable(representation_root / "INDICATORS_FROZEN.sha256", digest + "\n")
    return {
        "indicators_frozen": True,
        "scenario_count": len(expected),
        "indicators_frozen_sha256": digest,
    }


def run_path_generation_stage(
    experiment_config: dict[str, Any],
    llm_config_path: Path,
    run_root: Path,
    prompt_template_path: Path,
    workers: int,
    progress_callback: Callable[[str, int, int], None] | None = None,
    completion_provider: Callable[..., Callable[[str, str], LLMResponse]] | None = None,
) -> dict[str, Any]:
    """Run Phase B only against the immutable Phase A indicator set."""

    del workers
    _semantic_guard(run_root)
    llm_config = load_llm_config(llm_config_path, require_key=completion_provider is None)
    template = prompt_template_path.read_text(encoding="utf-8")
    rep_config = experiment_config["representation"]
    path_replication_count = int(
        experiment_config.get("path_replication", {}).get("replication_only_generations", 2)
    )
    maximum_repairs = int(rep_config["maximum_repair_rounds"])
    histories = run_root / "llm"
    representation_root = run_root / "representation"
    indicators_path = representation_root / "indicators_frozen.json"
    marker = representation_root / "INDICATORS_FROZEN.sha256"
    if not indicators_path.is_file() or not marker.is_file():
        raise FileNotFoundError("frozen indicator artifacts are incomplete")
    indicator_bundle_hash = _sha256_path(indicators_path)
    if marker.read_text(encoding="utf-8").strip() != indicator_bundle_hash:
        raise RuntimeError("frozen indicator hash mismatch")
    indicator_bundle = json.loads(indicators_path.read_text(encoding="utf-8"))
    selected_indicators = indicator_bundle["scenarios"]
    validation_path = representation_root / "indicator_generation_validation.json"
    validation_bundle: dict[str, Any] = json.loads(
        validation_path.read_text(encoding="utf-8")
    )

    primary_paths: dict[str, list[dict[str, Any]]] = {}
    prospective: dict[str, list[dict[str, Any]]] = {}
    derived: dict[str, list[dict[str, Any]]] = {}
    path_generations: dict[str, list[dict[str, Any]]] = {}
    total_path_calls = len(experiment_config["scenarios"]) * (1 + path_replication_count)
    done = 0
    for scenario, scenario_spec in sorted(experiment_config["scenarios"].items()):
        indicator_value = IndicatorGeneration.model_validate(selected_indicators[scenario])
        scenario_indicator_hash = sha256_json(selected_indicators[scenario])
        system, user = build_path_prompt(
            scenario, scenario_spec, rep_config, selected_indicators[scenario],
            scenario_indicator_hash, template,
        )
        accepted = []
        for index in range(1 + path_replication_count):
            result = _run_generation_call(
                root=histories / "path" / scenario / f"generation_{index:02d}",
                model=PathGeneration,
                validator=lambda value, indicators=indicator_value, digest=scenario_indicator_hash, spec=scenario_spec: validate_path_generation(value, indicators, digest, spec, rep_config),
                system=system,
                user=user,
                maximum_repairs=maximum_repairs,
                completion=_completion_for(completion_provider, scenario, index, "path"),
                llm_config=llm_config,
            )
            result["generation"] = index
            result["generation_role"] = "primary" if index == 0 else "replication_only"
            if result["status"] == "accepted":
                accepted.append(result)
            done += 1
            if progress_callback:
                progress_callback("Path generations", done, total_path_calls)
        primary = next((item for item in accepted if item["generation"] == 0), None)
        if primary is None:
            raise RuntimeError(f"primary path generation was not accepted for {scenario}")
        path_generations[scenario] = accepted
        payload = primary["accepted_generation"]
        primary_paths[scenario] = payload["candidate_paths"]
        prospective[scenario] = payload["prospective_predictions"]
        derived[scenario] = derive_candidate_edges(payload["candidate_paths"])
        representation = StructuredRepresentation.model_validate(
            {
                **selected_indicators[scenario],
                "candidate_paths": primary_paths[scenario],
                "candidate_edges": derived[scenario],
            }
        ).model_dump(mode="json", exclude_none=True)
        _write_immutable_json(
            representation_root / f"{scenario}_representation.json", representation
        )
        validation_bundle[scenario].update(
            {
                "indicator_set_sha256": scenario_indicator_hash,
                "primary_path_generation": 0,
                "path_selection_rule": "first accepted valid generation only",
                "path_selection_used_simulation_data": False,
                "path_generation_validation": primary["validation"],
                "path_replication_only_generation_count": path_replication_count,
            }
        )
    candidate_paths_bundle = {"schema_version": "1.0", "scenarios": primary_paths}
    derived_edges_bundle = {
        "schema_version": "1.0",
        "derivation": "deterministic projection of frozen candidate paths",
        "scenarios": derived,
    }
    prospective_bundle = {"schema_version": "2.0", "scenarios": prospective}
    paths_path = representation_root / "candidate_paths.json"
    edges_path = representation_root / "derived_candidate_edges.json"
    predictions_path = representation_root / "prospective_predictions.json"
    _write_immutable_json(paths_path, candidate_paths_bundle)
    _write_immutable_json(edges_path, derived_edges_bundle)
    _write_immutable_json(predictions_path, prospective_bundle)
    for artifact, marker in (
        (paths_path, "CANDIDATE_PATHS_FROZEN.sha256"),
        (predictions_path, "PROSPECTIVE_PREDICTIONS_FROZEN.sha256"),
    ):
        _write_immutable(representation_root / marker, _sha256_path(artifact) + "\n")
    validation_bundle.update(
        {
            "indicators_frozen_sha256": indicator_bundle_hash,
            "candidate_paths_sha256": _sha256_path(paths_path),
            "derived_candidate_edges_sha256": _sha256_path(edges_path),
            "prospective_predictions_sha256": _sha256_path(predictions_path),
            "all_semantics_frozen_before_baseline": True,
        }
    )
    _write_path_replication(representation_root, path_generations)
    _write_immutable_json(
        representation_root / "representation_validation.json", validation_bundle
    )
    return validation_bundle


def run_semantic_stage(
    experiment_config: dict[str, Any],
    llm_config_path: Path,
    run_root: Path,
    prompt_template_path: Path,
    workers: int,
    progress_callback: Callable[[str, int, int], None] | None = None,
    completion_provider: Callable[..., Callable[[str, str], LLMResponse]] | None = None,
) -> dict[str, Any]:
    """Compatibility wrapper that executes both semantic phases in strict order."""

    run_indicator_generation_stage(
        experiment_config,
        llm_config_path,
        run_root,
        prompt_template_path,
        workers,
        progress_callback,
        completion_provider,
    )
    freeze_indicator_stage(experiment_config, run_root)
    return run_path_generation_stage(
        experiment_config,
        llm_config_path,
        run_root,
        prompt_template_path,
        workers,
        progress_callback,
        completion_provider,
    )


def run_generation(
    scenario: str,
    generation_index: int,
    experiment_config: dict[str, Any],
    llm_config: dict[str, Any],
    prompt_template: str,
    output_root: Path,
    completion: Callable[[str, str], LLMResponse] | None = None,
) -> dict[str, Any]:
    """Compatibility entry point for a single Phase A generation."""
    spec = experiment_config["scenarios"][scenario]
    rep = experiment_config["representation"]
    system, user = build_indicator_prompt(scenario, spec, rep, prompt_template)
    result = _run_generation_call(
        root=output_root / "indicator" / scenario / f"generation_{generation_index:02d}",
        model=IndicatorGeneration,
        validator=lambda value: validate_indicator_generation(value, scenario, spec, rep),
        system=system,
        user=user,
        maximum_repairs=int(rep["maximum_repair_rounds"]),
        completion=completion,
        llm_config=llm_config,
    )
    return {"scenario": scenario, "generation": generation_index, **result}


def load_frozen_representations(run_root: Path) -> dict[str, dict[str, Any]]:
    validation_path = run_root / "representation" / "representation_validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError("semantic freeze validation is missing")
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    checks = {
        "indicators_frozen.json": (
            validation["indicators_frozen_sha256"], "INDICATORS_FROZEN.sha256"
        ),
        "candidate_paths.json": (
            validation["candidate_paths_sha256"], "CANDIDATE_PATHS_FROZEN.sha256"
        ),
        "derived_candidate_edges.json": (
            validation["derived_candidate_edges_sha256"], None
        ),
        "prospective_predictions.json": (
            validation["prospective_predictions_sha256"],
            "PROSPECTIVE_PREDICTIONS_FROZEN.sha256",
        ),
    }
    for name, (expected, marker_name) in checks.items():
        path = run_root / "representation" / name
        if not path.is_file() or _sha256_path(path) != expected:
            raise RuntimeError(f"frozen semantic artifact hash mismatch: {name}")
        if marker_name is not None:
            marker = run_root / "representation" / marker_name
            if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != expected:
                raise RuntimeError(f"frozen semantic marker mismatch: {marker_name}")
    root = run_root / "representation"
    indicator_bundle = json.loads(
        (root / "indicators_frozen.json").read_text(encoding="utf-8")
    )["scenarios"]
    path_bundle = json.loads(
        (root / "candidate_paths.json").read_text(encoding="utf-8")
    )["scenarios"]
    edge_bundle = json.loads(
        (root / "derived_candidate_edges.json").read_text(encoding="utf-8")
    )["scenarios"]
    if not set(indicator_bundle) == set(path_bundle) == set(edge_bundle):
        raise RuntimeError("frozen semantic scenario sets do not match")
    result: dict[str, dict[str, Any]] = {}
    for scenario in sorted(indicator_bundle):
        expected_edges = derive_candidate_edges(path_bundle[scenario])
        if edge_bundle[scenario] != expected_edges:
            raise RuntimeError(
                f"derived candidate relations do not match frozen paths: {scenario}"
            )
        assembled = StructuredRepresentation.model_validate(
            {
                **indicator_bundle[scenario],
                "candidate_paths": path_bundle[scenario],
                "candidate_edges": expected_edges,
            }
        ).model_dump(mode="json", exclude_none=True)
        scenario_path = root / f"{scenario}_representation.json"
        if not scenario_path.is_file():
            raise FileNotFoundError(f"frozen scenario representation is missing: {scenario}")
        stored = StructuredRepresentation.model_validate_json(
            scenario_path.read_text(encoding="utf-8")
        ).model_dump(mode="json", exclude_none=True)
        if stored != assembled:
            raise RuntimeError(f"frozen scenario representation mismatch: {scenario}")
        result[scenario] = assembled
    if not result:
        raise FileNotFoundError("no frozen representations were found")
    return result
