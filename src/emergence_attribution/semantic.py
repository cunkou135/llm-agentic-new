"""Data-blind semantic generation, repair, selection, and immutable freezing."""

from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path
from typing import Any, Callable

from pydantic import ValidationError

from .dsl import (
    DSLValidationError,
    computation_signature,
    expression_fields,
    grammar_description,
    validate_indicator_expression,
    validate_temporal_aggregation,
)
from .llm_client import LLMResponse, OpenAICompatibleClient, load_llm_config
from .raw_schemas import prompt_scenario_contract, raw_schema
from .schemas import SemanticGeneration


FORBIDDEN_PROMPT_KEYS = {
    "reference_branch",
    "truth_role",
    "truth_edge",
    "truth_lag",
    "truth_sign",
    "edge_f1",
    "shd",
    "temporal_results",
    "intervention_results",
    "baseline_numerical_summary",
    "mechanism_channel",
    "controlled latent",
    "reference process",
    "reference branch",
    "reference edge",
    "reference lag",
    "reference sign",
    "truth role",
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


def _write_immutable_text(path: Path, content: str) -> None:
    """Create an LLM history artifact once, or verify identical existing bytes."""

    if path.exists():
        if path.read_text(encoding="utf-8") != content:
            raise RuntimeError(f"refusing to overwrite completed LLM history: {path}")
        return
    path.write_text(content, encoding="utf-8")


def _write_generation_checkpoint(
    generation_root: Path,
    payload: dict[str, Any],
) -> dict[str, Any]:
    artifacts = {
        path.name: _sha256_path(path)
        for path in sorted(generation_root.glob("*"))
        if path.is_file()
        and (
            path.name == "prompt.md"
            or path.name.startswith("request_round_")
            or path.name.startswith("response_round_")
        )
    }
    completed = {
        **payload,
        "checkpoint": {
            "schema_version": "1.0",
            "payload_sha256": sha256_json(payload),
            "artifact_sha256": artifacts,
            "accepted_generation_sha256": sha256_json(payload["accepted_generation"])
            if payload.get("accepted_generation") is not None
            else None,
        },
    }
    result_path = generation_root / "generation_result.json"
    result_text = json.dumps(completed, indent=2, ensure_ascii=False)
    _write_immutable_text(result_path, result_text)
    _write_immutable_text(
        generation_root / "generation_result.sha256",
        _sha256_path(result_path) + "\n",
    )
    return completed


def _load_verified_generation_checkpoint(
    generation_root: Path,
    scenario: str,
    generation_index: int,
    expected_prompt: str,
    scenario_spec: dict[str, Any],
    representation_config: dict[str, Any],
) -> dict[str, Any] | None:
    result_path = generation_root / "generation_result.json"
    checksum_path = generation_root / "generation_result.sha256"
    if not result_path.exists() and not checksum_path.exists():
        return None
    if not result_path.is_file() or not checksum_path.is_file():
        raise RuntimeError(f"incomplete generation checkpoint: {generation_root}")
    expected_result_hash = checksum_path.read_text(encoding="utf-8").strip()
    if _sha256_path(result_path) != expected_result_hash:
        raise RuntimeError(f"generation result hash mismatch: {generation_root}")
    payload = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = payload.get("checkpoint")
    if not isinstance(checkpoint, dict):
        raise RuntimeError(f"generation checkpoint metadata missing: {generation_root}")
    unhashed = {key: value for key, value in payload.items() if key != "checkpoint"}
    if checkpoint.get("payload_sha256") != sha256_json(unhashed):
        raise RuntimeError(f"generation checkpoint payload hash mismatch: {generation_root}")
    if payload.get("scenario") != scenario or int(payload.get("generation", -1)) != generation_index:
        raise RuntimeError(f"generation checkpoint identity mismatch: {generation_root}")
    for name, expected_hash in checkpoint.get("artifact_sha256", {}).items():
        artifact = generation_root / name
        if not artifact.is_file() or _sha256_path(artifact) != expected_hash:
            raise RuntimeError(f"generation artifact hash mismatch: {artifact}")
    prompt_path = generation_root / "prompt.md"
    if not prompt_path.is_file() or prompt_path.read_text(encoding="utf-8") != expected_prompt:
        raise RuntimeError(f"generation prompt does not match current contract: {generation_root}")
    accepted = payload.get("accepted_generation")
    if payload.get("status") == "accepted":
        if checkpoint.get("accepted_generation_sha256") != sha256_json(accepted):
            raise RuntimeError(f"accepted generation hash mismatch: {generation_root}")
        value = SemanticGeneration.model_validate(accepted)
        validation = validate_generation(
            value, scenario, scenario_spec, representation_config
        )
        if not validation["valid"]:
            raise RuntimeError(
                f"accepted generation no longer validates: {generation_root}: "
                f"{validation['errors']}"
            )
    payload["resumed_from_verified_checkpoint"] = True
    return payload


def build_prompt(
    scenario: str,
    scenario_spec: dict[str, Any],
    representation_config: dict[str, Any],
    prompt_template: str,
) -> tuple[str, str]:
    contract = prompt_scenario_contract(scenario, scenario_spec)
    contract["generic_computation_grammar"] = grammar_description()
    contract["representation_budget"] = representation_config["budget"]
    contract["required_branch_count"] = representation_config["required_branch_count"]
    contract["candidate_edge_bounds"] = {
        "minimum": representation_config["minimum_candidate_edges"],
        "maximum": representation_config["maximum_candidate_edges"],
    }
    contract["structural_constraints"] = [
        "candidate edges may be micro to meso or meso to macro only",
        "candidate edges must remain within a generated branch",
        "every branch must include at least one connected micro to meso to macro path",
        "every micro node must have an outgoing candidate edge to a meso node",
        "every meso node must have an incoming micro edge and an outgoing macro edge",
        "every macro node must have an incoming meso edge",
        "every controllable parameter must have at least one direct micro-level association",
        "each prediction source must be such a direct micro-level association and its ordered path must exist in candidate_edges",
        "prospective validation criteria must be Boolean and edge-list fields, not prose",
        "the model must not infer any result from unprovided simulation statistics",
    ]
    user = (
        "Input contract:\n"
        + json.dumps(contract, indent=2, ensure_ascii=False)
        + "\n\nOutput JSON schema:\n"
        + json.dumps(SemanticGeneration.model_json_schema(), indent=2, ensure_ascii=False)
    )
    combined = (prompt_template + "\n" + user).lower()
    leaked = sorted(key for key in FORBIDDEN_PROMPT_KEYS if key in combined)
    if leaked:
        raise RuntimeError(f"prompt contains forbidden evaluation keys: {leaked}")
    return prompt_template.strip(), user


def _complete_paths(value: SemanticGeneration) -> dict[str, int]:
    lookup = {item.id: item for item in value.representation.indicators}
    first = [
        edge
        for edge in value.representation.candidate_edges
        if lookup[edge.source].scale == "micro" and lookup[edge.target].scale == "meso"
    ]
    second = [
        edge
        for edge in value.representation.candidate_edges
        if lookup[edge.source].scale == "meso" and lookup[edge.target].scale == "macro"
    ]
    counts: dict[str, int] = {}
    for left in first:
        for right in second:
            if left.target == right.source:
                branch = lookup[left.source].branch_id
                counts[branch] = counts.get(branch, 0) + 1
    return counts


def validate_generation(
    value: SemanticGeneration,
    scenario: str,
    scenario_spec: dict[str, Any],
    representation_config: dict[str, Any],
) -> dict[str, Any]:
    errors: list[str] = []
    representation = value.representation
    if representation.scenario != scenario:
        errors.append(f"representation.scenario must be {scenario}")
    budget = representation_config["budget"]
    scale_counts = {
        scale: sum(item.scale == scale for item in representation.indicators)
        for scale in ("micro", "meso", "macro")
    }
    for scale, required in budget.items():
        if scale_counts.get(scale, 0) != int(required):
            errors.append(f"scale {scale} requires exactly {required} indicators")
    branches = sorted({item.branch_id for item in representation.indicators})
    if len(branches) != int(representation_config["required_branch_count"]):
        errors.append(
            f"exactly {representation_config['required_branch_count']} branches are required"
        )
    for branch in branches:
        scales = {
            item.scale for item in representation.indicators if item.branch_id == branch
        }
        if scales != {"micro", "meso", "macro"}:
            errors.append(f"branch {branch} must contain all three scales")
        macro_count = sum(
            item.scale == "macro" and item.branch_id == branch
            for item in representation.indicators
        )
        if macro_count != 1:
            errors.append(f"branch {branch} must contain exactly one macro indicator")
    paths = _complete_paths(value)
    for branch in branches:
        if paths.get(branch, 0) < 1:
            errors.append(f"branch {branch} lacks a complete micro to meso to macro path")
    schema = raw_schema(scenario)
    for indicator in representation.indicators:
        try:
            validate_indicator_expression(indicator.computation, schema)
        except DSLValidationError as exc:
            errors.append(f"indicator {indicator.id}: {exc}")
        try:
            validate_temporal_aggregation(
                indicator.temporal_aggregation.model_dump(mode="json", exclude_none=True)
            )
        except DSLValidationError as exc:
            errors.append(f"indicator {indicator.id}: {exc}")
        actual_fields = expression_fields(indicator.computation)
        if actual_fields != set(indicator.source_fields):
            errors.append(
                f"indicator {indicator.id}: source_fields must exactly match AST fields {sorted(actual_fields)}"
            )
    parameter_names = set(scenario_spec["interventions"])
    proposed_parameters: set[str] = set()
    direct_micro_sources: dict[str, set[str]] = {name: set() for name in parameter_names}
    for indicator in representation.indicators:
        for association in indicator.parameter_associations:
            if association.parameter not in parameter_names:
                errors.append(
                    f"indicator {indicator.id}: unknown parameter {association.parameter}"
                )
            else:
                proposed_parameters.add(association.parameter)
                if association.relationship == "direct" and indicator.scale == "micro":
                    direct_micro_sources[association.parameter].add(indicator.id)
    if representation_config.get("require_all_parameters_associated", False):
        missing = sorted(parameter_names - proposed_parameters)
        if missing:
            errors.append(f"missing model-proposed parameter associations: {missing}")
        missing_direct = sorted(
            name for name, sources in direct_micro_sources.items() if not sources
        )
        if missing_direct:
            errors.append(
                f"parameters require at least one direct micro source: {missing_direct}"
            )
    edge_pairs = {(edge.source, edge.target) for edge in representation.candidate_edges}
    incoming = {item.id: 0 for item in representation.indicators}
    outgoing = {item.id: 0 for item in representation.indicators}
    for source, target in edge_pairs:
        outgoing[source] += 1
        incoming[target] += 1
    for indicator in representation.indicators:
        if indicator.scale == "micro" and outgoing[indicator.id] < 1:
            errors.append(f"micro indicator {indicator.id} needs an outgoing meso edge")
        if indicator.scale == "meso" and (
            incoming[indicator.id] < 1 or outgoing[indicator.id] < 1
        ):
            errors.append(f"meso indicator {indicator.id} needs incoming and outgoing edges")
        if indicator.scale == "macro" and incoming[indicator.id] < 1:
            errors.append(f"macro indicator {indicator.id} needs an incoming meso edge")
    edge_count = len(representation.candidate_edges)
    minimum_edges = int(representation_config["minimum_candidate_edges"])
    maximum_edges = int(representation_config["maximum_candidate_edges"])
    if not minimum_edges <= edge_count <= maximum_edges:
        errors.append(
            f"candidate edge count {edge_count} must be within [{minimum_edges}, {maximum_edges}]"
        )
    indicator_ids = {item.id for item in representation.indicators}
    prediction_ids: set[str] = set()
    for prediction in value.prospective_predictions:
        if prediction.prediction_id in prediction_ids:
            errors.append(f"duplicate prediction id {prediction.prediction_id}")
        prediction_ids.add(prediction.prediction_id)
        if prediction.parameter not in parameter_names:
            errors.append(f"prediction {prediction.prediction_id}: unknown parameter")
        elif prediction.source_indicator not in direct_micro_sources[prediction.parameter]:
            errors.append(
                f"prediction {prediction.prediction_id}: source must be a direct micro association for its parameter"
            )
        referenced = {prediction.source_indicator, *prediction.downstream_indicators}
        unknown = sorted(referenced - indicator_ids)
        if unknown:
            errors.append(
                f"prediction {prediction.prediction_id}: unknown indicators {unknown}"
            )
        if prediction.expected_temporal_order != [
            prediction.source_indicator,
            *prediction.downstream_indicators,
        ]:
            errors.append(
                f"prediction {prediction.prediction_id}: temporal order must list source then downstream indicators"
            )
        required_path = list(zip(
            prediction.expected_temporal_order,
            prediction.expected_temporal_order[1:],
        ))
        missing_path = [pair for pair in required_path if pair not in edge_pairs]
        if missing_path:
            errors.append(
                f"prediction {prediction.prediction_id}: ordered path is absent from candidate graph {missing_path}"
            )
        criteria = prediction.validation_criteria
        required_edges = {
            (item.source, item.target) for item in criteria.required_candidate_edges
        }
        if not criteria.required_source_response:
            errors.append(
                f"prediction {prediction.prediction_id}: required_source_response must be true"
            )
        if not criteria.required_temporal_order:
            errors.append(
                f"prediction {prediction.prediction_id}: required_temporal_order must be true"
            )
        if not all(criteria.required_downstream_response):
            errors.append(
                f"prediction {prediction.prediction_id}: all downstream responses must be required"
            )
        if not set(required_path).issubset(required_edges):
            errors.append(
                f"prediction {prediction.prediction_id}: validation criteria omit candidate path edges"
            )
    signatures = [
        json.dumps(computation_signature(item.computation), sort_keys=True)
        for item in representation.indicators
    ]
    if len(signatures) != len(set(signatures)):
        errors.append("indicator computations must have unique canonical signatures")
    lower_boundary = representation.interpretation_boundary.lower()
    if "temporal" not in lower_boundary or "intervention" not in lower_boundary:
        errors.append("interpretation_boundary must distinguish later temporal and intervention evidence")
    return {
        "valid": not errors,
        "errors": errors,
        "scale_counts": scale_counts,
        "branch_count": len(branches),
        "complete_paths": paths,
        "source_field_diversity": len(
            set().union(*(expression_fields(item.computation) for item in representation.indicators))
        ),
        "parameter_coverage": sorted(proposed_parameters),
        "indicator_count": len(representation.indicators),
        "candidate_edge_count": len(representation.candidate_edges),
        "direct_micro_parameter_sources": {
            name: sorted(sources) for name, sources in direct_micro_sources.items()
        },
    }


def _parse_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        lines = candidate.splitlines()
        candidate = "\n".join(lines[1:-1]).strip()
    start, end = candidate.find("{"), candidate.rfind("}")
    if start < 0 or end < start:
        raise ValueError("response does not contain a JSON object")
    return json.loads(candidate[start : end + 1])


def run_generation(
    scenario: str,
    generation_index: int,
    experiment_config: dict[str, Any],
    llm_config: dict[str, Any],
    prompt_template: str,
    output_root: Path,
    completion: Callable[[str, str], LLMResponse] | None = None,
) -> dict[str, Any]:
    scenario_spec = experiment_config["scenarios"][scenario]
    rep_config = experiment_config["representation"]
    system, user = build_prompt(scenario, scenario_spec, rep_config, prompt_template)
    generation_root = output_root / scenario / f"generation_{generation_index:02d}"
    generation_root.mkdir(parents=True, exist_ok=True)
    prompt_content = f"# System\n\n{system}\n\n# User\n\n{user}\n"
    _write_immutable_text(generation_root / "prompt.md", prompt_content)
    checkpoint = _load_verified_generation_checkpoint(
        generation_root,
        scenario,
        generation_index,
        prompt_content,
        scenario_spec,
        rep_config,
    )
    if checkpoint is not None:
        return checkpoint
    if completion is None:
        client = OpenAICompatibleClient(llm_config)
        completion = client.complete_json
    calls: list[dict[str, Any]] = []
    prior_text = ""
    errors: list[str] = []
    accepted: SemanticGeneration | None = None
    accepted_validation: dict[str, Any] | None = None
    started = time.perf_counter()
    for repair_round in range(int(rep_config["maximum_repair_rounds"]) + 1):
        repair = ""
        if repair_round:
            repair = (
                "\n\nThe previous JSON was rejected only for the following schema or executability errors:\n"
                + json.dumps(errors, indent=2, ensure_ascii=False)
                + "\nReturn a complete corrected object. Do not infer or request simulation outcomes.\nPrevious object:\n"
                + prior_text
            )
        request = user + repair
        request_path = generation_root / f"request_round_{repair_round:02d}.md"
        _write_immutable_text(
            request_path,
            f"# System\n\n{system}\n\n# User\n\n{request}\n",
        )
        call_started = time.perf_counter()
        response_path = generation_root / f"response_round_{repair_round:02d}.json"
        existing_response = (
            json.loads(response_path.read_text(encoding="utf-8"))
            if response_path.is_file()
            else None
        )
        try:
            if existing_response is None:
                response = completion(system, request)
            else:
                previous_call = existing_response["call"]
                response = LLMResponse(
                    text=str(existing_response["raw_text"]),
                    input_tokens=int(previous_call["input_tokens"]),
                    output_tokens=int(previous_call["output_tokens"]),
                    model=str(previous_call["model"]),
                )
            prior_text = response.text
            parsed = _parse_json(response.text)
            value = SemanticGeneration.model_validate(parsed)
            validation = validate_generation(value, scenario, scenario_spec, rep_config)
            errors = list(validation["errors"])
            status = "accepted" if validation["valid"] else "validator_rejected"
            if validation["valid"]:
                accepted = value
                accepted_validation = validation
        except (ValidationError, ValueError, json.JSONDecodeError) as exc:
            parsed = None
            validation = {"valid": False, "errors": [f"{type(exc).__name__}: {exc}"]}
            errors = list(validation["errors"])
            status = "schema_rejected"
            if existing_response is None:
                response = LLMResponse("", 0, 0, str(llm_config["model"]))
        call = (
            existing_response["call"]
            if existing_response is not None
            else {
                "repair_round": repair_round,
                "status": status,
                "duration_seconds": time.perf_counter() - call_started,
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "model": response.model,
                "validation": validation,
            }
        )
        if call["status"] != status:
            raise RuntimeError(f"stored response validation status changed: {response_path}")
        calls.append(call)
        if existing_response is None:
            _write_immutable_text(
                response_path,
                json.dumps(
                {"raw_text": response.text, "parsed_json": parsed, "call": call},
                indent=2,
                ensure_ascii=False,
                ),
            )
        if accepted is not None:
            break
    payload = {
        "scenario": scenario,
        "generation": generation_index,
        "status": "accepted" if accepted is not None else "rejected",
        "duration_seconds": time.perf_counter() - started,
        "repair_rounds": sum(call["status"] != "accepted" for call in calls),
        "accepted_generation": accepted.model_dump(mode="json", exclude_none=True) if accepted else None,
        "validation": accepted_validation,
        "calls": calls,
    }
    return _write_generation_checkpoint(generation_root, payload)


def _jaccard(first: set[Any], second: set[Any]) -> float:
    if not first and not second:
        return 1.0
    return len(first & second) / len(first | second)


def _agreement(accepted: list[dict[str, Any]]) -> dict[str, Any]:
    if len(accepted) < 2:
        return {"computation_signature_jaccard": None, "semantic_edge_jaccard": None}
    signatures: list[set[str]] = []
    edges: list[set[tuple[str, str]]] = []
    for item in accepted:
        generation = item["accepted_generation"]
        by_id = {
            node["id"]: json.dumps(computation_signature(node["computation"]), sort_keys=True)
            for node in generation["representation"]["indicators"]
        }
        signatures.append(set(by_id.values()))
        edges.append(
            {(by_id[edge["source"]], by_id[edge["target"]]) for edge in generation["representation"]["candidate_edges"]}
        )
    pairs = list(combinations(range(len(accepted)), 2))
    return {
        "computation_signature_jaccard": sum(
            _jaccard(signatures[a], signatures[b]) for a, b in pairs
        )
        / len(pairs),
        "semantic_edge_jaccard": sum(_jaccard(edges[a], edges[b]) for a, b in pairs)
        / len(pairs),
    }


def _selection_key(payload: dict[str, Any]) -> tuple[Any, ...]:
    validation = payload["validation"]
    canonical_hash = sha256_json(payload["accepted_generation"])
    return (
        -int(validation["source_field_diversity"]),
        -sum(int(value) for value in validation["complete_paths"].values()),
        int(payload["repair_rounds"]),
        canonical_hash,
    )


def run_semantic_stage(
    experiment_config: dict[str, Any],
    llm_config_path: Path,
    run_root: Path,
    prompt_template_path: Path,
    workers: int,
    progress_callback: Callable[[str, int, int], None] | None = None,
    completion_provider: Callable[[str, int], Callable[[str, str], LLMResponse]] | None = None,
) -> dict[str, Any]:
    llm_config = load_llm_config(
        llm_config_path, require_key=completion_provider is None
    )
    prompt_template = prompt_template_path.read_text(encoding="utf-8")
    output_root = run_root / "llm"
    output_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (scenario, index)
        for scenario in experiment_config["scenarios"]
        for index in range(int(experiment_config["representation"]["independent_generations"]))
    ]
    results: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, min(workers, len(jobs)))) as pool:
        futures = {
            pool.submit(
                run_generation,
                scenario,
                index,
                experiment_config,
                llm_config,
                prompt_template,
                output_root,
                completion_provider(scenario, index) if completion_provider else None,
            ): (scenario, index)
            for scenario, index in jobs
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress_callback:
                progress_callback("LLM generation", len(results), len(jobs))
    representation_root = run_root / "representation"
    representation_root.mkdir(parents=True, exist_ok=True)
    validation_bundle: dict[str, Any] = {}
    agreement_bundle: dict[str, Any] = {}
    predictions_bundle: dict[str, Any] = {"schema_version": "1.0", "scenarios": {}}
    for scenario in experiment_config["scenarios"]:
        accepted = [
            item
            for item in results
            if item["scenario"] == scenario and item["status"] == "accepted"
        ]
        if not accepted:
            raise RuntimeError(f"no valid formal semantic generation for {scenario}")
        selected = min(accepted, key=_selection_key)
        generation = selected["accepted_generation"]
        representation = generation["representation"]
        representation_path = representation_root / f"{scenario}_representation.json"
        representation_path.write_text(
            json.dumps(representation, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        predictions_bundle["scenarios"][scenario] = generation["prospective_predictions"]
        validation_bundle[scenario] = {
            "selected_generation": selected["generation"],
            "selection_rule": experiment_config["representation"]["selection_rule"],
            "selection_key": list(_selection_key(selected)),
            "selection_reason": "selected without simulation statistics or evaluation outcomes",
            "representation_sha256": hashlib.sha256(representation_path.read_bytes()).hexdigest(),
            "accepted_generation_count": len(accepted),
            "all_generations": [
                {
                    "generation": item["generation"],
                    "status": item["status"],
                    "repair_rounds": item["repair_rounds"],
                    "validation": item["validation"],
                }
                for item in sorted(
                    [entry for entry in results if entry["scenario"] == scenario],
                    key=lambda entry: entry["generation"],
                )
            ],
        }
        agreement_bundle[scenario] = _agreement(accepted)
    predictions_path = representation_root / "prospective_predictions.json"
    predictions_path.write_text(
        json.dumps(predictions_bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    validation_bundle["prospective_predictions_sha256"] = hashlib.sha256(
        predictions_path.read_bytes()
    ).hexdigest()
    (representation_root / "representation_validation.json").write_text(
        json.dumps(validation_bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (representation_root / "representation_agreement.json").write_text(
        json.dumps(agreement_bundle, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return validation_bundle


def load_frozen_representations(run_root: Path) -> dict[str, dict[str, Any]]:
    result: dict[str, dict[str, Any]] = {}
    for path in sorted((run_root / "representation").glob("*_representation.json")):
        scenario = path.name.removesuffix("_representation.json")
        result[scenario] = json.loads(path.read_text(encoding="utf-8"))
    if not result:
        raise FileNotFoundError("no frozen representations were found")
    return result
