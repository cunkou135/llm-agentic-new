"""Public, data-blind agreement metrics for independent semantic generations."""

from __future__ import annotations

import json
import os
from itertools import combinations
from pathlib import Path
from typing import Any

import pandas as pd

from .dsl import computation_signature
from .raw_schemas import public_raw_schema


PAIRWISE_COLUMNS = [
    "scenario",
    "generation_a",
    "generation_b",
    "role_a",
    "role_b",
    "comparison_group",
    "computation_signature_jaccard",
    "scale_assignment_jaccard",
    "source_family_jaccard",
    "indicator_structure_jaccard",
    "direct_parameter_source_jaccard",
    "macro_concept_jaccard",
]

METRIC_COLUMNS = PAIRWISE_COLUMNS[6:]


def _canonical(value: Any) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def _jaccard(first: set[Any], second: set[Any]) -> float:
    if not first and not second:
        return 1.0
    return len(first & second) / len(first | second)


def _families(scenario: str, source_fields: list[str]) -> tuple[str, ...]:
    lookup = {
        str(item["field_name"]): str(item["primitive_family"])
        for item in public_raw_schema(scenario)
    }
    return tuple(sorted({lookup[name] for name in source_fields if name in lookup}))


def _generation_features(scenario: str, payload: dict[str, Any]) -> dict[str, set[Any]]:
    representation = payload["accepted_generation"]
    indicators = representation["indicators"]
    by_id = {str(item["id"]): item for item in indicators}
    signatures = {
        node_id: _canonical(computation_signature(item["computation"]))
        for node_id, item in by_id.items()
    }
    families = {
        node_id: _families(scenario, list(item.get("source_fields", [])))
        for node_id, item in by_id.items()
    }
    indicator_structures = {
        _canonical((str(item["scale"]), families[node_id], signatures[node_id]))
        for node_id, item in by_id.items()
    }
    direct_parameter_sources = {
        (
            str(association["parameter"]),
            signatures[node_id],
            families[node_id],
        )
        for node_id, item in by_id.items()
        if str(item["scale"]) == "micro"
        for association in item.get("parameter_associations", [])
        if association.get("relationship") == "direct"
    }
    return {
        "computation_signature": set(signatures.values()),
        "scale_assignment": {
            (signature, str(by_id[node_id]["scale"]))
            for node_id, signature in signatures.items()
        },
        "source_family": {
            (str(by_id[node_id]["scale"]), family)
            for node_id in by_id
            for family in families[node_id]
        },
        "indicator_structure": indicator_structures,
        "direct_parameter_source": direct_parameter_sources,
        "macro_concept": {
            (signatures[node_id], families[node_id])
            for node_id, item in by_id.items()
            if str(item["scale"]) == "macro"
        },
    }


def _comparison_group(first_role: str, second_role: str) -> str:
    if first_role == second_role == "selection_eligible":
        return "within_selection"
    if first_role == second_role == "replication_only":
        return "within_replication_only"
    return "selection_vs_replication"


def write_replication_agreement(
    run_root: Path,
    results: list[dict[str, Any]],
    scenario_names: list[str],
    selection_generations: int,
) -> dict[str, Any]:
    """Persist pairwise public-semantic agreement without consulting simulation data."""

    rows: list[dict[str, Any]] = []
    scenario_summaries: dict[str, Any] = {}
    for scenario in sorted(scenario_names):
        scenario_results = sorted(
            (
                item
                for item in results
                if item.get("scenario") == scenario
                and item.get("status") == "accepted"
                and item.get("accepted_generation") is not None
            ),
            key=lambda item: int(item["generation"]),
        )
        feature_cache = {
            int(item["generation"]): _generation_features(scenario, item)
            for item in scenario_results
        }
        for first, second in combinations(scenario_results, 2):
            generation_a = int(first["generation"])
            generation_b = int(second["generation"])
            role_a = (
                "selection_eligible"
                if generation_a < selection_generations
                else "replication_only"
            )
            role_b = (
                "selection_eligible"
                if generation_b < selection_generations
                else "replication_only"
            )
            left, right = feature_cache[generation_a], feature_cache[generation_b]
            row = {
                "scenario": scenario,
                "generation_a": generation_a,
                "generation_b": generation_b,
                "role_a": role_a,
                "role_b": role_b,
                "comparison_group": _comparison_group(role_a, role_b),
            }
            for feature, column in (
                ("computation_signature", "computation_signature_jaccard"),
                ("scale_assignment", "scale_assignment_jaccard"),
                ("source_family", "source_family_jaccard"),
                ("indicator_structure", "indicator_structure_jaccard"),
                ("direct_parameter_source", "direct_parameter_source_jaccard"),
                ("macro_concept", "macro_concept_jaccard"),
            ):
                row[column] = _jaccard(left[feature], right[feature])
            rows.append(row)
        scenario_frame = pd.DataFrame(
            [row for row in rows if row["scenario"] == scenario],
            columns=PAIRWISE_COLUMNS,
        )
        by_group: dict[str, Any] = {}
        for group in (
            "within_selection",
            "selection_vs_replication",
            "within_replication_only",
        ):
            subset = scenario_frame[scenario_frame["comparison_group"] == group]
            by_group[group] = {
                "pair_count": int(len(subset)),
                **{
                    column: (
                        float(subset[column].mean()) if len(subset) else None
                    )
                    for column in METRIC_COLUMNS
                },
            }
        scenario_summaries[scenario] = {
            "accepted_generation_count": len(scenario_results),
            "selection_eligible_accepted": sum(
                int(item["generation"]) < selection_generations
                for item in scenario_results
            ),
            "replication_only_accepted": sum(
                int(item["generation"]) >= selection_generations
                for item in scenario_results
            ),
            "agreement_by_comparison_group": by_group,
        }
    pairwise = pd.DataFrame(rows, columns=PAIRWISE_COLUMNS)
    representation_root = run_root / "representation"
    representation_root.mkdir(parents=True, exist_ok=True)
    pairwise_path = representation_root / "replication_pairwise.csv"
    temporary_csv = pairwise_path.with_suffix(".tmp.csv")
    pairwise.to_csv(temporary_csv, index=False)
    os.replace(temporary_csv, pairwise_path)
    payload = {
        "schema_version": "1.0",
        "evaluation_track": "semantic_replication",
        "data_blind": True,
        "hidden_truth_used": False,
        "selection_generations": selection_generations,
        "selection_rule_boundary": (
            "only generation indices below selection_generations are eligible; "
            "replication-only generations cannot affect selection"
        ),
        "scenarios": scenario_summaries,
    }
    agreement_path = representation_root / "replication_agreement.json"
    temporary_json = agreement_path.with_suffix(".tmp.json")
    temporary_json.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary_json, agreement_path)
    return payload
