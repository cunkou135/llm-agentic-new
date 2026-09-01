"""Final evaluation-only alignment and controlled benchmark metrics."""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .dsl import computation_signature
from .reference_truth import reference_processes, reference_relations
from .temporal import TemporalEdge, load_graph_records


def _indicator_signature(indicator: dict[str, Any]) -> dict[str, Any]:
    return {
        **computation_signature(indicator["computation"]),
        "temporal_aggregation": indicator["temporal_aggregation"],
    }


def align_representation(
    representation: dict[str, Any], scenario: str
) -> dict[str, Any]:
    references = reference_processes(scenario)
    by_signature: dict[str, list[str]] = {}
    for process in references:
        key = json.dumps(process.signature, sort_keys=True, separators=(",", ":"))
        by_signature.setdefault(key, []).append(process.process_id)
    records: list[dict[str, Any]] = []
    mapping: dict[str, str] = {}
    for indicator in representation["indicators"]:
        signature = _indicator_signature(indicator)
        key = json.dumps(signature, sort_keys=True, separators=(",", ":"))
        candidates = by_signature.get(key, [])
        if len(candidates) == 1:
            status, aligned = "matched", candidates[0]
            mapping[indicator["id"]] = aligned
        elif len(candidates) == 0:
            status, aligned = "unmatched", None
        else:
            status, aligned = "ambiguous", None
        records.append(
            {
                "indicator_id": indicator["id"],
                "status": status,
                "aligned_process": aligned,
                "candidate_processes": candidates,
                "signature": signature,
                "alignment_inputs": [
                    "canonical_computation",
                    "source_field_signature",
                    "temporal_aggregation",
                ],
            }
        )
    return {
        "scenario": scenario,
        "mapping": mapping,
        "records": records,
        "matched_count": len(mapping),
        "unmatched_count": sum(item["status"] == "unmatched" for item in records),
        "ambiguous_count": sum(item["status"] == "ambiguous" for item in records),
        "prohibited_alignment_inputs": [
            "branch_id",
            "temporal_result",
            "intervention_result",
            "edge_f1",
            "lag_error",
        ],
    }


def graph_metrics(
    scenario: str,
    graph: Sequence[TemporalEdge],
    alignment: dict[str, Any],
) -> tuple[dict[str, float], dict[str, Any]]:
    mapping = alignment["mapping"]
    truth = reference_relations(scenario)
    truth_by_pair = {(item.source, item.target): item for item in truth}
    predicted_aligned: dict[tuple[str, str], TemporalEdge] = {}
    unmatched_edges: list[tuple[str, str]] = []
    for edge in graph:
        source = mapping.get(edge.source)
        target = mapping.get(edge.target)
        if source is None or target is None:
            unmatched_edges.append((edge.source, edge.target))
            continue
        predicted_aligned[(source, target)] = edge
    predicted_pairs = set(predicted_aligned)
    truth_pairs = set(truth_by_pair)
    true_positive = predicted_pairs & truth_pairs
    false_positive = predicted_pairs - truth_pairs
    false_negative = truth_pairs - predicted_pairs
    fp_count = len(false_positive) + len(unmatched_edges)
    precision = len(true_positive) / (len(true_positive) + fp_count) if true_positive or fp_count else 0.0
    recall = len(true_positive) / len(truth_pairs) if truth_pairs else 0.0
    edge_f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lag_errors = [
        abs(predicted_aligned[pair].lag - truth_by_pair[pair].lag)
        for pair in true_positive
        if predicted_aligned[pair].lag > 0
    ]
    direction_scores = [
        float(np.sign(predicted_aligned[pair].beta) == truth_by_pair[pair].sign)
        for pair in true_positive
        if np.isfinite(predicted_aligned[pair].beta) and predicted_aligned[pair].beta != 0
    ]
    finite_support = [edge.support for edge in graph if np.isfinite(edge.support)]
    metrics = {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": edge_f1,
        "shd": float(len(false_negative) + fp_count),
        "lag_mae": float(np.mean(lag_errors)) if lag_errors else float("nan"),
        "direction_accuracy": float(np.mean(direction_scores))
        if direction_scores
        else float("nan"),
        "stability": float(np.mean(finite_support)) if finite_support else float("nan"),
        "retained_edge_count": float(len(graph)),
        "aligned_edge_count": float(len(predicted_pairs)),
        "unmatched_predicted_edge_count": float(len(unmatched_edges)),
    }
    detail = {
        "correct_edges": [list(item) for item in sorted(true_positive)],
        "added_edges": [list(item) for item in sorted(false_positive)],
        "unmatched_added_edges": [list(item) for item in sorted(unmatched_edges)],
        "missed_edges": [list(item) for item in sorted(false_negative)],
    }
    return metrics, detail


def evaluate_main_graphs(
    run_root: Path,
    representations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    graphs = load_graph_records(run_root / "analysis" / "main_graphs.jsonl")
    alignments = {
        scenario: align_representation(representation, scenario)
        for scenario, representation in sorted(representations.items())
    }
    details: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []
    for (scenario, method), graph in sorted(graphs.items()):
        metrics, detail = graph_metrics(scenario, graph, alignments[scenario])
        rows.append(
            {
                "scenario": scenario,
                "method": method,
                **metrics,
                "intervention_f1": float("nan"),
                "mean_ci_width": float("nan"),
            }
        )
        details[f"{scenario}:{method}"] = detail
    analysis_root = run_root / "analysis"
    (analysis_root / "indicator_alignment.json").write_text(
        json.dumps(alignments, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (analysis_root / "graph_evaluation.json").write_text(
        json.dumps(details, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(analysis_root / "main_results.csv", index=False)
    return frame


def update_intervention_metrics(
    run_root: Path,
    representations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    results_path = run_root / "analysis" / "main_results.csv"
    results = pd.read_csv(results_path)
    classifications = pd.read_csv(
        run_root / "analysis" / "intervention_classifications.csv"
    )
    effects = pd.read_parquet(run_root / "analysis" / "paired_effects.parquet")
    alignments = json.loads(
        (run_root / "analysis" / "indicator_alignment.json").read_text(
            encoding="utf-8"
        )
    )
    for scenario in sorted(representations):
        mapping = alignments[scenario]["mapping"]
        truth_pairs = {
            (item.source, item.target) for item in reference_relations(scenario)
        }
        subset = classifications[classifications["scenario"] == scenario]
        supported_generated = {
            (row.source, row.target)
            for row in subset.itertuples()
            if row.primary_class == "supported"
        }
        supported_aligned = {
            (mapping[source], mapping[target])
            for source, target in supported_generated
            if source in mapping and target in mapping
        }
        unmatched_supported = sum(
            source not in mapping or target not in mapping
            for source, target in supported_generated
        )
        true_positive = supported_aligned & truth_pairs
        false_positive_count = len(supported_aligned - truth_pairs) + unmatched_supported
        false_negative = truth_pairs - supported_aligned
        precision = (
            len(true_positive) / (len(true_positive) + false_positive_count)
            if true_positive or false_positive_count
            else 0.0
        )
        recall = len(true_positive) / len(truth_pairs) if truth_pairs else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        mask = (results["scenario"] == scenario) & (
            results["method"] == "full_method"
        )
        results.loc[mask, "intervention_f1"] = f1
        results.loc[mask, "mean_ci_width"] = float(
            effects[effects["scenario"] == scenario]["ci_width"].mean()
        )
    results.to_csv(results_path, index=False)
    return results
