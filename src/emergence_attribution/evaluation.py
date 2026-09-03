"""Strictly separated Full Discovery and Controlled Recovery evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from .reference_truth import reference_processes, reference_relations
from .interventions import aggregate_edge_intervention_evidence
from .temporal import (
    TemporalEdge, load_graph_records, representation_candidates,
    unrestricted_candidates,
)


GRAPH_METHODS = (
    "unrestricted_temporal_search",
    "single_trajectory",
    "trajectory_vote",
    "full_method",
)


def candidate_count_for_method(
    method: str, representation: dict[str, Any]
) -> int:
    """Return the actual hypothesis space evaluated by a temporal method."""

    if method == "unrestricted_temporal_search":
        return len(unrestricted_candidates(representation))
    return len(representation_candidates(representation))


def temporal_qualification_rate(retained_count: int, candidate_count: int) -> float:
    rate = retained_count / max(candidate_count, 1)
    if rate > 1.0 + 1e-12:
        raise RuntimeError("temporal qualification rate exceeds one")
    return float(rate)


def intervention_classification_rates(
    classifications: pd.DataFrame,
) -> tuple[float, float, str]:
    """Summarise attempted interventions under the paper's estimand.

    ``not_applicable`` rows are outside the estimand.  Manipulation failures
    remain in the denominator because they are attempted, applicable
    intervention tests; only directional contradiction contributes to the
    contradiction numerator.
    """

    identity_columns = {"scenario", "source", "target"}
    if not identity_columns.issubset(classifications.columns):
        # Retain a narrow compatibility path for callers that only provide a
        # vector of attempt classes.  Production outputs always carry edge
        # identities and therefore use the frozen edge-level rule below.
        applicable = classifications[
            classifications["primary_class"] != "not_applicable"
        ]
        if applicable.empty:
            return np.nan, np.nan, "no_applicable_intervention_classifications"
        return (
            float(np.mean(applicable["primary_class"] == "supported")),
            float(
                np.mean(
                    applicable["primary_class"] == "directionally_contradicted"
                )
            ),
            "applicable_attempts_without_edge_identity;"
            "manipulation_failure_included",
        )
    edge_level = aggregate_edge_intervention_evidence(classifications)
    applicable = edge_level[edge_level["edge_class"] != "not_applicable"]
    if applicable.empty:
        return np.nan, np.nan, "no_applicable_intervention_classifications"
    return (
        float(np.mean(applicable["edge_class"] == "supported")),
        float(
            np.mean(
                applicable["edge_class"] == "directionally_contradicted"
            )
        ),
        "applicable_edges_after_frozen_attempt_aggregation;"
        "directional_contradiction_precedes_support",
    )


def _controlled_graph_metrics(
    scenario: str, graph: Sequence[TemporalEdge]
) -> tuple[dict[str, float], dict[str, Any]]:
    """Score fixed benchmark identities; no generated-node alignment is involved."""

    allowed = {item.process_id for item in reference_processes(scenario)}
    if any(edge.source not in allowed or edge.target not in allowed for edge in graph):
        raise RuntimeError("controlled graph contains a non-benchmark process id")
    truth = {(item.source, item.target): item for item in reference_relations(scenario)}
    predicted = {(edge.source, edge.target): edge for edge in graph}
    true_pairs, predicted_pairs = set(truth), set(predicted)
    true_positive = true_pairs & predicted_pairs
    false_positive = predicted_pairs - true_pairs
    false_negative = true_pairs - predicted_pairs
    precision = len(true_positive) / len(predicted_pairs) if predicted_pairs else 0.0
    recall = len(true_positive) / len(true_pairs) if true_pairs else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    lag_errors = [
        abs(predicted[pair].lag - truth[pair].lag)
        for pair in true_positive
        if predicted[pair].lag > 0
    ]
    directions = [
        float(np.sign(predicted[pair].beta) == truth[pair].sign)
        for pair in true_positive
        if np.isfinite(predicted[pair].beta) and predicted[pair].beta != 0
    ]
    supports = [edge.support for edge in graph if np.isfinite(edge.support)]
    return (
        {
            "edge_precision": precision,
            "edge_recall": recall,
            "edge_f1": f1,
            "shd": float(len(false_positive) + len(false_negative)),
            "lag_mae": float(np.mean(lag_errors)) if lag_errors else float("nan"),
            "direction_accuracy": float(np.mean(directions)) if directions else float("nan"),
            "stability": float(np.mean(supports)) if supports else float("nan"),
            "retained_edge_count": float(len(graph)),
        },
        {
            "correct_edges": [list(pair) for pair in sorted(true_positive)],
            "added_edges": [list(pair) for pair in sorted(false_positive)],
            "missed_edges": [list(pair) for pair in sorted(false_negative)],
        },
    )


def _runtime_lookup(path: Path) -> dict[tuple[str, str], float]:
    if not path.exists():
        return {}
    frame = pd.read_csv(path)
    return {
        (str(row.scenario), str(row.method)): float(row.runtime_seconds)
        for row in frame.itertuples()
    }


def evaluate_full_discovery(
    run_root: Path,
    representations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Evaluate generated observables without consulting hidden reference truth."""

    graphs = load_graph_records(run_root / "analysis" / "main_graphs.jsonl")
    runtime = _runtime_lookup(run_root / "analysis" / "method_runtime.csv")
    agreements = json.loads(
        (run_root / "representation" / "representation_agreement.json").read_text(
            encoding="utf-8"
        )
    )
    rows: list[dict[str, Any]] = []
    for (scenario, method), graph in sorted(graphs.items()):
        representation = representations[scenario]
        supports = [edge.support for edge in graph if np.isfinite(edge.support)]
        lag_supports = [edge.lag_support for edge in graph if np.isfinite(edge.lag_support)]
        lag_stds = [edge.lag_std for edge in graph if np.isfinite(edge.lag_std)]
        candidate_count = candidate_count_for_method(method, representation)
        if method == "llm_semantic_proposal":
            qualification_rate = float("nan")
            temporal_metric_reason = "not_temporally_qualified"
        else:
            qualification_rate = temporal_qualification_rate(
                len(graph), candidate_count
            )
            temporal_metric_reason = "temporally_qualified"
        rows.append(
            {
                "evaluation_track": "full_discovery",
                "scenario": scenario,
                "method": method,
                "indicator_count": len(representation["indicators"]),
                "candidate_edge_count": candidate_count,
                "retained_edge_count": len(graph),
                "temporal_qualification_rate": qualification_rate,
                "temporal_metric_reason": temporal_metric_reason,
                "stability": float(np.mean(supports)) if supports else float("nan"),
                "lag_support": float(np.mean(lag_supports)) if lag_supports else float("nan"),
                "lag_std": float(np.mean(lag_stds)) if lag_stds else float("nan"),
                "representation_agreement": agreements.get("scenarios", agreements)
                .get(scenario, {})
                .get("computation_agreement"),
                "runtime_seconds": runtime.get((scenario, method), float("nan")),
                "edge_f1": float("nan"),
                "shd": float("nan"),
                "lag_mae": float("nan"),
                "direction_accuracy": float("nan"),
                "reference_metric_reason": "not_applicable_without_generated_to_hidden_alignment",
                "intervention_support_rate": float("nan"),
                "contradiction_rate": float("nan"),
                "intervention_metric_reason": (
                    "not_applicable_semantic_proposal"
                    if method == "llm_semantic_proposal"
                    else "pending_intervention_stage"
                ),
                "mean_ci_width": float("nan"),
            }
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(run_root / "analysis" / "full_discovery_results.csv", index=False)
    return frame


def evaluate_controlled_recovery(run_root: Path) -> pd.DataFrame:
    graphs = load_graph_records(
        run_root / "analysis" / "controlled_recovery_graphs.jsonl"
    )
    runtime = _runtime_lookup(
        run_root / "analysis" / "controlled_recovery_runtime.csv"
    )
    rows: list[dict[str, Any]] = []
    details: dict[str, Any] = {}
    for (scenario, method), graph in sorted(graphs.items()):
        metrics, detail = _controlled_graph_metrics(scenario, graph)
        rows.append(
            {
                "evaluation_track": "controlled_recovery",
                "scenario": scenario,
                "method": method,
                **metrics,
                "runtime_seconds": runtime.get((scenario, method), float("nan")),
                "intervention_f1": float("nan"),
                "eligible_truth_edge_count": float("nan"),
                "supported_truth_edge_count": float("nan"),
                "intervention_precision": float("nan"),
                "intervention_recall": float("nan"),
                "intervention_metric_reason": "pending_intervention_stage",
                "mean_ci_width": float("nan"),
            }
        )
        details[f"{scenario}:{method}"] = detail
    analysis = run_root / "analysis"
    (analysis / "controlled_recovery_graph_evaluation.json").write_text(
        json.dumps(details, indent=2), encoding="utf-8"
    )
    frame = pd.DataFrame(rows)
    frame.to_csv(analysis / "controlled_recovery_results.csv", index=False)
    return frame


def evaluate_main_graphs(
    run_root: Path,
    representations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    full = evaluate_full_discovery(run_root, representations)
    controlled = evaluate_controlled_recovery(run_root)
    combined = pd.concat([full, controlled], ignore_index=True, sort=False)
    combined.to_csv(run_root / "analysis" / "main_results.csv", index=False)
    return combined


def update_intervention_metrics(
    run_root: Path,
    representations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Update natural Full Discovery metrics; never back-fill hidden-truth scores."""

    full_path = run_root / "analysis" / "full_discovery_results.csv"
    full = pd.read_csv(full_path)
    classifications = pd.read_csv(
        run_root / "analysis" / "intervention_classifications.csv"
    )
    effects = pd.read_parquet(run_root / "analysis" / "paired_effects.parquet")
    for scenario in sorted(representations):
        for method in GRAPH_METHODS:
            subset = classifications[
                (classifications["scenario"] == scenario)
                & (classifications["method"] == method)
            ]
            mask = (full["scenario"] == scenario) & (full["method"] == method)
            if subset.empty:
                continue
            support_rate, contradiction_rate, estimand = (
                intervention_classification_rates(subset)
            )
            if not np.isfinite(support_rate):
                full.loc[mask, "intervention_support_rate"] = np.nan
                full.loc[mask, "contradiction_rate"] = np.nan
                full.loc[mask, "intervention_metric_reason"] = (
                    "no_temporally_retained_edges"
                )
                continue
            full.loc[mask, "intervention_support_rate"] = support_rate
            full.loc[mask, "contradiction_rate"] = contradiction_rate
            full.loc[mask, "intervention_metric_reason"] = estimand
            full.loc[mask, "mean_ci_width"] = float(
                effects[effects["scenario"] == scenario]["ci_width"].mean()
            )
    full.to_csv(full_path, index=False)
    controlled = pd.read_csv(run_root / "analysis" / "controlled_recovery_results.csv")
    combined = pd.concat([full, controlled], ignore_index=True, sort=False)
    combined.to_csv(run_root / "analysis" / "main_results.csv", index=False)
    return combined
