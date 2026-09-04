"""Isolated Controlled Recovery benchmark.

This module is the only analysis path allowed to combine public trajectories
with withheld reference states.  Nothing here is imported by semantic prompt
construction or Full Discovery indicator compilation.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .dsl import compute_indicator
from .raw_schemas import hidden_reference_schema, public_raw_schema
from .reference_truth import reference_processes, reference_relations
from .simulation import METADATA_COLUMNS, sha256_file, trajectories
from .temporal import (
    discover_bootstrap_graph,
    discover_point_graph_from_blocks,
    discover_vote_graph,
    prepare_target_blocks,
    unrestricted_candidates,
    write_graph_records,
)


def controlled_representation(scenario: str) -> dict[str, Any]:
    """Return a fixed, hidden-known benchmark with deterministic distractors."""

    processes = reference_processes(scenario)
    direct_sources = (
        {
            "s_micro_satisfaction": "tolerance",
            "s_micro_relocation": "move_probability",
            "s_micro_destination_similarity": "destination_preference",
        }
        if scenario == "schelling"
        else {
            "d_micro_assimilation": "confidence_bound",
            "d_micro_shift": "assimilation_strength",
            "d_micro_repulsion": "backfire_threshold",
        }
    )
    indicators = [
        {
            "id": item.process_id,
            "scale": item.scale,
            "computation": item.computation,
            "temporal_aggregation": item.temporal_aggregation,
            "parameter_associations": (
                [{
                    "parameter": direct_sources[item.process_id],
                    "relationship": "direct",
                    "expected_indicator_direction": "unknown",
                    "rationale": "Fixed controlled-benchmark manipulation source.",
                }]
                if item.process_id in direct_sources
                else []
            ),
        }
        for index, item in enumerate(processes)
    ]
    truth = reference_relations(scenario)
    meso_to_macro = {
        item.source: item.target
        for item in truth
        if item.source.startswith(("s_meso_", "d_meso_"))
        and item.target.startswith(("s_macro_", "d_macro_"))
    }
    edges = [
        {
            "source": item.source,
            "target": item.target,
            "hypothesis_group_id": "macro_outcome_" + (
                item.target if item.target in meso_to_macro.values()
                else meso_to_macro[item.target]
            ),
            "expected_direction": "increase" if item.sign > 0 else "decrease",
        }
        for item in truth
    ]
    prefix = "s" if scenario == "schelling" else "d"
    micro_ids = [
        item["id"] for item in indicators if item["scale"] == "micro"
    ]
    for index in range(4):
        edges.extend(
            [
                {
                    "source": micro_ids[index],
                    "target": f"{prefix}_meso_{(index + 1) % 4}",
                    "hypothesis_group_id": f"macro_outcome_{prefix}_macro_{(index + 1) % 4}",
                    "expected_direction": "unknown",
                },
                {
                    "source": f"{prefix}_meso_{index}",
                    "target": f"{prefix}_macro_{(index + 1) % 4}",
                    "hypothesis_group_id": f"macro_outcome_{prefix}_macro_{(index + 1) % 4}",
                    "expected_direction": "unknown",
                },
            ]
        )
    candidate_paths = []
    for meso_source, macro_target in sorted(meso_to_macro.items()):
        for relation in truth:
            if relation.target == meso_source and relation.source.startswith(
                ("s_micro_", "d_micro_")
            ):
                candidate_paths.append(
                    {
                        "path_id": f"controlled_{relation.source}_{meso_source}_{macro_target}",
                        "micro_indicator": relation.source,
                        "meso_indicator": meso_source,
                        "macro_indicator": macro_target,
                    }
                )
    return {
        "scenario": scenario,
        "indicators": indicators,
        "candidate_paths": candidate_paths,
        "candidate_edges": edges,
    }


def controlled_intervention_recovery_metrics(
    truth_edges: set[tuple[str, str]],
    eligible_truth_edges: set[tuple[str, str]],
    supported_edges: set[tuple[str, str]],
) -> dict[str, float]:
    """Score recovery only against truth edges the simulator can manipulate."""

    if not eligible_truth_edges.issubset(truth_edges):
        raise ValueError("eligible intervention truth must be a subset of truth")
    supported_truth = supported_edges & eligible_truth_edges
    precision = (
        len(supported_truth) / len(supported_edges) if supported_edges else 0.0
    )
    recall = (
        len(supported_truth) / len(eligible_truth_edges)
        if eligible_truth_edges
        else np.nan
    )
    f1 = (
        2 * precision * recall / (precision + recall)
        if np.isfinite(recall) and precision + recall
        else 0.0 if np.isfinite(recall) else np.nan
    )
    return {
        "eligible_truth_edge_count": float(len(eligible_truth_edges)),
        "supported_truth_edge_count": float(len(supported_truth)),
        "intervention_precision": float(precision),
        "intervention_recall": float(recall),
        "intervention_f1": float(f1),
    }


def _manifest_records(run_root: Path, complete: bool) -> list[dict[str, Any]]:
    phases = ["baseline", "intervention"] if complete else ["baseline"]
    records: list[dict[str, Any]] = []
    for phase in phases:
        path = run_root / "data" / f"{phase}_simulation_manifest.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        records.extend(payload["task_records"])
    return records


def compile_controlled_dataset(
    run_root: Path,
    *,
    complete: bool,
) -> pd.DataFrame:
    """Compile reference processes without exposing them to Full Discovery."""

    kind = "complete" if complete else "baseline"
    output = run_root / "data" / f"controlled_recovery_trajectories_{kind}.parquet"
    manifest_path = output.with_suffix(".manifest.json")
    if output.exists() != manifest_path.exists():
        raise RuntimeError("controlled dataset checkpoint is incomplete")
    if output.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("sha256") != sha256_file(output):
            raise RuntimeError("controlled dataset checkpoint hash mismatch")
        return pd.read_parquet(output)
    rows: list[dict[str, Any]] = []
    for record in _manifest_records(run_root, complete):
        with np.load(run_root / record["raw_path"], allow_pickle=False) as archive:
            public = {name: archive[name] for name in archive.files}
        with np.load(run_root / record["hidden_path"], allow_pickle=False) as archive:
            hidden = {name: archive[name] for name in archive.files}
        if set(public) & set(hidden):
            raise RuntimeError("controlled compiler received overlapping public and hidden fields")
        raw = {**public, **hidden}
        schema = public_raw_schema(record["scenario"]) + hidden_reference_schema(
            record["scenario"]
        )
        series = {
            process.process_id: compute_indicator(
                process.computation,
                process.temporal_aggregation,
                raw,
                schema,
            )
            for process in reference_processes(record["scenario"])
        }
        for time_index in range(len(next(iter(series.values())))):
            row = {
                key: record[key]
                for key in (
                    "scenario", "seed", "condition", "intervention_parameter",
                    "intervention_direction", "mechanism_variant",
                )
            }
            row["time"] = time_index
            row.update({name: float(value[time_index]) for name, value in series.items()})
            rows.append(row)
    frame = pd.DataFrame(rows).sort_values(
        ["scenario", "condition", "seed", "time"], ignore_index=True
    )
    temporary = output.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, output)
    manifest = {
        "evaluation_track": "controlled_recovery",
        "kind": kind,
        "row_count": len(frame),
        "sha256": sha256_file(output),
        "hidden_inputs_isolated": True,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    return frame


def run_controlled_temporal_stage(
    config: dict[str, Any],
    run_root: Path,
    baseline_dataset: pd.DataFrame,
    workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    temporal = config["temporal"]
    records: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    for scenario in sorted(config["scenarios"]):
        representation = controlled_representation(scenario)
        by_seed = trajectories(baseline_dataset, scenario)
        frames = [by_seed[seed] for seed in sorted(by_seed)]
        candidates = representation["candidate_edges"]

        def progress(done: int, total: int) -> None:
            if progress_callback:
                progress_callback("Controlled recovery bootstrap", done, total, scenario)

        started = time.perf_counter()
        full, _ = discover_bootstrap_graph(
            frames, candidates, int(temporal["maximum_lag"]),
            float(temporal["parent_alpha"]), float(temporal["fdr_alpha"]),
            int(temporal["bootstrap_repetitions"]),
            float(temporal["support_threshold"]), int(config["master_seed"]),
            f"controlled:{scenario}:full", workers, progress,
        )
        runtimes.append({"scenario": scenario, "method": "full_method", "runtime_seconds": time.perf_counter() - started, "runtime_scope": "controlled_temporal_analysis"})
        started = time.perf_counter()
        single = discover_point_graph_from_blocks(
            prepare_target_blocks(frames[:1], candidates, int(temporal["maximum_lag"])),
            float(temporal["parent_alpha"]), float(temporal["fdr_alpha"]),
        )
        runtimes.append({"scenario": scenario, "method": "single_trajectory", "runtime_seconds": time.perf_counter() - started, "runtime_scope": "controlled_temporal_analysis"})
        started = time.perf_counter()
        vote = discover_vote_graph(
            frames, candidates, int(temporal["maximum_lag"]),
            float(temporal["parent_alpha"]), float(temporal["fdr_alpha"]),
            float(temporal["vote_threshold"]),
        )
        runtimes.append({"scenario": scenario, "method": "trajectory_vote", "runtime_seconds": time.perf_counter() - started, "runtime_scope": "controlled_temporal_analysis"})
        started = time.perf_counter()
        unrestricted, _ = discover_bootstrap_graph(
            frames, unrestricted_candidates(representation), int(temporal["maximum_lag"]),
            float(temporal["parent_alpha"]), float(temporal["fdr_alpha"]),
            int(temporal["bootstrap_repetitions"]),
            float(temporal["support_threshold"]), int(config["master_seed"]),
            f"controlled:{scenario}:unrestricted", workers,
        )
        runtimes.append({"scenario": scenario, "method": "unrestricted_temporal_search", "runtime_seconds": time.perf_counter() - started, "runtime_scope": "controlled_temporal_analysis"})
        for method, graph in {
            "unrestricted_temporal_search": unrestricted,
            "single_trajectory": single,
            "trajectory_vote": vote,
            "full_method": full,
        }.items():
            records.append({"scenario": scenario, "method": method, "edges": [asdict(edge) for edge in graph]})
    analysis = run_root / "analysis"
    write_graph_records(analysis / "controlled_recovery_graphs.jsonl", records)
    runtime_frame = pd.DataFrame(runtimes)
    runtime_frame["evaluation_track"] = "controlled_recovery"
    runtime_frame.to_csv(analysis / "controlled_recovery_runtime.csv", index=False)
    return {"graphs": records, "runtime": runtimes}


def run_controlled_intervention_stage(
    config: dict[str, Any],
    run_root: Path,
    complete_dataset: pd.DataFrame,
    workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Apply Stage 3 to the fixed benchmark and update only its result table."""

    from .interventions import (
        CLASSIFICATION_COLUMNS,
        aggregate_edge_intervention_evidence,
        classify_edge_interventions,
        estimate_all_effects,
        intervention_testable_edges,
        not_applicable_classification,
    )
    from .temporal import load_graph_records

    representations = {
        scenario: controlled_representation(scenario)
        for scenario in sorted(config["scenarios"])
    }
    effects, curves = estimate_all_effects(
        complete_dataset, config, representations, workers, paired=True,
        progress_callback=progress_callback,
    )
    graphs = load_graph_records(run_root / "analysis" / "controlled_recovery_graphs.jsonl")
    methods = (
        "unrestricted_temporal_search", "single_trajectory",
        "trajectory_vote", "full_method",
    )
    classified_frames: list[pd.DataFrame] = []
    for scenario, representation in representations.items():
        for method in methods:
            frame = classify_edge_interventions(
                scenario, graphs[(scenario, method)], effects, representation,
                int(config["intervention"]["lag_tolerance"]),
                observational_hard_gates=True,
            )
            if frame.empty:
                frame = pd.DataFrame(
                    [not_applicable_classification(
                        scenario, reason="no_temporally_retained_edges"
                    )],
                    columns=CLASSIFICATION_COLUMNS,
                )
            frame.insert(1, "method", method)
            classified_frames.append(frame)
    classifications = pd.concat(classified_frames, ignore_index=True)
    analysis = run_root / "analysis"
    effects.to_parquet(analysis / "controlled_recovery_paired_effects.parquet", index=False)
    curves.to_parquet(analysis / "controlled_recovery_effect_curves.parquet", index=False)
    classifications.to_csv(
        analysis / "controlled_recovery_intervention_classifications.csv", index=False
    )
    edge_classifications = aggregate_edge_intervention_evidence(classifications)
    edge_classifications.to_csv(
        analysis / "controlled_recovery_edge_intervention_classifications.csv",
        index=False,
    )
    results_path = analysis / "controlled_recovery_results.csv"
    results = pd.read_csv(results_path)
    for scenario in representations:
        truth = {(edge.source, edge.target) for edge in reference_relations(scenario)}
        eligible_truth = intervention_testable_edges(
            sorted(truth), representations[scenario]
        )
        for method in methods:
            subset = edge_classifications[
                (edge_classifications["scenario"] == scenario)
                & (edge_classifications["method"] == method)
            ]
            supported = {
                (row.source, row.target)
                for row in subset.itertuples()
                if row.edge_class == "supported"
            }
            mask = (results["scenario"] == scenario) & (results["method"] == method)
            metrics = controlled_intervention_recovery_metrics(
                truth, eligible_truth, supported
            )
            for name, value in metrics.items():
                results.loc[mask, name] = value
            results.loc[mask, "intervention_metric_reason"] = (
                "eligible_truth_edges_only"
                if eligible_truth
                else "no_truth_edge_has_a_legal_manipulation_route"
            )
            results.loc[mask, "mean_ci_width"] = float(
                effects[effects["scenario"] == scenario]["ci_width"].mean()
            )
    results.to_csv(results_path, index=False)
    return {
        "effect_rows": len(effects),
        "curve_rows": len(curves),
        "classification_rows": len(classifications),
    }
