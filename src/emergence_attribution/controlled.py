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
            "s_micro_similarity": "destination_preference",
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
            "branch_id": f"controlled_{index % 4}",
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
    edges = [
        {
            "source": item.source,
            "target": item.target,
            "branch_id": "controlled",
            "expected_direction": "increase" if item.sign > 0 else "decrease",
        }
        for item in truth
    ]
    prefix = "s" if scenario == "schelling" else "d"
    for index in range(4):
        edges.extend(
            [
                {
                    "source": f"{prefix}_micro_{('satisfaction', 'relocation', 'boundary', 'similarity')[index] if scenario == 'schelling' else ('assimilation', 'shift', 'repulsion', 'rejection')[index]}",
                    "target": f"{prefix}_meso_{(index + 1) % 4}",
                    "branch_id": "controlled",
                    "expected_direction": "unknown",
                },
                {
                    "source": f"{prefix}_meso_{index}",
                    "target": f"{prefix}_macro_{(index + 1) % 4}",
                    "branch_id": "controlled",
                    "expected_direction": "unknown",
                },
            ]
        )
    return {
        "scenario": scenario,
        "indicators": indicators,
        "candidate_edges": edges,
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

    from .interventions import classify_edge_interventions, estimate_all_effects
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
            )
            if frame.empty:
                frame = pd.DataFrame(
                    [{
                        "scenario": scenario,
                        "source": "", "target": "", "parameter": "",
                        "direction": "", "manipulation_success": False,
                        "primary_class": "not_applicable",
                        "underlying_class": "not_applicable",
                        "intervention_scope": "none", "source_onset": -1,
                        "target_onset": -1, "intervention_delay": np.nan,
                        "observational_lag": np.nan, "lag_difference": np.nan,
                        "source_effect": np.nan, "target_effect": np.nan,
                        "not_applicable_reason": "no_temporally_retained_edges",
                    }]
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
    results_path = analysis / "controlled_recovery_results.csv"
    results = pd.read_csv(results_path)
    for scenario in representations:
        truth = {(edge.source, edge.target) for edge in reference_relations(scenario)}
        for method in methods:
            subset = classifications[
                (classifications["scenario"] == scenario)
                & (classifications["method"] == method)
            ]
            supported = {
                (row.source, row.target)
                for row in subset.itertuples()
                if row.primary_class == "supported"
            }
            mask = (results["scenario"] == scenario) & (results["method"] == method)
            if bool((subset["primary_class"] == "not_applicable").all()):
                results.loc[mask, "intervention_f1"] = np.nan
                results.loc[mask, "intervention_metric_reason"] = "no_temporally_retained_edges"
            else:
                tp = len(supported & truth)
                precision = tp / len(supported) if supported else 0.0
                recall = tp / len(truth) if truth else 0.0
                f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
                results.loc[mask, "intervention_f1"] = f1
                results.loc[mask, "intervention_metric_reason"] = ""
            results.loc[mask, "mean_ci_width"] = float(
                effects[effects["scenario"] == scenario]["ci_width"].mean()
            )
    results.to_csv(results_path, index=False)
    return {
        "effect_rows": len(effects),
        "curve_rows": len(curves),
        "classification_rows": len(classifications),
    }
