"""Pre-specified ablations, sensitivity analyses, and scalability checks."""

from __future__ import annotations

import json
import copy
import hashlib
import time
from concurrent.futures import Executor, ProcessPoolExecutor, as_completed
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .interventions import (
    aggregate_edge_intervention_evidence,
    classify_edge_interventions,
    estimate_all_effects,
)
from .predefined import predefined_representation
from .reference_truth import mechanism_target_for_variant
from .simulation import compile_indicator_dataset, trajectories
from .temporal import (
    TemporalEdge,
    discover_bootstrap_graph,
    discover_point_graph_from_blocks,
    discover_vote_graph,
    load_graph_records,
    prepare_target_blocks,
    representation_candidates,
    stable_seed,
    unrestricted_candidates,
)


def _metric_row(
    scenario: str,
    variant: str,
    graph: Sequence[TemporalEdge],
    representation: dict[str, Any],
    *,
    candidate_count_override: int | None = None,
) -> dict[str, Any]:
    candidates = (
        int(candidate_count_override)
        if candidate_count_override is not None
        else len(representation_candidates(representation))
    )
    supports = [edge.support for edge in graph if np.isfinite(edge.support)]
    lag_supports = [edge.lag_support for edge in graph if np.isfinite(edge.lag_support)]
    lag_stds = [edge.lag_std for edge in graph if np.isfinite(edge.lag_std)]
    qualification_rate = len(graph) / max(candidates, 1)
    if qualification_rate > 1.0 + 1e-12:
        raise RuntimeError(f"temporal qualification rate exceeds one for {scenario}:{variant}")
    return {
        "evaluation_track": "full_discovery",
        "scenario": scenario,
        "variant": variant,
        "candidate_edge_count": candidates,
        "retained_edge_count": len(graph),
        "temporal_qualification_rate": qualification_rate,
        "stability": float(np.mean(supports)) if supports else np.nan,
        "lag_support": float(np.mean(lag_supports)) if lag_supports else np.nan,
        "lag_std": float(np.mean(lag_stds)) if lag_stds else np.nan,
        "edge_f1": np.nan,
        "shd": np.nan,
        "lag_mae": np.nan,
        "direction_accuracy": np.nan,
        "reference_metric_reason": "not_applicable_without_generated_to_hidden_alignment",
    }


def _merge_metric_fields(
    metadata: dict[str, Any], metrics: dict[str, Any]
) -> dict[str, Any]:
    overlap = sorted(set(metadata) & set(metrics))
    if overlap:
        raise ValueError(f"metric fields would be silently overwritten: {overlap}")
    return {**metadata, **metrics}


def _support_rate(frame: pd.DataFrame) -> float:
    if frame.empty or "primary_class" not in frame.columns:
        return float("nan")
    edge_level = aggregate_edge_intervention_evidence(frame)
    applicable = edge_level[edge_level["edge_class"] != "not_applicable"]
    if applicable.empty:
        return float("nan")
    return float(np.mean(applicable["edge_class"] == "supported"))


def _robustness_bootstrap_job(payload: dict[str, Any]) -> dict[str, Any]:
    """Execute one outer-parallel robustness condition without nested pools."""

    started = time.perf_counter()
    if payload.get("point_only", False):
        blocks = prepare_target_blocks(
            payload["frames"], payload["candidates"], payload["maximum_lag"]
        )
        graph = discover_point_graph_from_blocks(
            blocks, payload["parent_alpha"], payload["fdr_alpha"]
        )
        summary = None
    else:
        graph, summary = discover_bootstrap_graph(
            payload["frames"], payload["candidates"], payload["maximum_lag"],
            payload["parent_alpha"], payload["fdr_alpha"],
            payload["bootstrap_repetitions"], payload["support_threshold"],
            payload["master_seed"], payload["seed_label"], 1,
        )
    vote = (
        discover_vote_graph(
            payload["frames"], payload["candidates"], payload["maximum_lag"],
            payload["parent_alpha"], payload["fdr_alpha"],
            payload["vote_threshold"],
        )
        if payload.get("include_vote", False)
        else None
    )
    return {
        "job_index": payload["job_index"],
        "graph": graph,
        "summary": summary,
        "vote": vote,
        "runtime_seconds": time.perf_counter() - started,
    }


def _execute_robustness_bootstrap_jobs(
    payloads: list[dict[str, Any]],
    executor: Executor | None,
    progress_callback: Callable[[int, int, dict[str, Any]], None] | None = None,
) -> list[dict[str, Any]]:
    results: list[dict[str, Any]] = []
    if executor is None:
        for payload in payloads:
            results.append(_robustness_bootstrap_job(payload))
            if progress_callback:
                progress_callback(len(results), len(payloads), payload)
    else:
        futures = {
            executor.submit(_robustness_bootstrap_job, payload): payload
            for payload in payloads
        }
        for future in as_completed(futures):
            results.append(future.result())
            if progress_callback:
                progress_callback(len(results), len(payloads), futures[future])
    return sorted(results, key=lambda item: int(item["job_index"]))


def run_functional_ablations(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    baseline_dataset: pd.DataFrame,
    complete_dataset: pd.DataFrame,
    workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> pd.DataFrame:
    graphs = load_graph_records(run_root / "analysis" / "main_graphs.jsonl")
    main_results = pd.read_csv(run_root / "analysis" / "main_results.csv")
    effects = pd.read_parquet(run_root / "analysis" / "paired_effects.parquet")
    classifications = pd.read_csv(
        run_root / "analysis" / "intervention_classifications.csv"
    )
    bootstrap = json.loads(
        (run_root / "analysis" / "bootstrap_summary.json").read_text(encoding="utf-8")
    )
    rows: list[dict[str, Any]] = []
    for scenario, representation in sorted(representations.items()):
        full_row = main_results[
            (main_results["scenario"] == scenario)
            & (main_results["method"] == "full_method")
            & (main_results["evaluation_track"] == "full_discovery")
        ].iloc[0].to_dict()
        full_row.update({"variant": "full_method"})
        rows.append(full_row)
        single_row = main_results[
            (main_results["scenario"] == scenario)
            & (main_results["method"] == "single_trajectory")
            & (main_results["evaluation_track"] == "full_discovery")
        ].iloc[0].to_dict()
        single_classes = classifications[
            (classifications["scenario"] == scenario)
            & (classifications["method"] == "single_trajectory")
        ]
        single_row.update(
            {
                "variant": "without_joint_trajectories",
                "intervention_support_rate": _support_rate(single_classes),
                "mean_ci_width": float(
                    effects[effects["scenario"] == scenario]["ci_width"].mean()
                ),
            }
        )
        rows.append(single_row)
        point_graph = [
            TemporalEdge(**edge) for edge in bootstrap[scenario]["point_graph"]
        ]
        point_row = _metric_row(
            scenario, "without_bootstrap", point_graph, representation
        )
        point_row.update(
            {
                "intervention_support_rate": _support_rate(
                    classify_edge_interventions(
                        scenario, point_graph, effects, representation,
                        int(config["intervention"]["lag_tolerance"]),
                    )
                ),
                "mean_ci_width": float(
                    effects[effects["scenario"] == scenario]["ci_width"].mean()
                ),
            }
        )
        rows.append(point_row)
        unrestricted_row = main_results[
            (main_results["scenario"] == scenario)
            & (main_results["method"] == "unrestricted_temporal_search")
            & (main_results["evaluation_track"] == "full_discovery")
        ].iloc[0].to_dict()
        unrestricted_row.update({"variant": "without_structured_representation"})
        rows.append(unrestricted_row)
    unpaired_effects, _ = estimate_all_effects(
        complete_dataset,
        config,
        representations,
        workers,
        paired=False,
        progress_callback=progress_callback,
    )
    for scenario, representation in sorted(representations.items()):
        graph = graphs[(scenario, "full_method")]
        row = _metric_row(scenario, "without_paired_seeds", graph, representation)
        unpaired_classification = classify_edge_interventions(
            scenario,
            graph,
            unpaired_effects,
            representation,
            int(config["intervention"]["lag_tolerance"]),
        )
        row.update(
            {
                "intervention_support_rate": _support_rate(unpaired_classification),
                "mean_ci_width": float(
                    unpaired_effects[unpaired_effects["scenario"] == scenario][
                        "ci_width"
                    ].mean()
                ),
            }
        )
        rows.append(row)
    fixed_representations = {
        scenario: predefined_representation(scenario)
        for scenario in sorted(representations)
    }
    fixed_dataset = compile_indicator_dataset(
        config,
        run_root,
        fixed_representations,
        workers,
        complete=False,
        output_stem="predefined_observable_trajectories_baseline",
        progress_callback=progress_callback,
    )
    for scenario, representation in sorted(fixed_representations.items()):
        by_seed = trajectories(fixed_dataset, scenario)
        frames = [by_seed[seed] for seed in sorted(by_seed)]
        fixed_graph, _ = discover_bootstrap_graph(
            frames,
            representation_candidates(representation),
            int(config["temporal"]["maximum_lag"]),
            float(config["temporal"]["parent_alpha"]),
            float(config["temporal"]["fdr_alpha"]),
            int(config["temporal"]["bootstrap_repetitions"]),
            float(config["temporal"]["support_threshold"]),
            int(config["master_seed"]),
            f"{scenario}:predefined",
            workers,
        )
        row = _metric_row(
            scenario, "predefined_observable_baseline", fixed_graph, representation
        )
        row.update({"intervention_support_rate": np.nan, "mean_ci_width": np.nan})
        rows.append(row)
    frame = pd.DataFrame(rows)
    frame.to_csv(run_root / "analysis" / "functional_ablations.csv", index=False)
    return frame


def _bootstrap_metric_interval(
    summary: dict[str, Any],
    metric: str,
    candidate_count: int,
) -> tuple[float, float]:
    edge_sets = [
        {(edge["source"], edge["target"]) for edge in item["edges"]}
        for item in summary["edge_sets"]
    ]
    frequencies: dict[tuple[str, str], float] = {}
    for edges in edge_sets:
        for pair in edges:
            frequencies[pair] = frequencies.get(pair, 0.0) + 1.0 / max(len(edge_sets), 1)
    values: list[float] = []
    for pairs in edge_sets:
        if metric == "temporal_qualification_rate":
            values.append(len(pairs) / max(candidate_count, 1))
        elif metric == "stability":
            values.append(
                float(np.mean([frequencies[pair] for pair in pairs]))
                if pairs else 0.0
            )
        else:
            raise KeyError(metric)
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return np.nan, np.nan
    return float(np.quantile(finite, 0.025)), float(np.quantile(finite, 0.975))


def run_data_efficiency(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    baseline_dataset: pd.DataFrame,
    workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
    *,
    executor: Executor | None = None,
) -> pd.DataFrame:
    jobs: list[dict[str, Any]] = []
    metadata: dict[int, dict[str, Any]] = {}
    for scenario, representation in sorted(representations.items()):
        by_seed = trajectories(baseline_dataset, scenario)
        all_frames = [by_seed[seed] for seed in sorted(by_seed)]
        candidates = representation_candidates(representation)
        candidate_count = len(candidates)
        for trajectory_count in config["evaluation"]["trajectory_counts"]:
            for repetition in range(
                int(config["evaluation"]["repeated_subsampling_repetitions"])
            ):
                rng = np.random.default_rng(
                    stable_seed(
                        int(config["master_seed"]),
                        "data_efficiency",
                        scenario,
                        trajectory_count,
                        repetition,
                    )
                )
                indices = np.sort(
                    rng.choice(
                        len(all_frames), size=int(trajectory_count), replace=False
                    )
                )
                frames = [all_frames[int(index)] for index in indices]
                job_index = len(jobs)
                jobs.append(
                    {
                        "job_index": job_index,
                        "frames": frames,
                        "candidates": candidates,
                        "maximum_lag": int(config["temporal"]["maximum_lag"]),
                        "parent_alpha": float(config["temporal"]["parent_alpha"]),
                        "fdr_alpha": float(config["temporal"]["fdr_alpha"]),
                        "bootstrap_repetitions": int(
                            config["evaluation"]["data_efficiency_bootstrap_repetitions"]
                        ),
                        "support_threshold": float(config["temporal"]["support_threshold"]),
                        "master_seed": int(config["master_seed"]),
                        "seed_label": f"{scenario}:efficiency:{trajectory_count}:{repetition}",
                        "point_only": int(trajectory_count) == 1,
                        "include_vote": True,
                        "vote_threshold": float(config["temporal"]["vote_threshold"]),
                    }
                )
                metadata[job_index] = {
                    "scenario": scenario,
                    "representation": representation,
                    "trajectory_count": int(trajectory_count),
                    "repetition": repetition,
                    "candidate_count": candidate_count,
                }

    def progress(done: int, total: int, payload: dict[str, Any]) -> None:
        if progress_callback:
            item = metadata[int(payload["job_index"])]
            progress_callback(
                "Data efficiency", done, total,
                f"{item['scenario']}:n={item['trajectory_count']}:rep={item['repetition']}",
            )

    outputs = _execute_robustness_bootstrap_jobs(jobs, executor, progress)
    rows: list[dict[str, Any]] = []
    for output in outputs:
        item = metadata[int(output["job_index"])]
        scenario = item["scenario"]
        representation = item["representation"]
        trajectory_count = item["trajectory_count"]
        repetition = item["repetition"]
        candidate_count = item["candidate_count"]
        graph = output["graph"]
        summary = output["summary"]
        metrics = _metric_row(scenario, "full_method", graph, representation)
        estimable = trajectory_count > 1
        if estimable:
            edge_low, edge_high = _bootstrap_metric_interval(
                summary, "temporal_qualification_rate", candidate_count
            )
            stability_low, stability_high = _bootstrap_metric_interval(
                summary, "stability", candidate_count
            )
        else:
            edge_low = edge_high = stability_low = stability_high = np.nan
        rows.append(
            {
                "scenario": scenario,
                "method": "full_method",
                "trajectory_count": trajectory_count,
                "repetition": repetition,
                "temporal_qualification_rate": metrics["temporal_qualification_rate"],
                "stability": metrics["stability"] if estimable else np.nan,
                "lag_support": metrics["lag_support"] if estimable else np.nan,
                "lag_std": metrics["lag_std"] if estimable else np.nan,
                "stability_estimable": estimable,
                "temporal_qualification_rate_ci_low": edge_low,
                "temporal_qualification_rate_ci_high": edge_high,
                "stability_ci_low": stability_low,
                "stability_ci_high": stability_high,
            }
        )
        vote_metrics = _metric_row(
            scenario, "trajectory_vote", output["vote"], representation
        )
        rows.append(
            {
                "scenario": scenario,
                "method": "trajectory_vote",
                "trajectory_count": trajectory_count,
                "repetition": repetition,
                "temporal_qualification_rate": vote_metrics["temporal_qualification_rate"],
                "stability": vote_metrics["stability"] if estimable else np.nan,
                "lag_support": vote_metrics["lag_support"] if estimable else np.nan,
                "lag_std": vote_metrics["lag_std"] if estimable else np.nan,
                "stability_estimable": estimable,
                "temporal_qualification_rate_ci_low": np.nan,
                "temporal_qualification_rate_ci_high": np.nan,
                "stability_ci_low": np.nan,
                "stability_ci_high": np.nan,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["scenario", "method", "trajectory_count", "repetition"],
        ignore_index=True,
    )
    frame.to_csv(
        run_root / "analysis" / "data_efficiency_repeated_subsampling.csv",
        index=False,
    )
    return frame


def _perturb_frames(
    frames: Sequence[pd.DataFrame],
    noise: float,
    missing: float,
    seed: int,
) -> list[pd.DataFrame]:
    result = []
    for index, frame in enumerate(frames):
        rng = np.random.default_rng(stable_seed(seed, index))
        revised = frame.copy()
        values = revised.to_numpy(dtype=float)
        if noise > 0:
            finite = np.isfinite(values)
            counts = np.sum(finite, axis=0)
            means = np.divide(
                np.nansum(values, axis=0), counts,
                out=np.zeros(values.shape[1], dtype=float), where=counts > 0,
            )
            squared = np.nansum((values - means[None, :]) ** 2, axis=0)
            scales = np.sqrt(
                np.divide(
                    squared, counts - 1,
                    out=np.zeros(values.shape[1], dtype=float), where=counts > 1,
                )
            )
            values += rng.normal(size=values.shape) * scales[None, :] * noise
        if missing > 0:
            mask = rng.random(values.shape) < missing
            values[mask] = np.nan
        revised.loc[:, :] = values
        if missing > 0:
            revised = revised.interpolate(
                method="linear", axis=0, limit_direction="both"
            )
            revised = revised.fillna(revised.mean()).fillna(0.0)
        result.append(revised)
    return result


def run_observation_robustness(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    baseline_dataset: pd.DataFrame,
    workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
    *,
    executor: Executor | None = None,
) -> pd.DataFrame:
    repetitions = int(config["robustness"]["repetitions"])
    conditions = [
        ("observation_noise", float(level), 0.0, float(config["temporal"]["support_threshold"]))
        for level in config["robustness"]["noise_levels"]
    ] + [
        ("missing_values", 0.0, float(level), float(config["temporal"]["support_threshold"]))
        for level in config["robustness"]["missing_fractions"]
    ] + [
        ("support_threshold", 0.0, 0.0, float(level))
        for level in config["robustness"]["support_thresholds"]
    ]
    jobs: list[dict[str, Any]] = []
    metadata: dict[int, dict[str, Any]] = {}
    for scenario, representation in sorted(representations.items()):
        by_seed = trajectories(baseline_dataset, scenario)
        original_frames = [by_seed[seed] for seed in sorted(by_seed)]
        candidates = representation_candidates(representation)
        for factor, noise, missing, threshold in conditions:
            for repetition in range(repetitions):
                frames = _perturb_frames(
                    original_frames,
                    noise,
                    missing,
                    stable_seed(
                        int(config["master_seed"]),
                        factor,
                        scenario,
                        repetition,
                    ),
                )
                job_index = len(jobs)
                jobs.append(
                    {
                        "job_index": job_index,
                        "frames": frames,
                        "candidates": candidates,
                        "maximum_lag": int(config["temporal"]["maximum_lag"]),
                        "parent_alpha": float(config["temporal"]["parent_alpha"]),
                        "fdr_alpha": float(config["temporal"]["fdr_alpha"]),
                        "bootstrap_repetitions": int(
                            config["robustness"]["bootstrap_repetitions"]
                        ),
                        "support_threshold": threshold,
                        "master_seed": int(config["master_seed"]),
                        "seed_label": f"{scenario}:{factor}:{noise}:{missing}:{threshold}:{repetition}",
                        "point_only": False,
                        "include_vote": False,
                    }
                )
                metadata[job_index] = {
                    "scenario": scenario,
                    "representation": representation,
                    "factor": factor,
                    "noise": noise,
                    "missing": missing,
                    "threshold": threshold,
                    "repetition": repetition,
                }

    def progress(done: int, total: int, payload: dict[str, Any]) -> None:
        if progress_callback:
            item = metadata[int(payload["job_index"])]
            progress_callback(
                "Observation robustness", done, total,
                f"{item['scenario']}:{item['factor']}",
            )

    outputs = _execute_robustness_bootstrap_jobs(jobs, executor, progress)
    rows: list[dict[str, Any]] = []
    for output in outputs:
        item = metadata[int(output["job_index"])]
        graph = output["graph"]
        metrics = _metric_row(
            item["scenario"], item["factor"], graph, item["representation"]
        )
        rows.append(
            {
                "scenario": item["scenario"],
                "factor": item["factor"],
                "noise_level": item["noise"],
                "missing_fraction": item["missing"],
                "support_threshold": item["threshold"],
                "repetition": item["repetition"],
                "temporal_qualification_rate": metrics["temporal_qualification_rate"],
                "stability": metrics["stability"],
                "retained_edge_count": len(graph),
                "intervention_f1": np.nan,
            }
        )
    frame = pd.DataFrame(rows).sort_values(
        ["scenario", "factor", "noise_level", "missing_fraction", "support_threshold", "repetition"],
        ignore_index=True,
    )
    frame.to_csv(run_root / "analysis" / "observation_robustness.csv", index=False)
    return frame


def _corrupt_candidates_and_frames(
    representation: dict[str, Any],
    original_frames: Sequence[pd.DataFrame],
    operator: str,
    ratio: float,
    seed: int,
) -> tuple[list[dict[str, str]], list[pd.DataFrame]]:
    candidates = [dict(item) for item in representation_candidates(representation)]
    frames = [frame.copy() for frame in original_frames]
    if ratio <= 0 or not candidates:
        return candidates, frames
    rng = np.random.default_rng(seed)
    count = max(1, int(round(ratio * len(candidates))))
    selected = rng.choice(len(candidates), size=min(count, len(candidates)), replace=False)
    if operator == "delete_candidate_relation":
        removed = {int(index) for index in selected}
        candidates = [item for index, item in enumerate(candidates) if index not in removed]
    elif operator == "wrong_branch_assignment":
        for sequence, index in enumerate(selected):
            candidates[int(index)]["branch_id"] = f"misassigned_{sequence % 4}"
    elif operator == "cross_branch_relation":
        nodes = representation["indicators"]
        scales = {item["id"]: item["scale"] for item in nodes}
        branches = {item["id"]: item["branch_id"] for item in nodes}
        eligible = [
            (source["id"], target["id"])
            for source in nodes
            for target in nodes
            if branches[source["id"]] != branches[target["id"]]
            and (scales[source["id"]], scales[target["id"]])
            in {("micro", "meso"), ("meso", "macro")}
        ]
        existing = {(item["source"], item["target"]) for item in candidates}
        eligible = [pair for pair in eligible if pair not in existing]
        chosen = (
            rng.choice(len(eligible), size=min(count, len(eligible)), replace=False)
            if eligible
            else np.asarray([], dtype=int)
        )
        for index in chosen:
            source, target = eligible[int(index)]
            candidates.append(
                {
                    "source": source,
                    "target": target,
                    "branch_id": "cross_branch_corruption",
                    "expected_direction": "unknown",
                }
            )
    elif operator in {"irrelevant_indicator", "redundant_semantic_indicator"}:
        for sequence, index in enumerate(selected):
            original = candidates[int(index)]
            name = f"corrupt_{operator}_{sequence:03d}"
            for frame_index, frame in enumerate(frames):
                if operator == "irrelevant_indicator":
                    local_rng = np.random.default_rng(stable_seed(seed, sequence, frame_index))
                    frame[name] = local_rng.normal(size=len(frame))
                else:
                    frame[name] = frame[original["source"]].to_numpy(copy=True)
            candidates.append(
                {
                    "source": name,
                    "target": original["target"],
                    "branch_id": original["branch_id"],
                    "expected_direction": "unknown",
                }
            )
    else:
        raise KeyError(operator)
    unique: dict[tuple[str, str], dict[str, str]] = {}
    for item in candidates:
        unique[(item["source"], item["target"])] = item
    return list(unique.values()), frames


def run_representation_robustness(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    baseline_dataset: pd.DataFrame,
    workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
    *,
    executor: Executor | None = None,
) -> pd.DataFrame:
    operators = [
        "irrelevant_indicator",
        "cross_branch_relation",
        "delete_candidate_relation",
        "wrong_branch_assignment",
        "redundant_semantic_indicator",
    ]
    ratios = config["robustness"]["representation_error_ratios"]
    repetitions = int(config["robustness"]["representation_repetitions"])
    jobs: list[dict[str, Any]] = []
    metadata: dict[int, dict[str, Any]] = {}
    for scenario, representation in sorted(representations.items()):
        by_seed = trajectories(baseline_dataset, scenario)
        original_frames = [by_seed[seed] for seed in sorted(by_seed)]
        for operator in operators:
            for ratio in ratios:
                for repetition in range(repetitions):
                    seed = stable_seed(
                        int(config["master_seed"]),
                        "representation_robustness",
                        scenario,
                        operator,
                        ratio,
                        repetition,
                    )
                    candidates, frames = _corrupt_candidates_and_frames(
                        representation, original_frames, operator, float(ratio), seed
                    )
                    job_index = len(jobs)
                    jobs.append(
                        {
                            "job_index": job_index,
                            "frames": frames,
                            "candidates": candidates,
                            "maximum_lag": int(config["temporal"]["maximum_lag"]),
                            "parent_alpha": float(config["temporal"]["parent_alpha"]),
                            "fdr_alpha": float(config["temporal"]["fdr_alpha"]),
                            "bootstrap_repetitions": int(
                                config["robustness"]["bootstrap_repetitions"]
                            ),
                            "support_threshold": float(config["temporal"]["support_threshold"]),
                            "master_seed": int(config["master_seed"]),
                            "seed_label": f"{scenario}:{operator}:{ratio}:{repetition}",
                            "point_only": False,
                            "include_vote": False,
                        }
                    )
                    metadata[job_index] = {
                        "scenario": scenario,
                        "operator": operator,
                        "ratio": ratio,
                        "repetition": repetition,
                        "representation": representation,
                        "candidate_count": len(candidates),
                        "candidate_signature": [
                            [item["source"], item["target"], item["branch_id"]]
                            for item in sorted(
                                candidates,
                                key=lambda item: (
                                    item["source"], item["target"], item["branch_id"]
                                ),
                            )
                        ],
                    }

    def progress(done: int, total: int, payload: dict[str, Any]) -> None:
        if progress_callback:
            item = metadata[int(payload["job_index"])]
            progress_callback(
                "Representation robustness", done, total,
                f"{item['scenario']}:{item['operator']}:ratio={item['ratio']}",
            )

    outputs = _execute_robustness_bootstrap_jobs(jobs, executor, progress)
    rows: list[dict[str, Any]] = []
    for output in outputs:
        item = metadata[int(output["job_index"])]
        metrics = _metric_row(
            item["scenario"], item["operator"], output["graph"],
            item["representation"],
            candidate_count_override=item["candidate_count"],
        )
        rows.append(_merge_metric_fields(
                        {
                            "scenario": item["scenario"],
                            "operator": item["operator"],
                            "error_ratio": item["ratio"],
                            "repetition": item["repetition"],
                            "candidate_set_sha256": hashlib.sha256(
                                json.dumps(
                                    item["candidate_signature"],
                                    separators=(",", ":"),
                                ).encode("utf-8")
                            ).hexdigest(),
                        },
                        {
                            key: value
                            for key, value in metrics.items()
                            if key not in {"scenario", "variant"}
                        },
                    ))
    frame = pd.DataFrame(rows).sort_values(
        ["scenario", "operator", "error_ratio", "repetition"], ignore_index=True
    )
    frame.to_csv(run_root / "analysis" / "representation_robustness.csv", index=False)
    return frame


def run_mechanism_checks(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    complete_dataset: pd.DataFrame,
) -> pd.DataFrame:
    rows = []
    for scenario, representation in sorted(representations.items()):
        by_seed = trajectories(complete_dataset, scenario, "mechanism_disabled")
        frames = [by_seed[seed] for seed in sorted(by_seed)]
        blocks = prepare_target_blocks(
            frames,
            representation_candidates(representation),
            int(config["temporal"]["maximum_lag"]),
        )
        graph = discover_point_graph_from_blocks(
            blocks,
            float(config["temporal"]["parent_alpha"]),
            float(config["temporal"]["fdr_alpha"]),
        )
        metrics = _metric_row(
            scenario, "mechanism_disabled", graph, representation
        )
        variant = config["scenarios"][scenario]["mechanism_variant"]
        target = mechanism_target_for_variant(scenario, variant)
        metrics.update(
            {
                "mechanism_variant": variant,
                "targeted_mechanism": target,
                "metric_scope": "overall retained point graph under one targeted mechanism-disabled variant",
                "interpretation": (
                    "the targeted propagation pathway may weaken or disappear; "
                    "unrelated system dynamics are not expected to vanish"
                ),
            }
        )
        rows.append(metrics)
    frame = pd.DataFrame(rows)
    frame.to_csv(run_root / "analysis" / "mechanism_disabled_checks.csv", index=False)
    return frame


def run_causal_scalability(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    baseline_dataset: pd.DataFrame,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> pd.DataFrame:
    rows = []
    counts = config["robustness"]["candidate_indicator_counts"]
    repetitions = int(config["robustness"]["scalability_repetitions"])
    total = len(representations) * len(counts) * repetitions
    completed = 0
    for scenario, representation in sorted(representations.items()):
        by_seed = trajectories(baseline_dataset, scenario)
        base_frames = [by_seed[seed].copy() for seed in sorted(by_seed)]
        base_ids = [item["id"] for item in representation["indicators"]]
        for count in counts:
            for repetition in range(repetitions):
                frames = [frame.copy() for frame in base_frames]
                identifiers = list(base_ids[: min(int(count), len(base_ids))])
                while len(identifiers) < int(count):
                    name = f"null_candidate_{len(identifiers):03d}"
                    identifiers.append(name)
                    for frame_index, frame in enumerate(frames):
                        rng = np.random.default_rng(
                            stable_seed(
                                int(config["master_seed"]),
                                "scalability",
                                scenario,
                                repetition,
                                name,
                                frame_index,
                            )
                        )
                        frame[name] = rng.normal(size=len(frame))
                candidates = [
                    {
                        "source": source,
                        "target": target,
                        "branch_id": "scaled",
                        "expected_direction": "unknown",
                    }
                    for source in identifiers
                    for target in identifiers
                    if source != target
                ]
                started = time.perf_counter()
                blocks = prepare_target_blocks(
                    frames, candidates, int(config["temporal"]["maximum_lag"])
                )
                graph = discover_point_graph_from_blocks(
                    blocks,
                    float(config["temporal"]["parent_alpha"]),
                    float(config["temporal"]["fdr_alpha"]),
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "candidate_indicator_count": count,
                        "repetition": repetition,
                        "runtime_seconds": time.perf_counter() - started,
                        "discovered_edge_count": len(graph),
                    }
                )
                completed += 1
                if progress_callback:
                    progress_callback(
                        "Causal scalability", completed, total, f"{scenario}:p={count}"
                    )
    frame = pd.DataFrame(rows)
    frame.to_csv(run_root / "analysis" / "causal_scalability.csv", index=False)
    return frame


def run_robustness_stage(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    baseline_dataset: pd.DataFrame,
    complete_dataset: pd.DataFrame,
    workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    ablations = run_functional_ablations(
        config,
        run_root,
        representations,
        baseline_dataset,
        complete_dataset,
        workers,
        progress_callback,
    )
    data_jobs = (
        len(representations)
        * len(config["evaluation"]["trajectory_counts"])
        * int(config["evaluation"]["repeated_subsampling_repetitions"])
    )
    observation_jobs = (
        len(representations)
        * (
            len(config["robustness"]["noise_levels"])
            + len(config["robustness"]["missing_fractions"])
            + len(config["robustness"]["support_thresholds"])
        )
        * int(config["robustness"]["repetitions"])
    )
    representation_jobs = (
        len(representations)
        * 5
        * len(config["robustness"]["representation_error_ratios"])
        * int(config["robustness"]["representation_repetitions"])
    )
    sweep_started = time.perf_counter()

    def run_sweeps(executor: Executor | None):
        efficiency_frame = run_data_efficiency(
            config, run_root, representations, baseline_dataset, workers,
            progress_callback, executor=executor,
        )
        observation_frame = run_observation_robustness(
            config, run_root, representations, baseline_dataset, workers,
            progress_callback, executor=executor,
        )
        representation_frame = run_representation_robustness(
            config, run_root, representations, baseline_dataset, workers,
            progress_callback, executor=executor,
        )
        return efficiency_frame, observation_frame, representation_frame

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            efficiency, observation, representation_robustness = run_sweeps(executor)
        actual_pool_creations = 1
    else:
        efficiency, observation, representation_robustness = run_sweeps(None)
        actual_pool_creations = 0
    sweep_wall_time = time.perf_counter() - sweep_started
    legacy_pool_creations = (
        data_jobs + observation_jobs + representation_jobs if workers > 1 else 0
    )
    pool_profile = {
        "scientific_evidence": False,
        "architecture": "one reusable outer robustness process pool; bootstrap workers=1 inside jobs",
        "workers": workers,
        "outer_job_count": data_jobs + observation_jobs + representation_jobs,
        "legacy_estimated_pool_creations": legacy_pool_creations,
        "actual_pool_creations": actual_pool_creations,
        "nested_pool_creations": 0,
        "pool_creation_reduction": legacy_pool_creations - actual_pool_creations,
        "robustness_sweep_wall_time_seconds": sweep_wall_time,
    }
    (run_root / "analysis" / "robustness_pool_profile.json").write_text(
        json.dumps(pool_profile, indent=2), encoding="utf-8"
    )
    mechanism = run_mechanism_checks(
        config, run_root, representations, complete_dataset
    )
    scalability = run_causal_scalability(
        config, run_root, representations, baseline_dataset, progress_callback
    )
    return {
        "functional_ablation_rows": len(ablations),
        "data_efficiency_rows": len(efficiency),
        "observation_robustness_rows": len(observation),
        "representation_robustness_rows": len(representation_robustness),
        "mechanism_check_rows": len(mechanism),
        "causal_scalability_rows": len(scalability),
        "robustness_pool_profile": pool_profile,
    }
