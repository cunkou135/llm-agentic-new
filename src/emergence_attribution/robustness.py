"""Pre-specified ablations, sensitivity analyses, and scalability checks."""

from __future__ import annotations

import json
import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .interventions import (
    classify_edge_interventions,
    estimate_all_effects,
)
from .predefined import predefined_representation
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
) -> dict[str, Any]:
    candidates = len(representation_candidates(representation))
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


def _support_rate(frame: pd.DataFrame) -> float:
    if frame.empty or "primary_class" not in frame.columns:
        return float("nan")
    applicable = frame[frame["primary_class"] != "not_applicable"]
    if applicable.empty:
        return float("nan")
    return float(np.mean(applicable["primary_class"] == "supported"))


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
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    total_jobs = (
        len(representations)
        * len(config["evaluation"]["trajectory_counts"])
        * int(config["evaluation"]["repeated_subsampling_repetitions"])
    )
    completed = 0
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
                graph, summary = discover_bootstrap_graph(
                    frames,
                    candidates,
                    int(config["temporal"]["maximum_lag"]),
                    float(config["temporal"]["parent_alpha"]),
                    float(config["temporal"]["fdr_alpha"]),
                    int(config["evaluation"]["data_efficiency_bootstrap_repetitions"]),
                    float(config["temporal"]["support_threshold"]),
                    int(config["master_seed"]),
                    f"{scenario}:efficiency:{trajectory_count}:{repetition}",
                    workers,
                )
                metrics = _metric_row(scenario, "full_method", graph, representation)
                edge_low, edge_high = _bootstrap_metric_interval(
                    summary, "temporal_qualification_rate", candidate_count
                )
                stability_low, stability_high = _bootstrap_metric_interval(
                    summary, "stability", candidate_count
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "method": "full_method",
                        "trajectory_count": trajectory_count,
                        "repetition": repetition,
                        "temporal_qualification_rate": metrics["temporal_qualification_rate"],
                        "stability": metrics["stability"],
                        "temporal_qualification_rate_ci_low": edge_low,
                        "temporal_qualification_rate_ci_high": edge_high,
                        "stability_ci_low": stability_low,
                        "stability_ci_high": stability_high,
                    }
                )
                vote = discover_vote_graph(
                    frames,
                    candidates,
                    int(config["temporal"]["maximum_lag"]),
                    float(config["temporal"]["parent_alpha"]),
                    float(config["temporal"]["fdr_alpha"]),
                    float(config["temporal"]["vote_threshold"]),
                )
                vote_metrics = _metric_row(
                    scenario, "trajectory_vote", vote, representation
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "method": "trajectory_vote",
                        "trajectory_count": trajectory_count,
                        "repetition": repetition,
                        "temporal_qualification_rate": vote_metrics["temporal_qualification_rate"],
                        "stability": vote_metrics["stability"],
                        "temporal_qualification_rate_ci_low": np.nan,
                        "temporal_qualification_rate_ci_high": np.nan,
                        "stability_ci_low": np.nan,
                        "stability_ci_high": np.nan,
                    }
                )
                completed += 1
                if progress_callback:
                    progress_callback(
                        "Data efficiency",
                        completed,
                        total_jobs,
                        f"{scenario}:n={trajectory_count}:rep={repetition}",
                    )
    frame = pd.DataFrame(rows)
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
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
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
    total = len(representations) * len(conditions) * repetitions
    completed = 0
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
                graph, _ = discover_bootstrap_graph(
                    frames,
                    candidates,
                    int(config["temporal"]["maximum_lag"]),
                    float(config["temporal"]["parent_alpha"]),
                    float(config["temporal"]["fdr_alpha"]),
                    int(config["robustness"]["bootstrap_repetitions"]),
                    threshold,
                    int(config["master_seed"]),
                    f"{scenario}:{factor}:{noise}:{missing}:{threshold}:{repetition}",
                    workers,
                )
                metrics = _metric_row(
                    scenario, factor, graph, representation
                )
                rows.append(
                    {
                        "scenario": scenario,
                        "factor": factor,
                        "noise_level": noise,
                        "missing_fraction": missing,
                        "support_threshold": threshold,
                        "repetition": repetition,
                        "temporal_qualification_rate": metrics["temporal_qualification_rate"],
                        "stability": metrics["stability"],
                        "retained_edge_count": len(graph),
                        "intervention_f1": np.nan,
                    }
                )
                completed += 1
                if progress_callback:
                    progress_callback(
                        "Observation robustness", completed, total, f"{scenario}:{factor}"
                    )
    frame = pd.DataFrame(rows)
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
        for source, target in eligible[:count]:
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
    total = len(representations) * len(operators) * len(ratios) * repetitions
    completed = 0
    rows: list[dict[str, Any]] = []
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
                    graph, _ = discover_bootstrap_graph(
                        frames,
                        candidates,
                        int(config["temporal"]["maximum_lag"]),
                        float(config["temporal"]["parent_alpha"]),
                        float(config["temporal"]["fdr_alpha"]),
                        int(config["robustness"]["bootstrap_repetitions"]),
                        float(config["temporal"]["support_threshold"]),
                        int(config["master_seed"]),
                        f"{scenario}:{operator}:{ratio}:{repetition}",
                        workers,
                    )
                    metrics = _metric_row(
                        scenario, operator, graph, representation
                    )
                    rows.append(
                        {
                            "scenario": scenario,
                            "operator": operator,
                            "error_ratio": ratio,
                            "repetition": repetition,
                            "candidate_edge_count": len(candidates),
                            **metrics,
                        }
                    )
                    completed += 1
                    if progress_callback:
                        progress_callback(
                            "Representation robustness",
                            completed,
                            total,
                            f"{scenario}:{operator}:ratio={ratio}",
                        )
    frame = pd.DataFrame(rows)
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
        rows.append(
            {
                "scenario": scenario,
                "mechanism_variant": config["scenarios"][scenario]["mechanism_variant"],
                **metrics,
            }
        )
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
    efficiency = run_data_efficiency(
        config,
        run_root,
        representations,
        baseline_dataset,
        workers,
        progress_callback,
    )
    observation = run_observation_robustness(
        config,
        run_root,
        representations,
        baseline_dataset,
        workers,
        progress_callback,
    )
    representation_robustness = run_representation_robustness(
        config,
        run_root,
        representations,
        baseline_dataset,
        workers,
        progress_callback,
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
    }
