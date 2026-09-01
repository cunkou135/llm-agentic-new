"""Matched-seed parameter interventions and multiscale propagation evidence."""

from __future__ import annotations

import json
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import pandas as pd

from .temporal import TemporalEdge, load_graph_records, stable_seed


@dataclass(frozen=True)
class EffectSummary:
    scenario: str
    parameter: str
    direction: str
    node_id: str
    scale: str
    baseline_sd: float
    cumulative_effect_raw: float
    cumulative_effect_standardised: float
    cumulative_effect: float
    cumulative_ci_low_raw: float
    cumulative_ci_high_raw: float
    cumulative_ci_low_standardised: float
    cumulative_ci_high_standardised: float
    ci_width: float
    peak_time: int
    onset_time: int
    onset_ci_low: float
    onset_ci_high: float
    terminal_effect_standardised: float
    significant: bool
    effect_sign: int
    paired_seed_sign_consistency: float
    paired: bool


CLASSIFICATION_COLUMNS = [
    "scenario", "source", "target", "parameter", "direction",
    "manipulation_success", "primary_class", "underlying_class",
    "intervention_scope", "source_onset", "target_onset",
    "intervention_delay", "observational_lag", "lag_difference",
    "source_effect", "target_effect", "not_applicable_reason",
]


def detect_onset(
    values: np.ndarray,
    significant: np.ndarray,
    start: int,
    consecutive: int,
) -> int:
    values = np.asarray(values)
    significant = np.asarray(significant, dtype=bool)
    for index in range(max(0, start), len(values) - consecutive + 1):
        if np.all(significant[index : index + consecutive]):
            signs = np.sign(values[index : index + consecutive])
            if np.all(signs == signs[0]) and signs[0] != 0:
                return int(index)
    return -1


def _direction_arrays(
    dataset: pd.DataFrame,
    scenario: str,
    condition: str,
    node_ids: list[str],
) -> tuple[list[int], np.ndarray]:
    subset = dataset[
        (dataset["scenario"] == scenario) & (dataset["condition"] == condition)
    ]
    seeds = sorted(int(value) for value in subset["seed"].unique())
    arrays = []
    for seed in seeds:
        frame = subset[subset["seed"] == seed].sort_values("time")
        arrays.append(frame[node_ids].to_numpy(dtype=float))
    return seeds, np.stack(arrays, axis=0)


def _effect_job(payload: dict[str, Any]) -> dict[str, Any]:
    scenario = payload["scenario"]
    parameter = payload["parameter"]
    direction = payload["direction"]
    node_ids = payload["node_ids"]
    scales = payload["scales"]
    baseline = np.asarray(payload["baseline"], dtype=float)
    intervention = np.asarray(payload["intervention"], dtype=float)
    config = payload["config"]
    paired = bool(payload["paired"])
    if not paired:
        rng = np.random.default_rng(payload["seed"])
        baseline = baseline[rng.permutation(len(baseline))]
    raw_difference = intervention - baseline
    evaluation_start = int(config["evaluation_start"])
    baseline_sd = np.std(
        baseline[:, evaluation_start:, :].reshape(-1, baseline.shape[-1]),
        axis=0,
        ddof=1,
    )
    baseline_sd = np.where(np.isfinite(baseline_sd) & (baseline_sd > 1e-12), baseline_sd, 1.0)
    standardised = raw_difference / baseline_sd[None, None, :]
    repetitions = int(config["bootstrap_repetitions"])
    rng = np.random.default_rng(payload["seed"])
    indices = rng.integers(0, len(baseline), size=(repetitions, len(baseline)))
    bootstrap_raw = np.mean(raw_difference[indices], axis=1)
    bootstrap_standardised = np.mean(standardised[indices], axis=1)
    mean_raw = np.mean(raw_difference, axis=0)
    mean_standardised = np.mean(standardised, axis=0)
    alpha = (1.0 - float(config["confidence_level"])) / 2.0
    low_raw = np.quantile(bootstrap_raw, alpha, axis=0)
    high_raw = np.quantile(bootstrap_raw, 1.0 - alpha, axis=0)
    low_standardised = np.quantile(bootstrap_standardised, alpha, axis=0)
    high_standardised = np.quantile(bootstrap_standardised, 1.0 - alpha, axis=0)
    threshold = float(config["minimum_standardised_effect"])
    onset_start = int(config["onset_detection_start"])
    consecutive = int(config["onset_consecutive_steps"])
    terminal_window = int(config["terminal_window"])
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for node_index, node_id in enumerate(node_ids):
        significant_curve = (low_standardised[:, node_index] > threshold) | (
            high_standardised[:, node_index] < -threshold
        )
        onset = detect_onset(
            mean_standardised[:, node_index], significant_curve, onset_start, consecutive
        )
        onset_samples = np.asarray(
            [
                detect_onset(
                    curve[:, node_index],
                    np.abs(curve[:, node_index]) >= threshold,
                    onset_start,
                    consecutive,
                )
                for curve in bootstrap_standardised
            ]
        )
        valid_onsets = onset_samples[onset_samples >= 0]
        onset_low = float(np.quantile(valid_onsets, alpha)) if len(valid_onsets) else float("nan")
        onset_high = (
            float(np.quantile(valid_onsets, 1.0 - alpha))
            if len(valid_onsets)
            else float("nan")
        )
        raw_by_seed = np.mean(raw_difference[:, evaluation_start:, node_index], axis=1)
        standardised_by_seed = np.mean(
            standardised[:, evaluation_start:, node_index], axis=1
        )
        cumulative_raw = float(np.mean(raw_by_seed))
        cumulative_standardised = float(np.mean(standardised_by_seed))
        boot_cumulative_raw = np.mean(
            bootstrap_raw[:, evaluation_start:, node_index], axis=1
        )
        boot_cumulative_standardised = np.mean(
            bootstrap_standardised[:, evaluation_start:, node_index], axis=1
        )
        cumulative_low_raw = float(np.quantile(boot_cumulative_raw, alpha))
        cumulative_high_raw = float(np.quantile(boot_cumulative_raw, 1.0 - alpha))
        cumulative_low_standardised = float(
            np.quantile(boot_cumulative_standardised, alpha)
        )
        cumulative_high_standardised = float(
            np.quantile(boot_cumulative_standardised, 1.0 - alpha)
        )
        is_significant = bool(
            onset >= 0
            and (
                cumulative_low_standardised > threshold
                or cumulative_high_standardised < -threshold
            )
        )
        peak_time = int(
            evaluation_start
            + np.argmax(np.abs(mean_standardised[evaluation_start:, node_index]))
        )
        sign = int(np.sign(cumulative_standardised))
        consistency = (
            float(np.mean(np.sign(standardised_by_seed) == sign)) if sign else 0.0
        )
        summary = EffectSummary(
            scenario=scenario,
            parameter=parameter,
            direction=direction,
            node_id=node_id,
            scale=scales[node_id],
            baseline_sd=float(baseline_sd[node_index]),
            cumulative_effect_raw=cumulative_raw,
            cumulative_effect_standardised=cumulative_standardised,
            cumulative_effect=cumulative_standardised,
            cumulative_ci_low_raw=cumulative_low_raw,
            cumulative_ci_high_raw=cumulative_high_raw,
            cumulative_ci_low_standardised=cumulative_low_standardised,
            cumulative_ci_high_standardised=cumulative_high_standardised,
            ci_width=cumulative_high_standardised - cumulative_low_standardised,
            peak_time=peak_time,
            onset_time=onset,
            onset_ci_low=onset_low,
            onset_ci_high=onset_high,
            terminal_effect_standardised=float(
                np.mean(mean_standardised[-terminal_window:, node_index])
            ),
            significant=is_significant,
            effect_sign=sign,
            paired_seed_sign_consistency=consistency,
            paired=paired,
        )
        summaries.append(asdict(summary))
        for time_index in range(mean_raw.shape[0]):
            curves.append(
                {
                    "scenario": scenario,
                    "parameter": parameter,
                    "direction": direction,
                    "node_id": node_id,
                    "time": time_index,
                    "mean": float(mean_standardised[time_index, node_index]),
                    "ci_low": float(low_standardised[time_index, node_index]),
                    "ci_high": float(high_standardised[time_index, node_index]),
                    "mean_raw": float(mean_raw[time_index, node_index]),
                    "ci_low_raw": float(low_raw[time_index, node_index]),
                    "ci_high_raw": float(high_raw[time_index, node_index]),
                    "mean_standardised": float(mean_standardised[time_index, node_index]),
                    "ci_low_standardised": float(low_standardised[time_index, node_index]),
                    "ci_high_standardised": float(high_standardised[time_index, node_index]),
                    "centre_statistic": "mean",
                }
            )
    return {"summaries": summaries, "curves": curves}


def estimate_all_effects(
    dataset: pd.DataFrame,
    config: dict[str, Any],
    representations: dict[str, dict[str, Any]],
    workers: int,
    *,
    paired: bool,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    jobs: list[dict[str, Any]] = []
    for scenario, representation in sorted(representations.items()):
        node_ids = [item["id"] for item in representation["indicators"]]
        scales = {item["id"]: item["scale"] for item in representation["indicators"]}
        baseline_seeds, baseline = _direction_arrays(
            dataset, scenario, "baseline", node_ids
        )
        for parameter in sorted(config["scenarios"][scenario]["interventions"]):
            for direction in ("minus", "plus"):
                condition = f"{parameter}_{direction}"
                intervention_seeds, intervention = _direction_arrays(
                    dataset, scenario, condition, node_ids
                )
                if intervention_seeds != baseline_seeds:
                    raise RuntimeError(f"matched seed pool differs for {scenario}:{condition}")
                jobs.append(
                    {
                        "scenario": scenario,
                        "parameter": parameter,
                        "direction": direction,
                        "node_ids": node_ids,
                        "scales": scales,
                        "baseline": baseline,
                        "intervention": intervention,
                        "config": config["intervention"],
                        "paired": paired,
                        "seed": stable_seed(
                            int(config["master_seed"]),
                            "paired" if paired else "unpaired",
                            scenario,
                            parameter,
                            direction,
                        ),
                    }
                )
    results: list[dict[str, Any]] = []
    if workers <= 1:
        for job in jobs:
            results.append(_effect_job(job))
            if progress_callback:
                progress_callback(
                    "Paired-seed bootstrap" if paired else "Unpaired bootstrap",
                    len(results),
                    len(jobs),
                    f"{job['scenario']}:{job['parameter']}:{job['direction']}",
                )
    else:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(_effect_job, job): job for job in jobs}
            for future in as_completed(futures):
                results.append(future.result())
                job = futures[future]
                if progress_callback:
                    progress_callback(
                        "Paired-seed bootstrap" if paired else "Unpaired bootstrap",
                        len(results),
                        len(jobs),
                        f"{job['scenario']}:{job['parameter']}:{job['direction']}",
                    )
    summaries = pd.DataFrame(
        [record for result in results for record in result["summaries"]]
    ).sort_values(["scenario", "parameter", "direction", "node_id"], ignore_index=True)
    curves = pd.DataFrame(
        [record for result in results for record in result["curves"]]
    ).sort_values(
        ["scenario", "parameter", "direction", "node_id", "time"],
        ignore_index=True,
    )
    return summaries, curves


def direct_parameter_sources(representation: dict[str, Any]) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for indicator in representation["indicators"]:
        for association in indicator.get("parameter_associations", []):
            if association["relationship"] == "direct" and indicator["scale"] == "micro":
                result.setdefault(association["parameter"], []).append(indicator["id"])
    return {key: sorted(set(value)) for key, value in result.items()}


def classify_edge_interventions(
    scenario: str,
    graph: Sequence[TemporalEdge],
    effects: pd.DataFrame,
    representation: dict[str, Any],
    lag_tolerance: int,
) -> pd.DataFrame:
    if not graph:
        return pd.DataFrame(columns=CLASSIFICATION_COLUMNS)
    effect_lookup = {
        (row.parameter, row.direction, row.node_id): row
        for row in effects[effects["scenario"] == scenario].itertuples()
    }
    direct = direct_parameter_sources(representation)
    sources_to_parameters: dict[str, list[str]] = {}
    for parameter, nodes in direct.items():
        for node in nodes:
            sources_to_parameters.setdefault(node, []).append(parameter)
    rows: list[dict[str, Any]] = []
    for edge in graph:
        parameters = sorted(sources_to_parameters.get(edge.source, []))
        if not parameters:
            rows.append(
                {
                    "scenario": scenario,
                    "source": edge.source,
                    "target": edge.target,
                    "parameter": "",
                    "direction": "",
                    "manipulation_success": False,
                    "primary_class": "inconclusive",
                    "underlying_class": "inconclusive",
                    "intervention_scope": "unmapped",
                    "source_onset": -1,
                    "target_onset": -1,
                    "intervention_delay": np.nan,
                    "observational_lag": edge.lag,
                    "lag_difference": np.nan,
                    "source_effect": np.nan,
                    "target_effect": np.nan,
                    "not_applicable_reason": "",
                }
            )
            continue
        for parameter in parameters:
            scope = "non_local" if len(direct.get(parameter, [])) > 1 else "local"
            for direction in ("minus", "plus"):
                source = effect_lookup.get((parameter, direction, edge.source))
                target = effect_lookup.get((parameter, direction, edge.target))
                if source is None or target is None:
                    evidence = "inconclusive"
                    success = False
                    source_onset = target_onset = -1
                    source_effect = target_effect = delay = lag_difference = np.nan
                else:
                    success = bool(source.significant)
                    source_onset, target_onset = int(source.onset_time), int(target.onset_time)
                    source_effect = float(source.cumulative_effect_standardised)
                    target_effect = float(target.cumulative_effect_standardised)
                    delay = (
                        float(target_onset - source_onset)
                        if source_onset >= 0 and target_onset >= 0
                        else np.nan
                    )
                    lag_difference = delay - edge.lag if np.isfinite(delay) else np.nan
                    expected_target_sign = int(np.sign(source_effect * edge.beta))
                    observed_target_sign = int(np.sign(target_effect))
                    timing_ok = bool(
                        np.isfinite(delay)
                        and delay >= 0
                        and abs(lag_difference) <= lag_tolerance
                    )
                    if not source.significant:
                        evidence = "manipulation_failure"
                    elif not target.significant:
                        evidence = "no_stable_downstream_effect"
                    elif observed_target_sign != expected_target_sign:
                        evidence = "directionally_contradicted"
                    elif timing_ok:
                        evidence = "supported"
                    else:
                        evidence = "inconclusive"
                rows.append(
                    {
                        "scenario": scenario,
                        "source": edge.source,
                        "target": edge.target,
                        "parameter": parameter,
                        "direction": direction,
                        "manipulation_success": success,
                        "primary_class": evidence,
                        "underlying_class": evidence,
                        "intervention_scope": scope,
                        "source_onset": source_onset,
                        "target_onset": target_onset,
                        "intervention_delay": delay,
                        "observational_lag": edge.lag,
                        "lag_difference": lag_difference,
                        "source_effect": source_effect,
                        "target_effect": target_effect,
                        "not_applicable_reason": "",
                    }
                )
    return pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS)


def graph_paths(
    graph: Sequence[TemporalEdge], representation: dict[str, Any]
) -> list[tuple[str, str, str]]:
    scale = {item["id"]: item["scale"] for item in representation["indicators"]}
    first = [
        edge
        for edge in graph
        if scale.get(edge.source) == "micro" and scale.get(edge.target) == "meso"
    ]
    second = [
        edge
        for edge in graph
        if scale.get(edge.source) == "meso" and scale.get(edge.target) == "macro"
    ]
    return sorted(
        {
            (left.source, left.target, right.target)
            for left in first
            for right in second
            if left.target == right.source
        }
    )


def path_timing_summary(
    scenario: str,
    graph: Sequence[TemporalEdge],
    effects: pd.DataFrame,
    representation: dict[str, Any],
) -> pd.DataFrame:
    edge_lag = {(edge.source, edge.target): edge.lag for edge in graph}
    direct = direct_parameter_sources(representation)
    source_parameters: dict[str, list[str]] = {}
    for parameter, nodes in direct.items():
        for node in nodes:
            source_parameters.setdefault(node, []).append(parameter)
    rows: list[dict[str, Any]] = []
    for source, meso, macro in graph_paths(graph, representation):
        for parameter in sorted(source_parameters.get(source, [])):
            for direction in ("minus", "plus"):
                subset = effects[
                    (effects["scenario"] == scenario)
                    & (effects["parameter"] == parameter)
                    & (effects["direction"] == direction)
                    & (effects["node_id"].isin([source, meso, macro]))
                ]
                if len(subset) != 3:
                    continue
                lookup = {row.node_id: row for row in subset.itertuples()}
                path_id = f"{parameter}:{direction}:{source}>{meso}>{macro}"
                for scale, node, parent in (
                    ("micro", source, None),
                    ("meso", meso, source),
                    ("macro", macro, meso),
                ):
                    item = lookup[node]
                    parent_onset = lookup[parent].onset_time if parent else np.nan
                    response_delay = (
                        item.onset_time - parent_onset
                        if parent and item.onset_time >= 0 and parent_onset >= 0
                        else np.nan
                    )
                    observational_lag = edge_lag.get((parent, node), np.nan) if parent else np.nan
                    rows.append(
                        {
                            "scenario": scenario,
                            "path_id": path_id,
                            "parameter": parameter,
                            "direction": direction,
                            "source": source,
                            "meso": meso,
                            "macro": macro,
                            "node_id": node,
                            "scale": scale,
                            "onset_time": item.onset_time,
                            "onset_ci_low": item.onset_ci_low,
                            "onset_ci_high": item.onset_ci_high,
                            "observational_lag": observational_lag,
                            "response_delay": response_delay,
                            "lag_difference": response_delay - observational_lag
                            if np.isfinite(response_delay) and np.isfinite(observational_lag)
                            else np.nan,
                            "cumulative_effect": item.cumulative_effect_standardised,
                            "cumulative_effect_raw": item.cumulative_effect_raw,
                            "significant": item.significant,
                        }
                    )
    return pd.DataFrame(rows)


def select_representative_paths(path_summary: pd.DataFrame) -> dict[str, Any]:
    selected: dict[str, Any] = {
        "selection_rule": "closest_to_median_absolute_macro_cumulative_effect_among_complete_ordered_paths",
        "scenarios": {},
    }
    for scenario, scenario_frame in path_summary.groupby("scenario"):
        valid_ids = []
        for path_id, group in scenario_frame.groupby("path_id"):
            order = group.sort_values(
                "scale", key=lambda values: values.map({"micro": 0, "meso": 1, "macro": 2})
            )
            onsets = order["onset_time"].to_numpy(dtype=float)
            if len(order) == 3 and order["significant"].all() and np.all(onsets >= 0) and np.all(np.diff(onsets) >= 0):
                valid_ids.append(path_id)
        macro = scenario_frame[
            (scenario_frame["scale"] == "macro")
            & (scenario_frame["path_id"].isin(valid_ids))
        ].copy()
        if macro.empty:
            selected["scenarios"][scenario] = {"path_id": None, "reason": "no complete ordered path"}
            continue
        macro["absolute_effect"] = macro["cumulative_effect"].abs()
        median = float(macro["absolute_effect"].median())
        row = macro.loc[(macro["absolute_effect"] - median).abs().idxmin()]
        selected["scenarios"][scenario] = {
            "path_id": str(row["path_id"]),
            "median_absolute_effect": median,
        }
    return selected


def run_intervention_stage(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    complete_dataset: pd.DataFrame,
    workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    effects, curves = estimate_all_effects(
        complete_dataset,
        config,
        representations,
        workers,
        paired=True,
        progress_callback=progress_callback,
    )
    graphs = load_graph_records(run_root / "analysis" / "main_graphs.jsonl")
    classifications = []
    timings = []
    graph_methods = (
        "unrestricted_temporal_search",
        "single_trajectory",
        "trajectory_vote",
        "full_method",
    )
    for scenario, representation in sorted(representations.items()):
        for method in graph_methods:
            graph = graphs[(scenario, method)]
            classified = classify_edge_interventions(
                    scenario,
                    graph,
                    effects,
                    representation,
                    int(config["intervention"]["lag_tolerance"]),
                )
            if classified.empty:
                classified = pd.DataFrame(
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
            classified.insert(1, "method", method)
            classifications.append(classified)
        graph = graphs[(scenario, "full_method")]
        timings.append(
            path_timing_summary(scenario, graph, effects, representation)
        )
    classification_frame = pd.concat(classifications, ignore_index=True)
    timing_frame = pd.concat(timings, ignore_index=True) if timings else pd.DataFrame()
    analysis_root = run_root / "analysis"
    effects.to_parquet(analysis_root / "paired_effects.parquet", index=False)
    curves.to_parquet(analysis_root / "effect_curves.parquet", index=False)
    classification_frame.to_csv(
        analysis_root / "intervention_classifications.csv", index=False
    )
    timing_frame.to_csv(analysis_root / "path_timing_summary.csv", index=False)
    selection = select_representative_paths(timing_frame) if not timing_frame.empty else {
        "selection_rule": "no_path_available",
        "scenarios": {},
    }
    (analysis_root / "representative_path_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "effect_rows": len(effects),
        "curve_rows": len(curves),
        "classification_rows": len(classification_frame),
        "path_rows": len(timing_frame),
    }
