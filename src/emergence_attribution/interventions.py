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
    "scenario", "root_source", "edge_source", "edge_target",
    "source", "target", "parameter", "direction", "manipulation_level",
    "manipulation_success", "primary_class", "underlying_class",
    "intervention_scope", "root_onset", "source_onset", "target_onset",
    "intervention_delay", "observational_lag", "lag_difference",
    "root_effect", "source_effect", "target_effect", "not_applicable_reason",
]


INTERVENTION_CLASSES = {
    "supported",
    "directionally_contradicted",
    "no_stable_downstream_effect",
    "manipulation_failure",
    "inconclusive",
    "not_applicable",
}


def aggregate_edge_intervention_evidence(
    classifications: pd.DataFrame,
) -> pd.DataFrame:
    """Freeze attempt-to-edge aggregation across directions and root routes.

    Explicit directional contradiction has precedence over support.  A support
    remains sufficient when the other applicable attempt only failed to move
    the manipulation root, because perturbation strength may be asymmetric.
    """

    columns = [
        "scenario", "method", "source", "target", "edge_class",
        "attempt_count", "applicable_attempt_count", "supported_attempt_count",
        "contradiction_attempt_count",
    ]
    if classifications.empty:
        return pd.DataFrame(columns=columns)
    unknown = sorted(
        set(classifications["primary_class"].astype(str)) - INTERVENTION_CLASSES
    )
    if unknown:
        raise RuntimeError(f"unknown intervention classification values: {unknown}")
    group_columns = ["scenario"]
    if "method" in classifications.columns:
        group_columns.append("method")
    group_columns.extend(["source", "target"])
    rows: list[dict[str, Any]] = []
    for identity, group in classifications.groupby(
        group_columns, dropna=False, sort=True
    ):
        if not isinstance(identity, tuple):
            identity = (identity,)
        values = group["primary_class"].astype(str)
        applicable = values[values != "not_applicable"]
        if applicable.empty:
            edge_class = "not_applicable"
        elif bool((applicable == "directionally_contradicted").any()):
            edge_class = "directionally_contradicted"
        elif bool((applicable == "supported").any()):
            edge_class = "supported"
        elif bool((applicable == "manipulation_failure").all()):
            edge_class = "manipulation_failure"
        elif bool((applicable == "no_stable_downstream_effect").any()):
            edge_class = "no_stable_downstream_effect"
        else:
            edge_class = "inconclusive"
        row = dict(zip(group_columns, identity))
        if "method" not in row:
            row["method"] = ""
        row.update(
            {
                "edge_class": edge_class,
                "attempt_count": int(len(values)),
                "applicable_attempt_count": int(len(applicable)),
                "supported_attempt_count": int((applicable == "supported").sum()),
                "contradiction_attempt_count": int(
                    (applicable == "directionally_contradicted").sum()
                ),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


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
    estimable = np.isfinite(baseline_sd) & (baseline_sd > 1e-12)
    standardised = np.full_like(raw_difference, np.nan, dtype=float)
    standardised[:, :, estimable] = (
        raw_difference[:, :, estimable]
        / baseline_sd[None, None, estimable]
    )
    repetitions = int(config["bootstrap_repetitions"])
    rng = np.random.default_rng(payload["seed"])
    indices = rng.integers(0, len(baseline), size=(repetitions, len(baseline)))
    bootstrap_raw = np.mean(raw_difference[indices], axis=1)
    bootstrap_standardised = np.full(
        (repetitions, raw_difference.shape[1], raw_difference.shape[2]),
        np.nan,
        dtype=float,
    )
    bootstrap_standardised[:, :, estimable] = np.mean(
        standardised[indices][:, :, :, estimable], axis=1
    )
    mean_raw = np.mean(raw_difference, axis=0)
    mean_standardised = np.full_like(mean_raw, np.nan, dtype=float)
    mean_standardised[:, estimable] = np.mean(
        standardised[:, :, estimable], axis=0
    )
    alpha = (1.0 - float(config["confidence_level"])) / 2.0
    low_raw = np.quantile(bootstrap_raw, alpha, axis=0)
    high_raw = np.quantile(bootstrap_raw, 1.0 - alpha, axis=0)
    low_standardised = np.full_like(mean_raw, np.nan, dtype=float)
    high_standardised = np.full_like(mean_raw, np.nan, dtype=float)
    low_standardised[:, estimable] = np.quantile(
        bootstrap_standardised[:, :, estimable], alpha, axis=0
    )
    high_standardised[:, estimable] = np.quantile(
        bootstrap_standardised[:, :, estimable], 1.0 - alpha, axis=0
    )
    threshold = float(config["minimum_standardised_effect"])
    onset_start = int(config["onset_detection_start"])
    consecutive = int(config["onset_consecutive_steps"])
    terminal_window = int(config["terminal_window"])
    summaries: list[dict[str, Any]] = []
    curves: list[dict[str, Any]] = []
    for node_index, node_id in enumerate(node_ids):
        node_estimable = bool(estimable[node_index])
        if node_estimable:
            significant_curve = (low_standardised[:, node_index] > threshold) | (
                high_standardised[:, node_index] < -threshold
            )
            onset = detect_onset(
                mean_standardised[:, node_index], significant_curve,
                onset_start, consecutive,
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
            onset_low = (
                float(np.quantile(valid_onsets, alpha))
                if len(valid_onsets) else float("nan")
            )
            onset_high = (
                float(np.quantile(valid_onsets, 1.0 - alpha))
                if len(valid_onsets) else float("nan")
            )
        else:
            onset = -1
            onset_low = onset_high = float("nan")
        raw_by_seed = np.mean(raw_difference[:, evaluation_start:, node_index], axis=1)
        cumulative_raw = float(np.mean(raw_by_seed))
        boot_cumulative_raw = np.mean(
            bootstrap_raw[:, evaluation_start:, node_index], axis=1
        )
        cumulative_low_raw = float(np.quantile(boot_cumulative_raw, alpha))
        cumulative_high_raw = float(np.quantile(boot_cumulative_raw, 1.0 - alpha))
        if node_estimable:
            standardised_by_seed = np.mean(
                standardised[:, evaluation_start:, node_index], axis=1
            )
            cumulative_standardised = float(np.mean(standardised_by_seed))
            boot_cumulative_standardised = np.mean(
                bootstrap_standardised[:, evaluation_start:, node_index], axis=1
            )
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
                float(np.mean(np.sign(standardised_by_seed) == sign))
                if sign else 0.0
            )
            terminal_standardised = float(
                np.mean(mean_standardised[-terminal_window:, node_index])
            )
        else:
            cumulative_standardised = float("nan")
            cumulative_low_standardised = float("nan")
            cumulative_high_standardised = float("nan")
            is_significant = False
            peak_time = -1
            sign = 0
            consistency = float("nan")
            terminal_standardised = float("nan")
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
            terminal_effect_standardised=terminal_standardised,
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


def upstream_manipulation_routes(
    edge_source: str,
    edge_target: str,
    graph: Sequence[TemporalEdge],
    representation: dict[str, Any],
) -> list[dict[str, str]]:
    """Resolve legal simulator-parameter roots for one adjacent-scale edge.

    Parameters are never attached to or used to overwrite a generated
    observable.  A Micro->Meso edge is tested at its direct Micro root.  A
    Meso->Macro edge reuses a parameter attached to an upstream Micro parent
    and is therefore explicitly upstream-mediated.
    """

    scales = {item["id"]: item["scale"] for item in representation["indicators"]}
    branches = {
        item["id"]: item["branch_id"] for item in representation["indicators"]
    }
    direct = direct_parameter_sources(representation)
    root_parameters: dict[str, list[str]] = {}
    for parameter, roots in direct.items():
        for root in roots:
            root_parameters.setdefault(root, []).append(parameter)
    source_scale, target_scale = scales.get(edge_source), scales.get(edge_target)
    if (source_scale, target_scale) == ("micro", "meso"):
        roots = [edge_source]
        scope = "direct_root"
    elif (source_scale, target_scale) == ("meso", "macro"):
        candidate_pairs = {
            (edge["source"], edge["target"])
            for edge in representation.get("candidate_edges", [])
        }
        retained_pairs = {(edge.source, edge.target) for edge in graph}
        roots = sorted(
            source
            for source, target in candidate_pairs | retained_pairs
            if target == edge_source
            and scales.get(source) == "micro"
            and branches.get(source) == branches.get(edge_source)
        )
        scope = "upstream_mediated"
    else:
        return []
    return [
        {
            "root_source": root,
            "parameter": parameter,
            "intervention_scope": scope,
            "manipulation_level": "micro",
        }
        for root in sorted(set(roots))
        for parameter in sorted(set(root_parameters.get(root, [])))
    ]


def intervention_testable_edges(
    edges: Sequence[tuple[str, str]],
    representation: dict[str, Any],
) -> set[tuple[str, str]]:
    """Return truth/candidate relations with a legal simulator manipulation route."""

    return {
        (source, target)
        for source, target in edges
        if upstream_manipulation_routes(source, target, (), representation)
    }


def not_applicable_classification(
    scenario: str,
    *,
    source: str = "",
    target: str = "",
    observational_lag: float = np.nan,
    reason: str,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "root_source": "",
        "edge_source": source,
        "edge_target": target,
        "source": source,
        "target": target,
        "parameter": "",
        "direction": "",
        "manipulation_level": "none",
        "manipulation_success": False,
        "primary_class": "not_applicable",
        "underlying_class": "not_applicable",
        "intervention_scope": "none",
        "root_onset": -1,
        "source_onset": -1,
        "target_onset": -1,
        "intervention_delay": np.nan,
        "observational_lag": observational_lag,
        "lag_difference": np.nan,
        "root_effect": np.nan,
        "source_effect": np.nan,
        "target_effect": np.nan,
        "not_applicable_reason": reason,
    }


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
    rows: list[dict[str, Any]] = []
    for edge in graph:
        routes = upstream_manipulation_routes(
            edge.source, edge.target, graph, representation
        )
        if not routes:
            rows.append(
                not_applicable_classification(
                    scenario,
                    source=edge.source,
                    target=edge.target,
                    observational_lag=edge.lag,
                    reason="no_legal_upstream_micro_manipulation_root",
                )
            )
            continue
        for route in routes:
            parameter = route["parameter"]
            root_source = route["root_source"]
            for direction in ("minus", "plus"):
                root = effect_lookup.get((parameter, direction, root_source))
                source = effect_lookup.get((parameter, direction, edge.source))
                target = effect_lookup.get((parameter, direction, edge.target))
                if root is None or source is None or target is None:
                    evidence = "inconclusive"
                    success = False
                    root_onset = source_onset = target_onset = -1
                    root_effect = source_effect = target_effect = np.nan
                    delay = lag_difference = np.nan
                else:
                    success = bool(root.significant)
                    root_onset = int(root.onset_time)
                    source_onset, target_onset = int(source.onset_time), int(target.onset_time)
                    root_effect = float(root.cumulative_effect_standardised)
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
                    propagation_order_ok = bool(
                        root_onset >= 0
                        and source_onset >= root_onset
                        and target_onset >= source_onset
                    )
                    timing_ok = bool(
                        propagation_order_ok
                        and np.isfinite(delay)
                        and abs(lag_difference) <= lag_tolerance
                    )
                    if not root.significant:
                        evidence = "manipulation_failure"
                    elif not source.significant or not target.significant:
                        evidence = "no_stable_downstream_effect"
                    elif expected_target_sign == 0:
                        evidence = "inconclusive"
                    elif observed_target_sign != expected_target_sign:
                        evidence = "directionally_contradicted"
                    elif timing_ok:
                        evidence = "supported"
                    else:
                        evidence = "inconclusive"
                rows.append(
                    {
                        "scenario": scenario,
                        "root_source": root_source,
                        "edge_source": edge.source,
                        "edge_target": edge.target,
                        "source": edge.source,
                        "target": edge.target,
                        "parameter": parameter,
                        "direction": direction,
                        "manipulation_level": route["manipulation_level"],
                        "manipulation_success": success,
                        "primary_class": evidence,
                        "underlying_class": evidence,
                        "intervention_scope": route["intervention_scope"],
                        "root_onset": root_onset,
                        "source_onset": source_onset,
                        "target_onset": target_onset,
                        "intervention_delay": delay,
                        "observational_lag": edge.lag,
                        "lag_difference": lag_difference,
                        "root_effect": root_effect,
                        "source_effect": source_effect,
                        "target_effect": target_effect,
                        "not_applicable_reason": "",
                    }
                )
    frame = pd.DataFrame(rows, columns=CLASSIFICATION_COLUMNS)
    unknown = sorted(set(frame["primary_class"]) - INTERVENTION_CLASSES)
    if unknown:
        raise RuntimeError(f"unknown intervention classification values: {unknown}")
    return frame


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


def eligible_propagation_path_ids(
    path_summary: pd.DataFrame,
    classifications: pd.DataFrame | None = None,
    *,
    allowed_classes: tuple[str, ...] = ("supported",),
) -> set[str]:
    """Select complete, significant, ordered, intervention-supported paths."""

    if classifications is not None:
        required = {
            "scenario", "method", "parameter", "direction",
            "source", "target", "primary_class",
        }
        if not required.issubset(classifications.columns):
            raise ValueError("intervention classifications lack propagation-filter columns")
    valid_ids: set[str] = set()
    for path_id, group in path_summary.groupby("path_id"):
        order = group.sort_values(
            "scale", key=lambda values: values.map({"micro": 0, "meso": 1, "macro": 2})
        )
        onsets = order["onset_time"].to_numpy(dtype=float)
        if not (
            len(order) == 3
            and order["scale"].tolist() == ["micro", "meso", "macro"]
            and bool(order["significant"].all())
            and np.all(onsets >= 0)
            and np.all(np.diff(onsets) >= 0)
        ):
            continue
        if classifications is not None:
            first = order.iloc[0]
            scenario = str(first["scenario"])
            parameter = str(first["parameter"])
            direction = str(first["direction"])
            subset = classifications[
                (classifications["scenario"].astype(str) == scenario)
                & (classifications["method"] == "full_method")
                & (classifications["parameter"].astype(str) == parameter)
                & (classifications["direction"].astype(str) == direction)
            ]
            evidence = aggregate_edge_intervention_evidence(subset)
            supported = {
                (str(row.source), str(row.target))
                for row in evidence.itertuples()
                if row.edge_class in allowed_classes
            }
            keys = {
                (str(first["source"]), str(first["meso"])),
                (str(first["meso"]), str(first["macro"])),
            }
            if not keys.issubset(supported):
                continue
        valid_ids.add(str(path_id))
    return valid_ids


def select_representative_paths(
    path_summary: pd.DataFrame,
    classifications: pd.DataFrame | None = None,
) -> dict[str, Any]:
    selected: dict[str, Any] = {
        "selection_rule": "closest_to_median_absolute_macro_cumulative_effect_among_complete_ordered_full_method_intervention_supported_paths",
        "scenarios": {},
    }
    valid_ids = eligible_propagation_path_ids(path_summary, classifications)
    for scenario, scenario_frame in path_summary.groupby("scenario"):
        macro = scenario_frame[
            (scenario_frame["scale"] == "macro")
            & (scenario_frame["path_id"].astype(str).isin(valid_ids))
        ].copy()
        if macro.empty:
            selected["scenarios"][scenario] = {
                "path_id": None,
                "reason": "no complete ordered full-method intervention-supported path",
            }
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
                    [not_applicable_classification(
                        scenario, reason="no_temporally_retained_edges"
                    )],
                    columns=CLASSIFICATION_COLUMNS,
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
    aggregate_edge_intervention_evidence(classification_frame).to_csv(
        analysis_root / "edge_intervention_classifications.csv", index=False
    )
    timing_frame.to_csv(analysis_root / "path_timing_summary.csv", index=False)
    selection = select_representative_paths(
        timing_frame, classification_frame
    ) if not timing_frame.empty else {
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
