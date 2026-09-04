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
    "scenario", "hypothesis_group_id", "root_source", "edge_source", "edge_target",
    "source", "target", "parameter", "direction", "manipulation_level",
    "manipulation_success", "source_significant", "target_significant",
    "propagation_order_supported", "primary_class", "underlying_class",
    "intervention_scope", "root_onset", "source_onset", "target_onset",
    "intervention_delay", "observational_lag", "observational_beta",
    "lag_difference", "temporal_direction_concordant", "lag_concordant",
    "root_effect", "source_effect", "target_effect", "not_applicable_reason",
]


PATH_TIMING_COLUMNS = [
    "scenario",
    "path_id",
    "parameter",
    "direction",
    "source",
    "meso",
    "macro",
    "node_id",
    "scale",
    "onset_time",
    "onset_ci_low",
    "onset_ci_high",
    "observational_lag",
    "response_delay",
    "lag_difference",
    "cumulative_effect",
    "cumulative_effect_raw",
    "significant",
]


PATH_TIMING_CONCORDANCE_COLUMNS = [
    "scenario", "path_id", "parameter", "direction", "hypothesis_group_id",
    "micro", "meso", "macro",
    "expected_micro_response", "expected_meso_response", "expected_macro_response",
    "micro_effect", "meso_effect", "macro_effect",
    "micro_significant", "meso_significant", "macro_significant",
    "micro_onset", "meso_onset", "macro_onset",
    "micro_meso_observational_beta", "meso_macro_observational_beta",
    "micro_meso_observational_lag", "meso_macro_observational_lag",
    "micro_meso_intervention_delay", "meso_macro_intervention_delay",
    "micro_meso_temporal_direction_concordant",
    "meso_macro_temporal_direction_concordant",
    "micro_meso_lag_difference", "meso_macro_lag_difference",
    "micro_meso_lag_concordant", "meso_macro_lag_concordant",
    "observational_total_lag", "intervention_total_latency",
    "total_lag_difference", "total_lag_concordant",
    "manipulation_success", "response_direction_supported",
    "onset_order_supported", "path_classification", "reason",
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

    working = classifications.copy()
    if "hypothesis_group_id" not in working.columns:
        working["hypothesis_group_id"] = "ungrouped"
    identity_columns = ["scenario"]
    if "evaluation_track" in classifications.columns:
        identity_columns.append("evaluation_track")
    if "method" in classifications.columns:
        identity_columns.append("method")
    identity_columns.extend(["hypothesis_group_id", "source", "target"])
    columns = [
        *identity_columns, "edge_class",
        "attempt_count", "applicable_attempt_count", "supported_attempt_count",
        "contradiction_attempt_count",
    ]
    if classifications.empty:
        return pd.DataFrame(columns=columns)
    unknown = sorted(
        set(working["primary_class"].astype(str)) - INTERVENTION_CLASSES
    )
    if unknown:
        raise RuntimeError(f"unknown intervention classification values: {unknown}")
    group_columns = identity_columns
    rows: list[dict[str, Any]] = []
    for identity, group in working.groupby(
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


def mechanism_bidirectional_summary(
    effects: pd.DataFrame,
    representations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Expose both measured directions without changing path classification."""

    identity = [
        "scenario", "parameter", "micro", "meso", "macro",
        "primary_path_direction",
    ]
    measures = [
        f"{direction}_{scale}_{measure}"
        for direction in ("minus", "plus")
        for scale in ("micro", "meso", "macro")
        for measure in (
            "effect", "ci_low", "ci_high", "onset", "significant", "effect_sign",
        )
    ]
    rows: list[dict[str, Any]] = []
    for scenario, representation in sorted(representations.items()):
        scenario_effects = effects[effects["scenario"].astype(str) == scenario]
        for path in representation.get("candidate_paths", []):
            parameter = str(path["parameter"])
            nodes = {
                "micro": str(path["micro_indicator"]),
                "meso": str(path["meso_indicator"]),
                "macro": str(path["macro_indicator"]),
            }
            row: dict[str, Any] = {
                "scenario": scenario,
                "parameter": parameter,
                **nodes,
                "primary_path_direction": str(path["intervention_direction"]),
            }
            for direction in ("minus", "plus"):
                for scale, node_id in nodes.items():
                    match = scenario_effects[
                        (scenario_effects["parameter"].astype(str) == parameter)
                        & (scenario_effects["direction"].astype(str) == direction)
                        & (scenario_effects["node_id"].astype(str) == node_id)
                    ]
                    item = match.iloc[0] if len(match) else None
                    prefix = f"{direction}_{scale}"
                    row[f"{prefix}_effect"] = (
                        float(item["cumulative_effect_standardised"])
                        if item is not None else np.nan
                    )
                    row[f"{prefix}_ci_low"] = (
                        float(item["cumulative_ci_low_standardised"])
                        if item is not None else np.nan
                    )
                    row[f"{prefix}_ci_high"] = (
                        float(item["cumulative_ci_high_standardised"])
                        if item is not None else np.nan
                    )
                    row[f"{prefix}_onset"] = (
                        int(item["onset_time"]) if item is not None else -1
                    )
                    row[f"{prefix}_significant"] = (
                        bool(item["significant"]) if item is not None else False
                    )
                    row[f"{prefix}_effect_sign"] = (
                        int(item["effect_sign"]) if item is not None else 0
                    )
            rows.append(row)
    return pd.DataFrame(rows, columns=[*identity, *measures])


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
        paths = representation.get("candidate_paths", [])
        roots = sorted(
            str(path["micro_indicator"])
            for path in paths
            if str(path["meso_indicator"]) == edge_source
            and str(path["macro_indicator"]) == edge_target
        )
        if not paths:
            roots = sorted(
                str(edge["source"])
                for edge in representation.get("candidate_edges", [])
                if str(edge["target"]) == edge_source
                and scales.get(str(edge["source"])) == "micro"
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
    hypothesis_group_id: str = "ungrouped",
    source: str = "",
    target: str = "",
    observational_lag: float = np.nan,
    observational_beta: float = np.nan,
    reason: str,
) -> dict[str, Any]:
    return {
        "scenario": scenario,
        "hypothesis_group_id": hypothesis_group_id,
        "root_source": "",
        "edge_source": source,
        "edge_target": target,
        "source": source,
        "target": target,
        "parameter": "",
        "direction": "",
        "manipulation_level": "none",
        "manipulation_success": False,
        "source_significant": False,
        "target_significant": False,
        "propagation_order_supported": False,
        "primary_class": "not_applicable",
        "underlying_class": "not_applicable",
        "intervention_scope": "none",
        "root_onset": -1,
        "source_onset": -1,
        "target_onset": -1,
        "intervention_delay": np.nan,
        "observational_lag": observational_lag,
        "observational_beta": observational_beta,
        "lag_difference": np.nan,
        "temporal_direction_concordant": pd.NA,
        "lag_concordant": pd.NA,
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
    *,
    observational_hard_gates: bool = False,
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
                    hypothesis_group_id=edge.hypothesis_group_id,
                    source=edge.source,
                    target=edge.target,
                    observational_lag=edge.lag,
                    observational_beta=edge.beta,
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
                    source_significant = target_significant = False
                    propagation_order_supported = False
                    temporal_direction_concordant = lag_concordant = pd.NA
                    root_onset = source_onset = target_onset = -1
                    root_effect = source_effect = target_effect = np.nan
                    delay = lag_difference = np.nan
                else:
                    root_onset = int(root.onset_time)
                    source_onset, target_onset = int(source.onset_time), int(target.onset_time)
                    root_effect = float(root.cumulative_effect_standardised)
                    source_effect = float(source.cumulative_effect_standardised)
                    target_effect = float(target.cumulative_effect_standardised)
                    success = bool(root.significant and np.isfinite(root_effect))
                    source_significant = bool(source.significant)
                    target_significant = bool(target.significant)
                    delay = (
                        float(target_onset - source_onset)
                        if source_onset >= 0 and target_onset >= 0
                        else np.nan
                    )
                    lag_difference = delay - edge.lag if np.isfinite(delay) else np.nan
                    propagation_order_supported = bool(
                        source_onset >= 0
                        and target_onset >= 0
                        and target_onset >= source_onset
                    )
                    legacy_propagation_order_supported = bool(
                        root_onset >= 0
                        and source_onset >= root_onset
                        and target_onset >= source_onset
                    )
                    expected_target_sign = (
                        int(np.sign(source_effect * edge.beta))
                        if np.isfinite(source_effect) and np.isfinite(edge.beta)
                        else 0
                    )
                    observed_target_sign = (
                        int(np.sign(target_effect)) if np.isfinite(target_effect) else 0
                    )
                    temporal_direction_concordant = (
                        bool(observed_target_sign == expected_target_sign)
                        if expected_target_sign != 0 and observed_target_sign != 0
                        else pd.NA
                    )
                    lag_concordant = (
                        bool(abs(lag_difference) <= lag_tolerance)
                        if np.isfinite(lag_difference) else pd.NA
                    )
                    if not np.isfinite(root_effect):
                        evidence = "inconclusive"
                    elif not root.significant:
                        evidence = "manipulation_failure"
                    elif not np.isfinite(source_effect) or not np.isfinite(target_effect):
                        evidence = "inconclusive"
                    elif not source.significant or not target.significant:
                        evidence = "no_stable_downstream_effect"
                    elif observational_hard_gates:
                        if expected_target_sign == 0:
                            evidence = "inconclusive"
                        elif not bool(temporal_direction_concordant):
                            evidence = "directionally_contradicted"
                        elif (
                            legacy_propagation_order_supported
                            and bool(lag_concordant)
                        ):
                            evidence = "supported"
                        else:
                            evidence = "inconclusive"
                    elif propagation_order_supported:
                        evidence = "supported"
                    else:
                        evidence = "inconclusive"
                rows.append(
                    {
                        "scenario": scenario,
                        "hypothesis_group_id": edge.hypothesis_group_id,
                        "root_source": root_source,
                        "edge_source": edge.source,
                        "edge_target": edge.target,
                        "source": edge.source,
                        "target": edge.target,
                        "parameter": parameter,
                        "direction": direction,
                        "manipulation_level": route["manipulation_level"],
                        "manipulation_success": success,
                        "source_significant": source_significant,
                        "target_significant": target_significant,
                        "propagation_order_supported": propagation_order_supported,
                        "primary_class": evidence,
                        "underlying_class": evidence,
                        "intervention_scope": route["intervention_scope"],
                        "root_onset": root_onset,
                        "source_onset": source_onset,
                        "target_onset": target_onset,
                        "intervention_delay": delay,
                        "observational_lag": edge.lag,
                        "observational_beta": edge.beta,
                        "lag_difference": lag_difference,
                        "temporal_direction_concordant": temporal_direction_concordant,
                        "lag_concordant": lag_concordant,
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
    retained = {
        (edge.source, edge.target, edge.hypothesis_group_id) for edge in graph
    }
    return sorted(
        (
            str(path["micro_indicator"]),
            str(path["meso_indicator"]),
            str(path["macro_indicator"]),
        )
        for path in representation.get("candidate_paths", [])
        if (
            str(path["micro_indicator"]),
            str(path["meso_indicator"]),
            f"macro_outcome_{path['macro_indicator']}",
        ) in retained
        and (
            str(path["meso_indicator"]),
            str(path["macro_indicator"]),
            f"macro_outcome_{path['macro_indicator']}",
        ) in retained
    )


PATH_TEMPORAL_QUALIFICATION_COLUMNS = [
    "scenario", "path_id", "parameter", "hypothesis_group_id",
    "micro", "meso", "macro",
    "micro_meso_retained", "micro_meso_lag", "micro_meso_beta",
    "micro_meso_q", "micro_meso_support",
    "meso_macro_retained", "meso_macro_lag", "meso_macro_beta",
    "meso_macro_q", "meso_macro_support", "path_temporally_qualified",
]


def qualify_candidate_paths(
    scenario: str,
    graph: Sequence[TemporalEdge],
    representation: dict[str, Any],
) -> pd.DataFrame:
    """Map retained group-specific edges back to pre-frozen CandidatePath IDs."""

    retained = {
        (edge.source, edge.target, edge.hypothesis_group_id): edge for edge in graph
    }
    rows: list[dict[str, Any]] = []
    for path in representation.get("candidate_paths", []):
        micro = str(path["micro_indicator"])
        meso = str(path["meso_indicator"])
        macro = str(path["macro_indicator"])
        group = f"macro_outcome_{macro}"
        first = retained.get((micro, meso, group))
        second = retained.get((meso, macro, group))
        row = {
            "scenario": scenario,
            "path_id": str(path["path_id"]),
            "parameter": str(path["parameter"]),
            "hypothesis_group_id": group,
            "micro": micro,
            "meso": meso,
            "macro": macro,
            "micro_meso_retained": first is not None,
            "micro_meso_lag": first.lag if first else np.nan,
            "micro_meso_beta": first.beta if first else np.nan,
            "micro_meso_q": first.q_value if first else np.nan,
            "micro_meso_support": first.support if first else np.nan,
            "meso_macro_retained": second is not None,
            "meso_macro_lag": second.lag if second else np.nan,
            "meso_macro_beta": second.beta if second else np.nan,
            "meso_macro_q": second.q_value if second else np.nan,
            "meso_macro_support": second.support if second else np.nan,
            "path_temporally_qualified": first is not None and second is not None,
        }
        rows.append(row)
    return pd.DataFrame(rows, columns=PATH_TEMPORAL_QUALIFICATION_COLUMNS)


def path_timing_summary(
    scenario: str,
    graph: Sequence[TemporalEdge],
    effects: pd.DataFrame,
    representation: dict[str, Any],
) -> pd.DataFrame:
    qualified = qualify_candidate_paths(scenario, graph, representation)
    paths = {
        str(path["path_id"]): path for path in representation.get("candidate_paths", [])
    }
    rows: list[dict[str, Any]] = []
    for temporal in qualified.itertuples(index=False):
        if not bool(temporal.path_temporally_qualified):
            continue
        path = paths[str(temporal.path_id)]
        source, meso, macro = str(temporal.micro), str(temporal.meso), str(temporal.macro)
        parameter = str(path["parameter"])
        direction = str(path["intervention_direction"])
        subset = effects[
            (effects["scenario"] == scenario)
            & (effects["parameter"] == parameter)
            & (effects["direction"] == direction)
            & (effects["node_id"].isin([source, meso, macro]))
        ]
        if len(subset) != 3:
            continue
        lookup = {row.node_id: row for row in subset.itertuples()}
        lags = {meso: temporal.micro_meso_lag, macro: temporal.meso_macro_lag}
        for scale, node, parent in (
            ("micro", source, None), ("meso", meso, source), ("macro", macro, meso)
        ):
            item = lookup[node]
            parent_onset = lookup[parent].onset_time if parent else np.nan
            response_delay = (
                item.onset_time - parent_onset
                if parent and item.onset_time >= 0 and parent_onset >= 0 else np.nan
            )
            observational_lag = lags.get(node, np.nan)
            rows.append(
                {
                    "scenario": scenario, "path_id": str(path["path_id"]),
                    "parameter": parameter, "direction": direction,
                    "source": source, "meso": meso, "macro": macro,
                    "node_id": node, "scale": scale,
                    "onset_time": item.onset_time, "onset_ci_low": item.onset_ci_low,
                    "onset_ci_high": item.onset_ci_high,
                    "observational_lag": observational_lag,
                    "response_delay": response_delay,
                    "lag_difference": response_delay - observational_lag
                    if np.isfinite(response_delay) and np.isfinite(observational_lag) else np.nan,
                    "cumulative_effect": item.cumulative_effect_standardised,
                    "cumulative_effect_raw": item.cumulative_effect_raw,
                    "significant": item.significant,
                }
            )
    # An empty path set is a valid scientific result.  Preserve the stable
    # schema so exporters and renderers can distinguish zero records from a
    # malformed headerless file.
    return pd.DataFrame(rows, columns=PATH_TIMING_COLUMNS)


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
            "root_source", "source", "target", "primary_class",
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
            root_source = str(first["source"])
            subset = classifications[
                (classifications["scenario"].astype(str) == scenario)
                & (classifications["method"] == "full_method")
                & (classifications["parameter"].astype(str) == parameter)
                & (classifications["direction"].astype(str) == direction)
                & (classifications["root_source"].astype(str) == root_source)
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


PATH_INTERVENTION_CLASSIFICATION_COLUMNS = [
    "scenario", "path_id", "parameter", "direction", "hypothesis_group_id",
    "micro", "meso", "macro",
    "path_temporally_qualified", "manipulation_success",
    "micro_significant", "meso_significant", "macro_significant",
    "micro_meso_class", "meso_macro_class", "direction_supported",
    "onset_order_supported", "path_classification", "reason",
]


def classify_candidate_paths(
    temporal_qualification: pd.DataFrame,
    classifications: pd.DataFrame,
    representations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    """Classify every frozen hypothesis without reconstructing or cherry-picking paths."""

    rows: list[dict[str, Any]] = []
    for scenario, representation in sorted(representations.items()):
        paths = {
            str(path["path_id"]): path
            for path in representation.get("candidate_paths", [])
        }
        temporal_by_id = {
            str(row.path_id): row
            for row in temporal_qualification[
                temporal_qualification["scenario"].astype(str) == scenario
            ].itertuples(index=False)
        }
        for path_id, path in sorted(paths.items()):
            micro = str(path["micro_indicator"])
            meso = str(path["meso_indicator"])
            macro = str(path["macro_indicator"])
            parameter = str(path["parameter"])
            direction = str(path["intervention_direction"])
            hypothesis_group_id = f"macro_outcome_{macro}"
            temporal = temporal_by_id.get(path_id)
            qualified = bool(
                temporal is not None and temporal.path_temporally_qualified
            )
            subset = classifications[
                (classifications["scenario"].astype(str) == scenario)
                & (
                    classifications["method"].astype(str).isin(
                        {"full_method", "frozen_full_method"}
                    )
                )
                & (classifications["parameter"].astype(str) == parameter)
                & (classifications["direction"].astype(str) == direction)
                & (classifications["root_source"].astype(str) == micro)
                & (
                    classifications["hypothesis_group_id"].astype(str)
                    == hypothesis_group_id
                )
            ]
            evidence = aggregate_edge_intervention_evidence(subset)
            edge_classes = {
                (str(row.source), str(row.target)): str(row.edge_class)
                for row in evidence.itertuples(index=False)
            }
            first_class = edge_classes.get((micro, meso), "inconclusive")
            second_class = edge_classes.get((meso, macro), "inconclusive")
            first_rows = subset[
                (subset["source"].astype(str) == micro)
                & (subset["target"].astype(str) == meso)
            ]
            second_rows = subset[
                (subset["source"].astype(str) == meso)
                & (subset["target"].astype(str) == macro)
            ]
            manipulation_success = bool(
                len(first_rows) and first_rows["manipulation_success"].astype(bool).any()
            )
            first = first_rows.iloc[0] if len(first_rows) else None
            second = second_rows.iloc[0] if len(second_rows) else None
            effects = (
                float(first["root_effect"]) if first is not None else np.nan,
                float(first["target_effect"]) if first is not None else np.nan,
                float(second["target_effect"]) if second is not None else np.nan,
            )
            micro_significant = bool(
                first is not None and first.get("manipulation_success", False)
            )
            meso_significant = bool(
                first is not None and first.get("target_significant", False)
            )
            macro_significant = bool(
                second is not None and second.get("target_significant", False)
            )
            expected_responses = (
                str(path["expected_micro_response"]),
                str(path["expected_meso_response"]),
                str(path["expected_macro_response"]),
            )
            response_matches = [
                np.isfinite(effect)
                and int(np.sign(effect)) == (1 if expected == "increase" else -1)
                for effect, expected in zip(effects, expected_responses)
            ]
            direction_supported = bool(all(response_matches))
            onsets = (
                float(first["root_onset"]) if first is not None else np.nan,
                float(first["target_onset"]) if first is not None else np.nan,
                float(second["target_onset"]) if second is not None else np.nan,
            )
            onset_order_supported = bool(
                all(np.isfinite(value) and value >= 0 for value in onsets)
                and np.all(np.diff(np.asarray(onsets, dtype=float)) >= 0)
            )
            if not qualified:
                classification = "inconclusive"
                reason = "path_not_temporally_qualified"
            elif not manipulation_success:
                classification = "manipulation_failure"
                reason = "required_micro_manipulation_failed"
            elif not (
                micro_significant and meso_significant and macro_significant
                and all(np.isfinite(value) for value in effects)
            ):
                classification = "inconclusive"
                reason = "required_multiscale_response_not_stable"
            elif not direction_supported:
                classification = "contradicted"
                reason = "frozen_response_direction_contradicted"
            elif not onset_order_supported:
                classification = "inconclusive"
                reason = "intervention_onset_order_not_supported"
            else:
                classification = "supported"
                reason = "all_stage3_v2_hard_criteria_supported"
            rows.append(
                {
                    "scenario": scenario, "path_id": path_id,
                    "parameter": parameter, "direction": direction,
                    "hypothesis_group_id": hypothesis_group_id,
                    "micro": micro, "meso": meso, "macro": macro,
                    "path_temporally_qualified": qualified,
                    "manipulation_success": manipulation_success,
                    "micro_significant": micro_significant,
                    "meso_significant": meso_significant,
                    "macro_significant": macro_significant,
                    "micro_meso_class": first_class,
                    "meso_macro_class": second_class,
                    "direction_supported": direction_supported,
                    "onset_order_supported": onset_order_supported,
                    "path_classification": classification,
                    "reason": reason,
                }
            )
    return pd.DataFrame(rows, columns=PATH_INTERVENTION_CLASSIFICATION_COLUMNS)


def path_timing_concordance(
    temporal_qualification: pd.DataFrame,
    effects: pd.DataFrame,
    representations: dict[str, dict[str, Any]],
    path_classification: pd.DataFrame,
    lag_tolerance: int,
) -> pd.DataFrame:
    """Report observational/intervention concordance without using it as a gate."""

    classified = {
        (str(row.scenario), str(row.path_id)): row
        for row in path_classification.itertuples(index=False)
    }
    rows: list[dict[str, Any]] = []
    qualified = temporal_qualification[
        temporal_qualification["path_temporally_qualified"].astype(bool)
    ]
    for temporal in qualified.itertuples(index=False):
        scenario = str(temporal.scenario)
        path_id = str(temporal.path_id)
        paths = {
            str(path["path_id"]): path
            for path in representations[scenario].get("candidate_paths", [])
        }
        path = paths[path_id]
        parameter = str(path["parameter"])
        direction = str(path["intervention_direction"])
        micro = str(path["micro_indicator"])
        meso = str(path["meso_indicator"])
        macro = str(path["macro_indicator"])
        selected = effects[
            (effects["scenario"].astype(str) == scenario)
            & (effects["parameter"].astype(str) == parameter)
            & (effects["direction"].astype(str) == direction)
            & (effects["node_id"].astype(str).isin([micro, meso, macro]))
        ]
        lookup = {
            str(item.node_id): item for item in selected.itertuples(index=False)
        }

        def effect_value(node: str) -> float:
            item = lookup.get(node)
            return (
                float(item.cumulative_effect_standardised)
                if item is not None else np.nan
            )

        def significant(node: str) -> bool:
            item = lookup.get(node)
            return bool(item is not None and item.significant)

        def onset(node: str) -> int:
            item = lookup.get(node)
            return int(item.onset_time) if item is not None else -1

        micro_effect, meso_effect, macro_effect = (
            effect_value(micro), effect_value(meso), effect_value(macro)
        )
        micro_onset, meso_onset, macro_onset = (
            onset(micro), onset(meso), onset(macro)
        )
        micro_meso_delay = (
            float(meso_onset - micro_onset)
            if micro_onset >= 0 and meso_onset >= 0 else np.nan
        )
        meso_macro_delay = (
            float(macro_onset - meso_onset)
            if meso_onset >= 0 and macro_onset >= 0 else np.nan
        )
        micro_meso_lag = float(temporal.micro_meso_lag)
        meso_macro_lag = float(temporal.meso_macro_lag)
        micro_meso_difference = micro_meso_delay - micro_meso_lag
        meso_macro_difference = meso_macro_delay - meso_macro_lag

        def temporal_direction_concordant(
            parent_effect: float, child_effect: float, beta: float
        ) -> bool | Any:
            if not all(np.isfinite(value) for value in (parent_effect, child_effect, beta)):
                return pd.NA
            response_sign = int(np.sign(parent_effect * child_effect))
            beta_sign = int(np.sign(beta))
            return bool(response_sign != 0 and beta_sign != 0 and response_sign == beta_sign)

        observational_total_lag = micro_meso_lag + meso_macro_lag
        intervention_total_latency = (
            float(macro_onset - micro_onset)
            if micro_onset >= 0 and macro_onset >= 0 else np.nan
        )
        total_lag_difference = intervention_total_latency - observational_total_lag
        path_result = classified.get((scenario, path_id))
        rows.append(
            {
                "scenario": scenario, "path_id": path_id,
                "parameter": parameter, "direction": direction,
                "hypothesis_group_id": f"macro_outcome_{macro}",
                "micro": micro, "meso": meso, "macro": macro,
                "expected_micro_response": path["expected_micro_response"],
                "expected_meso_response": path["expected_meso_response"],
                "expected_macro_response": path["expected_macro_response"],
                "micro_effect": micro_effect, "meso_effect": meso_effect,
                "macro_effect": macro_effect,
                "micro_significant": significant(micro),
                "meso_significant": significant(meso),
                "macro_significant": significant(macro),
                "micro_onset": micro_onset, "meso_onset": meso_onset,
                "macro_onset": macro_onset,
                "micro_meso_observational_beta": temporal.micro_meso_beta,
                "meso_macro_observational_beta": temporal.meso_macro_beta,
                "micro_meso_observational_lag": micro_meso_lag,
                "meso_macro_observational_lag": meso_macro_lag,
                "micro_meso_intervention_delay": micro_meso_delay,
                "meso_macro_intervention_delay": meso_macro_delay,
                "micro_meso_temporal_direction_concordant": temporal_direction_concordant(
                    micro_effect, meso_effect, float(temporal.micro_meso_beta)
                ),
                "meso_macro_temporal_direction_concordant": temporal_direction_concordant(
                    meso_effect, macro_effect, float(temporal.meso_macro_beta)
                ),
                "micro_meso_lag_difference": micro_meso_difference,
                "meso_macro_lag_difference": meso_macro_difference,
                "micro_meso_lag_concordant": bool(
                    abs(micro_meso_difference) <= lag_tolerance
                ),
                "meso_macro_lag_concordant": bool(
                    abs(meso_macro_difference) <= lag_tolerance
                ),
                "observational_total_lag": observational_total_lag,
                "intervention_total_latency": intervention_total_latency,
                "total_lag_difference": total_lag_difference,
                "total_lag_concordant": bool(
                    np.isfinite(total_lag_difference)
                    and abs(total_lag_difference) <= lag_tolerance
                ),
                "manipulation_success": bool(
                    significant(micro) and np.isfinite(micro_effect)
                ),
                "response_direction_supported": bool(
                    path_result is not None and path_result.direction_supported
                ),
                "onset_order_supported": bool(
                    path_result is not None and path_result.onset_order_supported
                ),
                "path_classification": (
                    str(path_result.path_classification)
                    if path_result is not None else "inconclusive"
                ),
                "reason": (
                    str(path_result.reason)
                    if path_result is not None else "missing_path_classification"
                ),
            }
        )
    return pd.DataFrame(rows, columns=PATH_TIMING_CONCORDANCE_COLUMNS)


def select_representative_paths(
    path_classification: pd.DataFrame,
    holdout_confirmation: pd.DataFrame | None = None,
) -> dict[str, Any]:
    selected: dict[str, Any] = {
        "selection_rule": "supported_and_holdout_confirmed_then_lexical_path_id_else_primary_supported_then_lexical_path_id",
        "scenarios": {},
    }
    for scenario, scenario_frame in path_classification.groupby("scenario"):
        supported = scenario_frame[
            scenario_frame["path_classification"].astype(str) == "supported"
        ].copy()
        if supported.empty:
            selected["scenarios"][scenario] = {
                "path_id": None,
                "reason": "no frozen candidate path has complete intervention support",
            }
            continue
        confirmed: set[str] = set()
        if holdout_confirmation is not None and not holdout_confirmation.empty:
            column = (
                "holdout_confirmed"
                if "holdout_confirmed" in holdout_confirmation.columns
                else "confirmed"
            )
            if column in holdout_confirmation.columns:
                confirmed = set(
                    holdout_confirmation[
                        (holdout_confirmation["scenario"].astype(str) == str(scenario))
                        & (holdout_confirmation[column].astype(bool))
                    ]["path_id"].astype(str)
                )
        preferred = supported[supported["path_id"].astype(str).isin(confirmed)]
        pool = preferred if not preferred.empty else supported
        path_id = sorted(pool["path_id"].astype(str))[0]
        selected["scenarios"][scenario] = {
            "path_id": path_id,
            "evidence_scope": "holdout_confirmed" if path_id in confirmed else "primary_only",
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
    qualifications = []
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
        qualifications.append(
            qualify_candidate_paths(scenario, graph, representation)
        )
        timings.append(
            path_timing_summary(scenario, graph, effects, representation)
        )
    classification_frame = pd.concat(classifications, ignore_index=True)
    timing_frame = pd.concat(timings, ignore_index=True) if timings else pd.DataFrame()
    qualification_frame = (
        pd.concat(qualifications, ignore_index=True)
        if qualifications else pd.DataFrame(columns=PATH_TEMPORAL_QUALIFICATION_COLUMNS)
    )
    path_classification = classify_candidate_paths(
        qualification_frame, classification_frame, representations
    )
    timing_concordance = path_timing_concordance(
        qualification_frame,
        effects,
        representations,
        path_classification,
        int(config["intervention"]["lag_tolerance"]),
    )
    effects.insert(0, "evaluation_track", "primary_discovery")
    curves.insert(0, "evaluation_track", "primary_discovery")
    classification_frame.insert(0, "evaluation_track", "primary_discovery")
    if "evaluation_track" not in timing_frame.columns:
        timing_frame.insert(0, "evaluation_track", "primary_discovery")
    analysis_root = run_root / "analysis"
    effects.to_parquet(analysis_root / "paired_effects.parquet", index=False)
    curves.to_parquet(analysis_root / "effect_curves.parquet", index=False)
    mechanism_bidirectional_summary(effects, representations).to_csv(
        analysis_root / "mechanism_bidirectional_summary.csv", index=False
    )
    classification_frame.to_csv(
        analysis_root / "intervention_classifications.csv", index=False
    )
    edge_evidence = aggregate_edge_intervention_evidence(classification_frame)
    edge_evidence.to_csv(analysis_root / "edge_intervention_classifications.csv", index=False)
    qualification_frame.to_csv(
        analysis_root / "path_temporal_qualification.csv", index=False
    )
    path_classification.to_csv(
        analysis_root / "path_intervention_classification.csv", index=False
    )
    timing_frame.to_csv(analysis_root / "path_timing_summary.csv", index=False)
    timing_concordance.to_csv(
        analysis_root / "path_timing_concordance.csv", index=False
    )
    selection = select_representative_paths(path_classification)
    (analysis_root / "representative_path_selection.json").write_text(
        json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "effect_rows": len(effects),
        "curve_rows": len(curves),
        "classification_rows": len(classification_frame),
        "path_rows": len(timing_frame),
        "path_classification_rows": len(path_classification),
        "path_timing_concordance_rows": len(timing_concordance),
        "stage3_method_version": "intervention_path_classification_v2",
    }
