"""Secondary, independent-confirmation, and falsification analyses.

The functions in this module deliberately operate *after* the primary graph,
paths, and prospective predictions have been frozen.  They never select a
representation, graph, lag, threshold, path, or prediction, and none of their
outputs is a valid input to the primary intervention classifier.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Mapping, Sequence
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from .prospective import classify_prediction_requirements
from .temporal import (
    TemporalEdge,
    discover_bootstrap_graph,
    representation_candidates,
    stable_seed,
)


DOSE_LABELS = ("minus", "mid_minus", "baseline", "mid_plus", "plus")
NORMALIZED_DOSE = {
    "minus": -1.0,
    "mid_minus": -0.5,
    "baseline": 0.0,
    "mid_plus": 0.5,
    "plus": 1.0,
}

DOSE_EFFECT_COLUMNS = [
    "evaluation_track",
    "scenario",
    "parameter",
    "dose_label",
    "dose_value",
    "normalized_dose",
    "node_id",
    "scale",
    "mean_raw_effect",
    "mean_standardized_effect",
    "bootstrap_ci_low",
    "bootstrap_ci_high",
    "bootstrap_ci_low_raw",
    "bootstrap_ci_high_raw",
    "paired_seed_sign_consistency",
    "paired_seed_count",
    "baseline_sd",
    "is_mid_dose",
    "primary_support_input",
]

DOSE_SUMMARY_COLUMNS = [
    "evaluation_track",
    "scenario",
    "parameter",
    "node_id",
    "scale",
    "dose_point_count",
    "dose_response_slope",
    "slope_ci_low",
    "slope_ci_high",
    "spearman_rho",
    "monotonic_direction_score",
    "adjacent_dose_consistency",
    "quadratic_delta_r2",
    "nonlinearity_indicator",
    "slope_bootstrap_unit",
    "primary_classification_changed",
]

HOLDOUT_PATH_COLUMNS = [
    "evaluation_track",
    "scenario",
    "path_id",
    "parameter",
    "direction",
    "hypothesis_group_id",
    "micro",
    "source",
    "meso",
    "macro",
    "micro_meso_confirmation",
    "meso_macro_confirmation",
    "classification",
    "holdout_confirmed",
    "frozen_definition_sha256",
    "primary_result_unchanged",
]

HOLDOUT_PROSPECTIVE_COLUMNS = [
    "evaluation_track",
    "scenario",
    "prediction_id",
    "candidate_path_id",
    "parameter",
    "intervention_direction",
    "micro",
    "meso",
    "macro",
    "classification",
    "path_temporally_qualified",
    "direction_supported",
    "onset_order_supported",
    "required_downstream_responses_supported",
    "prediction_sha256",
    "primary_result_unchanged",
]

HOLDOUT_MECHANISM_COLUMNS = [
    "evaluation_track",
    "scenario",
    "root_source",
    "source",
    "target",
    "parameter",
    "direction",
    "primary_class",
    "holdout_class",
    "classification",
    "retained_in_holdout_baseline",
    "retained_in_holdout_mechanism_disabled",
    "frozen_definition_sha256",
    "primary_result_unchanged",
]

NEGATIVE_CONTROL_COLUMNS = [
    "evaluation_track",
    "scenario",
    "repetition",
    "candidate_edge_count",
    "retained_edge_count",
    "qualification_rate",
    "candidate_path_count",
    "qualified_path_count",
    "path_qualification_rate",
    "stability",
    "primary_retained_edge_count",
    "retained_edge_difference_from_primary",
    "retained_edge_ratio_to_primary",
    "minimum_applied_shift",
    "maximum_applied_shift",
    "maximum_lag",
    "control_only",
]

SHIFT_RECORD_COLUMNS = [
    "scenario",
    "repetition",
    "trajectory_index",
    "trajectory_seed",
    "parent",
    "shift",
    "maximum_lag",
]

PATH_ATTENUATION_COLUMNS = [
    "evaluation_track",
    "scenario",
    "path_id",
    "mechanism_variant",
    "node_id",
    "scale",
    "baseline_effect",
    "disabled_effect",
    "disabled_to_baseline_ratio",
    "attenuation_ratio",
    "temporal_edge_retained_baseline",
    "temporal_edge_retained_disabled",
    "uses_hidden_truth",
]


def _empty(columns: Sequence[str]) -> pd.DataFrame:
    return pd.DataFrame(columns=list(columns))


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], label: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{label} is missing required columns: {missing}")


def _canonical_hash(value: Mapping[str, Any]) -> str:
    payload = json.dumps(
        dict(value), sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def five_point_dose_levels(levels: Sequence[float] | Mapping[str, float]) -> dict[str, float]:
    """Construct the frozen five-point grid within the original safe range."""

    if isinstance(levels, Mapping):
        try:
            minus = float(levels["minus"])
            baseline = float(levels["baseline"])
            plus = float(levels["plus"])
        except KeyError as exc:
            raise ValueError("dose mapping requires minus, baseline, and plus") from exc
    else:
        if len(levels) != 3:
            raise ValueError("intervention levels must contain minus, baseline, plus")
        minus, baseline, plus = (float(value) for value in levels)
    if not all(np.isfinite([minus, baseline, plus])):
        raise ValueError("dose values must be finite")
    if not minus < baseline < plus:
        raise ValueError("dose values must satisfy minus < baseline < plus")
    return {
        "minus": minus,
        "mid_minus": (minus + baseline) / 2.0,
        "baseline": baseline,
        "mid_plus": (baseline + plus) / 2.0,
        "plus": plus,
    }


def _paired_dose_seed_effects(
    dataset: pd.DataFrame,
    representations: Mapping[str, Mapping[str, Any]],
    scenario_specs: Mapping[str, Mapping[str, Any]],
    *,
    evaluation_start: int,
) -> pd.DataFrame:
    _require_columns(dataset, {"scenario", "seed", "condition", "time"}, "dataset")
    rows: list[dict[str, Any]] = []
    for scenario, representation in sorted(representations.items()):
        if scenario not in scenario_specs:
            raise ValueError(f"missing scenario specification: {scenario}")
        indicators = list(representation.get("indicators", []))
        node_ids = [str(item["id"]) for item in indicators]
        scales = {str(item["id"]): str(item["scale"]) for item in indicators}
        _require_columns(dataset, node_ids, f"{scenario} indicator dataset")
        scenario_data = dataset[dataset["scenario"].astype(str) == str(scenario)]
        # A caller may pass a concatenated primary+holdout table.  Dose
        # characterisation is a primary-seed secondary analysis and must not
        # silently pool independent confirmation rows with discovery rows.
        if "data_partition" in scenario_data.columns:
            scenario_data = scenario_data[
                scenario_data["data_partition"].astype(str) == "primary"
            ]
        baseline = scenario_data[scenario_data["condition"].astype(str) == "baseline"]
        if baseline.empty:
            raise ValueError(f"{scenario} has no primary baseline rows")
        seed_ids = sorted(int(seed) for seed in baseline["seed"].unique())
        baseline_eval = baseline[baseline["time"].astype(int) >= int(evaluation_start)]
        baseline_sd = baseline_eval[node_ids].std(axis=0, ddof=1).to_dict()
        spec = scenario_specs[scenario]
        for parameter, original_levels in sorted(spec["interventions"].items()):
            doses = five_point_dose_levels(original_levels)
            for dose_label in DOSE_LABELS:
                condition = (
                    "baseline" if dose_label == "baseline" else f"{parameter}_{dose_label}"
                )
                treated = scenario_data[
                    scenario_data["condition"].astype(str) == condition
                ]
                treated_seeds = sorted(int(seed) for seed in treated["seed"].unique())
                if treated_seeds != seed_ids:
                    raise ValueError(
                        f"paired seed mismatch for {scenario}:{parameter}:{dose_label}"
                    )
                for seed in seed_ids:
                    left = baseline[
                        (baseline["seed"].astype(int) == seed)
                        & (baseline["time"].astype(int) >= int(evaluation_start))
                    ].sort_values("time")
                    right = treated[
                        (treated["seed"].astype(int) == seed)
                        & (treated["time"].astype(int) >= int(evaluation_start))
                    ].sort_values("time")
                    if left["time"].astype(int).tolist() != right["time"].astype(int).tolist():
                        raise ValueError(
                            f"paired time mismatch for {scenario}:{parameter}:{dose_label}:{seed}"
                        )
                    raw = right[node_ids].to_numpy(float) - left[node_ids].to_numpy(float)
                    for index, node_id in enumerate(node_ids):
                        raw_effect = float(np.mean(raw[:, index]))
                        sd = float(baseline_sd[node_id])
                        standardized = (
                            raw_effect / sd
                            if np.isfinite(sd) and sd > 1e-12
                            else float("nan")
                        )
                        rows.append(
                            {
                                "scenario": scenario,
                                "parameter": str(parameter),
                                "dose_label": dose_label,
                                "dose_value": float(doses[dose_label]),
                                "normalized_dose": NORMALIZED_DOSE[dose_label],
                                "node_id": node_id,
                                "scale": scales[node_id],
                                "seed": seed,
                                "raw_effect": raw_effect,
                                "standardized_effect": standardized,
                                "baseline_sd": sd,
                            }
                        )
    return pd.DataFrame(rows)


def _bootstrap_mean_interval(
    values: np.ndarray,
    *,
    repetitions: int,
    confidence_level: float,
    seed: int,
) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    if not len(values) or not np.all(np.isfinite(values)):
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    indices = rng.integers(0, len(values), size=(int(repetitions), len(values)))
    means = np.mean(values[indices], axis=1)
    alpha = (1.0 - float(confidence_level)) / 2.0
    return float(np.quantile(means, alpha)), float(np.quantile(means, 1.0 - alpha))


def dose_response_effects(
    dataset: pd.DataFrame,
    representations: Mapping[str, Mapping[str, Any]],
    scenario_specs: Mapping[str, Mapping[str, Any]],
    intervention_config: Mapping[str, Any],
    *,
    master_seed: int,
) -> pd.DataFrame:
    """Estimate paired five-dose effects without producing primary evidence.

    The baseline, minus, and plus values are the already frozen primary values;
    only the two arithmetic midpoints are additional.  This function never
    calls the primary edge classifier, and every output row explicitly carries
    ``primary_support_input=False``.
    """

    seed_effects = _paired_dose_seed_effects(
        dataset,
        representations,
        scenario_specs,
        evaluation_start=int(intervention_config["evaluation_start"]),
    )
    if seed_effects.empty:
        result = _empty(DOSE_EFFECT_COLUMNS)
        result.attrs["paired_seed_effects"] = seed_effects
        return result
    repetitions = int(intervention_config["bootstrap_repetitions"])
    confidence = float(intervention_config["confidence_level"])
    rows: list[dict[str, Any]] = []
    keys = ["scenario", "parameter", "dose_label", "node_id", "scale"]
    for identity, group in seed_effects.groupby(keys, sort=True, dropna=False):
        scenario, parameter, dose_label, node_id, scale = identity
        raw = group["raw_effect"].to_numpy(float)
        standardized = group["standardized_effect"].to_numpy(float)
        raw_low, raw_high = _bootstrap_mean_interval(
            raw,
            repetitions=repetitions,
            confidence_level=confidence,
            seed=stable_seed(
                master_seed, "dose", scenario, parameter, dose_label, node_id, "raw"
            ),
        )
        std_low, std_high = _bootstrap_mean_interval(
            standardized,
            repetitions=repetitions,
            confidence_level=confidence,
            seed=stable_seed(
                master_seed, "dose", scenario, parameter, dose_label, node_id, "std"
            ),
        )
        mean_standardized = (
            float(np.mean(standardized))
            if len(standardized) and np.all(np.isfinite(standardized))
            else float("nan")
        )
        sign = int(np.sign(mean_standardized)) if np.isfinite(mean_standardized) else 0
        consistency = (
            float(np.mean(np.sign(standardized) == sign))
            if np.all(np.isfinite(standardized))
            else float("nan")
        )
        rows.append(
            {
                "evaluation_track": "secondary_dose_response",
                "scenario": scenario,
                "parameter": parameter,
                "dose_label": dose_label,
                "dose_value": float(group["dose_value"].iloc[0]),
                "normalized_dose": float(group["normalized_dose"].iloc[0]),
                "node_id": node_id,
                "scale": scale,
                "mean_raw_effect": float(np.mean(raw)),
                "mean_standardized_effect": mean_standardized,
                "bootstrap_ci_low": std_low,
                "bootstrap_ci_high": std_high,
                "bootstrap_ci_low_raw": raw_low,
                "bootstrap_ci_high_raw": raw_high,
                "paired_seed_sign_consistency": consistency,
                "paired_seed_count": int(group["seed"].nunique()),
                "baseline_sd": float(group["baseline_sd"].iloc[0]),
                "is_mid_dose": dose_label in {"mid_minus", "mid_plus"},
                "primary_support_input": False,
            }
        )
    result = pd.DataFrame(rows, columns=DOSE_EFFECT_COLUMNS).sort_values(
        ["scenario", "parameter", "node_id", "normalized_dose"], ignore_index=True
    )
    result.attrs["paired_seed_effects"] = seed_effects
    return result


def _r_squared(y: np.ndarray, fitted: np.ndarray) -> float:
    denominator = float(np.sum((y - np.mean(y)) ** 2))
    if denominator <= 1e-15:
        return 1.0 if np.allclose(y, fitted) else float("nan")
    return float(1.0 - np.sum((y - fitted) ** 2) / denominator)


def dose_response_summary(
    effects: pd.DataFrame,
    *,
    bootstrap_repetitions: int,
    confidence_level: float,
    master_seed: int,
    paired_seed_effects: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarise slope, monotonicity, and nonlinearity as secondary metrics."""

    if effects.empty:
        return _empty(DOSE_SUMMARY_COLUMNS)
    _require_columns(effects, DOSE_EFFECT_COLUMNS, "dose-response effects")
    seed_effects = paired_seed_effects
    if seed_effects is None:
        candidate = effects.attrs.get("paired_seed_effects")
        seed_effects = candidate if isinstance(candidate, pd.DataFrame) else None
    keys = ["scenario", "parameter", "node_id", "scale"]
    rows: list[dict[str, Any]] = []
    for identity, group in effects.groupby(keys, sort=True, dropna=False):
        scenario, parameter, node_id, scale = identity
        group = group.sort_values("normalized_dose")
        labels = tuple(group["dose_label"].astype(str))
        if len(group) != 5 or set(labels) != set(DOSE_LABELS):
            raise ValueError(
                f"incomplete five-point dose grid for {scenario}:{parameter}:{node_id}"
            )
        x = group["normalized_dose"].to_numpy(float)
        y = group["mean_standardized_effect"].to_numpy(float)
        if not np.all(np.isfinite(y)):
            slope = low = high = rho = monotonic = adjacent = float("nan")
            delta_r2 = float("nan")
            nonlinear: bool | float = float("nan")
            bootstrap_unit = "not_estimable"
        else:
            linear = np.polyfit(x, y, 1)
            slope = float(linear[0])
            rho = float(spearmanr(x, y).statistic)
            expected_sign = int(np.sign(slope))
            pair_differences = np.asarray(
                [y[right] - y[left] for left in range(4) for right in range(left + 1, 5)]
            )
            adjacent_differences = np.diff(y)
            if expected_sign == 0:
                monotonic = float(np.mean(np.isclose(pair_differences, 0.0)))
                adjacent = float(np.mean(np.isclose(adjacent_differences, 0.0)))
            else:
                monotonic = float(np.mean(expected_sign * pair_differences >= 0.0))
                adjacent = float(np.mean(expected_sign * adjacent_differences >= 0.0))
            linear_fit = np.polyval(linear, x)
            quadratic = np.polyfit(x, y, 2)
            delta_r2 = float(
                max(0.0, _r_squared(y, np.polyval(quadratic, x)) - _r_squared(y, linear_fit))
            )
            nonlinear = bool(delta_r2 > 0.05)
            bootstrap_slopes: list[float] = []
            relevant_seed_effects = None
            if seed_effects is not None and not seed_effects.empty:
                relevant_seed_effects = seed_effects[
                    (seed_effects["scenario"].astype(str) == str(scenario))
                    & (seed_effects["parameter"].astype(str) == str(parameter))
                    & (seed_effects["node_id"].astype(str) == str(node_id))
                ]
            rng = np.random.default_rng(
                stable_seed(master_seed, "dose_slope", scenario, parameter, node_id)
            )
            if relevant_seed_effects is not None and not relevant_seed_effects.empty:
                pivot = relevant_seed_effects.pivot(
                    index="seed", columns="dose_label", values="standardized_effect"
                ).reindex(columns=list(DOSE_LABELS))
                if pivot.notna().all().all():
                    matrix = pivot.to_numpy(float)
                    for _ in range(int(bootstrap_repetitions)):
                        sampled = matrix[rng.integers(0, len(matrix), size=len(matrix))]
                        means = np.mean(sampled, axis=0)
                        bootstrap_slopes.append(
                            float(np.polyfit(np.asarray(list(NORMALIZED_DOSE.values())), means, 1)[0])
                        )
                    bootstrap_unit = "paired_seeds"
                else:
                    bootstrap_unit = "not_estimable"
            else:
                residuals = y - linear_fit
                for _ in range(int(bootstrap_repetitions)):
                    sampled = linear_fit + residuals[
                        rng.integers(0, len(residuals), size=len(residuals))
                    ]
                    bootstrap_slopes.append(float(np.polyfit(x, sampled, 1)[0]))
                bootstrap_unit = "dose_point_residuals"
            if bootstrap_slopes:
                alpha = (1.0 - float(confidence_level)) / 2.0
                low = float(np.quantile(bootstrap_slopes, alpha))
                high = float(np.quantile(bootstrap_slopes, 1.0 - alpha))
            else:
                low = high = float("nan")
        rows.append(
            {
                "evaluation_track": "secondary_dose_response",
                "scenario": scenario,
                "parameter": parameter,
                "node_id": node_id,
                "scale": scale,
                "dose_point_count": 5,
                "dose_response_slope": slope,
                "slope_ci_low": low,
                "slope_ci_high": high,
                "spearman_rho": rho,
                "monotonic_direction_score": monotonic,
                "adjacent_dose_consistency": adjacent,
                "quadratic_delta_r2": delta_r2,
                "nonlinearity_indicator": nonlinear,
                "slope_bootstrap_unit": bootstrap_unit,
                "primary_classification_changed": False,
            }
        )
    return pd.DataFrame(rows, columns=DOSE_SUMMARY_COLUMNS)


def _confirmation_class(value: str) -> str:
    if value == "supported":
        return "confirmed"
    if value == "manipulation_failure":
        return "manipulation_failure"
    if value in {"directionally_contradicted", "no_stable_downstream_effect"}:
        return "failed_confirmation"
    return "inconclusive"


def holdout_path_confirmation(
    frozen_primary_paths: pd.DataFrame,
    holdout_classifications: pd.DataFrame,
) -> pd.DataFrame:
    """Confirm exact frozen paths; never searches for a replacement path."""

    if frozen_primary_paths.empty:
        return _empty(HOLDOUT_PATH_COLUMNS)
    _require_columns(
        frozen_primary_paths,
        {"scenario", "path_id", "parameter", "direction", "meso", "macro"},
        "frozen primary paths",
    )
    _require_columns(
        holdout_classifications,
        {
            "scenario", "hypothesis_group_id", "root_source", "source", "target",
            "parameter", "direction", "primary_class",
        },
        "holdout classifications",
    )
    rows: list[dict[str, Any]] = []
    for (_, path_id), group in frozen_primary_paths.groupby(
        ["scenario", "path_id"], sort=True
    ):
        first = group.iloc[0]
        micro_column = "micro" if "micro" in group.columns else "source"
        definition = {
            key: str(first[key])
            for key in ("scenario", "path_id", "parameter", "direction", "meso", "macro")
        }
        definition["micro"] = str(first[micro_column])
        definition["source"] = definition["micro"]
        definition["hypothesis_group_id"] = f"macro_outcome_{definition['macro']}"
        subset = holdout_classifications[
            (holdout_classifications["scenario"].astype(str) == definition["scenario"])
            & (holdout_classifications["parameter"].astype(str) == definition["parameter"])
            & (holdout_classifications["direction"].astype(str) == definition["direction"])
            & (holdout_classifications["root_source"].astype(str) == definition["source"])
            & (
                holdout_classifications["hypothesis_group_id"].astype(str)
                == definition["hypothesis_group_id"]
            )
        ]
        if "method" in subset.columns:
            subset = subset[
                subset["method"].astype(str).isin(
                    {"full_method", "frozen_full_method"}
                )
            ]
        edge_classes: list[str] = []
        for source, target in (
            (definition["source"], definition["meso"]),
            (definition["meso"], definition["macro"]),
        ):
            attempts = subset[
                (subset["source"].astype(str) == source)
                & (subset["target"].astype(str) == target)
            ]["primary_class"].astype(str).tolist()
            if not attempts:
                edge_classes.append("inconclusive")
            elif "directionally_contradicted" in attempts:
                edge_classes.append("directionally_contradicted")
            elif "supported" in attempts:
                edge_classes.append("supported")
            elif "manipulation_failure" in attempts:
                edge_classes.append("manipulation_failure")
            elif "no_stable_downstream_effect" in attempts:
                edge_classes.append("no_stable_downstream_effect")
            else:
                edge_classes.append("inconclusive")
        confirmations = [_confirmation_class(value) for value in edge_classes]
        if "manipulation_failure" in confirmations:
            classification = "manipulation_failure"
        elif all(value == "confirmed" for value in confirmations):
            classification = "confirmed"
        elif "failed_confirmation" in confirmations:
            classification = "failed_confirmation"
        else:
            classification = "inconclusive"
        rows.append(
            {
                "evaluation_track": "holdout_confirmation",
                **definition,
                "micro_meso_confirmation": confirmations[0],
                "meso_macro_confirmation": confirmations[1],
                "classification": classification,
                "holdout_confirmed": classification == "confirmed",
                "frozen_definition_sha256": _canonical_hash(definition),
                "primary_result_unchanged": True,
            }
        )
    return pd.DataFrame(rows, columns=HOLDOUT_PATH_COLUMNS)


def _edge_pairs(graph: Any, scenario: str | None = None) -> set[tuple[str, str]]:
    if graph is None:
        return set()
    if isinstance(graph, Mapping):
        if scenario in graph:
            return _edge_pairs(graph[scenario], scenario)
        for key in ((scenario, "full_method"), (scenario, "frozen_full_method")):
            if key in graph:
                return _edge_pairs(graph[key], scenario)
        return set()
    if isinstance(graph, pd.DataFrame):
        frame = graph
        if scenario is not None and "scenario" in frame:
            frame = frame[frame["scenario"].astype(str) == str(scenario)]
        source_column = "source" if "source" in frame else "edge_source"
        target_column = "target" if "target" in frame else "edge_target"
        return set(zip(frame[source_column].astype(str), frame[target_column].astype(str)))
    pairs: set[tuple[str, str]] = set()
    for edge in graph:
        if isinstance(edge, tuple) and len(edge) >= 2:
            pairs.add((str(edge[0]), str(edge[1])))
        elif isinstance(edge, Mapping):
            pairs.add((str(edge["source"]), str(edge["target"])))
        else:
            pairs.add((str(edge.source), str(edge.target)))
    return pairs


def holdout_prospective_confirmation(
    frozen_predictions: Mapping[str, Any],
    frozen_primary_graphs: Any,
    holdout_effects: pd.DataFrame,
    representations: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Evaluate candidate_path_id-bound predictions without selecting replacements."""

    scenarios = frozen_predictions.get("scenarios", frozen_predictions)
    if not scenarios:
        return _empty(HOLDOUT_PROSPECTIVE_COLUMNS)
    _require_columns(
        holdout_effects,
        {"scenario", "parameter", "direction", "node_id", "significant", "effect_sign", "onset_time"},
        "holdout effects",
    )
    rows: list[dict[str, Any]] = []
    for scenario, predictions in sorted(scenarios.items()):
        primary_pairs = _edge_pairs(frozen_primary_graphs, str(scenario))
        paths = {
            str(path["path_id"]): path
            for path in (representations or {}).get(str(scenario), {}).get(
                "candidate_paths", []
            )
        }
        for prediction in predictions:
            path_id = str(prediction["candidate_path_id"])
            if path_id not in paths:
                raise ValueError(f"holdout prediction references unknown frozen path {path_id}")
            path = paths[path_id]
            indicators = [
                str(path["micro_indicator"]), str(path["meso_indicator"]),
                str(path["macro_indicator"]),
            ]
            expected_edges = list(zip(indicators, indicators[1:]))
            retained = [edge in primary_pairs for edge in expected_edges]
            selected = holdout_effects[
                (holdout_effects["scenario"].astype(str) == str(scenario))
                & (holdout_effects["parameter"].astype(str) == str(path["parameter"]))
                & (holdout_effects["direction"].astype(str) == str(path["intervention_direction"]))
                & (holdout_effects["node_id"].astype(str).isin(indicators))
            ]
            lookup = {str(row.node_id): row for row in selected.itertuples(index=False)}
            values = [lookup.get(node) for node in indicators]
            signs = [
                1 if expected == "increase" else -1
                for expected in (
                    path["expected_micro_response"], path["expected_meso_response"],
                    path["expected_macro_response"],
                )
            ]
            direction_matches = [
                item is not None and bool(item.significant) and int(item.effect_sign) == sign
                for item, sign in zip(values, signs)
            ]
            onsets = [float(item.onset_time) if item is not None else np.nan for item in values]
            ordered = bool(
                all(np.isfinite(value) and value >= 0 for value in onsets)
                and np.all(np.diff(np.asarray(onsets, dtype=float)) >= 0)
            )
            downstream_supported = all(
                item is not None and bool(item.significant) for item in values[1:]
            )
            if values[0] is None or not bool(values[0].significant):
                confirmation = "manipulation_failure"
            elif any(item is None for item in values) or not downstream_supported:
                confirmation = "inconclusive"
            elif not all(direction_matches) or not ordered or not all(retained):
                confirmation = "failed_confirmation"
            else:
                confirmation = "confirmed"
            prediction_hash = _canonical_hash(prediction)
            rows.append(
                {
                    "evaluation_track": "holdout_confirmation",
                    "scenario": scenario,
                    "prediction_id": prediction["prediction_id"],
                    "candidate_path_id": path_id,
                    "parameter": path["parameter"],
                    "intervention_direction": path["intervention_direction"],
                    "micro": indicators[0], "meso": indicators[1], "macro": indicators[2],
                    "classification": confirmation,
                    "path_temporally_qualified": all(retained),
                    "direction_supported": all(direction_matches),
                    "onset_order_supported": ordered,
                    "required_downstream_responses_supported": downstream_supported,
                    "prediction_sha256": prediction_hash,
                    "primary_result_unchanged": True,
                }
            )
    return pd.DataFrame(rows, columns=HOLDOUT_PROSPECTIVE_COLUMNS)


def holdout_mechanism_confirmation(
    frozen_primary_classifications: pd.DataFrame,
    holdout_classifications: pd.DataFrame,
    *,
    holdout_baseline_graphs: Any = None,
    holdout_mechanism_disabled_graphs: Any = None,
) -> pd.DataFrame:
    """Confirm exact primary-supported public edges on independent seeds."""

    if frozen_primary_classifications.empty:
        return _empty(HOLDOUT_MECHANISM_COLUMNS)
    required = {
        "scenario", "root_source", "source", "target", "parameter", "direction", "primary_class"
    }
    _require_columns(frozen_primary_classifications, required, "primary classifications")
    _require_columns(holdout_classifications, required, "holdout classifications")
    holdout = holdout_classifications.copy()
    if "method" in holdout.columns:
        holdout = holdout[
            holdout["method"].astype(str).isin(
                {"full_method", "frozen_full_method"}
            )
        ]
    primary = frozen_primary_classifications.copy()
    if "method" in primary:
        primary = primary[primary["method"].astype(str) == "full_method"]
    primary = primary[primary["primary_class"].astype(str) == "supported"]
    identity_columns = [
        "scenario", "root_source", "source", "target", "parameter", "direction"
    ]
    primary = primary.drop_duplicates(identity_columns)
    rows: list[dict[str, Any]] = []
    for item in primary.itertuples(index=False):
        definition = {column: str(getattr(item, column)) for column in identity_columns}
        subset = holdout
        for column, value in definition.items():
            subset = subset[subset[column].astype(str) == value]
        attempts = subset["primary_class"].astype(str).tolist()
        if "directionally_contradicted" in attempts:
            holdout_class = "directionally_contradicted"
        elif "supported" in attempts:
            holdout_class = "supported"
        elif "manipulation_failure" in attempts:
            holdout_class = "manipulation_failure"
        elif "no_stable_downstream_effect" in attempts:
            holdout_class = "no_stable_downstream_effect"
        else:
            holdout_class = "inconclusive"
        scenario = definition["scenario"]
        pair = (definition["source"], definition["target"])
        rows.append(
            {
                "evaluation_track": "holdout_confirmation",
                **definition,
                "primary_class": "supported",
                "holdout_class": holdout_class,
                "classification": _confirmation_class(holdout_class),
                "retained_in_holdout_baseline": (
                    pair in _edge_pairs(holdout_baseline_graphs, scenario)
                    if holdout_baseline_graphs is not None
                    else pd.NA
                ),
                "retained_in_holdout_mechanism_disabled": (
                    pair in _edge_pairs(holdout_mechanism_disabled_graphs, scenario)
                    if holdout_mechanism_disabled_graphs is not None
                    else pd.NA
                ),
                "frozen_definition_sha256": _canonical_hash(definition),
                "primary_result_unchanged": True,
            }
        )
    return pd.DataFrame(rows, columns=HOLDOUT_MECHANISM_COLUMNS)


def circular_shift_parent_series(
    trajectory_frames: Sequence[pd.DataFrame],
    candidates: Sequence[Mapping[str, Any]],
    *,
    scenario: str,
    repetition: int,
    maximum_lag: int,
    minimum_shift: int,
    maximum_shift: int,
    master_seed: int,
) -> tuple[list[pd.DataFrame], pd.DataFrame]:
    """Return shifted copies; input trajectory frames are never mutated."""

    if int(minimum_shift) <= int(maximum_lag):
        raise ValueError("minimum temporal-control shift must exceed maximum lag")
    if int(maximum_shift) < int(minimum_shift):
        raise ValueError("maximum shift must be at least minimum shift")
    parents = sorted({str(edge["source"]) for edge in candidates})
    copies: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for index, frame in enumerate(trajectory_frames):
        shifted = frame.copy(deep=True)
        _require_columns(frame, parents, "negative-control trajectory")
        if len(frame) <= int(maximum_shift) + int(maximum_lag):
            raise ValueError(
                "trajectory is too short for the requested circular shift range"
            )
        trajectory_seed = (
            int(frame["seed"].iloc[0]) if "seed" in frame and len(frame) else index
        )
        for parent in parents:
            rng = np.random.default_rng(
                stable_seed(
                    master_seed,
                    "temporal_negative_control",
                    scenario,
                    repetition,
                    trajectory_seed,
                    parent,
                )
            )
            shift = int(rng.integers(int(minimum_shift), int(maximum_shift) + 1))
            shifted[parent] = np.roll(frame[parent].to_numpy(copy=True), shift)
            records.append(
                {
                    "scenario": scenario,
                    "repetition": int(repetition),
                    "trajectory_index": index,
                    "trajectory_seed": trajectory_seed,
                    "parent": parent,
                    "shift": shift,
                    "maximum_lag": int(maximum_lag),
                }
            )
        copies.append(shifted)
    return copies, pd.DataFrame(records, columns=SHIFT_RECORD_COLUMNS)


def temporal_negative_control(
    scenario: str,
    trajectory_frames: Sequence[pd.DataFrame],
    representation: Mapping[str, Any],
    temporal_config: Mapping[str, Any],
    control_config: Mapping[str, Any],
    *,
    master_seed: int,
    workers: int = 1,
    primary_retained_edges: Sequence[TemporalEdge] | None = None,
) -> pd.DataFrame:
    """Run deterministically shifted data through the unchanged Stage 2 core."""

    candidates = representation_candidates(dict(representation))
    repetitions = int(control_config["repetitions"])
    maximum_lag = int(temporal_config["maximum_lag"])
    rows: list[dict[str, Any]] = []
    for repetition in range(repetitions):
        shifted, shift_records = circular_shift_parent_series(
            trajectory_frames,
            candidates,
            scenario=scenario,
            repetition=repetition,
            maximum_lag=maximum_lag,
            minimum_shift=int(control_config["minimum_shift"]),
            maximum_shift=int(control_config["maximum_shift"]),
            master_seed=master_seed,
        )
        graph, _ = discover_bootstrap_graph(
            shifted,
            candidates,
            maximum_lag,
            float(temporal_config["parent_alpha"]),
            float(temporal_config["fdr_alpha"]),
            int(temporal_config["bootstrap_repetitions"]),
            float(temporal_config["support_threshold"]),
            master_seed,
            f"{scenario}:temporal-negative:{repetition}",
            workers,
        )
        supports = [float(edge.support) for edge in graph if np.isfinite(edge.support)]
        retained_group_edges = {
            (edge.source, edge.target, edge.hypothesis_group_id) for edge in graph
        }
        frozen_paths = list(representation.get("candidate_paths", []))
        qualified_paths = sum(
            (
                str(path["micro_indicator"]), str(path["meso_indicator"]),
                f"macro_outcome_{path['macro_indicator']}",
            ) in retained_group_edges
            and (
                str(path["meso_indicator"]), str(path["macro_indicator"]),
                f"macro_outcome_{path['macro_indicator']}",
            ) in retained_group_edges
            for path in frozen_paths
        )
        primary_count = (
            len(primary_retained_edges) if primary_retained_edges is not None else None
        )
        retained = len(graph)
        rows.append(
            {
                "evaluation_track": "falsification_control",
                "scenario": scenario,
                "repetition": repetition,
                "candidate_edge_count": len(candidates),
                "retained_edge_count": retained,
                "qualification_rate": retained / max(len(candidates), 1),
                "candidate_path_count": len(frozen_paths),
                "qualified_path_count": qualified_paths,
                "path_qualification_rate": qualified_paths / max(len(frozen_paths), 1),
                "stability": float(np.mean(supports)) if supports else float("nan"),
                "primary_retained_edge_count": primary_count if primary_count is not None else np.nan,
                "retained_edge_difference_from_primary": (
                    retained - primary_count if primary_count is not None else np.nan
                ),
                "retained_edge_ratio_to_primary": (
                    retained / primary_count
                    if primary_count is not None and primary_count > 0
                    else np.nan
                ),
                "minimum_applied_shift": int(shift_records["shift"].min()),
                "maximum_applied_shift": int(shift_records["shift"].max()),
                "maximum_lag": maximum_lag,
                "control_only": True,
            }
        )
    return pd.DataFrame(rows, columns=NEGATIVE_CONTROL_COLUMNS)


def _metric_lookup(
    frame: pd.DataFrame, *, value_column: str
) -> dict[tuple[str, str], float]:
    if frame.empty:
        return {}
    _require_columns(frame, {"scenario", "node_id", value_column}, "mechanism metric")
    return {
        (str(row.scenario), str(row.node_id)): float(getattr(row, value_column))
        for row in frame.itertuples(index=False)
    }


def path_mechanism_attenuation(
    frozen_primary_paths: pd.DataFrame,
    baseline_metrics: pd.DataFrame,
    disabled_metrics: pd.DataFrame,
    *,
    baseline_graphs: Any,
    disabled_graphs: Any,
    mechanism_variants: Mapping[str, str],
    value_column: str = "effect",
) -> pd.DataFrame:
    """Measure attenuation only for public Full Discovery path identities."""

    if frozen_primary_paths.empty:
        return _empty(PATH_ATTENUATION_COLUMNS)
    _require_columns(
        frozen_primary_paths,
        {"scenario", "path_id", "source", "meso", "macro"},
        "frozen primary paths",
    )
    baseline = _metric_lookup(baseline_metrics, value_column=value_column)
    disabled = _metric_lookup(disabled_metrics, value_column=value_column)
    rows: list[dict[str, Any]] = []
    for (_, path_id), group in frozen_primary_paths.groupby(
        ["scenario", "path_id"], sort=True
    ):
        first = group.iloc[0]
        scenario = str(first["scenario"])
        nodes = [
            (str(first["source"]), "micro", None),
            (str(first["meso"]), "meso", str(first["source"])),
            (str(first["macro"]), "macro", str(first["meso"])),
        ]
        baseline_pairs = _edge_pairs(baseline_graphs, scenario)
        disabled_pairs = _edge_pairs(disabled_graphs, scenario)
        for node_id, scale, parent in nodes:
            baseline_effect = baseline.get((scenario, node_id), float("nan"))
            disabled_effect = disabled.get((scenario, node_id), float("nan"))
            if np.isfinite(baseline_effect) and abs(baseline_effect) > 1e-12 and np.isfinite(disabled_effect):
                ratio = abs(disabled_effect) / abs(baseline_effect)
                attenuation = 1.0 - ratio
            else:
                ratio = attenuation = float("nan")
            incoming = (parent, node_id) if parent is not None else None
            rows.append(
                {
                    "evaluation_track": "falsification_control",
                    "scenario": scenario,
                    "path_id": str(path_id),
                    "mechanism_variant": str(mechanism_variants.get(scenario, "")),
                    "node_id": node_id,
                    "scale": scale,
                    "baseline_effect": baseline_effect,
                    "disabled_effect": disabled_effect,
                    "disabled_to_baseline_ratio": ratio,
                    "attenuation_ratio": attenuation,
                    "temporal_edge_retained_baseline": (
                        incoming in baseline_pairs if incoming is not None else pd.NA
                    ),
                    "temporal_edge_retained_disabled": (
                        incoming in disabled_pairs if incoming is not None else pd.NA
                    ),
                    "uses_hidden_truth": False,
                }
            )
    return pd.DataFrame(rows, columns=PATH_ATTENUATION_COLUMNS)
