"""High-level runners for the frozen final-protocol secondary tracks."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from .final_protocol import (
    dose_response_effects,
    dose_response_summary,
    holdout_mechanism_confirmation,
    holdout_path_confirmation,
    holdout_prospective_confirmation,
    path_mechanism_attenuation,
    temporal_negative_control,
)
from .interventions import (
    CLASSIFICATION_COLUMNS,
    classify_edge_interventions,
    eligible_propagation_path_ids,
    estimate_all_effects,
)
from .primary_freeze import verify_primary_contract
from .simulation import trajectories
from .temporal import (
    discover_bootstrap_graph,
    load_graph_records,
    representation_candidates,
)


ProgressCallback = Callable[[str, int, int, str], None]


def _atomic_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp.csv")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _relevant_dose_nodes(
    representations: dict[str, dict[str, Any]],
) -> set[tuple[str, str, str]]:
    relevant: set[tuple[str, str, str]] = set()
    for scenario, representation in representations.items():
        indicators = {item["id"]: item for item in representation["indicators"]}
        edges = {
            str(edge["source"]): {
                str(item["target"])
                for item in representation["candidate_edges"]
                if str(item["source"]) == str(edge["source"])
            }
            for edge in representation["candidate_edges"]
        }
        for node_id, indicator in indicators.items():
            if indicator["scale"] != "micro":
                continue
            parameters = {
                str(item["parameter"])
                for item in indicator.get("parameter_associations", [])
                if item.get("relationship") == "direct"
            }
            for parameter in parameters:
                relevant.add((scenario, parameter, str(node_id)))
                for meso in edges.get(str(node_id), set()):
                    if indicators.get(meso, {}).get("scale") != "meso":
                        continue
                    relevant.add((scenario, parameter, meso))
                    for macro in edges.get(meso, set()):
                        if indicators.get(macro, {}).get("scale") == "macro":
                            relevant.add((scenario, parameter, macro))
    return relevant


def run_dose_response_analysis(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    complete_dataset: pd.DataFrame,
) -> dict[str, Any]:
    effects = dose_response_effects(
        complete_dataset,
        representations,
        config["scenarios"],
        config["intervention"],
        master_seed=int(config["master_seed"]),
    )
    relevant = _relevant_dose_nodes(representations)
    if not effects.empty:
        keep = [
            (str(row.scenario), str(row.parameter), str(row.node_id)) in relevant
            for row in effects.itertuples()
        ]
        seed_effects = effects.attrs.get("paired_seed_effects")
        effects = effects.loc[keep].reset_index(drop=True)
        if isinstance(seed_effects, pd.DataFrame):
            seed_keep = [
                (str(row.scenario), str(row.parameter), str(row.node_id)) in relevant
                for row in seed_effects.itertuples()
            ]
            seed_effects = seed_effects.loc[seed_keep].reset_index(drop=True)
        else:
            seed_effects = None
    else:
        seed_effects = effects.attrs.get("paired_seed_effects")
    summary = dose_response_summary(
        effects,
        bootstrap_repetitions=int(config["intervention"]["bootstrap_repetitions"]),
        confidence_level=float(config["intervention"]["confidence_level"]),
        master_seed=int(config["master_seed"]),
        paired_seed_effects=seed_effects,
    )
    analysis = run_root / "analysis"
    _atomic_csv(effects, analysis / "dose_response_effects.csv")
    _atomic_csv(summary, analysis / "dose_response_summary.csv")
    return {
        "evaluation_track": "secondary_dose_response",
        "effect_rows": len(effects),
        "summary_rows": len(summary),
        "primary_classification_changed": False,
    }


def _validated_primary_paths(run_root: Path) -> pd.DataFrame:
    timing = pd.read_csv(run_root / "analysis" / "path_timing_summary.csv")
    classifications = pd.read_csv(
        run_root / "analysis" / "intervention_classifications.csv"
    )
    if timing.empty:
        return timing
    identifiers = eligible_propagation_path_ids(timing, classifications)
    return timing[timing["path_id"].astype(str).isin(identifiers)].copy()


def _attach_holdout_mechanism_metrics(
    frame: pd.DataFrame,
    dataset: pd.DataFrame,
    config: dict[str, Any],
) -> pd.DataFrame:
    columns = [
        "mechanism_variant",
        "source_baseline_mean",
        "source_disabled_mean",
        "source_disabled_minus_baseline",
        "target_baseline_mean",
        "target_disabled_mean",
        "target_disabled_minus_baseline",
        "mechanism_disabled_evaluated",
    ]
    if frame.empty:
        for column in columns:
            frame[column] = pd.Series(dtype="bool" if column.endswith("evaluated") else "object")
        return frame
    evaluation_start = int(config["intervention"]["evaluation_start"])
    rows: list[dict[str, Any]] = []
    for item in frame.itertuples(index=False):
        scenario = str(item.scenario)
        subset = dataset[
            (dataset["scenario"].astype(str) == scenario)
            & (dataset["time"].astype(int) >= evaluation_start)
        ]
        baseline = subset[subset["condition"].astype(str) == "baseline"]
        disabled = subset[
            subset["condition"].astype(str) == "mechanism_disabled"
        ]
        source_baseline = float(baseline[str(item.source)].mean())
        source_disabled = float(disabled[str(item.source)].mean())
        target_baseline = float(baseline[str(item.target)].mean())
        target_disabled = float(disabled[str(item.target)].mean())
        rows.append(
            {
                "mechanism_variant": str(
                    config["scenarios"][scenario]["mechanism_variant"]
                ),
                "source_baseline_mean": source_baseline,
                "source_disabled_mean": source_disabled,
                "source_disabled_minus_baseline": source_disabled - source_baseline,
                "target_baseline_mean": target_baseline,
                "target_disabled_mean": target_disabled,
                "target_disabled_minus_baseline": target_disabled - target_baseline,
                "mechanism_disabled_evaluated": bool(len(baseline) and len(disabled)),
            }
        )
    metrics = pd.DataFrame(rows, columns=columns)
    return pd.concat([frame.reset_index(drop=True), metrics], axis=1)


def run_holdout_confirmation_analysis(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    holdout_dataset: pd.DataFrame,
    workers: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    verify_primary_contract(run_root)
    effects, curves = estimate_all_effects(
        holdout_dataset,
        config,
        representations,
        workers,
        paired=True,
        progress_callback=progress_callback,
    )
    primary_graphs = load_graph_records(run_root / "analysis" / "main_graphs.jsonl")
    classifications: list[pd.DataFrame] = []
    for scenario, representation in sorted(representations.items()):
        classified = classify_edge_interventions(
            scenario,
            primary_graphs[(scenario, "full_method")],
            effects,
            representation,
            int(config["intervention"]["lag_tolerance"]),
        )
        if classified.empty:
            classified = pd.DataFrame(columns=CLASSIFICATION_COLUMNS)
        classified.insert(0, "method", "frozen_full_method")
        classified.insert(0, "evaluation_track", "holdout_confirmation")
        classifications.append(classified)
    holdout_classifications = pd.concat(classifications, ignore_index=True)
    frozen_paths = _validated_primary_paths(run_root)
    path_confirmation = holdout_path_confirmation(
        frozen_paths, holdout_classifications
    )
    frozen_predictions = json.loads(
        (run_root / "representation" / "prospective_predictions.json").read_text(
            encoding="utf-8"
        )
    )
    prospective_confirmation = holdout_prospective_confirmation(
        frozen_predictions, primary_graphs, effects
    )
    primary_classifications = pd.read_csv(
        run_root / "analysis" / "intervention_classifications.csv"
    )
    mechanism_confirmation = holdout_mechanism_confirmation(
        primary_classifications, holdout_classifications
    )
    mechanism_confirmation = _attach_holdout_mechanism_metrics(
        mechanism_confirmation, holdout_dataset, config
    )
    effects.insert(0, "evaluation_track", "holdout_confirmation")
    curves.insert(0, "evaluation_track", "holdout_confirmation")
    analysis = run_root / "analysis"
    effects.to_parquet(analysis / "holdout_paired_effects.parquet", index=False)
    curves.to_parquet(analysis / "holdout_effect_curves.parquet", index=False)
    _atomic_csv(
        holdout_classifications,
        analysis / "holdout_intervention_classifications.csv",
    )
    _atomic_csv(path_confirmation, analysis / "holdout_path_confirmation.csv")
    _atomic_csv(
        prospective_confirmation,
        analysis / "holdout_prospective_confirmation.csv",
    )
    _atomic_csv(
        mechanism_confirmation,
        analysis / "holdout_mechanism_confirmation.csv",
    )
    verify_primary_contract(run_root)
    return {
        "evaluation_track": "holdout_confirmation",
        "effect_rows": len(effects),
        "classification_rows": len(holdout_classifications),
        "path_confirmation_rows": len(path_confirmation),
        "prospective_confirmation_rows": len(prospective_confirmation),
        "mechanism_confirmation_rows": len(mechanism_confirmation),
        "primary_result_unchanged": True,
    }


def _node_level_means(
    dataset: pd.DataFrame,
    representations: dict[str, dict[str, Any]],
    condition: str,
    evaluation_start: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for scenario, representation in sorted(representations.items()):
        subset = dataset[
            (dataset["scenario"].astype(str) == scenario)
            & (dataset["condition"].astype(str) == condition)
            & (dataset["time"].astype(int) >= evaluation_start)
        ]
        for indicator in representation["indicators"]:
            node_id = str(indicator["id"])
            rows.append(
                {
                    "scenario": scenario,
                    "node_id": node_id,
                    "effect": float(subset[node_id].mean()),
                }
            )
    return pd.DataFrame(rows, columns=["scenario", "node_id", "effect"])


def run_falsification_analyses(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    baseline_dataset: pd.DataFrame,
    complete_dataset: pd.DataFrame,
    workers: int,
    progress_callback: ProgressCallback | None = None,
) -> dict[str, Any]:
    primary_graphs = load_graph_records(run_root / "analysis" / "main_graphs.jsonl")
    negative_frames: list[pd.DataFrame] = []
    disabled_graphs: dict[str, list[Any]] = {}
    for scenario, representation in sorted(representations.items()):
        baseline_by_seed = trajectories(baseline_dataset, scenario)
        frames = [
            baseline_by_seed[seed].assign(seed=int(seed))
            for seed in sorted(baseline_by_seed)
        ]
        negative_frames.append(
            temporal_negative_control(
                scenario,
                frames,
                representation,
                config["temporal"],
                config["temporal_negative_control"],
                master_seed=int(config["master_seed"]),
                workers=workers,
                primary_retained_edges=primary_graphs[(scenario, "full_method")],
            )
        )
        if progress_callback:
            progress_callback(
                "Negative temporal controls",
                len(negative_frames),
                len(representations),
                scenario,
            )
        disabled_by_seed = trajectories(
            complete_dataset, scenario, "mechanism_disabled"
        )
        disabled_frames = [
            disabled_by_seed[seed] for seed in sorted(disabled_by_seed)
        ]
        graph, _ = discover_bootstrap_graph(
            disabled_frames,
            representation_candidates(representation),
            int(config["temporal"]["maximum_lag"]),
            float(config["temporal"]["parent_alpha"]),
            float(config["temporal"]["fdr_alpha"]),
            int(config["temporal"]["bootstrap_repetitions"]),
            float(config["temporal"]["support_threshold"]),
            int(config["master_seed"]),
            f"{scenario}:mechanism-disabled:path-attenuation",
            workers,
        )
        disabled_graphs[scenario] = graph
    negative = pd.concat(negative_frames, ignore_index=True)
    paths = _validated_primary_paths(run_root)
    baseline_metrics = _node_level_means(
        complete_dataset,
        representations,
        "baseline",
        int(config["intervention"]["evaluation_start"]),
    )
    disabled_metrics = _node_level_means(
        complete_dataset,
        representations,
        "mechanism_disabled",
        int(config["intervention"]["evaluation_start"]),
    )
    attenuation = path_mechanism_attenuation(
        paths,
        baseline_metrics,
        disabled_metrics,
        baseline_graphs=primary_graphs,
        disabled_graphs=disabled_graphs,
        mechanism_variants={
            scenario: str(spec["mechanism_variant"])
            for scenario, spec in config["scenarios"].items()
        },
        value_column="effect",
    )
    analysis = run_root / "analysis"
    _atomic_csv(negative, analysis / "temporal_negative_control.csv")
    _atomic_csv(attenuation, analysis / "path_mechanism_attenuation.csv")
    return {
        "evaluation_track": "falsification_control",
        "negative_control_rows": len(negative),
        "path_attenuation_rows": len(attenuation),
        "primary_data_overwritten": False,
        "hidden_truth_used_for_path_attenuation": False,
    }
