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
    classify_candidate_paths,
    classify_edge_interventions,
    estimate_all_effects,
    qualify_candidate_paths,
    select_representative_paths,
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
    path_rows: list[dict[str, Any]] = []
    qualification_path = analysis / "path_temporal_qualification.csv"
    path_classification_path = analysis / "path_intervention_classification.csv"
    qualified_ids: set[tuple[str, str]] = set()
    supported_ids: set[tuple[str, str]] = set()
    if qualification_path.is_file():
        qualified = pd.read_csv(qualification_path)
        qualified_ids = {
            (str(row.scenario), str(row.path_id))
            for row in qualified.itertuples(index=False)
            if bool(row.path_temporally_qualified)
        }
    if path_classification_path.is_file():
        primary = pd.read_csv(path_classification_path)
        supported_ids = {
            (str(row.scenario), str(row.path_id))
            for row in primary.itertuples(index=False)
            if str(row.path_classification) == "supported"
        }
    summary_lookup = {
        (str(row.scenario), str(row.parameter), str(row.node_id)): row
        for row in summary.itertuples(index=False)
    }
    for scenario, representation in sorted(representations.items()):
        for path in representation.get("candidate_paths", []):
            key = (scenario, str(path["path_id"]))
            if key not in qualified_ids and key not in supported_ids:
                continue
            nodes = [
                str(path["micro_indicator"]), str(path["meso_indicator"]),
                str(path["macro_indicator"]),
            ]
            trends = [
                summary_lookup.get((scenario, str(path["parameter"]), node))
                for node in nodes
            ]
            path_rows.append(
                {
                    "scenario": scenario,
                    "path_id": str(path["path_id"]),
                    "parameter": str(path["parameter"]),
                    "temporally_qualified": key in qualified_ids,
                    "primary_supported": key in supported_ids,
                    "micro_dose_trend": getattr(trends[0], "dose_response_slope", np.nan),
                    "meso_dose_trend": getattr(trends[1], "dose_response_slope", np.nan),
                    "macro_dose_trend": getattr(trends[2], "dose_response_slope", np.nan),
                    "primary_classification_changed": False,
                }
            )
    path_dose = pd.DataFrame(
        path_rows,
        columns=[
            "scenario", "path_id", "parameter", "temporally_qualified",
            "primary_supported", "micro_dose_trend", "meso_dose_trend",
            "macro_dose_trend", "primary_classification_changed",
        ],
    )
    _atomic_csv(path_dose, analysis / "path_dose_response_summary.csv")
    return {
        "evaluation_track": "secondary_dose_response",
        "effect_rows": len(effects),
        "summary_rows": len(summary),
        "path_summary_rows": len(path_dose),
        "primary_classification_changed": False,
    }


def _validated_primary_paths(run_root: Path) -> pd.DataFrame:
    paths = pd.read_csv(
        run_root / "analysis" / "path_intervention_classification.csv"
    )
    supported = paths[
        paths["path_classification"].astype(str) == "supported"
    ].copy()
    supported["source"] = supported["micro"]
    return supported


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
    temporal_qualifications: list[pd.DataFrame] = []
    for scenario, representation in sorted(representations.items()):
        graph = primary_graphs[(scenario, "full_method")]
        classified = classify_edge_interventions(
            scenario,
            graph,
            effects,
            representation,
            int(config["intervention"]["lag_tolerance"]),
        )
        if classified.empty:
            classified = pd.DataFrame(columns=CLASSIFICATION_COLUMNS)
        classified.insert(0, "method", "frozen_full_method")
        classified.insert(0, "evaluation_track", "holdout_confirmation")
        classifications.append(classified)
        temporal_qualifications.append(
            qualify_candidate_paths(scenario, graph, representation)
        )
    holdout_classifications = pd.concat(classifications, ignore_index=True)
    holdout_temporal_qualification = pd.concat(
        temporal_qualifications, ignore_index=True
    )
    holdout_path_classification = classify_candidate_paths(
        holdout_temporal_qualification,
        holdout_classifications,
        representations,
    )
    frozen_paths = _validated_primary_paths(run_root)
    path_confirmation = holdout_path_confirmation(
        frozen_paths, holdout_path_classification
    )
    frozen_predictions = json.loads(
        (run_root / "representation" / "prospective_predictions.json").read_text(
            encoding="utf-8"
        )
    )
    prospective_confirmation = holdout_prospective_confirmation(
        frozen_predictions, primary_graphs, effects, representations
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
    primary_paths = pd.read_csv(
        analysis / "path_intervention_classification.csv"
    )
    representative = select_representative_paths(primary_paths, path_confirmation)
    (analysis / "representative_path_selection.json").write_text(
        json.dumps(representative, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporal_paths = pd.read_csv(analysis / "path_temporal_qualification.csv")
    prospective_primary = pd.read_csv(analysis / "prospective_validation.csv")
    funnel_rows: list[dict[str, Any]] = []
    for scenario, representation in sorted(representations.items()):
        candidate_count = len(representation.get("candidate_paths", []))
        temporal_subset = temporal_paths[
            temporal_paths["scenario"].astype(str) == scenario
        ]
        primary_subset = primary_paths[
            primary_paths["scenario"].astype(str) == scenario
        ]
        holdout_subset = path_confirmation[
            path_confirmation["scenario"].astype(str) == scenario
        ]
        prospective_subset = prospective_primary[
            prospective_primary["scenario"].astype(str) == scenario
        ]
        temporal_count = int(
            temporal_subset["path_temporally_qualified"].astype(bool).sum()
        )
        supported_count = int(
            (primary_subset["path_classification"].astype(str) == "supported").sum()
        )
        holdout_count = int(
            holdout_subset.get(
                "holdout_confirmed", pd.Series(False, index=holdout_subset.index)
            ).astype(bool).sum()
        )
        funnel_rows.append(
            {
                "scenario": scenario,
                "candidate_path_count": candidate_count,
                "temporally_qualified_path_count": temporal_count,
                "temporally_qualified_path_rate": temporal_count / max(candidate_count, 1),
                "intervention_supported_path_count": supported_count,
                "intervention_supported_path_rate": supported_count / max(candidate_count, 1),
                "contradicted_path_count": int((primary_subset["path_classification"] == "contradicted").sum()),
                "inconclusive_path_count": int((primary_subset["path_classification"] == "inconclusive").sum()),
                "manipulation_failure_path_count": int((primary_subset["path_classification"] == "manipulation_failure").sum()),
                "holdout_confirmed_path_count": holdout_count,
                "holdout_confirmation_rate": holdout_count / max(supported_count, 1),
                "prospective_supported_path_count": int((prospective_subset["classification"] == "supported").sum()),
            }
        )
    _atomic_csv(pd.DataFrame(funnel_rows), analysis / "path_funnel_summary.csv")
    verify_primary_contract(run_root)
    return {
        "evaluation_track": "holdout_confirmation",
        "effect_rows": len(effects),
        "classification_rows": len(holdout_classifications),
        "path_confirmation_rows": len(path_confirmation),
        "holdout_path_classification_rows": len(holdout_path_classification),
        "prospective_confirmation_rows": len(prospective_confirmation),
        "mechanism_confirmation_rows": len(mechanism_confirmation),
        "primary_result_unchanged": True,
        "stage3_method_version": "intervention_path_classification_v2",
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
    _atomic_csv(
        negative[
            [
                "evaluation_track", "scenario", "repetition",
                "candidate_path_count", "qualified_path_count",
                "path_qualification_rate", "control_only",
            ]
        ],
        analysis / "path_temporal_negative_control.csv",
    )
    _atomic_csv(attenuation, analysis / "path_mechanism_attenuation.csv")
    return {
        "evaluation_track": "falsification_control",
        "negative_control_rows": len(negative),
        "path_negative_control_rows": len(negative),
        "path_attenuation_rows": len(attenuation),
        "primary_data_overwritten": False,
        "hidden_truth_used_for_path_attenuation": False,
    }
