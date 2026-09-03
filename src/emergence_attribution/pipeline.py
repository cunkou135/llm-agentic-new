"""Checkpointed orchestration for immutable formal and isolated dev runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from .controlled import (
    compile_controlled_dataset,
    run_controlled_intervention_stage,
    run_controlled_temporal_stage,
)
from .evaluation import evaluate_main_graphs, update_intervention_metrics
from .exporting import create_visualization_bundle, generate_tables, integrate_evidence
from .interventions import (
    classify_candidate_paths,
    qualify_candidate_paths,
    run_intervention_stage,
    select_representative_paths,
)
from .llm_client import LLMResponse
from .progress import ProgressReporter
from .primary_freeze import freeze_primary_contract, verify_primary_contract
from .protocol_stages import (
    run_dose_response_analysis,
    run_falsification_analyses,
    run_holdout_confirmation_analysis,
)
from .prospective import validate_prospective_predictions
from .provenance import RunContractError, RunManager
from .rendering import render_all_figures
from .robustness import run_robustness_stage
from .semantic import (
    freeze_indicator_stage,
    load_frozen_representations,
    run_indicator_generation_stage,
    run_path_generation_stage,
)
from .simulation import (
    HOLDOUT_PARTITION,
    compile_indicator_dataset,
    run_baseline_simulation_stage,
    run_holdout_baseline_simulation_stage,
    run_holdout_intervention_simulation_stage,
    run_intervention_simulation_stage,
    verify_simulation_manifest,
)
from .temporal import load_graph_records, run_temporal_stage


STAGE_ORDER = [
    "indicator_generation",
    "indicator_freeze",
    "path_generation",
    "semantic_freeze",
    "baseline_simulation",
    "temporal",
    "path_temporal_qualification",
    "intervention_simulation",
    "intervention",
    "path_intervention_classification",
    "prospective",
    "primary_freeze",
    "dose_response",
    "holdout_simulation",
    "holdout_confirmation",
    "temporal_negative_control",
    "robustness",
    "export",
    "render",
]


def load_experiment_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "master_seed", "random_seeds", "representation", "temporal",
        "intervention", "robustness", "render", "scenarios",
        "confirmation_seeds", "dose_response", "semantic_replication", "path_replication",
        "temporal_negative_control",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"experiment configuration fields are missing: {missing}")
    representation = config["representation"]
    if int(representation["minimum_candidate_paths"]) != 16:
        raise ValueError("minimum candidate path count must equal 16")
    if int(representation["maximum_candidate_paths"]) != 24:
        raise ValueError("maximum candidate path count must equal 24")
    if int(representation["maximum_candidate_paths"]) < int(
        representation["minimum_candidate_paths"]
    ):
        raise ValueError("maximum candidate path count is below the minimum")
    if bool(config.get("formal_run", True)):
        intervention = config["intervention"]
        frozen_intervention = {
            "bootstrap_repetitions": 500,
            "confidence_level": 0.95,
            "onset_detection_start": 0,
            "minimum_standardised_effect": 0.10,
            "onset_consecutive_steps": 4,
            "evaluation_start": 15,
            "terminal_window": 24,
            "lag_tolerance": 2,
        }
        for key, expected in frozen_intervention.items():
            if intervention.get(key) != expected:
                raise ValueError(f"formal intervention setting {key} must equal {expected}")
        frozen_temporal = {
            "maximum_lag": 5,
            "parent_alpha": 0.10,
            "fdr_alpha": 0.05,
            "bootstrap_repetitions": 100,
            "support_threshold": 0.65,
        }
        for key, expected in frozen_temporal.items():
            if config["temporal"].get(key) != expected:
                raise ValueError(f"formal temporal setting {key} must equal {expected}")
        frozen_representation = {
            "independent_generations": 3,
            "maximum_repair_rounds": 3,
            "minimum_candidate_paths": 16,
            "maximum_candidate_paths": 24,
            "prospective_prediction_count": 6,
        }
        for key, expected in frozen_representation.items():
            if representation.get(key) != expected:
                raise ValueError(
                    f"formal representation capacity {key} must equal {expected}"
                )
        if representation.get("budget") != {"micro": 16, "meso": 8, "macro": 4}:
            raise ValueError(
                "formal representation capacity must be 16 Micro, 8 Meso, and 4 Macro"
            )
        if len(config["random_seeds"]) != 24:
            raise ValueError("formal runs require exactly 24 random seeds")
        if config["random_seeds"] != list(range(3101, 3125)):
            raise ValueError("formal primary seeds must be exactly 3101 through 3124")
        if config["confirmation_seeds"] != list(range(4101, 4113)):
            raise ValueError("formal holdout seeds must be exactly 4101 through 4112")
        if set(config["random_seeds"]) & set(config["confirmation_seeds"]):
            raise ValueError("primary and holdout seed pools must be disjoint")
        if config["dose_response"].get("enabled") is not True:
            raise ValueError("formal dose-response experiment must be enabled")
        if config["dose_response"].get("levels") != [
            "minus", "mid_minus", "baseline", "mid_plus", "plus"
        ]:
            raise ValueError("formal dose-response levels are frozen at five points")
        if config["dose_response"].get("primary_support_levels") != ["minus", "plus"]:
            raise ValueError("primary support must remain restricted to minus and plus")
        replication = config["semantic_replication"]
        if replication.get("selection_generations") != 3:
            raise ValueError("formal semantic selection requires exactly three generations")
        if replication.get("replication_only_generations") != 3:
            raise ValueError("formal semantic replication requires exactly three held-out generations")
        path_replication = config["path_replication"]
        if path_replication.get("primary_generations") != 1:
            raise ValueError("formal path protocol requires one primary generation")
        if path_replication.get("replication_only_generations") != 2:
            raise ValueError("formal path replication requires two replication-only generations")
        negative = config["temporal_negative_control"]
        if (
            negative.get("enabled") is not True
            or negative.get("repetitions") != 20
            or negative.get("minimum_shift") != 20
            or negative.get("maximum_shift") != 60
        ):
            raise ValueError("formal temporal negative-control settings are frozen")
        if int(negative["minimum_shift"]) <= int(config["temporal"]["maximum_lag"]):
            raise ValueError("negative-control shifts must exceed the maximum tested lag")
        if len(config["scenarios"]) != 2 or any(
            len(spec.get("interventions", {})) != 3
            for spec in config["scenarios"].values()
        ):
            raise ValueError("formal task contract requires two scenarios with three parameters each")
        primary_count = len(config["scenarios"]) * len(config["random_seeds"]) * (1 + 3 * 4 + 1)
        holdout_count = len(config["scenarios"]) * len(config["confirmation_seeds"]) * (1 + 3 * 2 + 1)
        if (primary_count, holdout_count, primary_count + holdout_count) != (672, 192, 864):
            raise ValueError("formal simulator task matrix must equal 672 primary + 192 holdout = 864")
    else:
        if set(config["random_seeds"]) & set(config["confirmation_seeds"]):
            raise ValueError("primary and holdout seed pools must be disjoint")
        if int(config["temporal_negative_control"]["minimum_shift"]) <= int(
            config["temporal"]["maximum_lag"]
        ):
            raise ValueError("negative-control shifts must exceed the maximum tested lag")
    return config


def _require_stages(manager: RunManager, stages: Iterable[str]) -> None:
    missing = [stage for stage in stages if not manager.stage_complete(stage)]
    if missing:
        raise RunContractError(f"required stages are not complete: {missing}")


def _files(root: Path, patterns: Iterable[str]) -> list[Path]:
    result: list[Path] = []
    for pattern in patterns:
        result.extend(path for path in root.glob(pattern) if path.is_file())
    return sorted(set(result))


def _progress_callback(reporter: ProgressReporter):
    def callback(label: str, completed: int, total: int, detail: str = "") -> None:
        reporter.update(label, completed, total, detail)

    return callback


def run_stage(
    stage: str,
    manager: RunManager,
    workers: int,
    reporter: ProgressReporter,
    prompt_template_path: Path,
    llm_config_path: Path,
    plot_repo: Path | None,
    completion_provider: Callable[[str, int], Callable[[str, str], LLMResponse]] | None = None,
) -> None:
    run_root, config = manager.run_root, manager.config
    phase_for_stage = {
        "baseline_simulation": "baseline",
        "intervention_simulation": "intervention",
    }
    if manager.stage_complete(stage):
        if stage in phase_for_stage:
            verify_simulation_manifest(run_root, phase_for_stage[stage])
        elif stage == "holdout_simulation":
            verify_primary_contract(run_root)
            verify_simulation_manifest(
                run_root, "baseline", dataset_partition=HOLDOUT_PARTITION
            )
            verify_simulation_manifest(
                run_root, "intervention", dataset_partition=HOLDOUT_PARTITION
            )
        reporter.start_stage(stage)
        reporter.finish_stage(stage, skipped=True)
        return
    dependencies = {
        "indicator_freeze": ["indicator_generation"],
        "path_generation": ["indicator_freeze"],
        "semantic_freeze": ["path_generation"],
        "baseline_simulation": ["semantic_freeze"],
        "temporal": ["semantic_freeze", "baseline_simulation"],
        "path_temporal_qualification": ["temporal"],
        "intervention_simulation": ["path_temporal_qualification"],
        "intervention": ["intervention_simulation"],
        "path_intervention_classification": ["intervention"],
        "prospective": ["path_intervention_classification"],
        "primary_freeze": ["prospective"],
        "dose_response": ["primary_freeze"],
        "holdout_simulation": ["dose_response"],
        "holdout_confirmation": ["holdout_simulation"],
        "temporal_negative_control": ["holdout_confirmation"],
        "robustness": ["temporal_negative_control"],
        "export": ["robustness"],
        "render": ["export"],
    }
    _require_stages(manager, dependencies.get(stage, []))
    reporter.start_stage(stage)
    callback = _progress_callback(reporter)
    manager.record_timestamp(f"{stage}_start_unix_time")
    started = time.perf_counter()
    details: dict[str, Any]
    outputs: list[Path]
    if stage == "indicator_generation":
        details = run_indicator_generation_stage(
            config, llm_config_path, run_root, prompt_template_path, workers,
            lambda label, done, total: reporter.update(label, done, total),
            completion_provider,
        )
        outputs = _files(run_root, [
            "llm/indicator/**/*",
            "representation/indicator_selection.json",
            "representation/indicator_generation_validation.json",
            "representation/indicator_replication_pairwise.csv",
            "representation/representation_agreement.json",
        ])
    elif stage == "indicator_freeze":
        details = freeze_indicator_stage(config, run_root)
        outputs = _files(run_root, [
            "representation/INDICATORS_FROZEN.sha256",
            "representation/indicators_frozen.json",
        ])
    elif stage == "path_generation":
        details = run_path_generation_stage(
            config, llm_config_path, run_root, prompt_template_path, workers,
            lambda label, done, total: reporter.update(label, done, total),
            completion_provider,
        )
        outputs = _files(run_root, ["llm/path/**/*", "representation/*"])
    elif stage == "semantic_freeze":
        representations = load_frozen_representations(run_root)
        details = {
            "semantic_scenarios": len(representations),
            "all_semantics_frozen_before_baseline": True,
        }
        outputs = _files(run_root, ["representation/*"])
    elif stage == "baseline_simulation":
        details = run_baseline_simulation_stage(config, run_root, workers, callback)
        outputs = _files(run_root, [
            "data/baseline_simulation_manifest.json",
            "data/primary/baseline_simulation_manifest.json",
            "data/primary/raw_logs/**/baseline/*.npz",
            "data/primary/raw_logs/**/*.sha256",
            "data/primary/reference_hidden/**/*.npz",
            "data/primary/reference_hidden/**/*.sha256",
        ])
    elif stage == "temporal":
        verify_simulation_manifest(run_root, "baseline")
        representations = load_frozen_representations(run_root)
        baseline = compile_indicator_dataset(
            config, run_root, representations, workers, complete=False,
            progress_callback=callback,
        )
        controlled_baseline = compile_controlled_dataset(run_root, complete=False)
        full_details = run_temporal_stage(
            config, run_root, representations, baseline, workers, callback
        )
        controlled_details = run_controlled_temporal_stage(
            config, run_root, controlled_baseline, workers, callback
        )
        evaluate_main_graphs(run_root, representations)
        details = {
            "full_discovery_graph_records": len(full_details["graphs"]),
            "controlled_recovery_graph_records": len(controlled_details["graphs"]),
        }
        outputs = _files(run_root, [
            "data/indicator_trajectories_baseline.*",
            "data/controlled_recovery_trajectories_baseline.*",
            "analysis/main_graphs.jsonl", "analysis/controlled_recovery_graphs.jsonl",
            "analysis/bootstrap_summary.json",
            "analysis/method_runtime.csv", "analysis/controlled_recovery_runtime.csv",
        ])
    elif stage == "path_temporal_qualification":
        representations = load_frozen_representations(run_root)
        graphs = load_graph_records(run_root / "analysis" / "main_graphs.jsonl")
        frames = [
            qualify_candidate_paths(
                scenario, graphs[(scenario, "full_method")], representation
            )
            for scenario, representation in sorted(representations.items())
        ]
        qualification = pd.concat(frames, ignore_index=True)
        output = run_root / "analysis" / "path_temporal_qualification.csv"
        qualification.to_csv(output, index=False)
        details = {
            "candidate_paths": len(qualification),
            "temporally_qualified_paths": int(
                qualification["path_temporally_qualified"].astype(bool).sum()
            ),
        }
        outputs = [output]
    elif stage == "intervention_simulation":
        details = run_intervention_simulation_stage(config, run_root, workers, callback)
        outputs = _files(run_root, [
            "data/intervention_simulation_manifest.json",
            "data/primary/intervention_simulation_manifest.json",
            "data/primary/raw_logs/**/*.npz",
            "data/primary/raw_logs/**/*.sha256",
            "data/primary/reference_hidden/**/*.npz",
            "data/primary/reference_hidden/**/*.sha256",
        ])
    elif stage == "intervention":
        verify_simulation_manifest(run_root, "baseline")
        verify_simulation_manifest(run_root, "intervention")
        representations = load_frozen_representations(run_root)
        complete = compile_indicator_dataset(
            config, run_root, representations, workers, complete=True,
            progress_callback=callback,
        )
        controlled_complete = compile_controlled_dataset(run_root, complete=True)
        details = run_intervention_stage(
            config, run_root, representations, complete, workers, callback,
        )
        controlled_details = run_controlled_intervention_stage(
            config, run_root, controlled_complete, workers, callback
        )
        details["controlled_recovery"] = controlled_details
        update_intervention_metrics(run_root, representations)
        outputs = _files(run_root, [
            "data/indicator_trajectories_complete.*",
            "data/controlled_recovery_trajectories_complete.*",
            "analysis/paired_effects.parquet", "analysis/effect_curves.parquet",
            "analysis/mechanism_bidirectional_summary.csv",
            "analysis/intervention_classifications.csv",
            "analysis/edge_intervention_classifications.csv",
            "analysis/path_timing_summary.csv",
            "analysis/controlled_recovery_*", "analysis/main_results.csv",
        ])
    elif stage == "prospective":
        representations = load_frozen_representations(run_root)
        prospective = validate_prospective_predictions(run_root, representations)
        details = {
            "prospective_validation_rows": len(prospective),
            "holdout_used_for_primary": False,
        }
        outputs = [run_root / "analysis" / "prospective_validation.csv"]
    elif stage == "path_intervention_classification":
        representations = load_frozen_representations(run_root)
        qualification = pd.read_csv(
            run_root / "analysis" / "path_temporal_qualification.csv"
        )
        edge_attempts = pd.read_csv(
            run_root / "analysis" / "intervention_classifications.csv"
        )
        path_classification = classify_candidate_paths(
            qualification, edge_attempts, representations
        )
        output = run_root / "analysis" / "path_intervention_classification.csv"
        path_classification.to_csv(output, index=False)
        selection = select_representative_paths(path_classification)
        selection_path = run_root / "analysis" / "representative_path_selection.json"
        selection_path.write_text(
            json.dumps(selection, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        details = {
            "classified_paths": len(path_classification),
            "supported_paths": int(
                (path_classification["path_classification"] == "supported").sum()
            ),
        }
        outputs = [output, selection_path]
    elif stage == "primary_freeze":
        primary_freeze = freeze_primary_contract(run_root)
        details = {"primary_discovery_frozen": True, "holdout_used_for_primary": False}
        outputs = [primary_freeze]
    elif stage == "dose_response":
        verify_primary_contract(run_root)
        representations = load_frozen_representations(run_root)
        complete = pd.read_parquet(
            run_root / "data" / "indicator_trajectories_complete.parquet"
        )
        details = run_dose_response_analysis(
            config, run_root, representations, complete
        )
        verify_primary_contract(run_root)
        outputs = _files(run_root, [
            "analysis/dose_response_effects.csv",
            "analysis/dose_response_summary.csv",
            "analysis/path_dose_response_summary.csv",
        ])
    elif stage == "holdout_simulation":
        verify_primary_contract(run_root)
        baseline_details = run_holdout_baseline_simulation_stage(
            config, run_root, workers, callback
        )
        intervention_details = run_holdout_intervention_simulation_stage(
            config, run_root, workers, callback
        )
        verify_primary_contract(run_root)
        details = {
            "baseline_tasks": int(baseline_details["requested_tasks"]),
            "intervention_tasks": int(intervention_details["requested_tasks"]),
            "total_holdout_tasks": int(baseline_details["requested_tasks"])
            + int(intervention_details["requested_tasks"]),
            "primary_result_unchanged": True,
        }
        outputs = _files(run_root, [
            "data/holdout/*_simulation_manifest.json",
            "data/holdout/raw_logs/**/*.npz",
            "data/holdout/raw_logs/**/*.sha256",
            "data/holdout/reference_hidden/**/*.npz",
            "data/holdout/reference_hidden/**/*.sha256",
        ])
    elif stage == "holdout_confirmation":
        verify_primary_contract(run_root)
        verify_simulation_manifest(
            run_root, "baseline", dataset_partition=HOLDOUT_PARTITION
        )
        verify_simulation_manifest(
            run_root, "intervention", dataset_partition=HOLDOUT_PARTITION
        )
        representations = load_frozen_representations(run_root)
        holdout = compile_indicator_dataset(
            config,
            run_root,
            representations,
            workers,
            complete=True,
            dataset_partition=HOLDOUT_PARTITION,
            progress_callback=callback,
        )
        details = run_holdout_confirmation_analysis(
            config, run_root, representations, holdout, workers, callback
        )
        outputs = _files(run_root, [
            "data/holdout_indicator_trajectories_complete.*",
            "analysis/holdout_paired_effects.parquet",
            "analysis/holdout_effect_curves.parquet",
            "analysis/holdout_intervention_classifications.csv",
            "analysis/holdout_path_confirmation.csv",
            "analysis/holdout_prospective_confirmation.csv",
            "analysis/holdout_mechanism_confirmation.csv",
            "analysis/path_funnel_summary.csv",
            "analysis/representative_path_selection.json",
        ])
    elif stage == "temporal_negative_control":
        verify_primary_contract(run_root)
        representations = load_frozen_representations(run_root)
        baseline = pd.read_parquet(
            run_root / "data" / "indicator_trajectories_baseline.parquet"
        )
        complete = pd.read_parquet(
            run_root / "data" / "indicator_trajectories_complete.parquet"
        )
        details = run_falsification_analyses(
            config,
            run_root,
            representations,
            baseline,
            complete,
            workers,
            callback,
        )
        verify_primary_contract(run_root)
        outputs = _files(run_root, [
            "analysis/temporal_negative_control.csv",
            "analysis/path_temporal_negative_control.csv",
            "analysis/path_mechanism_attenuation.csv",
        ])
    elif stage == "robustness":
        representations = load_frozen_representations(run_root)
        baseline = pd.read_parquet(run_root / "data" / "indicator_trajectories_baseline.parquet")
        complete = pd.read_parquet(run_root / "data" / "indicator_trajectories_complete.parquet")
        details = run_robustness_stage(
            config, run_root, representations, baseline, complete, workers, callback
        )
        outputs = _files(run_root, [
            "analysis/*.csv",
            "analysis/robustness_pool_profile.json",
            "data/predefined_observable_trajectories_baseline.*",
        ])
    elif stage == "export":
        representations = load_frozen_representations(run_root)
        attribution = integrate_evidence(run_root, representations)
        tables = generate_tables(run_root)
        bundle = create_visualization_bundle(run_root)
        details = {
            "attribution_scenarios": len(attribution["scenarios"]),
            "table_count": len(tables), "figure_input_count": len(bundle["figures"]),
        }
        outputs = [
            path for path in _files(run_root, [
                "analysis/attribution_objects.json",
                "analysis/comparative_method_intervention_evidence.csv",
                "tables/*",
                "visualization_input/**/*",
            ])
            if path != run_root / "visualization_input" / "render_manifest.json"
        ]
    elif stage == "render":
        details = render_all_figures(run_root, plot_repo)
        outputs = _files(run_root, ["figures/*", "visualization_input/render_manifest.json"])
    else:
        raise KeyError(stage)
    duration = time.perf_counter() - started
    manager.mark_stage_completed(stage, outputs, duration, details)
    manager.record_timestamp(f"{stage}_complete_unix_time")
    if stage == "semantic_freeze":
        manager.record_timestamp("semantic_freeze_unix_time")
        manager.record_timestamp("prediction_freeze_unix_time")
    elif stage == "primary_freeze":
        manager.record_timestamp("primary_discovery_freeze_unix_time")
    reporter.finish_stage(stage)


def run_selected_stages(
    stages: list[str],
    manager: RunManager,
    workers: int,
    prompt_template_path: Path,
    llm_config_path: Path,
    *,
    no_render: bool,
    plot_repo: Path | None,
    completion_provider: Callable[[str, int], Callable[[str, str], LLMResponse]] | None = None,
) -> None:
    expanded = STAGE_ORDER if stages == ["all"] else stages
    if no_render:
        expanded = [stage for stage in expanded if stage != "render"]
    with ProgressReporter(manager.run_root, workers) as reporter:
        for stage in expanded:
            run_stage(
                stage, manager, workers, reporter, prompt_template_path,
                llm_config_path, plot_repo, completion_provider,
            )
    if all(manager.stage_complete(stage) for stage in STAGE_ORDER):
        manager.finalise()
    elif expanded == [stage for stage in STAGE_ORDER if stage != "render"]:
        manifest = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "analysis_completed_waiting_for_render"
        manager.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
