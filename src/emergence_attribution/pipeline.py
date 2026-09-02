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
from .interventions import run_intervention_stage
from .llm_client import LLMResponse
from .progress import ProgressReporter
from .prospective import validate_prospective_predictions
from .provenance import RunContractError, RunManager
from .rendering import render_all_figures
from .robustness import run_robustness_stage
from .semantic import load_frozen_representations, run_semantic_stage
from .simulation import (
    compile_indicator_dataset,
    run_baseline_simulation_stage,
    run_intervention_simulation_stage,
    verify_simulation_manifest,
)
from .temporal import run_temporal_stage


STAGE_ORDER = [
    "semantic",
    "baseline_simulation",
    "temporal",
    "intervention_simulation",
    "intervention",
    "prospective",
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
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"experiment configuration fields are missing: {missing}")
    representation = config["representation"]
    if int(representation["minimum_candidate_edges"]) < 28:
        raise ValueError("minimum candidate edge count must be at least 28")
    if int(representation["maximum_candidate_edges"]) < int(
        representation["minimum_candidate_edges"]
    ):
        raise ValueError("maximum candidate edge count is below the minimum")
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
            "required_branch_count": 4,
            "minimum_candidate_edges": 28,
            "maximum_candidate_edges": 48,
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
        reporter.start_stage(stage)
        reporter.finish_stage(stage, skipped=True)
        return
    dependencies = {
        "baseline_simulation": ["semantic"],
        "temporal": ["semantic", "baseline_simulation"],
        "intervention_simulation": ["temporal"],
        "intervention": ["intervention_simulation"],
        "prospective": ["intervention"],
        "robustness": ["prospective"],
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
    if stage == "semantic":
        details = run_semantic_stage(
            config, llm_config_path, run_root, prompt_template_path, workers,
            lambda label, done, total: reporter.update(label, done, total),
            completion_provider,
        )
        outputs = _files(run_root, ["llm/**/*", "representation/*"])
    elif stage == "baseline_simulation":
        details = run_baseline_simulation_stage(config, run_root, workers, callback)
        outputs = _files(run_root, [
            "data/baseline_simulation_manifest.json", "data/raw_logs/**/baseline/*.npz",
            "data/raw_logs/**/*.sha256", "data/reference_hidden/**/*.npz",
            "data/reference_hidden/**/*.sha256",
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
    elif stage == "intervention_simulation":
        details = run_intervention_simulation_stage(config, run_root, workers, callback)
        outputs = _files(run_root, [
            "data/intervention_simulation_manifest.json", "data/raw_logs/**/*.npz",
            "data/raw_logs/**/*.sha256", "data/reference_hidden/**/*.npz",
            "data/reference_hidden/**/*.sha256",
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
            "analysis/intervention_classifications.csv",
            "analysis/edge_intervention_classifications.csv",
            "analysis/path_timing_summary.csv",
            "analysis/controlled_recovery_*", "analysis/main_results.csv",
        ])
    elif stage == "prospective":
        representations = load_frozen_representations(run_root)
        prospective = validate_prospective_predictions(run_root, representations)
        details = {"prospective_validation_rows": len(prospective)}
        outputs = [run_root / "analysis" / "prospective_validation.csv"]
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
    if stage == "semantic":
        manager.record_timestamp("semantic_freeze_unix_time")
        manager.record_timestamp("prediction_freeze_unix_time")
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
