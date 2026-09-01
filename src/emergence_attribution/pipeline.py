"""Stage orchestration for immutable formal runs."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from .evaluation import evaluate_main_graphs, update_intervention_metrics
from .exporting import create_visualization_bundle, generate_tables, integrate_evidence
from .interventions import run_intervention_stage
from .progress import ProgressReporter
from .prospective import validate_prospective_predictions
from .provenance import RunContractError, RunManager
from .rendering import render_all_figures
from .robustness import run_robustness_stage
from .semantic import load_frozen_representations, run_semantic_stage
from .simulation import (
    compile_indicator_dataset,
    run_simulation_stage,
    verify_simulation_manifest,
)
from .temporal import run_temporal_stage


STAGE_ORDER = [
    "simulation",
    "semantic",
    "temporal",
    "intervention",
    "robustness",
    "export",
    "render",
]


def load_experiment_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    config = json.loads(path.read_text(encoding="utf-8"))
    required = {
        "master_seed",
        "random_seeds",
        "representation",
        "temporal",
        "intervention",
        "robustness",
        "render",
        "scenarios",
    }
    missing = sorted(required - set(config))
    if missing:
        raise ValueError(f"experiment configuration fields are missing: {missing}")
    intervention = config["intervention"]
    frozen_values = {
        "onset_detection_start": 0,
        "minimum_standardised_effect": 0.10,
        "onset_consecutive_steps": 4,
        "evaluation_start": 15,
    }
    for key, expected in frozen_values.items():
        if intervention.get(key) != expected:
            raise ValueError(f"formal intervention setting {key} must equal {expected}")
    temporal = config["temporal"]
    if temporal.get("maximum_lag") != 5:
        raise ValueError("formal candidate lags must be exactly 1 through 5")
    if temporal.get("bootstrap_repetitions") != 100:
        raise ValueError("formal trajectory bootstrap repetitions must be 100")
    if intervention.get("bootstrap_repetitions") != 500:
        raise ValueError("formal paired bootstrap repetitions must be 500")
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
) -> None:
    run_root, config = manager.run_root, manager.config
    if manager.stage_complete(stage):
        if stage == "simulation":
            verify_simulation_manifest(run_root)
        reporter.start_stage(stage)
        reporter.finish_stage(stage, skipped=True)
        return
    dependencies = {
        "temporal": ["simulation", "semantic"],
        "intervention": ["simulation", "semantic", "temporal"],
        "robustness": ["simulation", "semantic", "temporal", "intervention"],
        "export": ["simulation", "semantic", "temporal", "intervention", "robustness"],
        "render": ["export"],
    }
    _require_stages(manager, dependencies.get(stage, []))
    reporter.start_stage(stage)
    callback = _progress_callback(reporter)
    started = time.perf_counter()
    details: dict[str, Any]
    outputs: list[Path]
    if stage == "simulation":
        details = run_simulation_stage(config, run_root, workers, callback)
        outputs = _files(
            run_root,
            ["data/simulation_manifest.json", "data/raw_logs/**/*.npz", "data/raw_logs/**/*.sha256"],
        )
    elif stage == "semantic":
        details = run_semantic_stage(
            config,
            llm_config_path,
            run_root,
            prompt_template_path,
            workers,
            lambda label, done, total: reporter.update(label, done, total),
        )
        outputs = _files(run_root, ["llm/**/*", "representation/*"])
    elif stage == "temporal":
        verify_simulation_manifest(run_root)
        representations = load_frozen_representations(run_root)
        baseline = compile_indicator_dataset(
            config,
            run_root,
            representations,
            workers,
            complete=False,
            progress_callback=callback,
        )
        details = run_temporal_stage(
            config, run_root, representations, baseline, workers, callback
        )
        evaluate_main_graphs(run_root, representations)
        outputs = _files(
            run_root,
            [
                "data/indicator_trajectories_baseline.*",
                "analysis/main_graphs.jsonl",
                "analysis/main_results.csv",
                "analysis/bootstrap_summary.json",
                "analysis/indicator_alignment.json",
                "analysis/graph_evaluation.json",
            ],
        )
        details = {
            "scenario_count": len(representations),
            "graph_record_count": len(details["graphs"]),
        }
    elif stage == "intervention":
        verify_simulation_manifest(run_root)
        representations = load_frozen_representations(run_root)
        complete = compile_indicator_dataset(
            config,
            run_root,
            representations,
            workers,
            complete=True,
            progress_callback=callback,
        )
        details = run_intervention_stage(
            config, run_root, representations, complete, workers, callback
        )
        prospective = validate_prospective_predictions(run_root, representations)
        update_intervention_metrics(run_root, representations)
        details["prospective_validation_rows"] = len(prospective)
        outputs = _files(
            run_root,
            [
                "data/indicator_trajectories_complete.*",
                "analysis/paired_effects.parquet",
                "analysis/effect_curves.parquet",
                "analysis/intervention_classifications.csv",
                "analysis/path_timing_summary.csv",
                "analysis/representative_path_selection.json",
                "analysis/prospective_validation.csv",
                "analysis/main_results.csv",
            ],
        )
    elif stage == "robustness":
        representations = load_frozen_representations(run_root)
        baseline = pd.read_parquet(
            run_root / "data" / "indicator_trajectories_baseline.parquet"
        )
        complete = pd.read_parquet(
            run_root / "data" / "indicator_trajectories_complete.parquet"
        )
        details = run_robustness_stage(
            config,
            run_root,
            representations,
            baseline,
            complete,
            workers,
            callback,
        )
        outputs = _files(
            run_root,
            [
                "analysis/functional_ablations.csv",
                "analysis/data_efficiency_repeated_subsampling.csv",
                "analysis/observation_robustness.csv",
                "analysis/representation_robustness.csv",
                "analysis/mechanism_disabled_checks.csv",
                "analysis/causal_scalability.csv",
                "data/predefined_observable_trajectories_baseline.*",
            ],
        )
    elif stage == "export":
        representations = load_frozen_representations(run_root)
        attribution = integrate_evidence(run_root, representations)
        table_paths = generate_tables(run_root)
        bundle = create_visualization_bundle(run_root)
        details = {
            "attribution_scenarios": len(attribution["scenarios"]),
            "table_count": len(table_paths),
            "figure_input_count": len(bundle["figures"]),
        }
        outputs = _files(
            run_root,
            [
                "analysis/attribution_objects.json",
                "tables/*",
                "visualization_input/**/*",
            ],
        )
        outputs = [
            path
            for path in outputs
            if path != run_root / "visualization_input" / "render_manifest.json"
        ]
    elif stage == "render":
        details = render_all_figures(run_root, plot_repo)
        outputs = _files(run_root, ["figures/*", "visualization_input/render_manifest.json"])
    else:
        raise KeyError(stage)
    duration = time.perf_counter() - started
    manager.mark_stage_completed(stage, outputs, duration, details)
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
) -> None:
    expanded = STAGE_ORDER if stages == ["all"] else stages
    if no_render:
        expanded = [stage for stage in expanded if stage != "render"]
    with ProgressReporter(manager.run_root, workers) as reporter:
        for stage in expanded:
            run_stage(
                stage,
                manager,
                workers,
                reporter,
                prompt_template_path,
                llm_config_path,
                plot_repo,
            )
    if all(manager.stage_complete(stage) for stage in STAGE_ORDER):
        manager.finalise()
    elif expanded == [stage for stage in STAGE_ORDER if stage != "render"]:
        manifest = json.loads(manager.manifest_path.read_text(encoding="utf-8"))
        manifest["status"] = "analysis_completed_waiting_for_render"
        manager.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
