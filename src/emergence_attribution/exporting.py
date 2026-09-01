"""Evidence integration, paper-data tables, and dynamic visualisation bundles."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .provenance import sha256_file
from .temporal import load_graph_records


CORE_ANALYSIS_FILES = [
    "main_graphs.jsonl",
    "controlled_recovery_graphs.jsonl",
    "main_results.csv",
    "full_discovery_results.csv",
    "controlled_recovery_results.csv",
    "data_efficiency_repeated_subsampling.csv",
    "effect_curves.parquet",
    "paired_effects.parquet",
    "path_timing_summary.csv",
    "representative_path_selection.json",
    "intervention_classifications.csv",
    "edge_intervention_classifications.csv",
    "controlled_recovery_edge_intervention_classifications.csv",
    "comparative_method_intervention_evidence.csv",
    "observation_robustness.csv",
    "representation_robustness.csv",
    "robustness_pool_profile.json",
    "causal_scalability.csv",
    "prospective_validation.csv",
]


def integrate_evidence(
    run_root: Path, representations: dict[str, dict[str, Any]]
) -> dict[str, Any]:
    graphs = load_graph_records(run_root / "analysis" / "main_graphs.jsonl")
    classifications = pd.read_csv(
        run_root / "analysis" / "intervention_classifications.csv"
    )
    if "method" not in classifications.columns:
        raise RuntimeError("intervention classifications do not identify their method")
    primary_classifications = classifications[
        classifications["method"] == "full_method"
    ].copy()
    comparative = classifications[
        classifications["method"] != "full_method"
    ].copy()
    comparative.to_csv(
        run_root / "analysis" / "comparative_method_intervention_evidence.csv",
        index=False,
    )
    result: dict[str, Any] = {
        "schema_version": "1.0",
        "interpretation": "semantic hypotheses, temporal evidence, and intervention evidence remain distinct",
        "scenarios": {},
    }
    for scenario, representation in sorted(representations.items()):
        graph = graphs[(scenario, "full_method")]
        intervention_lookup: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for row in primary_classifications[
            primary_classifications["scenario"] == scenario
        ].to_dict(
            orient="records"
        ):
            intervention_lookup.setdefault((row["source"], row["target"]), []).append(row)
        retained = {(edge.source, edge.target): edge for edge in graph}
        relations = []
        for semantic in representation["candidate_edges"]:
            pair = (semantic["source"], semantic["target"])
            temporal = retained.get(pair)
            relations.append(
                {
                    "source": semantic["source"],
                    "target": semantic["target"],
                    "semantic_hypothesis": {
                        "expected_direction": semantic["expected_direction"],
                        "rationale": semantic["rationale"],
                    },
                    "temporal_evidence": asdict(temporal) if temporal else None,
                    "intervention_evidence": intervention_lookup.get(pair, []),
                }
            )
        result["scenarios"][scenario] = {
            "phenomenon": representation["phenomenon"],
            "representation": representation,
            "relations": relations,
        }
    path = run_root / "analysis" / "attribution_objects.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    return result


def generate_tables(run_root: Path) -> list[Path]:
    table_root = run_root / "tables"
    table_root.mkdir(parents=True, exist_ok=True)
    main = pd.read_csv(run_root / "analysis" / "main_results.csv")
    main.to_csv(table_root / "main_results_source.csv", index=False)
    outputs = [table_root / "main_results_source.csv"]
    ablation_path = run_root / "analysis" / "functional_ablations.csv"
    if ablation_path.exists():
        ablation = pd.read_csv(ablation_path)
        ablation.to_csv(table_root / "functional_ablation_source.csv", index=False)
        outputs.append(table_root / "functional_ablation_source.csv")
    prospective_path = run_root / "analysis" / "prospective_validation.csv"
    if prospective_path.exists():
        prospective = pd.read_csv(prospective_path)
        summary = (
            prospective.groupby(["scenario", "classification"], dropna=False)
            .size()
            .reset_index(name="count")
        )
        summary.to_csv(table_root / "prospective_validation_summary.csv", index=False)
        outputs.append(table_root / "prospective_validation_summary.csv")
    return outputs


def _frame_manifest(path: Path, required_columns: list[str]) -> dict[str, Any]:
    if path.suffix == ".csv":
        frame = pd.read_csv(path)
        file_format = "csv"
    else:
        frame = pd.read_parquet(path)
        file_format = "parquet"
    missing = sorted(set(required_columns) - set(frame.columns))
    if missing:
        raise RuntimeError(f"{path.name} lacks visualisation columns: {missing}")
    return {
        "path": path.as_posix(),
        "format": file_format,
        "records": len(frame),
        "required_columns": required_columns,
    }


def create_visualization_bundle(run_root: Path) -> dict[str, Any]:
    bundle_root = run_root / "visualization_input"
    manifest_path = bundle_root / "figure_inputs.generated.json"
    if manifest_path.is_file():
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    if (run_root / "RUN_FROZEN").exists():
        raise RuntimeError(
            "the run is frozen but has no visualization manifest; refusing to modify it"
        )
    data_root = bundle_root / "data"
    analysis_root = bundle_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    for name in CORE_ANALYSIS_FILES:
        source = run_root / "analysis" / name
        if source.is_file():
            shutil.copy2(source, analysis_root / name)
    raw_items = []
    simulation_manifest = json.loads(
        (run_root / "data" / "baseline_simulation_manifest.json").read_text(encoding="utf-8")
    )
    for scenario in sorted({item["scenario"] for item in simulation_manifest["task_records"]}):
        baseline = sorted(
            (
                item
                for item in simulation_manifest["task_records"]
                if item["scenario"] == scenario and item["condition"] == "baseline"
            ),
            key=lambda item: int(item["seed"]),
        )[0]
        source = run_root / baseline["raw_path"]
        destination = data_root / "raw_logs" / scenario / "baseline" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)
        with np.load(destination, allow_pickle=False) as archive:
            arrays = {name: list(archive[name].shape) for name in archive.files}
        raw_items.append(
            {
                "path": destination.relative_to(bundle_root).as_posix(),
                "format": "npz",
                "arrays": arrays,
            }
        )
    relative_analysis = lambda name: Path("analysis") / name
    figures = {
        "02": {
            "paper_pdf_figure": 2,
            "output": "figure_2_simulation_dynamics.png",
            "inputs": raw_items,
        },
        "03": {
            "paper_pdf_figure": 3,
            "output": "figure_3_graph_recovery.png",
            "inputs": [
                {
                    "path": relative_analysis("controlled_recovery_graphs.jsonl").as_posix(),
                    "format": "jsonl",
                    "records": sum(
                        1
                        for line in (analysis_root / "controlled_recovery_graphs.jsonl")
                        .read_text(encoding="utf-8")
                        .splitlines()
                        if line.strip()
                    ),
                },
                _frame_manifest(
                    analysis_root / "controlled_recovery_results.csv",
                    ["evaluation_track", "scenario", "method", "edge_f1", "shd", "stability", "lag_mae"],
                ),
            ],
        },
        "04": {
            "paper_pdf_figure": 4,
            "output": "figure_4_data_efficiency.png",
            "inputs": [
                _frame_manifest(
                    analysis_root / "data_efficiency_repeated_subsampling.csv",
                    [
                        "scenario",
                        "method",
                        "trajectory_count",
                        "repetition",
                        "temporal_qualification_rate",
                        "stability",
                        "temporal_qualification_rate_ci_low",
                        "temporal_qualification_rate_ci_high",
                        "stability_ci_low",
                        "stability_ci_high",
                    ],
                )
            ],
        },
        "05": {
            "paper_pdf_figure": 5,
            "output": "figure_5_intervention_timing.png",
            "inputs": [
                _frame_manifest(
                    analysis_root / "effect_curves.parquet",
                    ["scenario", "parameter", "direction", "node_id", "time", "mean", "ci_low", "ci_high"],
                ),
                _frame_manifest(
                    analysis_root / "path_timing_summary.csv",
                    ["scenario", "path_id", "scale", "onset_time", "observational_lag", "lag_difference", "cumulative_effect", "significant"],
                ),
                {
                    "path": relative_analysis("representative_path_selection.json").as_posix(),
                    "format": "json",
                },
            ],
        },
        "06": {
            "paper_pdf_figure": 6,
            "output": "figure_6_effect_matrix.png",
            "inputs": [
                _frame_manifest(
                    analysis_root / "paired_effects.parquet",
                    ["scenario", "parameter", "direction", "node_id", "scale", "cumulative_effect", "significant"],
                )
            ],
        },
        "07": {
            "paper_pdf_figure": 7,
            "output": "figure_7_multiscale_propagation.png",
            "inputs": [
                _frame_manifest(
                    analysis_root / "path_timing_summary.csv",
                    ["scenario", "path_id", "parameter", "direction", "source", "meso", "macro", "scale", "onset_time", "cumulative_effect", "significant"],
                ),
                _frame_manifest(
                    analysis_root / "intervention_classifications.csv",
                    ["scenario", "source", "target", "parameter", "direction", "manipulation_success", "primary_class", "underlying_class"],
                ),
            ],
        },
        "08": {
            "paper_pdf_figure": 8,
            "output": "figure_8_robustness_efficiency.png",
            "inputs": [
                _frame_manifest(
                    analysis_root / "observation_robustness.csv",
                    ["scenario", "factor", "noise_level", "missing_fraction", "support_threshold", "repetition", "temporal_qualification_rate", "stability", "retained_edge_count"],
                ),
                _frame_manifest(
                    analysis_root / "causal_scalability.csv",
                    ["scenario", "candidate_indicator_count", "repetition", "runtime_seconds", "discovered_edge_count"],
                ),
            ],
        },
    }
    for figure in figures.values():
        for item in figure["inputs"]:
            if item.get("format") in {"csv", "parquet"}:
                item["path"] = Path(item["path"]).relative_to(bundle_root).as_posix()
    manifest = {
        "schema_version": "2.0",
        "source_run": run_root.name,
        "render_backend": "dynamic Python/matplotlib adapter",
        "dynamic_row_counts": True,
        "figures": figures,
    }
    manifest_path.write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    hashes = {}
    for path in sorted(bundle_root.rglob("*")):
        if path.is_file() and path.name not in {"SHA256SUMS", "render_manifest.json"}:
            hashes[path.relative_to(bundle_root).as_posix()] = sha256_file(path)
    (bundle_root / "SHA256SUMS").write_text(
        "".join(f"{digest}  {relative}\n" for relative, digest in hashes.items()),
        encoding="ascii",
    )
    render_manifest = {
        "schema_version": "1.0",
        "status": "not_rendered",
        "source_run": run_root.name,
        "input_manifest_sha256": sha256_file(manifest_path),
    }
    (bundle_root / "render_manifest.json").write_text(
        json.dumps(render_manifest, indent=2), encoding="utf-8"
    )
    return manifest


def export_bundle_to_plot_repository(run_root: Path, plot_repo: Path) -> Path:
    if not plot_repo.is_dir():
        raise FileNotFoundError(f"plotting repository does not exist: {plot_repo}")
    source = run_root / "visualization_input"
    if not source.is_dir():
        create_visualization_bundle(run_root)
    destination = plot_repo / "data" / "generated_runs" / run_root.name
    if destination.exists():
        raise FileExistsError(
            f"generated bundle destination already exists and will not be overwritten: {destination}"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(source, destination)
    return destination
