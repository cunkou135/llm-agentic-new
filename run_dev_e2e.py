"""Run the isolated NON_SCIENTIFIC development pipeline with resume coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergence_attribution.llm_client import load_llm_config  # noqa: E402
from emergence_attribution.mock_semantic import mock_completion_provider  # noqa: E402
from emergence_attribution.pipeline import (  # noqa: E402
    STAGE_ORDER,
    load_experiment_config,
    run_selected_stages,
)
from emergence_attribution.provenance import RunManager  # noqa: E402
from emergence_attribution.controlled import controlled_representation  # noqa: E402
from emergence_attribution.interventions import graph_paths  # noqa: E402
from emergence_attribution.semantic import load_frozen_representations  # noqa: E402
from emergence_attribution.temporal import load_graph_records  # noqa: E402


def _stage3_coverage(
    classifications: pd.DataFrame,
    representations: dict[str, dict],
    graphs: dict,
) -> dict:
    totals = {
        "micro_to_meso_testable_edges": 0,
        "meso_to_macro_testable_edges": 0,
        "supported_micro_to_meso": 0,
        "supported_meso_to_macro": 0,
        "unmapped_meso_to_macro": 0,
        "complete_testable_paths": 0,
        "complete_supported_paths": 0,
        "scenarios": {},
    }
    for scenario, representation in sorted(representations.items()):
        scale = {item["id"]: item["scale"] for item in representation["indicators"]}
        subset = classifications[
            (classifications["scenario"] == scenario)
            & (classifications["method"] == "full_method")
        ]
        applicable = subset[subset["primary_class"] != "not_applicable"]
        testable_pairs = {
            (str(row.source), str(row.target)) for row in applicable.itertuples()
        }
        supported_pairs = {
            (str(row.source), str(row.target))
            for row in applicable.itertuples()
            if row.primary_class == "supported"
        }
        micro_meso = {
            pair for pair in testable_pairs
            if scale.get(pair[0]) == "micro" and scale.get(pair[1]) == "meso"
        }
        meso_macro = {
            pair for pair in testable_pairs
            if scale.get(pair[0]) == "meso" and scale.get(pair[1]) == "macro"
        }
        unmapped = {
            (str(row.source), str(row.target))
            for row in subset.itertuples()
            if row.primary_class == "not_applicable"
            and scale.get(str(row.source)) == "meso"
            and scale.get(str(row.target)) == "macro"
        }
        paths = graph_paths(graphs[(scenario, "full_method")], representation)
        complete_testable = {
            path for path in paths
            if (path[0], path[1]) in testable_pairs
            and (path[1], path[2]) in testable_pairs
        }
        complete_supported = {
            path for path in paths
            if (path[0], path[1]) in supported_pairs
            and (path[1], path[2]) in supported_pairs
        }
        metrics = {
            "micro_to_meso_testable_edges": len(micro_meso),
            "meso_to_macro_testable_edges": len(meso_macro),
            "supported_micro_to_meso": len(micro_meso & supported_pairs),
            "supported_meso_to_macro": len(meso_macro & supported_pairs),
            "unmapped_meso_to_macro": len(unmapped),
            "complete_testable_paths": len(complete_testable),
            "complete_supported_paths": len(complete_supported),
        }
        totals["scenarios"][scenario] = metrics
        for name, value in metrics.items():
            totals[name] += value
    return totals


def write_dev_stage3_audit(run_root: Path) -> dict:
    full_representations = load_frozen_representations(run_root)
    controlled_representations = {
        scenario: controlled_representation(scenario)
        for scenario in full_representations
    }
    full = _stage3_coverage(
        pd.read_csv(run_root / "analysis" / "intervention_classifications.csv"),
        full_representations,
        load_graph_records(run_root / "analysis" / "main_graphs.jsonl"),
    )
    controlled = _stage3_coverage(
        pd.read_csv(
            run_root / "analysis" / "controlled_recovery_intervention_classifications.csv"
        ),
        controlled_representations,
        load_graph_records(run_root / "analysis" / "controlled_recovery_graphs.jsonl"),
    )
    passed = bool(
        set(controlled["scenarios"]) == {"schelling", "deffuant"}
        and controlled["meso_to_macro_testable_edges"] > 0
        and controlled["complete_testable_paths"] > 0
    )
    audit = {
        "status": "passed" if passed else "failed",
        "scientific_evidence": False,
        "real_llm_api_called": False,
        "full_discovery": full,
        "controlled_recovery": controlled,
        **{
            name: controlled[name]
            for name in (
                "micro_to_meso_testable_edges",
                "meso_to_macro_testable_edges",
                "supported_micro_to_meso",
                "supported_meso_to_macro",
                "unmapped_meso_to_macro",
                "complete_testable_paths",
                "complete_supported_paths",
            )
        },
    }
    path = run_root / "analysis" / "dev_stage3_audit.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"development Stage 3 path-aware audit failed: {audit}")
    return audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--plot-repo", type=Path, default=None)
    args = parser.parse_args()
    config_path = PROJECT_ROOT / "config" / "dev_experiment.json"
    llm_path = PROJECT_ROOT / "config" / "llm_api.mock.json"
    config = load_experiment_config(config_path)
    llm = load_llm_config(llm_path, require_key=False)
    manager = RunManager.initialise(
        PROJECT_ROOT, args.run_id, config, llm, resume=False,
        output_family="dev_runs",
    )
    (manager.run_root / "NON_SCIENTIFIC.json").write_text(
        json.dumps({"scientific_output": False, "real_llm_api_called": False}, indent=2),
        encoding="utf-8",
    )
    run_selected_stages(
        ["semantic", "baseline_simulation"], manager, args.workers,
        PROJECT_ROOT / "config" / "semantic_prompt.txt", llm_path,
        no_render=False, plot_repo=args.plot_repo,
        completion_provider=mock_completion_provider,
    )
    resumed = RunManager.initialise(
        PROJECT_ROOT, args.run_id, config, llm, resume=True,
        output_family="dev_runs",
    )
    run_selected_stages(
        STAGE_ORDER[2:-1], resumed, args.workers,
        PROJECT_ROOT / "config" / "semantic_prompt.txt", llm_path,
        no_render=False, plot_repo=args.plot_repo,
        completion_provider=mock_completion_provider,
    )
    audit = write_dev_stage3_audit(resumed.run_root)
    run_selected_stages(
        ["render"], resumed, args.workers,
        PROJECT_ROOT / "config" / "semantic_prompt.txt", llm_path,
        no_render=False, plot_repo=args.plot_repo,
        completion_provider=mock_completion_provider,
    )
    print(json.dumps(audit, indent=2))
    print(resumed.run_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
