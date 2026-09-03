"""Run the isolated NON_SCIENTIFIC development pipeline with resume coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
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
from emergence_attribution.interventions import (  # noqa: E402
    aggregate_edge_intervention_evidence,
    eligible_propagation_path_ids,
    graph_paths,
)
from emergence_attribution.dsl import compute_indicator  # noqa: E402
from emergence_attribution.raw_schemas import public_raw_schema  # noqa: E402
from emergence_attribution.reference_truth import (  # noqa: E402
    mechanism_target_for_variant,
    reference_processes,
    reference_relations,
)
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
        edge_level = aggregate_edge_intervention_evidence(subset)
        applicable = edge_level[edge_level["edge_class"] != "not_applicable"]
        testable_pairs = {
            (str(row.source), str(row.target)) for row in applicable.itertuples()
        }
        supported_pairs = {
            (str(row.source), str(row.target))
            for row in applicable.itertuples()
            if row.edge_class == "supported"
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
    """Validate software/data contracts only; scientific outcomes are irrelevant."""

    representations = load_frozen_representations(run_root)
    config = json.loads(
        (run_root / "config" / "experiment_config.snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    temporal_paths = pd.read_csv(
        run_root / "analysis" / "path_temporal_qualification.csv"
    )
    intervention_paths = pd.read_csv(
        run_root / "analysis" / "path_intervention_classification.csv"
    )
    prospective = json.loads(
        (run_root / "representation" / "prospective_predictions.json").read_text(
            encoding="utf-8"
        )
    )["scenarios"]
    holdout = pd.read_csv(run_root / "analysis" / "holdout_path_confirmation.csv")
    checks: dict[str, bool] = {}
    checks["indicator_budget_16_8_4"] = all(
        {
            scale: sum(item["scale"] == scale for item in rep["indicators"])
            for scale in ("micro", "meso", "macro")
        } == {"micro": 16, "meso": 8, "macro": 4}
        for rep in representations.values()
    )
    checks["candidate_path_count_16_24"] = all(
        16 <= len(rep["candidate_paths"]) <= 24
        for rep in representations.values()
    )
    checks["all_path_ids_frozen"] = True
    checks["derived_edges_exact"] = True
    checks["prospective_bound_to_paths"] = True
    for scenario, rep in representations.items():
        indicator_ids = {item["id"] for item in rep["indicators"]}
        path_ids = {item["path_id"] for item in rep["candidate_paths"]}
        checks["all_path_ids_frozen"] &= all(
            {
                path["micro_indicator"], path["meso_indicator"],
                path["macro_indicator"],
            }.issubset(indicator_ids)
            for path in rep["candidate_paths"]
        )
        derived_pairs = {
            (edge["source"], edge["target"]) for edge in rep["candidate_edges"]
        }
        expected_pairs = {
            pair
            for path in rep["candidate_paths"]
            for pair in (
                (path["micro_indicator"], path["meso_indicator"]),
                (path["meso_indicator"], path["macro_indicator"]),
            )
        }
        checks["derived_edges_exact"] &= derived_pairs == expected_pairs
        checks["prospective_bound_to_paths"] &= all(
            item["candidate_path_id"] in path_ids
            for item in prospective[scenario]
        )
    checks["path_temporal_output_complete"] = len(temporal_paths) == sum(
        len(rep["candidate_paths"]) for rep in representations.values()
    )
    checks["path_intervention_output_complete"] = len(intervention_paths) == sum(
        len(rep["candidate_paths"]) for rep in representations.values()
    )
    checks["path_classification_closed_set"] = set(
        intervention_paths["path_classification"].astype(str)
    ).issubset({"supported", "contradicted", "inconclusive", "manipulation_failure"})
    checks["holdout_no_replacement_path"] = set(holdout["path_id"].astype(str)).issubset(
        set(intervention_paths["path_id"].astype(str))
    )
    checks["secondary_outputs_present"] = all(
        (run_root / relative).is_file()
        for relative in (
            "analysis/path_dose_response_summary.csv",
            "analysis/path_temporal_negative_control.csv",
            "analysis/path_funnel_summary.csv",
            "analysis/attribution_objects.json",
        )
    )
    checks["thresholds_unchanged"] = bool(
        config["temporal"]["maximum_lag"] == 5
        and config["temporal"]["parent_alpha"] == 0.10
        and config["temporal"]["fdr_alpha"] == 0.05
        and config["temporal"]["support_threshold"] == 0.65
        and config["intervention"]["onset_detection_start"] == 0
        and config["intervention"]["onset_consecutive_steps"] == 4
        and config["intervention"]["lag_tolerance"] == 2
    )
    passed = all(checks.values())
    audit = {
        "status": "passed" if passed else "failed",
        "scientific_evidence": False,
        "real_llm_api_called": False,
        "formal_experiment_executed": False,
        "supported_path_required_for_dev_pass": False,
        "supported_path_count": int(
            (intervention_paths["path_classification"] == "supported").sum()
        ),
        "contract_checks": checks,
    }
    path = run_root / "analysis" / "dev_stage3_audit.json"
    path.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    if not passed:
        raise RuntimeError(f"development path-centered contract audit failed: {audit}")
    return audit

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
    config = json.loads(
        (run_root / "config" / "experiment_config.snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    mechanism_frame = pd.read_csv(
        run_root / "analysis" / "mechanism_disabled_checks.csv"
    )
    mechanism_expected = {
        scenario: mechanism_target_for_variant(
            scenario, config["scenarios"][scenario]["mechanism_variant"]
        )
        for scenario in controlled_representations
    }
    mechanism_alignment = True
    for scenario, target in mechanism_expected.items():
        row = mechanism_frame[mechanism_frame["scenario"] == scenario].iloc[0]
        target_edges = [
            edge for edge in reference_relations(scenario) if edge.mechanism == target
        ]
        public_source = next(
            item for item in reference_processes(scenario)
            if item.process_id == target_edges[0].source
        )
        expected_field = (
            "destination_similarity" if scenario == "schelling"
            else "interaction_backfire"
        )
        mechanism_alignment &= bool(
            row["targeted_mechanism"] == target
            and len(target_edges) == 2
            and expected_field in json.dumps(public_source.computation)
        )

    representation_robustness = pd.read_csv(
        run_root / "analysis" / "representation_robustness.csv"
    )
    positive_error = representation_robustness[
        representation_robustness["error_ratio"] > 0
    ]
    repetition_checks = positive_error.groupby(
        ["scenario", "operator", "error_ratio"], dropna=False
    ).agg(
        repetitions=("repetition", "nunique"),
        candidate_sets=("candidate_set_sha256", "nunique"),
    )
    random_repetitions = bool(
        len(repetition_checks)
        and (repetition_checks["repetitions"] >= 2).all()
        and (
            repetition_checks["candidate_sets"]
            == repetition_checks["repetitions"]
        ).all()
    )

    efficiency = pd.read_csv(
        run_root / "analysis" / "data_efficiency_repeated_subsampling.csv"
    )
    n1 = efficiency[efficiency["trajectory_count"] == 1]
    n1_stability = bool(
        len(n1)
        and not n1["stability_estimable"].astype(bool).any()
        and n1[["stability", "lag_support", "lag_std", "stability_ci_low", "stability_ci_high"]]
        .isna().all().all()
    )

    predictions = json.loads(
        (run_root / "representation" / "prospective_predictions.json").read_text(
            encoding="utf-8"
        )
    )
    prospective_exact = True
    for scenario, scenario_predictions in predictions["scenarios"].items():
        candidates = {
            (edge["source"], edge["target"])
            for edge in full_representations[scenario]["candidate_edges"]
        }
        for prediction in scenario_predictions:
            path = prediction["expected_temporal_order"]
            required = [
                (edge["source"], edge["target"])
                for edge in prediction["validation_criteria"]["required_candidate_edges"]
            ]
            adjacent = list(zip(path, path[1:]))
            prospective_exact &= bool(
                len(required) == len(adjacent)
                and set(required) == set(adjacent)
                and set(required).issubset(candidates)
            )

    expression = {
        "op": "network_assortativity",
        "values": {"op": "field", "name": "state_opinion"},
        "edges": {"op": "field", "name": "network_edges"},
    }
    values = np.asarray([[0.1, 0.5, -0.4, 0.9], [0.2, -0.2, 0.8, 0.4]])
    edges = np.asarray([[0, 1], [0, 3], [1, 2]], dtype=int)
    permutation = np.asarray([2, 0, 3, 1])
    relabelled_values = np.empty_like(values)
    relabelled_values[:, permutation] = values
    original_assortativity = compute_indicator(
        expression, {"op": "identity"},
        {"state_opinion": values, "network_edges": edges},
        public_raw_schema("deffuant"),
    )
    relabelled_assortativity = compute_indicator(
        expression, {"op": "identity"},
        {
            "state_opinion": relabelled_values,
            "network_edges": permutation[edges],
        },
        public_raw_schema("deffuant"),
    )
    assortativity_invariant = bool(
        np.allclose(original_assortativity, relabelled_assortativity)
    )

    controlled_attempts = pd.read_csv(
        run_root / "analysis" / "controlled_recovery_intervention_classifications.csv"
    )
    stored_edge_evidence = pd.read_csv(
        run_root / "analysis" / "controlled_recovery_edge_intervention_classifications.csv"
    )
    recomputed_edge_evidence = aggregate_edge_intervention_evidence(controlled_attempts)
    edge_evidence_matches = bool(
        stored_edge_evidence.sort_values(
            ["scenario", "method", "source", "target"]
        ).reset_index(drop=True).equals(
            recomputed_edge_evidence.sort_values(
                ["scenario", "method", "source", "target"]
            ).reset_index(drop=True)
        )
    )
    precedence_probe = aggregate_edge_intervention_evidence(
        pd.DataFrame(
            [
                {"scenario": "probe", "source": "a", "target": "b", "primary_class": "supported"},
                {"scenario": "probe", "source": "a", "target": "b", "primary_class": "directionally_contradicted"},
            ]
        )
    )
    edge_aggregation = bool(
        edge_evidence_matches
        and precedence_probe.iloc[0]["edge_class"] == "directionally_contradicted"
    )

    timing = pd.read_csv(run_root / "analysis" / "path_timing_summary.csv")
    attempts = pd.read_csv(run_root / "analysis" / "intervention_classifications.csv")
    eligible_paths = eligible_propagation_path_ids(timing, attempts)
    selected_paths = json.loads(
        (run_root / "analysis" / "representative_path_selection.json").read_text(
            encoding="utf-8"
        )
    )
    figure7_filter = all(
        item.get("path_id") is None or str(item["path_id"]) in eligible_paths
        for item in selected_paths.get("scenarios", {}).values()
    )
    figures = {
        name: (run_root / "figures" / name).is_file()
        for name in (
            "figure_4_data_efficiency.png",
            "figure_7_multiscale_propagation.png",
        )
    }
    pool_profile = json.loads(
        (run_root / "analysis" / "robustness_pool_profile.json").read_text(
            encoding="utf-8"
        )
    )
    pool_lifecycle = bool(
        pool_profile["actual_pool_creations"] <= 1
        and pool_profile["nested_pool_creations"] == 0
        and pool_profile["actual_pool_creations"]
        < pool_profile["legacy_estimated_pool_creations"]
    )
    passed = bool(
        set(controlled["scenarios"]) == {"schelling", "deffuant"}
        and controlled["meso_to_macro_testable_edges"] > 0
        and controlled["complete_testable_paths"] > 0
        and mechanism_alignment
        and random_repetitions
        and n1_stability
        and prospective_exact
        and assortativity_invariant
        and edge_aggregation
        and figure7_filter
        and all(figures.values())
        and pool_lifecycle
    )
    audit = {
        "status": "passed" if passed else "failed",
        "scientific_evidence": False,
        "real_llm_api_called": False,
        "full_discovery": full,
        "controlled_recovery": controlled,
        "final_prerun_checks": {
            "mechanism_disabled_semantic_alignment": mechanism_alignment,
            "representation_repetitions_distinct": random_repetitions,
            "single_trajectory_stability_missing": n1_stability,
            "prospective_required_edges_exact": prospective_exact,
            "undirected_assortativity_relabel_invariant": assortativity_invariant,
            "edge_level_contradiction_precedence": edge_aggregation,
            "figure_4_rendered_with_n1_missing": figures["figure_4_data_efficiency.png"],
            "figure_7_complete_ordered_supported_only": bool(
                figures["figure_7_multiscale_propagation.png"] and figure7_filter
            ),
            "robustness_pool_lifecycle": pool_lifecycle,
        },
        "robustness_pool_profile": pool_profile,
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
        STAGE_ORDER[:5], manager, args.workers,
        PROJECT_ROOT / "config" / "semantic_prompt.txt", llm_path,
        no_render=False, plot_repo=args.plot_repo,
        completion_provider=mock_completion_provider,
    )
    resumed = RunManager.initialise(
        PROJECT_ROOT, args.run_id, config, llm, resume=True,
        output_family="dev_runs",
    )
    run_selected_stages(
        STAGE_ORDER[5:-1], resumed, args.workers,
        PROJECT_ROOT / "config" / "semantic_prompt.txt", llm_path,
        no_render=False, plot_repo=args.plot_repo,
        completion_provider=mock_completion_provider,
    )
    run_selected_stages(
        ["render"], resumed, args.workers,
        PROJECT_ROOT / "config" / "semantic_prompt.txt", llm_path,
        no_render=False, plot_repo=args.plot_repo,
        completion_provider=mock_completion_provider,
    )
    audit = write_dev_stage3_audit(resumed.run_root)
    print(json.dumps(audit, indent=2))
    print(resumed.run_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
