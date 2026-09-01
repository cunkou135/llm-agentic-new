from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from emergence_attribution.exporting import create_visualization_bundle
from emergence_attribution.llm_client import redacted_llm_config
from emergence_attribution.provenance import RunContractError, RunManager


def _contract_config() -> dict:
    return {
        "random_seeds": [1],
        "scenarios": {"toy": {"baseline": {}, "interventions": {}}},
        "representation": {},
        "temporal": {},
        "intervention": {},
        "robustness": {},
        "render": {},
        "formal_run": False,
    }


def _llm_config() -> dict:
    return {
        "base_url": "https://example.invalid/v1",
        "api_key": "secret-value",
        "model": "test-model",
        "temperature": 0.1,
        "max_tokens": 100,
        "timeout": 10,
        "max_retries": 0,
    }


def test_api_key_redaction() -> None:
    redacted = redacted_llm_config(_llm_config())
    assert redacted["api_key"] == "***REDACTED***"
    assert "secret-value" not in json.dumps(redacted)
    assert redacted["model"] == "test-model"


def test_resume_checkpoint_verifies_output_hash(tmp_path: Path) -> None:
    project = tmp_path / "project"
    project.mkdir()
    (project / "source.py").write_text("VALUE = 1\n", encoding="utf-8")
    manager = RunManager.initialise(
        project, "test_run", _contract_config(), _llm_config(), resume=False
    )
    output = manager.run_root / "analysis" / "value.txt"
    output.write_text("stable\n", encoding="utf-8")
    manager.mark_stage_completed("toy", [output], 0.1)
    resumed = RunManager.initialise(
        project, "test_run", _contract_config(), _llm_config(), resume=True
    )
    assert resumed.stage_complete("toy")
    output.write_text("changed\n", encoding="utf-8")
    with pytest.raises(RunContractError, match="modified output"):
        resumed.stage_complete("toy")


def _write_visual_inputs(run_root: Path) -> None:
    analysis = run_root / "analysis"
    analysis.mkdir(parents=True)
    data = run_root / "data" / "raw_logs" / "toy" / "baseline"
    data.mkdir(parents=True)
    raw_path = data / "seed_1.npz"
    np.savez_compressed(
        raw_path,
        state_grid=np.zeros((3, 2, 2), dtype=np.int8),
        unhappy_count=np.zeros(3, dtype=np.int32),
        agent_count=np.asarray([2], dtype=np.int32),
    )
    (run_root / "data" / "simulation_manifest.json").write_text(
        json.dumps(
            {
                "task_records": [
                    {
                        "scenario": "toy",
                        "condition": "baseline",
                        "seed": 1,
                        "raw_path": "data/raw_logs/toy/baseline/seed_1.npz",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    (analysis / "main_graphs.jsonl").write_text(
        json.dumps({"scenario": "toy", "method": "full_method", "edges": []}) + "\n",
        encoding="utf-8",
    )
    pd.DataFrame(
        [
            {
                "scenario": "toy",
                "method": "full_method",
                "edge_f1": 0.0,
                "shd": 0.0,
                "stability": 0.0,
                "lag_mae": np.nan,
            }
        ]
    ).to_csv(analysis / "main_results.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "toy",
                "method": "full_method",
                "trajectory_count": 1,
                "repetition": 0,
                "edge_f1": 0.0,
                "stability": 0.0,
                "edge_f1_ci_low": 0.0,
                "edge_f1_ci_high": 0.0,
                "stability_ci_low": 0.0,
                "stability_ci_high": 0.0,
            }
        ]
    ).to_csv(analysis / "data_efficiency_repeated_subsampling.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "toy",
                "parameter": "p",
                "direction": "plus",
                "node_id": "n",
                "time": 0,
                "mean": 0.0,
                "ci_low": -0.1,
                "ci_high": 0.1,
            }
        ]
    ).to_parquet(analysis / "effect_curves.parquet", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "toy",
                "parameter": "p",
                "direction": "plus",
                "node_id": "n",
                "scale": "micro",
                "cumulative_effect": 0.0,
                "significant": False,
            }
        ]
    ).to_parquet(analysis / "paired_effects.parquet", index=False)
    timing = {
        "scenario": "toy",
        "path_id": "path",
        "parameter": "p",
        "direction": "plus",
        "source": "a",
        "meso": "b",
        "macro": "c",
        "scale": "micro",
        "onset_time": -1,
        "observational_lag": 1,
        "lag_difference": np.nan,
        "cumulative_effect": 0.0,
        "significant": False,
    }
    pd.DataFrame([timing]).to_csv(analysis / "path_timing_summary.csv", index=False)
    (analysis / "representative_path_selection.json").write_text("{}", encoding="utf-8")
    pd.DataFrame(
        [
            {
                "scenario": "toy",
                "source": "a",
                "target": "b",
                "parameter": "p",
                "direction": "plus",
                "manipulation_success": False,
                "primary_class": "inconclusive",
                "underlying_class": "inconclusive",
            }
        ]
    ).to_csv(analysis / "intervention_classifications.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "toy",
                "factor": "observation_noise",
                "noise_level": 0.0,
                "missing_fraction": 0.0,
                "support_threshold": 0.65,
                "repetition": 0,
                "edge_f1": 0.0,
                "stability": 0.0,
                "retained_edge_count": 0,
                "intervention_f1": np.nan,
            }
        ]
    ).to_csv(analysis / "observation_robustness.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "toy",
                "candidate_indicator_count": 1,
                "repetition": 0,
                "runtime_seconds": 0.0,
                "discovered_edge_count": 0,
            }
        ]
    ).to_csv(analysis / "causal_scalability.csv", index=False)
    pd.DataFrame(
        [
            {
                "scenario": "toy",
                "prediction_id": "pred",
                "classification": "inconclusive",
            }
        ]
    ).to_csv(analysis / "prospective_validation.csv", index=False)


def test_visualization_bundle_uses_dynamic_schema(tmp_path: Path) -> None:
    run_root = tmp_path / "runs" / "example"
    _write_visual_inputs(run_root)
    manifest = create_visualization_bundle(run_root)
    assert manifest["dynamic_row_counts"] is True
    assert set(manifest["figures"]) == {"02", "03", "04", "05", "06", "07", "08"}
    generated = json.loads(
        (run_root / "visualization_input" / "figure_inputs.generated.json").read_text(
            encoding="utf-8"
        )
    )
    assert generated["source_run"] == "example"

