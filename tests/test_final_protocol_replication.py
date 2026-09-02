from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from emergence_attribution.llm_client import LLMResponse
from emergence_attribution.mock_semantic import mock_generation
from emergence_attribution.pipeline import load_experiment_config
from emergence_attribution.semantic import run_semantic_stage


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _provider(replication_suffix: str):
    def provider(scenario: str, generation_index: int):
        payload = mock_generation(scenario)
        if generation_index >= 2:
            payload["representation"]["interpretation_boundary"] += replication_suffix
        text = json.dumps(payload, ensure_ascii=False)

        def complete(_system: str, _user: str) -> LLMResponse:
            return LLMResponse(
                text=text,
                input_tokens=0,
                output_tokens=0,
                model=f"replication-test-{generation_index}",
            )

        return complete

    return provider


def _run(tmp_path: Path, name: str, suffix: str) -> Path:
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    run_root = tmp_path / name
    run_semantic_stage(
        config,
        PROJECT_ROOT / "config" / "llm_api.mock.json",
        run_root,
        PROJECT_ROOT / "config" / "semantic_prompt.txt",
        workers=2,
        completion_provider=_provider(suffix),
    )
    return run_root


def test_formal_semantic_generation_roles_are_frozen_three_plus_three() -> None:
    config = json.loads(
        (PROJECT_ROOT / "config" / "experiment.json").read_text(encoding="utf-8")
    )
    assert config["semantic_replication"] == {
        "selection_generations": 3,
        "replication_only_generations": 3,
    }


def test_replication_only_generation_cannot_change_selected_representation(
    tmp_path: Path,
) -> None:
    first = _run(tmp_path, "first", " replication variant A")
    second = _run(tmp_path, "second", " replication variant B")
    for scenario in ("schelling", "deffuant"):
        first_representation = json.loads(
            (first / "representation" / f"{scenario}_representation.json").read_text(
                encoding="utf-8"
            )
        )
        second_representation = json.loads(
            (second / "representation" / f"{scenario}_representation.json").read_text(
                encoding="utf-8"
            )
        )
        assert first_representation == second_representation
        validation = json.loads(
            (first / "representation" / "representation_validation.json").read_text(
                encoding="utf-8"
            )
        )[scenario]
        assert validation["selected_generation"] < 2
        assert [item["selection_eligible"] for item in validation["all_generations"]] == [
            True,
            True,
            False,
        ]
    pairwise = pd.read_csv(first / "representation" / "replication_pairwise.csv")
    assert len(pairwise) == 6
    assert set(pairwise["comparison_group"]) == {
        "within_selection",
        "selection_vs_replication",
    }
    agreement = json.loads(
        (first / "representation" / "replication_agreement.json").read_text(
            encoding="utf-8"
        )
    )
    assert agreement["hidden_truth_used"] is False
    assert "scale_assignment_jaccard" in pairwise.columns
    assert "source_family_jaccard" in pairwise.columns
