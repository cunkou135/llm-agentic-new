"""Contracts for the two-phase, path-centered formal protocol."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from emergence_attribution.interventions import (
    classify_candidate_paths,
    classify_edge_interventions,
    path_timing_concordance,
    qualify_candidate_paths,
)
from emergence_attribution.mock_semantic import (
    mock_completion_provider,
    mock_indicator_generation,
    mock_path_generation,
)
from emergence_attribution.pipeline import load_experiment_config
from emergence_attribution.schemas import (
    IndicatorGeneration,
    PathGeneration,
    ProspectivePrediction,
)
from emergence_attribution.semantic import (
    derive_candidate_edges,
    freeze_indicator_stage,
    load_frozen_representations,
    run_indicator_generation_stage,
    run_path_generation_stage,
    sha256_json,
    validate_indicator_generation,
    validate_path_generation,
)
from emergence_attribution.temporal import TemporalEdge


def _config() -> dict:
    return load_experiment_config(Path("config/dev_experiment.json"))


def _indicator_and_path(scenario: str = "schelling"):
    config = _config()
    indicator_payload = mock_indicator_generation(scenario)
    indicator = IndicatorGeneration.model_validate(indicator_payload)
    path = PathGeneration.model_validate(mock_path_generation(scenario))
    return config, indicator_payload, indicator, path


@pytest.mark.parametrize("forbidden", ["candidate_edges", "candidate_paths", "prospective_predictions"])
def test_indicator_generation_rejects_relationship_outputs(forbidden: str) -> None:
    payload = mock_indicator_generation("schelling")
    payload[forbidden] = []
    with pytest.raises(ValidationError):
        IndicatorGeneration.model_validate(payload)


def test_indicator_generation_has_exact_budget_without_group_field() -> None:
    config, _, indicator, _ = _indicator_and_path()
    result = validate_indicator_generation(
        indicator, "schelling", config["scenarios"]["schelling"],
        config["representation"],
    )
    assert result["valid"]
    assert result["scale_counts"] == {"micro": 16, "meso": 8, "macro": 4}
    assert "branch_id" not in IndicatorGeneration.model_json_schema()["$defs"]["IndicatorSpec"]["properties"]


def test_indicator_hash_is_stable_and_path_input_matches_it() -> None:
    _, payload, indicator, path = _indicator_and_path()
    canonical = indicator.model_dump(mode="json", exclude_none=True)
    assert sha256_json(canonical) == sha256_json(deepcopy(canonical))
    assert path.indicator_set_sha256 == sha256_json(canonical)


def test_two_semantic_stages_are_separate_and_tampering_fails_closed(
    tmp_path: Path,
) -> None:
    config = _config()
    llm_config = Path("config/llm_api.mock.json")
    prompt = Path("config/semantic_prompt.txt")
    run_indicator_generation_stage(
        config, llm_config, tmp_path, prompt, 1,
        completion_provider=mock_completion_provider,
    )
    representation_root = tmp_path / "representation"
    assert (representation_root / "indicator_selection.json").is_file()
    assert not (representation_root / "candidate_paths.json").exists()
    freeze_indicator_stage(config, tmp_path)
    assert (representation_root / "INDICATORS_FROZEN.sha256").is_file()
    run_path_generation_stage(
        config, llm_config, tmp_path, prompt, 1,
        completion_provider=mock_completion_provider,
    )
    assert len(load_frozen_representations(tmp_path)) == 2
    path_file = representation_root / "candidate_paths.json"
    payload = json.loads(path_file.read_text(encoding="utf-8"))
    payload["scenarios"]["schelling"][0]["path_id"] = "tampered_path"
    path_file.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="hash mismatch"):
        load_frozen_representations(tmp_path)


def test_unknown_frozen_indicator_is_rejected() -> None:
    config, payload, indicator, path = _indicator_and_path()
    changed = path.model_dump(mode="json")
    changed["candidate_paths"][0]["micro_indicator"] = "unknown_observable"
    value = PathGeneration.model_validate(changed)
    result = validate_path_generation(
        value, indicator, sha256_json(indicator.model_dump(mode="json", exclude_none=True)),
        config["scenarios"]["schelling"], config["representation"],
    )
    assert not result["valid"]
    assert any("unknown frozen indicator" in item for item in result["errors"])


def test_wrong_scale_path_reference_is_rejected() -> None:
    config, _, indicator, path = _indicator_and_path()
    changed = path.model_dump(mode="json")
    changed["candidate_paths"][0]["micro_indicator"] = changed["candidate_paths"][0]["meso_indicator"]
    value = PathGeneration.model_validate(changed)
    result = validate_path_generation(
        value, indicator, sha256_json(indicator.model_dump(mode="json", exclude_none=True)),
        config["scenarios"]["schelling"], config["representation"],
    )
    assert not result["valid"]
    assert any("Micro-Meso-Macro" in item for item in result["errors"])


def test_path_requires_real_parameter_and_direct_micro_association() -> None:
    config, _, indicator, path = _indicator_and_path()
    changed = path.model_dump(mode="json")
    changed["candidate_paths"][0]["parameter"] = "invented_parameter"
    value = PathGeneration.model_validate(changed)
    result = validate_path_generation(
        value, indicator, sha256_json(indicator.model_dump(mode="json", exclude_none=True)),
        config["scenarios"]["schelling"], config["representation"],
    )
    assert not result["valid"]
    assert any("unknown controllable parameter" in item for item in result["errors"])


def test_duplicate_indicator_triple_is_rejected() -> None:
    payload = mock_path_generation("schelling")
    payload["candidate_paths"][1]["micro_indicator"] = payload["candidate_paths"][0]["micro_indicator"]
    payload["candidate_paths"][1]["meso_indicator"] = payload["candidate_paths"][0]["meso_indicator"]
    payload["candidate_paths"][1]["macro_indicator"] = payload["candidate_paths"][0]["macro_indicator"]
    with pytest.raises(ValidationError, match="duplicate indicator triple"):
        PathGeneration.model_validate(payload)


def test_formal_path_capacity_and_coverage_contract() -> None:
    config, _, indicator, path = _indicator_and_path()
    result = validate_path_generation(
        path, indicator, sha256_json(indicator.model_dump(mode="json", exclude_none=True)),
        config["scenarios"]["schelling"], config["representation"],
    )
    assert result["valid"]
    assert 16 <= result["candidate_path_count"] <= 24
    assert min(result["parameter_path_coverage"].values()) >= 4
    assert min(result["macro_path_coverage"].values()) >= 2


def test_derived_edges_deduplicate_shared_meso_macro_edge() -> None:
    paths = [
        {
            "path_id": "path_one", "micro_indicator": "micro_one",
            "meso_indicator": "meso_one", "macro_indicator": "macro_one",
            "micro_to_meso_expected_direction": "increase",
            "meso_to_macro_expected_direction": "increase",
        },
        {
            "path_id": "path_two", "micro_indicator": "micro_two",
            "meso_indicator": "meso_one", "macro_indicator": "macro_one",
            "micro_to_meso_expected_direction": "increase",
            "meso_to_macro_expected_direction": "increase",
        },
    ]
    edges = derive_candidate_edges(paths)
    assert {(item["source"], item["target"]) for item in edges} == {
        ("micro_one", "meso_one"), ("micro_two", "meso_one"),
        ("meso_one", "macro_one"),
    }


def _edge(source: str, target: str, group: str) -> TemporalEdge:
    return TemporalEdge(
        source=source, target=target, lag=1, beta=0.4, p_value=0.01,
        q_value=0.02, effect_direction="increase", support=0.8,
        lag_support=0.8, lag_std=0.0, hypothesis_group_id=group,
    )


def _one_path_representation() -> dict:
    path = mock_path_generation("schelling")["candidate_paths"][0]
    return {"candidate_paths": [path]}


def test_path_temporal_qualification_requires_both_group_specific_edges() -> None:
    representation = _one_path_representation()
    path = representation["candidate_paths"][0]
    group = f"macro_outcome_{path['macro_indicator']}"
    graph = [
        _edge(path["micro_indicator"], path["meso_indicator"], group),
        _edge(path["meso_indicator"], path["macro_indicator"], group),
    ]
    qualified = qualify_candidate_paths("schelling", graph, representation)
    assert bool(qualified.iloc[0]["path_temporally_qualified"])
    wrong_group = [graph[0], _edge(path["meso_indicator"], path["macro_indicator"], "other")]
    rejected = qualify_candidate_paths("schelling", wrong_group, representation)
    assert not bool(rejected.iloc[0]["path_temporally_qualified"])


def _stage3_v2_result(
    *,
    effects: tuple[float, float, float] = (0.5, 0.4, 0.3),
    significant: tuple[bool, bool, bool] = (True, True, True),
    onsets: tuple[int, int, int] = (0, 0, 5),
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    path = {
        "path_id": "path_v2", "parameter": "theta",
        "intervention_direction": "plus", "micro_indicator": "micro_a",
        "meso_indicator": "meso_b", "macro_indicator": "macro_c",
        "expected_micro_response": "increase",
        "expected_meso_response": "increase",
        "expected_macro_response": "increase",
        "micro_to_meso_expected_direction": "increase",
        "meso_to_macro_expected_direction": "increase",
    }
    representation = {
        "indicators": [
            {
                "id": "micro_a", "scale": "micro",
                "parameter_associations": [
                    {"parameter": "theta", "relationship": "direct"}
                ],
            },
            {"id": "meso_b", "scale": "meso", "parameter_associations": []},
            {"id": "macro_c", "scale": "macro", "parameter_associations": []},
        ],
        "candidate_paths": [path],
    }
    group = "macro_outcome_macro_c"
    graph = [
        TemporalEdge(
            source="micro_a", target="meso_b", lag=4, beta=-0.4,
            p_value=0.01, q_value=0.02, effect_direction="decrease",
            support=0.8, lag_support=0.8, lag_std=0.0,
            hypothesis_group_id=group,
        ),
        TemporalEdge(
            source="meso_b", target="macro_c", lag=1, beta=-0.4,
            p_value=0.01, q_value=0.02, effect_direction="decrease",
            support=0.8, lag_support=0.8, lag_std=0.0,
            hypothesis_group_id=group,
        ),
    ]
    effect_frame = pd.DataFrame(
        [
            {
                "scenario": "toy", "parameter": "theta", "direction": "plus",
                "node_id": node, "cumulative_effect_standardised": effect,
                "onset_time": onset, "significant": is_significant,
            }
            for node, effect, onset, is_significant in zip(
                ("micro_a", "meso_b", "macro_c"), effects, onsets, significant
            )
        ]
    )
    qualification = qualify_candidate_paths("toy", graph, representation)
    edge_classification = classify_edge_interventions(
        "toy", graph, effect_frame, representation, lag_tolerance=2
    )
    edge_classification.insert(1, "method", "full_method")
    path_classification = classify_candidate_paths(
        qualification, edge_classification, {"toy": representation}
    )
    concordance = path_timing_concordance(
        qualification, effect_frame, {"toy": representation},
        path_classification, lag_tolerance=2,
    )
    return edge_classification, path_classification, concordance


def test_stage3_v2_lag_and_beta_mismatch_are_auxiliary() -> None:
    edges, paths, concordance = _stage3_v2_result()
    plus = edges[edges["direction"] == "plus"]
    assert (plus["primary_class"] == "supported").all()
    assert not plus["temporal_direction_concordant"].astype(bool).any()
    assert not plus["lag_concordant"].astype(bool).any()
    assert paths.iloc[0]["path_classification"] == "supported"
    row = concordance.iloc[0]
    assert not bool(row["micro_meso_lag_concordant"])
    assert not bool(row["meso_macro_lag_concordant"])
    assert row["observational_total_lag"] == 5
    assert row["intervention_total_latency"] == 5


def test_stage3_v2_frozen_macro_response_contradiction() -> None:
    _, paths, _ = _stage3_v2_result(effects=(0.5, 0.4, -0.3))
    assert paths.iloc[0]["path_classification"] == "contradicted"
    assert paths.iloc[0]["reason"] == "frozen_response_direction_contradicted"


def test_stage3_v2_requires_stable_response_at_all_scales() -> None:
    _, paths, _ = _stage3_v2_result(significant=(True, False, True))
    assert paths.iloc[0]["path_classification"] == "inconclusive"
    assert paths.iloc[0]["reason"] == "required_multiscale_response_not_stable"


def test_stage3_v2_requires_ordered_multiscale_onsets() -> None:
    _, paths, _ = _stage3_v2_result(onsets=(2, 5, 3))
    assert paths.iloc[0]["path_classification"] == "inconclusive"
    assert paths.iloc[0]["reason"] == "intervention_onset_order_not_supported"


def test_path_intervention_evidence_isolated_by_macro_hypothesis_group() -> None:
    paths = [
        {
            "path_id": "path_c", "parameter": "theta",
            "intervention_direction": "plus", "micro_indicator": "micro_a",
            "meso_indicator": "meso_b", "macro_indicator": "macro_c",
            "expected_micro_response": "increase",
            "expected_meso_response": "increase",
            "expected_macro_response": "increase",
            "micro_to_meso_expected_direction": "increase",
            "meso_to_macro_expected_direction": "increase",
        },
        {
            "path_id": "path_d", "parameter": "theta",
            "intervention_direction": "plus", "micro_indicator": "micro_a",
            "meso_indicator": "meso_b", "macro_indicator": "macro_d",
            "expected_micro_response": "increase",
            "expected_meso_response": "increase",
            "expected_macro_response": "increase",
            "micro_to_meso_expected_direction": "increase",
            "meso_to_macro_expected_direction": "increase",
        },
    ]
    qualification = pd.DataFrame(
        [
            {"scenario": "toy", "path_id": path["path_id"],
             "path_temporally_qualified": True}
            for path in paths
        ]
    )
    common = {
        "scenario": "toy", "method": "full_method", "root_source": "micro_a",
        "parameter": "theta", "direction": "plus", "manipulation_success": True,
        "target_significant": True, "root_onset": 0,
        "root_effect": 0.5, "source_effect": 0.5,
    }
    classifications = pd.DataFrame(
        [
            {**common, "hypothesis_group_id": "macro_outcome_macro_c",
             "source": "micro_a", "target": "meso_b", "primary_class": "supported",
             "target_effect": 0.4, "source_onset": 0, "target_onset": 1},
            {**common, "hypothesis_group_id": "macro_outcome_macro_c",
             "source": "meso_b", "target": "macro_c", "primary_class": "supported",
             "target_effect": 0.3, "source_onset": 1, "target_onset": 2},
            {**common, "hypothesis_group_id": "macro_outcome_macro_d",
             "source": "micro_a", "target": "meso_b",
             "primary_class": "directionally_contradicted", "target_effect": -0.4,
             "source_onset": 0, "target_onset": 1},
            {**common, "hypothesis_group_id": "macro_outcome_macro_d",
             "source": "meso_b", "target": "macro_d", "primary_class": "supported",
             "target_effect": 0.3, "source_onset": 1, "target_onset": 2},
        ]
    )

    result = classify_candidate_paths(
        qualification, classifications, {"toy": {"candidate_paths": paths}}
    ).set_index("path_id")

    assert result.loc["path_c", "path_classification"] == "supported"
    assert result.loc["path_d", "path_classification"] == "contradicted"


def test_prospective_prediction_cannot_invent_path() -> None:
    payload = mock_path_generation("schelling")
    payload["prospective_predictions"][0]["candidate_path_id"] = "not_frozen"
    with pytest.raises(ValidationError, match="unknown path"):
        PathGeneration.model_validate(payload)


def test_prospective_schema_requires_candidate_path_id() -> None:
    with pytest.raises(ValidationError):
        ProspectivePrediction.model_validate(
            {
                "prediction_id": "prediction_one",
                "scientific_rationale": "Data-blind prospective statement.",
                "falsification_condition": "The frozen path is not supported.",
            }
        )


def test_formal_thresholds_and_new_seed_pools_are_frozen() -> None:
    from pathlib import Path

    config = load_experiment_config(Path("config/experiment.json"))
    assert config["random_seeds"] == list(range(3101, 3125))
    assert config["confirmation_seeds"] == list(range(4101, 4113))
    assert config["temporal"] == {
        "maximum_lag": 5, "parent_alpha": 0.10, "fdr_alpha": 0.05,
        "bootstrap_repetitions": 100, "support_threshold": 0.65,
        "vote_threshold": 0.50,
    }
    assert config["intervention"]["onset_detection_start"] == 0
    assert config["intervention"]["onset_consecutive_steps"] == 4
