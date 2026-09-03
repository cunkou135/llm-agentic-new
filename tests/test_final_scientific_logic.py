from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest

from emergence_attribution.controlled import (
    controlled_intervention_recovery_metrics,
    controlled_representation,
)
from emergence_attribution.dsl import expression_fields
from emergence_attribution.evaluation import (
    candidate_count_for_method,
    evaluate_full_discovery,
    intervention_classification_rates,
    temporal_qualification_rate,
)
from emergence_attribution.exporting import integrate_evidence
from emergence_attribution.interventions import (
    classify_edge_interventions,
    eligible_propagation_path_ids,
    intervention_testable_edges,
)
from emergence_attribution.llm_client import LLMResponse
from emergence_attribution.mock_semantic import mock_generation, mock_indicator_generation
from emergence_attribution.pipeline import load_experiment_config
from emergence_attribution.prospective import classify_prediction_requirements
from emergence_attribution.reference_truth import reference_relations
from emergence_attribution.semantic import run_generation
from emergence_attribution.temporal import TemporalEdge, write_graph_records


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _path_representation() -> dict:
    return {
        "scenario": "toy",
        "indicators": [
            {
                "id": "micro_a", "scale": "micro",
                "parameter_associations": [{
                    "parameter": "theta", "relationship": "direct",
                    "expected_indicator_direction": "increase", "rationale": "rule",
                }],
            },
            {"id": "meso_b", "scale": "meso", "parameter_associations": []},
            {"id": "macro_c", "scale": "macro", "parameter_associations": []},
        ],
        "candidate_paths": [{
            "path_id": "path_toy_01", "parameter": "theta",
            "intervention_direction": "plus", "micro_indicator": "micro_a",
            "meso_indicator": "meso_b", "macro_indicator": "macro_c",
            "micro_to_meso_expected_direction": "increase",
            "meso_to_macro_expected_direction": "increase",
            "expected_micro_response": "increase",
            "expected_meso_response": "increase",
            "expected_macro_response": "increase",
        }],
        "candidate_edges": [
            {"source": "micro_a", "target": "meso_b", "hypothesis_group_ids": ["macro_outcome_macro_c"], "expected_direction": "increase"},
            {"source": "meso_b", "target": "macro_c", "hypothesis_group_ids": ["macro_outcome_macro_c"], "expected_direction": "increase"},
        ],
    }


def _edge(source: str, target: str, lag: int = 1) -> TemporalEdge:
    return TemporalEdge(
        source=source, target=target, lag=lag, beta=0.7, p_value=0.001,
        q_value=0.002, effect_direction="increase", support=0.9,
        lag_support=0.8, lag_std=0.1,
        hypothesis_group_id="macro_outcome_macro_c",
    )


def _effects() -> pd.DataFrame:
    rows = []
    for direction, sign in (("minus", -1), ("plus", 1)):
        for node, onset, magnitude in (
            ("micro_a", 1, 0.8), ("meso_b", 2, 0.6), ("macro_c", 3, 0.5)
        ):
            rows.append({
                "scenario": "toy", "parameter": "theta", "direction": direction,
                "node_id": node, "significant": True, "onset_time": onset,
                "cumulative_effect_standardised": sign * magnitude,
            })
    return pd.DataFrame(rows)


def test_micro_to_meso_intervention_uses_direct_root() -> None:
    graph = [_edge("micro_a", "meso_b"), _edge("meso_b", "macro_c")]
    frame = classify_edge_interventions(
        "toy", graph, _effects(), _path_representation(), 2
    )
    row = frame[(frame["source"] == "micro_a") & (frame["direction"] == "plus")].iloc[0]
    assert row["root_source"] == "micro_a"
    assert row["intervention_scope"] == "direct_root"
    assert row["manipulation_level"] == "micro"


def test_meso_to_macro_intervention_uses_upstream_micro_root() -> None:
    graph = [_edge("micro_a", "meso_b"), _edge("meso_b", "macro_c")]
    frame = classify_edge_interventions(
        "toy", graph, _effects(), _path_representation(), 2
    )
    row = frame[(frame["source"] == "meso_b") & (frame["direction"] == "plus")].iloc[0]
    assert row["root_source"] == "micro_a"
    assert row["parameter"] == "theta"
    assert row["intervention_scope"] == "upstream_mediated"
    assert row["root_onset"] <= row["source_onset"] <= row["target_onset"]


def test_meso_to_macro_is_not_automatically_unmapped() -> None:
    frame = classify_edge_interventions(
        "toy", [_edge("meso_b", "macro_c")], _effects(),
        _path_representation(), 2,
    )
    assert set(frame["intervention_scope"]) == {"upstream_mediated"}
    assert "not_applicable" not in set(frame["primary_class"])


def test_controlled_intervention_recall_uses_only_testable_truth_edges() -> None:
    truth = {("a", "b"), ("b", "c")}
    metrics = controlled_intervention_recovery_metrics(
        truth, {("a", "b")}, {("a", "b")}
    )
    assert metrics["eligible_truth_edge_count"] == 1
    assert metrics["supported_truth_edge_count"] == 1
    assert metrics["intervention_recall"] == 1.0


def test_controlled_direct_parameter_mapping_is_rule_consistent() -> None:
    representation = controlled_representation("schelling")
    mapped = {
        association["parameter"]: item
        for item in representation["indicators"]
        for association in item["parameter_associations"]
        if association["relationship"] == "direct"
    }
    destination = mapped["destination_preference"]
    assert destination["id"] == "s_micro_destination_similarity"
    assert expression_fields(destination["computation"]) == {"destination_similarity"}
    assert "local_similarity" not in expression_fields(destination["computation"])


def test_directionally_contradicted_contributes_to_contradiction_rate() -> None:
    frame = pd.DataFrame({
        "primary_class": [
            "directionally_contradicted", "supported", "manipulation_failure",
            "not_applicable",
        ]
    })
    support, contradiction, estimand = intervention_classification_rates(frame)
    assert support == pytest.approx(1 / 3)
    assert contradiction == pytest.approx(1 / 3)
    assert "manipulation_failure_included" in estimand


def test_unrestricted_candidate_denominator_is_actual_search_space() -> None:
    representation = mock_generation("schelling")["representation"]
    assert len(representation["indicators"]) == 28
    assert candidate_count_for_method("unrestricted_temporal_search", representation) == 28 * 27
    assert candidate_count_for_method("full_method", representation) == sum(
        len(edge["hypothesis_group_ids"])
        for edge in representation["candidate_edges"]
    )


def test_qualification_rate_never_exceeds_one() -> None:
    assert temporal_qualification_rate(27, 28) < 1
    with pytest.raises(RuntimeError, match="exceeds one"):
        temporal_qualification_rate(29, 28)


def _response(significant: bool, sign: int = 1, onset: int = 1) -> SimpleNamespace:
    return SimpleNamespace(significant=significant, effect_sign=sign, onset_time=onset)


@pytest.mark.parametrize("missing_index", [0, 1])
def test_required_downstream_absence_falsifies_prediction(missing_index: int) -> None:
    downstream = [_response(True, onset=2), _response(True, onset=3)]
    downstream[missing_index] = _response(False, onset=-1)
    classification, *_ = classify_prediction_requirements(
        _response(True, onset=1), downstream, ["increase"] * 3, [True, True],
        source_required=True, order_required=True,
        observational_edges_retained=[True, True],
    )
    assert classification == "contradicted"


def test_wrong_direction_falsifies_prediction() -> None:
    classification, *_ = classify_prediction_requirements(
        _response(True, onset=1),
        [_response(True, onset=2), _response(True, sign=-1, onset=3)],
        ["increase"] * 3, [True, True], source_required=True,
        order_required=True, observational_edges_retained=[True, True],
    )
    assert classification == "contradicted"


def test_all_prediction_requirements_pass_is_supported() -> None:
    classification, *_ = classify_prediction_requirements(
        _response(True, onset=1),
        [_response(True, onset=2), _response(True, onset=3)],
        ["increase"] * 3, [True, True], source_required=True,
        order_required=True, observational_edges_retained=[True, True],
    )
    assert classification == "supported"


def test_primary_attribution_contains_only_full_method(tmp_path: Path) -> None:
    analysis = tmp_path / "analysis"
    analysis.mkdir()
    representation = mock_generation("schelling")["representation"]
    semantic = representation["candidate_edges"][0]
    edge = _edge(semantic["source"], semantic["target"])
    write_graph_records(analysis / "main_graphs.jsonl", [{
        "scenario": "schelling", "method": "full_method",
        "edges": [edge.__dict__],
    }])
    pd.DataFrame([
        {"scenario": "schelling", "method": "full_method", "source": edge.source, "target": edge.target, "primary_class": "supported"},
        {"scenario": "schelling", "method": "trajectory_vote", "source": edge.source, "target": edge.target, "primary_class": "directionally_contradicted"},
    ]).to_csv(analysis / "intervention_classifications.csv", index=False)
    pd.DataFrame(columns=["scenario", "path_id"]).to_csv(
        analysis / "path_temporal_qualification.csv", index=False
    )
    pd.DataFrame(columns=["scenario", "path_id", "path_classification"]).to_csv(
        analysis / "path_intervention_classification.csv", index=False
    )
    result = integrate_evidence(tmp_path, {"schelling": representation})
    evidence = [
        item for relation in result["scenarios"]["schelling"]["relations"]
        for item in relation["intervention_evidence"]
    ]
    assert evidence
    assert {item["method"] for item in evidence} == {"full_method"}
    comparative = pd.read_csv(analysis / "comparative_method_intervention_evidence.csv")
    assert set(comparative["method"]) == {"trajectory_vote"}


def test_semantic_proposal_has_no_temporal_qualification_rate(
    tmp_path: Path,
) -> None:
    analysis = tmp_path / "analysis"
    representation_root = tmp_path / "representation"
    analysis.mkdir()
    representation_root.mkdir()
    representation = _path_representation()
    semantic_edges = [_edge("micro_a", "meso_b"), _edge("meso_b", "macro_c")]
    write_graph_records(
        analysis / "main_graphs.jsonl",
        [
            {
                "scenario": "toy",
                "method": "llm_semantic_proposal",
                "edges": [edge.__dict__ for edge in semantic_edges],
            },
            {
                "scenario": "toy",
                "method": "full_method",
                "edges": [semantic_edges[0].__dict__],
            },
        ],
    )
    (representation_root / "representation_agreement.json").write_text(
        json.dumps({"toy": {"computation_signature_jaccard": 1.0}}),
        encoding="utf-8",
    )

    result = evaluate_full_discovery(tmp_path, {"toy": representation})
    proposal = result[result["method"] == "llm_semantic_proposal"].iloc[0]
    qualified = result[result["method"] == "full_method"].iloc[0]
    assert pd.isna(proposal["temporal_qualification_rate"])
    assert proposal["temporal_metric_reason"] == "not_temporally_qualified"
    assert qualified["temporal_qualification_rate"] == pytest.approx(0.5)
    assert qualified["temporal_metric_reason"] == "temporally_qualified"


def test_controlled_fdr_is_macro_outcome_group_local() -> None:
    for scenario in ("schelling", "deffuant"):
        groups = {
            edge["hypothesis_group_id"]
            for edge in controlled_representation(scenario)["candidate_edges"]
        }
        prefix = "s" if scenario == "schelling" else "d"
        assert groups == {f"macro_outcome_{prefix}_macro_{index}" for index in range(4)}


def test_controlled_recovery_uses_multiple_macro_outcome_groups() -> None:
    representation = controlled_representation("schelling")
    assert all("hypothesis_group_id" in edge for edge in representation["candidate_edges"])
    assert len({edge["hypothesis_group_id"] for edge in representation["candidate_edges"]}) == 4


def test_controlled_unmanipulable_truth_is_not_eligible() -> None:
    representation = controlled_representation("schelling")
    truth = {(edge.source, edge.target) for edge in reference_relations("schelling")}
    eligible = intervention_testable_edges(sorted(truth), representation)
    assert len(eligible) == 6
    assert ("s_micro_boundary", "s_meso_2") not in eligible
    assert ("s_meso_2", "s_macro_2") not in eligible


def test_semantic_generation_resume_does_not_repeat_completed_api_calls(
    tmp_path: Path,
) -> None:
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    template = (PROJECT_ROOT / "config" / "semantic_prompt.txt").read_text(encoding="utf-8")
    llm = {"model": "resume-test"}
    counts: defaultdict[int, int] = defaultdict(int)

    def provider(index: int, *, fail: bool = False):
        def complete(_system: str, _user: str) -> LLMResponse:
            counts[index] += 1
            if fail:
                raise RuntimeError("simulated provider outage")
            return LLMResponse(
                json.dumps(mock_indicator_generation("schelling")), 0, 0, "resume-test"
            )

        return complete

    for index in (0, 1):
        run_generation(
            "schelling", index, config, llm, template, tmp_path,
            completion=provider(index),
        )
    with pytest.raises(RuntimeError, match="provider outage"):
        run_generation(
            "schelling", 2, config, llm, template, tmp_path,
            completion=provider(2, fail=True),
        )
    for index in (0, 1, 2):
        run_generation(
            "schelling", index, config, llm, template, tmp_path,
            completion=provider(index),
        )
    assert counts == {0: 1, 1: 1, 2: 2}


def test_figure7_filters_incomplete_or_unordered_paths() -> None:
    rows = []
    for path_id, onsets, scales in (
        ("valid", [1, 2, 3], ["micro", "meso", "macro"]),
        ("unordered", [1, 4, 3], ["micro", "meso", "macro"]),
        ("incomplete", [1, 2], ["micro", "meso"]),
    ):
        for scale, onset in zip(scales, onsets):
            rows.append({
                "scenario": "toy", "path_id": path_id, "parameter": "theta",
                "direction": "plus", "source": "micro_a", "meso": "meso_b",
                "macro": "macro_c", "scale": scale, "onset_time": onset,
                "significant": True, "cumulative_effect": 0.5,
            })
    timing = pd.DataFrame(rows)
    classifications = pd.DataFrame([
        {"scenario": "toy", "method": "full_method", "parameter": "theta", "direction": "plus", "root_source": "micro_a", "source": "micro_a", "target": "meso_b", "primary_class": "supported"},
        {"scenario": "toy", "method": "full_method", "parameter": "theta", "direction": "plus", "root_source": "micro_a", "source": "meso_b", "target": "macro_c", "primary_class": "supported"},
    ])
    assert eligible_propagation_path_ids(timing, classifications) == {"valid"}


def test_propagation_path_cannot_borrow_support_from_another_direction() -> None:
    timing = pd.DataFrame([
        {
            "scenario": "toy", "path_id": "theta:plus:path", "parameter": "theta",
            "direction": "plus", "source": "micro_a", "meso": "meso_b",
            "macro": "macro_c", "scale": scale, "onset_time": onset,
            "significant": True, "cumulative_effect": 0.5,
        }
        for scale, onset in (("micro", 1), ("meso", 2), ("macro", 3))
    ])
    classifications = pd.DataFrame([
        {
            "scenario": "toy", "method": "full_method", "parameter": "theta",
            "direction": "plus", "root_source": "micro_a",
            "source": "micro_a", "target": "meso_b",
            "primary_class": "supported",
        },
        {
            "scenario": "toy", "method": "full_method", "parameter": "theta",
            "direction": "plus", "root_source": "micro_a",
            "source": "meso_b", "target": "macro_c",
            "primary_class": "manipulation_failure",
        },
        {
            "scenario": "toy", "method": "full_method", "parameter": "theta",
            "direction": "minus", "root_source": "micro_a",
            "source": "meso_b", "target": "macro_c",
            "primary_class": "supported",
        },
    ])
    assert eligible_propagation_path_ids(timing, classifications) == set()
    classifications.loc[
        classifications["direction"] == "plus", "primary_class"
    ] = "supported"
    assert eligible_propagation_path_ids(timing, classifications) == {
        "theta:plus:path"
    }


def test_propagation_path_cannot_borrow_support_from_another_root() -> None:
    timing = pd.DataFrame([
        {
            "scenario": "toy", "path_id": "theta:plus:path", "parameter": "theta",
            "direction": "plus", "source": "micro_a", "meso": "meso_b",
            "macro": "macro_c", "scale": scale, "onset_time": onset,
            "significant": True, "cumulative_effect": 0.5,
        }
        for scale, onset in (("micro", 1), ("meso", 2), ("macro", 3))
    ])
    classifications = pd.DataFrame([
        {
            "scenario": "toy", "method": "full_method", "parameter": "theta",
            "direction": "plus", "root_source": "micro_a",
            "source": "micro_a", "target": "meso_b",
            "primary_class": "supported",
        },
        {
            "scenario": "toy", "method": "full_method", "parameter": "theta",
            "direction": "plus", "root_source": "micro_d",
            "source": "meso_b", "target": "macro_c",
            "primary_class": "supported",
        },
    ])

    assert eligible_propagation_path_ids(timing, classifications) == set()
    classifications.loc[len(classifications)] = {
        "scenario": "toy", "method": "full_method", "parameter": "theta",
        "direction": "plus", "root_source": "micro_a",
        "source": "meso_b", "target": "macro_c",
        "primary_class": "supported",
    }
    assert eligible_propagation_path_ids(timing, classifications) == {
        "theta:plus:path"
    }
