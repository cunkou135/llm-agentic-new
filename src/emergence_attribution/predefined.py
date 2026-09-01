"""Fixed-observable comparator used only by the requested ablation."""

from __future__ import annotations

from typing import Any

from .dsl import expression_fields


def _field(name: str) -> dict[str, Any]:
    return {"op": "field", "name": name}


def _reduce(op: str, name: str, axis: str = "agent", **extra: Any) -> dict[str, Any]:
    return {"op": op, "input": _field(name), "axis": axis, **extra}


def _channel(index: int) -> dict[str, Any]:
    return {"op": "select", "input": _field("mechanism_channel"), "axis": "channel", "index": index}


def _micro_expressions(scenario: str) -> list[tuple[str, str, dict[str, Any]]]:
    if scenario == "schelling":
        return [
            ("unsatisfied fraction", "fraction of agents below the satisfaction threshold", _reduce("fraction", "unhappy")),
            ("relocation fraction", "fraction of agents relocating in a step", _reduce("fraction", "moved")),
            ("boundary fraction", "fraction of agents adjacent to another group", _reduce("fraction", "boundary_agent")),
            ("mean local similarity", "mean same-group neighbourhood share", _reduce("mean", "local_similarity")),
            ("similarity dispersion", "standard deviation of local similarity", _reduce("std", "local_similarity")),
            ("lower similarity quantile", "lower quartile of local similarity", _reduce("quantile", "local_similarity", q=0.25)),
            ("upper similarity quantile", "upper quartile of local similarity", _reduce("quantile", "local_similarity", q=0.75)),
            ("mean neighbour count", "mean occupied-neighbour count", _reduce("mean", "neighbour_count")),
            ("neighbour-count dispersion", "standard deviation of occupied-neighbour count", _reduce("std", "neighbour_count")),
            ("mean relocation distance", "mean normalised relocation distance", _reduce("mean", "move_distance")),
            ("upper relocation distance", "upper decile relocation distance", _reduce("quantile", "move_distance", q=0.90)),
            ("mean destination similarity", "mean destination same-group fraction", _reduce("mean", "destination_similarity")),
            ("destination-similarity dispersion", "dispersion of destination similarity", _reduce("std", "destination_similarity")),
            ("spatial component count", "number of periodic same-group components", {"op": "connected_component_count", "input": _field("state_grid")}),
            ("largest component fraction", "fraction in the largest same-group component", {"op": "largest_component_fraction", "input": _field("state_grid")}),
            ("neighbour spatial agreement", "same-group agreement across periodic cell adjacencies", {"op": "spatial_neighbor_similarity", "input": _field("state_grid")}),
        ]
    return [
        ("accepted interaction fraction", "fraction of interactions that assimilate", _reduce("fraction", "interaction_accepted")),
        ("repulsive interaction fraction", "fraction of interactions that produce repulsion", _reduce("fraction", "interaction_backfire")),
        ("rejected interaction fraction", "fraction of interactions without an update", _reduce("fraction", "interaction_rejected")),
        ("sign-flip fraction", "fraction of agents crossing opinion zero", _reduce("fraction", "sign_flip")),
        ("mean opinion", "population mean opinion", _reduce("mean", "state_opinion")),
        ("opinion dispersion", "population opinion standard deviation", _reduce("std", "state_opinion")),
        ("lower opinion quantile", "lower opinion quartile", _reduce("quantile", "state_opinion", q=0.25)),
        ("upper opinion quantile", "upper opinion quartile", _reduce("quantile", "state_opinion", q=0.75)),
        ("mean absolute opinion", "mean opinion extremity", {"op": "mean", "input": {"op": "abs", "input": _field("state_opinion")}, "axis": "agent"}),
        ("upper absolute opinion", "upper decile opinion extremity", {"op": "quantile", "input": {"op": "abs", "input": _field("state_opinion")}, "axis": "agent", "q": 0.90}),
        ("mean interaction distance", "mean sampled opinion distance", _reduce("mean", "interaction_distance")),
        ("mean absolute update", "mean absolute signed opinion update", {"op": "mean", "input": {"op": "abs", "input": _field("agent_shift")}, "axis": "agent"}),
        ("update dispersion", "standard deviation of signed opinion updates", _reduce("std", "agent_shift")),
        ("upper interaction distance", "upper decile sampled opinion distance", _reduce("quantile", "interaction_distance", q=0.90)),
        ("network opinion assortativity", "correlation of endpoint opinions", {"op": "network_assortativity", "values": _field("state_opinion"), "edges": _field("network_edges")}),
        ("opinion variance", "population opinion variance", _reduce("variance", "state_opinion")),
    ]


def predefined_representation(scenario: str) -> dict[str, Any]:
    prefix = "sfix" if scenario == "schelling" else "dfix"
    micro = _micro_expressions(scenario)
    parameters = (
        {0: "tolerance", 4: "move_probability", 11: "destination_preference"}
        if scenario == "schelling"
        else {0: "confidence_bound", 11: "assimilation_strength", 1: "backfire_threshold"}
    )
    indicators: list[dict[str, Any]] = []
    for index, (name, definition, expression) in enumerate(micro):
        associations = []
        if index in parameters:
            associations.append(
                {
                    "parameter": parameters[index],
                    "relationship": "direct",
                    "expected_indicator_direction": "unknown",
                    "rationale": "Fixed comparator association derived from the documented simulator rule.",
                }
            )
        indicators.append(
            {
                "id": f"{prefix}_micro_{index:02d}",
                "semantic_name": name,
                "scientific_definition": definition,
                "phenomenon": "fixed observable comparator",
                "scale": "micro",
                "branch_id": f"branch_{index // 4}",
                "entities": "agent interactions or local states",
                "source_fields": sorted(expression_fields(expression)),
                "computation": expression,
                "temporal_aggregation": {"op": "identity"},
                "parameter_associations": associations,
                "scientific_rationale": "Predefined only for the fixed-observable ablation comparator.",
            }
        )
    for index in range(8):
        expression = (
            _channel(index // 2)
            if index % 2 == 0
            else {"op": "rolling_mean", "input": _channel(index // 2), "window": 3}
        )
        indicators.append(
            {
                "id": f"{prefix}_meso_{index:02d}",
                "semantic_name": f"fixed intermediate organisation {index + 1}",
                "scientific_definition": "A documented intermediate organisation summary for ablation.",
                "phenomenon": "fixed observable comparator",
                "scale": "meso",
                "branch_id": f"branch_{index // 2}",
                "entities": "system channel",
                "source_fields": ["mechanism_channel"],
                "computation": expression,
                "temporal_aggregation": {"op": "identity"},
                "parameter_associations": [],
                "scientific_rationale": "Predefined only for the fixed-observable ablation comparator.",
            }
        )
    for index in range(4):
        expression = _channel(index + 4)
        indicators.append(
            {
                "id": f"{prefix}_macro_{index:02d}",
                "semantic_name": f"fixed collective outcome {index + 1}",
                "scientific_definition": "A documented collective state summary for ablation.",
                "phenomenon": "fixed observable comparator",
                "scale": "macro",
                "branch_id": f"branch_{index}",
                "entities": "system channel",
                "source_fields": ["mechanism_channel"],
                "computation": expression,
                "temporal_aggregation": {"op": "identity"},
                "parameter_associations": [],
                "scientific_rationale": "Predefined only for the fixed-observable ablation comparator.",
            }
        )
    edges: list[dict[str, Any]] = []
    for branch in range(4):
        micro_ids = [f"{prefix}_micro_{index:02d}" for index in range(branch * 4, branch * 4 + 4)]
        meso_ids = [f"{prefix}_meso_{branch * 2:02d}", f"{prefix}_meso_{branch * 2 + 1:02d}"]
        macro_id = f"{prefix}_macro_{branch:02d}"
        for index, micro_id in enumerate(micro_ids):
            edges.append(
                {
                    "source": micro_id,
                    "target": meso_ids[index // 2],
                    "expected_direction": "unknown",
                    "rationale": "Fixed adjacent-scale comparator relation defined before evaluation.",
                }
            )
        for meso_id in meso_ids:
            edges.append(
                {
                    "source": meso_id,
                    "target": macro_id,
                    "expected_direction": "unknown",
                    "rationale": "Fixed adjacent-scale comparator relation defined before evaluation.",
                }
            )
    return {
        "scenario": scenario,
        "phenomenon": "fixed observable comparator",
        "indicators": indicators,
        "candidate_edges": edges,
        "interpretation_boundary": "This fixed comparator receives temporal and intervention evaluation only in the ablation and never supplies the full method.",
    }

