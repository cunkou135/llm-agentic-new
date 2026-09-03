"""Fixed-observable comparator used only by the requested ablation."""

from __future__ import annotations

from typing import Any

from .dsl import expression_fields


def _field(name: str) -> dict[str, Any]:
    return {"op": "field", "name": name}


def _reduce(op: str, name: str, axis: str = "agent", **extra: Any) -> dict[str, Any]:
    return {"op": op, "input": _field(name), "axis": axis, **extra}


def _group_stat(
    value: dict[str, Any], reducer: str, across: str = "variance"
) -> dict[str, Any]:
    grouped = {
        "op": "group_reduce",
        "values": value,
        "groups": _field("district_id"),
        "axis": "agent",
        "reducer": reducer,
    }
    return {"op": across, "input": grouped, "axis": "group"}


def _network_neighborhood(value: dict[str, Any], reducer: str) -> dict[str, Any]:
    return {
        "op": "network_neighborhood_reduce",
        "values": value,
        "edges": _field("network_edges"),
        "reducer": reducer,
    }


def _quantile_range(
    value: dict[str, Any], axis: str, lower: float = 0.25, upper: float = 0.75
) -> dict[str, Any]:
    return {
        "op": "subtract",
        "left": {"op": "quantile", "input": value, "axis": axis, "q": upper},
        "right": {"op": "quantile", "input": value, "axis": axis, "q": lower},
    }


def _micro_expressions(
    scenario: str,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    if scenario == "schelling":
        return [
            ("unsatisfied event prevalence", "prevalence of individual dissatisfaction events", "elementary_event", _reduce("fraction", "unhappy")),
            ("relocation event prevalence", "prevalence of individual relocation events", "elementary_event", _reduce("fraction", "moved")),
            ("boundary exposure prevalence", "prevalence of individual cross-group boundary exposure", "local_process", _reduce("fraction", "boundary_agent")),
            ("mean local exposure", "mean individual same-group Moore-neighborhood exposure", "local_process", _reduce("mean", "local_similarity")),
            ("local exposure dispersion", "dispersion of individual local exposure", "local_process", _reduce("std", "local_similarity")),
            ("lower local exposure", "lower quartile of individual local exposure", "local_process", _reduce("quantile", "local_similarity", q=0.25)),
            ("upper local exposure", "upper quartile of individual local exposure", "local_process", _reduce("quantile", "local_similarity", q=0.75)),
            ("mean occupied neighborhood", "mean local occupied-neighbor count per agent", "local_process", _reduce("mean", "neighbour_count")),
            ("sparse-neighborhood prevalence", "prevalence of agents with at most two occupied neighbors", "local_process", {"op": "fraction", "input": {"op": "less_equal", "left": _field("neighbour_count"), "right": {"op": "constant", "value": 2}}, "axis": "agent"}),
            ("mean relocation distance", "mean distance of individual relocation events including zero for no move", "elementary_event", _reduce("mean", "move_distance")),
            ("upper relocation distance", "upper decile of individual relocation distance", "elementary_event", _reduce("quantile", "move_distance", q=0.90)),
            ("mean destination outcome", "mean local similarity outcome of individual destination choices", "elementary_event", _reduce("mean", "destination_similarity")),
            ("destination outcome dispersion", "dispersion of individual destination-choice outcomes", "elementary_event", _reduce("std", "destination_similarity")),
            ("long-move prevalence", "prevalence of individual relocations longer than one quarter of maximum periodic distance", "elementary_event", {"op": "fraction", "input": {"op": "greater", "left": _field("move_distance"), "right": {"op": "constant", "value": 0.25}}, "axis": "agent"}),
            ("high-similarity destination prevalence", "prevalence of destination choices with high same-group exposure", "elementary_event", {"op": "fraction", "input": {"op": "greater_equal", "left": _field("destination_similarity"), "right": {"op": "constant", "value": 0.75}}, "axis": "agent"}),
            ("isolated-agent prevalence", "prevalence of individual agents with no occupied Moore neighbor", "local_process", {"op": "fraction", "input": {"op": "equal", "left": _field("neighbour_count"), "right": {"op": "constant", "value": 0}}, "axis": "agent"}),
        ]
    return [
        ("accepted encounter prevalence", "prevalence of pairwise assimilation encounters", "interaction", _reduce("fraction", "interaction_accepted")),
        ("backfire encounter prevalence", "prevalence of pairwise repulsive-update encounters", "interaction", _reduce("fraction", "interaction_backfire")),
        ("rejected encounter prevalence", "prevalence of pairwise no-update encounters", "interaction", _reduce("fraction", "interaction_rejected")),
        ("rewired encounter prevalence", "prevalence of focal interaction ties successfully replaced", "elementary_event", _reduce("fraction", "edge_rewired")),
        ("mean encounter distance", "mean opinion distance in pairwise encounters", "interaction", _reduce("mean", "interaction_distance")),
        ("upper encounter distance", "upper decile of pairwise opinion distance", "interaction", _reduce("quantile", "interaction_distance", q=0.90)),
        ("mean update magnitude", "mean magnitude of individual elementary opinion updates", "elementary_event", {"op": "mean", "input": {"op": "abs", "input": _field("agent_shift")}, "axis": "agent"}),
        ("update dispersion", "dispersion of individual signed opinion updates", "elementary_event", _reduce("std", "agent_shift")),
        ("sign-crossing prevalence", "prevalence of individual updates crossing opinion zero", "elementary_event", _reduce("fraction", "sign_flip")),
        ("extreme-state prevalence", "prevalence of individual agents with absolute opinion at least 0.75", "individual", {"op": "fraction", "input": {"op": "greater_equal", "left": {"op": "abs", "input": _field("state_opinion")}, "right": {"op": "constant", "value": 0.75}}, "axis": "agent"}),
        ("mean individual extremity", "mean absolute individual opinion state", "individual", {"op": "mean", "input": {"op": "abs", "input": _field("state_opinion")}, "axis": "agent"}),
        ("positive-state prevalence", "prevalence of individual agents with positive opinion", "individual", {"op": "fraction", "input": {"op": "greater", "left": _field("state_opinion"), "right": {"op": "constant", "value": 0.0}}, "axis": "agent"}),
        ("large-update prevalence", "prevalence of elementary opinion updates exceeding 0.05 in magnitude", "elementary_event", {"op": "fraction", "input": {"op": "greater", "left": {"op": "abs", "input": _field("agent_shift")}, "right": {"op": "constant", "value": 0.05}}, "axis": "agent"}),
        ("mean accepted distance", "mean pairwise distance among accepted encounters", "interaction", {"op": "mean", "input": {"op": "where", "condition": _field("interaction_accepted"), "input": _field("interaction_distance")}, "axis": "agent"}),
        ("mean backfire distance", "mean pairwise distance among backfire encounters", "interaction", {"op": "mean", "input": {"op": "where", "condition": _field("interaction_backfire"), "input": _field("interaction_distance")}, "axis": "agent"}),
        ("lower encounter distance", "lower quartile of pairwise opinion distance", "interaction", _reduce("quantile", "interaction_distance", q=0.25)),
    ]


def _meso_expressions(
    scenario: str,
) -> list[tuple[str, str, str, dict[str, Any]]]:
    if scenario == "schelling":
        return [
            ("district composition heterogeneity", "between-district variance in group composition", "district", _group_stat(_field("agent_group"), "mean")),
            ("district composition diversity", "mean within-district group-label entropy", "district", _group_stat(_field("agent_group"), "entropy", across="mean")),
            ("district dissatisfaction heterogeneity", "between-district variance in dissatisfaction prevalence", "district", _group_stat(_field("unhappy"), "fraction")),
            ("district turnover heterogeneity", "between-district variance in relocation-event prevalence", "district", _group_stat(_field("moved"), "fraction")),
            ("district turnover interquartile spread", "interquartile range across districts in relocation-event prevalence", "district", _quantile_range({"op": "group_reduce", "values": _field("moved"), "groups": _field("district_id"), "axis": "agent", "reducer": "fraction"}, "group")),
            ("district boundary heterogeneity", "between-district variance in boundary exposure prevalence", "district", _group_stat(_field("boundary_agent"), "fraction")),
            ("district destination heterogeneity", "between-district variance in destination-choice outcome", "district", _group_stat(_field("destination_similarity"), "mean")),
            ("district mobility-distance heterogeneity", "between-district variance in mean relocation distance", "district", _group_stat(_field("move_distance"), "mean")),
        ]
    neighborhood_mean = _network_neighborhood(_field("state_opinion"), "mean")
    extreme_state = {
        "op": "greater_equal",
        "left": {"op": "abs", "input": _field("state_opinion")},
        "right": {"op": "constant", "value": 0.75},
    }
    positive_state = {
        "op": "greater",
        "left": _field("state_opinion"),
        "right": {"op": "constant", "value": 0.0},
    }
    return [
        ("neighborhood opinion mismatch", "mean absolute difference between each opinion and its network-neighborhood mean", "neighborhood", {"op": "mean", "input": {"op": "distance", "left": _field("state_opinion"), "right": neighborhood_mean}, "axis": "agent"}),
        ("neighborhood mean separation", "variance across agents in network-neighborhood mean opinion", "neighborhood", {"op": "variance", "input": neighborhood_mean, "axis": "agent"}),
        ("neighborhood opinion dispersion", "mean within-network-neighborhood opinion standard deviation", "neighborhood", {"op": "mean", "input": _network_neighborhood(_field("state_opinion"), "std"), "axis": "agent"}),
        ("neighborhood mean interquartile spread", "interquartile range across agents in network-neighborhood mean opinion", "neighborhood", _quantile_range(neighborhood_mean, "agent")),
        ("neighborhood extreme-state heterogeneity", "variance across agents in neighborhood extreme-opinion prevalence", "neighborhood", {"op": "variance", "input": _network_neighborhood(extreme_state, "fraction"), "axis": "agent"}),
        ("neighborhood sign-composition separation", "standard deviation across agents in neighborhood positive-opinion prevalence", "neighborhood", {"op": "std", "input": _network_neighborhood(positive_state, "fraction"), "axis": "agent"}),
        ("neighborhood acceptance heterogeneity", "variance across agents in neighborhood acceptance prevalence", "neighborhood", {"op": "variance", "input": _network_neighborhood(_field("interaction_accepted"), "fraction"), "axis": "agent"}),
        ("neighborhood rewiring heterogeneity", "variance across agents in neighborhood rewiring-event prevalence", "neighborhood", {"op": "variance", "input": _network_neighborhood(_field("edge_rewired"), "fraction"), "axis": "agent"}),
    ]


def _macro_expressions(
    scenario: str,
) -> list[tuple[str, str, dict[str, Any]]]:
    if scenario == "schelling":
        return [
            ("whole-grid segregation", "whole-grid fraction belonging to the largest periodic same-group domain", {"op": "largest_component_fraction", "input": _field("state_grid")}),
            ("city-wide mobility", "short-window whole-grid prevalence of relocation events", {"op": "rolling_mean", "input": _reduce("fraction", "moved"), "window": 3}),
            ("global spatial fragmentation", "whole-grid count of periodic same-group components", {"op": "connected_component_count", "input": _field("state_grid")}),
            ("global spatial agreement", "whole-grid same-group agreement across periodic cell adjacencies", {"op": "spatial_neighbor_similarity", "input": _field("state_grid")}),
        ]
    return [
        ("whole-network opinion dispersion", "whole-system standard deviation of opinions", _reduce("std", "state_opinion")),
        ("whole-network opinion center", "whole-system mean signed opinion", _reduce("mean", "state_opinion")),
        ("whole-network opinion assortativity", "whole-network opinion correlation across current adaptive-network endpoints", {"op": "network_assortativity", "values": _field("state_opinion"), "edges": _field("network_edges")}),
        ("whole-network opinion entropy", "whole-system binned entropy of the opinion distribution", {"op": "binned_entropy", "input": _field("state_opinion"), "axis": "agent", "bins": 10}),
    ]


def predefined_representation(scenario: str) -> dict[str, Any]:
    prefix = "sfix" if scenario == "schelling" else "dfix"
    micro = _micro_expressions(scenario)
    parameters = (
        {0: "tolerance", 1: "move_probability", 11: "destination_preference"}
        if scenario == "schelling"
        else {0: "confidence_bound", 6: "assimilation_strength", 1: "backfire_threshold"}
    )
    indicators: list[dict[str, Any]] = []
    for index, (name, definition, entity_scope, expression) in enumerate(micro):
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
                "entity_scope": entity_scope,
                "entities": "individual agent interactions, events, or local primitive states",
                "source_fields": sorted(expression_fields(expression)),
                "computation": expression,
                "temporal_aggregation": {"op": "identity"},
                "parameter_associations": associations,
                "scientific_rationale": "Predefined only for the fixed-observable ablation comparator.",
            }
        )
    for index, (name, definition, entity_scope, expression) in enumerate(
        _meso_expressions(scenario)
    ):
        indicators.append(
            {
                "id": f"{prefix}_meso_{index:02d}",
                "semantic_name": name,
                "scientific_definition": definition,
                "phenomenon": "fixed observable comparator",
                "scale": "meso",
                "entity_scope": entity_scope,
                "entities": "public district, network-neighborhood, or community organization",
                "source_fields": sorted(expression_fields(expression)),
                "computation": expression,
                "temporal_aggregation": {"op": "identity"},
                "parameter_associations": [],
                "scientific_rationale": "Predefined only for the fixed-observable ablation comparator.",
            }
        )
    for index, (name, definition, expression) in enumerate(
        _macro_expressions(scenario)
    ):
        indicators.append(
            {
                "id": f"{prefix}_macro_{index:02d}",
                "semantic_name": name,
                "scientific_definition": definition,
                "phenomenon": "fixed observable comparator",
                "scale": "macro",
                "entity_scope": "whole_system",
                "entities": "public system-level simulator summary",
                "source_fields": sorted(expression_fields(expression)),
                "computation": expression,
                "temporal_aggregation": {"op": "identity"},
                "parameter_associations": [],
                "scientific_rationale": "Predefined only for the fixed-observable ablation comparator.",
            }
        )
    edges: list[dict[str, Any]] = []
    for macro_index in range(4):
        micro_ids = [f"{prefix}_micro_{index:02d}" for index in range(macro_index * 4, macro_index * 4 + 4)]
        meso_ids = [f"{prefix}_meso_{macro_index * 2:02d}", f"{prefix}_meso_{macro_index * 2 + 1:02d}"]
        macro_id = f"{prefix}_macro_{macro_index:02d}"
        for index, micro_id in enumerate(micro_ids):
            edges.append(
                {
                    "source": micro_id,
                    "target": meso_ids[index // 2],
                    "expected_direction": "unknown",
                    "hypothesis_group_ids": [f"macro_outcome_{macro_id}"],
                    "path_ids": [f"fixed_{micro_id}_{meso_ids[index // 2]}_{macro_id}"],
                }
            )
        for meso_id in meso_ids:
            edges.append(
                {
                    "source": meso_id,
                    "target": macro_id,
                    "expected_direction": "unknown",
                    "hypothesis_group_ids": [f"macro_outcome_{macro_id}"],
                    "path_ids": [f"fixed_{meso_id}_{macro_id}"],
                }
            )
        edges.append(
            {
                "source": micro_ids[0],
                "target": meso_ids[1],
                "expected_direction": "unknown",
                "hypothesis_group_ids": [f"macro_outcome_{macro_id}"],
                "path_ids": [f"fixed_{micro_ids[0]}_{meso_ids[1]}_{macro_id}"],
            }
        )
    return {
        "scenario": scenario,
        "phenomenon": "fixed observable comparator",
        "indicators": indicators,
        "candidate_edges": edges,
        "interpretation_boundary": "This fixed comparator receives temporal and intervention evaluation only in the ablation and never supplies the full method.",
    }
