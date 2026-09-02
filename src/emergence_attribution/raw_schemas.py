"""Prompt-safe simulator descriptions and raw field schemas."""

from __future__ import annotations

from copy import deepcopy
from typing import Any


_COMMON_METADATA = [
    {
        "field_name": "num_steps",
        "dtype": "int32",
        "shape": [1],
        "semantic_meaning": "number of recorded simulation steps",
        "entity_level": "run",
    },
    {
        "field_name": "agent_count",
        "dtype": "int32",
        "shape": [1],
        "semantic_meaning": "number of agents in the simulation",
        "entity_level": "run",
    },
]


PUBLIC_RAW_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    "schelling": _COMMON_METADATA
    + [
        {
            "field_name": "state_grid",
            "dtype": "int8",
            "shape": ["time", "grid_y", "grid_x"],
            "semantic_meaning": "periodic occupancy grid with -1 for vacancy and 0 or 1 for group",
            "entity_level": "cell",
        },
        {
            "field_name": "agent_id",
            "dtype": "int32",
            "shape": ["agent"],
            "semantic_meaning": "stable public identifier of each agent",
            "entity_level": "agent",
        },
        {
            "field_name": "agent_group",
            "dtype": "int8",
            "shape": ["agent"],
            "semantic_meaning": "fixed group label of each agent",
            "entity_level": "agent",
        },
        {
            "field_name": "agent_position",
            "dtype": "int32",
            "shape": ["time", "agent", "coordinate"],
            "semantic_meaning": "row and column occupied by every agent",
            "entity_level": "agent",
        },
        {
            "field_name": "district_id",
            "dtype": "int16",
            "shape": ["time", "agent"],
            "semantic_meaning": "fixed spatial district containing each agent at the start of the recorded step",
            "entity_level": "district membership",
        },
        {
            "field_name": "local_similarity",
            "dtype": "float32",
            "shape": ["time", "agent"],
            "semantic_meaning": "same-group fraction among occupied Moore neighbours",
            "entity_level": "agent",
        },
        {
            "field_name": "neighbour_count",
            "dtype": "int16",
            "shape": ["time", "agent"],
            "semantic_meaning": "occupied Moore-neighbour count",
            "entity_level": "agent",
        },
        {
            "field_name": "unhappy",
            "dtype": "bool",
            "shape": ["time", "agent"],
            "semantic_meaning": "whether local similarity is below tolerance",
            "entity_level": "agent",
        },
        {
            "field_name": "unhappy_count",
            "dtype": "int32",
            "shape": ["time"],
            "semantic_meaning": "number of unsatisfied agents in each step",
            "entity_level": "system log aggregate",
        },
        {
            "field_name": "moved",
            "dtype": "bool",
            "shape": ["time", "agent"],
            "semantic_meaning": "whether the agent relocated during the step",
            "entity_level": "agent",
        },
        {
            "field_name": "move_distance",
            "dtype": "float32",
            "shape": ["time", "agent"],
            "semantic_meaning": "periodic Manhattan distance of relocation, zero without a move",
            "entity_level": "agent",
        },
        {
            "field_name": "destination_similarity",
            "dtype": "float32",
            "shape": ["time", "agent"],
            "semantic_meaning": "same-group neighbour fraction at the selected destination, zero without a move",
            "entity_level": "agent",
        },
        {
            "field_name": "boundary_agent",
            "dtype": "bool",
            "shape": ["time", "agent"],
            "semantic_meaning": "whether an agent has at least one occupied different-group neighbour",
            "entity_level": "agent",
        },
    ],
    "deffuant": _COMMON_METADATA
    + [
        {
            "field_name": "state_opinion",
            "dtype": "float32",
            "shape": ["time", "agent"],
            "semantic_meaning": "agent opinions bounded to [-1, 1] before each step update",
            "entity_level": "agent",
        },
        {
            "field_name": "network_edges",
            "dtype": "int32",
            "shape": ["time", "edge", "endpoint"],
            "semantic_meaning": "undirected interaction-network endpoint pairs used for partner sampling at the start of each step",
            "entity_level": "edge",
        },
        {
            "field_name": "partner_id",
            "dtype": "int32",
            "shape": ["time", "agent"],
            "semantic_meaning": "network neighbour sampled by each agent at each step",
            "entity_level": "interaction",
        },
        {
            "field_name": "interaction_distance",
            "dtype": "float32",
            "shape": ["time", "agent"],
            "semantic_meaning": "absolute opinion distance in each sampled interaction",
            "entity_level": "interaction",
        },
        {
            "field_name": "interaction_accepted",
            "dtype": "bool",
            "shape": ["time", "agent"],
            "semantic_meaning": "whether sampled opinion distance permits assimilation",
            "entity_level": "interaction",
        },
        {
            "field_name": "interaction_backfire",
            "dtype": "bool",
            "shape": ["time", "agent"],
            "semantic_meaning": "whether sampled opinion distance triggers repulsive updating",
            "entity_level": "interaction",
        },
        {
            "field_name": "interaction_rejected",
            "dtype": "bool",
            "shape": ["time", "agent"],
            "semantic_meaning": "whether an interaction produces neither assimilation nor repulsion",
            "entity_level": "interaction",
        },
        {
            "field_name": "edge_rewired",
            "dtype": "bool",
            "shape": ["time", "agent"],
            "semantic_meaning": "whether the sampled rejected or backfire tie was successfully replaced for this focal agent",
            "entity_level": "interaction",
        },
        {
            "field_name": "agent_shift",
            "dtype": "float32",
            "shape": ["time", "agent"],
            "semantic_meaning": "signed opinion update applied to each agent",
            "entity_level": "agent",
        },
        {
            "field_name": "sign_flip",
            "dtype": "bool",
            "shape": ["time", "agent"],
            "semantic_meaning": "whether an update crosses opinion zero",
            "entity_level": "agent",
        },
        {
            "field_name": "extreme_agent_count",
            "dtype": "int32",
            "shape": ["time"],
            "semantic_meaning": "number of agents with absolute opinion at least 0.75",
            "entity_level": "system log aggregate",
        },
    ],
    "toy": [
        {
            "field_name": "x",
            "dtype": "float64",
            "shape": ["time", "agent"],
            "semantic_meaning": "toy lower-level measurements",
            "entity_level": "agent",
        },
        {
            "field_name": "group_id",
            "dtype": "int32",
            "shape": ["time", "agent"],
            "semantic_meaning": "toy intermediate group membership",
            "entity_level": "district membership",
        },
        {
            "field_name": "y",
            "dtype": "float64",
            "shape": ["time"],
            "semantic_meaning": "toy aggregate measurement",
            "entity_level": "system",
        },
    ],
}


# Public semantic lineage only: these labels identify which raw fields are
# alternate records of the same observable primitive.  They contain no hidden
# edge, lag, mechanism, or outcome information.  ``statistic_role`` lets the
# validator recognize an exposed aggregate count versus its per-entity event
# without guessing from field names.
_PUBLIC_PRIMITIVE_METADATA: dict[str, dict[str, tuple[str, str]]] = {
    "schelling": {
        "num_steps": ("simulation_length", "normalizer"),
        "agent_count": ("population_size", "normalizer"),
        "state_grid": ("spatial_configuration", "system_state"),
        "agent_id": ("agent_identity", "identifier"),
        "agent_group": ("social_group", "categorical_state"),
        "agent_position": ("spatial_position", "individual_state"),
        "district_id": ("district_membership", "membership"),
        "local_similarity": ("local_group_exposure", "individual_measure"),
        "neighbour_count": ("local_occupancy", "individual_measure"),
        "unhappy": ("dissatisfaction_event", "elementary_event"),
        "unhappy_count": ("dissatisfaction_event", "aggregate_count"),
        "moved": ("relocation_event", "elementary_event"),
        "move_distance": ("relocation_distance", "event_measure"),
        "destination_similarity": ("destination_exposure", "event_measure"),
        "boundary_agent": ("boundary_exposure", "binary_state"),
    },
    "deffuant": {
        "num_steps": ("simulation_length", "normalizer"),
        "agent_count": ("population_size", "normalizer"),
        "state_opinion": ("opinion_state", "individual_state"),
        "network_edges": ("network_topology", "system_structure"),
        "partner_id": ("interaction_partner", "interaction_record"),
        "interaction_distance": ("encounter_distance", "interaction_measure"),
        "interaction_accepted": ("assimilation_event", "interaction_event"),
        "interaction_backfire": ("repulsion_event", "interaction_event"),
        "interaction_rejected": ("rejection_event", "interaction_event"),
        "edge_rewired": ("rewiring_event", "interaction_event"),
        "agent_shift": ("opinion_update", "event_measure"),
        "sign_flip": ("sign_crossing_event", "elementary_event"),
        "extreme_agent_count": ("extreme_opinion_state", "aggregate_count"),
    },
    "toy": {
        "x": ("toy_lower_measure", "individual_measure"),
        "group_id": ("toy_group_membership", "membership"),
        "y": ("toy_aggregate_outcome", "system_state"),
    },
}

for _scenario, _schema in PUBLIC_RAW_SCHEMAS.items():
    _metadata = _PUBLIC_PRIMITIVE_METADATA[_scenario]
    for _field in _schema:
        _family, _role = _metadata[str(_field["field_name"])]
        _field["primitive_family"] = _family
        _field["statistic_role"] = _role


# These fields are never included in prompts, public NPZ files, or Full Discovery
# compilation.  They exist only for the separately labelled Controlled Recovery
# benchmark and are persisted below data/reference_hidden/.
HIDDEN_REFERENCE_SCHEMAS: dict[str, list[dict[str, Any]]] = {
    scenario: [
        {
            "field_name": "mechanism_channel",
            "dtype": "float32",
            "shape": ["time", "channel"],
            "semantic_meaning": "withheld controlled-benchmark state",
            "entity_level": "hidden reference",
        }
    ]
    for scenario in ("schelling", "deffuant")
}

HIDDEN_REFERENCE_FIELD_NAMES = frozenset(
    item["field_name"]
    for schema in HIDDEN_REFERENCE_SCHEMAS.values()
    for item in schema
)


RULES = {
    "schelling": [
        "Each occupied cell contains one fixed-group agent and the grid is periodic.",
        "The periodic grid is partitioned into fixed public spatial districts; district membership follows an agent's current cell and is not a precomputed segregation outcome.",
        "An agent is unsatisfied when its occupied-neighbour same-group fraction is below tolerance.",
        "An unsatisfied agent attempts movement with move_probability.",
        "A moving agent samples vacancies and, with destination_preference, chooses the sampled vacancy with the highest local similarity.",
        "Disabling homophilic relocation makes destination choice non-preferential while preserving other rules.",
    ],
    "deffuant": [
        "Each agent samples one neighbour from the current undirected network per step.",
        "Opinion distance within confidence_bound produces assimilation proportional to assimilation_strength.",
        "Distance at least backfire_threshold produces a weak repulsive update when the mechanism is enabled.",
        "Other encounters are rejected and opinions remain bounded to [-1, 1].",
        "Rejected or backfire encounters may replace the sampled tie at a fixed adaptive_rewiring_probability; replacement candidates exclude self-loops and duplicate edges, preserve at least one tie per agent, and use a disclosed weak homophilic preference.",
        "The time-varying fixed-edge-count edge list records the network used at the start of each step; successful rewiring affects the next step.",
        "Disabling backfire sets repulsive update strength to zero while preserving rejection-triggered adaptive rewiring.",
    ],
}


def public_raw_schema(scenario: str) -> list[dict[str, Any]]:
    """Return the only schema visible to semantic generation and Full Discovery."""

    return deepcopy(PUBLIC_RAW_SCHEMAS[scenario])


def hidden_reference_schema(scenario: str) -> list[dict[str, Any]]:
    """Return the isolated schema used only by the Controlled Recovery evaluator."""

    return deepcopy(HIDDEN_REFERENCE_SCHEMAS.get(scenario, []))


def raw_schema(scenario: str) -> list[dict[str, Any]]:
    """Backward-compatible public-schema alias."""

    return public_raw_schema(scenario)


def prompt_scenario_contract(scenario: str, spec: dict[str, Any]) -> dict[str, Any]:
    contract = {
        "scenario": scenario,
        "description": spec["description"],
        "agent_rules": list(RULES[scenario]),
        "controllable_parameters": [
            {
                "name": name,
                "meaning": spec["parameter_descriptions"][name],
                "baseline": spec["baseline"][name],
                "minus": levels[0],
                "plus": levels[2],
            }
            for name, levels in spec["interventions"].items()
        ],
        "raw_field_schema": public_raw_schema(scenario),
    }
    if scenario == "schelling":
        contract["public_environment_structure"] = {
            "periodic_grid": True,
            "district_rows": int(spec.get("district_rows", 3)),
            "district_columns": int(spec.get("district_columns", 3)),
            "district_semantics": (
                "fixed spatial domains defined before simulation; district_id is a "
                "primitive membership label, not an inferred outcome"
            ),
        }
    elif scenario == "deffuant":
        contract["public_environment_structure"] = {
            "initial_network_model": "Watts-Strogatz undirected fixed-edge-count network",
            "initial_network_rewire_probability": float(
                spec["network_rewire_probability"]
            ),
            "adaptive_rewiring_probability": float(
                spec["adaptive_rewiring_probability"]
            ),
            "rewiring_homophily_probability": float(
                spec["rewiring_homophily_probability"]
            ),
            "adaptive_rule": (
                "rejected or backfire encounters can replace the sampled tie; "
                "no self-loop, duplicate edge, or isolated agent is permitted and "
                "edge count is preserved"
            ),
        }
    return contract
