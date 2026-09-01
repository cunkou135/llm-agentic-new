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


RAW_SCHEMAS: dict[str, list[dict[str, Any]]] = {
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
        {
            "field_name": "mechanism_channel",
            "dtype": "float32",
            "shape": ["time", "channel"],
            "semantic_meaning": "instrumented continuous internal organisation states recorded by the controlled simulator",
            "entity_level": "system channel",
            "channel_semantics": [
                "local satisfaction coordination",
                "relocation organisation",
                "spatial interface organisation",
                "group mixing organisation",
                "system segregation",
                "cluster concentration",
                "interface permeability",
                "collective spatial integration",
            ],
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
            "shape": ["edge", "endpoint"],
            "semantic_meaning": "undirected static interaction-network endpoint pairs",
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
        {
            "field_name": "mechanism_channel",
            "dtype": "float32",
            "shape": ["time", "channel"],
            "semantic_meaning": "instrumented continuous internal organisation states recorded by the controlled simulator",
            "entity_level": "system channel",
            "channel_semantics": [
                "local assimilation coordination",
                "opinion-update contraction",
                "repulsive interaction organisation",
                "interaction rejection organisation",
                "population consensus",
                "opinion-cluster concentration",
                "population extremity",
                "collective opinion integration",
            ],
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
            "field_name": "y",
            "dtype": "float64",
            "shape": ["time"],
            "semantic_meaning": "toy aggregate measurement",
            "entity_level": "system",
        },
    ],
}


RULES = {
    "schelling": [
        "Each occupied cell contains one fixed-group agent and the grid is periodic.",
        "An agent is unsatisfied when its occupied-neighbour same-group fraction is below tolerance.",
        "An unsatisfied agent attempts movement with move_probability.",
        "A moving agent samples vacancies and, with destination_preference, chooses the sampled vacancy with the highest local similarity.",
        "Disabling homophilic relocation makes destination choice non-preferential while preserving other rules.",
    ],
    "deffuant": [
        "Each agent samples one network neighbour per step.",
        "Opinion distance within confidence_bound produces assimilation proportional to assimilation_strength.",
        "Distance at least backfire_threshold produces a weak repulsive update when the mechanism is enabled.",
        "Other encounters are rejected and opinions remain bounded to [-1, 1].",
        "Disabling backfire sets repulsive update strength to zero while preserving other rules.",
    ],
}


def raw_schema(scenario: str) -> list[dict[str, Any]]:
    return deepcopy(RAW_SCHEMAS[scenario])


def prompt_scenario_contract(scenario: str, spec: dict[str, Any]) -> dict[str, Any]:
    return {
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
        "raw_field_schema": raw_schema(scenario),
    }
