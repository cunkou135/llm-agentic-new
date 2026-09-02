"""Evaluation-only controlled process identities and delayed relations.

This module is intentionally consumed only by simulator internals and final
evaluation. It is never imported by prompt construction, semantic validation,
temporal fitting, bootstrap filtering, or intervention classification.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .dsl import computation_signature


@dataclass(frozen=True)
class ReferenceProcess:
    process_id: str
    scenario: str
    scale: str
    computation: dict[str, Any]
    temporal_aggregation: dict[str, Any]

    @property
    def signature(self) -> dict[str, Any]:
        return {
            **computation_signature(self.computation),
            "temporal_aggregation": self.temporal_aggregation,
        }


@dataclass(frozen=True)
class ReferenceRelation:
    source: str
    target: str
    lag: int
    sign: int
    mechanism: str


def _field(name: str) -> dict[str, Any]:
    return {"op": "field", "name": name}


def _fraction(name: str) -> dict[str, Any]:
    return {"op": "fraction", "input": _field(name), "axis": "agent"}


def _mean(name: str) -> dict[str, Any]:
    return {"op": "mean", "input": _field(name), "axis": "agent"}


def _mean_expression(input_: dict[str, Any]) -> dict[str, Any]:
    return {"op": "mean", "input": input_, "axis": "agent"}


def _channel(index: int) -> dict[str, Any]:
    return {"op": "select", "input": _field("mechanism_channel"), "axis": "channel", "index": index}


def _constant(value: float) -> dict[str, Any]:
    return {"op": "constant", "value": value}


def _scale(input_: dict[str, Any], factor: float) -> dict[str, Any]:
    return {"op": "multiply", "left": input_, "right": _constant(factor)}


def _clip01(input_: dict[str, Any]) -> dict[str, Any]:
    return {"op": "clip", "input": input_, "minimum": 0.0, "maximum": 1.0}


def _district_variance(field: str, reducer: str) -> dict[str, Any]:
    return _clip01(
        _scale(
            {
                "op": "variance",
                "input": {
                    "op": "group_reduce",
                    "values": _field(field),
                    "groups": _field("district_id"),
                    "axis": "agent",
                    "reducer": reducer,
                },
                "axis": "group",
            },
            4.0,
        )
    )


def _neighborhood(values: dict[str, Any], reducer: str) -> dict[str, Any]:
    return {
        "op": "network_neighborhood_reduce",
        "values": values,
        "edges": _field("network_edges"),
        "reducer": reducer,
    }


_MICRO = {
    "schelling": [
        ("s_micro_satisfaction", _fraction("unhappy")),
        ("s_micro_relocation", _fraction("moved")),
        ("s_micro_boundary", _fraction("boundary_agent")),
        ("s_micro_destination_similarity", _mean("destination_similarity")),
    ],
    "deffuant": [
        ("d_micro_assimilation", _fraction("interaction_accepted")),
        ("d_micro_shift", {"op": "mean", "input": {"op": "abs", "input": _field("agent_shift")}, "axis": "agent"}),
        ("d_micro_repulsion", _fraction("interaction_backfire")),
        ("d_micro_rewiring", _fraction("edge_rewired")),
    ],
}


_STRUCTURAL_CONTEXT = {
    "schelling": [
        (
            _district_variance("unhappy", "fraction"),
            _mean("local_similarity"),
        ),
        (
            _district_variance("moved", "fraction"),
            _fraction("moved"),
        ),
        (
            _district_variance("boundary_agent", "fraction"),
            _clip01(
                {
                    "op": "subtract",
                    "left": _constant(1.0),
                    "right": {
                        "op": "spatial_neighbor_similarity",
                        "input": _field("state_grid"),
                    },
                }
            ),
        ),
        (
            _district_variance("destination_similarity", "mean"),
            _mean("destination_similarity"),
        ),
    ],
    "deffuant": [
        (
            _clip01(
                _scale(
                    {
                        "op": "variance",
                        "input": _neighborhood(
                            _field("interaction_accepted"), "fraction"
                        ),
                        "axis": "agent",
                    },
                    4.0,
                )
            ),
            _clip01(
                {
                    "op": "subtract",
                    "left": _constant(1.0),
                    "right": {
                        "op": "std",
                        "input": _field("state_opinion"),
                        "axis": "agent",
                    },
                }
            ),
        ),
        (
            _clip01(
                _scale(
                    {
                        "op": "variance",
                        "input": _neighborhood(
                            {"op": "abs", "input": _field("agent_shift")}, "mean"
                        ),
                        "axis": "agent",
                    },
                    4.0,
                )
            ),
            _clip01(
                {
                    "op": "subtract",
                    "left": _constant(1.0),
                    "right": {
                        "op": "std",
                        "input": _field("state_opinion"),
                        "axis": "agent",
                    },
                }
            ),
        ),
        (
            _clip01(
                {
                    "op": "divide",
                    "left": {
                        "op": "mean",
                        "input": {
                            "op": "distance",
                            "left": _field("state_opinion"),
                            "right": _neighborhood(_field("state_opinion"), "mean"),
                        },
                        "axis": "agent",
                    },
                    "right": _constant(2.0),
                }
            ),
            _clip01(
                _mean_expression({"op": "abs", "input": _field("state_opinion")})
            ),
        ),
        (
            _clip01(
                _scale(
                    {
                        "op": "variance",
                        "input": _neighborhood(_field("edge_rewired"), "fraction"),
                        "axis": "agent",
                    },
                    4.0,
                )
            ),
            _clip01(
                {
                    "op": "divide",
                    "left": {
                        "op": "network_component_count",
                        "edges": _field("network_edges"),
                        "node_count": _field("agent_count"),
                    },
                    "right": _field("agent_count"),
                }
            ),
        ),
    ],
}


def controlled_structural_contexts(
    scenario: str,
) -> tuple[tuple[dict[str, Any], dict[str, Any]], ...]:
    """Return hidden benchmark contexts anchored to actual public structure."""

    return tuple(_STRUCTURAL_CONTEXT[scenario])


_LAGS = {
    "schelling": [(1, 1), (2, 1), (3, 1), (5, -1)],
    "deffuant": [(1, 1), (2, 1), (3, 1), (5, 1)],
}


def reference_processes(scenario: str) -> tuple[ReferenceProcess, ...]:
    prefix = "s" if scenario == "schelling" else "d"
    processes = [
        ReferenceProcess(process_id, scenario, "micro", expression, {"op": "identity"})
        for process_id, expression in _MICRO[scenario]
    ]
    processes.extend(
        ReferenceProcess(
            f"{prefix}_meso_{index}",
            scenario,
            "meso",
            _channel(index),
            {"op": "identity"},
        )
        for index in range(4)
    )
    processes.extend(
        ReferenceProcess(
            f"{prefix}_macro_{index}",
            scenario,
            "macro",
            _channel(index + 4),
            {"op": "identity"},
        )
        for index in range(4)
    )
    return tuple(processes)


def reference_relations(scenario: str) -> tuple[ReferenceRelation, ...]:
    prefix = "s" if scenario == "schelling" else "d"
    mechanism_names = (
        (
            "satisfaction",
            "relocation",
            "interface",
            "homophilic_destination_selection",
        )
        if scenario == "schelling"
        else ("assimilation", "contraction", "repulsion", "adaptive_rewiring")
    )
    relations: list[ReferenceRelation] = []
    for index, ((micro_id, _), (lag, sign), mechanism) in enumerate(
        zip(_MICRO[scenario], _LAGS[scenario], mechanism_names)
    ):
        relations.append(
            ReferenceRelation(micro_id, f"{prefix}_meso_{index}", lag, sign, mechanism)
        )
        relations.append(
            ReferenceRelation(
                f"{prefix}_meso_{index}",
                f"{prefix}_macro_{index}",
                (2, 3, 4, 5)[index],
                1,
                mechanism,
            )
        )
    return tuple(relations)


def disabled_mechanism(scenario: str) -> str:
    return (
        "homophilic_destination_selection"
        if scenario == "schelling"
        else "repulsion"
    )


def mechanism_target_for_variant(scenario: str, mechanism_variant: str) -> str | None:
    """Map a simulator variant to the exact controlled-reference mechanism."""

    if mechanism_variant == "baseline":
        return None
    expected_variant = {
        "schelling": "disable_homophilic_relocation",
        "deffuant": "disable_backfire",
    }[scenario]
    if mechanism_variant != expected_variant:
        raise ValueError(
            f"unknown mechanism variant for {scenario}: {mechanism_variant}"
        )
    return disabled_mechanism(scenario)
