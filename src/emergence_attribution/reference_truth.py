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


def _channel(index: int) -> dict[str, Any]:
    return {"op": "select", "input": _field("mechanism_channel"), "axis": "channel", "index": index}


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
        ("d_micro_rejection", _fraction("interaction_rejected")),
    ],
}


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
        ("satisfaction", "relocation", "interface", "mixing")
        if scenario == "schelling"
        else ("assimilation", "contraction", "repulsion", "rejection")
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
    return "relocation" if scenario == "schelling" else "repulsion"
