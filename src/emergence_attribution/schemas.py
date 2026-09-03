"""Strict, data-blind contracts for the two semantic phases.

Indicator generation is separated from complete mechanism hypothesis
generation. The first response cannot contain edges, paths, or predictions;
the second can only reference the immutable indicator set.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Scale = Literal["micro", "meso", "macro"]
Direction = Literal["increase", "decrease", "mixed", "unknown"]
SignedDirection = Literal["increase", "decrease"]
EntityScope = Literal[
    "individual", "interaction", "elementary_event", "local_process",
    "neighborhood", "district", "community", "cluster", "local_domain",
    "whole_system",
]


class TemporalAggregationSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")

    op: Literal[
        "identity", "rolling_mean", "rolling_std", "difference", "cumulative_mean"
    ]
    window: int | None = None

    @model_validator(mode="after")
    def validate_window(self) -> "TemporalAggregationSpec":
        if self.op in {"rolling_mean", "rolling_std"}:
            if self.window is None or isinstance(self.window, bool) or self.window <= 0:
                raise ValueError(f"{self.op} requires a positive integer window")
        elif self.window is not None:
            raise ValueError(f"{self.op} does not accept window")
        return self


class ParameterAssociation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    parameter: str
    relationship: Literal["direct", "indirect"] = "direct"
    expected_indicator_direction: Direction
    rationale: str = Field(min_length=12)


class IndicatorSpec(BaseModel):
    """One executable observable with no LLM-authored grouping field."""

    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    semantic_name: str = Field(min_length=3)
    scientific_definition: str = Field(min_length=12)
    phenomenon: str = Field(min_length=3)
    scale: Scale
    entity_scope: EntityScope
    entities: str = Field(min_length=2)
    source_fields: list[str] = Field(min_length=1)
    computation: dict[str, Any]
    temporal_aggregation: TemporalAggregationSpec
    parameter_associations: list[ParameterAssociation] = Field(default_factory=list)
    scientific_rationale: str = Field(min_length=12)

    @model_validator(mode="after")
    def unique_fields_and_associations(self) -> "IndicatorSpec":
        if len(self.source_fields) != len(set(self.source_fields)):
            raise ValueError("source_fields must be unique")
        parameters = [item.parameter for item in self.parameter_associations]
        if len(parameters) != len(set(parameters)):
            raise ValueError("parameter associations must be unique per indicator")
        return self


class StructuredIndicatorSet(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str
    phenomenon: str = Field(min_length=3)
    indicators: list[IndicatorSpec]
    interpretation_boundary: str = Field(min_length=30)

    @model_validator(mode="after")
    def unique_indicator_ids(self) -> "StructuredIndicatorSet":
        identifiers = [item.id for item in self.indicators]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate indicator id")
        return self


class IndicatorGeneration(StructuredIndicatorSet):
    """Phase A response; forbidden extras reject paths, edges, and predictions."""


class CandidatePath(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,95}$")
    parameter: str
    intervention_direction: Literal["minus", "plus"]
    micro_indicator: str
    meso_indicator: str
    macro_indicator: str
    micro_to_meso_expected_direction: SignedDirection
    meso_to_macro_expected_direction: SignedDirection
    expected_micro_response: SignedDirection
    expected_meso_response: SignedDirection
    expected_macro_response: SignedDirection
    scientific_rationale: str = Field(min_length=12)
    mechanistic_explanation: str = Field(min_length=12)
    falsification_condition: str = Field(min_length=12)


class ProspectivePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,95}$")
    candidate_path_id: str
    prospective_priority: int = Field(default=0, ge=0)
    scientific_rationale: str = Field(min_length=12)
    falsification_condition: str = Field(min_length=12)


class PathGeneration(BaseModel):
    """Phase B response over one immutable indicator set."""

    model_config = ConfigDict(extra="forbid")

    scenario: str
    indicator_set_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    candidate_paths: list[CandidatePath]
    prospective_predictions: list[ProspectivePrediction]

    @model_validator(mode="after")
    def unique_paths_and_bound_predictions(self) -> "PathGeneration":
        identifiers = [item.path_id for item in self.candidate_paths]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate candidate path id")
        triples = [
            (item.micro_indicator, item.meso_indicator, item.macro_indicator)
            for item in self.candidate_paths
        ]
        if len(triples) != len(set(triples)):
            raise ValueError("duplicate indicator triple")
        path_ids = set(identifiers)
        prediction_ids = [item.prediction_id for item in self.prospective_predictions]
        if len(prediction_ids) != len(set(prediction_ids)):
            raise ValueError("duplicate prospective prediction id")
        unknown = sorted(
            item.candidate_path_id
            for item in self.prospective_predictions
            if item.candidate_path_id not in path_ids
        )
        if unknown:
            raise ValueError(f"prospective prediction references unknown path: {unknown}")
        return self


class CandidateEdge(BaseModel):
    """Deterministic projection of one or more frozen CandidatePath objects."""

    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    expected_direction: Direction
    hypothesis_group_ids: list[str] = Field(default_factory=lambda: ["legacy_group"], min_length=1)
    path_ids: list[str] = Field(default_factory=lambda: ["legacy_unbound"], min_length=1)

class StructuredRepresentation(StructuredIndicatorSet):
    """Internal downstream view assembled after both semantic phases freeze."""

    candidate_paths: list[CandidatePath] = Field(default_factory=list)
    candidate_edges: list[CandidateEdge]

    @model_validator(mode="after")
    def references_are_frozen_and_adjacent(self) -> "StructuredRepresentation":
        lookup = {item.id: item for item in self.indicators}
        path_ids = {item.path_id for item in self.candidate_paths}
        seen: set[tuple[str, str]] = set()
        for edge in self.candidate_edges:
            if edge.source not in lookup or edge.target not in lookup:
                raise ValueError("candidate edge references an unknown indicator")
            if (lookup[edge.source].scale, lookup[edge.target].scale) not in {
                ("micro", "meso"), ("meso", "macro")
            }:
                raise ValueError("candidate edge violates adjacent-scale order")
            key = (edge.source, edge.target)
            if key in seen:
                raise ValueError("duplicate derived candidate edge")
            seen.add(key)
            if path_ids and not set(edge.path_ids).issubset(path_ids):
                raise ValueError("candidate edge references an unknown path")
        return self


class SemanticGeneration(BaseModel):
    """Deprecated import compatibility; never used as an LLM output schema."""

    model_config = ConfigDict(extra="forbid")
    representation: dict[str, Any]
    prospective_predictions: list[dict[str, Any]]


def identifier_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"item_{slug}"
    return slug[:64]
