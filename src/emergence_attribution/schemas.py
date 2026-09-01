"""Strict structured objects for semantic hypotheses and prospective tests."""

from __future__ import annotations

import re
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


Scale = Literal["micro", "meso", "macro"]
Direction = Literal["increase", "decrease", "mixed", "unknown"]


class TemporalAggregationSpec(BaseModel):
    """Strict temporal transformation applied after an indicator is computed."""

    model_config = ConfigDict(extra="forbid")

    op: Literal["identity", "rolling_mean", "difference", "cumulative_mean"]
    window: int | None = None

    @model_validator(mode="after")
    def validate_window(self) -> "TemporalAggregationSpec":
        if self.op == "rolling_mean":
            if self.window is None or isinstance(self.window, bool) or self.window <= 0:
                raise ValueError("rolling_mean requires a positive integer window")
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
    model_config = ConfigDict(extra="forbid")

    id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    semantic_name: str = Field(min_length=3)
    scientific_definition: str = Field(min_length=12)
    phenomenon: str = Field(min_length=3)
    scale: Scale
    branch_id: str = Field(pattern=r"^[a-z][a-z0-9_]{1,31}$")
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


class CandidateEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str
    expected_direction: Direction
    rationale: str = Field(min_length=12)


class RequiredCandidateEdge(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str
    target: str


class ProspectiveValidationCriteria(BaseModel):
    model_config = ConfigDict(extra="forbid")

    required_source_response: bool = True
    required_downstream_response: list[bool] = Field(min_length=1)
    required_temporal_order: bool = True
    required_candidate_edges: list[RequiredCandidateEdge] = Field(min_length=1)


class StructuredRepresentation(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scenario: str
    phenomenon: str = Field(min_length=3)
    indicators: list[IndicatorSpec]
    candidate_edges: list[CandidateEdge]
    interpretation_boundary: str = Field(min_length=30)

    @model_validator(mode="after")
    def structural_integrity(self) -> "StructuredRepresentation":
        identifiers = [item.id for item in self.indicators]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("duplicate indicator id")
        lookup = {item.id: item for item in self.indicators}
        pairs: set[tuple[str, str]] = set()
        for edge in self.candidate_edges:
            if edge.source not in lookup or edge.target not in lookup:
                raise ValueError("candidate edge references an unknown indicator")
            source = lookup[edge.source]
            target = lookup[edge.target]
            if source.branch_id != target.branch_id:
                raise ValueError("candidate edge crosses branches")
            if (source.scale, target.scale) not in {
                ("micro", "meso"),
                ("meso", "macro"),
            }:
                raise ValueError("candidate edge violates adjacent-scale order")
            pair = (edge.source, edge.target)
            if pair in pairs:
                raise ValueError("duplicate candidate edge")
            pairs.add(pair)
        return self


class ProspectivePrediction(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prediction_id: str = Field(pattern=r"^[a-z][a-z0-9_]{2,63}$")
    phenomenon: str
    parameter: str
    intervention_direction: Literal["minus", "plus"]
    source_indicator: str
    expected_source_direction: Literal["increase", "decrease"]
    downstream_indicators: list[str] = Field(min_length=1)
    expected_downstream_direction: list[Literal["increase", "decrease"]] = Field(
        min_length=1
    )
    expected_temporal_order: list[str] = Field(min_length=2)
    validation_criteria: ProspectiveValidationCriteria
    scientific_rationale: str = Field(min_length=12)
    falsification_condition: str = Field(min_length=12)

    @model_validator(mode="after")
    def matching_downstream_lengths(self) -> "ProspectivePrediction":
        if len(self.downstream_indicators) != len(
            self.expected_downstream_direction
        ):
            raise ValueError("downstream indicators and directions must have equal length")
        if len(self.validation_criteria.required_downstream_response) != len(
            self.downstream_indicators
        ):
            raise ValueError(
                "required downstream responses and downstream indicators must have equal length"
            )
        return self


class SemanticGeneration(BaseModel):
    model_config = ConfigDict(extra="forbid")

    representation: StructuredRepresentation
    prospective_predictions: list[ProspectivePrediction] = Field(min_length=1)


def identifier_slug(value: str) -> str:
    """Create a stable safe identifier for generated labels."""

    slug = re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")
    if not slug or not slug[0].isalpha():
        slug = f"item_{slug}"
    return slug[:64]
