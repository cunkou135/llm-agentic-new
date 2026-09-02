"""Safe declarative indicator computation with no dynamic code execution."""

from __future__ import annotations

import json
import math
import warnings
from dataclasses import dataclass
from typing import Any, Iterable

import networkx as nx
import numpy as np


class DSLValidationError(ValueError):
    pass


@dataclass(frozen=True)
class ValueType:
    dtype: str
    dimensions: tuple[str, ...]


_REDUCERS = {
    "mean", "sum", "count", "fraction", "variance", "std", "quantile",
    "entropy", "binned_entropy",
}
_GROUP_REDUCERS = {"mean", "sum", "count", "fraction", "variance", "std", "entropy"}
_NETWORK_REDUCERS = {"mean", "sum", "count", "fraction", "variance", "std"}
_BINARY = {"add", "subtract", "multiply", "divide", "safe_ratio", "distance"}
_UNARY = {"abs", "negate", "sqrt", "log1p"}
_COMPARISONS = {"greater", "greater_equal", "less", "less_equal", "equal", "not_equal"}
_OPERATORS = {
    "field",
    "constant",
    "clip",
    "where",
    "time_difference",
    "rolling_mean",
    "rolling_std",
    "select",
    "group_reduce",
    "connected_component_count",
    "largest_component_fraction",
    "spatial_neighbor_similarity",
    "network_assortativity",
    "network_density",
    "network_neighborhood_reduce",
    "network_component_count",
    "network_largest_component_fraction",
} | _REDUCERS | _BINARY | _UNARY | _COMPARISONS
_OPERATORS.add("correlation")

# Scientific-scope classes are deliberately separate from type/shape checking.
# A scalar [time] output does not by itself establish a scientific scale.
ELEMENTARY_OR_LOCAL_OPERATORS = frozenset(
    {
        "field",
        "where",
        "select",
        "correlation",
    }
)
GENUINE_MESO_OPERATORS = frozenset(
    {
        "group_reduce",
        "network_neighborhood_reduce",
    }
)
GLOBAL_STRUCTURE_OPERATORS = frozenset(
    {
        "spatial_neighbor_similarity",
        "connected_component_count",
        "largest_component_fraction",
        "network_assortativity",
        "network_density",
        "network_component_count",
        "network_largest_component_fraction",
    }
)
TRIVIAL_WRAPPERS = frozenset(
    {
        "time_difference",
        "rolling_mean",
        "rolling_std",
        "negate",
        "sqrt",
        "log1p",
        "clip",
    }
)

# Backward-compatible public name.  Whole-system operators are intentionally no
# longer classified as mesoscopic structure.
MESOSCOPIC_STRUCTURE_OPERATORS = GENUINE_MESO_OPERATORS

_ORGANIZATION_REDUCERS = frozenset(
    {"std", "variance", "entropy", "binned_entropy"}
)
_NONTRIVIAL_WITHIN_STRUCTURE_REDUCERS = frozenset(
    {"std", "variance", "entropy"}
)
_NORMALIZER_PRIMITIVE_FAMILIES = frozenset(
    {"population_size", "simulation_length"}
)


def grammar_description() -> dict[str, Any]:
    """Machine-readable grammar included in model prompts."""

    contracts: dict[str, dict[str, Any]] = {}
    for op in sorted(_OPERATORS):
        required = ["op"]
        optional: list[str] = []
        types: dict[str, str] = {"op": "string literal"}
        axis = "not applicable"
        example: dict[str, Any] = {"op": op}
        output = "inherits the validated input dimensions"
        if op == "field":
            required += ["name"]
            types["name"] = "public raw field name"
            example["name"] = "local_similarity"
            output = "raw field dtype and dimensions"
        elif op == "constant":
            required += ["value"]
            types["value"] = "finite number or boolean"
            example["value"] = 0.5
            output = "scalar"
        elif op in _UNARY | {"clip", "time_difference", "rolling_mean", "rolling_std"}:
            required += ["input"]
            types["input"] = "AST object"
            example["input"] = {"op": "field", "name": "local_similarity"}
            if op == "clip":
                required += ["minimum", "maximum"]
                types.update({"minimum": "number", "maximum": "number"})
                example.update({"minimum": 0.0, "maximum": 1.0})
            if op in {"rolling_mean", "rolling_std"}:
                required += ["window"]
                types["window"] = "positive integer"
                example["window"] = 3
        elif op in _BINARY | _COMPARISONS:
            required += ["left", "right"]
            types.update({"left": "AST object", "right": "AST object"})
            example.update({
                "left": {"op": "field", "name": "unhappy_count"},
                "right": {"op": "constant", "value": 1},
            })
            output = "boolean dimensions" if op in _COMPARISONS else output
        elif op in _REDUCERS:
            required += ["input", "axis"]
            types.update({"input": "AST object", "axis": "named input dimension"})
            axis = "required; removed from output"
            example.update({
                "input": {"op": "field", "name": "local_similarity"},
                "axis": "agent",
            })
            if op == "quantile":
                required += ["q"]
                types["q"] = "number in [0,1]"
                example["q"] = 0.5
            if op == "binned_entropy":
                required += ["bins"]
                types["bins"] = "integer in [2,128]"
                example["bins"] = 10
            output = "numeric input dimensions with axis removed"
        elif op == "where":
            required += ["condition", "input"]
            types.update({"condition": "boolean AST object", "input": "AST object"})
            example.update({
                "condition": {"op": "field", "name": "unhappy"},
                "input": {"op": "field", "name": "local_similarity"},
            })
        elif op == "select":
            required += ["input", "axis", "index"]
            types.update({"input": "AST object", "axis": "named input dimension", "index": "non-negative integer"})
            axis = "required; selected and removed from output"
            example.update({"input": {"op": "field", "name": "agent_position"}, "axis": "coordinate", "index": 0})
        elif op == "group_reduce":
            required += ["values", "groups", "axis", "reducer"]
            types.update(
                {
                    "values": "numeric, integer, or boolean AST object",
                    "groups": "integer group-membership AST object",
                    "axis": "shared entity dimension",
                    "reducer": f"one of {sorted(_GROUP_REDUCERS)}",
                }
            )
            axis = "required entity axis; replaced by a generic group dimension"
            example.update(
                {
                    "values": {"op": "field", "name": "agent_group"},
                    "groups": {"op": "field", "name": "district_id"},
                    "axis": "agent",
                    "reducer": "mean",
                }
            )
            output = "numeric dimensions with entity axis replaced by group"
        elif op == "correlation":
            required += ["left", "right", "axis"]
            types.update({"left": "numeric AST object", "right": "numeric AST object", "axis": "shared named dimension"})
            axis = "required; removed from output"
            example.update({"left": {"op": "field", "name": "state_opinion"}, "right": {"op": "field", "name": "agent_shift"}, "axis": "agent"})
        elif op in {"connected_component_count", "largest_component_fraction", "spatial_neighbor_similarity"}:
            required += ["input"]
            types["input"] = "time-grid AST object"
            example["input"] = {"op": "field", "name": "state_grid"}
            output = "numeric [time]"
        elif op == "network_assortativity":
            required += ["values", "edges"]
            types.update({"values": "numeric [time,agent] AST", "edges": "integer [time,edge,endpoint] AST"})
            example.update({"values": {"op": "field", "name": "state_opinion"}, "edges": {"op": "field", "name": "network_edges"}})
            output = "numeric [time]"
        elif op == "network_density":
            required += ["edges", "node_count"]
            types.update({"edges": "integer [time,edge,endpoint] AST", "node_count": "scalar AST"})
            example.update({"edges": {"op": "field", "name": "network_edges"}, "node_count": {"op": "field", "name": "agent_count"}})
            output = "numeric [time] for dynamic edges"
        elif op == "network_neighborhood_reduce":
            required += ["values", "edges", "reducer"]
            types.update(
                {
                    "values": "numeric, integer, or boolean [time,agent] AST",
                    "edges": "integer [time,edge,endpoint] AST",
                    "reducer": f"one of {sorted(_NETWORK_REDUCERS)}",
                }
            )
            example.update(
                {
                    "values": {"op": "field", "name": "state_opinion"},
                    "edges": {"op": "field", "name": "network_edges"},
                    "reducer": "mean",
                }
            )
            output = "numeric [time,agent] neighborhood statistic"
        elif op in {"network_component_count", "network_largest_component_fraction"}:
            required += ["edges", "node_count"]
            types.update(
                {
                    "edges": "integer [time,edge,endpoint] AST",
                    "node_count": "positive scalar AST",
                }
            )
            example.update(
                {
                    "edges": {"op": "field", "name": "network_edges"},
                    "node_count": {"op": "field", "name": "agent_count"},
                }
            )
            output = "numeric [time]"
        contracts[op] = {
            "required": required,
            "optional": optional,
            "types": types,
            "axis_semantics": axis,
            "output": output,
            "example": example,
        }
    return {
        "ast_rule": "Every node is a JSON object with an op. No source code or expression strings are allowed.",
        "operators": contracts,
        "required_output": "A scalar time series with dimensions [time].",
    }


def field_types(schema: Iterable[dict[str, Any]]) -> dict[str, ValueType]:
    result: dict[str, ValueType] = {}
    for item in schema:
        shape = tuple(str(value) for value in item["shape"] if isinstance(value, str))
        raw_dtype = str(item["dtype"]).lower()
        if raw_dtype == "bool":
            dtype = "bool"
        elif raw_dtype.startswith(("int", "uint")):
            dtype = "integer"
        else:
            dtype = "numeric"
        result[str(item["field_name"])] = ValueType(dtype, shape)
    return result


def expression_fields(expression: dict[str, Any]) -> set[str]:
    fields: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("op") == "field" and isinstance(value.get("name"), str):
                fields.add(value["name"])
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(expression)
    return fields


def canonical_expression(expression: dict[str, Any]) -> str:
    return json.dumps(expression, sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def computation_signature(expression: dict[str, Any]) -> dict[str, Any]:
    return {
        "expression": canonical_expression(expression),
        "source_fields": sorted(expression_fields(expression)),
    }


def expression_operators(expression: dict[str, Any]) -> set[str]:
    operators: set[str] = set()

    def visit(value: Any) -> None:
        if isinstance(value, dict):
            if isinstance(value.get("op"), str):
                operators.add(value["op"])
            for nested in value.values():
                visit(nested)
        elif isinstance(value, list):
            for nested in value:
                visit(nested)

    visit(expression)
    return operators


def mesoscopic_structure_operators(expression: dict[str, Any]) -> set[str]:
    return expression_operators(expression) & MESOSCOPIC_STRUCTURE_OPERATORS


def global_structure_operators(expression: dict[str, Any]) -> set[str]:
    """Return inherently whole-grid/whole-network operators in an AST."""

    return expression_operators(expression) & GLOBAL_STRUCTURE_OPERATORS


def _contains_operator(expression: Any, operators: frozenset[str]) -> bool:
    if not isinstance(expression, dict):
        return False
    if expression.get("op") in operators:
        return True
    return any(
        _contains_operator(value, operators)
        for value in expression.values()
        if isinstance(value, dict)
    )


def _is_quantile_range(expression: dict[str, Any]) -> bool:
    if expression.get("op") not in {"subtract", "distance"}:
        return False
    left = expression.get("left")
    right = expression.get("right")
    if not isinstance(left, dict) or not isinstance(right, dict):
        return False
    if left.get("op") != "quantile" or right.get("op") != "quantile":
        return False
    left_q, right_q = left.get("q"), right.get("q")
    if not isinstance(left_q, (int, float)) or not isinstance(right_q, (int, float)):
        return False
    return (
        float(left_q) != float(right_q)
        and _contains_operator(left, GENUINE_MESO_OPERATORS)
        and _contains_operator(right, GENUINE_MESO_OPERATORS)
    )


def _contains_local_structural_contrast(expression: Any) -> bool:
    """Detect an explicit entity-versus-neighborhood/group contrast."""

    if not isinstance(expression, dict):
        return False
    if expression.get("op") in {"distance", "subtract"}:
        left = expression.get("left")
        right = expression.get("right")
        left_structural = _contains_operator(left, GENUINE_MESO_OPERATORS)
        right_structural = _contains_operator(right, GENUINE_MESO_OPERATORS)
        nonstructural = right if left_structural else left
        if (
            left_structural != right_structural
            and _contains_operator(nonstructural, frozenset({"field"}))
        ):
            return True
    return any(
        _contains_local_structural_contrast(value)
        for value in expression.values()
        if isinstance(value, dict)
    )


def is_genuine_meso_expression(expression: dict[str, Any]) -> bool:
    """Whether an AST measures organization across real intermediate entities.

    A group/neighborhood primitive is necessary but not sufficient.  The final
    statistic must express between-entity heterogeneity, within-entity
    heterogeneity, a quantile range, or an explicit local-versus-structural
    contrast.  Merely averaging/summing group or neighborhood means is rejected.
    """

    if global_structure_operators(expression):
        return False
    if not _contains_operator(expression, GENUINE_MESO_OPERATORS):
        return False
    if _is_quantile_range(expression):
        return True
    if _contains_local_structural_contrast(expression):
        return True

    def visit(node: Any) -> bool:
        if not isinstance(node, dict):
            return False
        op = node.get("op")
        if _is_quantile_range(node):
            return True
        if op in _ORGANIZATION_REDUCERS and _contains_operator(
            node.get("input"), GENUINE_MESO_OPERATORS
        ):
            return True
        if (
            op in GENUINE_MESO_OPERATORS
            and node.get("reducer") in _NONTRIVIAL_WITHIN_STRUCTURE_REDUCERS
        ):
            return True
        return any(
            visit(value) for value in node.values() if isinstance(value, dict)
        )

    return visit(expression)


def _constant_node(value: Any) -> bool:
    return isinstance(value, dict) and value.get("op") == "constant"


def strip_trivial_wrappers(expression: dict[str, Any]) -> dict[str, Any]:
    """Strip temporal and one-to-one mathematical wrappers from an AST lineage."""

    node = expression
    while isinstance(node, dict):
        op = node.get("op")
        if op in TRIVIAL_WRAPPERS:
            node = node.get("input")
            continue
        if op in {"add", "subtract", "multiply", "divide", "safe_ratio"}:
            left, right = node.get("left"), node.get("right")
            if _constant_node(left) and isinstance(right, dict):
                node = right
                continue
            if _constant_node(right) and isinstance(left, dict):
                node = left
                continue
        break
    if not isinstance(node, dict):
        return {"op": "invalid_lineage"}
    return node


def primitive_family_metadata(
    schema: Iterable[dict[str, Any]],
) -> dict[str, dict[str, str]]:
    """Public raw-field lineage metadata keyed by field name."""

    return {
        str(item["field_name"]): {
            "primitive_family": str(item.get("primitive_family", item["field_name"])),
            "statistic_role": str(item.get("statistic_role", "primitive")),
        }
        for item in schema
    }


def expression_primitive_families(
    expression: dict[str, Any], schema: Iterable[dict[str, Any]]
) -> set[str]:
    metadata = primitive_family_metadata(schema)
    return {
        metadata[name]["primitive_family"]
        for name in expression_fields(expression)
        if name in metadata
        and metadata[name]["primitive_family"] not in _NORMALIZER_PRIMITIVE_FAMILIES
    }


def _family_lineage_ast(
    expression: dict[str, Any], metadata: dict[str, dict[str, str]]
) -> dict[str, Any]:
    """Replace public field aliases and equivalent count/rate forms canonically."""

    node = strip_trivial_wrappers(expression)
    op = node.get("op")
    if op == "field":
        field = str(node.get("name"))
        item = metadata.get(
            field, {"primitive_family": field, "statistic_role": "primitive"}
        )
        return {
            "op": "primitive",
            "family": item["primitive_family"],
            "role": item["statistic_role"],
        }
    if op == "fraction":
        nested = node.get("input")
        if isinstance(nested, dict):
            nested_ast = _family_lineage_ast(nested, metadata)
            if (
                nested_ast.get("op") == "primitive"
                and nested_ast.get("role")
                in {"elementary_event", "interaction_event", "binary_state"}
            ):
                families = sorted(
                    family
                    for family in _family_names_from_ast(nested_ast)
                    if family not in _NORMALIZER_PRIMITIVE_FAMILIES
                )
                if families:
                    return {"op": "primitive_rate", "families": families}
    if op in {"divide", "safe_ratio"}:
        left = node.get("left")
        right = node.get("right")
        if isinstance(left, dict) and isinstance(right, dict):
            left_ast = _family_lineage_ast(left, metadata)
            right_ast = _family_lineage_ast(right, metadata)
            left_families = sorted(
                family
                for family in _family_names_from_ast(left_ast)
                if family not in _NORMALIZER_PRIMITIVE_FAMILIES
            )
            right_families = _family_names_from_ast(right_ast)
            left_is_count = (
                left_ast.get("op") == "primitive"
                and left_ast.get("role") == "aggregate_count"
            )
            if (
                left_is_count
                and left_families
                and right_families
                and right_families <= _NORMALIZER_PRIMITIVE_FAMILIES
            ):
                return {"op": "primitive_rate", "families": left_families}
    result: dict[str, Any] = {}
    for key, value in node.items():
        if isinstance(value, dict):
            result[key] = _family_lineage_ast(value, metadata)
        else:
            result[key] = value
    return result


def _family_names_from_ast(expression: Any) -> set[str]:
    if not isinstance(expression, dict):
        return set()
    families: set[str] = set()
    family = expression.get("family")
    if isinstance(family, str):
        families.add(family)
    values = expression.get("families")
    if isinstance(values, list):
        families.update(str(value) for value in values)
    for value in expression.values():
        if isinstance(value, dict):
            families.update(_family_names_from_ast(value))
    return families


def canonical_source_family_lineage(
    expression: dict[str, Any],
    temporal_aggregation: dict[str, Any],
    schema: Iterable[dict[str, Any]],
) -> str:
    aggregation_op = temporal_aggregation.get("op")
    if aggregation_op not in {
        "identity",
        "rolling_mean",
        "rolling_std",
        "difference",
        "cumulative_mean",
    }:
        raise DSLValidationError(f"unknown temporal lineage wrapper: {aggregation_op!r}")
    return canonical_expression(
        _family_lineage_ast(expression, primitive_family_metadata(schema))
    )


def is_trivial_micro_macro_lineage(
    micro_expression: dict[str, Any],
    micro_aggregation: dict[str, Any],
    macro_expression: dict[str, Any],
    macro_aggregation: dict[str, Any],
    schema: Iterable[dict[str, Any]],
) -> bool:
    """Reject a complete path whose Macro is only the Micro in another guise."""

    if canonical_computational_lineage(
        micro_expression, micro_aggregation
    ) == canonical_computational_lineage(macro_expression, macro_aggregation):
        return True
    return canonical_source_family_lineage(
        micro_expression, micro_aggregation, schema
    ) == canonical_source_family_lineage(
        macro_expression, macro_aggregation, schema
    )


def canonical_computational_lineage(
    expression: dict[str, Any], temporal_aggregation: dict[str, Any]
) -> str:
    """Canonical core after removing scale-irrelevant temporal/math wrappers."""

    aggregation_op = temporal_aggregation.get("op")
    if aggregation_op not in {
        "identity",
        "rolling_mean",
        "rolling_std",
        "difference",
        "cumulative_mean",
    }:
        raise DSLValidationError(f"unknown temporal lineage wrapper: {aggregation_op!r}")
    return canonical_expression(strip_trivial_wrappers(expression))


def is_trivial_cross_scale_transform(
    source_expression: dict[str, Any],
    source_aggregation: dict[str, Any],
    target_expression: dict[str, Any],
    target_aggregation: dict[str, Any],
) -> bool:
    source_core = canonical_computational_lineage(
        source_expression, source_aggregation
    )
    target_core = canonical_computational_lineage(
        target_expression, target_aggregation
    )
    if source_core != target_core:
        return False
    source_structure = mesoscopic_structure_operators(source_expression)
    target_structure = mesoscopic_structure_operators(target_expression)
    return not bool(target_structure - source_structure)


def _same_or_scalar(left: ValueType, right: ValueType) -> tuple[str, ...]:
    if not left.dimensions:
        return right.dimensions
    if not right.dimensions:
        return left.dimensions
    if left.dimensions != right.dimensions:
        raise DSLValidationError(
            f"incompatible dimensions {left.dimensions} and {right.dimensions}"
        )
    return left.dimensions


def validate_expression(expression: Any, fields: dict[str, ValueType]) -> ValueType:
    if not isinstance(expression, dict):
        raise DSLValidationError("each AST node must be an object")
    op = expression.get("op")
    if op not in _OPERATORS:
        raise DSLValidationError(f"illegal computation operator: {op!r}")
    if op == "field":
        name = expression.get("name")
        if name not in fields:
            raise DSLValidationError(f"unknown raw field: {name!r}")
        return fields[name]
    if op == "constant":
        if not isinstance(expression.get("value"), (int, float, bool)):
            raise DSLValidationError("constant value must be numeric or boolean")
        if not isinstance(expression["value"], bool) and not math.isfinite(
            float(expression["value"])
        ):
            raise DSLValidationError("constant value must be finite")
        if isinstance(expression["value"], bool):
            dtype = "bool"
        elif isinstance(expression["value"], int):
            dtype = "integer"
        else:
            dtype = "numeric"
        return ValueType(dtype, ())
    if op in _UNARY:
        value = validate_expression(expression.get("input"), fields)
        if value.dtype not in {"numeric", "integer"}:
            raise DSLValidationError(f"{op} requires numeric input")
        return value
    if op in _BINARY or op in _COMPARISONS:
        left = validate_expression(expression.get("left"), fields)
        right = validate_expression(expression.get("right"), fields)
        dimensions = _same_or_scalar(left, right)
        if op not in _COMPARISONS and (
            left.dtype not in {"numeric", "integer"}
            or right.dtype not in {"numeric", "integer"}
        ):
            raise DSLValidationError(f"{op} requires numeric inputs")
        return ValueType("bool" if op in _COMPARISONS else "numeric", dimensions)
    if op == "correlation":
        left = validate_expression(expression.get("left"), fields)
        right = validate_expression(expression.get("right"), fields)
        if left.dtype not in {"numeric", "integer"} or right.dtype not in {"numeric", "integer"}:
            raise DSLValidationError("correlation requires numeric inputs")
        if left.dimensions != right.dimensions:
            raise DSLValidationError("correlation inputs must have equal dimensions")
        axis = expression.get("axis")
        if axis not in left.dimensions:
            raise DSLValidationError("correlation axis is not present")
        dimensions = list(left.dimensions)
        dimensions.remove(axis)
        return ValueType("numeric", tuple(dimensions))
    if op in _REDUCERS:
        value = validate_expression(expression.get("input"), fields)
        axis = expression.get("axis")
        if axis not in value.dimensions:
            raise DSLValidationError(f"axis {axis!r} is not present in {value.dimensions}")
        if op == "entropy" and value.dtype not in {"bool", "integer"}:
            raise DSLValidationError(
                "entropy accepts only categorical, boolean, or integer-like input; use binned_entropy for continuous values"
            )
        if op == "binned_entropy" and value.dtype not in {"numeric", "integer"}:
            raise DSLValidationError("binned_entropy requires numeric input")
        if op not in {"count", "fraction", "entropy", "binned_entropy"} and value.dtype not in {"numeric", "integer"}:
            raise DSLValidationError(f"{op} requires numeric input")
        dimensions = list(value.dimensions)
        dimensions.remove(axis)
        if op == "quantile":
            q = expression.get("q")
            if not isinstance(q, (int, float)) or not 0 <= float(q) <= 1:
                raise DSLValidationError("quantile q must be within [0, 1]")
        if op == "binned_entropy":
            bins = expression.get("bins")
            if isinstance(bins, bool) or not isinstance(bins, int) or not 2 <= bins <= 128:
                raise DSLValidationError("binned_entropy bins must be an integer in [2, 128]")
        return ValueType("numeric", tuple(dimensions))
    if op == "group_reduce":
        values = validate_expression(expression.get("values"), fields)
        groups = validate_expression(expression.get("groups"), fields)
        axis = expression.get("axis")
        reducer = expression.get("reducer")
        if groups.dtype != "integer":
            raise DSLValidationError("group_reduce groups must be integer labels")
        if axis not in groups.dimensions or axis not in values.dimensions:
            raise DSLValidationError("group_reduce axis must be present in values and groups")
        broadcast_static_values = (
            values.dimensions == (axis,)
            and groups.dimensions == ("time", axis)
        )
        if values.dimensions != groups.dimensions and not broadcast_static_values:
            raise DSLValidationError(
                "group_reduce values and groups must share dimensions, except static entity values may broadcast over time"
            )
        if reducer not in _GROUP_REDUCERS:
            raise DSLValidationError(f"illegal group_reduce reducer: {reducer!r}")
        if reducer == "entropy" and values.dtype not in {"bool", "integer"}:
            raise DSLValidationError("group_reduce entropy requires categorical values")
        if reducer not in {"count", "fraction", "entropy"} and values.dtype not in {
            "numeric", "integer"
        }:
            raise DSLValidationError(f"group_reduce {reducer} requires numeric values")
        dimensions = list(groups.dimensions)
        dimensions.remove(axis)
        dimensions.append("group")
        return ValueType("numeric", tuple(dimensions))
    if op == "clip":
        value = validate_expression(expression.get("input"), fields)
        if value.dtype not in {"numeric", "integer"}:
            raise DSLValidationError("clip requires numeric input")
        if not all(isinstance(expression.get(key), (int, float)) for key in ("minimum", "maximum")):
            raise DSLValidationError("clip requires numeric minimum and maximum")
        if float(expression["minimum"]) > float(expression["maximum"]):
            raise DSLValidationError("clip minimum exceeds maximum")
        return value
    if op == "where":
        condition = validate_expression(expression.get("condition"), fields)
        value = validate_expression(expression.get("input"), fields)
        if condition.dtype != "bool":
            raise DSLValidationError("where condition must be boolean")
        _same_or_scalar(condition, value)
        return value
    if op in {"time_difference", "rolling_mean", "rolling_std"}:
        value = validate_expression(expression.get("input"), fields)
        if value.dtype not in {"numeric", "integer"} or "time" not in value.dimensions:
            raise DSLValidationError(f"{op} requires numeric time-indexed input")
        if op in {"rolling_mean", "rolling_std"} and (
            not isinstance(expression.get("window"), int) or expression["window"] < 1
        ):
            raise DSLValidationError(f"{op} window must be a positive integer")
        return value
    if op == "select":
        value = validate_expression(expression.get("input"), fields)
        axis = expression.get("axis")
        index = expression.get("index")
        if axis not in value.dimensions:
            raise DSLValidationError(f"select axis {axis!r} is not present")
        if not isinstance(index, int) or index < 0:
            raise DSLValidationError("select index must be a non-negative integer")
        maximum = {"channel": 8, "coordinate": 2, "endpoint": 2}.get(axis)
        if maximum is not None and index >= maximum:
            raise DSLValidationError(
                f"select index {index} exceeds fixed axis size {maximum}"
            )
        dimensions = list(value.dimensions)
        dimensions.remove(axis)
        return ValueType(value.dtype, tuple(dimensions))
    if op in {"connected_component_count", "largest_component_fraction", "spatial_neighbor_similarity"}:
        value = validate_expression(expression.get("input"), fields)
        if value.dimensions != ("time", "grid_y", "grid_x"):
            raise DSLValidationError(f"{op} requires a time-indexed grid")
        return ValueType("numeric", ("time",))
    if op == "network_assortativity":
        values = validate_expression(expression.get("values"), fields)
        edges = validate_expression(expression.get("edges"), fields)
        if values.dimensions != ("time", "agent") or edges.dimensions not in {
            ("edge", "endpoint"),
            ("time", "edge", "endpoint"),
        }:
            raise DSLValidationError("network_assortativity requires time-agent values and valid edge lists")
        return ValueType("numeric", ("time",))
    if op == "network_density":
        edges = validate_expression(expression.get("edges"), fields)
        if edges.dimensions not in {
            ("edge", "endpoint"),
            ("time", "edge", "endpoint"),
        }:
            raise DSLValidationError("network_density requires valid edge lists")
        if not isinstance(expression.get("node_count"), dict):
            raise DSLValidationError("network_density requires node_count expression")
        node_count = validate_expression(expression["node_count"], fields)
        if node_count.dimensions:
            raise DSLValidationError("network_density node_count must be scalar")
        return ValueType(
            "numeric", ("time",) if edges.dimensions[0] == "time" else ()
        )
    if op == "network_neighborhood_reduce":
        values = validate_expression(expression.get("values"), fields)
        edges = validate_expression(expression.get("edges"), fields)
        reducer = expression.get("reducer")
        if values.dimensions != ("time", "agent"):
            raise DSLValidationError(
                "network_neighborhood_reduce requires time-agent values"
            )
        if edges.dimensions not in {
            ("edge", "endpoint"),
            ("time", "edge", "endpoint"),
        }:
            raise DSLValidationError(
                "network_neighborhood_reduce requires valid edge lists"
            )
        if reducer not in _NETWORK_REDUCERS:
            raise DSLValidationError(
                f"illegal network neighborhood reducer: {reducer!r}"
            )
        if reducer not in {"count", "fraction"} and values.dtype not in {
            "numeric", "integer"
        }:
            raise DSLValidationError(
                f"network neighborhood {reducer} requires numeric values"
            )
        return ValueType("numeric", ("time", "agent"))
    if op in {"network_component_count", "network_largest_component_fraction"}:
        edges = validate_expression(expression.get("edges"), fields)
        node_count = validate_expression(expression.get("node_count"), fields)
        if edges.dimensions != ("time", "edge", "endpoint"):
            raise DSLValidationError(f"{op} requires dynamic time-indexed edges")
        if node_count.dimensions or node_count.dtype not in {"integer", "numeric"}:
            raise DSLValidationError(f"{op} node_count must be numeric scalar")
        return ValueType("numeric", ("time",))
    raise DSLValidationError(f"unhandled operator: {op}")


def validate_indicator_expression(expression: dict[str, Any], schema: Iterable[dict[str, Any]]) -> None:
    result = validate_expression(expression, field_types(schema))
    if result.dimensions != ("time",) or result.dtype not in {"numeric", "integer"}:
        raise DSLValidationError(
            f"indicator computation must return numeric [time], got {result}"
        )


def _axis_index(expression: dict[str, Any], raw_schema: dict[str, ValueType]) -> int:
    value_type = validate_expression(expression["input"], raw_schema)
    return value_type.dimensions.index(expression["axis"])


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result[index] = np.nanmean(
                values[max(0, index - window + 1) : index + 1], axis=0
            )
    return result


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    result = np.empty_like(values, dtype=float)
    for index in range(len(values)):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            result[index] = np.nanstd(
                values[max(0, index - window + 1) : index + 1], axis=0
            )
    return result


def _reduce_vector(values: np.ndarray, reducer: str) -> float:
    array = np.asarray(values)
    finite = np.isfinite(array)
    usable = array[finite]
    if reducer == "count":
        return float(len(usable))
    if not len(usable):
        return float("nan")
    if reducer == "mean":
        return float(np.mean(usable))
    if reducer == "sum":
        return float(np.sum(usable))
    if reducer == "fraction":
        return float(np.mean(usable.astype(bool)))
    if reducer == "variance":
        return float(np.var(usable))
    if reducer == "std":
        return float(np.std(usable))
    if reducer == "entropy":
        _, counts = np.unique(usable, return_counts=True)
        probabilities = counts / np.sum(counts)
        return float(-np.sum(probabilities * np.log(probabilities + 1e-15)))
    raise RuntimeError(f"unknown reducer reached runtime: {reducer}")


def _group_reduce_runtime(
    values: np.ndarray,
    groups: np.ndarray,
    values_type: ValueType,
    groups_type: ValueType,
    axis: str,
    reducer: str,
) -> np.ndarray:
    values = np.asarray(values)
    groups = np.asarray(groups)
    if values_type.dimensions == (axis,) and groups_type.dimensions == ("time", axis):
        values = np.broadcast_to(values[None, :], groups.shape)
        values_type = ValueType(values_type.dtype, groups_type.dimensions)
    value_axis = values_type.dimensions.index(axis)
    group_axis = groups_type.dimensions.index(axis)
    values = np.moveaxis(values, value_axis, -1)
    groups = np.moveaxis(groups, group_axis, -1)
    if values.shape != groups.shape:
        raise DSLValidationError(
            f"group_reduce runtime shape mismatch: {values.shape} versus {groups.shape}"
        )
    finite_groups = np.isfinite(groups) & (groups >= 0)
    group_count = (
        int(np.max(groups[finite_groups])) + 1 if np.any(finite_groups) else 1
    )
    output = np.full(values.shape[:-1] + (group_count,), np.nan, dtype=float)
    for index in np.ndindex(values.shape[:-1]):
        row_groups = groups[index]
        row_values = values[index]
        for group_id in range(group_count):
            mask = np.isfinite(row_groups) & (row_groups == group_id)
            output[index + (group_id,)] = _reduce_vector(
                row_values[mask], reducer
            )
    return output


def _validated_edges(edges: np.ndarray, node_count: int) -> np.ndarray:
    array = np.asarray(edges)
    if array.ndim != 2 or array.shape[1:] != (2,):
        raise DSLValidationError(
            f"network edge slice must have shape [edge,2], got {array.shape}"
        )
    if not np.isfinite(array).all():
        raise DSLValidationError("network edges contain non-finite endpoints")
    array = array.astype(np.int64)
    if np.any(array < 0) or np.any(array >= node_count):
        raise DSLValidationError("network edge endpoint is outside the node range")
    if np.any(array[:, 0] == array[:, 1]):
        raise DSLValidationError("network edges contain a self-loop")
    canonical = np.sort(array, axis=1)
    if len({tuple(item) for item in canonical}) != len(canonical):
        raise DSLValidationError("network edges contain a duplicate undirected edge")
    return canonical


def _edge_slice(edges: np.ndarray, time: int, node_count: int) -> np.ndarray:
    array = np.asarray(edges)
    current = array if array.ndim == 2 else array[time]
    return _validated_edges(current, node_count)


def _component_stats(grid: np.ndarray) -> tuple[float, float]:
    occupied = np.asarray(grid) >= 0
    height, width = occupied.shape
    visited = np.zeros_like(occupied, dtype=bool)
    components = 0
    largest = 0
    for row in range(height):
        for column in range(width):
            if not occupied[row, column] or visited[row, column]:
                continue
            group_value = grid[row, column]
            stack = [(row, column)]
            visited[row, column] = True
            size = 0
            while stack:
                x, y = stack.pop()
                size += 1
                for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nx_, ny_ = (x + dx) % height, (y + dy) % width
                    if not visited[nx_, ny_] and grid[nx_, ny_] == group_value:
                        visited[nx_, ny_] = True
                        stack.append((nx_, ny_))
            components += 1
            largest = max(largest, size)
    total = int(np.sum(occupied))
    return float(components), float(largest / total if total else 0.0)


def _spatial_similarity(grid: np.ndarray) -> float:
    occupied = grid >= 0
    same = 0
    total = 0
    for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
        neighbour = np.roll(np.roll(grid, dx, axis=0), dy, axis=1)
        valid = occupied & (neighbour >= 0)
        total += int(np.sum(valid))
        same += int(np.sum(valid & (neighbour == grid)))
    return float(same / total) if total else 1.0


def execute_expression(
    expression: dict[str, Any],
    raw: dict[str, np.ndarray],
    schema: Iterable[dict[str, Any]],
) -> np.ndarray:
    types = field_types(schema)
    validate_expression(expression, types)

    def run(node: dict[str, Any]) -> np.ndarray:
        op = node["op"]
        if op == "field":
            return np.asarray(raw[node["name"]])
        if op == "constant":
            return np.asarray(node["value"])
        if op in _UNARY:
            value = np.asarray(run(node["input"]), dtype=float)
            return {
                "abs": np.abs,
                "negate": np.negative,
                "sqrt": lambda x: np.sqrt(np.clip(x, 0.0, None)),
                "log1p": lambda x: np.log1p(np.clip(x, -0.999999, None)),
            }[op](value)
        if op in _BINARY or op in _COMPARISONS:
            left = run(node["left"])
            right = run(node["right"])
            if op == "add":
                return left + right
            if op == "subtract":
                return left - right
            if op == "multiply":
                return left * right
            if op in {"divide", "safe_ratio"}:
                return np.divide(left, right, out=np.zeros(np.broadcast_shapes(np.shape(left), np.shape(right))), where=np.asarray(right) != 0)
            if op == "distance":
                return np.abs(left - right)
            return {
                "greater": np.greater,
                "greater_equal": np.greater_equal,
                "less": np.less,
                "less_equal": np.less_equal,
                "equal": np.equal,
                "not_equal": np.not_equal,
            }[op](left, right)
        if op == "correlation":
            left = np.asarray(run(node["left"]), dtype=float)
            right = np.asarray(run(node["right"]), dtype=float)
            value_type = validate_expression(node["left"], types)
            axis = value_type.dimensions.index(node["axis"])
            left = np.moveaxis(left, axis, -1)
            right = np.moveaxis(right, axis, -1)
            left_centered = left - np.nanmean(left, axis=-1, keepdims=True)
            right_centered = right - np.nanmean(right, axis=-1, keepdims=True)
            numerator = np.nansum(left_centered * right_centered, axis=-1)
            denominator = np.sqrt(
                np.nansum(left_centered**2, axis=-1)
                * np.nansum(right_centered**2, axis=-1)
            )
            return np.divide(
                numerator,
                denominator,
                out=np.zeros_like(numerator, dtype=float),
                where=denominator > 0,
            )
        if op in _REDUCERS:
            values = run(node["input"])
            axis = _axis_index(node, types)
            if op == "mean":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    return np.nanmean(values, axis=axis)
            if op == "sum":
                return np.nansum(values, axis=axis)
            if op == "count":
                input_type = validate_expression(node["input"], types)
                return (
                    np.sum(np.asarray(values, dtype=bool), axis=axis)
                    if input_type.dtype == "bool"
                    else np.sum(np.isfinite(values), axis=axis)
                )
            if op == "fraction":
                return np.nanmean(np.asarray(values, dtype=float), axis=axis)
            if op == "variance":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    return np.nanvar(values, axis=axis)
            if op == "std":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    return np.nanstd(values, axis=axis)
            if op == "quantile":
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", category=RuntimeWarning)
                    return np.nanquantile(values, float(node["q"]), axis=axis)
            if op == "entropy":
                array = np.asarray(values)
                moved = np.moveaxis(array, axis, -1)
                output = np.empty(moved.shape[:-1], dtype=float)
                for index in np.ndindex(output.shape):
                    _, counts = np.unique(moved[index], return_counts=True)
                    probabilities = counts / max(np.sum(counts), 1)
                    output[index] = -np.sum(probabilities * np.log(probabilities + 1e-15))
                return output
            if op == "binned_entropy":
                array = np.asarray(values, dtype=float)
                moved = np.moveaxis(array, axis, -1)
                output = np.empty(moved.shape[:-1], dtype=float)
                bins = int(node["bins"])
                for index in np.ndindex(output.shape):
                    row = moved[index]
                    finite = row[np.isfinite(row)]
                    if not finite.size or np.nanmin(finite) == np.nanmax(finite):
                        output[index] = 0.0
                        continue
                    counts, _ = np.histogram(finite, bins=bins)
                    probabilities = counts[counts > 0] / np.sum(counts)
                    output[index] = -np.sum(probabilities * np.log(probabilities))
                return output
        if op == "group_reduce":
            values_type = validate_expression(node["values"], types)
            groups_type = validate_expression(node["groups"], types)
            return _group_reduce_runtime(
                run(node["values"]),
                run(node["groups"]),
                values_type,
                groups_type,
                str(node["axis"]),
                str(node["reducer"]),
            )
        if op == "clip":
            return np.clip(run(node["input"]), float(node["minimum"]), float(node["maximum"]))
        if op == "where":
            return np.where(run(node["condition"]), run(node["input"]), np.nan)
        if op == "time_difference":
            values = np.asarray(run(node["input"]), dtype=float)
            return np.diff(values, axis=0, prepend=values[[0]])
        if op == "rolling_mean":
            return _rolling_mean(run(node["input"]), int(node["window"]))
        if op == "rolling_std":
            return _rolling_std(run(node["input"]), int(node["window"]))
        if op == "select":
            value = run(node["input"])
            value_type = validate_expression(node["input"], types)
            axis = value_type.dimensions.index(node["axis"])
            return np.take(value, int(node["index"]), axis=axis)
        if op in {"connected_component_count", "largest_component_fraction"}:
            stats = [_component_stats(grid) for grid in run(node["input"])]
            column = 0 if op == "connected_component_count" else 1
            return np.asarray([item[column] for item in stats], dtype=float)
        if op == "spatial_neighbor_similarity":
            return np.asarray([_spatial_similarity(grid) for grid in run(node["input"])])
        if op == "network_assortativity":
            values = np.asarray(run(node["values"]), dtype=float)
            edges = np.asarray(run(node["edges"]))
            result = []
            for time, row in enumerate(values):
                current_edges = _edge_slice(edges, time, len(row))
                # The simulator graph is undirected.  Include both orientations
                # so the statistic cannot depend on the stored min/max node-id
                # ordering of an edge.
                forward_left = row[current_edges[:, 0]]
                forward_right = row[current_edges[:, 1]]
                left = np.concatenate([forward_left, forward_right])
                right = np.concatenate([forward_right, forward_left])
                finite = np.isfinite(left) & np.isfinite(right)
                left, right = left[finite], right[finite]
                if (
                    not len(left)
                    or np.std(left) <= 1e-12
                    or np.std(right) <= 1e-12
                ):
                    result.append(0.0)
                else:
                    correlation = float(np.corrcoef(left, right)[0, 1])
                    result.append(correlation if np.isfinite(correlation) else 0.0)
            return np.asarray(result)
        if op == "network_density":
            edges = np.asarray(run(node["edges"]))
            n = int(np.ravel(run(node["node_count"]))[0])
            if n <= 0:
                raise DSLValidationError("network node_count must be positive")
            if edges.ndim == 2:
                current = _validated_edges(edges, n)
                return np.asarray(2 * len(current) / max(n * (n - 1), 1))
            return np.asarray(
                [
                    2 * len(_edge_slice(edges, time, n)) / max(n * (n - 1), 1)
                    for time in range(len(edges))
                ],
                dtype=float,
            )
        if op == "network_neighborhood_reduce":
            values = np.asarray(run(node["values"]))
            edges = np.asarray(run(node["edges"]))
            reducer = str(node["reducer"])
            result = np.full(values.shape, np.nan, dtype=float)
            for time, row in enumerate(values):
                current_edges = _edge_slice(edges, time, len(row))
                neighbours: list[list[int]] = [[] for _ in range(len(row))]
                for left, right in current_edges:
                    neighbours[int(left)].append(int(right))
                    neighbours[int(right)].append(int(left))
                for agent, connected in enumerate(neighbours):
                    result[time, agent] = _reduce_vector(
                        row[np.asarray(connected, dtype=int)], reducer
                    )
            return result
        if op in {"network_component_count", "network_largest_component_fraction"}:
            edges = np.asarray(run(node["edges"]))
            n = int(np.ravel(run(node["node_count"]))[0])
            if n <= 0:
                raise DSLValidationError("network node_count must be positive")
            result = []
            for time in range(len(edges)):
                current_edges = _edge_slice(edges, time, n)
                graph = nx.Graph()
                graph.add_nodes_from(range(n))
                graph.add_edges_from((int(left), int(right)) for left, right in current_edges)
                components = list(nx.connected_components(graph))
                if op == "network_component_count":
                    result.append(float(len(components)))
                else:
                    largest = max((len(component) for component in components), default=0)
                    result.append(float(largest / n))
            return np.asarray(result, dtype=float)
        raise RuntimeError(f"unsupported operator reached executor: {op}")

    return np.asarray(run(expression), dtype=float)


def validate_temporal_aggregation(aggregation: dict[str, Any]) -> None:
    if not isinstance(aggregation, dict):
        raise DSLValidationError("temporal aggregation must be an object")
    allowed = {
        "identity", "rolling_mean", "rolling_std", "difference", "cumulative_mean"
    }
    op = aggregation.get("op")
    if op not in allowed:
        raise DSLValidationError(f"illegal temporal aggregation: {op!r}")
    extra = set(aggregation) - (
        {"op", "window"} if op in {"rolling_mean", "rolling_std"} else {"op"}
    )
    if extra:
        raise DSLValidationError(f"unexpected temporal aggregation fields: {sorted(extra)}")
    if op in {"rolling_mean", "rolling_std"}:
        window = aggregation.get("window")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise DSLValidationError(
                f"temporal {op} requires a positive integer window"
            )


def apply_temporal_aggregation(values: np.ndarray, aggregation: dict[str, Any]) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    validate_temporal_aggregation(aggregation)
    op = aggregation["op"]
    if op == "identity":
        return values
    if op == "rolling_mean":
        window = aggregation.get("window")
        if not isinstance(window, int) or window < 1:
            raise DSLValidationError("temporal rolling_mean requires a positive window")
        return _rolling_mean(values, window)
    if op == "rolling_std":
        window = aggregation.get("window")
        if not isinstance(window, int) or window < 1:
            raise DSLValidationError("temporal rolling_std requires a positive window")
        return _rolling_std(values, window)
    if op == "difference":
        return np.diff(values, prepend=values[0])
    if op == "cumulative_mean":
        return np.cumsum(values) / np.arange(1, len(values) + 1)
    raise DSLValidationError(f"illegal temporal aggregation: {op!r}")


def compute_indicator(
    expression: dict[str, Any],
    aggregation: dict[str, Any],
    raw: dict[str, np.ndarray],
    schema: Iterable[dict[str, Any]],
    *,
    allow_all_nan: bool = False,
) -> np.ndarray:
    validate_indicator_expression(expression, schema)
    values = execute_expression(expression, raw, schema)
    values = apply_temporal_aggregation(values, aggregation)
    if values.ndim != 1:
        raise DSLValidationError(f"indicator execution returned shape {values.shape}, expected one dimension")
    if not allow_all_nan and not np.isfinite(values).any():
        raise DSLValidationError("indicator execution returned no finite values")
    return values
