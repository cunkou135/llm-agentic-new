"""Safe declarative indicator computation with no dynamic code execution."""

from __future__ import annotations

import json
import math
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
    "select",
    "connected_component_count",
    "largest_component_fraction",
    "spatial_neighbor_similarity",
    "network_assortativity",
    "network_density",
} | _REDUCERS | _BINARY | _UNARY | _COMPARISONS
_OPERATORS.add("correlation")


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
        elif op in _UNARY | {"clip", "time_difference", "rolling_mean"}:
            required += ["input"]
            types["input"] = "AST object"
            example["input"] = {"op": "field", "name": "local_similarity"}
            if op == "clip":
                required += ["minimum", "maximum"]
                types.update({"minimum": "number", "maximum": "number"})
                example.update({"minimum": 0.0, "maximum": 1.0})
            if op == "rolling_mean":
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
            types.update({"values": "numeric [time,agent] AST", "edges": "integer [edge,endpoint] AST"})
            example.update({"values": {"op": "field", "name": "state_opinion"}, "edges": {"op": "field", "name": "network_edges"}})
            output = "numeric [time]"
        elif op == "network_density":
            required += ["edges", "node_count"]
            types.update({"edges": "integer [edge,endpoint] AST", "node_count": "scalar AST"})
            example.update({"edges": {"op": "field", "name": "network_edges"}, "node_count": {"op": "field", "name": "agent_count"}})
            output = "numeric scalar"
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
    if op in {"time_difference", "rolling_mean"}:
        value = validate_expression(expression.get("input"), fields)
        if value.dtype not in {"numeric", "integer"} or "time" not in value.dimensions:
            raise DSLValidationError(f"{op} requires numeric time-indexed input")
        if op == "rolling_mean" and (
            not isinstance(expression.get("window"), int) or expression["window"] < 1
        ):
            raise DSLValidationError("rolling_mean window must be a positive integer")
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
        if values.dimensions != ("time", "agent") or edges.dimensions != ("edge", "endpoint"):
            raise DSLValidationError("network_assortativity requires time-agent values and static edges")
        return ValueType("numeric", ("time",))
    if op == "network_density":
        edges = validate_expression(expression.get("edges"), fields)
        if edges.dimensions != ("edge", "endpoint"):
            raise DSLValidationError("network_density requires static edges")
        if not isinstance(expression.get("node_count"), dict):
            raise DSLValidationError("network_density requires node_count expression")
        validate_expression(expression["node_count"], fields)
        return ValueType("numeric", ())
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
        result[index] = np.nanmean(values[max(0, index - window + 1) : index + 1], axis=0)
    return result


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
                return np.nanvar(values, axis=axis)
            if op == "std":
                return np.nanstd(values, axis=axis)
            if op == "quantile":
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
        if op == "clip":
            return np.clip(run(node["input"]), float(node["minimum"]), float(node["maximum"]))
        if op == "where":
            return np.where(run(node["condition"]), run(node["input"]), np.nan)
        if op == "time_difference":
            values = np.asarray(run(node["input"]), dtype=float)
            return np.diff(values, axis=0, prepend=values[[0]])
        if op == "rolling_mean":
            return _rolling_mean(run(node["input"]), int(node["window"]))
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
            edges = np.asarray(run(node["edges"]), dtype=int)
            result = []
            for row in values:
                left, right = row[edges[:, 0]], row[edges[:, 1]]
                if np.std(left) == 0 or np.std(right) == 0:
                    result.append(0.0)
                else:
                    result.append(float(np.corrcoef(left, right)[0, 1]))
            return np.asarray(result)
        if op == "network_density":
            edges = np.asarray(run(node["edges"]))
            n = int(np.ravel(run(node["node_count"]))[0])
            return np.asarray(2 * len(edges) / max(n * (n - 1), 1))
        raise RuntimeError(f"unsupported operator reached executor: {op}")

    return np.asarray(run(expression), dtype=float)


def validate_temporal_aggregation(aggregation: dict[str, Any]) -> None:
    if not isinstance(aggregation, dict):
        raise DSLValidationError("temporal aggregation must be an object")
    allowed = {"identity", "rolling_mean", "difference", "cumulative_mean"}
    op = aggregation.get("op")
    if op not in allowed:
        raise DSLValidationError(f"illegal temporal aggregation: {op!r}")
    extra = set(aggregation) - ({"op", "window"} if op == "rolling_mean" else {"op"})
    if extra:
        raise DSLValidationError(f"unexpected temporal aggregation fields: {sorted(extra)}")
    if op == "rolling_mean":
        window = aggregation.get("window")
        if isinstance(window, bool) or not isinstance(window, int) or window < 1:
            raise DSLValidationError("temporal rolling_mean requires a positive integer window")


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
) -> np.ndarray:
    validate_indicator_expression(expression, schema)
    values = execute_expression(expression, raw, schema)
    values = apply_temporal_aggregation(values, aggregation)
    if values.ndim != 1:
        raise DSLValidationError(f"indicator execution returned shape {values.shape}, expected one dimension")
    if not np.isfinite(values).any():
        raise DSLValidationError("indicator execution returned no finite values")
    return values
