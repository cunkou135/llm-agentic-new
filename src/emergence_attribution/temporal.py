"""Lagged OLS with pre-frozen Macro-outcome FDR hypothesis groups."""

from __future__ import annotations

import hashlib
import json
import math
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import pandas as pd
from scipy.stats import t as student_t

from .simulation import trajectories


@dataclass(frozen=True)
class TemporalEdge:
    source: str
    target: str
    lag: int
    beta: float
    p_value: float
    q_value: float
    effect_direction: str
    support: float
    lag_support: float
    lag_std: float
    hypothesis_group_id: str = "ungrouped"


@dataclass(frozen=True)
class TargetBlock:
    target: str
    sources: tuple[str, ...]
    hypothesis_groups: dict[tuple[str, str], tuple[str, ...]]
    y_blocks: tuple[np.ndarray, ...]
    x_blocks: tuple[np.ndarray, ...]
    terms: tuple[tuple[str, int, bool], ...]


def stable_seed(master_seed: int, *parts: Any) -> int:
    text = ":".join([str(master_seed), *(str(part) for part in parts)])
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32 - 1)


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if values.size == 0:
        return values
    order = np.argsort(values)
    ranked = values[order]
    adjusted = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    result = np.empty_like(adjusted)
    result[order] = adjusted
    return result


def representation_candidates(representation: dict[str, Any]) -> list[dict[str, str]]:
    candidates: list[dict[str, str]] = []
    for edge in representation["candidate_edges"]:
        groups = edge.get("hypothesis_group_ids")
        if groups is None:
            groups = [edge.get("hypothesis_group_id", "ungrouped")]
        for group in groups:
            candidates.append(
                {
                    "source": edge["source"], "target": edge["target"],
                    "hypothesis_group_id": str(group),
                    "expected_direction": edge["expected_direction"],
                }
            )
    return candidates


def unrestricted_candidates(representation: dict[str, Any]) -> list[dict[str, str]]:
    identifiers = sorted(item["id"] for item in representation["indicators"])
    return [
        {
            "source": source,
            "target": target,
            "hypothesis_group_id": "unrestricted",
            "expected_direction": "unknown",
        }
        for source in identifiers
        for target in identifiers
        if source != target
    ]


def prepare_target_blocks(
    trajectory_frames: Sequence[pd.DataFrame],
    candidates: Sequence[dict[str, str]],
    maximum_lag: int,
) -> tuple[TargetBlock, ...]:
    target_sources: dict[str, set[str]] = {}
    groups: dict[tuple[str, str], set[str]] = {}
    for edge in candidates:
        target_sources.setdefault(edge["target"], set()).add(edge["source"])
        groups.setdefault((edge["source"], edge["target"]), set()).add(
            edge.get("hypothesis_group_id", "ungrouped")
        )
    blocks: list[TargetBlock] = []
    for target, source_set in sorted(target_sources.items()):
        sources = tuple(sorted(source_set))
        terms: list[tuple[str, int, bool]] = [
            (target, lag, True) for lag in range(1, maximum_lag + 1)
        ]
        terms.extend(
            (source, lag, False)
            for source in sources
            for lag in range(1, maximum_lag + 1)
        )
        y_blocks: list[np.ndarray] = []
        x_blocks: list[np.ndarray] = []
        for frame in trajectory_frames:
            required = {target, *sources}
            missing = sorted(required - set(frame.columns))
            if missing:
                raise KeyError(f"trajectory is missing indicators: {missing}")
            if len(frame) <= maximum_lag:
                continue
            y = frame[target].to_numpy(dtype=float)[maximum_lag:]
            columns = [
                frame[name].to_numpy(dtype=float)[maximum_lag - lag : len(frame) - lag]
                for name, lag, _ in terms
            ]
            x = np.column_stack(columns)
            valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
            if np.any(valid):
                y_blocks.append(y[valid])
                x_blocks.append(x[valid])
        blocks.append(
            TargetBlock(
                target=target,
                sources=sources,
                hypothesis_groups={
                    key: tuple(sorted(values)) for key, values in groups.items()
                },
                y_blocks=tuple(y_blocks),
                x_blocks=tuple(x_blocks),
                terms=tuple(terms),
            )
        )
    return tuple(blocks)


def _standardised_ols(y: np.ndarray, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    x = np.asarray(x, dtype=float)
    valid = np.isfinite(y) & np.all(np.isfinite(x), axis=1)
    y, x = y[valid], x[valid]
    coefficients = np.zeros(x.shape[1], dtype=float)
    p_values = np.ones(x.shape[1], dtype=float)
    if len(y) <= x.shape[1] + 2:
        return coefficients, p_values
    x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0, ddof=1)
    y_mean, y_std = np.mean(y), np.std(y, ddof=1)
    usable = np.isfinite(x_std) & (x_std > 1e-12)
    if not np.isfinite(y_std) or y_std <= 1e-12 or not np.any(usable):
        return coefficients, p_values
    design = (x[:, usable] - x_mean[usable]) / x_std[usable]
    response = (y - y_mean) / y_std
    beta, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    residual = response - design @ beta
    dof = max(len(response) - int(rank), 1)
    sigma2 = float(residual @ residual / dof)
    covariance = sigma2 * np.linalg.pinv(design.T @ design)
    standard_error = np.sqrt(np.clip(np.diag(covariance), 0.0, None))
    statistic = np.divide(
        beta,
        standard_error,
        out=np.zeros_like(beta),
        where=standard_error > 0,
    )
    p = 2.0 * student_t.sf(np.abs(statistic), dof)
    coefficients[usable] = beta
    p_values[usable] = np.clip(p, 0.0, 1.0)
    return coefficients, p_values


def _sampled_arrays(
    block: TargetBlock, sample_indices: Sequence[int] | None
) -> tuple[np.ndarray, np.ndarray]:
    if not block.y_blocks:
        return np.empty(0), np.empty((0, len(block.terms)))
    indices = range(len(block.y_blocks)) if sample_indices is None else sample_indices
    return (
        np.concatenate([block.y_blocks[int(index)] for index in indices]),
        np.row_stack([block.x_blocks[int(index)] for index in indices]),
    )


def discover_point_graph_from_blocks(
    blocks: Sequence[TargetBlock],
    parent_alpha: float,
    fdr_alpha: float,
    sample_indices: Sequence[int] | None = None,
) -> list[TemporalEdge]:
    provisional_by_group: dict[
        str, list[tuple[str, str, int, float, float]]
    ] = {}
    for block in blocks:
        y, x = _sampled_arrays(block, sample_indices)
        if not len(y):
            continue
        _, screening_p = _standardised_ols(y, x)
        self_indices = [index for index, term in enumerate(block.terms) if term[2]]
        candidate_indices = [
            index
            for index, term in enumerate(block.terms)
            if not term[2] and screening_p[index] < parent_alpha
        ]
        if not candidate_indices:
            continue
        refit_indices = self_indices + candidate_indices
        coefficients, p_values = _standardised_ols(y, x[:, refit_indices])
        offset = len(self_indices)
        for local_index, term_index in enumerate(candidate_indices):
            source, lag, _ = block.terms[term_index]
            for group in block.hypothesis_groups[(source, block.target)]:
                provisional_by_group.setdefault(group, []).append(
                    (
                        source,
                        block.target,
                        lag,
                        float(coefficients[offset + local_index]),
                        float(p_values[offset + local_index]),
                    )
                )
    result: list[TemporalEdge] = []
    for group, provisional in sorted(provisional_by_group.items()):
        q_values = benjamini_hochberg([item[4] for item in provisional])
        by_pair: dict[tuple[str, str], list[tuple[int, float, float, float]]] = {}
        for item, q_value in zip(provisional, q_values):
            source, target, lag, beta, p_value = item
            by_pair.setdefault((source, target), []).append(
                (lag, beta, p_value, float(q_value))
            )
        for (source, target), choices in sorted(by_pair.items()):
            lag, beta, p_value, q_value = min(choices, key=lambda item: (item[3], item[0]))
            if q_value < fdr_alpha:
                result.append(
                    TemporalEdge(
                        source=source,
                        target=target,
                        lag=int(lag),
                        beta=float(beta),
                        p_value=float(p_value),
                        q_value=float(q_value),
                        effect_direction="increase" if beta > 0 else "decrease",
                        support=float("nan"),
                        lag_support=float("nan"),
                        lag_std=float("nan"),
                        hypothesis_group_id=group,
                    )
                )
    return result


_WORKER_BLOCKS: tuple[TargetBlock, ...] = ()
_WORKER_PARENT_ALPHA = 0.10
_WORKER_FDR_ALPHA = 0.05


def _initialise_bootstrap_worker(
    blocks: tuple[TargetBlock, ...], parent_alpha: float, fdr_alpha: float
) -> None:
    global _WORKER_BLOCKS, _WORKER_PARENT_ALPHA, _WORKER_FDR_ALPHA
    _WORKER_BLOCKS = blocks
    _WORKER_PARENT_ALPHA = parent_alpha
    _WORKER_FDR_ALPHA = fdr_alpha


def _bootstrap_worker(payload: tuple[int, np.ndarray]) -> tuple[int, list[TemporalEdge]]:
    replicate, indices = payload
    graph = discover_point_graph_from_blocks(
        _WORKER_BLOCKS, _WORKER_PARENT_ALPHA, _WORKER_FDR_ALPHA, indices
    )
    return replicate, graph


def discover_bootstrap_graph(
    trajectory_frames: Sequence[pd.DataFrame],
    candidates: Sequence[dict[str, str]],
    maximum_lag: int,
    parent_alpha: float,
    fdr_alpha: float,
    bootstrap_repetitions: int,
    support_threshold: float,
    master_seed: int,
    seed_label: str,
    workers: int,
    progress_callback: Callable[[int, int], None] | None = None,
) -> tuple[list[TemporalEdge], dict[str, Any]]:
    blocks = prepare_target_blocks(trajectory_frames, candidates, maximum_lag)
    point = discover_point_graph_from_blocks(blocks, parent_alpha, fdr_alpha)
    trajectory_count = len(trajectory_frames)
    payloads = []
    for replicate in range(bootstrap_repetitions):
        rng = np.random.default_rng(stable_seed(master_seed, seed_label, replicate))
        payloads.append(
            (replicate, rng.integers(0, trajectory_count, size=trajectory_count))
        )
    bootstrap_graphs: list[tuple[int, list[TemporalEdge]]] = []
    if workers <= 1:
        _initialise_bootstrap_worker(blocks, parent_alpha, fdr_alpha)
        for payload in payloads:
            bootstrap_graphs.append(_bootstrap_worker(payload))
            if progress_callback:
                progress_callback(len(bootstrap_graphs), bootstrap_repetitions)
    else:
        with ProcessPoolExecutor(
            max_workers=workers,
            initializer=_initialise_bootstrap_worker,
            initargs=(blocks, parent_alpha, fdr_alpha),
        ) as pool:
            futures = [pool.submit(_bootstrap_worker, payload) for payload in payloads]
            for future in as_completed(futures):
                bootstrap_graphs.append(future.result())
                if progress_callback:
                    progress_callback(len(bootstrap_graphs), bootstrap_repetitions)
    bootstrap_graphs.sort(key=lambda item: item[0])
    pair_count: dict[tuple[str, str, str], int] = {}
    lag_count: dict[tuple[str, str, str, int], int] = {}
    lag_samples: dict[tuple[str, str, str], list[int]] = {}
    edge_sets: list[dict[str, Any]] = []
    for replicate, graph in bootstrap_graphs:
        pairs = {
            (edge.source, edge.target, edge.hypothesis_group_id) for edge in graph
        }
        for pair in pairs:
            pair_count[pair] = pair_count.get(pair, 0) + 1
        for edge in graph:
            key = (
                edge.source, edge.target, edge.hypothesis_group_id, edge.lag
            )
            lag_count[key] = lag_count.get(key, 0) + 1
            lag_samples.setdefault(
                (edge.source, edge.target, edge.hypothesis_group_id), []
            ).append(edge.lag)
        edge_sets.append(
            {
                "replicate": replicate,
                "edges": [
                    {
                        **asdict(edge),
                        "support": None,
                        "lag_support": None,
                        "lag_std": None,
                    }
                    for edge in graph
                ],
            }
        )
    retained: list[TemporalEdge] = []
    for edge in point:
        pair = (edge.source, edge.target, edge.hypothesis_group_id)
        support = pair_count.get(pair, 0) / bootstrap_repetitions
        if support < support_threshold:
            continue
        lags = lag_samples.get(pair, [])
        retained.append(
            replace(
                edge,
                support=float(support),
                lag_support=lag_count.get(
                    (
                        edge.source, edge.target, edge.hypothesis_group_id,
                        edge.lag,
                    ), 0
                )
                / bootstrap_repetitions,
                lag_std=float(np.std(lags, ddof=1)) if len(lags) > 1 else 0.0,
            )
        )
    return retained, {
        "bootstrap_repetitions": bootstrap_repetitions,
        "trajectory_count": trajectory_count,
        "support_threshold": support_threshold,
        "edge_sets": edge_sets,
    }


def discover_vote_graph(
    trajectory_frames: Sequence[pd.DataFrame],
    candidates: Sequence[dict[str, str]],
    maximum_lag: int,
    parent_alpha: float,
    fdr_alpha: float,
    vote_threshold: float,
) -> list[TemporalEdge]:
    individual = []
    for frame in trajectory_frames:
        blocks = prepare_target_blocks([frame], candidates, maximum_lag)
        individual.append(discover_point_graph_from_blocks(blocks, parent_alpha, fdr_alpha))
    grouped: dict[tuple[str, str, str], list[TemporalEdge]] = {}
    for graph in individual:
        for edge in graph:
            grouped.setdefault(
                (edge.source, edge.target, edge.hypothesis_group_id), []
            ).append(edge)
    result: list[TemporalEdge] = []
    for pair, edges in sorted(grouped.items()):
        support = len(edges) / len(individual)
        if support < vote_threshold:
            continue
        lag_counts: dict[int, int] = {}
        for edge in edges:
            lag_counts[edge.lag] = lag_counts.get(edge.lag, 0) + 1
        selected_lag = min(lag_counts, key=lambda lag: (-lag_counts[lag], lag))
        matching = [edge for edge in edges if edge.lag == selected_lag]
        template = matching[0]
        result.append(
            replace(
                template,
                beta=float(np.mean([edge.beta for edge in matching])),
                p_value=float(np.mean([edge.p_value for edge in matching])),
                q_value=float(np.mean([edge.q_value for edge in matching])),
                support=support,
                lag_support=lag_counts[selected_lag] / len(individual),
                lag_std=float(np.std([edge.lag for edge in edges], ddof=1))
                if len(edges) > 1
                else 0.0,
            )
        )
    return result


def semantic_graph(representation: dict[str, Any]) -> list[TemporalEdge]:
    direction_value = {"increase": 1.0, "decrease": -1.0, "mixed": 0.0, "unknown": 0.0}
    return [
        TemporalEdge(
            source=edge["source"],
            target=edge["target"],
            lag=0,
            beta=direction_value[edge["expected_direction"]],
            p_value=float("nan"),
            q_value=float("nan"),
            effect_direction=edge["expected_direction"],
            support=float("nan"),
            lag_support=float("nan"),
            lag_std=float("nan"),
            hypothesis_group_id=group,
        )
        for edge in representation["candidate_edges"]
        for group in edge["hypothesis_group_ids"]
    ]


def write_graph_records(path: Path, records: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False, allow_nan=True) + "\n")


def run_temporal_stage(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    baseline_dataset: pd.DataFrame,
    workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    temporal_config = config["temporal"]
    records: list[dict[str, Any]] = []
    bootstrap_summaries: dict[str, Any] = {}
    runtime_rows: list[dict[str, Any]] = []
    for scenario, representation in sorted(representations.items()):
        semantic_seconds = 0.0
        selection_generation_count = int(
            config.get("semantic_replication", {}).get(
                "selection_generations",
                config["representation"]["independent_generations"],
            )
        )
        for path in sorted((run_root / "llm" / scenario).glob("generation_*/generation_result.json")):
            generation_index = int(path.parent.name.rsplit("_", 1)[-1])
            if generation_index >= selection_generation_count:
                continue
            semantic_seconds += float(
                json.loads(path.read_text(encoding="utf-8"))["duration_seconds"]
            )
        runtime_rows.append({
            "scenario": scenario,
            "method": "llm_semantic_proposal",
            "runtime_seconds": semantic_seconds,
            "runtime_scope": "semantic_generation",
        })
        by_seed = trajectories(baseline_dataset, scenario)
        frames = [by_seed[seed] for seed in sorted(by_seed)]
        candidates = representation_candidates(representation)

        def bootstrap_progress(completed: int, total: int) -> None:
            if progress_callback:
                progress_callback("Trajectory bootstrap", completed, total, scenario)

        started = time.perf_counter()
        full_graph, bootstrap = discover_bootstrap_graph(
            frames,
            candidates,
            int(temporal_config["maximum_lag"]),
            float(temporal_config["parent_alpha"]),
            float(temporal_config["fdr_alpha"]),
            int(temporal_config["bootstrap_repetitions"]),
            float(temporal_config["support_threshold"]),
            int(config["master_seed"]),
            f"{scenario}:full",
            workers,
            bootstrap_progress,
        )
        runtime_rows.append({"scenario": scenario, "method": "full_method", "runtime_seconds": time.perf_counter() - started, "runtime_scope": "temporal_analysis"})
        point_blocks = prepare_target_blocks(
            frames, candidates, int(temporal_config["maximum_lag"])
        )
        point_graph = discover_point_graph_from_blocks(
            point_blocks,
            float(temporal_config["parent_alpha"]),
            float(temporal_config["fdr_alpha"]),
        )
        started = time.perf_counter()
        single_blocks = prepare_target_blocks(
            frames[:1], candidates, int(temporal_config["maximum_lag"])
        )
        single_graph = discover_point_graph_from_blocks(
            single_blocks,
            float(temporal_config["parent_alpha"]),
            float(temporal_config["fdr_alpha"]),
        )
        runtime_rows.append({"scenario": scenario, "method": "single_trajectory", "runtime_seconds": time.perf_counter() - started, "runtime_scope": "temporal_analysis"})
        started = time.perf_counter()
        vote_graph = discover_vote_graph(
            frames,
            candidates,
            int(temporal_config["maximum_lag"]),
            float(temporal_config["parent_alpha"]),
            float(temporal_config["fdr_alpha"]),
            float(temporal_config["vote_threshold"]),
        )
        runtime_rows.append({"scenario": scenario, "method": "trajectory_vote", "runtime_seconds": time.perf_counter() - started, "runtime_scope": "temporal_analysis"})
        unrestricted = unrestricted_candidates(representation)
        started = time.perf_counter()
        unrestricted_graph, unrestricted_bootstrap = discover_bootstrap_graph(
            frames,
            unrestricted,
            int(temporal_config["maximum_lag"]),
            float(temporal_config["parent_alpha"]),
            float(temporal_config["fdr_alpha"]),
            int(temporal_config["bootstrap_repetitions"]),
            float(temporal_config["support_threshold"]),
            int(config["master_seed"]),
            f"{scenario}:unrestricted",
            workers,
        )
        runtime_rows.append({"scenario": scenario, "method": "unrestricted_temporal_search", "runtime_seconds": time.perf_counter() - started, "runtime_scope": "temporal_analysis"})
        methods = {
            "llm_semantic_proposal": semantic_graph(representation),
            "unrestricted_temporal_search": unrestricted_graph,
            "single_trajectory": single_graph,
            "trajectory_vote": vote_graph,
            "full_method": full_graph,
        }
        records.extend(
            {
                "scenario": scenario,
                "method": method,
                "edges": [asdict(edge) for edge in graph],
            }
            for method, graph in methods.items()
        )
        bootstrap_summaries[scenario] = bootstrap
        bootstrap_summaries[scenario]["point_graph"] = [asdict(edge) for edge in point_graph]
        bootstrap_summaries[scenario]["unrestricted"] = unrestricted_bootstrap
    analysis_root = run_root / "analysis"
    analysis_root.mkdir(parents=True, exist_ok=True)
    write_graph_records(analysis_root / "main_graphs.jsonl", records)
    (analysis_root / "bootstrap_summary.json").write_text(
        json.dumps(bootstrap_summaries, indent=2, ensure_ascii=False, allow_nan=True),
        encoding="utf-8",
    )
    pd.DataFrame(runtime_rows).to_csv(
        analysis_root / "method_runtime.csv", index=False
    )
    return {"graphs": records, "bootstrap": bootstrap_summaries, "runtime": runtime_rows}


def load_graph_records(path: Path) -> dict[tuple[str, str], list[TemporalEdge]]:
    result: dict[tuple[str, str], list[TemporalEdge]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        result[(record["scenario"], record["method"])] = [
            TemporalEdge(**edge) for edge in record["edges"]
        ]
    return result
