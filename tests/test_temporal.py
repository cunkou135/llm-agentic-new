from __future__ import annotations

import numpy as np
import pandas as pd

from emergence_attribution.temporal import (
    benjamini_hochberg,
    discover_bootstrap_graph,
    discover_point_graph_from_blocks,
    prepare_target_blocks,
)


def _synthetic_frames(count: int = 8, steps: int = 260) -> list[pd.DataFrame]:
    frames = []
    for seed in range(count):
        rng = np.random.default_rng(seed)
        source = rng.normal(size=steps)
        target = np.zeros(steps)
        for time in range(5, steps):
            target[time] = 0.45 * target[time - 1] + 0.85 * source[time - 2] + rng.normal(0, 0.25)
        frames.append(pd.DataFrame({"source": source, "target": target}))
    return frames


CANDIDATES = [
    {
        "source": "source",
        "target": "target",
        "hypothesis_group_id": "macro_outcome_target",
        "expected_direction": "increase",
    }
]


def test_bh_fdr_known_values() -> None:
    adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
    np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008])


def test_temporal_ols_recovers_lag_two() -> None:
    blocks = prepare_target_blocks(_synthetic_frames(), CANDIDATES, 5)
    graph = discover_point_graph_from_blocks(blocks, 0.10, 0.05)
    assert len(graph) == 1
    assert graph[0].lag == 2
    assert graph[0].beta > 0
    assert graph[0].q_value < 0.05


def test_trajectory_bootstrap_reproducible() -> None:
    frames = _synthetic_frames(count=6, steps=180)
    first, first_summary = discover_bootstrap_graph(
        frames, CANDIDATES, 5, 0.10, 0.05, 10, 0.60, 123, "repeat", 1
    )
    second, second_summary = discover_bootstrap_graph(
        frames, CANDIDATES, 5, 0.10, 0.05, 10, 0.60, 123, "repeat", 1
    )
    assert first == second
    assert first_summary["edge_sets"] == second_summary["edge_sets"]


def test_workers_1_equals_workers_2() -> None:
    frames = _synthetic_frames(count=6, steps=180)
    single, summary_single = discover_bootstrap_graph(
        frames, CANDIDATES, 5, 0.10, 0.05, 8, 0.50, 321, "parallel", 1
    )
    parallel, summary_parallel = discover_bootstrap_graph(
        frames, CANDIDATES, 5, 0.10, 0.05, 8, 0.50, 321, "parallel", 2
    )
    assert single == parallel
    assert summary_single["edge_sets"] == summary_parallel["edge_sets"]
