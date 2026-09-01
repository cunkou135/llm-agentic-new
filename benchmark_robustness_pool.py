"""NON_SCIENTIFIC process-pool lifecycle benchmark for robustness jobs."""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergence_attribution.pipeline import load_experiment_config  # noqa: E402
from emergence_attribution.robustness import (  # noqa: E402
    _execute_robustness_bootstrap_jobs,
)
from emergence_attribution.temporal import discover_bootstrap_graph  # noqa: E402


def _job_count(config: dict) -> int:
    scenario_count = len(config["scenarios"])
    data = (
        scenario_count
        * len(config["evaluation"]["trajectory_counts"])
        * int(config["evaluation"]["repeated_subsampling_repetitions"])
    )
    observation = (
        scenario_count
        * (
            len(config["robustness"]["noise_levels"])
            + len(config["robustness"]["missing_fractions"])
            + len(config["robustness"]["support_thresholds"])
        )
        * int(config["robustness"]["repetitions"])
    )
    representation = (
        scenario_count
        * 5
        * len(config["robustness"]["representation_error_ratios"])
        * int(config["robustness"]["representation_repetitions"])
    )
    return data + observation + representation


def _normalise(graph, summary) -> str:
    return json.dumps(
        {
            "graph": [edge.__dict__ for edge in graph],
            "summary": summary,
        },
        sort_keys=True,
        allow_nan=True,
        separators=(",", ":"),
    )


def _legacy_run(payloads: list[dict], workers: int) -> tuple[list[str], float]:
    outputs: list[str] = []
    started = time.perf_counter()
    for payload in payloads:
        graph, summary = discover_bootstrap_graph(
            payload["frames"], payload["candidates"], payload["maximum_lag"],
            payload["parent_alpha"], payload["fdr_alpha"],
            payload["bootstrap_repetitions"], payload["support_threshold"],
            payload["master_seed"], payload["seed_label"], workers,
        )
        outputs.append(_normalise(graph, summary))
    return outputs, time.perf_counter() - started


def _reused_pool_run(
    payloads: list[dict], workers: int
) -> tuple[list[str], float]:
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as executor:
        results = _execute_robustness_bootstrap_jobs(payloads, executor)
    elapsed = time.perf_counter() - started
    return [
        _normalise(item["graph"], item["summary"])
        for item in results
    ], elapsed


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=2)
    args = parser.parse_args()
    if args.workers < 2:
        raise ValueError("the lifecycle benchmark requires at least two workers")
    output_root = PROJECT_ROOT / "smoke_runs" / args.run_id
    if output_root.exists():
        raise FileExistsError(f"benchmark output already exists: {output_root}")
    output_root.mkdir(parents=True)
    config = load_experiment_config(PROJECT_ROOT / "config" / "dev_experiment.json")
    rng = np.random.default_rng(20260902)
    frames = []
    for _ in range(2):
        source = rng.normal(size=36)
        target = np.roll(source, 1) + rng.normal(0.0, 0.08, size=36)
        frames.append(pd.DataFrame({"source": source, "target": target}))
    candidates = [
        {
            "source": "source", "target": "target",
            "branch_id": "benchmark_branch", "expected_direction": "unknown",
        }
    ]
    count = _job_count(config)
    payloads = [
        {
            "job_index": index,
            "frames": frames,
            "candidates": candidates,
            "maximum_lag": int(config["temporal"]["maximum_lag"]),
            "parent_alpha": float(config["temporal"]["parent_alpha"]),
            "fdr_alpha": float(config["temporal"]["fdr_alpha"]),
            "bootstrap_repetitions": int(
                config["robustness"]["bootstrap_repetitions"]
            ),
            "support_threshold": float(config["temporal"]["support_threshold"]),
            "master_seed": int(config["master_seed"]),
            "seed_label": f"dev-lifecycle-benchmark:{index}",
            "point_only": False,
            "include_vote": False,
        }
        for index in range(count)
    ]
    legacy_outputs, legacy_seconds = _legacy_run(payloads, args.workers)
    reused_outputs, reused_seconds = _reused_pool_run(payloads, args.workers)
    identical = legacy_outputs == reused_outputs
    report = {
        "status": "passed" if identical else "failed",
        "scientific_evidence": False,
        "workload": "synthetic temporal fits using NON_SCIENTIFIC dev configuration job count and robustness bootstrap repetitions",
        "workers": args.workers,
        "job_count": count,
        "bootstrap_repetitions_per_job": int(
            config["robustness"]["bootstrap_repetitions"]
        ),
        "legacy": {
            "pool_creations": count,
            "wall_time_seconds": legacy_seconds,
        },
        "reused_outer_pool": {
            "pool_creations": 1,
            "nested_pool_creations": 0,
            "wall_time_seconds": reused_seconds,
        },
        "pool_creation_reduction": count - 1,
        "wall_time_ratio_reused_over_legacy": reused_seconds / max(legacy_seconds, 1e-12),
        "outputs_identical": identical,
    }
    (output_root / "robustness_pool_benchmark.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, indent=2))
    print(output_root.resolve())
    return 0 if identical else 1


if __name__ == "__main__":
    raise SystemExit(main())
