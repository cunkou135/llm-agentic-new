"""Deterministic non-scientific wiring smoke check."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergence_attribution.dsl import compute_indicator  # noqa: E402
from emergence_attribution.interventions import _effect_job  # noqa: E402
from emergence_attribution.raw_schemas import raw_schema  # noqa: E402
from emergence_attribution.simulators import simulate_toy  # noqa: E402
from emergence_attribution.temporal import (  # noqa: E402
    discover_bootstrap_graph,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", default="smoke_local")
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    output = PROJECT_ROOT / "smoke_runs" / args.run_id
    if output.exists():
        raise FileExistsError(f"smoke output already exists: {output}")
    output.mkdir(parents=True)
    (output / "NON_SCIENTIFIC").write_text(
        "Toy software-wiring output. Never use as scientific evidence.\n",
        encoding="utf-8",
    )
    frames = []
    for seed in range(6):
        raw = simulate_toy(seed, steps=100, agents=12)
        x_mean = compute_indicator(
            {"op": "mean", "input": {"op": "field", "name": "x"}, "axis": "agent"},
            {"op": "identity"},
            raw,
            raw_schema("toy"),
        )
        frames.append(pd.DataFrame({"x_mean": x_mean, "y": raw["y"]}))
    candidates = [
        {
            "source": "x_mean",
            "target": "y",
            "branch_id": "toy_branch",
            "expected_direction": "increase",
        }
    ]
    graph, summary = discover_bootstrap_graph(
        frames,
        candidates,
        5,
        0.10,
        0.05,
        8,
        0.50,
        1234,
        "toy_smoke",
        args.workers,
    )
    baseline = np.zeros((6, 40, 1))
    intervention = np.ones((6, 40, 1))
    effects = _effect_job(
        {
            "scenario": "toy",
            "parameter": "toy_parameter",
            "direction": "plus",
            "node_ids": ["toy_indicator"],
            "scales": {"toy_indicator": "micro"},
            "baseline": baseline,
            "intervention": intervention,
            "config": {
                "bootstrap_repetitions": 20,
                "confidence_level": 0.95,
                "onset_detection_start": 0,
                "minimum_standardised_effect": 0.10,
                "onset_consecutive_steps": 4,
                "evaluation_start": 15,
                "terminal_window": 10,
            },
            "paired": True,
            "seed": 99,
        }
    )
    report = {
        "status": "passed",
        "scientific_evidence": False,
        "temporal_edges": [edge.__dict__ for edge in graph],
        "bootstrap_repetitions": summary["bootstrap_repetitions"],
        "paired_effect": effects["summaries"][0],
    }
    (output / "smoke_report.json").write_text(
        json.dumps(report, indent=2, allow_nan=True), encoding="utf-8"
    )
    print(output.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

