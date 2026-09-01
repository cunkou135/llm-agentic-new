"""Run the isolated NON_SCIENTIFIC development pipeline with resume coverage."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergence_attribution.llm_client import load_llm_config  # noqa: E402
from emergence_attribution.mock_semantic import mock_completion_provider  # noqa: E402
from emergence_attribution.pipeline import (  # noqa: E402
    STAGE_ORDER,
    load_experiment_config,
    run_selected_stages,
)
from emergence_attribution.provenance import RunManager  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--plot-repo", type=Path, default=None)
    args = parser.parse_args()
    config_path = PROJECT_ROOT / "config" / "dev_experiment.json"
    llm_path = PROJECT_ROOT / "config" / "llm_api.mock.json"
    config = load_experiment_config(config_path)
    llm = load_llm_config(llm_path, require_key=False)
    manager = RunManager.initialise(
        PROJECT_ROOT, args.run_id, config, llm, resume=False,
        output_family="dev_runs",
    )
    (manager.run_root / "NON_SCIENTIFIC.json").write_text(
        json.dumps({"scientific_output": False, "real_llm_api_called": False}, indent=2),
        encoding="utf-8",
    )
    run_selected_stages(
        ["semantic", "baseline_simulation"], manager, args.workers,
        PROJECT_ROOT / "config" / "semantic_prompt.txt", llm_path,
        no_render=False, plot_repo=args.plot_repo,
        completion_provider=mock_completion_provider,
    )
    resumed = RunManager.initialise(
        PROJECT_ROOT, args.run_id, config, llm, resume=True,
        output_family="dev_runs",
    )
    run_selected_stages(
        STAGE_ORDER[2:], resumed, args.workers,
        PROJECT_ROOT / "config" / "semantic_prompt.txt", llm_path,
        no_render=False, plot_repo=args.plot_repo,
        completion_provider=mock_completion_provider,
    )
    print(resumed.run_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

