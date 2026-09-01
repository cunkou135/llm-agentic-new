"""Single-command entry point for the complete formal experiment."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergence_attribution.llm_client import load_llm_config  # noqa: E402
from emergence_attribution.pipeline import (  # noqa: E402
    STAGE_ORDER,
    load_experiment_config,
    run_selected_stages,
)
from emergence_attribution.provenance import RunManager  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the reproducible multiscale emergence attribution experiment."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--llm-config", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument(
        "--stage",
        action="append",
        choices=["all", *STAGE_ORDER],
        default=None,
        help="Repeat this option to request several stages; default is all.",
    )
    parser.add_argument("--workers", default="auto")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--no-render", action="store_true")
    parser.add_argument("--plot-repo", type=Path, default=None)
    return parser.parse_args()


def resolve_workers(value: str) -> int:
    if value.lower() == "auto":
        return max(1, (os.cpu_count() or 2) - 1)
    workers = int(value)
    if workers < 1:
        raise ValueError("workers must be a positive integer or auto")
    return workers


def main() -> int:
    args = parse_args()
    stages = args.stage or ["all"]
    if "all" in stages and len(stages) > 1:
        raise ValueError("--stage all cannot be combined with another stage")
    config_path = args.config.resolve()
    llm_config_path = args.llm_config.resolve()
    config = load_experiment_config(config_path)
    requires_key = "all" in stages or "semantic" in stages
    llm_config = load_llm_config(llm_config_path, require_key=requires_key)
    manager = RunManager.initialise(
        PROJECT_ROOT,
        args.run_id,
        config,
        llm_config,
        resume=args.resume,
    )
    run_selected_stages(
        stages,
        manager,
        resolve_workers(args.workers),
        PROJECT_ROOT / "config" / "semantic_prompt.txt",
        llm_config_path,
        no_render=args.no_render,
        plot_repo=args.plot_repo.resolve() if args.plot_repo else None,
    )
    print(manager.run_root.resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

