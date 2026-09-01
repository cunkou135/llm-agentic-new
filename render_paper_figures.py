"""Render publication Figure 2--8 from a generated run."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergence_attribution.provenance import RunManager  # noqa: E402
from emergence_attribution.rendering import render_all_figures  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plot-repo", type=Path, required=True)
    parser.add_argument(
        "--formats", nargs="+", choices=["png", "svg", "pdf", "tiff"], default=None
    )
    args = parser.parse_args()
    run_root = args.run.resolve()
    config = json.loads(
        (run_root / "config" / "experiment_config.snapshot.json").read_text(
            encoding="utf-8"
        )
    )
    redacted = json.loads(
        (run_root / "config" / "llm_config.redacted.json").read_text(
            encoding="utf-8"
        )
    )
    redacted.pop("api_key_present", None)
    redacted["api_key"] = ""
    manager = RunManager.initialise(
        PROJECT_ROOT, run_root.name, config, redacted, resume=True
    )
    if manager.stage_complete("render"):
        raise RuntimeError(
            "render stage is already complete; refusing to overwrite published figures"
        )
    started = time.perf_counter()
    manifest = render_all_figures(
        run_root, args.plot_repo.resolve(), args.formats
    )
    outputs = [run_root / relative for relative in manifest["outputs"]]
    outputs.extend(
        [
            run_root / "figures" / "render_manifest.json",
            run_root / "visualization_input" / "render_manifest.json",
        ]
    )
    manager.mark_stage_completed(
        "render", outputs, time.perf_counter() - started, manifest
    )
    if all(manager.stage_complete(stage) for stage in ("simulation", "semantic", "temporal", "intervention", "robustness", "export", "render")):
        manager.finalise()
    print(run_root / "figures")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
