"""Export a generated visualisation bundle to a local plotting repository."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from emergence_attribution.exporting import (  # noqa: E402
    create_visualization_bundle,
    export_bundle_to_plot_repository,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--plot-repo", type=Path, required=True)
    args = parser.parse_args()
    run_root = args.run.resolve()
    create_visualization_bundle(run_root)
    destination = export_bundle_to_plot_repository(
        run_root, args.plot_repo.resolve()
    )
    print(destination)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

