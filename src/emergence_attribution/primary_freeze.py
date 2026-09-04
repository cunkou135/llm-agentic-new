"""Immutable boundary between primary discovery and independent confirmation."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .provenance import RunContractError, sha256_file


PRIMARY_FREEZE_INPUTS = [
    "representation/indicators_frozen.json",
    "representation/INDICATORS_FROZEN.sha256",
    "representation/candidate_paths.json",
    "representation/CANDIDATE_PATHS_FROZEN.sha256",
    "representation/derived_candidate_edges.json",
    "representation/prospective_predictions.json",
    "representation/PROSPECTIVE_PREDICTIONS_FROZEN.sha256",
    "representation/representation_validation.json",
    "analysis/main_graphs.jsonl",
    "analysis/paired_effects.parquet",
    "analysis/mechanism_bidirectional_summary.csv",
    "analysis/intervention_classifications.csv",
    "analysis/edge_intervention_classifications.csv",
    "analysis/path_timing_summary.csv",
    "analysis/path_timing_concordance.csv",
    "analysis/path_temporal_qualification.csv",
    "analysis/path_intervention_classification.csv",
    "analysis/prospective_validation.csv",
]


def _representation_paths(run_root: Path) -> list[Path]:
    return sorted((run_root / "representation").glob("*_representation.json"))


def freeze_primary_contract(run_root: Path) -> Path:
    """Hash primary decisions once; confirmation is forbidden before this marker."""

    marker = run_root / "provenance" / "PRIMARY_DISCOVERY_FROZEN.json"
    if marker.exists():
        verify_primary_contract(run_root)
        return marker
    paths = [run_root / relative for relative in PRIMARY_FREEZE_INPUTS]
    paths.extend(_representation_paths(run_root))
    missing = [path.relative_to(run_root).as_posix() for path in paths if not path.is_file()]
    if missing:
        raise RunContractError(f"cannot freeze incomplete primary discovery: {missing}")
    payload: dict[str, Any] = {
        "schema_version": "1.0",
        "status": "frozen",
        "holdout_used_for_primary": False,
        "files": {
            path.relative_to(run_root).as_posix(): sha256_file(path)
            for path in sorted(set(paths))
        },
    }
    marker.parent.mkdir(parents=True, exist_ok=True)
    temporary = marker.with_suffix(".tmp.json")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    os.replace(temporary, marker)
    return marker


def verify_primary_contract(run_root: Path) -> dict[str, Any]:
    marker = run_root / "provenance" / "PRIMARY_DISCOVERY_FROZEN.json"
    if not marker.is_file():
        raise RunContractError(
            "holdout access is forbidden until primary discovery and prospective "
            "classification are frozen"
        )
    payload = json.loads(marker.read_text(encoding="utf-8"))
    if payload.get("status") != "frozen":
        raise RunContractError("primary discovery freeze marker is invalid")
    mismatches = []
    for relative, expected in payload.get("files", {}).items():
        path = run_root / relative
        if not path.is_file() or sha256_file(path) != expected:
            mismatches.append(relative)
    if mismatches:
        raise RunContractError(
            f"primary discovery changed after freezing: {sorted(mismatches)}"
        )
    return payload
