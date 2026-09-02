"""Immutable run contracts, stage checkpoints, hashes, and environment capture."""

from __future__ import annotations

import hashlib
import importlib.metadata
import json
import os
import platform
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .llm_client import public_config_hash, redacted_llm_config


class RunContractError(RuntimeError):
    pass


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _source_files(project_root: Path) -> list[Path]:
    included_suffixes = {".py", ".json", ".toml", ".txt", ".md"}
    excluded_parts = {
        "runs", "smoke_runs", "dev_runs", ".venv", "__pycache__",
        ".pytest_cache", "build", "dist",
    }
    files = []
    for path in project_root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in included_suffixes:
            continue
        relative = path.relative_to(project_root)
        if any(part in excluded_parts for part in relative.parts):
            continue
        if path.name == "llm_api.local.json":
            continue
        files.append(path)
    return sorted(files)


def source_manifest(project_root: Path) -> dict[str, str]:
    return {
        path.relative_to(project_root).as_posix(): sha256_file(path)
        for path in _source_files(project_root)
    }


def _environment_text() -> str:
    distributions = sorted(
        f"{distribution.metadata['Name']}=={distribution.version}"
        for distribution in importlib.metadata.distributions()
        if distribution.metadata.get("Name")
    )
    lines = [
        f"python={sys.version}",
        f"executable={sys.executable}",
        f"platform={platform.platform()}",
        f"processor={platform.processor()}",
        "",
        "installed_distributions:",
        *distributions,
    ]
    return "\n".join(lines) + "\n"


def _relative_hashes(run_root: Path, paths: Iterable[Path]) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(set(paths)):
        if path.is_file():
            result[path.relative_to(run_root).as_posix()] = sha256_file(path)
    return result


@dataclass
class RunManager:
    project_root: Path
    run_root: Path
    config: dict[str, Any]
    llm_config: dict[str, Any]
    resume: bool

    @classmethod
    def initialise(
        cls,
        project_root: Path,
        run_id: str,
        config: dict[str, Any],
        llm_config: dict[str, Any],
        *,
        resume: bool,
        output_family: str = "runs",
    ) -> "RunManager":
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", run_id):
            raise RunContractError("run id contains unsupported characters")
        if output_family not in {"runs", "dev_runs", "smoke_runs"}:
            raise RunContractError("unsupported run output family")
        if bool(config.get("formal_run", True)) and output_family != "runs":
            raise RunContractError("formal outputs must be written below runs")
        if not bool(config.get("formal_run", True)) and output_family == "runs":
            raise RunContractError("non-scientific outputs must not be written below runs")
        run_root = project_root / output_family / run_id
        manager = cls(project_root, run_root, config, llm_config, resume)
        if run_root.exists():
            if not resume:
                raise RunContractError(
                    f"run already exists and will not be overwritten: {run_root}"
                )
            manager._verify_existing_contract()
        else:
            if resume:
                raise RunContractError(f"cannot resume a run that does not exist: {run_root}")
            manager._create_contract()
        return manager

    @property
    def manifest_path(self) -> Path:
        return self.run_root / "provenance" / "run_manifest.json"

    def _contract(self) -> dict[str, Any]:
        sources = source_manifest(self.project_root)
        return {
            "schema_version": "1.0",
            "experiment_config_sha256": sha256_json(self.config),
            "llm_public_config_sha256": public_config_hash(self.llm_config),
            "source_manifest_sha256": sha256_json(sources),
            "simulation_contract_sha256": sha256_json(
                {
                    "random_seeds": self.config["random_seeds"],
                    "scenarios": self.config["scenarios"],
                }
            ),
            "representation_contract_sha256": sha256_json(
                self.config["representation"]
            ),
            "temporal_contract_sha256": sha256_json(self.config["temporal"]),
            "intervention_contract_sha256": sha256_json(
                self.config["intervention"]
            ),
            "robustness_contract_sha256": sha256_json(self.config["robustness"]),
            "render_contract_sha256": sha256_json(self.config["render"]),
        }

    def _create_contract(self) -> None:
        for relative in (
            "config",
            "provenance/stages",
            "llm",
            "representation",
            "data/raw_logs",
            "data/reference_hidden",
            "analysis",
            "visualization_input",
            "figures",
            "tables",
            "logs",
        ):
            (self.run_root / relative).mkdir(parents=True, exist_ok=True)
        (self.run_root / "config" / "experiment_config.snapshot.json").write_text(
            json.dumps(self.config, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.run_root / "config" / "llm_config.redacted.json").write_text(
            json.dumps(redacted_llm_config(self.llm_config), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        sources = source_manifest(self.project_root)
        (self.run_root / "provenance" / "source_manifest.json").write_text(
            json.dumps(sources, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.run_root / "provenance" / "environment.txt").write_text(
            _environment_text(), encoding="utf-8"
        )
        manifest = {
            **self._contract(),
            "run_id": self.run_root.name,
            "status": "running",
            "created_unix_time": time.time(),
            "stage_status": {},
            "manual_artifact_edits": 0,
            "formal_output": bool(self.config.get("formal_run", True)),
        }
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _verify_existing_contract(self) -> None:
        if (self.run_root / "RUN_FROZEN").exists():
            raise RunContractError("completed run is frozen and cannot be resumed")
        if not self.manifest_path.is_file():
            raise RunContractError("existing run lacks its provenance manifest")
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        current = self._contract()
        mismatches = {
            key: {"saved": manifest.get(key), "current": value}
            for key, value in current.items()
            if manifest.get(key) != value
        }
        if mismatches:
            raise RunContractError(
                "run contract changed; create a new run id: "
                + json.dumps(mismatches, indent=2)
            )

    def stage_complete(self, stage: str) -> bool:
        checkpoint = self.run_root / "provenance" / "stages" / f"{stage}.json"
        if not checkpoint.is_file():
            return False
        value = json.loads(checkpoint.read_text(encoding="utf-8"))
        if value.get("status") != "completed":
            return False
        for relative, expected in value.get("outputs", {}).items():
            path = self.run_root / relative
            if not path.is_file() or sha256_file(path) != expected:
                raise RunContractError(
                    f"completed stage {stage} has a missing or modified output: {relative}"
                )
        return True

    def mark_stage_completed(
        self,
        stage: str,
        outputs: Iterable[Path],
        duration_seconds: float,
        details: dict[str, Any] | None = None,
    ) -> None:
        output_hashes = _relative_hashes(self.run_root, outputs)
        checkpoint = {
            "stage": stage,
            "status": "completed",
            "duration_seconds": duration_seconds,
            "outputs": output_hashes,
            "details": details or {},
        }
        checkpoint_path = self.run_root / "provenance" / "stages" / f"{stage}.json"
        checkpoint_path.write_text(
            json.dumps(checkpoint, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest["stage_status"][stage] = "completed"
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        self._write_timing_summary()

    def record_timestamp(self, name: str) -> None:
        """Record an auditable wall-clock boundary in the run manifest."""

        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest.setdefault("event_timestamps", {})[name] = time.time()
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )

    def _write_timing_summary(self) -> None:
        stages = {}
        for path in sorted((self.run_root / "provenance" / "stages").glob("*.json")):
            value = json.loads(path.read_text(encoding="utf-8"))
            stages[value["stage"]] = float(value.get("duration_seconds", 0.0))
        summary = {
            "stage_seconds": stages,
            "total_stage_seconds": sum(stages.values()),
        }
        method_path = self.run_root / "analysis" / "method_runtime.csv"
        if method_path.exists():
            import csv

            with method_path.open("r", encoding="utf-8", newline="") as handle:
                method_rows = list(csv.DictReader(handle))
            summary["method_runtime_seconds"] = method_rows
            full = {
                row["scenario"]: float(row["runtime_seconds"])
                for row in method_rows
                if row["method"] == "full_method"
            }
            summary["speedup_vs_full"] = [
                {
                    **row,
                    "speedup": full.get(row["scenario"], float("nan"))
                    / max(float(row["runtime_seconds"]), 1e-12),
                }
                for row in method_rows
                if row.get("runtime_scope") == "temporal_analysis"
            ]
        (self.run_root / "analysis" / "timing_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )

    def finalise(self) -> None:
        excluded = {
            "provenance/hashes.json",
            "provenance/run_manifest.json",
            "RUN_FROZEN",
        }
        hashes = {}
        for path in sorted(self.run_root.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(self.run_root).as_posix()
            if relative in excluded:
                continue
            hashes[relative] = sha256_file(path)
        hashes_path = self.run_root / "provenance" / "hashes.json"
        hashes_path.write_text(
            json.dumps(hashes, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        manifest = json.loads(self.manifest_path.read_text(encoding="utf-8"))
        manifest.update(
            {
                "status": "completed",
                "completed_unix_time": time.time(),
                "artifact_count": len(hashes),
                "hashes_manifest_sha256": sha256_file(hashes_path),
            }
        )
        self.manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        (self.run_root / "RUN_FROZEN").write_text(
            "This run completed under an immutable contract. Create a new run id for any change.\n",
            encoding="utf-8",
        )
