"""Simulation task matrix, atomic persistence, indicator compilation, and resume."""

from __future__ import annotations

import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np
import pandas as pd

from .dsl import compute_indicator
from .raw_schemas import raw_schema
from .raw_schemas import HIDDEN_REFERENCE_FIELD_NAMES
from .simulators import run_scenario_with_hidden


METADATA_COLUMNS = {
    "scenario",
    "seed",
    "condition",
    "intervention_parameter",
    "intervention_direction",
    "mechanism_variant",
    "time",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SimulationTask:
    task_id: str
    scenario: str
    seed: int
    condition: str
    intervention_parameter: str
    intervention_direction: str
    mechanism_variant: str
    scenario_spec: dict[str, Any]
    parameters: dict[str, float]
    raw_path: str
    hidden_path: str
    phase: str
    formal_run: bool


@dataclass(frozen=True)
class SimulationTaskResult:
    task_id: str
    status: str
    raw_path: str
    sha256: str
    hidden_path: str
    hidden_sha256: str
    duration_seconds: float
    error: str | None


def build_simulation_tasks(
    config: dict[str, Any],
    raw_root: Path,
    hidden_root: Path | None = None,
    *,
    phase: str = "all",
) -> list[SimulationTask]:
    if phase not in {"baseline", "intervention", "all"}:
        raise ValueError(f"unknown simulation phase: {phase}")
    hidden_root = hidden_root or raw_root.parent / "reference_hidden"
    tasks: list[SimulationTask] = []
    for scenario, spec in sorted(config["scenarios"].items()):
        baseline = {name: float(value) for name, value in spec["baseline"].items()}
        conditions: list[tuple[str, str, str, str, dict[str, float]]] = []
        if phase in {"baseline", "all"}:
            conditions.append(("baseline", "", "baseline", "baseline", baseline))
        if phase in {"intervention", "all"}:
            intervention_conditions: list[tuple[str, str, str, str, dict[str, float]]] = []
        else:
            intervention_conditions = []
        for parameter, levels in sorted(spec["interventions"].items()):
            for direction, value in (("minus", levels[0]), ("plus", levels[2])):
                revised = dict(baseline)
                revised[parameter] = float(value)
                intervention_conditions.append(
                    (f"{parameter}_{direction}", parameter, direction, "baseline", revised)
                )
        if phase in {"intervention", "all"}:
            intervention_conditions.append(
                (
                    "mechanism_disabled",
                    "",
                    "disabled",
                    str(spec["mechanism_variant"]),
                    baseline,
                )
            )
            conditions.extend(intervention_conditions)
        for condition, parameter, direction, mechanism, parameters in conditions:
            for seed in config["random_seeds"]:
                raw_path = raw_root / scenario / condition / f"seed_{int(seed)}.npz"
                hidden_path = hidden_root / scenario / condition / f"seed_{int(seed)}.npz"
                tasks.append(
                    SimulationTask(
                        task_id=f"{scenario}:{condition}:{int(seed)}",
                        scenario=scenario,
                        seed=int(seed),
                        condition=condition,
                        intervention_parameter=parameter,
                        intervention_direction=direction,
                        mechanism_variant=mechanism,
                        scenario_spec=spec,
                        parameters=parameters,
                        raw_path=str(raw_path),
                        hidden_path=str(hidden_path),
                        phase="baseline" if condition == "baseline" else "intervention",
                        formal_run=bool(config.get("formal_run", False)),
                    )
                )
    return sorted(tasks, key=lambda item: item.task_id)


def _verify_npz(path: Path) -> None:
    with np.load(path, allow_pickle=False) as archive:
        if not archive.files:
            raise RuntimeError("empty NPZ archive")
        for name in archive.files:
            _ = archive[name].shape


def execute_simulation_task(task: SimulationTask) -> SimulationTaskResult:
    started = time.perf_counter()
    path = Path(task.raw_path)
    sidecar = path.with_suffix(path.suffix + ".sha256")
    hidden_path = Path(task.hidden_path)
    hidden_sidecar = hidden_path.with_suffix(hidden_path.suffix + ".sha256")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        hidden_path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() or hidden_path.exists():
            if not path.exists() or not hidden_path.exists():
                raise RuntimeError("public/hidden checkpoint pair is incomplete")
            _verify_npz(path)
            _verify_npz(hidden_path)
            actual = sha256_file(path)
            hidden_actual = sha256_file(hidden_path)
            if not sidecar.exists() or not hidden_sidecar.exists():
                if task.formal_run:
                    raise RuntimeError("formal existing NPZ requires its original sidecar")
                raise RuntimeError("existing NPZ checkpoint pair requires sidecars")
            expected = sidecar.read_text(encoding="ascii").strip()
            hidden_expected = hidden_sidecar.read_text(encoding="ascii").strip()
            if expected != actual or hidden_expected != hidden_actual:
                raise RuntimeError("existing NPZ hash does not match its checkpoint")
            with np.load(path, allow_pickle=False) as archive:
                overlap = set(archive.files) & HIDDEN_REFERENCE_FIELD_NAMES
            if overlap:
                raise RuntimeError(f"public NPZ contains hidden fields: {sorted(overlap)}")
            return SimulationTaskResult(
                task.task_id, "verified_existing", str(path), actual,
                str(hidden_path), hidden_actual, time.perf_counter() - started, None
            )
        raw, hidden = run_scenario_with_hidden(
            task.scenario,
            task.seed,
            task.scenario_spec,
            task.parameters,
            task.mechanism_variant,
        )
        overlap = set(raw) & (set(hidden) | HIDDEN_REFERENCE_FIELD_NAMES)
        if overlap:
            raise RuntimeError(f"public simulator payload contains hidden fields: {sorted(overlap)}")
        temporary = path.with_suffix(".tmp.npz")
        hidden_temporary = hidden_path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **raw)
        np.savez_compressed(hidden_temporary, **hidden)
        _verify_npz(temporary)
        _verify_npz(hidden_temporary)
        os.replace(temporary, path)
        os.replace(hidden_temporary, hidden_path)
        digest = sha256_file(path)
        hidden_digest = sha256_file(hidden_path)
        sidecar.write_text(digest + "\n", encoding="ascii")
        hidden_sidecar.write_text(hidden_digest + "\n", encoding="ascii")
        return SimulationTaskResult(
            task.task_id, "completed", str(path), digest,
            str(hidden_path), hidden_digest, time.perf_counter() - started, None
        )
    except Exception as exc:
        return SimulationTaskResult(
            task.task_id,
            "failed",
            str(path),
            "",
            str(hidden_path),
            "",
            time.perf_counter() - started,
            f"{type(exc).__name__}: {exc}",
        )


def _run_jobs(
    function: Callable[[Any], Any],
    jobs: list[Any],
    workers: int,
    callback: Callable[[int, int, Any], None] | None = None,
) -> list[Any]:
    results: list[Any] = []
    if workers <= 1:
        for job in jobs:
            result = function(job)
            results.append(result)
            if callback:
                callback(len(results), len(jobs), result)
        return results
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(function, job): job for job in jobs}
        for future in as_completed(futures):
            result = future.result()
            results.append(result)
            if callback:
                callback(len(results), len(jobs), result)
    return results


def run_simulation_stage(
    config: dict[str, Any],
    run_root: Path,
    workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
    *,
    phase: str = "all",
) -> dict[str, Any]:
    data_root = run_root / "data"
    raw_root = data_root / "raw_logs"
    hidden_root = data_root / "reference_hidden"
    tasks = build_simulation_tasks(
        config, raw_root, hidden_root, phase=phase
    )

    def update(completed: int, total: int, result: SimulationTaskResult) -> None:
        if progress_callback:
            progress_callback(f"{phase.title()} simulation", completed, total, result.task_id)

    results = _run_jobs(execute_simulation_task, tasks, workers, update)
    failures = [item for item in results if item.status == "failed"]
    result_lookup = {item.task_id: item for item in results}
    task_records = []
    for task in tasks:
        result = result_lookup[task.task_id]
        record = asdict(task)
        record.pop("scenario_spec")
        record.pop("parameters")
        record.update(
            {
                "status": result.status,
                "sha256": result.sha256,
                "duration_seconds": result.duration_seconds,
                "error": result.error,
                "raw_path": str(Path(result.raw_path).relative_to(run_root)),
                "hidden_path": str(Path(result.hidden_path).relative_to(run_root)),
                "hidden_sha256": result.hidden_sha256,
            }
        )
        task_records.append(record)
    manifest = {
        "schema_version": "1.0",
        "phase": phase,
        "simulation_contract_sha256": sha256_json(
            {
                "random_seeds": config["random_seeds"],
                "scenarios": config["scenarios"],
            }
        ),
        "requested_tasks": len(tasks),
        "completed_tasks": len(tasks) - len(failures),
        "failed_tasks": len(failures),
        "pairing": "same initial state and counter-based random streams by scenario and seed",
        "task_records": task_records,
    }
    data_root.mkdir(parents=True, exist_ok=True)
    manifest_path = data_root / f"{phase}_simulation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        raise RuntimeError(
            f"{len(failures)} simulation tasks failed; inspect {manifest_path} and resume after correcting the environment"
        )
    return manifest


def run_baseline_simulation_stage(
    config: dict[str, Any], run_root: Path, workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    return run_simulation_stage(
        config, run_root, workers, progress_callback, phase="baseline"
    )


def run_intervention_simulation_stage(
    config: dict[str, Any], run_root: Path, workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    return run_simulation_stage(
        config, run_root, workers, progress_callback, phase="intervention"
    )


def verify_simulation_manifest(run_root: Path, phase: str = "all") -> None:
    manifest_path = run_root / "data" / f"{phase}_simulation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for record in manifest["task_records"]:
        path = run_root / record["raw_path"]
        if not path.is_file():
            errors.append(f"missing {record['raw_path']}")
        elif sha256_file(path) != record["sha256"]:
            errors.append(f"hash mismatch {record['raw_path']}")
        hidden_path = run_root / record["hidden_path"]
        if not hidden_path.is_file():
            errors.append(f"missing {record['hidden_path']}")
        elif sha256_file(hidden_path) != record["hidden_sha256"]:
            errors.append(f"hash mismatch {record['hidden_path']}")
    if errors:
        raise RuntimeError("simulation manifest verification failed: " + "; ".join(errors[:10]))


@dataclass(frozen=True)
class IndicatorCompilationTask:
    scenario: str
    seed: int
    condition: str
    intervention_parameter: str
    intervention_direction: str
    mechanism_variant: str
    raw_path: str
    indicators: list[dict[str, Any]]


def compile_indicator_task(task: IndicatorCompilationTask) -> list[dict[str, Any]]:
    with np.load(task.raw_path, allow_pickle=False) as archive:
        raw = {name: archive[name] for name in archive.files}
    schema = raw_schema(task.scenario)
    values: dict[str, np.ndarray] = {}
    for indicator in task.indicators:
        values[indicator["id"]] = compute_indicator(
            indicator["computation"],
            indicator["temporal_aggregation"],
            raw,
            schema,
        )
    steps = len(next(iter(values.values())))
    records: list[dict[str, Any]] = []
    for time_index in range(steps):
        record: dict[str, Any] = {
            "scenario": task.scenario,
            "seed": task.seed,
            "condition": task.condition,
            "intervention_parameter": task.intervention_parameter,
            "intervention_direction": task.intervention_direction,
            "mechanism_variant": task.mechanism_variant,
            "time": time_index,
        }
        record.update({name: float(series[time_index]) for name, series in values.items()})
        records.append(record)
    return records


def compile_indicator_dataset(
    config: dict[str, Any],
    run_root: Path,
    representations: dict[str, dict[str, Any]],
    workers: int,
    *,
    complete: bool,
    output_stem: str | None = None,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> pd.DataFrame:
    data_root = run_root / "data"
    kind = "complete" if complete else "baseline"
    stem = output_stem or f"indicator_trajectories_{kind}"
    output_path = data_root / f"{stem}.parquet"
    manifest_path = data_root / f"{stem}.manifest.json"
    representation_hash = sha256_json(representations)
    if output_path.exists() != manifest_path.exists():
        raise RuntimeError("saved indicator dataset checkpoint is incomplete")
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("representation_sha256") != representation_hash:
            raise RuntimeError("saved indicator dataset belongs to another representation")
        if manifest.get("sha256") != sha256_file(output_path):
            raise RuntimeError("saved indicator dataset hash mismatch")
        return pd.read_parquet(output_path)
    phases = ["baseline", "intervention"] if complete else ["baseline"]
    records: list[dict[str, Any]] = []
    for phase in phases:
        phase_path = data_root / f"{phase}_simulation_manifest.json"
        if not phase_path.exists():
            legacy = data_root / "all_simulation_manifest.json"
            if legacy.exists():
                phase_path = legacy
            else:
                raise FileNotFoundError(f"missing simulation phase manifest: {phase_path}")
        phase_records = json.loads(phase_path.read_text(encoding="utf-8"))["task_records"]
        if phase == "baseline":
            phase_records = [item for item in phase_records if item["condition"] == "baseline"]
        elif phase_path.name == "all_simulation_manifest.json":
            phase_records = [item for item in phase_records if item["condition"] != "baseline"]
        records.extend(phase_records)
    tasks = [
        IndicatorCompilationTask(
            scenario=item["scenario"],
            seed=int(item["seed"]),
            condition=item["condition"],
            intervention_parameter=item["intervention_parameter"],
            intervention_direction=item["intervention_direction"],
            mechanism_variant=item["mechanism_variant"],
            raw_path=str(run_root / item["raw_path"]),
            indicators=representations[item["scenario"]]["indicators"],
        )
        for item in records
    ]

    def update(completed: int, total: int, _result: Any) -> None:
        if progress_callback:
            progress_callback("Indicator compilation", completed, total, kind)

    compiled = _run_jobs(compile_indicator_task, tasks, workers, update)
    flat = [record for task_records in compiled for record in task_records]
    frame = pd.DataFrame(flat).sort_values(
        ["scenario", "condition", "seed", "time"], ignore_index=True
    )
    temporary = output_path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    manifest = {
        "schema_version": "1.0",
        "kind": kind,
        "representation_sha256": representation_hash,
        "row_count": len(frame),
        "columns": list(frame.columns),
        "sha256": sha256_file(output_path),
    }
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return frame


def trajectories(
    dataset: pd.DataFrame,
    scenario: str,
    condition: str = "baseline",
    columns: Iterable[str] | None = None,
) -> dict[int, pd.DataFrame]:
    subset = dataset[
        (dataset["scenario"] == scenario) & (dataset["condition"] == condition)
    ]
    if columns is None:
        selected = [column for column in subset.columns if column not in METADATA_COLUMNS]
    else:
        selected = list(columns)
    return {
        int(seed): group.sort_values("time")[selected].reset_index(drop=True)
        for seed, group in subset.groupby("seed", sort=True)
    }
