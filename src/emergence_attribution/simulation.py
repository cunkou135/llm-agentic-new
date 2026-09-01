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
from .simulators import run_scenario


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


@dataclass(frozen=True)
class SimulationTaskResult:
    task_id: str
    status: str
    raw_path: str
    sha256: str
    duration_seconds: float
    error: str | None


def build_simulation_tasks(config: dict[str, Any], raw_root: Path) -> list[SimulationTask]:
    tasks: list[SimulationTask] = []
    for scenario, spec in sorted(config["scenarios"].items()):
        baseline = {name: float(value) for name, value in spec["baseline"].items()}
        conditions: list[tuple[str, str, str, str, dict[str, float]]] = [
            ("baseline", "", "baseline", "baseline", baseline)
        ]
        for parameter, levels in sorted(spec["interventions"].items()):
            for direction, value in (("minus", levels[0]), ("plus", levels[2])):
                revised = dict(baseline)
                revised[parameter] = float(value)
                conditions.append(
                    (f"{parameter}_{direction}", parameter, direction, "baseline", revised)
                )
        conditions.append(
            (
                "mechanism_disabled",
                "",
                "disabled",
                str(spec["mechanism_variant"]),
                baseline,
            )
        )
        for condition, parameter, direction, mechanism, parameters in conditions:
            for seed in config["random_seeds"]:
                raw_path = raw_root / scenario / condition / f"seed_{int(seed)}.npz"
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
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            _verify_npz(path)
            actual = sha256_file(path)
            if sidecar.exists():
                expected = sidecar.read_text(encoding="ascii").strip()
                if expected != actual:
                    raise RuntimeError("existing raw file hash does not match its checkpoint")
            else:
                sidecar.write_text(actual + "\n", encoding="ascii")
            return SimulationTaskResult(
                task.task_id, "verified_existing", str(path), actual, time.perf_counter() - started, None
            )
        raw = run_scenario(
            task.scenario,
            task.seed,
            task.scenario_spec,
            task.parameters,
            task.mechanism_variant,
        )
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(temporary, **raw)
        _verify_npz(temporary)
        os.replace(temporary, path)
        digest = sha256_file(path)
        sidecar.write_text(digest + "\n", encoding="ascii")
        return SimulationTaskResult(
            task.task_id, "completed", str(path), digest, time.perf_counter() - started, None
        )
    except Exception as exc:
        return SimulationTaskResult(
            task.task_id,
            "failed",
            str(path),
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
) -> dict[str, Any]:
    data_root = run_root / "data"
    raw_root = data_root / "raw_logs"
    tasks = build_simulation_tasks(config, raw_root)

    def update(completed: int, total: int, result: SimulationTaskResult) -> None:
        if progress_callback:
            progress_callback("Simulation", completed, total, result.task_id)

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
            }
        )
        task_records.append(record)
    manifest = {
        "schema_version": "1.0",
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
    manifest_path = data_root / "simulation_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    if failures:
        raise RuntimeError(
            f"{len(failures)} simulation tasks failed; inspect {manifest_path} and resume after correcting the environment"
        )
    return manifest


def verify_simulation_manifest(run_root: Path) -> None:
    manifest_path = run_root / "data" / "simulation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    errors = []
    for record in manifest["task_records"]:
        path = run_root / record["raw_path"]
        if not path.is_file():
            errors.append(f"missing {record['raw_path']}")
        elif sha256_file(path) != record["sha256"]:
            errors.append(f"hash mismatch {record['raw_path']}")
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
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("representation_sha256") != representation_hash:
            raise RuntimeError("saved indicator dataset belongs to another representation")
        if manifest.get("sha256") != sha256_file(output_path):
            raise RuntimeError("saved indicator dataset hash mismatch")
        return pd.read_parquet(output_path)
    simulation_manifest = json.loads(
        (data_root / "simulation_manifest.json").read_text(encoding="utf-8")
    )
    records = simulation_manifest["task_records"]
    if not complete:
        records = [item for item in records if item["condition"] == "baseline"]
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
