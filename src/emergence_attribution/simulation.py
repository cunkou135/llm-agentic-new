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
from .primary_freeze import verify_primary_contract
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
    "evaluation_track",
    "data_partition",
    "phase",
    "dose_label",
    "time",
}


PRIMARY_PARTITION = "primary"
HOLDOUT_PARTITION = "holdout"
DATA_PARTITIONS = {PRIMARY_PARTITION, HOLDOUT_PARTITION}


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
    evaluation_track: str = "primary_discovery"
    data_partition: str = PRIMARY_PARTITION
    dose_label: str = "baseline"


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
    dataset_partition: str | None = None,
) -> list[SimulationTask]:
    """Build a deterministic, non-overlapping simulation matrix.

    ``dataset_partition=None`` preserves the pre-final API: it uses
    ``random_seeds`` and only the canonical minus/plus interventions.  Final
    protocol callers must select ``primary`` or ``holdout`` explicitly.  The
    primary partition adds the two pre-specified in-range intermediate doses;
    the holdout partition uses an independent seed pool and canonical
    minus/plus conditions only.
    """
    if phase not in {"baseline", "intervention", "all"}:
        raise ValueError(f"unknown simulation phase: {phase}")
    if dataset_partition is not None and dataset_partition not in DATA_PARTITIONS:
        raise ValueError(f"unknown simulation dataset partition: {dataset_partition}")
    primary_seeds = tuple(int(seed) for seed in config["random_seeds"])
    confirmation_seeds = tuple(int(seed) for seed in config.get("confirmation_seeds", ()))
    overlap = sorted(set(primary_seeds) & set(confirmation_seeds))
    if overlap:
        raise ValueError(f"primary and holdout seed pools overlap: {overlap}")
    if dataset_partition == HOLDOUT_PARTITION:
        if not confirmation_seeds:
            raise ValueError("holdout simulation requires confirmation_seeds")
        seeds = confirmation_seeds
    else:
        seeds = primary_seeds
    use_intermediate_doses = bool(
        dataset_partition == PRIMARY_PARTITION
        and config.get("dose_response", {}).get("enabled", False)
    )
    hidden_root = hidden_root or raw_root.parent / "reference_hidden"
    tasks: list[SimulationTask] = []
    for scenario, spec in sorted(config["scenarios"].items()):
        baseline = {name: float(value) for name, value in spec["baseline"].items()}
        conditions: list[
            tuple[str, str, str, str, dict[str, float], str, str]
        ] = []
        if phase in {"baseline", "all"}:
            baseline_track = (
                "holdout_confirmation"
                if dataset_partition == HOLDOUT_PARTITION
                else "primary_discovery"
            )
            conditions.append(
                ("baseline", "", "baseline", "baseline", baseline, baseline_track, "baseline")
            )
        if phase in {"intervention", "all"}:
            intervention_conditions: list[
                tuple[str, str, str, str, dict[str, float], str, str]
            ] = []
        else:
            intervention_conditions = []
        for parameter, levels in sorted(spec["interventions"].items()):
            if len(levels) != 3:
                raise ValueError(
                    f"{scenario}:{parameter} must define [minus, baseline, plus]"
                )
            if not np.isclose(float(levels[1]), baseline[parameter]):
                raise ValueError(
                    f"{scenario}:{parameter} intervention centre differs from baseline"
                )
            dose_values: list[tuple[str, float]] = [
                ("minus", float(levels[0])),
                ("plus", float(levels[2])),
            ]
            if use_intermediate_doses:
                dose_values = [
                    ("minus", float(levels[0])),
                    ("mid_minus", (float(levels[0]) + float(levels[1])) / 2.0),
                    ("mid_plus", (float(levels[1]) + float(levels[2])) / 2.0),
                    ("plus", float(levels[2])),
                ]
            for direction, value in dose_values:
                revised = dict(baseline)
                revised[parameter] = float(value)
                if dataset_partition == HOLDOUT_PARTITION:
                    evaluation_track = "holdout_confirmation"
                elif direction in {"mid_minus", "mid_plus"}:
                    evaluation_track = "dose_response"
                else:
                    evaluation_track = "primary_discovery"
                intervention_conditions.append(
                    (
                        f"{parameter}_{direction}", parameter, direction,
                        "baseline", revised, evaluation_track, direction,
                    )
                )
        if phase in {"intervention", "all"}:
            mechanism_track = (
                "holdout_confirmation"
                if dataset_partition == HOLDOUT_PARTITION
                else "falsification_control"
            )
            intervention_conditions.append(
                (
                    "mechanism_disabled",
                    "",
                    "disabled",
                    str(spec["mechanism_variant"]),
                    baseline,
                    mechanism_track,
                    "mechanism_disabled",
                )
            )
            conditions.extend(intervention_conditions)
        for (
            condition, parameter, direction, mechanism, parameters,
            evaluation_track, dose_label,
        ) in conditions:
            for seed in seeds:
                raw_path = raw_root / scenario / condition / f"seed_{int(seed)}.npz"
                hidden_path = hidden_root / scenario / condition / f"seed_{int(seed)}.npz"
                partition_label = dataset_partition or "legacy"
                tasks.append(
                    SimulationTask(
                        task_id=(
                            f"{partition_label}:{scenario}:{condition}:{int(seed)}"
                            if dataset_partition is not None
                            else f"{scenario}:{condition}:{int(seed)}"
                        ),
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
                        evaluation_track=evaluation_track,
                        data_partition=partition_label,
                        dose_label=dose_label,
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
    dataset_partition: str = PRIMARY_PARTITION,
) -> dict[str, Any]:
    if dataset_partition not in DATA_PARTITIONS:
        raise ValueError(f"unknown simulation dataset partition: {dataset_partition}")
    if dataset_partition == HOLDOUT_PARTITION:
        _require_primary_freeze(run_root)
    data_root = run_root / "data"
    partition_root = data_root / dataset_partition
    raw_root = partition_root / "raw_logs"
    hidden_root = partition_root / "reference_hidden"
    tasks = build_simulation_tasks(
        config, raw_root, hidden_root, phase=phase,
        dataset_partition=dataset_partition,
    )

    def update(completed: int, total: int, result: SimulationTaskResult) -> None:
        if progress_callback:
            label = {
                (PRIMARY_PARTITION, "baseline"): "Primary baseline",
                (PRIMARY_PARTITION, "intervention"): "Primary dose interventions",
                (HOLDOUT_PARTITION, "baseline"): "Holdout baseline",
                (HOLDOUT_PARTITION, "intervention"): "Holdout interventions",
            }[(dataset_partition, phase)]
            progress_callback(label, completed, total, result.task_id)

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
        "schema_version": "2.0",
        "phase": phase,
        "data_partition": dataset_partition,
        "evaluation_tracks": sorted({task.evaluation_track for task in tasks}),
        "simulation_contract_sha256": sha256_json(
            {
                "data_partition": dataset_partition,
                "seeds": (
                    config["random_seeds"]
                    if dataset_partition == PRIMARY_PARTITION
                    else config["confirmation_seeds"]
                ),
                "scenarios": config["scenarios"],
                "dose_response": config.get("dose_response", {}),
            }
        ),
        "requested_tasks": len(tasks),
        "completed_tasks": len(tasks) - len(failures),
        "failed_tasks": len(failures),
        "pairing": "same initial state and counter-based random streams by scenario and seed",
        "task_records": task_records,
    }
    partition_root.mkdir(parents=True, exist_ok=True)
    manifest_text = json.dumps(manifest, indent=2, ensure_ascii=False)
    manifest_path = partition_root / f"{phase}_simulation_manifest.json"
    manifest_path.write_text(manifest_text, encoding="utf-8")
    # Primary aliases keep pre-final readers functional while every payload is
    # physically isolated under data/primary.  Holdout never receives aliases,
    # so it cannot be mistaken for discovery input.
    if dataset_partition == PRIMARY_PARTITION:
        (data_root / f"{phase}_simulation_manifest.json").write_text(
            manifest_text, encoding="utf-8"
        )
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
        config, run_root, workers, progress_callback, phase="baseline",
        dataset_partition=PRIMARY_PARTITION,
    )


def run_intervention_simulation_stage(
    config: dict[str, Any], run_root: Path, workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    return run_simulation_stage(
        config, run_root, workers, progress_callback, phase="intervention",
        dataset_partition=PRIMARY_PARTITION,
    )


def _require_primary_freeze(run_root: Path) -> None:
    """Fail closed until the primary prospective stage is frozen."""
    verify_primary_contract(run_root)


def run_holdout_baseline_simulation_stage(
    config: dict[str, Any], run_root: Path, workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    return run_simulation_stage(
        config, run_root, workers, progress_callback, phase="baseline",
        dataset_partition=HOLDOUT_PARTITION,
    )


def run_holdout_intervention_simulation_stage(
    config: dict[str, Any], run_root: Path, workers: int,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> dict[str, Any]:
    return run_simulation_stage(
        config, run_root, workers, progress_callback, phase="intervention",
        dataset_partition=HOLDOUT_PARTITION,
    )


def verify_simulation_manifest(
    run_root: Path,
    phase: str = "all",
    *,
    dataset_partition: str = PRIMARY_PARTITION,
) -> None:
    if dataset_partition not in DATA_PARTITIONS:
        raise ValueError(f"unknown simulation dataset partition: {dataset_partition}")
    manifest_path = (
        run_root / "data" / dataset_partition / f"{phase}_simulation_manifest.json"
    )
    if not manifest_path.is_file() and dataset_partition == PRIMARY_PARTITION:
        manifest_path = run_root / "data" / f"{phase}_simulation_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("data_partition", dataset_partition) != dataset_partition:
        raise RuntimeError("simulation manifest belongs to another data partition")
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
    evaluation_track: str = "primary_discovery"
    data_partition: str = PRIMARY_PARTITION
    phase: str = "baseline"
    dose_label: str = "baseline"


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
            # A well-defined event-conditioned observable can become undefined
            # when an intervention or mechanism-disabled variant removes every
            # such event.  Preserve that scientific NaN for Stage 3 to classify
            # as inconclusive; baseline trajectories still fail closed because
            # Stage 2 cannot estimate an entirely undefined observable.
            allow_all_nan=task.condition != "baseline",
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
            "evaluation_track": task.evaluation_track,
            "data_partition": task.data_partition,
            "phase": task.phase,
            "dose_label": task.dose_label,
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
    dataset_partition: str = PRIMARY_PARTITION,
    output_stem: str | None = None,
    progress_callback: Callable[[str, int, int, str], None] | None = None,
) -> pd.DataFrame:
    if dataset_partition not in DATA_PARTITIONS:
        raise ValueError(f"unknown simulation dataset partition: {dataset_partition}")
    if dataset_partition == HOLDOUT_PARTITION:
        _require_primary_freeze(run_root)
    data_root = run_root / "data"
    kind = "complete" if complete else "baseline"
    stem = output_stem or (
        f"indicator_trajectories_{kind}"
        if dataset_partition == PRIMARY_PARTITION
        else f"holdout_indicator_trajectories_{kind}"
    )
    output_path = data_root / f"{stem}.parquet"
    manifest_path = data_root / f"{stem}.manifest.json"
    representation_hash = sha256_json(representations)
    if output_path.exists() != manifest_path.exists():
        raise RuntimeError("saved indicator dataset checkpoint is incomplete")
    if output_path.exists() and manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("data_partition", dataset_partition) != dataset_partition:
            raise RuntimeError("saved indicator dataset belongs to another data partition")
        if manifest.get("representation_sha256") != representation_hash:
            raise RuntimeError("saved indicator dataset belongs to another representation")
        if manifest.get("sha256") != sha256_file(output_path):
            raise RuntimeError("saved indicator dataset hash mismatch")
        return pd.read_parquet(output_path)
    phases = ["baseline", "intervention"] if complete else ["baseline"]
    records: list[dict[str, Any]] = []
    for phase in phases:
        phase_path = (
            data_root / dataset_partition / f"{phase}_simulation_manifest.json"
        )
        if not phase_path.exists():
            primary_alias = data_root / f"{phase}_simulation_manifest.json"
            legacy = data_root / "all_simulation_manifest.json"
            if dataset_partition == PRIMARY_PARTITION and primary_alias.exists():
                phase_path = primary_alias
            if not phase_path.exists() and legacy.exists():
                phase_path = legacy
            elif not phase_path.exists():
                raise FileNotFoundError(f"missing simulation phase manifest: {phase_path}")
        phase_manifest = json.loads(phase_path.read_text(encoding="utf-8"))
        if phase_manifest.get("data_partition", dataset_partition) != dataset_partition:
            raise RuntimeError("indicator compiler received another data partition")
        phase_records = phase_manifest["task_records"]
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
            evaluation_track=item.get("evaluation_track", "primary_discovery"),
            data_partition=item.get("data_partition", dataset_partition),
            phase=item.get(
                "phase", "baseline" if item["condition"] == "baseline" else "intervention"
            ),
            dose_label=item.get("dose_label", item["intervention_direction"]),
        )
        for item in records
    ]

    def update(completed: int, total: int, _result: Any) -> None:
        if progress_callback:
            progress_callback("Indicator compilation", completed, total, kind)

    compiled = _run_jobs(compile_indicator_task, tasks, workers, update)
    flat = [record for task_records in compiled for record in task_records]
    frame = pd.DataFrame(flat).sort_values(
        ["data_partition", "scenario", "condition", "seed", "time"],
        ignore_index=True,
    )
    temporary = output_path.with_suffix(".tmp.parquet")
    frame.to_parquet(temporary, index=False)
    os.replace(temporary, output_path)
    manifest = {
        "schema_version": "1.0",
        "kind": kind,
        "data_partition": dataset_partition,
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
