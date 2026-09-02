from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from emergence_attribution.simulation import (
    HOLDOUT_PARTITION,
    PRIMARY_PARTITION,
    build_simulation_tasks,
    run_holdout_baseline_simulation_stage,
)
from emergence_attribution.simulators import run_scenario


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _config(name: str = "experiment.json") -> dict:
    return json.loads((PROJECT_ROOT / "config" / name).read_text(encoding="utf-8"))


def _tasks(config: dict, tmp_path: Path, partition: str, phase: str = "all"):
    partition_root = tmp_path / "data" / partition
    return build_simulation_tasks(
        config,
        partition_root / "raw_logs",
        partition_root / "reference_hidden",
        phase=phase,
        dataset_partition=partition,
    )


def test_formal_seed_pools_are_frozen_and_disjoint() -> None:
    config = _config()
    assert config["random_seeds"] == list(range(1101, 1125))
    assert config["confirmation_seeds"] == list(range(2101, 2113))
    assert not (set(config["random_seeds"]) & set(config["confirmation_seeds"]))


def test_final_formal_task_matrix_has_exact_864_trajectories(tmp_path: Path) -> None:
    config = _config()
    primary_baseline = _tasks(config, tmp_path, PRIMARY_PARTITION, "baseline")
    primary_intervention = _tasks(config, tmp_path, PRIMARY_PARTITION, "intervention")
    holdout = _tasks(config, tmp_path, HOLDOUT_PARTITION)

    assert len(primary_baseline) == 48
    assert len(primary_intervention) == 624
    assert len(primary_baseline) + len(primary_intervention) == 672
    assert len(holdout) == 192
    assert len(primary_baseline) + len(primary_intervention) + len(holdout) == 864
    assert len({task.task_id for task in [*primary_baseline, *primary_intervention, *holdout]}) == 864


def test_primary_has_13_interventions_and_holdout_has_eight_total_conditions_per_scenario(
    tmp_path: Path,
) -> None:
    config = _config()
    primary = _tasks(config, tmp_path, PRIMARY_PARTITION)
    holdout = _tasks(config, tmp_path, HOLDOUT_PARTITION)
    for scenario in config["scenarios"]:
        primary_conditions = {
            task.condition for task in primary
            if task.scenario == scenario and task.phase == "intervention"
        }
        holdout_conditions = {
            task.condition for task in holdout if task.scenario == scenario
        }
        assert len(primary_conditions) == 13
        assert len(holdout_conditions) == 8
        assert "baseline" not in primary_conditions
        assert "baseline" in holdout_conditions
        assert not any("mid_" in condition for condition in holdout_conditions)


def test_five_point_doses_are_in_range_and_primary_extremes_are_unchanged(
    tmp_path: Path,
) -> None:
    config = _config()
    tasks = _tasks(config, tmp_path, PRIMARY_PARTITION)
    seed = config["random_seeds"][0]
    for scenario, spec in config["scenarios"].items():
        for parameter, configured in spec["interventions"].items():
            selected = {
                task.dose_label: task.parameters[parameter]
                for task in tasks
                if task.scenario == scenario
                and task.seed == seed
                and task.intervention_parameter == parameter
            }
            assert selected == {
                "minus": pytest.approx(configured[0]),
                "mid_minus": pytest.approx((configured[0] + configured[1]) / 2.0),
                "mid_plus": pytest.approx((configured[1] + configured[2]) / 2.0),
                "plus": pytest.approx(configured[2]),
            }
            assert selected["minus"] == configured[0]
            assert selected["plus"] == configured[2]


def test_task_tracks_prevent_mid_doses_and_controls_from_becoming_primary_support(
    tmp_path: Path,
) -> None:
    tasks = _tasks(_config(), tmp_path, PRIMARY_PARTITION)
    for task in tasks:
        assert task.data_partition == PRIMARY_PARTITION
        assert task.phase in {"baseline", "intervention"}
        if task.dose_label in {"mid_minus", "mid_plus"}:
            assert task.evaluation_track == "dose_response"
        elif task.condition == "mechanism_disabled":
            assert task.evaluation_track == "falsification_control"
        else:
            assert task.evaluation_track == "primary_discovery"


def test_holdout_tasks_are_physically_and_semantically_isolated(tmp_path: Path) -> None:
    config = _config()
    primary = _tasks(config, tmp_path, PRIMARY_PARTITION)
    holdout = _tasks(config, tmp_path, HOLDOUT_PARTITION)
    assert {task.seed for task in primary} == set(config["random_seeds"])
    assert {task.seed for task in holdout} == set(config["confirmation_seeds"])
    assert not ({task.seed for task in primary} & {task.seed for task in holdout})
    assert all(task.evaluation_track == "holdout_confirmation" for task in holdout)
    assert all("data\\holdout" in task.raw_path or "data/holdout" in task.raw_path for task in holdout)
    assert all("data\\primary" in task.raw_path or "data/primary" in task.raw_path for task in primary)


def test_overlapping_holdout_seed_pool_is_rejected(tmp_path: Path) -> None:
    config = _config("dev_experiment.json")
    config["confirmation_seeds"] = [config["random_seeds"][0]]
    with pytest.raises(ValueError, match="seed pools overlap"):
        _tasks(config, tmp_path, HOLDOUT_PARTITION)


def test_holdout_execution_is_locked_before_primary_prospective_freeze(
    tmp_path: Path,
) -> None:
    with pytest.raises(RuntimeError, match="holdout access is forbidden"):
        run_holdout_baseline_simulation_stage(
            _config("dev_experiment.json"), tmp_path / "run", workers=1
        )
    assert not (tmp_path / "run" / "data" / "holdout").exists()


@pytest.mark.parametrize(
    ("scenario", "initial_fields"),
    [
        ("schelling", ("state_grid", "agent_group", "agent_position", "district_id")),
        ("deffuant", ("state_opinion", "network_edges")),
    ],
)
def test_all_primary_doses_share_matched_seed_initial_state(
    tmp_path: Path,
    scenario: str,
    initial_fields: tuple[str, ...],
) -> None:
    config = _config("dev_experiment.json")
    seed = config["random_seeds"][0]
    tasks = _tasks(config, tmp_path, PRIMARY_PARTITION)
    baseline = next(
        task for task in tasks
        if task.scenario == scenario and task.seed == seed and task.condition == "baseline"
    )
    parameter = sorted(config["scenarios"][scenario]["interventions"])[0]
    matched = [baseline] + [
        task for task in tasks
        if task.scenario == scenario
        and task.seed == seed
        and task.intervention_parameter == parameter
    ]
    assert {task.dose_label for task in matched} == {
        "baseline", "minus", "mid_minus", "mid_plus", "plus"
    }
    payloads = [
        run_scenario(
            task.scenario,
            task.seed,
            task.scenario_spec,
            task.parameters,
            task.mechanism_variant,
        )
        for task in matched
    ]
    for field in initial_fields:
        reference = payloads[0][field]
        reference = reference[0] if reference.ndim > 1 and field != "agent_group" else reference
        for payload in payloads[1:]:
            actual = payload[field]
            actual = actual[0] if actual.ndim > 1 and field != "agent_group" else actual
            np.testing.assert_array_equal(actual, reference)


def test_same_seed_and_condition_is_bitwise_deterministic(tmp_path: Path) -> None:
    config = _config("dev_experiment.json")
    task = next(
        item for item in _tasks(config, tmp_path, PRIMARY_PARTITION)
        if item.scenario == "deffuant" and item.condition == "confidence_bound_mid_plus"
    )
    first = run_scenario(
        task.scenario, task.seed, task.scenario_spec, task.parameters, task.mechanism_variant
    )
    second = run_scenario(
        task.scenario, task.seed, task.scenario_spec, task.parameters, task.mechanism_variant
    )
    assert set(first) == set(second)
    for name in first:
        np.testing.assert_array_equal(first[name], second[name])


def test_formal_stage_two_and_three_thresholds_remain_frozen() -> None:
    config = _config()
    assert config["temporal"] == {
        "maximum_lag": 5,
        "parent_alpha": 0.10,
        "fdr_alpha": 0.05,
        "bootstrap_repetitions": 100,
        "support_threshold": 0.65,
        "vote_threshold": 0.50,
    }
    intervention = config["intervention"]
    assert intervention["bootstrap_repetitions"] == 500
    assert intervention["confidence_level"] == 0.95
    assert intervention["onset_detection_start"] == 0
    assert intervention["minimum_standardised_effect"] == 0.10
    assert intervention["onset_consecutive_steps"] == 4
    assert intervention["evaluation_start"] == 15
    assert intervention["terminal_window"] == 24
    assert intervention["lag_tolerance"] == 2
