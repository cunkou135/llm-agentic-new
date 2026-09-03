"""Post-experiment validation of frozen prospective predictions."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .temporal import load_graph_records


def _expected_sign(direction: str) -> int:
    return 1 if direction == "increase" else -1


def classify_prediction_requirements(
    source: Any | None,
    downstream: list[Any | None],
    expected_directions: list[str],
    required_downstream: list[bool],
    *,
    source_required: bool,
    order_required: bool,
    observational_edges_retained: list[bool],
) -> tuple[str, list[bool | None], bool, bool]:
    """Apply the criteria frozen before baseline simulation.

    Missing required responses, wrong directions, temporal-order violations,
    and missing required observational edges are falsifications rather than
    partial support.  ``partially_supported`` is reserved for optional
    evidence and is therefore unreachable under the current all-required
    formal semantic contract.
    """

    if source_required and (source is None or not bool(source.significant)):
        return "manipulation_failure", [], False, False
    items = [source, *downstream]
    direction_matches: list[bool | None] = []
    onsets: list[float] = []
    for item, expected_direction in zip(items, expected_directions):
        if item is None or not bool(item.significant):
            direction_matches.append(None)
            onsets.append(np.nan)
        else:
            direction_matches.append(
                int(item.effect_sign) == _expected_sign(expected_direction)
            )
            onsets.append(float(item.onset_time))
    downstream_supported = all(
        (not required) or (item is not None and bool(item.significant))
        for item, required in zip(downstream, required_downstream)
    )
    finite_order = bool(
        len(onsets) > 0
        and all(np.isfinite(value) and value >= 0 for value in onsets)
        and np.all(np.diff(np.asarray(onsets, dtype=float)) >= 0)
    )
    if not downstream_supported:
        return "contradicted", direction_matches, finite_order, False
    if any(value is False for value in direction_matches):
        return "contradicted", direction_matches, finite_order, True
    if order_required and not finite_order:
        return "contradicted", direction_matches, finite_order, True
    if not all(observational_edges_retained):
        return "contradicted", direction_matches, finite_order, True
    required_direction_indices = [0] + [
        index + 1 for index, required in enumerate(required_downstream) if required
    ]
    if all(direction_matches[index] is True for index in required_direction_indices):
        return "supported", direction_matches, finite_order, True
    if any(value is True for value in direction_matches):
        return "partially_supported", direction_matches, finite_order, True
    return "inconclusive", direction_matches, finite_order, True


def validate_prospective_predictions(
    run_root: Path,
    representations: dict[str, dict[str, Any]],
) -> pd.DataFrame:
    prediction_path = run_root / "representation" / "prospective_predictions.json"
    validation_path = run_root / "representation" / "representation_validation.json"
    validation = json.loads(validation_path.read_text(encoding="utf-8"))
    expected_hash = validation["prospective_predictions_sha256"]
    actual_hash = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    if actual_hash != expected_hash:
        raise RuntimeError("frozen prospective prediction hash mismatch")
    predictions = json.loads(prediction_path.read_text(encoding="utf-8"))["scenarios"]
    path_results = pd.read_csv(
        run_root / "analysis" / "path_intervention_classification.csv"
    )
    rows: list[dict[str, Any]] = []
    for scenario, scenario_predictions in sorted(predictions.items()):
        frozen_paths = {
            str(path["path_id"]): path
            for path in representations[scenario]["candidate_paths"]
        }
        for prediction in scenario_predictions:
            path_id = str(prediction["candidate_path_id"])
            if path_id not in frozen_paths:
                raise RuntimeError(
                    f"prospective prediction references non-frozen path {path_id}"
                )
            selected = path_results[
                (path_results["scenario"].astype(str) == scenario)
                & (path_results["path_id"].astype(str) == path_id)
            ]
            if len(selected) != 1:
                raise RuntimeError(
                    f"prospective path classification is not unique: {scenario}:{path_id}"
                )
            result = selected.iloc[0]
            path = frozen_paths[path_id]
            classification = str(result["path_classification"])
            rows.append(
                {
                    "evaluation_track": "primary_discovery",
                    "scenario": scenario,
                    "prediction_id": prediction["prediction_id"],
                    "candidate_path_id": path_id,
                    "parameter": path["parameter"],
                    "intervention_direction": path["intervention_direction"],
                    "micro": path["micro_indicator"],
                    "meso": path["meso_indicator"],
                    "macro": path["macro_indicator"],
                    "classification": classification,
                    "path_temporally_qualified": bool(
                        result["path_temporally_qualified"]
                    ),
                    "direction_supported": bool(result["direction_supported"]),
                    "onset_order_supported": bool(result["onset_order_supported"]),
                    "falsification_condition": prediction["falsification_condition"],
                    "prediction_sha256": hashlib.sha256(
                        json.dumps(
                            prediction,
                            sort_keys=True,
                            separators=(",", ":"),
                            ensure_ascii=False,
                        ).encode("utf-8")
                    ).hexdigest(),
                }
            )
    frame = pd.DataFrame(rows)
    frame.to_csv(run_root / "analysis" / "prospective_validation.csv", index=False)
    return frame
