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
    effects = pd.read_parquet(run_root / "analysis" / "paired_effects.parquet")
    graphs = load_graph_records(run_root / "analysis" / "main_graphs.jsonl")
    rows: list[dict[str, Any]] = []
    for scenario, scenario_predictions in sorted(predictions.items()):
        graph_pairs = {
            (edge.source, edge.target)
            for edge in graphs[(scenario, "full_method")]
        }
        for prediction in scenario_predictions:
            criteria = prediction["validation_criteria"]
            ordered = prediction["expected_temporal_order"]
            expected_edges = [
                (item["source"], item["target"])
                for item in criteria["required_candidate_edges"]
            ]
            observational_edges_retained = [pair in graph_pairs for pair in expected_edges]
            indicators = [prediction["source_indicator"], *prediction["downstream_indicators"]]
            expected_directions = [
                prediction["expected_source_direction"],
                *prediction["expected_downstream_direction"],
            ]
            selected = effects[
                (effects["scenario"] == scenario)
                & (effects["parameter"] == prediction["parameter"])
                & (effects["direction"] == prediction["intervention_direction"])
                & (effects["node_id"].isin(indicators))
            ]
            lookup = {row.node_id: row for row in selected.itertuples()}
            source = lookup.get(prediction["source_indicator"])
            source_required = bool(criteria["required_source_response"])
            downstream_required = list(criteria["required_downstream_response"])
            order_required = bool(criteria["required_temporal_order"])
            if source_required and (source is None or not bool(source.significant)):
                classification = "manipulation_failure"
                direction_matches: list[bool | None] = []
                onset_order_supported = False
            else:
                direction_matches = []
                significant_count = 0
                onsets = []
                for indicator, expected_direction in zip(indicators, expected_directions):
                    item = lookup.get(indicator)
                    if item is None or not bool(item.significant):
                        direction_matches.append(None)
                        onsets.append(np.nan)
                    else:
                        significant_count += 1
                        direction_matches.append(
                            int(item.effect_sign) == _expected_sign(expected_direction)
                        )
                        onsets.append(float(item.onset_time))
                finite_onsets = [value for value in onsets if np.isfinite(value) and value >= 0]
                onset_order_supported = bool(
                    len(finite_onsets) == len(onsets)
                    and np.all(np.diff(np.asarray(onsets)) >= 0)
                )
                downstream_responses_supported = all(
                    (not required)
                    or (
                        lookup.get(indicator) is not None
                        and bool(lookup[indicator].significant)
                    )
                    for indicator, required in zip(
                        prediction["downstream_indicators"], downstream_required
                    )
                )
                if any(value is False for value in direction_matches) or (
                    significant_count == len(indicators)
                    and order_required
                    and not onset_order_supported
                ):
                    classification = "contradicted"
                elif (
                    all(value is True for value in direction_matches)
                    and downstream_responses_supported
                    and (onset_order_supported or not order_required)
                    and all(observational_edges_retained)
                ):
                    classification = "supported"
                elif significant_count > 1 or any(observational_edges_retained):
                    classification = "partially_supported"
                else:
                    classification = "inconclusive"
            rows.append(
                {
                    "scenario": scenario,
                    "prediction_id": prediction["prediction_id"],
                    "phenomenon": prediction["phenomenon"],
                    "parameter": prediction["parameter"],
                    "intervention_direction": prediction["intervention_direction"],
                    "source_indicator": prediction["source_indicator"],
                    "downstream_indicators": json.dumps(
                        prediction["downstream_indicators"], ensure_ascii=False
                    ),
                    "classification": classification,
                    "observational_edges_retained": json.dumps(
                        observational_edges_retained
                    ),
                    "direction_matches": json.dumps(direction_matches),
                    "onset_order_supported": onset_order_supported,
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
