"""Compact live terminal progress with durable JSONL events."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class ProgressReporter:
    STAGES = [
        "semantic",
        "baseline_simulation",
        "temporal",
        "intervention_simulation",
        "intervention",
        "prospective",
        "dose_response",
        "holdout_simulation",
        "holdout_confirmation",
        "temporal_negative_control",
        "robustness",
        "export",
        "render",
    ]

    def __init__(self, run_root: Path, workers: int):
        self.run_root = run_root
        self.workers = workers
        self.log_path = run_root / "logs" / "progress.jsonl"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TextColumn("{task.completed:.0f}/{task.total:.0f}"),
            TimeElapsedColumn(),
            TimeRemainingColumn(),
            TextColumn(f"workers={workers}"),
            console=Console(),
            refresh_per_second=6,
        )
        self.overall_task = self.progress.add_task(
            "Overall experiment", total=len(self.STAGES)
        )
        self.stage_task: int | None = None
        self.stage_name = ""
        self.stage_total = 1

    def __enter__(self) -> "ProgressReporter":
        self.progress.start()
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.progress.stop()

    def _write(self, event: dict[str, Any]) -> None:
        event = {"time": time.time(), "workers": self.workers, **event}
        with self.log_path.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(event, ensure_ascii=False) + "\n")

    def start_stage(self, stage: str, total: int = 1) -> None:
        if self.stage_task is not None:
            self.progress.remove_task(self.stage_task)
        self.stage_name = stage
        self.stage_total = max(total, 1)
        self.stage_task = self.progress.add_task(stage.title(), total=max(total, 1))
        self._write({"event": "stage_started", "stage": stage, "total": total})

    def update(self, label: str, completed: int, total: int, detail: str = "") -> None:
        if self.stage_task is None:
            return
        description = f"{self.stage_name.title()} - {label}"
        if detail:
            description += f" [{detail}]"
        self.progress.update(
            self.stage_task,
            description=description,
            total=max(total, 1),
            completed=completed,
        )
        self.stage_total = max(total, 1)
        self._write(
            {
                "event": "progress",
                "stage": self.stage_name,
                "label": label,
                "completed": completed,
                "total": total,
                "percentage": 100.0 * completed / max(total, 1),
                "detail": detail,
            }
        )

    def finish_stage(self, stage: str, skipped: bool = False) -> None:
        if self.stage_task is not None:
            self.progress.update(
                self.stage_task, total=self.stage_total, completed=self.stage_total
            )
        self.progress.advance(self.overall_task, 1)
        self._write(
            {
                "event": "stage_skipped" if skipped else "stage_completed",
                "stage": stage,
            }
        )
