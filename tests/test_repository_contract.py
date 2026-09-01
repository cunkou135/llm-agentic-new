from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_forbidden_historical_name_absent_case_insensitively() -> None:
    forbidden = "ca" + "mo"
    excluded = {".git", ".venv", "__pycache__", ".pytest_cache", "runs", "smoke_runs"}
    matches = []
    for path in PROJECT_ROOT.rglob("*"):
        if any(part in excluded for part in path.relative_to(PROJECT_ROOT).parts):
            continue
        if forbidden in path.name.lower():
            matches.append(str(path.relative_to(PROJECT_ROOT)))
            continue
        if path.is_file():
            try:
                text = path.read_text(encoding="utf-8")
            except UnicodeDecodeError:
                continue
            if forbidden in text.lower():
                matches.append(str(path.relative_to(PROJECT_ROOT)))
    assert matches == []


def test_reference_module_not_imported_by_inference_stages() -> None:
    forbidden_import = "reference_" + "truth"
    for relative in (
        "src/emergence_attribution/semantic.py",
        "src/emergence_attribution/temporal.py",
        "src/emergence_attribution/interventions.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert forbidden_import not in text

