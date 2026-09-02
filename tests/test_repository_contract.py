from __future__ import annotations

from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_reference_module_not_imported_by_inference_stages() -> None:
    forbidden_import = "reference_" + "truth"
    for relative in (
        "src/emergence_attribution/semantic.py",
        "src/emergence_attribution/temporal.py",
        "src/emergence_attribution/interventions.py",
    ):
        text = (PROJECT_ROOT / relative).read_text(encoding="utf-8")
        assert forbidden_import not in text
