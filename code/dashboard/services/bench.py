"""Read the Epic 10 formula-bench summary written by pytest_sessionfinish."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from dashboard.config import OUTPUT_ROOT, REPO_ROOT

BENCH_REPORT_PATH = OUTPUT_ROOT / "bench" / "latest.json"


def _rel_or_abs(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path.resolve())


def load_bench_summary(path: Path | None = None) -> dict[str, Any]:
    """Return the latest bench report, or an empty-state dict if none exists."""
    report_path = path if path is not None else BENCH_REPORT_PATH
    source = _rel_or_abs(report_path)
    empty = {
        "available": False,
        "message": "no bench report yet",
        "totals": {},
        "files": {},
        "source_path": source,
        "passed": None,
        "failed": None,
        "xfailed": None,
        "skipped": None,
        "exitstatus": None,
    }
    if not report_path.is_file():
        return empty
    try:
        data = json.loads(report_path.read_text())
    except (OSError, json.JSONDecodeError):
        return empty
    totals = data.get("totals") or {}
    return {
        "available": True,
        "message": None,
        "source_path": source,
        "exitstatus": data.get("exitstatus"),
        "totals": totals,
        "files": data.get("files") or {},
        "passed": int(totals.get("passed", 0)),
        "failed": int(totals.get("failed", 0)),
        "xfailed": int(totals.get("xfailed", 0)),
        "skipped": int(totals.get("skipped", 0)),
    }
