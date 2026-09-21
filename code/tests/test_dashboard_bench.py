"""Dashboard formula-bench empty-state service."""
from __future__ import annotations

import json
from pathlib import Path

from dashboard.services.bench import load_bench_summary


def test_load_bench_summary_empty_state(tmp_path: Path):
    missing = tmp_path / "latest.json"
    out = load_bench_summary(missing)
    assert out["available"] is False
    assert out["message"] == "no bench report yet"
    assert out["source_path"]
    assert out["passed"] is None


def test_load_bench_summary_corrupt_json(tmp_path: Path):
    p = tmp_path / "latest.json"
    p.write_text("{not-json")
    out = load_bench_summary(p)
    assert out["available"] is False
    assert out["message"] == "no bench report yet"
    assert "passed" not in out or out["passed"] is None


def test_load_bench_summary_reads_file(tmp_path: Path):
    p = tmp_path / "latest.json"
    p.write_text(json.dumps({
        "exitstatus": 0,
        "totals": {"passed": 3, "failed": 0, "xfailed": 1, "skipped": 2},
        "files": {"tests/test_bench_x.py": {"passed": 3, "failed": 0, "xfailed": 1}},
    }))
    out = load_bench_summary(p)
    assert out["available"] is True
    assert out["passed"] == 3
    assert out["xfailed"] == 1
    assert out["skipped"] == 2
    assert out["source_path"]
    assert out["files"]["tests/test_bench_x.py"]["passed"] == 3
