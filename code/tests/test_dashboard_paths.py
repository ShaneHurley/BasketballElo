"""Path sandbox and run listing tests."""
from __future__ import annotations

import pytest

from dashboard.config import OUTPUT_ROOT
from dashboard.paths import PathSandboxError, list_runs, results_csv_path, safe_resolve


def test_safe_resolve_allows_output(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.paths.OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr("dashboard.paths.ALLOWED_ROOTS", (tmp_path.resolve(),))
    f = tmp_path / "runA" / "x.txt"
    f.parent.mkdir()
    f.write_text("ok")
    assert safe_resolve(f) == f.resolve()


def test_safe_resolve_blocks_escape(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.paths.ALLOWED_ROOTS", (tmp_path.resolve(),))
    with pytest.raises(PathSandboxError):
        safe_resolve("/etc/passwd")


def test_run_id_rejects_traversal():
    from dashboard.paths import run_dir_for_id
    with pytest.raises(PathSandboxError):
        run_dir_for_id("../secret")


def test_list_runs_empty(monkeypatch, tmp_path):
    monkeypatch.setattr("dashboard.paths.OUTPUT_ROOT", tmp_path)
    assert list_runs() == []


def test_results_csv_fallback_to_latest_run(monkeypatch, tmp_path):
    run = tmp_path / "run_new"
    (run / "betting").mkdir(parents=True)
    csv = run / "betting" / "backtest_results.csv"
    csv.write_text("A\n1\n")
    monkeypatch.setattr("dashboard.paths.OUTPUT_ROOT", tmp_path)
    monkeypatch.setattr("dashboard.paths.CODE_ROOT", tmp_path / "code")
    monkeypatch.setattr(
        "dashboard.paths.ALLOWED_ROOTS",
        (tmp_path.resolve(),),
    )
    from dashboard.paths import results_csv_path
    assert results_csv_path(None) == csv
    assert results_csv_path("run_new") == csv
