"""Tests for pipeline_python probing."""
from __future__ import annotations

import sys
from pathlib import Path

from dashboard.jobs.util import (
    PIPELINE_REQUIREMENTS_FILE,
    ensure_pipeline_ready,
    install_hint,
    pipeline_python,
    pipeline_requirement_imports,
    probe_python,
)


def test_requirements_pipeline_file_exists():
    assert PIPELINE_REQUIREMENTS_FILE.exists()


def test_pipeline_requirement_imports_includes_catboost():
    required, optional = pipeline_requirement_imports()
    assert "catboost" in required
    assert "matplotlib" in required
    assert "lightgbm" in optional


def test_probe_python_current():
    info = probe_python(sys.executable)
    assert "python" in info
    assert isinstance(info["present"], list)
    assert isinstance(info["missing_required"], list)
    assert "ready" in info
    assert info["score"] >= 0


def test_pipeline_python_returns_path():
    log: list[str] = []
    py = pipeline_python(log=log)
    assert py
    assert log, "expected probe log line"
    assert "pipeline_python" in log[0]


def test_install_hint_mentions_requirements():
    hint = install_hint(sys.executable)
    assert "requirements-pipeline.txt" in hint
    assert "-m pip install" in hint


def test_ensure_pipeline_ready_or_skip():
    info = probe_python(sys.executable)
    if info["ready"]:
        ensure_pipeline_ready(sys.executable)
    else:
        try:
            ensure_pipeline_ready(sys.executable)
            raise AssertionError("expected RuntimeError for incomplete python")
        except RuntimeError as e:
            assert "missing required" in str(e).lower() or "Install with" in str(e)
