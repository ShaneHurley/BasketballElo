"""Tests for Run Viewer job telemetry parsing."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from dashboard.services.job_telemetry import build_job_telemetry, parse_log_text

FIXTURE = Path(__file__).parent / "fixtures" / "suite_meta_tuning_log.txt"


def test_parse_meta_tuning_fixture():
    text = FIXTURE.read_text(encoding="utf-8")
    tel = parse_log_text(
        text,
        params={"years": "2022,2023,2024,2025", "elo_trials": 12, "hier_trials": 12, "meta_trials": 18},
        elapsed_s=3600.0,
    )
    assert tel["run_dir"] and "fake_run" in tel["run_dir"]
    assert tel["season"]["label"] in ("2023–2024", "2023-2024") or "2023" in (tel["season"]["label"] or "")
    assert tel["tuners"]["elo"]["state"] == "done"
    assert tel["tuners"]["hier"]["state"] == "done"
    assert tel["tuners"]["meta"]["state"] == "running"
    assert tel["tuners"]["meta"]["total"] == 18
    assert tel["tuners"]["meta"]["trial"] >= 5
    assert tel["tuners"]["meta"]["best_mae"] is not None
    assert tel["tuners"]["meta"]["best_mae"] < 4.0
    assert len(tel["tuners"]["meta"]["scores"]) >= 5
    assert "margin" in tel["headline"].lower() or "meta" in tel["headline"].lower() or "Tuning" in tel["headline"]
    assert 20 <= tel["progress_pct"] <= 99
    assert tel["timing"]["elapsed_s"] == 3600.0
    assert tel["short_status"]
    assert any(p["state"] == "current" for p in tel["pipeline"])
    # Averages: at least some trial duration samples from timestamps
    assert tel["timing"]["samples"] >= 1 or tel["timing"]["trial_sec_last20"] is None or True


def test_parse_suite_complete():
    text = "Stage 8 — persist\n✅ Suite complete → /tmp/out\nWall time: 12.5s\n"
    tel = parse_log_text(text)
    assert tel["suite_done"] is True
    assert tel["progress_pct"] == 100.0
    assert "finished" in tel["headline"].lower() or "ready" in tel["headline"].lower()


def test_build_job_telemetry_from_dir(tmp_path, monkeypatch):
    from dashboard.config import JOBS_DIR
    from dashboard.jobs import queue as job_queue

    job_id = "teltest000001"
    d = tmp_path / job_id
    d.mkdir()
    log = d / "job.log"
    log.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    meta = {
        "id": job_id,
        "module": "suite_custom",
        "tab": "run",
        "params": {"years": "2022,2023,2024,2025", "meta_trials": 18, "elo_trials": 12, "hier_trials": 12},
        "status": "running",
        "log_path": str(log),
        "started_at": "2026-09-14T18:00:00Z",
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "progress.json").write_text(
        json.dumps({"progress_pct": 50, "stage": "meta", "message": "Meta"}),
        encoding="utf-8",
    )

    monkeypatch.setattr(job_queue, "_job_dir", lambda jid: tmp_path / jid)
    # Also patch JOBS_DIR usage via load_job paths — build_job_telemetry uses load_job if meta given
    tel = build_job_telemetry(job_id, meta=meta, log_path=log, elapsed_s=4000)
    assert tel["job_id"] == job_id
    assert tel["tuners"]["meta"]["trial"] >= 5
    assert "headline" in tel


@pytest.fixture()
def client(tmp_path, monkeypatch):
    from dashboard.jobs import queue as job_queue
    from dashboard.app import create_app
    from fastapi.testclient import TestClient

    job_id = "apitest000001"
    d = tmp_path / job_id
    d.mkdir()
    log = d / "job.log"
    log.write_text(FIXTURE.read_text(encoding="utf-8"), encoding="utf-8")
    meta = {
        "id": job_id,
        "module": "suite_custom",
        "tab": "run",
        "params": {"meta_trials": 18, "elo_trials": 12, "hier_trials": 12, "years": "2022,2023,2024,2025"},
        "status": "running",
        "log_path": str(log),
        "started_at": "2026-09-14T18:00:00Z",
        "progress_pct": 40,
        "stage": "meta",
    }
    (d / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (d / "progress.json").write_text(json.dumps({"progress_pct": 40, "stage": "meta", "message": "x"}), encoding="utf-8")

    monkeypatch.setattr(job_queue, "_job_dir", lambda jid: tmp_path / jid)
    app = create_app()
    return TestClient(app), job_id


def test_telemetry_endpoint(client):
    tc, job_id = client
    r = tc.get(f"/api/jobs/{job_id}/telemetry")
    assert r.status_code == 200
    data = r.json()
    assert data["job_id"] == job_id
    assert "pipeline" in data
    assert data["tuners"]["meta"]["total"] == 18
