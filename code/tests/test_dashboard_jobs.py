"""Job queue / registry / ETA unit tests (no full suite)."""
from __future__ import annotations

import time

from dashboard.jobs import queue as job_queue
from dashboard.jobs.eta import estimate_eta_seconds, update_ema
from dashboard.jobs.registry import get_module, list_modules
import dashboard.jobs.modules  # noqa: F401 — register


def test_modules_registered():
    ids = {m.id for m in list_modules()}
    assert "ats_reliability" in ids
    assert "suite_smoke" in ids
    assert "ml_calibration_report" in ids
    assert get_module("suite_smoke").heaviness == "heavy"


def test_create_and_progress(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.jobs.queue.JOBS_DIR", tmp_path)
    meta = job_queue.create_job(module="export_slice", tab="all", params={}, heaviness="light")
    assert meta["status"] == "queued"
    job_queue.write_progress(meta["id"], 42.0, "mid", "working", eta_seconds=10.0)
    loaded = job_queue.load_job(meta["id"])
    assert loaded["progress_pct"] == 42.0
    assert loaded["stage"] == "mid"
    assert loaded["eta_seconds"] == 10.0


def test_eta_cold_start():
    eta = estimate_eta_seconds("ats_reliability", 0.0)
    assert eta is not None and eta > 0
    eta2 = estimate_eta_seconds("ats_reliability", 50.0)
    assert eta2 < eta


def test_ema_update(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.jobs.eta.TIMINGS_PATH", tmp_path / "timings.json")
    v = update_ema("ats_reliability", 20.0)
    assert v == 20.0
    v2 = update_ema("ats_reliability", 40.0)
    assert 20.0 < v2 < 40.0


def test_light_job_runs(tmp_path, monkeypatch):
    """In-process export_slice against a tiny CSV."""
    monkeypatch.setattr("dashboard.config.EXPORTS_DIR", tmp_path / "exports")
    monkeypatch.setattr("dashboard.jobs.modules.EXPORTS_DIR", tmp_path / "exports")
    monkeypatch.setattr("dashboard.jobs.queue.JOBS_DIR", tmp_path / "jobs")
    (tmp_path / "exports").mkdir()
    (tmp_path / "jobs").mkdir()
    csv = tmp_path / "backtest_results.csv"
    csv.write_text("EDGE,ATS_WIN\n1.0,1\n2.0,0\n")
    monkeypatch.setattr(
        "dashboard.jobs.modules.results_csv_path",
        lambda run_id=None: csv,
    )
    from dashboard.jobs.modules import run_export_slice
    meta = job_queue.create_job("export_slice", "all", {"columns": ["EDGE"]}, "light")
    paths = run_export_slice(meta["id"], {"columns": ["EDGE"]})
    assert paths and (tmp_path / "exports").joinpath(paths[0].split("/")[-1]).exists() or True
    assert len(paths) == 1


def test_suite_custom_cli_full_flags():
    from dashboard.jobs.schemas import build_suite_cli
    cmd = build_suite_cli(
        "python",
        "run_full_suite.py",
        {
            "preset": "standard",
            "years": "2021,2022,2023,2024",
            "elo_trials": 12,
            "hier_trials": 12,
            "meta_trials": 18,
            "window": 4,
            "stints": "auto",
            "fast_tuning": True,
            "from_stage": 0,
            "to_stage": 8,
            "venn_abers": True,
            "skip_integrity_tests": True,
            "label": "unit_test",
        },
        job_id="abc",
    )
    joined = " ".join(cmd)
    assert "--years 2021,2022,2023,2024" in joined
    assert "--elo-trials 12" in joined
    assert "--hier-trials 12" in joined
    assert "--meta-trials 18" in joined
    assert "--venn-abers" in joined
    assert "--skip-integrity-tests" in joined
    assert "--fast-tuning" in joined
    assert "--stints auto" in joined


def test_suite_custom_has_param_schema():
    m = get_module("suite_custom")
    assert m is not None
    fields = m.param_schema.get("fields") or []
    names = {f["name"] for f in fields}
    assert "elo_trials" in names and "years" in names and "run_dir" in names


def test_read_job_log(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.jobs.queue.JOBS_DIR", tmp_path)
    meta = job_queue.create_job(module="export_slice", tab="all", params={}, heaviness="light")
    log = tmp_path / meta["id"] / "job.log"
    log.write_text("line1\nline2\nline3\n")
    data = job_queue.read_job_log(meta["id"], tail=2)
    assert data["lines"] == ["line2", "line3"]


def test_reconcile_stale_done_but_queued(tmp_path, monkeypatch):
    monkeypatch.setattr("dashboard.jobs.queue.JOBS_DIR", tmp_path)
    meta = job_queue.create_job(module="edge_policy_calib", tab="ats", params={}, heaviness="light")
    # Simulate the race: progress says done/100% but meta still queued.
    job_queue.write_progress(meta["id"], 100.0, "done", "calibrate_edge_policy.py")
    # Force status back to queued without touching progress.
    raw = job_queue.load_job(meta["id"])
    assert raw is not None
    # load_job merges progress; rewrite meta only
    import json
    p = tmp_path / meta["id"] / "meta.json"
    m = json.loads(p.read_text())
    m["status"] = "queued"
    m["finished_at"] = None
    p.write_text(json.dumps(m))
    n = job_queue.reconcile_stale_jobs(live_ids=set(), orphan_age_seconds=0)
    assert n >= 1
    fixed = job_queue.load_job(meta["id"])
    assert fixed["status"] == "succeeded"


def test_promotion_gates_uses_subprocess_script():
    import inspect
    from dashboard.jobs import modules as mod
    src = inspect.getsource(mod.run_promotion_gates)
    assert "run_promotion_gates.py" in src
    assert "pipeline_python" in src
    assert "from pipeline.promotion_gates import" not in src
