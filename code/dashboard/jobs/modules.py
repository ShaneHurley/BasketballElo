"""In-process and subprocess job runners registered for the dashboard."""
from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard.config import CODE_ROOT, EXPORTS_DIR, HEAVY_MODULES, JOBS_DIR
from dashboard.jobs import queue as job_queue
from dashboard.jobs.eta import estimate_eta_seconds, now_iso, update_ema
from dashboard.jobs.registry import register
from dashboard.jobs.schemas import (
    BACKTEST_SCHEMA,
    SUITE_CUSTOM_SCHEMA,
    SUITE_SMOKE_SCHEMA,
    build_backtest_cli,
    build_suite_cli,
)
from dashboard.jobs.util import ensure_ats_win, ensure_pipeline_ready, pipeline_python
from dashboard.paths import results_csv_path, run_dir_for_id, safe_resolve


def _load_results(run_id: str | None) -> pd.DataFrame:
    path = results_csv_path(run_id)
    if path is None or not path.exists():
        hint = f"run_id={run_id!r}" if run_id else "no run_id (and no latest results under output/)"
        raise FileNotFoundError(f"No backtest_results.csv found ({hint})")
    return ensure_ats_win(pd.read_csv(path))


def _append_job_log(job_id: str, line: str) -> None:
    """Append a timestamped line to the job's log file (light + heavy)."""
    meta = job_queue.load_job(job_id)
    log_path = Path(meta["log_path"]) if meta and meta.get("log_path") else JOBS_DIR / job_id / "job.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    ts = now_iso()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{ts}] {line.rstrip()}\n")


def _progress(job_id: str, pct: float, stage: str, module: str, msg: str = "") -> None:
    eta = estimate_eta_seconds(module, pct, stage)
    text = msg or stage
    job_queue.write_progress(job_id, pct, stage, text, eta_seconds=eta)
    _append_job_log(job_id, f"{module} {pct:.0f}% stage={stage} {text}")


@register("ats_reliability", tab="ats", label="ATS reliability / CLV report", heaviness="light")
def run_ats_reliability(job_id: str, params: dict[str, Any]) -> list[str]:
    _progress(job_id, 5, "load", "ats_reliability", "loading results")
    df = _load_results(params.get("run_id"))
    _progress(job_id, 40, "compute", "ats_reliability", "computing buckets")
    out_rows = []
    season_col = "simulated_season_window" if "simulated_season_window" in df.columns else None
    cover = df.get("CALIBRATED_COVER_PROB", df.get("COVER_PROB_CALIBRATED"))
    ats_win = df.get("ATS_WIN")
    if cover is not None and ats_win is not None:
        tmp = pd.DataFrame({"p": pd.to_numeric(cover, errors="coerce"),
                            "y": pd.to_numeric(ats_win, errors="coerce")}).dropna()
        if len(tmp):
            tmp["bucket"] = pd.cut(tmp["p"], bins=np.linspace(0, 1, 11), include_lowest=True)
            g = tmp.groupby("bucket", observed=False).agg(n=("y", "size"), hit=("y", "mean"), mean_p=("p", "mean"))
            for idx, row in g.iterrows():
                out_rows.append({"kind": "reliability", "bucket": str(idx), **row.to_dict()})
    if season_col and "ATS_WIN" in df.columns:
        for season, g in df.groupby(season_col):
            y = pd.to_numeric(g["ATS_WIN"], errors="coerce")
            out_rows.append({
                "kind": "season", "season": season, "n": int(y.notna().sum()),
                "ats_hit": float(y.mean()) if y.notna().any() else None,
            })
    _progress(job_id, 80, "write", "ats_reliability")
    out_path = EXPORTS_DIR / f"ats_reliability_{job_id}.csv"
    pd.DataFrame(out_rows).to_csv(out_path, index=False)
    _progress(job_id, 100, "done", "ats_reliability")
    return [str(out_path)]


@register("ml_calibration_report", tab="ml", label="ML Brier / logloss report", heaviness="light")
def run_ml_calibration_report(job_id: str, params: dict[str, Any]) -> list[str]:
    _progress(job_id, 10, "load", "ml_calibration_report")
    df = _load_results(params.get("run_id"))
    wp = pd.to_numeric(df.get("WIN_PROB", df.get("P_HOME")), errors="coerce")
    act = None
    if "ACTUAL_HOME" in df.columns and "ACTUAL_AWAY" in df.columns:
        act = (pd.to_numeric(df["ACTUAL_HOME"], errors="coerce")
               > pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")).astype(float)
    rows = []
    if wp is not None and act is not None:
        mask = wp.notna() & act.notna()
        p = wp[mask].clip(1e-6, 1 - 1e-6).to_numpy()
        y = act[mask].to_numpy()
        brier = float(np.mean((p - y) ** 2))
        logloss = float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))
        rows.append({"metric": "brier", "value": brier, "n": int(mask.sum())})
        rows.append({"metric": "logloss", "value": logloss, "n": int(mask.sum())})
        # Simple ECE
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for i in range(len(bins) - 1):
            m = (p >= bins[i]) & (p < bins[i + 1] if i < len(bins) - 2 else p <= bins[i + 1])
            if m.sum() == 0:
                continue
            ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
        rows.append({"metric": "ece", "value": float(ece), "n": int(mask.sum())})
    out_path = EXPORTS_DIR / f"ml_calibration_{job_id}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    _progress(job_id, 100, "done", "ml_calibration_report")
    return [str(out_path)]


@register("totals_calibration_report", tab="totals", label="Totals MAE / residual report", heaviness="light")
def run_totals_calibration_report(job_id: str, params: dict[str, Any]) -> list[str]:
    _progress(job_id, 10, "load", "totals_calibration_report")
    df = _load_results(params.get("run_id"))
    rows = []
    if "PRED_TOTAL" in df.columns and "ACTUAL_HOME" in df.columns:
        pred = pd.to_numeric(df["PRED_TOTAL"], errors="coerce")
        actual = (pd.to_numeric(df["ACTUAL_HOME"], errors="coerce")
                  + pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce"))
        err = (pred - actual).dropna()
        rows.append({"metric": "mae", "value": float(err.abs().mean()), "n": int(len(err))})
        rows.append({"metric": "rmse", "value": float(np.sqrt((err ** 2).mean())), "n": int(len(err))})
        rows.append({"metric": "bias", "value": float(err.mean()), "n": int(len(err))})
    if "MARKET_TOTAL" in df.columns and "ACTUAL_HOME" in df.columns:
        mkt = pd.to_numeric(df["MARKET_TOTAL"], errors="coerce")
        actual = (pd.to_numeric(df["ACTUAL_HOME"], errors="coerce")
                  + pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce"))
        err = (mkt - actual).dropna()
        err = err[mkt.dropna().index.intersection(err.index)] if False else err
        if len(err):
            rows.append({"metric": "market_mae", "value": float(err.abs().mean()), "n": int(len(err))})
    out_path = EXPORTS_DIR / f"totals_calibration_{job_id}.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    _progress(job_id, 100, "done", "totals_calibration_report")
    return [str(out_path)]


@register("promotion_gates", tab="all", label="Run promotion gates on results", heaviness="light")
def run_promotion_gates(job_id: str, params: dict[str, Any]) -> list[str]:
    """Subprocess helper so dashboard venv doesn't need tqdm/sklearn for pipeline imports."""
    _progress(job_id, 5, "load", "promotion_gates", "resolving results CSV")
    csv_path = results_csv_path(params.get("run_id"))
    if csv_path is None or not csv_path.exists():
        raise FileNotFoundError("No backtest_results.csv found for promotion gates")
    out_path = EXPORTS_DIR / f"promotion_gates_{job_id}.json"
    script = CODE_ROOT / "scripts" / "run_promotion_gates.py"
    probe_log: list[str] = []
    py = pipeline_python(log=probe_log)
    for line in probe_log:
        _append_job_log(job_id, line)
    cmd = [py, str(script), "--results", str(csv_path), "--out", str(out_path)]
    _append_job_log(job_id, "CMD " + " ".join(cmd))
    _progress(job_id, 20, "subprocess", "promotion_gates", "running run_promotion_gates.py")
    proc = subprocess.run(
        cmd, cwd=str(CODE_ROOT), capture_output=True, text=True, timeout=600,
    )
    _append_job_log(job_id, (proc.stdout or "")[-2000:])
    if proc.stderr:
        _append_job_log(job_id, "STDERR:\n" + proc.stderr[-2000:])
    if proc.returncode != 0:
        raise RuntimeError(
            f"promotion_gates exited {proc.returncode}: {(proc.stderr or proc.stdout or '')[:400]}"
        )
    if not out_path.exists():
        raise RuntimeError("promotion_gates produced no output JSON")
    _progress(job_id, 100, "done", "promotion_gates")
    return [str(out_path)]


@register("export_slice", tab="all", label="Export filtered CSV", heaviness="light")
def run_export_slice(job_id: str, params: dict[str, Any]) -> list[str]:
    _progress(job_id, 10, "load", "export_slice")
    df = _load_results(params.get("run_id"))
    cols = params.get("columns") or []
    if cols:
        cols = [c for c in cols if c in df.columns]
        if cols:
            df = df[cols]
    # Simple equality filter: {"col": "value"}
    filt = params.get("filter") or {}
    for k, v in filt.items():
        if k in df.columns:
            df = df[df[k].astype(str) == str(v)]
    name = params.get("filename") or f"export_{job_id}.csv"
    out_path = EXPORTS_DIR / Path(name).name
    df.to_csv(out_path, index=False)
    _progress(job_id, 100, "done", "export_slice")
    return [str(out_path)]


@register("edge_policy_calib", tab="ats", label="Edge policy grid (post-hoc)", heaviness="light")
def run_edge_policy_calib(job_id: str, params: dict[str, Any]) -> list[str]:
    _progress(job_id, 5, "load", "edge_policy_calib")
    csv_path = results_csv_path(params.get("run_id"))
    if csv_path is None:
        raise FileNotFoundError("results CSV required")
    out_dir = EXPORTS_DIR / f"edge_policy_{job_id}"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Prefer the real calibrator (positional CSV arg) under a full pipeline Python.
    script = CODE_ROOT / "scripts" / "calibrate_edge_policy.py"
    out_csv = out_dir / "edge_policy_grid.csv"
    if script.exists():
        _progress(job_id, 25, "subprocess", "edge_policy_calib", "calibrate_edge_policy.py")
        cmd = [
            pipeline_python(), str(script), str(csv_path),
            "--out-dir", str(out_dir), "--compare-avoid",
        ]
        try:
            proc = subprocess.run(
                cmd, check=False, cwd=str(CODE_ROOT), timeout=300,
                capture_output=True, text=True,
            )
            (out_dir / "calibrate_edge_policy.log").write_text(
                (proc.stdout or "") + "\n" + (proc.stderr or "")
            )
            if out_csv.exists() or (out_dir / "edge_policy_recommendation.json").exists():
                _progress(job_id, 100, "done", "edge_policy_calib")
                paths = [str(p) for p in out_dir.iterdir() if p.is_file()]
                return paths or [str(out_dir)]
        except Exception as exc:
            (out_dir / "calibrate_edge_policy.log").write_text(str(exc))

    # In-process fallback grid (derive ATS_WIN when CSV lacks it).
    _progress(job_id, 60, "grid", "edge_policy_calib")
    df = ensure_ats_win(pd.read_csv(csv_path))
    if "EDGE" not in df.columns:
        raise ValueError("EDGE column required")
    if "ATS_WIN" not in df.columns:
        raise ValueError(
            "Could not grade ATS (need ACTUAL_MARGIN + DECISION/MARKET_SPREAD + DIRECTION/EDGE_LEAN)"
        )
    rows = []
    edge = pd.to_numeric(df["EDGE"], errors="coerce")
    for thr in [0.0, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5]:
        sub = df[edge.abs() >= thr]
        y = pd.to_numeric(sub["ATS_WIN"], errors="coerce")
        rows.append({
            "min_abs_edge": thr,
            "n": int(y.notna().sum()),
            "ats_hit": float(y.mean()) if y.notna().any() else None,
        })
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    _progress(job_id, 100, "done", "edge_policy_calib")
    return [str(out_csv)]


def _run_subprocess_job(job_id: str, module: str, cmd: list[str], cwd: Path) -> list[str]:
    """Spawn a heavy CLI job; progress comes from parsed suite/backtest telemetry."""
    from dashboard.services.job_telemetry import build_job_telemetry

    meta = job_queue.load_job(job_id)
    log_path = Path(meta["log_path"]) if meta else JOBS_DIR / job_id / "job.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)

    if os.name == "nt":
        popen_kwargs = {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}  # type: ignore[attr-defined]
    else:
        popen_kwargs = {"start_new_session": True}

    _progress(job_id, 1, "starting", module, " ".join(cmd[:6]))
    # Preserve any prior structured lines (CMD …) then stream subprocess stdout.
    with open(log_path, "a", encoding="utf-8") as logf:
        logf.write(f"\n--- subprocess start {now_iso()} ---\n")
        logf.write(" ".join(cmd) + "\n")
        logf.flush()
        proc = subprocess.Popen(
            cmd, cwd=str(cwd), stdout=logf, stderr=subprocess.STDOUT, **popen_kwargs,
        )
    if meta:
        meta["pid"] = proc.pid
        meta["status"] = "running"
        meta["started_at"] = now_iso()
        job_queue.save_job(meta)

    t0 = time.time()
    while True:
        rc = proc.poll()
        if rc is not None:
            break
        try:
            tel = build_job_telemetry(job_id, elapsed_s=time.time() - t0)
            pct = float(tel.get("progress_pct") or 5.0)
            stage = f"stage{tel.get('current_stage', '')}"
            if tel.get("active_tuner"):
                stage = str(tel["active_tuner"])
            msg = tel.get("short_status") or tel.get("headline") or f"pid={proc.pid}"
            job_queue.write_progress(
                job_id,
                pct,
                stage,
                str(msg)[:180],
                eta_seconds=tel.get("timing", {}).get("eta_s"),
            )
            # Avoid duplicating telemetry spam into job.log (headline already in progress).
        except Exception:
            job_queue.write_progress(
                job_id, 5.0, "running", f"pid={proc.pid}", eta_seconds=None,
            )
        time.sleep(5.0)
    elapsed = time.time() - t0
    update_ema(module, elapsed)
    if proc.returncode != 0:
        raise RuntimeError(f"subprocess exited {proc.returncode}; see {log_path}")
    _progress(job_id, 100, "done", module, "complete")
    return [str(log_path)]


def _resolve_pipeline_python(job_id: str) -> str:
    """Pick + validate pipeline interpreter; write probe lines to the job log."""
    probe_log: list[str] = []
    py = pipeline_python(log=probe_log)
    ensure_pipeline_ready(py, log=probe_log)
    for line in probe_log:
        _append_job_log(job_id, line)
    return py


@register(
    "suite_smoke",
    tab="run",
    label="Full suite smoke (mini)",
    heaviness="heavy",
    param_schema={"fields": SUITE_SMOKE_SCHEMA},
)
def run_suite_smoke(job_id: str, params: dict[str, Any]) -> list[str]:
    script = CODE_ROOT / "run_full_suite.py"
    py = _resolve_pipeline_python(job_id)
    p = dict(params or {})
    p.setdefault("preset", "smoke")
    p.setdefault("fast_tuning", True)
    p.setdefault("skip_integrity_tests", p.get("skip_integrity", True))
    if not p.get("label"):
        p["label"] = f"dash_{job_id}"
    cmd = build_suite_cli(py, str(script), p, job_id=job_id)
    _append_job_log(job_id, "CMD " + " ".join(cmd))
    return _run_subprocess_job(job_id, "suite_smoke", cmd, CODE_ROOT)


@register(
    "suite_custom",
    tab="run",
    label="Full suite (custom / notebook-equivalent)",
    heaviness="heavy",
    param_schema={"fields": SUITE_CUSTOM_SCHEMA},
)
def run_suite_custom(job_id: str, params: dict[str, Any]) -> list[str]:
    script = CODE_ROOT / "run_full_suite.py"
    py = _resolve_pipeline_python(job_id)
    cmd = build_suite_cli(py, str(script), params or {}, job_id=job_id)
    _append_job_log(job_id, "CMD " + " ".join(cmd))
    return _run_subprocess_job(job_id, "suite_custom", cmd, CODE_ROOT)


@register(
    "backtest_quick",
    tab="run",
    label="Walk-forward backtest",
    heaviness="heavy",
    param_schema={"fields": BACKTEST_SCHEMA},
)
def run_backtest_quick(job_id: str, params: dict[str, Any]) -> list[str]:
    script = CODE_ROOT / "run_backtest.py"
    py = _resolve_pipeline_python(job_id)
    cmd = build_backtest_cli(py, str(script), params or {})
    _append_job_log(job_id, "CMD " + " ".join(cmd))
    return _run_subprocess_job(job_id, "backtest_quick", cmd, CODE_ROOT)


@register("player_rating_smoke", tab="player", label="Player ratings smoke (quick backtest)", heaviness="heavy")
def run_player_rating_smoke(job_id: str, params: dict[str, Any]) -> list[str]:
    return run_backtest_quick(job_id, params)
