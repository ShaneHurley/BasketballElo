"""Durable on-disk job registry under dashboard_data/jobs/."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Any

from dashboard.config import JOBS_DIR
from dashboard.jobs.eta import now_iso


def _job_dir(job_id: str) -> Path:
    return JOBS_DIR / job_id


def _meta_path(job_id: str) -> Path:
    return _job_dir(job_id) / "meta.json"


def _progress_path(job_id: str) -> Path:
    return _job_dir(job_id) / "progress.json"


def create_job(
    module: str,
    tab: str = "all",
    params: dict | None = None,
    heaviness: str = "light",
) -> dict[str, Any]:
    job_id = uuid.uuid4().hex[:12]
    d = _job_dir(job_id)
    d.mkdir(parents=True, exist_ok=True)
    meta = {
        "id": job_id,
        "module": module,
        "tab": tab,
        "params": params or {},
        "status": "queued",
        "progress_pct": 0.0,
        "stage": "queued",
        "eta_seconds": None,
        "started_at": None,
        "finished_at": None,
        "error": None,
        "result_paths": [],
        "heaviness": heaviness,
        "pid": None,
        "cancel_requested": False,
        "log_path": str(d / "job.log"),
    }
    _meta_path(job_id).write_text(json.dumps(meta, indent=2, default=str))
    _progress_path(job_id).write_text(json.dumps({
        "progress_pct": 0.0, "stage": "queued", "message": "queued",
    }))
    return meta


def load_job(job_id: str) -> dict[str, Any] | None:
    p = _meta_path(job_id)
    if not p.exists():
        return None
    meta = json.loads(p.read_text())
    prog_p = _progress_path(job_id)
    if prog_p.exists():
        try:
            prog = json.loads(prog_p.read_text())
            meta["progress_pct"] = prog.get("progress_pct", meta.get("progress_pct", 0))
            meta["stage"] = prog.get("stage", meta.get("stage", ""))
            if "eta_seconds" in prog:
                meta["eta_seconds"] = prog["eta_seconds"]
            if prog.get("message"):
                meta["message"] = prog["message"]
        except Exception:
            pass
    return meta


def save_job(meta: dict[str, Any]) -> None:
    job_id = meta["id"]
    _job_dir(job_id).mkdir(parents=True, exist_ok=True)
    _meta_path(job_id).write_text(json.dumps(meta, indent=2, default=str))


def write_progress(
    job_id: str,
    progress_pct: float,
    stage: str,
    message: str = "",
    eta_seconds: float | None = None,
    status: str | None = None,
) -> None:
    payload = {
        "progress_pct": float(progress_pct),
        "stage": stage,
        "message": message,
        "eta_seconds": eta_seconds,
        "updated_at": now_iso(),
    }
    if status is not None:
        payload["status"] = status
    _progress_path(job_id).write_text(json.dumps(payload, indent=2))
    # Update meta without re-reading progress (avoids merge churn).
    meta_p = _meta_path(job_id)
    if meta_p.exists():
        meta = json.loads(meta_p.read_text())
        meta["progress_pct"] = payload["progress_pct"]
        meta["stage"] = stage
        meta["eta_seconds"] = eta_seconds
        if status is not None:
            meta["status"] = status
        if message:
            meta["message"] = message
        save_job(meta)


_FINISHED = frozenset({"succeeded", "failed", "cancelled"})


def list_jobs(limit: int = 50, *, reconcile: bool = True) -> list[dict[str, Any]]:
    if reconcile:
        reconcile_stale_jobs()
    rows = []
    if not JOBS_DIR.exists():
        return rows
    for d in sorted(JOBS_DIR.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        meta = load_job(d.name)
        if meta:
            rows.append(meta)
        if len(rows) >= limit:
            break
    return rows


def reconcile_stale_jobs(
    *,
    live_ids: set[str] | None = None,
    orphan_age_seconds: float = 30.0,
) -> int:
    """Flip stuck queued/running jobs that finished or were orphaned by a restart.

    Returns the number of jobs whose status was corrected.
    """
    import time

    live = live_ids if live_ids is not None else set()
    fixed = 0
    if not JOBS_DIR.exists():
        return 0
    now = time.time()
    for d in JOBS_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_p = _meta_path(d.name)
        if not meta_p.exists():
            continue
        try:
            meta = json.loads(meta_p.read_text())
        except Exception:
            continue
        status = meta.get("status")
        if status not in ("queued", "running", "pending"):
            continue

        # Progress file says we're done / failed / cancelled.
        prog_p = _progress_path(d.name)
        prog: dict[str, Any] = {}
        if prog_p.exists():
            try:
                prog = json.loads(prog_p.read_text())
            except Exception:
                prog = {}
        prog_status = prog.get("status")
        pct = float(prog.get("progress_pct", meta.get("progress_pct") or 0) or 0)
        stage = str(prog.get("stage") or meta.get("stage") or "")

        new_status = None
        if prog_status in _FINISHED:
            new_status = prog_status
        elif pct >= 100.0 or stage in ("done", "error"):
            new_status = "failed" if stage == "error" else "succeeded"
        elif d.name not in live:
            # Orphan: no live worker. Use mtime age so we don't race brand-new jobs.
            age = now - d.stat().st_mtime
            if age >= orphan_age_seconds:
                new_status = "failed"
                meta["error"] = meta.get("error") or "job orphaned (server restarted)"

        if new_status is None:
            continue
        meta["status"] = new_status
        meta["progress_pct"] = pct if pct else meta.get("progress_pct", 0)
        if stage:
            meta["stage"] = stage
        if prog.get("message"):
            meta["message"] = prog["message"]
        meta["finished_at"] = meta.get("finished_at") or now_iso()
        save_job(meta)
        # Keep progress.json in sync so the drawer doesn't flip back.
        write_progress(
            d.name,
            float(meta.get("progress_pct") or 100),
            meta.get("stage") or new_status,
            meta.get("message") or "",
            status=new_status,
        )
        fixed += 1
    return fixed


def count_running_heavy() -> int:
    n = 0
    for j in list_jobs(limit=100, reconcile=False):
        if j.get("heaviness") == "heavy" and j.get("status") in ("queued", "running"):
            n += 1
    return n


def delete_job(job_id: str) -> bool:
    """Remove a finished job directory from disk. Refuses active jobs."""
    import shutil
    meta = load_job(job_id)
    if meta is None:
        return False
    if meta.get("status") not in _FINISHED and not meta.get("cancel_requested"):
        if meta.get("status") in ("queued", "running", "pending"):
            raise ValueError("cannot delete an active job; cancel it first")
    d = _job_dir(job_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    return True


def read_job_log(job_id: str, tail: int = 200) -> dict[str, Any]:
    """Return the last ``tail`` lines of a job's log file."""
    meta = load_job(job_id)
    if meta is None:
        return {"job_id": job_id, "lines": [], "error": "job not found"}
    log_path = Path(meta.get("log_path") or str(_job_dir(job_id) / "job.log"))
    if not log_path.exists():
        return {"job_id": job_id, "lines": [], "path": str(log_path), "missing": True}
    try:
        text = log_path.read_text(errors="replace")
    except Exception as exc:
        return {"job_id": job_id, "lines": [], "error": str(exc)}
    lines = text.splitlines()
    if tail and tail > 0:
        lines = lines[-int(tail):]
    return {
        "job_id": job_id,
        "path": str(log_path),
        "lines": lines,
        "total_lines": text.count("\n") + (1 if text and not text.endswith("\n") else 0),
        "status": meta.get("status"),
    }


def clear_finished_jobs() -> int:
    """Delete all succeeded/failed/cancelled job records. Returns count removed."""
    removed = 0
    for j in list_jobs(limit=500):
        if j.get("status") in _FINISHED:
            if delete_job(j["id"]):
                removed += 1
    return removed
