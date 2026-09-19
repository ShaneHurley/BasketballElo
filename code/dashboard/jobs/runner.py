"""Background job executor with heavy-job concurrency lock."""
from __future__ import annotations

import os
import signal
import threading
import traceback
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from dashboard.config import HEAVY_MODULES, MAX_HEAVY_JOBS, MAX_LIGHT_WORKERS
from dashboard.jobs import queue as job_queue
from dashboard.jobs.eta import now_iso
from dashboard.jobs.registry import get_module, list_modules

# Import side-effect: register modules
import dashboard.jobs.modules  # noqa: F401


_heavy_lock = threading.Semaphore(MAX_HEAVY_JOBS)
_executor = ThreadPoolExecutor(max_workers=MAX_LIGHT_WORKERS + MAX_HEAVY_JOBS)
_running: dict[str, Any] = {}  # job_id -> future


def live_job_ids() -> set[str]:
    return set(_running.keys())


def bootstrap_jobs() -> int:
    """Reconcile orphans left on disk from a previous server process."""
    return job_queue.reconcile_stale_jobs(live_ids=live_job_ids(), orphan_age_seconds=0.0)


def _execute(job_id: str) -> None:
    meta = job_queue.load_job(job_id)
    if meta is None:
        return
    module_id = meta["module"]
    info = get_module(module_id)
    if info is None:
        meta["status"] = "failed"
        meta["error"] = f"Unknown module: {module_id}"
        meta["finished_at"] = now_iso()
        job_queue.save_job(meta)
        return

    heavy = module_id in HEAVY_MODULES or info.heaviness == "heavy"
    acquired = False
    try:
        if heavy:
            if not _heavy_lock.acquire(blocking=False):
                meta["status"] = "queued"
                meta["stage"] = "waiting_for_heavy_slot"
                meta["message"] = "Another heavy job is running; waiting…"
                job_queue.save_job(meta)
                _heavy_lock.acquire(blocking=True)
            acquired = True
            meta = job_queue.load_job(job_id) or meta
            if meta.get("cancel_requested") or meta.get("status") == "cancelled":
                meta["status"] = "cancelled"
                meta["finished_at"] = now_iso()
                job_queue.save_job(meta)
                return

        meta["status"] = "running"
        meta["started_at"] = meta.get("started_at") or now_iso()
        job_queue.save_job(meta)

        result_paths = info.runner(job_id, meta.get("params") or {})
        meta = job_queue.load_job(job_id) or meta
        if meta.get("cancel_requested"):
            meta["status"] = "cancelled"
        else:
            meta["status"] = "succeeded"
            meta["progress_pct"] = 100.0
            meta["stage"] = "done"
            meta["result_paths"] = list(result_paths or [])
        meta["finished_at"] = now_iso()
        # Persist terminal status BEFORE progress so a crash can't leave status=queued.
        job_queue.save_job(meta)
        job_queue.write_progress(
            job_id,
            float(meta.get("progress_pct") or 100),
            meta.get("stage") or "done",
            meta.get("message") or "",
            status=meta["status"],
        )
    except Exception as exc:
        meta = job_queue.load_job(job_id) or meta
        meta["status"] = "failed"
        meta["error"] = f"{exc}\n{traceback.format_exc()}"
        meta["finished_at"] = now_iso()
        meta["stage"] = "error"
        job_queue.save_job(meta)
        job_queue.write_progress(
            job_id,
            float(meta.get("progress_pct") or 0),
            "error",
            str(exc),
            status="failed",
        )
    finally:
        if acquired:
            _heavy_lock.release()
        _running.pop(job_id, None)


def submit_job(tab: str, module: str, params: dict[str, Any] | None = None) -> dict:
    info = get_module(module)
    if info is None:
        raise ValueError(f"Unknown module: {module}")
    params = dict(params or {})
    meta = job_queue.create_job(
        tab=tab or info.tab,
        module=module,
        params=params,
        heaviness=info.heaviness,
    )
    fut = _executor.submit(_execute, meta["id"])
    _running[meta["id"]] = fut
    return meta


def cancel_job(job_id: str) -> dict | None:
    meta = job_queue.load_job(job_id)
    if meta is None:
        return None
    meta["cancel_requested"] = True
    pid = meta.get("pid")
    if pid and meta.get("status") in ("running", "queued"):
        try:
            if os.name == "nt":
                os.kill(int(pid), signal.CTRL_BREAK_EVENT)  # type: ignore[attr-defined]
            else:
                os.killpg(int(pid), signal.SIGTERM)
        except Exception:
            try:
                os.kill(int(pid), signal.SIGTERM)
            except Exception:
                pass
    if meta.get("status") in ("queued", "pending"):
        meta["status"] = "cancelled"
        meta["finished_at"] = now_iso()
    job_queue.save_job(meta)
    return meta


def list_available_modules() -> list[dict]:
    return [
        {
            "id": m.id,
            "tab": m.tab,
            "label": m.label,
            "heaviness": m.heaviness,
            "param_schema": m.param_schema,
        }
        for m in list_modules()
    ]
