"""Local T-60 Analysis Dashboard — FastAPI app factory."""
from __future__ import annotations

import asyncio
import json
import time
from pathlib import Path
from typing import Any, Optional

from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from dashboard.config import DEFAULT_HOST, DEFAULT_PORT, DASHBOARD_DATA_DIR
from dashboard.jobs import queue as job_queue
from dashboard.jobs.runner import (
    bootstrap_jobs,
    cancel_job,
    list_available_modules,
    live_job_ids,
    submit_job,
)
from dashboard.models import ExportRequest, JobCreate
from dashboard.paths import load_ui_state, save_ui_state
from dashboard.services import ats as ats_svc
from dashboard.services import bench as bench_svc
from dashboard.services import charts as charts_svc
from dashboard.services import exports as exports_svc
from dashboard.services import health as health_svc
from dashboard.services.job_telemetry import build_job_telemetry
from dashboard.services import live_runs as live_runs_svc
from dashboard.services import ml as ml_svc
from dashboard.services import players as players_svc
from dashboard.services import runs as runs_svc
from dashboard.services import totals as totals_svc

HERE = Path(__file__).resolve().parent
TEMPLATES = Jinja2Templates(directory=str(HERE / "templates"))


def create_app() -> FastAPI:
    app = FastAPI(title="T-60 Analysis Dashboard", docs_url="/api/docs")
    app.mount("/static", StaticFiles(directory=str(HERE / "static")), name="static")

    @app.on_event("startup")
    def _startup_reconcile():
        n = bootstrap_jobs()
        if n:
            print(f"[dashboard] reconciled {n} stale job(s) on startup")

    @app.get("/favicon.ico", include_in_schema=False)
    def favicon():
        return Response(status_code=204)

    @app.get("/", response_class=HTMLResponse)
    def index(request: Request):
        return TEMPLATES.TemplateResponse(
            "index.html",
            {
                "request": request,
                "tabs": [
                    ("all", "All Data"),
                    ("run", "Run Lab"),
                    ("player", "Player"),
                    ("ats", "ATS"),
                    ("ml", "ML"),
                    ("totals", "Totals"),
                    ("jobs", "Jobs"),
                    ("docs", "Documentation"),
                ],
            },
        )

    # ---- Runs / All Data ----
    @app.get("/api/runs")
    def api_runs():
        return runs_svc.get_run_summaries()

    @app.get("/api/runs/{run_id}/summary")
    def api_run_summary(run_id: str):
        return runs_svc.summarize_results(None if run_id == "latest" else run_id)

    @app.get("/api/runs/{run_id}/results")
    def api_run_results(
        run_id: str,
        page: int = 1,
        page_size: int = 50,
        search: Optional[str] = None,
        columns: Optional[str] = None,
    ):
        cols = [c.strip() for c in columns.split(",") if c.strip()] if columns else None
        rid = None if run_id == "latest" else run_id
        return runs_svc.page_results(rid, page=page, page_size=page_size, columns=cols, search=search)

    @app.get("/api/runs/{run_id}/artifacts")
    def api_artifacts(run_id: str):
        return runs_svc.list_artifacts(None if run_id == "latest" else run_id)

    # ---- Charts ----
    @app.get("/api/charts")
    def api_charts(tab: Optional[str] = None):
        return charts_svc.list_charts(tab)

    @app.get("/api/charts/{chart_id}")
    def api_chart(chart_id: str, run_id: Optional[str] = None):
        return charts_svc.build_chart(chart_id, run_id)

    # ---- Tab scorecards ----
    @app.get("/api/ats/scorecard")
    def api_ats(run_id: Optional[str] = None):
        return {"scorecard": ats_svc.ats_scorecard(run_id), "presets": ats_svc.ATS_PRESETS}

    @app.get("/api/ml/scorecard")
    def api_ml(run_id: Optional[str] = None):
        return {"scorecard": ml_svc.ml_scorecard(run_id), "presets": ml_svc.ML_PRESETS}

    @app.get("/api/totals/scorecard")
    def api_totals(run_id: Optional[str] = None):
        return {"scorecard": totals_svc.totals_scorecard(run_id), "presets": totals_svc.TOTALS_PRESETS}

    @app.get("/api/players")
    def api_players(run_id: Optional[str] = None, search: Optional[str] = None):
        return players_svc.load_player_snapshot(run_id, search=search)

    @app.post("/api/players/export")
    def api_players_export(run_id: Optional[str] = Query(None)):
        path = players_svc.export_players_csv(run_id)
        return {"path": path}

    # ---- Export ----
    @app.post("/api/export")
    def api_export(body: ExportRequest):
        try:
            path = exports_svc.export_csv(
                body.run_id,
                columns=body.columns or None,
                filename=body.filename,
            )
        except FileNotFoundError as exc:
            raise HTTPException(404, str(exc)) from exc
        return {"path": path}

    # ---- Jobs ----
    @app.get("/api/modules")
    def api_modules(tab: Optional[str] = None):
        mods = list_available_modules()
        if tab:
            mods = [m for m in mods if m["tab"] in (tab, "all")]
        return mods

    @app.get("/api/jobs")
    def api_jobs(limit: int = 40):
        # Reconcile using live worker set so we don't orphan brand-new jobs.
        job_queue.reconcile_stale_jobs(live_ids=live_job_ids(), orphan_age_seconds=45.0)
        return job_queue.list_jobs(limit=limit, reconcile=False)

    @app.post("/api/jobs/clear-finished")
    def api_clear_finished():
        n = job_queue.clear_finished_jobs()
        return {"cleared": n}

    @app.get("/api/jobs/{job_id}/log")
    def api_job_log(job_id: str, tail: int = Query(200, ge=1, le=5000)):
        data = job_queue.read_job_log(job_id, tail=tail)
        if data.get("error") == "job not found":
            raise HTTPException(404, "job not found")
        return data

    @app.get("/api/jobs/{job_id}/telemetry")
    def api_job_telemetry(job_id: str):
        meta = job_queue.load_job(job_id)
        if meta is None:
            raise HTTPException(404, "job not found")
        return build_job_telemetry(job_id, meta=meta)

    @app.get("/api/presets/suite")
    def api_suite_presets():
        from dashboard.jobs.schemas import SUITE_PRESETS
        return SUITE_PRESETS

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str):
        meta = job_queue.load_job(job_id)
        if meta is None:
            raise HTTPException(404, "job not found")
        return meta

    @app.post("/api/jobs")
    def api_create_job(body: JobCreate):
        params = dict(body.params or {})
        if body.run_id:
            params.setdefault("run_id", body.run_id)
        # Light reports need a CSV; default to newest run with results.
        if not params.get("run_id") and body.module not in (
            "suite_smoke", "suite_custom", "backtest_quick", "player_rating_smoke",
        ):
            for row in runs_svc.get_run_summaries():
                if row.get("has_results"):
                    params["run_id"] = row["id"]
                    break
        try:
            meta = submit_job(body.tab, body.module, params)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        print(f"[dashboard] job created id={meta['id']} module={body.module} tab={body.tab} params_keys={list(params)}")
        return meta

    @app.post("/api/jobs/{job_id}/cancel")
    def api_cancel(job_id: str):
        meta = cancel_job(job_id)
        if meta is None:
            raise HTTPException(404, "job not found")
        print(f"[dashboard] job cancel id={job_id} status={meta.get('status')}")
        return meta

    @app.delete("/api/jobs/{job_id}")
    def api_delete_job(job_id: str):
        try:
            ok = job_queue.delete_job(job_id)
        except ValueError as exc:
            raise HTTPException(400, str(exc)) from exc
        if not ok:
            raise HTTPException(404, "job not found")
        return {"deleted": job_id}

    @app.get("/api/jobs/{job_id}/events")
    async def api_job_events(job_id: str):
        """SSE stream of progress updates (+ telemetry) until the job finishes."""
        if job_queue.load_job(job_id) is None:
            raise HTTPException(404, "job not found")

        async def gen():
            last = None
            while True:
                meta = job_queue.load_job(job_id) or {}
                try:
                    tel = build_job_telemetry(job_id, meta=meta)
                except Exception:
                    tel = None
                payload_obj = {**meta, "telemetry": tel}
                payload = json.dumps(payload_obj, default=str)
                if payload != last:
                    yield f"data: {payload}\n\n"
                    last = payload
                if meta.get("status") in ("succeeded", "failed", "cancelled"):
                    break
                await asyncio.sleep(1.0)

        return StreamingResponse(gen(), media_type="text/event-stream")

    # ---- UI state ----
    @app.get("/api/ui-state")
    def api_get_ui():
        return load_ui_state()

    @app.post("/api/ui-state")
    def api_set_ui(body: dict[str, Any]):
        cur = load_ui_state()
        cur.update(body)
        save_ui_state(cur)
        return cur

    @app.get("/api/bench")
    def api_bench():
        return bench_svc.load_bench_summary()

    @app.get("/api/health")
    def api_model_health(run_id: Optional[str] = None):
        """Walk-forward model health for the Documentation tab (Epic 7.5)."""
        rid = None if run_id in (None, "", "latest") else run_id
        return health_svc.load_model_health(rid)

    @app.get("/api/ping")
    def api_ping():
        return {"ok": True, "data_dir": str(DASHBOARD_DATA_DIR)}

    @app.get("/api/live_runs")
    def api_live_runs(refresh: bool = Query(True)):
        """Host-side ablation / calib / watchdog status for the Live runs panel."""
        return live_runs_svc.get_live_runs(refresh=refresh)

    return app


app = create_app()


def main(host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> None:
    import uvicorn
    uvicorn.run("dashboard.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
