"""Load run summaries and paginated results tables."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.config import OUTPUT_ROOT, STATE_DIR
from dashboard.paths import list_runs, results_csv_path, run_dir_for_id, safe_resolve


def get_run_summaries() -> list[dict]:
    return list_runs()


def load_ats_status(run_id: str | None = None) -> dict | None:
    candidates = []
    rd = run_dir_for_id(run_id) if run_id else None
    if rd:
        candidates.append(rd / "ats_status.json")
        candidates.append(rd / "artifacts" / "ats_status.json")
    candidates.append(STATE_DIR / "ats_status.json")
    for p in candidates:
        if p.exists():
            try:
                return json.loads(p.read_text())
            except Exception:
                continue
    return None


def load_odds_provenance(run_id: str | None = None) -> dict | None:
    rd = run_dir_for_id(run_id) if run_id else None
    for base in ([rd] if rd else []) + [OUTPUT_ROOT, STATE_DIR]:
        if base is None:
            continue
        for name in ("odds_provenance.json", "artifacts/odds_provenance.json"):
            p = Path(base) / name
            if p.exists():
                try:
                    return json.loads(p.read_text())
                except Exception:
                    pass
    return None


def summarize_results(run_id: str | None) -> dict[str, Any]:
    path = results_csv_path(run_id)
    summary: dict[str, Any] = {
        "run_id": run_id,
        "results_path": str(path) if path else None,
        "n_games": 0,
        "spread_mae": None,
        "total_mae": None,
        "ats_hit_rate": None,
        "roi": None,
        "actionable_bets": None,
        "quote_source": None,
        "ats_status": load_ats_status(run_id),
        "odds_provenance": load_odds_provenance(run_id),
        "research_only_banner": False,
        "banner_message": None,
    }
    if path is None or not path.exists():
        return summary
    df = pd.read_csv(path, nrows=50000)
    summary["n_games"] = len(df)
    if "SPREAD_ERR" in df.columns:
        summary["spread_mae"] = float(pd.to_numeric(df["SPREAD_ERR"], errors="coerce").abs().mean())
    elif "PRED_MARGIN" in df.columns and "ACTUAL_HOME" in df.columns:
        pred = pd.to_numeric(df["PRED_MARGIN"], errors="coerce")
        act = pd.to_numeric(df["ACTUAL_HOME"], errors="coerce") - pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")
        summary["spread_mae"] = float((pred - act).abs().mean())
    if "TOTAL_ERR" in df.columns:
        summary["total_mae"] = float(pd.to_numeric(df["TOTAL_ERR"], errors="coerce").abs().mean())
    if "ATS_WIN" in df.columns:
        summary["ats_hit_rate"] = float(pd.to_numeric(df["ATS_WIN"], errors="coerce").mean())
    if "ROI" in df.columns:
        summary["roi"] = float(pd.to_numeric(df["ROI"], errors="coerce").mean())
    elif "PNL" in df.columns:
        pnl = pd.to_numeric(df["PNL"], errors="coerce").dropna()
        if len(pnl):
            summary["roi"] = float(pnl.mean())
    if "ACTIONABLE" in df.columns:
        summary["actionable_bets"] = int(pd.to_numeric(df["ACTIONABLE"], errors="coerce").fillna(0).sum())
    for col in ("quote_source", "QUOTE_SOURCE", "odds_source"):
        if col in df.columns:
            vals = df[col].dropna().astype(str)
            if len(vals):
                summary["quote_source"] = vals.mode().iloc[0]
                break
    # Honesty banners
    qs = (summary.get("quote_source") or "").lower()
    prov = summary.get("odds_provenance") or {}
    tip = "tip_proxy" in qs or prov.get("quote_source") == "tip_proxy" or prov.get("uses_tip_proxy")
    if tip:
        summary["research_only_banner"] = True
        summary["banner_message"] = (
            "Research-only: quote_source=tip_proxy. Do not use for promotion ROI."
        )
    ats_st = summary.get("ats_status") or {}
    if ats_st.get("blocked") or ats_st.get("status") in ("blocked", "reject", "research_only"):
        summary["research_only_banner"] = True
        msg = ats_st.get("message") or ats_st.get("reason") or "ATS blocked / research-only per ats_status."
        summary["banner_message"] = (summary.get("banner_message") or "") + (" | " if summary.get("banner_message") else "") + str(msg)
    return summary


def page_results(
    run_id: str | None,
    page: int = 1,
    page_size: int = 50,
    columns: list[str] | None = None,
    search: str | None = None,
) -> dict[str, Any]:
    path = results_csv_path(run_id)
    if path is None or not path.exists():
        return {"rows": [], "total": 0, "page": page, "page_size": page_size, "columns": []}
    df = pd.read_csv(path)
    all_cols = list(df.columns)
    if search:
        mask = pd.Series(False, index=df.index)
        for c in df.columns:
            mask |= df[c].astype(str).str.contains(search, case=False, na=False)
        df = df[mask]
    if columns:
        use = [c for c in columns if c in df.columns]
        if use:
            df = df[use]
    total = len(df)
    start = max(0, (page - 1) * page_size)
    end = start + page_size
    chunk = df.iloc[start:end]
    return {
        "rows": chunk.fillna("").to_dict(orient="records"),
        "total": total,
        "page": page,
        "page_size": page_size,
        "columns": list(chunk.columns) if len(chunk.columns) else all_cols,
        "all_columns": all_cols,
    }


def list_artifacts(run_id: str | None) -> list[dict]:
    rd = run_dir_for_id(run_id) if run_id else None
    if rd is None:
        return []
    out = []
    for pattern in ("checkpoints/*.json", "artifacts/*", "plots/*", "analysis_plots/*", "*.png", "**/*.pkl"):
        for p in rd.glob(pattern):
            if not p.is_file():
                continue
            try:
                safe_resolve(p, rd)
            except Exception:
                continue
            out.append({
                "path": str(p.relative_to(rd)),
                "size": p.stat().st_size,
                "suffix": p.suffix,
            })
    # Dedupe by path
    seen = set()
    uniq = []
    for a in out:
        if a["path"] in seen:
            continue
        seen.add(a["path"])
        uniq.append(a)
    return sorted(uniq, key=lambda x: x["path"])[:500]
