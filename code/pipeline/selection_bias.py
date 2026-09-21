"""Path-variance / selection-bias reporting (not CPCV training).

Reports season-path variance of primary metrics so anti-overfit review can
compare a candidate run against a persisted baseline. Never used as a train
gate or DSR promote criterion.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def season_metric_paths(
    results: pd.DataFrame,
    *,
    season_col: str = "season",
    mae_col: str = "PRED_SPREAD",
    actual_col: str = "ACTUAL_MARGIN",
) -> pd.DataFrame:
    """Per-season MAE / bet-count / ATS path (read-only)."""
    if results is None or results.empty:
        return pd.DataFrame()
    df = results.copy()
    if season_col not in df.columns:
        return pd.DataFrame()
    rows = []
    for season, g in df.groupby(season_col):
        mae = np.nan
        if mae_col in g.columns and actual_col in g.columns:
            pred = pd.to_numeric(g[mae_col], errors="coerce")
            act = pd.to_numeric(g[actual_col], errors="coerce")
            mask = pred.notna() & act.notna()
            if mask.any():
                mae = float((pred[mask] - act[mask]).abs().mean())
        n_bets = 0
        ats = np.nan
        if "DIRECTION" in g.columns:
            act_b = g[g["DIRECTION"].astype(str).ne("Pass")]
            n_bets = len(act_b)
            if "HIT" in act_b.columns and n_bets:
                hits = pd.to_numeric(act_b["HIT"], errors="coerce")
                ats = float(hits.mean()) if hits.notna().any() else np.nan
        clv = np.nan
        if "CLV" in g.columns and n_bets:
            act_b = g[g["DIRECTION"].astype(str).ne("Pass")]
            clv = float(pd.to_numeric(act_b["CLV"], errors="coerce").mean())
        rows.append({
            "season": season,
            "spread_mae": mae,
            "n_bets": n_bets,
            "ats_pct": ats,
            "mean_clv": clv,
        })
    return pd.DataFrame(rows)


def path_variance_summary(path_df: pd.DataFrame) -> dict[str, Any]:
    """Aggregate path variance for review / path_variance.json."""
    if path_df is None or path_df.empty:
        return {"n_seasons": 0}
    mae = pd.to_numeric(path_df.get("spread_mae"), errors="coerce")
    bets = pd.to_numeric(path_df.get("n_bets"), errors="coerce")
    clv = pd.to_numeric(path_df.get("mean_clv"), errors="coerce")
    out: dict[str, Any] = {
        "n_seasons": int(len(path_df)),
        "mae_mean": float(mae.mean()) if mae.notna().any() else np.nan,
        "mae_std": float(mae.std(ddof=0)) if mae.notna().sum() > 1 else 0.0,
        "n_bets_total": int(bets.fillna(0).sum()),
        "n_bets_std": float(bets.std(ddof=0)) if bets.notna().sum() > 1 else 0.0,
        "clv_mean": float(clv.mean()) if clv.notna().any() else np.nan,
        "seasons": path_df.to_dict(orient="records"),
    }
    return out


def compare_to_baseline(
    current: dict[str, Any],
    baseline: dict[str, Any] | None,
    *,
    mae_tol: float = 0.25,
    ece_tol: float = 0.02,
) -> tuple[bool, str]:
    """Return (ok, message). Fail if MAE/ECE regress vs baseline on path means."""
    if not baseline:
        return True, "no baseline — first suite path; store review_baseline.json"
    cur_mae = current.get("mae_mean", current.get("spread_mae_mean"))
    # Accept either a flat stability dict (spread_mae_mean) or a nested review payload.
    base_mae = (
        baseline.get("mae_mean")
        if baseline.get("mae_mean") is not None
        else baseline.get("spread_mae_mean")
    )
    if base_mae is None:
        base_mae = (baseline.get("stability") or {}).get("spread_mae_mean")
    cur_ece = current.get("ece_mean", current.get("ece"))
    base_ece = (
        baseline.get("ece_mean")
        if baseline.get("ece_mean") is not None
        else (baseline.get("stability") or {}).get("ece_mean")
    )
    msgs = []
    ok = True
    if (
        cur_mae is not None and base_mae is not None
        and np.isfinite(cur_mae) and np.isfinite(base_mae)
        and float(cur_mae) > float(base_mae) + mae_tol
    ):
        ok = False
        msgs.append(f"MAE path mean {cur_mae:.3f} > baseline {base_mae:.3f}+{mae_tol}")
    if (
        cur_ece is not None and base_ece is not None
        and np.isfinite(cur_ece) and np.isfinite(base_ece)
        and float(cur_ece) > float(base_ece) + ece_tol
    ):
        ok = False
        msgs.append(f"ECE path mean {cur_ece:.4f} > baseline {base_ece:.4f}+{ece_tol}")
    if ok:
        return True, "path variance vs baseline OK (MAE/ECE)"
    return False, "; ".join(msgs)


def write_path_variance(run_dir: Path, results: pd.DataFrame) -> dict[str, Any]:
    """Compute and write ``checkpoints/path_variance.json``."""
    import json
    from datetime import datetime, timezone

    path_df = season_metric_paths(results)
    summary = path_variance_summary(path_df)
    summary["at"] = datetime.now(timezone.utc).isoformat()
    ck = Path(run_dir) / "checkpoints"
    ck.mkdir(parents=True, exist_ok=True)
    (ck / "path_variance.json").write_text(json.dumps(summary, indent=2, default=str))
    return summary
