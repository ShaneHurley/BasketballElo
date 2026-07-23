"""Post-hoc ablation on saved backtest CSV (no re-run of walk-forward training)."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.ablation import compute_metrics, passes_ablation_gate
from pipeline.config import MAX_QUANTILE_WIDTH, MIN_CONFIDENCE_SCORE, MIN_EDGE_BUCKET
from pipeline.metrics import add_all_profile_columns


def _apply_edge_bucket_filter(df: pd.DataFrame, min_edge: float = MIN_EDGE_BUCKET) -> pd.DataFrame:
    out = df.copy()
    if "DIRECTION" not in out.columns:
        return out
    active = out["DIRECTION"] != "Pass"
    edge_ok = out["EDGE"].abs() >= float(min_edge)
    width = out.get("CONF_WIDTH", pd.Series(24.0, index=out.index))
    width_ok = width.fillna(24.0) <= MAX_QUANTILE_WIDTH
    out.loc[active & ~(edge_ok & width_ok), "DIRECTION"] = "Pass"
    return out


def _apply_confidence_gate(df: pd.DataFrame, min_conf: float | None = None) -> pd.DataFrame:
    out = df.copy()
    if "CONFIDENCE" not in out.columns or "DIRECTION" not in out.columns:
        return out
    thr = float(MIN_CONFIDENCE_SCORE if min_conf is None else min_conf)
    active = out["DIRECTION"] != "Pass"
    out.loc[active & (out["CONFIDENCE"].fillna(0) < thr), "DIRECTION"] = "Pass"
    return out


def _apply_tight_width(df: pd.DataFrame, max_width: float = 22.0) -> pd.DataFrame:
    out = df.copy()
    if "CONF_WIDTH" not in out.columns or "DIRECTION" not in out.columns:
        return out
    active = out["DIRECTION"] != "Pass"
    out.loc[active & (out["CONF_WIDTH"].fillna(99) > max_width), "DIRECTION"] = "Pass"
    return out


def simulate_selection_mode(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    """Return a copy of results with bet selection applied post-hoc."""
    out = df.copy()
    if mode == "edge_bucket" or mode == "edge_bucket_ats":
        out = _apply_edge_bucket_filter(out)
    elif mode == "edge_bucket_conf":
        out = _apply_edge_bucket_filter(out)
        out = _apply_confidence_gate(out)
    elif mode == "edge_bucket_width22":
        out = _apply_edge_bucket_filter(out)
        out = _apply_tight_width(out, max_width=22.0)
    elif mode == "edge_bucket_full":
        out = _apply_edge_bucket_filter(out)
        out = _apply_tight_width(out, max_width=22.0)
        out = _apply_confidence_gate(out)
    return out


def posthoc_ablation_summary(
    results_df: pd.DataFrame,
    modes: tuple[str, ...] = (
        "baseline_current",
        "edge_bucket",
        "edge_bucket_ats",
        "edge_bucket_conf",
        "edge_bucket_width22",
        "edge_bucket_full",
    ),
) -> pd.DataFrame:
    """Compare selection modes on one backtest export."""
    rows = []
    for mode in modes:
        sim = results_df if mode == "baseline_current" else simulate_selection_mode(results_df, mode)
        m = compute_metrics(sim)
        if m.empty:
            continue
        overall = m[m["season"] == "ALL"].iloc[0].to_dict()
        overall["config"] = mode
        overall["season"] = "ALL"
        rows.append(overall)
    return pd.DataFrame(rows)


def passes_flip_gate(metrics: dict, *, roi_gain: float = 0.02, min_ats: float = 0.55, min_bets: int = 400) -> bool:
    """Plan checklist: pooled ROI +2%, ATS >= 55%, n_bets >= 400, MAE not worse."""
    roi = float(metrics.get("roi", np.nan))
    ats = float(metrics.get("ats_pct", np.nan))
    n = int(metrics.get("n_bets", 0) or 0)
    return (
        np.isfinite(roi)
        and np.isfinite(ats)
        and n >= min_bets
        and ats >= min_ats
        and roi >= roi_gain
    )
