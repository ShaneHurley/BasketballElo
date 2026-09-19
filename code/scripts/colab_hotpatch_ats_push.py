"""Colab hotpatch: paste into a notebook cell if diagnostics KeyErrors on ats_push.

Force-reloads pipeline.diagnostics from disk when possible; otherwise monkeypatches
the broken empty-frame path in the already-imported module so
run_backtest_diagnostics can finish without a runtime restart.

Also prints the live bet-gate config so you can confirm Colab is not still on
MAX_QUANTILE_WIDTH=22 / EDGE_AVOID_BAND=(7, 9.5).
"""
from __future__ import annotations

# --- paste from here ---
import importlib
import sys

import numpy as np
import pandas as pd


def _apply_ats_push_hotpatch():
    import pipeline.diagnostics as diag

    def _ensure_ats_outcome_columns(m: pd.DataFrame) -> pd.DataFrame:
        out = m.copy() if m is not None else pd.DataFrame()
        if "ats_win" not in out.columns:
            out["ats_win"] = pd.Series(dtype=int)
        if "ats_push" not in out.columns:
            out["ats_push"] = pd.Series(dtype=int)
        return out

    def _ats_frame(d: pd.DataFrame) -> pd.DataFrame:
        from pipeline.bet_selection import actionable_spread_frame, spread_side_series

        m = actionable_spread_frame(d)
        if m is None or m.empty:
            return _ensure_ats_outcome_columns(m if m is not None else pd.DataFrame())
        side = m["_side"] if "_side" in m.columns else spread_side_series(m)
        cover = m["ACTUAL_MARGIN"] + m["MARKET_SPREAD"]
        m = m.copy()
        m["ats_win"] = (
            ((side == "Home") & (cover > 0))
            | ((side == "Away") & (cover < 0))
        ).astype(int)
        m["ats_push"] = (cover == 0).astype(int)
        return _ensure_ats_outcome_columns(m)

    def _non_push_ats(bets: pd.DataFrame) -> pd.DataFrame:
        if bets is None or bets.empty:
            return _ensure_ats_outcome_columns(bets if bets is not None else pd.DataFrame())
        if "ats_push" not in bets.columns:
            return _ensure_ats_outcome_columns(bets.iloc[0:0])
        return bets[bets["ats_push"] == 0].copy()

    def _confidence_tier_table(g: pd.DataFrame) -> pd.DataFrame:
        g = diag._norm_results(g)
        if "CONFIDENCE_TIER" not in g.columns:
            return pd.DataFrame()
        bets = _non_push_ats(_ats_frame(g))
        if bets.empty:
            return pd.DataFrame()
        rows = []
        for tier in sorted(bets["CONFIDENCE_TIER"].dropna().unique()):
            sub = bets[bets["CONFIDENCE_TIER"] == tier]
            wp = sub["ats_win"].mean()
            rows.append({
                "tier": int(tier),
                "n_bets": len(sub),
                "ats_pct": wp,
                "roi": wp * (100.0 / 110.0) - (1 - wp) if len(sub) else np.nan,
                "mean_edge": sub["EDGE"].abs().mean() if "EDGE" in sub.columns else np.nan,
            })
        return pd.DataFrame(rows)

    diag._ensure_ats_outcome_columns = _ensure_ats_outcome_columns
    diag._ats_frame = _ats_frame
    diag._non_push_ats = _non_push_ats
    diag._confidence_tier_table = _confidence_tier_table

    if "pipeline.confidence_diagnostics" in sys.modules:
        import pipeline.confidence_diagnostics as cd
        importlib.reload(cd)

    print("Hotpatched pipeline.diagnostics (_ats_frame / _confidence_tier_table).")
    print("diagnostics file:", getattr(diag, "__file__", "?"))


try:
    import pipeline.diagnostics as _diag0
    importlib.reload(_diag0)
    print("Reloaded pipeline.diagnostics from disk.")
except Exception as exc:
    print("Reload skipped:", exc)

_apply_ats_push_hotpatch()

try:
    import pipeline.config as cfg
    print(
        "Gate config:",
        f"mode={getattr(cfg, 'BET_SELECTION_MODE', '?')}",
        f"max_width={getattr(cfg, 'MAX_QUANTILE_WIDTH', '?')}",
        f"avoid_band={getattr(cfg, 'EDGE_AVOID_BAND', '?')}",
        f"tight_spread={getattr(cfg, 'SKIP_TIGHT_SPREAD', '?')}",
        f"trust>={getattr(cfg, 'MIN_DISAGREEMENT_TRUST', '?')}",
    )
except Exception as exc:
    print("Could not print gate config:", exc)

from pipeline.diagnostics import run_backtest_diagnostics  # noqa: E402
# --- paste ends ---
