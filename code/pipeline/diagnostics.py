"""Per-season diagnostics: text summaries + matplotlib graphs.

Two entry points:
  - ``diagnose_loaded_data``  — after stints/odds load (coverage, dates, games)
  - ``run_backtest_diagnostics`` — after walk-forward backtest (MAE, edge, ATS, …)

Designed for both local scripts and the Colab notebook (inline ``plt.show()``).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

BREAKEVEN = 110.0 / 210.0  # -110 break-even ATS win rate
EDGE_BUCKETS = [0, 2, 4, 6, 8, np.inf]
EDGE_LABELS = ["0-2", "2-4", "4-6", "6-8", "8+"]
CLOSE_MARGINS = (3, 5, 7)


def _season_col(df: pd.DataFrame) -> str:
    for c in ("simulated_season_window", "season", "SEASON"):
        if c in df.columns:
            return c
    return "season"


def _norm_results(df: pd.DataFrame) -> pd.DataFrame:
    """Standardise column names used across the pipeline."""
    out = df.copy()
    rename = {}
    if "DATE" in out.columns and "date" not in out.columns:
        rename["DATE"] = "date"
    if rename:
        out = out.rename(columns=rename)
    if "date" in out.columns:
        out["date"] = pd.to_datetime(out["date"], errors="coerce")
    sc = _season_col(out)
    out["_season"] = out[sc].astype(str)
    if "ACTUAL_MARGIN" not in out.columns and {"ACTUAL_HOME", "ACTUAL_AWAY"}.issubset(out.columns):
        out["ACTUAL_MARGIN"] = out["ACTUAL_HOME"] - out["ACTUAL_AWAY"]
    if "SPREAD_ERR" not in out.columns and "PRED_SPREAD" in out.columns:
        out["SPREAD_ERR"] = (out["PRED_SPREAD"] - out["ACTUAL_MARGIN"]).abs()
    if "RAW_SPREAD_ERR" not in out.columns and "RAW_PRED_MARGIN" in out.columns:
        out["RAW_SPREAD_ERR"] = (out["RAW_PRED_MARGIN"] - out["ACTUAL_MARGIN"]).abs()
    if "TOTAL_ERR" not in out.columns and "PRED_TOTAL" in out.columns:
        out["TOTAL_ERR"] = out["PRED_TOTAL"] - (out["ACTUAL_HOME"] + out["ACTUAL_AWAY"])
    return out


def _spread_side_series(d: pd.DataFrame) -> pd.Series:
    if "EDGE_LEAN" in d.columns:
        lean = d["EDGE_LEAN"].fillna("Pass").astype(str)
    else:
        lean = d.get("DIRECTION", pd.Series("Pass", index=d.index)).fillna("Pass").astype(str)
    if "EDGE" in d.columns:
        edge = pd.to_numeric(d["EDGE"], errors="coerce").fillna(0)
        need = lean.isin(("Pass", "nan", "")) | lean.isna()
        inferred = np.where(edge > 0, "Home", np.where(edge < 0, "Away", "Pass"))
        lean = lean.where(~need, inferred)
    return lean


def _ats_lean_frame(d: pd.DataFrame) -> pd.DataFrame:
    """ATS outcomes for every lined game using model lean (includes sub-threshold edges)."""
    if d is None or d.empty or "MARKET_SPREAD" not in d.columns:
        return _ensure_ats_outcome_columns(pd.DataFrame())
    m = d[d["MARKET_SPREAD"].notna()].copy()
    if m.empty or "ACTUAL_MARGIN" not in m.columns:
        return _ensure_ats_outcome_columns(m)
    side = _spread_side_series(m)
    m = m[side != "Pass"].copy()
    m["_side"] = side.loc[m.index]
    cover = m["ACTUAL_MARGIN"] + m["MARKET_SPREAD"]
    m["ats_win"] = (
        ((m["_side"] == "Home") & (cover > 0))
        | ((m["_side"] == "Away") & (cover < 0))
    ).astype(int)
    m["ats_push"] = (cover == 0).astype(int)
    return m


def _ensure_ats_outcome_columns(m: pd.DataFrame) -> pd.DataFrame:
    """Guarantee ``ats_win`` / ``ats_push`` exist (empty-frame safe)."""
    out = m.copy() if m is not None else pd.DataFrame()
    if "ats_win" not in out.columns:
        out["ats_win"] = pd.Series(dtype=int)
    if "ats_push" not in out.columns:
        out["ats_push"] = pd.Series(dtype=int)
    return out


def _ats_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Actionable ATS bets with ``ats_win`` / ``ats_push`` always present.

    An empty actionable set still returns those columns so callers can safely
    filter on ``ats_push`` without KeyError (e.g. confidence-only mode with
    zero placed bets for a season).
    """
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
    """Filter push rows; no-op/empty-safe when ``ats_push`` is missing."""
    if bets is None or bets.empty:
        return _ensure_ats_outcome_columns(bets if bets is not None else pd.DataFrame())
    if "ats_push" not in bets.columns:
        return _ensure_ats_outcome_columns(bets.iloc[0:0])
    return bets[bets["ats_push"] == 0].copy()


def _ou_frame(d: pd.DataFrame) -> pd.DataFrame:
    """O/U bets with win flag."""
    if "OU_DIRECTION" not in d.columns or "MARKET_TOTAL" not in d.columns:
        return pd.DataFrame()
    m = d[(d["OU_DIRECTION"] != "Pass") & d["MARKET_TOTAL"].notna()].copy()
    if m.empty:
        return m
    if "OU_WIN" in m.columns:
        m["ou_win"] = pd.to_numeric(m["OU_WIN"], errors="coerce")
    else:
        actual = m["ACTUAL_HOME"] + m["ACTUAL_AWAY"]
        mkt = pd.to_numeric(m["MARKET_TOTAL"], errors="coerce")
        m["ou_win"] = np.where(
            m["OU_DIRECTION"] == "Over",
            (actual > mkt).astype(float),
            (actual < mkt).astype(float),
        )
    return m


def _ou_edge_bucket_table(g: pd.DataFrame) -> pd.DataFrame:
    """O/U win rate by |TOTAL_EDGE| bucket."""
    g = _norm_results(g)
    if "TOTAL_EDGE" not in g.columns:
        return pd.DataFrame()
    ou = _ou_frame(g)
    if ou.empty:
        return pd.DataFrame()
    ou = ou.copy()
    ou["ebin"] = pd.cut(ou["TOTAL_EDGE"].abs(), bins=EDGE_BUCKETS, labels=EDGE_LABELS, right=False)
    rows = []
    for lbl in EDGE_LABELS:
        sub = ou[ou["ebin"] == lbl]
        if sub.empty:
            continue
        wp = float(sub["ou_win"].mean())
        rows.append({"bucket": lbl, "n": len(sub), "ou_win_pct": wp, "roi": wp * (100 / 110) - (1 - wp)})
    return pd.DataFrame(rows)


def _tight_spread_table(g: pd.DataFrame) -> pd.DataFrame:
    """ATS on actionable bets when |market spread| is within TIGHT_SPREAD_MAX."""
    from pipeline.config import TIGHT_SPREAD_MAX

    g = _norm_results(g)
    bets = _ats_frame(g)
    if bets.empty or "MARKET_SPREAD" not in bets.columns:
        return pd.DataFrame()
    tight = bets[bets["MARKET_SPREAD"].abs() <= float(TIGHT_SPREAD_MAX)]
    tight = _non_push_ats(tight)
    if tight.empty:
        return pd.DataFrame()
    wp = float(tight["ats_win"].mean())
    return pd.DataFrame([{
        "max_spread": float(TIGHT_SPREAD_MAX),
        "n_bets": len(tight),
        "ats_pct": wp,
        "roi": wp * (100.0 / 110.0) - (1 - wp),
    }])


def _season_metrics(g: pd.DataFrame) -> dict:
    """One row of summary stats for a single season."""
    g = _norm_results(g)
    n = len(g)
    has_mkt = g["MARKET_SPREAD"].notna() if "MARKET_SPREAD" in g.columns else pd.Series(False, index=g.index)
    n_mkt = int(has_mkt.sum())
    spread_mae = float(g["SPREAD_ERR"].mean()) if "SPREAD_ERR" in g else np.nan
    raw_spread_mae = float(g["RAW_SPREAD_ERR"].mean()) if "RAW_SPREAD_ERR" in g else np.nan
    spread_bias = float((g["PRED_SPREAD"] - g["ACTUAL_MARGIN"]).mean()) if "PRED_SPREAD" in g else np.nan
    total_mae = float(g["TOTAL_ERR"].abs().mean()) if "TOTAL_ERR" in g else np.nan
    brier = np.nan
    log_loss = np.nan
    ece = np.nan
    if "WIN_PROB" in g.columns:
        hw = (g["ACTUAL_MARGIN"] > 0).astype(int)
        brier = float(((g["WIN_PROB"] - hw) ** 2).mean())
        try:
            from pipeline.calibration_metrics import compute_ece
            ece = float(compute_ece(hw.values, g["WIN_PROB"].values))
        except Exception:
            ece = np.nan
        try:
            from sklearn.metrics import log_loss as sk_log_loss
            probs = np.clip(g["WIN_PROB"].values.astype(float), 1e-6, 1 - 1e-6)
            log_loss = float(sk_log_loss(hw.values, np.column_stack([1 - probs, probs])))
        except Exception:
            log_loss = np.nan
    # Prefer calibration metrics over hit-rate for ML (kept for reference only)
    ml_acc = float(g["MODEL_ML_CORRECT"].mean()) if "MODEL_ML_CORRECT" in g else np.nan

    crps = np.nan
    if "PRED_TOTAL" in g.columns and "ACTUAL_HOME" in g.columns and "ACTUAL_AWAY" in g.columns:
        from pipeline.market import crps_gaussian
        from pipeline.config import OU_DEFAULT_SIGMA
        actual_total = g["ACTUAL_HOME"] + g["ACTUAL_AWAY"]
        sigma = g["SIGMA_TOTAL"] if "SIGMA_TOTAL" in g.columns else OU_DEFAULT_SIGMA
        if isinstance(sigma, (int, float)):
            sigma = pd.Series(float(sigma), index=g.index)
        crps_vals = [
            crps_gaussian(a, m, s)
            for a, m, s in zip(actual_total, g["PRED_TOTAL"], sigma.fillna(OU_DEFAULT_SIGMA))
        ]
        crps_vals = [c for c in crps_vals if c is not None and np.isfinite(c)]
        crps = float(np.mean(crps_vals)) if crps_vals else np.nan

    close = {}
    for cm in CLOSE_MARGINS:
        close[f"pct_decided_{cm}"] = float((g["ACTUAL_MARGIN"].abs() <= cm).mean()) if n else np.nan

    edge = g["EDGE"].abs() if "EDGE" in g else pd.Series(dtype=float)
    edge_stats = {
        "edge_mean": float(edge.mean()) if len(edge) else np.nan,
        "edge_median": float(edge.median()) if len(edge) else np.nan,
        "edge_p90": float(edge.quantile(0.9)) if len(edge) else np.nan,
    }

    ats = _ats_frame(g)
    non_push = _non_push_ats(ats)
    ats_n = int(len(non_push))
    if ats_n:
        wp = non_push["ats_win"].mean()
        roi = wp * (100.0 / 110.0) - (1 - wp)
    else:
        wp = roi = np.nan

    lean = _ats_lean_frame(g)
    lean_non_push = _non_push_ats(lean)
    lean_n = int(len(lean_non_push))
    lean_wp = lean_non_push["ats_win"].mean() if lean_n else np.nan

    ou_n = ou_wp = ou_roi = np.nan
    ou = _ou_frame(g)
    if not ou.empty:
        ou_n = len(ou)
        ou_wp = float(ou["ou_win"].mean())
        ou_roi = ou_wp * (100.0 / 110.0) - (1 - ou_wp)

    cover_ece = np.nan
    cover_brier = np.nan
    sharpe = np.nan
    max_dd = np.nan
    if "COVER_PROB_CALIBRATED" in g.columns and not ats.empty:
        from pipeline.calibration_metrics import compute_brier, compute_ece
        bets = _non_push_ats(ats).copy()
        if len(bets):
            cover_brier = compute_brier(bets["ats_win"].values, bets["COVER_PROB_CALIBRATED"].values)
            cover_ece = compute_ece(bets["ats_win"].values, bets["COVER_PROB_CALIBRATED"].values)
    if "PROFIT_MODERATE" in g.columns and "STAKE_MODERATE" in g.columns:
        from pipeline.metrics import sharpe_ratio, max_drawdown
        active = g[g["STAKE_MODERATE"] > 0]
        if len(active) >= 5:
            profits = active["PROFIT_MODERATE"].values
            sharpe = sharpe_ratio(profits)
            max_dd = max_drawdown(np.cumsum(profits))

    dates_ok = True
    if "date" in g.columns:
        dates_ok = g["date"].dropna().dt.normalize().nunique() > 1

    return {
        "n_games": n,
        "n_with_odds": n_mkt,
        "pct_with_odds": n_mkt / n if n else np.nan,
        "spread_mae": spread_mae,
        "raw_spread_mae": raw_spread_mae,
        "spread_bias": spread_bias,
        "total_mae": total_mae,
        "crps": crps,
        "brier": brier,
        "log_loss": log_loss,
        "ece": ece,
        "cover_brier": cover_brier,
        "cover_ece": cover_ece,
        "sharpe_per_bet": sharpe,
        "max_drawdown": max_dd,
        "ml_acc": ml_acc,  # reference only — do not promote models on hit-rate
        "n_bets": ats_n,
        "n_leans": lean_n,
        "lean_ats_pct": lean_wp,
        "ats_pct": wp,
        "roi": roi,
        "ou_n_bets": ou_n,
        "ou_win_pct": ou_wp,
        "ou_roi": ou_roi,
        "dates_ok": dates_ok,
        **close,
        **edge_stats,
    }


def _confidence_tier_table(g: pd.DataFrame) -> pd.DataFrame:
    """ATS win rate and ROI by calibrated confidence tier."""
    g = _norm_results(g)
    if "CONFIDENCE_TIER" not in g.columns:
        return pd.DataFrame()
    bets = _ats_frame(g)
    bets = _non_push_ats(bets)
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


def _confidence_score_table(g: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """ATS win rate and ROI by confidence score deciles."""
    from pipeline.confidence_diagnostics import confidence_score_table
    return confidence_score_table(g, score_col="CONFIDENCE", n_bins=n_bins)


def _edge_bucket_table(g: pd.DataFrame, include_ats: bool = True) -> pd.DataFrame:
    g = _norm_results(g)
    if "EDGE" not in g.columns:
        return pd.DataFrame()
    g = g.copy()
    g["edge_bucket"] = pd.cut(g["EDGE"].abs(), bins=EDGE_BUCKETS, labels=EDGE_LABELS, right=False)
    rows = []
    for lbl in EDGE_LABELS:
        sub = g[g["edge_bucket"] == lbl]
        row = {"bucket": lbl, "n_games": len(sub), "pct_games": len(sub) / len(g) if len(g) else 0}
        if include_ats and "MARKET_SPREAD" in g.columns:
            bets = _non_push_ats(_ats_lean_frame(sub))
            row["n_bets"] = len(bets)
            row["ats_pct"] = bets["ats_win"].mean() if len(bets) else np.nan
            if "WIN_PCT" in sub.columns and len(sub):
                row["mean_win_pct"] = float(pd.to_numeric(sub["WIN_PCT"], errors="coerce").mean())
            elif "CONFIDENCE" in sub.columns and len(sub):
                row["mean_win_pct"] = float(pd.to_numeric(sub["CONFIDENCE"], errors="coerce").mean())
            if len(bets) and bets["ats_win"].notna().any():
                wp = bets["ats_win"].mean()
                row["roi"] = wp * (100.0 / 110.0) - (1 - wp)
            else:
                row["roi"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def diagnose_loaded_data(stints_df: pd.DataFrame, odds_dict: dict | None = None) -> pd.DataFrame:
    """Print a per-season load summary (games, dates, odds coverage)."""
    if stints_df is None or stints_df.empty:
        print("No stint data loaded.")
        return pd.DataFrame()

    from pipeline.market import get_game_odds

    df = stints_df.copy()
    df["game_date"] = pd.to_datetime(df["game_date"], errors="coerce")
    games = (
        df[["season", "GAME_ID", "game_date", "home_team", "away_team"]]
        .drop_duplicates(["season", "GAME_ID"])
    )

    rows = []
    print("\n" + "=" * 72)
    print("LOAD DIAGNOSTICS · per season")
    print("=" * 72)
    for season in sorted(games["season"].unique()):
        sg = games[games["season"] == season]
        n_g = len(sg)
        dmin, dmax = sg["game_date"].min(), sg["game_date"].max()
        n_dates = sg["game_date"].dropna().dt.normalize().nunique()
        dates_ok = n_dates > 1
        n_odds = 0
        n_distinct_clv = 0
        if odds_dict:
            for _, r in sg.iterrows():
                d = r["game_date"]
                if pd.isna(d):
                    continue
                go = get_game_odds(d, r["home_team"], odds_dict)
                if pd.notna(go.get("spread")):
                    n_odds += 1
                dec, clo = go.get("decision_spread"), go.get("closing_spread")
                if pd.notna(dec) and pd.notna(clo) and float(dec) != float(clo):
                    n_distinct_clv += 1
        pct_odds = n_odds / n_g if n_g else 0
        flag = "" if dates_ok else "  ⚠️ collapsed dates"
        print(f"\n── Season {season}{flag}")
        print(f"   games={n_g:,}  unique_dates={n_dates}  range={str(dmin)[:10]} → {str(dmax)[:10]}")
        print(f"   odds matched (via get_game_odds): {n_odds:,} ({pct_odds:.1%})")
        print(f"   distinct decision≠close (CLV-capable): {n_distinct_clv:,}")
        rows.append({
            "season": season, "n_games": n_g, "n_dates": n_dates,
            "date_min": dmin, "date_max": dmax, "dates_ok": dates_ok,
            "n_odds_matched": n_odds, "pct_odds": pct_odds,
            "n_distinct_decision_close": n_distinct_clv,
        })
    return pd.DataFrame(rows)


def assert_odds_coverage(
    coverage: pd.DataFrame,
    *,
    min_rate: float | None = None,
    require_any: bool = True,
) -> None:
    """Raise SystemExit if odds match rate is too low on seasons with games."""
    from pipeline.config import MIN_ODDS_MATCH_RATE

    thresh = float(MIN_ODDS_MATCH_RATE if min_rate is None else min_rate)
    if coverage is None or coverage.empty:
        if require_any:
            raise SystemExit("Odds coverage empty — refuse walk-forward")
        return
    bad = coverage[coverage["pct_odds"] < thresh]
    if bad.empty:
        return
    detail = ", ".join(
        f"{int(r.season)}={r.pct_odds:.1%}" for r in bad.itertuples()
    )
    raise SystemExit(
        f"Odds match rate below {thresh:.0%} for seasons: {detail}. "
        "Fix team/date join (canonicalize_odds_dict) or drop seasons. "
        "Refuse silent walk-forward with blind market metrics."
    )


def warn_pinnacle_clv_gap(
    coverage: pd.DataFrame,
    *,
    pinnacle_path: str | Path | None,
    pin_date_min=None,
    pin_date_max=None,
) -> list[int]:
    """Hard-warn when seasons overlap Pinnacle coverage but have 0 decision≠close.

    Returns list of season labels that look covered by Pinnacle yet lack CLV-capable
    lines (typical cause: suite years excluded 2025–26 while pin file is 2025-only).
    """
    if coverage is None or coverage.empty or pinnacle_path is None:
        return []
    p = Path(pinnacle_path)
    if not p.exists():
        return []

    dmin, dmax = pin_date_min, pin_date_max
    if dmin is None or dmax is None:
        try:
            # Cheap header+timestamp scan for range (avoid full parse cost when provided).
            pdf = pd.read_csv(p, usecols=["timestamp"])
            ts = pd.to_datetime(pdf["timestamp"], errors="coerce").dropna()
            if ts.empty:
                return []
            dmin, dmax = ts.min(), ts.max()
        except Exception:
            return []

    dmin = pd.Timestamp(dmin)
    dmax = pd.Timestamp(dmax)
    flagged: list[int] = []
    for r in coverage.itertuples():
        smin = getattr(r, "date_min", None)
        smax = getattr(r, "date_max", None)
        if smin is None or smax is None or pd.isna(smin) or pd.isna(smax):
            continue
        smin, smax = pd.Timestamp(smin), pd.Timestamp(smax)
        # Overlap with Pinnacle feed window?
        if smax < dmin or smin > dmax:
            continue
        n_dist = int(getattr(r, "n_distinct_decision_close", 0) or 0)
        if n_dist == 0:
            flagged.append(int(r.season))
    if flagged:
        print(
            f"\n  ❌ PINNACLE CLV GAP: seasons {flagged} overlap Pinnacle "
            f"({str(dmin)[:10]}→{str(dmax)[:10]}) but n_distinct_decision_close=0. "
            "Include start-year 2025 (2025–26) in --years, or check schedule/pin merge. "
            "CLV will stay NaN — do not promote ROI/ATS."
        )
    return flagged


def _setup_plt():
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.grid": True, "grid.alpha": 0.3,
        "axes.spines.top": False, "axes.spines.right": False, "font.size": 10,
    })
    return plt


def _save_or_show(fig, save_dir: Path | None, name: str):
    import matplotlib.pyplot as plt
    if save_dir is not None:
        save_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_dir / name, dpi=150, bbox_inches="tight")
        plt.close(fig)
    else:
        fig.tight_layout()
        plt.show()


def _autoscaled_ylim(values, *, pad_frac: float = 0.08, floor: float | None = None, ceil: float | None = None):
    """Data-driven y-limits with fractional padding (avoids hard clips)."""
    arr = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return None
    lo, hi = float(np.nanmin(arr)), float(np.nanmax(arr))
    if lo == hi:
        lo, hi = lo - 0.05, hi + 0.05
    pad = max((hi - lo) * pad_frac, 0.02)
    y0, y1 = lo - pad, hi + pad
    if floor is not None:
        y0 = max(y0, floor)
    if ceil is not None:
        y1 = min(y1, ceil)
    if y0 >= y1:
        y0, y1 = lo - pad, hi + pad
    return y0, y1


def _autoscaled_lim_pair(xs, ys, *, pad_frac: float = 0.08):
    """Joint padded limits for reliability-style scatter axes."""
    xlim = _autoscaled_ylim(xs, pad_frac=pad_frac, floor=0.0, ceil=1.0)
    ylim = _autoscaled_ylim(ys, pad_frac=pad_frac, floor=0.0, ceil=1.0)
    return xlim, ylim


def _print_gate_attrition(g: pd.DataFrame) -> None:
    """Explain zero actionable ATS bets via successive confidence-gate filters."""
    import pipeline.config as cfg

    n = len(g)
    if n == 0:
        return
    edge = pd.to_numeric(g["EDGE"] if "EDGE" in g.columns else pd.Series(np.nan, index=g.index), errors="coerce").abs()
    conf = pd.to_numeric(
        g["CONFIDENCE"] if "CONFIDENCE" in g.columns else (
            g["WIN_PCT"] if "WIN_PCT" in g.columns else pd.Series(np.nan, index=g.index)
        ),
        errors="coerce",
    )
    width = pd.to_numeric(
        g["CONF_WIDTH"] if "CONF_WIDTH" in g.columns else (
            g["SPREAD_QUANTILE_WIDTH"] if "SPREAD_QUANTILE_WIDTH" in g.columns
            else pd.Series(np.nan, index=g.index)
        ),
        errors="coerce",
    )
    mkt = pd.to_numeric(
        g["MARKET_SPREAD"] if "MARKET_SPREAD" in g.columns else pd.Series(np.nan, index=g.index),
        errors="coerce",
    ).abs()
    trust = pd.to_numeric(
        g["DISAGREEMENT_TRUST"] if "DISAGREEMENT_TRUST" in g.columns
        else pd.Series(1.0, index=g.index),
        errors="coerce",
    ).fillna(1.0)
    phantom = (
        pd.to_numeric(g["PHANTOM_INJURY_FLAG"], errors="coerce").fillna(0).astype(bool)
        if "PHANTOM_INJURY_FLAG" in g.columns
        else pd.Series(False, index=g.index)
    )

    has_odds = mkt.notna()
    pass_edge = has_odds & edge.notna() & (edge >= float(cfg.CONFIDENCE_MIN_EDGE))
    floor = float(cfg.MIN_CONFIDENCE_SCORE)
    if cfg.CONFIDENCE_EDGE_SCALED:
        need = np.where(edge < 5.0, floor + 2.0, np.where(edge < 8.0, floor + 1.0, floor))
    else:
        need = floor
    # Prefer per-row MIN_CONFIDENCE_SCORE from the backtest export when present
    # (adaptive floor may have lowered the season gate below config default).
    if "MIN_CONFIDENCE_SCORE" in g.columns:
        row_floor = pd.to_numeric(g["MIN_CONFIDENCE_SCORE"], errors="coerce")
        if cfg.CONFIDENCE_EDGE_SCALED:
            need = np.where(
                edge < 5.0, row_floor + 2.0,
                np.where(edge < 8.0, row_floor + 1.0, row_floor),
            )
        else:
            need = row_floor.fillna(floor)
        floor = float(row_floor.dropna().iloc[0]) if row_floor.notna().any() else floor
    pass_conf = pass_edge & conf.notna() & (conf >= need)
    if cfg.MAX_QUANTILE_WIDTH is None:
        pass_width = pass_conf
    else:
        pass_width = pass_conf & (width.isna() | (width <= float(cfg.MAX_QUANTILE_WIDTH)))
    pass_trust = pass_width & (trust >= float(cfg.MIN_DISAGREEMENT_TRUST))
    pass_phantom = pass_trust & (~phantom) if cfg.SKIP_PHANTOM_INJURY else pass_trust
    pass_tight = (
        pass_phantom & (mkt > float(cfg.TIGHT_SPREAD_MAX))
        if cfg.SKIP_TIGHT_SPREAD
        else pass_phantom
    )
    from pipeline.bet_selection import normalize_edge_avoid_bands
    bands = normalize_edge_avoid_bands(cfg.EDGE_AVOID_BAND)
    if bands:
        in_band = pd.Series(False, index=g.index)
        for lo, hi in bands:
            in_band = in_band | ((edge >= float(lo)) & (edge < float(hi)))
        agree = pd.to_numeric(
            g["ELO_META_AGREEMENT"] if "ELO_META_AGREEMENT" in g.columns
            else pd.Series(1.0, index=g.index),
            errors="coerce",
        ).fillna(1.0)
        pass_band = pass_tight & (~in_band | (agree >= float(cfg.EDGE_AVOID_BAND_MIN_ELO_AGREE)))
    else:
        pass_band = pass_tight

    print("  ⚠️  0 ATS bets — gate attrition (rows surviving each filter):")
    print(f"     games={n:,}  |edge|>={cfg.CONFIDENCE_MIN_EDGE}: {int(pass_edge.sum()):,}")
    print(
        f"     + conf>={cfg.MIN_CONFIDENCE_SCORE}"
        f"{' (edge-scaled)' if cfg.CONFIDENCE_EDGE_SCALED else ''}: {int(pass_conf.sum()):,}"
    )
    print(f"     + width<={cfg.MAX_QUANTILE_WIDTH}: {int(pass_width.sum()):,}")
    print(f"     + trust>={cfg.MIN_DISAGREEMENT_TRUST}: {int(pass_trust.sum()):,}")
    if cfg.SKIP_PHANTOM_INJURY:
        print(f"     + no phantom injury: {int(pass_phantom.sum()):,}")
    if cfg.SKIP_TIGHT_SPREAD:
        print(f"     + |market|>{cfg.TIGHT_SPREAD_MAX}: {int(pass_tight.sum()):,}")
    print(f"     + edge avoid band {bands or None}: {int(pass_band.sum()):,}")
    if width.notna().any():
        print(
            f"     conf width: median={float(width.median()):.1f}  "
            f"p90={float(width.quantile(0.9)):.1f}  "
            f"(cap={cfg.MAX_QUANTILE_WIDTH})"
        )


def run_backtest_diagnostics(
    results_df: pd.DataFrame,
    save_dir: str | Path | None = None,
    show_graphs: bool = True,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame]]:
    """Print per-season backtest diagnostics and draw graphs.

    Returns (season_summary_df, {season: edge_bucket_df}).
    """
    if results_df is None or results_df.empty:
        print("No backtest results to diagnose.")
        return pd.DataFrame(), {}

    df = _norm_results(results_df)
    seasons = sorted(df["_season"].unique())
    save_path = Path(save_dir) if save_dir else None
    plt = _setup_plt() if show_graphs else None

    summary_rows = []
    edge_tables: dict[str, pd.DataFrame] = {}

    print("\n" + "=" * 72)
    print("BACKTEST DIAGNOSTICS · per season")
    print("=" * 72)
    print("(PRED_SPREAD is the bias/slope-corrected model spread)")

    for s in seasons:
        g = df[df["_season"] == s]
        m = _season_metrics(g)
        m["season"] = s
        summary_rows.append(m)
        eb = _edge_bucket_table(g)
        edge_tables[s] = eb

        flag = "" if m["dates_ok"] else "  ⚠️ suspect dates"
        print(f"\n{'─' * 72}")
        print(f"Season {s}{flag}")
        print(f"  games={m['n_games']:,}  with_odds={m['n_with_odds']:,} ({m['pct_with_odds']:.1%})")
        print(f"  spread MAE={m['spread_mae']:.2f}  raw MAE={m.get('raw_spread_mae', float('nan')):.2f}  "
              f"bias(corrected)={m['spread_bias']:+.2f}  total MAE={m['total_mae']:.2f}  "
              f"CRPS={m.get('crps', float('nan')):.2f}")
        print(f"  Brier={m['brier']:.3f}  LogLoss={m.get('log_loss', float('nan')):.3f}  "
              f"ECE={m.get('ece', float('nan')):.3f}  cover ECE={m.get('cover_ece', float('nan')):.3f}  "
              f"(ML hit-rate ref={m['ml_acc']:.1%})")
        if pd.notna(m.get("sharpe_per_bet")):
            print(f"  moderate Sharpe={m['sharpe_per_bet']:.2f}  max DD={m.get('max_drawdown', float('nan')):.3f}")
        print(f"  close games: ≤3pt={m['pct_decided_3']:.1%}  ≤5pt={m['pct_decided_5']:.1%}  ≤7pt={m['pct_decided_7']:.1%}")
        print(f"  edge |model-market|: mean={m['edge_mean']:.2f}  median={m['edge_median']:.2f}  p90={m['edge_p90']:.2f}")
        print(f"  ATS bets={m['n_bets']:,}  win%={m['ats_pct']:.1%}  ROI={m['roi']:+.1%}  (break-even={BREAKEVEN:.1%})")
        if int(m["n_bets"] or 0) == 0:
            _print_gate_attrition(g)
        if not eb.empty:
            print("  edge buckets (|edge| pts):")
            for _, r in eb.iterrows():
                ats_s = f"  ATS={r['ats_pct']:.1%} n={int(r['n_bets'])}" if pd.notna(r.get("ats_pct")) else ""
                print(f"    {r['bucket']:>4}: {int(r['n_games']):4d} games ({r['pct_games']:.1%}){ats_s}")
        ct = _confidence_tier_table(g)
        if not ct.empty:
            print("  confidence tiers (calibrated):")
            for _, r in ct.iterrows():
                print(
                    f"    tier {int(r['tier'])}: n={int(r['n_bets']):4d}  "
                    f"ATS={r['ats_pct']:.1%}  ROI={r['roi']:+.1%}  "
                    f"mean|edge|={r['mean_edge']:.2f}"
                )

    summary = pd.DataFrame(summary_rows)

    # ── pooled overall ──
    om = _season_metrics(df)
    print(f"\n{'═' * 72}")
    print("OVERALL (all seasons pooled)")
    print(f"  games={om['n_games']:,}  spread MAE={om['spread_mae']:.2f}  total MAE={om['total_mae']:.2f}")
    print(f"  ATS {om['n_bets']:,} bets @ {om['ats_pct']:.1%}  ROI={om['roi']:+.1%}")
    if pd.notna(om.get("lean_ats_pct")):
        print(f"  all leans: n={int(om.get('n_leans', 0)):,}  win%={om['lean_ats_pct']:.1%}")
    tight = _tight_spread_table(df)
    if not tight.empty:
        r = tight.iloc[0]
        print(
            f"  tight spread (|line|<={r['max_spread']:.1f}): "
            f"n={int(r['n_bets'])}  ATS={r['ats_pct']:.1%}  ROI={r['roi']:+.1%}"
        )
    eb_all = _edge_bucket_table(df)
    if not eb_all.empty:
        print("  pooled edge buckets:")
        print(eb_all.to_string(index=False, float_format=lambda x: f"{x:.3f}" if isinstance(x, float) else str(x)))

    if not show_graphs:
        return summary, edge_tables

    palette = {s: c for s, c in zip(seasons, plt.cm.tab10.colors)}

    # 1 · ATS win% by season
    fig, ax = plt.subplots(figsize=(9, 5))
    vals = [summary.loc[summary["season"] == s, "ats_pct"].iloc[0] for s in seasons]
    ns = [summary.loc[summary["season"] == s, "n_bets"].iloc[0] for s in seasons]
    colors = ["#55A868" if summary.loc[summary["season"] == s, "dates_ok"].iloc[0] else "#C44E52" for s in seasons]
    bars = ax.bar([str(s) for s in seasons], vals, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(BREAKEVEN, ls="--", color="black", lw=1.2, label=f"Break-even ({BREAKEVEN:.1%})")
    for b, v, n in zip(bars, vals, ns):
        if pd.notna(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.1%}\n(n={int(n)})", ha="center", fontsize=9)
    ylim = _autoscaled_ylim(list(vals) + [BREAKEVEN], pad_frac=0.12, floor=0.0, ceil=1.0)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_ylabel("ATS win rate")
    ax.set_title("ATS Win Rate by Season (active spread bets)")
    ax.legend()
    _save_or_show(fig, save_path, "01_ats_by_season.png")

    # 2 · Spread MAE + bias by season
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(seasons))
    mae_v = [summary.loc[summary["season"] == s, "spread_mae"].iloc[0] for s in seasons]
    bias_v = [summary.loc[summary["season"] == s, "spread_bias"].iloc[0] for s in seasons]
    ax.bar(x - 0.2, mae_v, 0.4, label="Spread MAE (corrected pred)", color="#4C72B0", edgecolor="black")
    ax.bar(x + 0.2, np.abs(bias_v), 0.4, label="|Bias| (pred − actual)", color="#DD8452", edgecolor="black")
    ax.set_xticks(x)
    ax.set_xticklabels([str(s) for s in seasons], rotation=15, ha="right")
    ax.set_ylabel("Points")
    ax.set_title("Corrected Spread Error by Season")
    ax.legend()
    _save_or_show(fig, save_path, "02_spread_mae_bias.png")

    # 3b · ATS win% by confidence tier (pooled)
    ct_all = _confidence_tier_table(df)
    if not ct_all.empty:
        fig, ax = plt.subplots(figsize=(8, 5))
        tiers = ct_all["tier"].astype(int).astype(str)
        ax.bar(tiers, ct_all["ats_pct"], color="#8172B3", edgecolor="black", alpha=0.85)
        ax.axhline(BREAKEVEN, ls="--", color="black", lw=1)
        for i, (_, r) in enumerate(ct_all.iterrows()):
            ax.text(i, r["ats_pct"] + 0.01, f"n={int(r['n_bets'])}", ha="center", fontsize=9)
        ax.set_xlabel("Confidence tier")
        ax.set_ylabel("ATS win rate")
        ax.set_title("ATS Win % by Calibrated Confidence Tier (pooled)")
        _save_or_show(fig, save_path, "03b_ats_by_confidence_tier.png")

    # 3 · ATS win% by edge bucket (grouped by season)
    fig, ax = plt.subplots(figsize=(11, 6))
    width = 0.8 / max(len(seasons), 1)
    for i, s in enumerate(seasons):
        eb = edge_tables[s]
        if eb.empty:
            continue
        xs = np.arange(len(EDGE_LABELS)) + i * width
        ys = [eb.loc[eb["bucket"] == lbl, "ats_pct"].iloc[0] if lbl in eb["bucket"].values else np.nan
              for lbl in EDGE_LABELS]
        ax.bar(xs, ys, width, label=str(s), color=palette[s], edgecolor="black", alpha=0.85)
    ax.axhline(BREAKEVEN, ls="--", color="black", lw=1)
    ax.set_xticks(np.arange(len(EDGE_LABELS)) + width * (len(seasons) - 1) / 2)
    ax.set_xticklabels(EDGE_LABELS)
    ax.set_xlabel("|Model edge| bucket (pts)")
    ax.set_ylabel("ATS win rate")
    ax.set_title("ATS Win % by Edge Bucket (per season)")
    ax.legend(fontsize=8, ncol=2)
    _save_or_show(fig, save_path, "03_ats_by_edge_bucket.png")

    # 3c · Rolling ATS vs |edge| (data-driven curve, not fixed buckets)
    try:
        from pipeline.edge_analysis import rolling_edge_ats, segment_breakpoints

        roll = rolling_edge_ats(df)
        if not roll.empty:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.plot(roll["edge_center"], roll["ats_pct"], color="#4C72B0", lw=2, marker="o", ms=4)
            ax.axhline(BREAKEVEN, ls="--", color="black", lw=1, label=f"Break-even ({BREAKEVEN:.1%})")
            seg = segment_breakpoints(df)
            for _, r in seg.iterrows():
                if r["flat_roi"] < 0:
                    ax.axvspan(r["edge_lo"], min(r["edge_hi"], 14), alpha=0.12, color="red")
            ax.set_xlabel("|Model edge| (pts, rolling ±0.75 pt)")
            ax.set_ylabel("ATS win rate")
            ax.set_title("Rolling ATS vs Edge (valleys ≈ injury/noise zones)", fontweight="bold")
            ax.legend()
            _save_or_show(fig, save_path, "03c_rolling_ats_vs_edge.png")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠ rolling edge plot skipped: {e}")

    # 4 · Edge distribution (% of games in each bucket) stacked
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(EDGE_LABELS))
    for s in seasons:
        eb = edge_tables[s]
        if eb.empty:
            continue
        pcts = [eb.loc[eb["bucket"] == lbl, "pct_games"].iloc[0] if lbl in eb["bucket"].values else 0
                for lbl in EDGE_LABELS]
        ax.bar(EDGE_LABELS, pcts, bottom=bottom, label=str(s), color=palette[s], edgecolor="white", alpha=0.9)
        bottom += np.array(pcts)
    ax.set_ylabel("Share of games")
    ax.set_xlabel("|Model − market| edge bucket (pts)")
    ax.set_title("Edge Distribution by Season (stacked % of games)")
    ax.legend(fontsize=8, ncol=2)
    _save_or_show(fig, save_path, "04_edge_distribution.png")

    # 5 · Close-game rates
    fig, ax = plt.subplots(figsize=(9, 5))
    w = 0.25
    for j, cm in enumerate(CLOSE_MARGINS):
        vals = [summary.loc[summary["season"] == s, f"pct_decided_{cm}"].iloc[0] for s in seasons]
        ax.bar(np.arange(len(seasons)) + (j - 1) * w, vals, w, label=f"≤{cm} pts", edgecolor="black", alpha=0.85)
    ax.set_xticks(np.arange(len(seasons)))
    ax.set_xticklabels([str(s) for s in seasons], rotation=15, ha="right")
    ax.set_ylabel("% of games")
    ax.set_title("Games Decided by Close Margins")
    ax.legend()
    _save_or_show(fig, save_path, "05_close_games.png")

    # 6 · Total MAE by season
    if summary["total_mae"].notna().any():
        fig, ax = plt.subplots(figsize=(9, 5))
        tm = [summary.loc[summary["season"] == s, "total_mae"].iloc[0] for s in seasons]
        ax.bar([str(s) for s in seasons], tm, color="#8172B3", edgecolor="black", alpha=0.85)
        ax.set_ylabel("Total MAE (pts)")
        ax.set_title("Game Total Prediction Error by Season")
        _save_or_show(fig, save_path, "06_total_mae.png")

    # 7 · Win-probability calibration (one panel per season + pooled)
    n_panels = len(seasons) + 1
    ncols = min(3, n_panels)
    nrows = int(np.ceil(n_panels / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
    axes_flat = axes.flatten()
    pb = np.linspace(0, 1, 11)
    for idx, (label, g) in enumerate(list(zip(seasons, [df[df["_season"] == s] for s in seasons])) + [("ALL", df)]):
        ax = axes_flat[idx]
        g = g.copy()
        g["home_win"] = (g["ACTUAL_MARGIN"] > 0).astype(int)
        g["pbin"] = pd.cut(g["WIN_PROB"], pb)
        cal = g.groupby("pbin", observed=True).agg(
            pred=("WIN_PROB", "mean"), obs=("home_win", "mean"), n=("home_win", "size")
        ).dropna()
        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
        if not cal.empty:
            ax.scatter(cal["pred"], cal["obs"], s=np.clip(cal["n"] / 5, 20, 200),
                       color="#4C72B0", edgecolor="black", zorder=3)
            ax.plot(cal["pred"], cal["obs"], color="#4C72B0", lw=1.2)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(str(label), fontsize=10)
        ax.set_xlabel("Pred win prob")
        ax.set_ylabel("Observed")
    for j in range(n_panels, len(axes_flat)):
        axes_flat[j].set_visible(False)
    fig.suptitle("Win-Probability Calibration (corrected spread → win prob)", fontweight="bold")
    _save_or_show(fig, save_path, "07_calibration.png")

    # 8 · Rolling spread MAE through each season
    fig, ax = plt.subplots(figsize=(10, 5.5))
    for s in seasons:
        g = df[df["_season"] == s].copy()
        if g["date"].notna().any() and g["date"].dropna().dt.normalize().nunique() > 1:
            g = g.sort_values("date")
        else:
            g = g.sort_values("GAME_ID")
        err = (g["PRED_SPREAD"] - g["ACTUAL_MARGIN"]).abs()
        roll = err.rolling(50, min_periods=15).mean()
        prog = np.linspace(0, 1, len(g))
        ax.plot(prog, roll, label=str(s), color=palette[s], lw=2 if g["date"].dropna().dt.normalize().nunique() > 1 else 1.2)
    ax.set_xlabel("Season progress (0 → 1)")
    ax.set_ylabel("Rolling spread MAE (50-game window)")
    ax.set_title("Corrected Spread MAE Through the Season")
    ax.legend(fontsize=8, ncol=2)
    _save_or_show(fig, save_path, "08_rolling_mae.png")

    if save_path:
        print(f"\n📊 Diagnostic plots saved → {save_path}/")
        try:
            from pipeline.edge_analysis import run_edge_analysis
            edge_report = run_edge_analysis(df, verbose=True)
            for key in ("fine_bins", "rolling", "segments", "injury_split"):
                tbl = edge_report.get(key)
                if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                    tbl.to_csv(save_path / f"edge_{key}.csv", index=False)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠ edge curve analysis skipped: {e}")

    # Upset / blowout / favorite-size residual slices
    try:
        upset_tbl = favorite_loss_by_spread_bin(df)
        blow_tbl = blowout_accuracy_table(df)
        if not upset_tbl.empty:
            print("\n  Favorite-loss rate by |MARKET_SPREAD| bin:")
            for _, r in upset_tbl.iterrows():
                print(
                    f"    {r['bin']:>8}: n={int(r['n']):4d}  fav_loss={r['fav_loss_rate']:.1%}  "
                    f"mean_win_prob_fav={r['mean_win_prob_fav']:.3f}  MAE={r['mae']:.2f}"
                )
        if not blow_tbl.empty:
            print("\n  Blowout realization vs predicted P(|m|>=k):")
            for _, r in blow_tbl.iterrows():
                print(
                    f"    |m|>={int(r['threshold'])}: rate={r['actual_rate']:.1%}  "
                    f"mean_pred_p={r['mean_pred_p']:.3f}  n={int(r['n'])}"
                )
    except Exception as e:  # noqa: BLE001
        print(f"  (upset/blowout diagnostics skipped: {e})")

    return summary, edge_tables


def generate_betting_plots(df, save_dir=None):
    """Bankroll curves, ROI by confidence tier, ATS reliability diagram."""
    import matplotlib.pyplot as plt
    from pipeline.metrics import add_all_profile_columns

    save_path = Path(save_dir) if save_dir else None
    if df is None or df.empty:
        return

    df = _norm_results(df)

    if "PROFIT_MODERATE" not in df.columns:
        df = add_all_profile_columns(df)

    fig, ax = plt.subplots(figsize=(11, 6))
    df_sorted = df.sort_values("DATE") if "DATE" in df.columns else df
    for profile, color in [("conservative", "#4C72B0"), ("moderate", "#55A868"), ("aggressive", "#C44E52")]:
        pc = f"PROFIT_{profile.upper()}"
        if pc not in df_sorted.columns:
            continue
        cum = df_sorted[pc].fillna(0).cumsum()
        ax.plot(range(len(cum)), cum, label=profile, color=color, lw=2)
    ax.axhline(0, ls="--", color="gray")
    ax.set_xlabel("Bet sequence")
    ax.set_ylabel("Cumulative profit (bankroll units)")
    ax.set_title("Bankroll Curves by Stake Profile", fontweight="bold")
    ax.legend()
    _save_or_show(fig, save_path, "10_bankroll_by_profile.png")

    if "CONFIDENCE_TIER" in df.columns:
        fig, ax = plt.subplots(figsize=(8, 5))
        tiers = sorted(df["CONFIDENCE_TIER"].dropna().unique())
        rois = []
        for t in tiers:
            sub = df[df["CONFIDENCE_TIER"] == t]
            sc, pc = "STAKE_MODERATE", "PROFIT_MODERATE"
            if sc in sub.columns and sub[sc].sum() > 0:
                rois.append(sub[pc].sum() / sub[sc].sum())
            else:
                rois.append(np.nan)
        ax.bar([str(int(t)) for t in tiers], rois, color="#55A868", edgecolor="black")
        ax.axhline(0, ls="--", color="black")
        ax.set_xlabel("Confidence tier")
        ax.set_ylabel("Moderate profile ROI")
        ax.set_title("ROI by Calibrated Confidence Tier", fontweight="bold")
        _save_or_show(fig, save_path, "11_roi_by_confidence_tier.png")

    if "COVER_PROB_CALIBRATED" in df.columns and "DIRECTION" in df.columns:
        fig, ax = plt.subplots(figsize=(7, 7))
        bets = df[df["DIRECTION"] != "Pass"].copy()
        if not bets.empty:
            cover = bets["ACTUAL_MARGIN"] + bets["MARKET_SPREAD"]
            home = bets["DIRECTION"] == "Home"
            bets["won"] = (home & (cover > 0)) | (~home & (cover < 0))
            bins = np.linspace(0.45, 0.75, 8)
            bets["pb"] = pd.cut(bets["COVER_PROB_CALIBRATED"], bins=bins)
            cal = bets.groupby("pb", observed=True)["won"].mean()
            centers = [iv.mid for iv in cal.index]
            ax.plot([0, 1], [0, 1], ls="--", color="gray", label="Perfect")
            ax.scatter(centers, cal.values, s=80, color="#4C72B0", zorder=3)
            ax.set_xlabel("Calibrated cover probability")
            ax.set_ylabel("Actual cover rate")
            ax.set_title("ATS Reliability (Calibrated Confidence)", fontweight="bold")
            ax.legend()
        _save_or_show(fig, save_path, "12_confidence_reliability.png")

    # 12b · Cover-prob reliability by season (pooled panels)
    if "COVER_PROB_CALIBRATED" in df.columns and "DIRECTION" in df.columns:
        from pipeline.calibration_metrics import compute_ece, reliability_bins
        seasons = sorted(df["_season"].unique())
        n_panels = len(seasons) + 1
        ncols = min(3, n_panels)
        nrows = int(np.ceil(n_panels / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(4.5 * ncols, 4 * nrows), squeeze=False)
        panels = list(zip(seasons, [df[df["_season"] == s] for s in seasons])) + [("ALL", df)]
        for idx, (label, g) in enumerate(panels):
            ax = axes.flatten()[idx]
            bets = g[g["DIRECTION"] != "Pass"].copy()
            if bets.empty:
                ax.set_visible(False)
                continue
            cover = bets["ACTUAL_MARGIN"] + bets["MARKET_SPREAD"]
            home = bets["DIRECTION"] == "Home"
            y = ((home & (cover > 0)) | (~home & (cover < 0))).astype(int)
            p = bets["COVER_PROB_CALIBRATED"].astype(float)
            rb = reliability_bins(y, p, n_bins=8)
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            if not rb.empty:
                ax.plot(rb["pred_mean"], rb["obs_rate"], "o-", color="#8172B3", lw=1.5)
            ece = compute_ece(y, p)
            ax.set_title(f"{label}  ECE={ece:.3f}" if np.isfinite(ece) else str(label), fontsize=10)
            if not rb.empty:
                xlim, ylim = _autoscaled_lim_pair(
                    list(rb["pred_mean"]) + [0.5],
                    list(rb["obs_rate"]) + [0.5],
                    pad_frac=0.1,
                )
                if xlim:
                    ax.set_xlim(*xlim)
                if ylim:
                    ax.set_ylim(*ylim)
            else:
                ax.set_xlim(0, 1)
                ax.set_ylim(0, 1)
        for j in range(len(panels), len(axes.flatten())):
            axes.flatten()[j].set_visible(False)
        fig.suptitle("ATS Cover-Prob Reliability by Season", fontweight="bold")
        _save_or_show(fig, save_path, "12b_cover_prob_reliability_by_season.png")

    if save_path:
        print(f"📊 Betting plots saved → {save_path}/")


def favorite_loss_by_spread_bin(df: pd.DataFrame, bins=None) -> pd.DataFrame:
    """Favorite loss rate and MAE by |MARKET_SPREAD| bins."""
    from pipeline.upset_classifier import favorite_lost_label

    g = _norm_results(df)
    if g.empty or "MARKET_SPREAD" not in g.columns:
        return pd.DataFrame()
    bins = bins or [0, 3, 6, 9, 12, 99]
    rows = []
    mkt = pd.to_numeric(g["MARKET_SPREAD"], errors="coerce").abs()
    g = g.assign(_abs_mkt=mkt).dropna(subset=["_abs_mkt"])
    g["_bin"] = pd.cut(g["_abs_mkt"], bins=bins, right=False)
    for label, sub in g.groupby("_bin", observed=True):
        labs = [favorite_lost_label(r) for _, r in sub.iterrows()]
        labs = [x for x in labs if x is not None]
        if len(labs) < 10:
            continue
        win_prob = pd.to_numeric(sub.get("WIN_PROB", 0.5), errors="coerce").fillna(0.5)
        home_fav = pd.to_numeric(sub["MARKET_SPREAD"], errors="coerce") < 0
        win_prob_fav = np.where(home_fav, win_prob, 1.0 - win_prob)
        mae = np.nan
        if "SPREAD_ERR" in sub.columns:
            mae = float(pd.to_numeric(sub["SPREAD_ERR"], errors="coerce").abs().mean())
        elif "PRED_SPREAD" in sub.columns and "ACTUAL_MARGIN" in sub.columns:
            mae = float((sub["PRED_SPREAD"] - sub["ACTUAL_MARGIN"]).abs().mean())
        rows.append({
            "bin": str(label),
            "n": len(labs),
            "fav_loss_rate": float(np.mean(labs)),
            "mean_win_prob_fav": float(np.mean(win_prob_fav)),
            "mae": mae,
        })
    return pd.DataFrame(rows)


def blowout_accuracy_table(df: pd.DataFrame, thresholds=(10, 15, 20)) -> pd.DataFrame:
    """Compare realized |margin| rates to predicted blowout probabilities when present."""
    g = _norm_results(df)
    if g.empty or "ACTUAL_MARGIN" not in g.columns:
        return pd.DataFrame()
    abs_m = pd.to_numeric(g["ACTUAL_MARGIN"], errors="coerce").abs()
    rows = []
    for thr in thresholds:
        col = f"P_BLOWOUT_{int(thr)}"
        pred = pd.to_numeric(g[col], errors="coerce") if col in g.columns else pd.Series(np.nan, index=g.index)
        mask = abs_m.notna()
        actual = (abs_m[mask] >= float(thr)).astype(float)
        rows.append({
            "threshold": int(thr),
            "n": int(mask.sum()),
            "actual_rate": float(actual.mean()) if mask.any() else np.nan,
            "mean_pred_p": float(pred[mask].mean()) if pred[mask].notna().any() else np.nan,
        })
    return pd.DataFrame(rows)