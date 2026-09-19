"""Backtest metrics and betting edge grid search."""
from __future__ import annotations

import numpy as np
import pandas as pd

BREAKEVEN_ATS = 110.0 / 210.0


def variance_aware_edge_threshold(
    base_threshold: float,
    quantile_width: float,
    *,
    base_width: float = 20.0,
    scale: float = 0.05,
    max_bump: float = 2.5,
) -> float:
    """Require higher edge when quantile spread uncertainty is wide (#19)."""
    if not np.isfinite(quantile_width) or quantile_width <= base_width:
        return float(base_threshold)
    bump = min(max_bump, scale * (float(quantile_width) - base_width))
    return float(base_threshold) + bump


def _active_spread_bets(df):
    """Rows with a market line and a placed / analysis spread side.

    Prefer explicit ``DIRECTION != Pass`` so post-hoc edge grids that rewrite
    DIRECTION from EDGE still grade correctly under ``confidence_only`` exports
    (where ``ACTIONABLE`` is 0 for every row). Fall back to the mode-aware
    actionable frame when DIRECTION is all Pass.
    """
    if df is None or df.empty:
        return pd.DataFrame()
    m = df
    if "MARKET_SPREAD" in df.columns:
        m = df[df["MARKET_SPREAD"].notna()]
    if m.empty:
        return m
    if "DIRECTION" in m.columns:
        directed = m[m["DIRECTION"].astype(str).isin(("Home", "Away"))]
        if not directed.empty:
            return directed.copy()
    from pipeline.bet_selection import spread_bet_frame
    return spread_bet_frame(df)


def compute_clv(model_spread, bet_spread, closing_spread, side: str | None = None):
    """Bet-side point CLV (Task 038).

    Preferred: pass ``side`` (``Home``/``Away``) with ``bet_spread`` as the
    decision/T-60 home line and ``closing_spread`` as the close home line.

    Legacy callers pass ``(model_spread, bet_spread, closing_spread)`` without
    ``side``; the bet side is then inferred from ``sign(model + bet)``.
    Point CLV is never a multiplicative return — use ``PRICE_CLV`` separately.
    """
    from pipeline.bet_grading import bet_side_point_clv

    if pd.isna(bet_spread) or pd.isna(closing_spread):
        return np.nan
    if side is None:
        if pd.isna(model_spread):
            return np.nan
        edge = float(model_spread) + float(bet_spread)
        side = "Home" if edge >= 0 else "Away"
    return bet_side_point_clv(bet_spread, closing_spread, side)


def add_clv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Add point CLV and (when available) price CLV columns.

    Point CLV uses decision vs close home spreads and the bet DIRECTION.
    Prefers ``DECISION_SPREAD`` (T-60) when present so legacy conflation of
    MARKET_SPREAD==CLOSING_SPREAD does not force CLV to zero.
    Price/probability CLV is kept in a separate column and never compounded
    with point CLV via a multiplicative return identity (Task 028/038).
    """
    from pipeline.bet_grading import bet_side_point_clv, bet_side_price_clv

    out = df.copy()
    if "CLOSING_SPREAD" not in out.columns:
        out["CLOSING_SPREAD"] = out.get("MARKET_SPREAD", np.nan)
    if "DECISION_SPREAD" not in out.columns:
        # Prefer explicit decision/T-60; fall back to MARKET_SPREAD.
        if "decision_spread" in out.columns:
            out["DECISION_SPREAD"] = out["decision_spread"]
        else:
            out["DECISION_SPREAD"] = out.get("MARKET_SPREAD", np.nan)
    if all(c in out.columns for c in ("PRED_SPREAD", "ACTUAL_MARGIN")):
        close_for_resid = out["CLOSING_SPREAD"]
        out["MARKET_RESIDUAL"] = out["ACTUAL_MARGIN"] + close_for_resid
        out["MODEL_VS_CLOSE"] = out["PRED_SPREAD"] + close_for_resid
    if all(c in out.columns for c in ("DECISION_SPREAD", "CLOSING_SPREAD")):
        decision = out["DECISION_SPREAD"]
        close = out["CLOSING_SPREAD"]
        if "DIRECTION" in out.columns:
            sides = out["DIRECTION"].fillna("Pass")
        elif "EDGE" in out.columns:
            sides = np.where(out["EDGE"] > 0, "Home", np.where(out["EDGE"] < 0, "Away", "Pass"))
        else:
            sides = pd.Series(["Pass"] * len(out), index=out.index)
        point_vals = []
        for dec, clo, side in zip(decision, close, sides):
            if str(side) in ("Pass", "nan") or pd.isna(dec) or pd.isna(clo):
                point_vals.append(np.nan)
            elif float(dec) == float(clo):
                # Identical decision/close is conflation, not true zero CLV.
                point_vals.append(np.nan)
            else:
                point_vals.append(bet_side_point_clv(dec, clo, side))
        out["CLV"] = point_vals
        out["POINT_CLV"] = out["CLV"]
        if "DECISION_FAIR_PROB" in out.columns and "CLOSE_FAIR_PROB" in out.columns:
            out["PRICE_CLV"] = [
                bet_side_price_clv(d, c)
                for d, c in zip(out["DECISION_FAIR_PROB"], out["CLOSE_FAIR_PROB"])
            ]
    return out


def bootstrap_ci(values, n_boot: int = 1000, alpha: float = 0.05, stat="mean", seed: int = 42):
    """Bootstrap confidence interval for mean or rate.

    NOTE: when ``stat="mean"`` on per-bet profits this is a *mean-profit*
    interval, NOT an ROI interval. Use ``bootstrap_roi_ci`` for
    ``sum(profit)/sum(stake)`` with date/week block resampling (Task 042).
    """
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(seed)
    stats = []
    for _ in range(n_boot):
        sample = rng.choice(arr, size=len(arr), replace=True)
        stats.append(sample.mean() if stat == "mean" else np.mean(sample > 0))
    lo = float(np.quantile(stats, alpha / 2))
    hi = float(np.quantile(stats, 1 - alpha / 2))
    return float(np.mean(stats)), lo, hi


def bootstrap_roi_ci(
    profits,
    stakes,
    *,
    block_ids=None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 42,
):
    """Block-bootstrap CI for ``sum(profit)/sum(stake)`` (Task 042).

    ``block_ids`` groups bets into date/week blocks resampled as wholes.
    Returns ``(roi_point, lo, hi)``. Distinct from ``bootstrap_ci`` mean-profit.
    """
    profits = np.asarray(profits, dtype=float)
    stakes = np.asarray(stakes, dtype=float)
    mask = np.isfinite(profits) & np.isfinite(stakes) & (stakes > 0)
    profits, stakes = profits[mask], stakes[mask]
    if len(profits) < 5:
        return np.nan, np.nan, np.nan
    if block_ids is None:
        block_ids = np.arange(len(profits))
    else:
        block_ids = np.asarray(block_ids)[mask]
    unique_blocks = np.unique(block_ids)
    if len(unique_blocks) < 3:
        # Fall back to bet-level only when too few blocks — still label as ROI.
        unique_blocks = np.arange(len(profits))
        block_ids = unique_blocks
    rng = np.random.default_rng(seed)
    rois = []
    for _ in range(n_boot):
        sampled = rng.choice(unique_blocks, size=len(unique_blocks), replace=True)
        p_sum = 0.0
        s_sum = 0.0
        for b in sampled:
            sel = block_ids == b
            p_sum += float(profits[sel].sum())
            s_sum += float(stakes[sel].sum())
        if s_sum > 0:
            rois.append(p_sum / s_sum)
    if not rois:
        return np.nan, np.nan, np.nan
    point = float(np.sum(profits) / np.sum(stakes))
    lo = float(np.quantile(rois, alpha / 2))
    hi = float(np.quantile(rois, 1 - alpha / 2))
    return point, lo, hi


def assert_actionable_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Reject inconsistent actionable-bet rows (Task 043).

    Requires agreement among ACTIONABLE, DIRECTION, stake availability, and
    market presence for rows marked actionable.
    """
    if df is None or df.empty:
        return df
    required = {"DIRECTION", "MARKET_SPREAD"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"actionable frame missing columns: {sorted(missing)}")
    out = df.copy()
    if "ACTIONABLE" not in out.columns:
        out["ACTIONABLE"] = out["DIRECTION"].ne("Pass") & out["MARKET_SPREAD"].notna()
    bad = []
    for i, r in out.iterrows():
        actionable = bool(r.get("ACTIONABLE"))
        direction = r.get("DIRECTION", "Pass")
        has_market = pd.notna(r.get("MARKET_SPREAD"))
        if actionable:
            if direction in (None, "Pass", "nan") or not has_market:
                bad.append(i)
        if direction not in (None, "Pass", "nan", "Home", "Away"):
            bad.append(i)
        stake_cols = [c for c in out.columns if c.startswith("STAKE_")]
        for sc in stake_cols:
            stake = r.get(sc, 0) or 0
            if float(stake) > 0 and (direction in (None, "Pass") or not has_market):
                bad.append(i)
    if bad:
        raise ValueError(
            f"actionable-frame consistency failed for {len(set(bad))} rows "
            f"(e.g. index {next(iter(set(bad)))})"
        )
    return out


def ats_win_series(df):
    """Boolean series: active non-push spread bets that covered (Task 040).

    Pushes are excluded (NaN), not counted as losses.
    """
    from pipeline.bet_grading import grade_spread_bet

    d = _active_spread_bets(df)
    if d.empty:
        return pd.Series(dtype=bool)
    outcomes = [
        grade_spread_bet(m, s, side)
        for m, s, side in zip(d["ACTUAL_MARGIN"], d["MARKET_SPREAD"], d["DIRECTION"])
    ]
    result = pd.Series(
        [np.nan if o == "push" else (o == "win") for o in outcomes],
        index=d.index,
        dtype=float,
    )
    return result.dropna().astype(bool)


def clv_weighted_roi(df):
    """ROI weighted by positive CLV (rewards beating the close)."""
    d = add_clv_columns(_active_spread_bets(df))
    if d.empty or "CLV" not in d.columns:
        return np.nan
    wins = ats_win_series(d)
    if len(wins) == 0:
        return np.nan
    weights = np.clip(d.loc[wins.index, "CLV"].fillna(0) + 1.0, 0.5, 2.0)
    wp = (wins.astype(float) * weights).sum() / weights.sum()
    return wp * (100.0 / 110.0) - (1 - wp)


def sharpe_ratio(profits, eps: float = 1e-9) -> float:
    """Per-bet Sharpe (mean / std)."""
    arr = np.asarray(profits, dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 5:
        return float("nan")
    sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    if sd < eps:
        return float("nan")
    return float(arr.mean() / sd)


def max_drawdown(cumulative: np.ndarray) -> float:
    if len(cumulative) == 0:
        return float("nan")
    peak = np.maximum.accumulate(cumulative)
    dd = cumulative - peak
    return float(dd.min())


def walkforward_edge_threshold(
    prior_df,
    edges=(2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0),
    default=3.5,
    min_bets=40,
    optimize="bucket_roi",
    require_positive_ci=True,
    min_floor=None,
):
    """Pick ATS edge threshold from prior seasons.

    optimize: "roi" | "clv_roi" | "ats" | "bucket_roi"
    bucket_roi: maximize ROI with penalty for bets landing in the weak 2–4 pt bucket.
    require_positive_ci: reject thresholds whose bootstrap ROI 95% CI lower bound < break-even
    """
    if prior_df is None or prior_df.empty or "EDGE" not in prior_df.columns:
        return default
    d = prior_df[prior_df["MARKET_SPREAD"].notna()].copy()
    if d.empty:
        return default
    best, best_score = default, -1e9
    cover = d["ACTUAL_MARGIN"] + d["MARKET_SPREAD"]
    d = add_clv_columns(d)
    for t in edges:
        active = d["EDGE"].abs() >= t
        bets = d[active]
        if len(bets) < min_bets:
            continue
        home = bets["EDGE"] > 0
        c = cover[active]
        win = (home & (c > 0)) | (~home & (c < 0))
        wp = win.mean()
        roi = wp * (100.0 / 110.0) - (1 - wp)
        if optimize == "ats":
            score = wp
        elif optimize == "clv_roi":
            score = clv_weighted_roi(bets)
            if np.isnan(score):
                score = roi
        elif optimize == "bucket_roi":
            abs_e = bets["EDGE"].abs()
            weak_share = (abs_e < 4.0).mean()
            core = abs_e >= 4.0
            if core.sum() >= max(20, min_bets // 2):
                core_home = home[core]
                core_c = c[core]
                core_win = (core_home & (core_c > 0)) | (~core_home & (core_c < 0))
                core_wp = core_win.mean()
                core_roi = core_wp * (100.0 / 110.0) - (1 - core_wp)
                score = core_roi - 0.08 * weak_share
            else:
                score = roi - 0.10 * weak_share
        else:
            score = roi
        if require_positive_ci:
            wins_arr = win.astype(float).values
            _, lo, _ = bootstrap_ci(wins_arr, n_boot=500)
            if np.isfinite(lo) and lo < BREAKEVEN_ATS:
                continue
        if score > best_score or (abs(score - best_score) < 1e-9 and t > best):
            best_score, best = score, t
    if min_floor is not None and best < min_floor:
        best = float(min_floor)
    return best


def adaptive_min_confidence(
    scores,
    *,
    default: float | None = None,
    min_clear_frac: float = 0.08,
    target_frac: float | None = None,
    hard_floor: float | None = None,
) -> float:
    """Return a confidence gate that never undercuts the production floor.

    Historically this lowered the gate when scores were compressed (causing
    46 / 60 / 46 season swings). Production policy: keep ``default`` /
    ``MIN_CONFIDENCE_SCORE`` as a hard floor; only raise toward a high
    percentile when almost no scores clear the floor (signal to fix weights,
    not to bet mid-40s).
    """
    from pipeline.config import (
        CONFIDENCE_ADAPTIVE_FLOOR,
        CONFIDENCE_ADAPTIVE_TARGET_FRAC,
        CONFIDENCE_GATE_MAX_LIFT,
        MIN_CONFIDENCE_SCORE,
    )

    default = float(MIN_CONFIDENCE_SCORE if default is None else default)
    if not CONFIDENCE_ADAPTIVE_FLOOR:
        return default
    target_frac = float(
        CONFIDENCE_ADAPTIVE_TARGET_FRAC if target_frac is None else target_frac
    )
    arr = np.asarray(list(scores), dtype=float)
    arr = arr[np.isfinite(arr)]
    if len(arr) < 25:
        return default
    clear_frac = float((arr >= default).mean())
    if clear_frac >= min_clear_frac:
        return default
    # Scores compressed below floor — do NOT lower the gate; keep floor.
    # Optional slight raise if the top quantile is above the floor (rare).
    q = float(np.quantile(arr, max(0.0, min(1.0, 1.0 - target_frac))))
    ceiling = default + float(CONFIDENCE_GATE_MAX_LIFT)
    return float(max(default, min(ceiling, q if q >= default else default)))


def _clamp_confidence_gate(best_min: float, default_min: float) -> float:
    from pipeline.config import CONFIDENCE_GATE_MAX_LIFT, MIN_CONFIDENCE_SCORE

    floor = float(max(float(default_min), float(MIN_CONFIDENCE_SCORE)))
    ceiling = floor + float(CONFIDENCE_GATE_MAX_LIFT)
    return float(min(ceiling, max(floor, float(best_min))))


def walkforward_confidence_gate(
    prior_df,
    thresholds_min=None,
    thresholds_max: tuple[int | None, ...] = (None, 61, 62, 63),
    default_min: float | None = None,
    default_max: float | None = None,
    min_bets: int = 80,
    bet_calibrator=None,
    min_gated_bets: int = 100,
) -> tuple[float, float | None]:
    """Walk-forward min/max WIN_PCT band for actionable spread bets."""
    from pipeline.bet_selection import actionable_spread_frame, lean_spread_frame
    from pipeline.bet_confidence import _pick_confidence_band_gated_roi, _scored_ats_bets
    from pipeline.config import CONFIDENCE_GATE_MAX_LIFT, MAX_CONFIDENCE_SCORE, MIN_CONFIDENCE_SCORE

    if default_min is None:
        default_min = float(MIN_CONFIDENCE_SCORE)
    if default_max is None:
        default_max = MAX_CONFIDENCE_SCORE
    floor = float(max(float(default_min), float(MIN_CONFIDENCE_SCORE)))
    ceiling = floor + float(CONFIDENCE_GATE_MAX_LIFT)
    if thresholds_min is None:
        # Search only at/above the hard floor up to the lift ceiling.
        thresholds_min = tuple(
            t for t in (55, 56, 58, 60, 61, 62, 63) if floor <= t <= ceiling
        ) or (int(floor),)

    if prior_df is None or prior_df.empty:
        return floor, default_max

    d = actionable_spread_frame(prior_df)
    if d.empty:
        # Early seasons may not have ACTIONABLE populated consistently; fallback to leans.
        d = lean_spread_frame(prior_df)
    if d.empty:
        return floor, default_max

    score_pool: list[float] = []
    if bet_calibrator is not None and getattr(bet_calibrator, "_fitted", False):
        scored = _scored_ats_bets(bet_calibrator, d)
        score_pool = [float(cs) for _y, cs, _p in scored]
        if len(scored) >= min_bets:
            roi, best_min, best_max, best_n = _pick_confidence_band_gated_roi(
                scored,
                min_thresholds=thresholds_min,
                max_thresholds=thresholds_max,
                min_bets=min_bets,
            )
            if best_n >= min_bets and roi > -1e8:
                best_min = _clamp_confidence_gate(best_min, floor)
                if best_max is not None and default_max is not None:
                    best_max = min(float(best_max), float(default_max))
                gated_n = sum(
                    1 for _y, cs, _p in scored
                    if cs >= best_min and (best_max is None or cs <= best_max)
                )
                if gated_n >= min(min_gated_bets, max(40, int(0.15 * len(scored)))):
                    return best_min, best_max
                return floor, default_max

    # Fallback: min-only on stored WIN_PCT / CONFIDENCE
    best_min = walkforward_min_confidence(
        prior_df,
        thresholds=thresholds_min,
        default=floor,
        min_bets=min_bets,
        require_positive_ci=False,
        bet_calibrator=None,
    )
    if not score_pool:
        conf_col = "CONFIDENCE" if "CONFIDENCE" in d.columns else "WIN_PCT"
        if conf_col in d.columns:
            score_pool = pd.to_numeric(d[conf_col], errors="coerce").dropna().tolist()
    best_min = _clamp_confidence_gate(best_min, floor)
    return best_min, default_max


def walkforward_min_confidence(
    prior_df,
    thresholds=(50, 52, 54, 56, 58, 60, 62, 64, 66, 68, 70),
    default: float = 64,
    min_bets: int = 40,
    require_positive_ci: bool = True,
    bet_calibrator=None,
):
    """Pick minimum WIN_PCT / CONFIDENCE for ATS leans from prior seasons only.

    When *bet_calibrator* is fitted, leans are re-scored through the same isotonic
    path used at simulation time (avoids tuning on stale raw WIN_PCT).
    """
    from pipeline.bet_selection import lean_spread_frame, win_pct_column
    from pipeline.config import MIN_CONFIDENCE_SCORE

    if prior_df is None or prior_df.empty:
        return float(default)

    d = lean_spread_frame(prior_df)
    if d.empty:
        return float(MIN_CONFIDENCE_SCORE if default is None else default)

    if bet_calibrator is not None and getattr(bet_calibrator, "_fitted", False):
        from pipeline.bet_confidence import _pick_threshold_gated_roi, _scored_ats_bets

        scored = _scored_ats_bets(bet_calibrator, d)
        if len(scored) >= min_bets:
            _, _, best_thr, best_n = _pick_threshold_gated_roi(
                scored, thresholds=thresholds, min_bets=min_bets,
            )
            if best_n >= min_bets:
                return float(best_thr)
        if scored:
            best_thr = float(default)
            best_roi = -1e9
            for thr in thresholds:
                sub = [y for y, cs, _p in scored if cs >= thr]
                if len(sub) < min_bets:
                    continue
                wp = float(np.mean(sub))
                roi = wp * (100.0 / 110.0) - (1.0 - wp)
                if roi > best_roi:
                    best_roi, best_thr = roi, float(thr)
            if best_roi > -1e8:
                return best_thr

    score_col = win_pct_column(d)
    if score_col not in d.columns:
        return float(default)

    cover = d["ACTUAL_MARGIN"] + d["MARKET_SPREAD"]
    side = d["_side"]
    win = (side.eq("Home") & (cover > 0)) | (side.eq("Away") & (cover < 0))
    d = d.assign(_win=win.astype(float))

    best_thr = float(default)
    best_roi = -1e9
    for thr in thresholds:
        bets = d[pd.to_numeric(d[score_col], errors="coerce") >= thr]
        if len(bets) < min_bets:
            continue
        wp = float(bets["_win"].mean())
        roi = wp * (100.0 / 110.0) - (1 - wp)
        if require_positive_ci:
            _, lo, _ = bootstrap_ci(bets["_win"].astype(float).values, n_boot=500)
            if np.isfinite(lo) and lo < BREAKEVEN_ATS:
                continue
        if roi > best_roi:
            best_roi, best_thr = roi, float(thr)
    return best_thr


def apply_edge_threshold(df, threshold):
    """Re-mark spread DIRECTION/EDGE selection at a given edge threshold."""
    df = df.copy()
    e = df["EDGE"]
    active = e.abs() >= threshold
    df["DIRECTION"] = np.where(active, np.where(e > 0, "Home", "Away"), "Pass")
    df["EDGE_THRESHOLD"] = threshold
    return df


def grid_search_bet_edge(results_df, edges=(1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 5.5, 6.0)):
    """Find ATS edge threshold that maximizes ROI; re-derive DIRECTION from EDGE each step."""
    best_edge, best_roi = 2.5, -1.0
    rows = []
    df = results_df.copy()
    if "MARKET_SPREAD" not in df.columns:
        return best_edge, pd.DataFrame(rows)
    has_odds = df["MARKET_SPREAD"].notna()
    for edge in edges:
        sim = apply_edge_threshold(df[has_odds], edge)
        bets = sim[sim["DIRECTION"] != "Pass"]
        if len(bets) < 20:
            continue
        wins = ats_win_series(bets)
        win_pct = wins.mean()
        roi = win_pct * (100 / 110) - (1 - win_pct)
        rows.append({"edge": edge, "n_bets": len(bets), "win_pct": win_pct, "roi": roi})
        if roi > best_roi:
            best_roi, best_edge = roi, edge
    return best_edge, pd.DataFrame(rows)


def compare_selection_strategies(
    results_df,
    *,
    min_conf: float | None = None,
    confidence_min_edge: float | None = None,
    edge_thresholds: tuple[float, ...] = (5.5, 6.0),
) -> pd.DataFrame:
    """Apples-to-apples ROI table on one backtest export (no re-simulation)."""
    from pipeline.bet_selection import edge_scaled_min_confidence
    from pipeline.config import CONFIDENCE_MIN_EDGE, MIN_CONFIDENCE_SCORE

    if results_df is None or results_df.empty or "MARKET_SPREAD" not in results_df.columns:
        return pd.DataFrame()

    min_conf = float(MIN_CONFIDENCE_SCORE if min_conf is None else min_conf)
    confidence_min_edge = float(CONFIDENCE_MIN_EDGE if confidence_min_edge is None else confidence_min_edge)
    df = results_df[results_df["MARKET_SPREAD"].notna()].copy()
    # confidence_only leaves DIRECTION=Pass on non-actionable rows; rebuild leans for analysis.
    if "DIRECTION" in df.columns and (df["DIRECTION"].astype(str) == "Pass").mean() > 0.85:
        from pipeline.bet_selection import spread_side_series
        df["DIRECTION"] = spread_side_series(df)
    conf_col = "CONFIDENCE" if "CONFIDENCE" in df.columns else "WIN_PCT"
    edge_s = pd.to_numeric(df.get("EDGE"), errors="coerce").abs()
    rows = []

    def _row(mode: str, bets: pd.DataFrame) -> dict:
        if bets.empty:
            return {"mode": mode, "n_bets": 0, "win_pct": np.nan, "roi": np.nan}
        wins = ats_win_series(bets)
        if wins.empty:
            return {"mode": mode, "n_bets": 0, "win_pct": np.nan, "roi": np.nan}
        wp = float(wins.mean())
        return {
            "mode": mode,
            "n_bets": int(len(bets)),
            "win_pct": wp,
            "roi": float(wp * (100.0 / 110.0) - (1.0 - wp)),
        }

    if conf_col in df.columns:
        hybrid = df[
            (df["DIRECTION"] != "Pass")
            & (edge_s >= confidence_min_edge)
            & (pd.to_numeric(df[conf_col], errors="coerce") >= min_conf)
        ]
        rows.append(_row(
            f"confidence_hybrid|edge|>={confidence_min_edge:g}&conf>={min_conf:g}",
            hybrid,
        ))
        scaled_thr = edge_scaled_min_confidence(confidence_min_edge, base=min_conf)
        edge_scaled = df[
            (df["DIRECTION"] != "Pass")
            & (edge_s >= confidence_min_edge)
            & (
                pd.to_numeric(df[conf_col], errors="coerce")
                >= edge_s.map(lambda ae: edge_scaled_min_confidence(ae, base=min_conf))
            )
        ]
        rows.append(_row(
            f"confidence_edge_scaled|edge|>={confidence_min_edge:g}&scaled_conf>={scaled_thr:g}+",
            edge_scaled,
        ))

    for thr in edge_thresholds:
        sim = apply_edge_threshold(df, thr)
        edge_bets = sim[sim["DIRECTION"] != "Pass"]
        rows.append(_row(f"edge_only|edge|>={thr:g}", edge_bets))

    return pd.DataFrame(rows)


def walkforward_favorite_decimal(
    prior_df,
    candidates=(1.35, 1.40, 1.45, 1.50, 1.55, 1.60, 1.65, 1.70),
    default=None,
    min_bets=30,
):
    """Pick max favorite decimal odds for ML bets from prior seasons."""
    from pipeline.config import ML_MAX_FAVORITE_DECIMAL_DEFAULT
    if default is None:
        default = ML_MAX_FAVORITE_DECIMAL_DEFAULT
    if prior_df is None or prior_df.empty or "ML_DIRECTION" not in prior_df.columns:
        return default
    best, best_roi = default, -1e9
    for dec in candidates:
        bets = prior_df[prior_df["ML_DIRECTION"] != "Pass"].copy()
        if bets.empty:
            continue
        profits = []
        for _, r in bets.iterrows():
            ml = r["MARKET_ML"]
            if pd.isna(ml):
                continue
            d_home = 1 + ml / 100.0 if ml > 0 else 1 + 100.0 / abs(ml)
            d_away = 1 + (-ml) / 100.0 if -ml > 0 else 1 + 100.0 / abs(-ml)
            d = d_home if r["ML_DIRECTION"] == "Home" else d_away
            if d < dec:
                continue
            home_win = r["ACTUAL_HOME"] > r["ACTUAL_AWAY"]
            won = home_win if r["ML_DIRECTION"] == "Home" else not home_win
            profits.append((d - 1) if won else -1.0)
        if len(profits) < min_bets:
            continue
        wp = np.mean([p > 0 for p in profits])
        roi = np.mean(profits)
        if roi > best_roi:
            best_roi, best = roi, dec
    return best


def apply_ou_threshold(df, threshold):
    """Re-mark O/U direction from TOTAL_EDGE at a given points threshold."""
    df = df.copy()
    if "TOTAL_EDGE" not in df.columns:
        if {"PRED_TOTAL", "MARKET_TOTAL"}.issubset(df.columns):
            df["TOTAL_EDGE"] = df["PRED_TOTAL"] - df["MARKET_TOTAL"]
        else:
            df["OU_DIRECTION"] = "Pass"
            return df
    te = pd.to_numeric(df["TOTAL_EDGE"], errors="coerce")
    mkt = pd.to_numeric(df["MARKET_TOTAL"], errors="coerce")
    has_total = mkt.notna() & (mkt > 0)
    active = has_total & (te.abs() >= float(threshold))
    direction = pd.Series("Pass", index=df.index, dtype=object)
    direction.loc[active & (te > 0)] = "Over"
    direction.loc[active & (te < 0)] = "Under"
    df["OU_DIRECTION"] = direction
    if {"ACTUAL_HOME", "ACTUAL_AWAY", "MARKET_TOTAL"}.issubset(df.columns):
        from pipeline.bet_grading import grade_total_bet
        actual_total = df["ACTUAL_HOME"] + df["ACTUAL_AWAY"]
        mkt = pd.to_numeric(df["MARKET_TOTAL"], errors="coerce")
        ou_win = []
        for tot, line, direction in zip(actual_total, mkt, df["OU_DIRECTION"]):
            if direction == "Pass" or pd.isna(tot) or pd.isna(line):
                ou_win.append(np.nan)
                continue
            outcome = grade_total_bet(tot, line, direction)
            if outcome == "push":
                ou_win.append(np.nan)  # excluded from hit-rate (Task 040)
            else:
                ou_win.append(1.0 if outcome == "win" else 0.0)
        df["OU_WIN"] = ou_win
    df["OU_MIN_EDGE"] = threshold
    return df


def walkforward_ou_edge(
    prior_df,
    edges=(3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0),
    default=4.5,
    min_bets=40,
    require_positive_ci: bool = True,
):
    """Pick O/U edge threshold from prior seasons (re-derive all O/U bets each step)."""
    from pipeline.config import OU_MIN_EDGE, OU_REQUIRE_POSITIVE_CI

    if default is None:
        default = OU_MIN_EDGE
    if require_positive_ci is None:
        require_positive_ci = OU_REQUIRE_POSITIVE_CI
    if prior_df is None or prior_df.empty:
        return float(default)

    base = prior_df[prior_df["MARKET_TOTAL"].notna()].copy()
    if base.empty:
        return float(default)
    if "TOTAL_EDGE" not in base.columns and {"PRED_TOTAL", "MARKET_TOTAL"}.issubset(base.columns):
        base["TOTAL_EDGE"] = base["PRED_TOTAL"] - base["MARKET_TOTAL"]

    best, best_roi = float(default), -1e9
    for t in edges:
        sim = apply_ou_threshold(base, t)
        bets = sim[sim["OU_DIRECTION"] != "Pass"]
        if len(bets) < min_bets or "OU_WIN" not in bets.columns:
            continue
        wins = bets["OU_WIN"].astype(float)
        wp = float(wins.mean())
        roi = wp * (100.0 / 110.0) - (1 - wp)
        if require_positive_ci:
            _, lo, _ = bootstrap_ci(wins.values, n_boot=500)
            if np.isfinite(lo) and lo < BREAKEVEN_ATS:
                continue
        if roi > best_roi:
            best_roi, best = roi, float(t)
    return best


def compute_stake_profits(df, profile: str = "moderate"):
    """Add per-profile stake and profit columns from backtest results.

    Task 041: always apply slate/daily caps to stakes *before* computing
    profit. Task 040: pushes return zero profit and are not treated as losses.
    """
    from pipeline.bet_selection import edge_bucket_min_for_stakes, edge_stake_multiplier, uses_edge_gates
    from pipeline.bet_grading import exact_price_profit, grade_spread_bet
    from pipeline.stake_profiles import apply_daily_caps, compute_stake

    if df is None or df.empty:
        return df
    out = df.copy()
    stake_col = f"STAKE_{profile.upper()}"
    profit_col = f"PROFIT_{profile.upper()}"

    thr = float(out["EDGE_THRESHOLD"].iloc[0]) if "EDGE_THRESHOLD" in out.columns else 2.5
    edge_bucket_min = edge_bucket_min_for_stakes()

    # Fast path: stakes already present — still must cap before profit.
    if stake_col in out.columns:
        out = apply_daily_caps(out, stake_col, profile=profile)
        profits = []
        for _, r in out.iterrows():
            stake = float(r.get(stake_col, 0) or 0)
            direction = r.get("DIRECTION", "Pass")
            juice = r.get("SPREAD_PRICE", r.get("JUICE", -110))
            if juice is None or (isinstance(juice, float) and np.isnan(juice)):
                juice = -110
            if stake <= 0 or direction in (None, "Pass") or pd.isna(r.get("MARKET_SPREAD")):
                profits.append(0.0)
                continue
            outcome = grade_spread_bet(r["ACTUAL_MARGIN"], r["MARKET_SPREAD"], direction)
            profits.append(exact_price_profit(stake, outcome, juice))
        out[profit_col] = profits
        return out

    stakes = []
    for _, r in out.iterrows():
        direction = r.get("DIRECTION", "Pass")
        edge = float(r.get("EDGE", 0) or 0)
        cover_prob = float(r.get("COVER_PROB_CALIBRATED", r.get("WIN_PROB", 0.5)) or 0.5)
        tier = int(r.get("CONFIDENCE_TIER", 2) or 2)
        conf_score = r.get("CONFIDENCE")
        conf_score = int(conf_score) if pd.notna(conf_score) else None
        edge_mult = edge_stake_multiplier(abs(edge)) if uses_edge_gates() else 1.0
        vol_mult = float(r.get("VOL_STAKE_MULT", 1.0) or 1.0)
        juice = r.get("SPREAD_PRICE", r.get("JUICE", -110))
        if juice is None or (isinstance(juice, float) and np.isnan(juice)):
            juice = -110
        stake = compute_stake(
            profile,
            direction=direction,
            edge_pts=edge,
            edge_threshold=thr,
            cover_prob=cover_prob,
            confidence_tier=tier,
            conf_width=float(r.get("CONF_WIDTH", 24) or 24),
            rating_uncertainty=float(r.get("RATING_UNCERTAINTY", 350) or 350),
            juice=float(juice) if abs(float(juice)) >= 100 else -110,
            edge_stake_mult=edge_mult,
            edge_bucket_min=edge_bucket_min,
            volatility_mult=vol_mult,
            confidence_score=conf_score,
        )
        stakes.append(stake)
    out[stake_col] = stakes
    # Caps BEFORE profit (Task 041) — never the reverse.
    out = apply_daily_caps(out, stake_col, profile=profile)
    profits = []
    for _, r in out.iterrows():
        stake = float(r.get(stake_col, 0) or 0)
        direction = r.get("DIRECTION", "Pass")
        juice = r.get("SPREAD_PRICE", r.get("JUICE", -110))
        if juice is None or (isinstance(juice, float) and np.isnan(juice)):
            juice = -110
        if stake <= 0 or direction in (None, "Pass") or pd.isna(r.get("MARKET_SPREAD")):
            profits.append(0.0)
            continue
        outcome = grade_spread_bet(r["ACTUAL_MARGIN"], r["MARKET_SPREAD"], direction)
        profits.append(exact_price_profit(stake, outcome, juice))
    out[profit_col] = profits
    return out


def benchmark_betting_roi(df, profile: str = "moderate", n_boot: int = 1000):
    """Print ROI, drawdown, and bootstrap CI for a stake profile."""
    from pipeline.stake_profiles import PROFILES

    if df is None or df.empty:
        print(f"No results for profile {profile}.")
        return {}
    prof_col = f"PROFIT_{profile.upper()}"
    stake_col = f"STAKE_{profile.upper()}"
    if prof_col not in df.columns:
        df = compute_stake_profits(df, profile=profile)
    active = df[df[stake_col] > 0]
    if active.empty:
        print(f"Profile {profile}: no active bets.")
        return {"roi": np.nan, "n_bets": 0}
    profits = active[prof_col].values
    stakes = active[stake_col].values
    staked = stakes.sum()
    total_profit = profits.sum()
    roi = total_profit / staked if staked > 0 else np.nan
    cum = np.cumsum(profits)
    dd = float((cum - np.maximum.accumulate(cum)).min()) if len(cum) else 0.0
    # Hit rate excludes zero-profit pushes (profit==0 with stake>0 may also be
    # floating noise; grade-aware path already returns 0 for pushes).
    graded_wins = profits > 0
    graded_decided = profits != 0
    wp = float(graded_wins[graded_decided].mean()) if graded_decided.any() else np.nan
    sharpe = sharpe_ratio(profits)
    # Task 042: label mean-profit CI separately from true ROI CI.
    mean_profit, mean_lo, mean_hi = bootstrap_ci(profits, n_boot=n_boot, stat="mean")
    date_col = "DATE" if "DATE" in active.columns else ("game_date" if "game_date" in active.columns else None)
    block_ids = active[date_col].astype(str).values if date_col else None
    roi_point, roi_lo, roi_hi = bootstrap_roi_ci(
        profits, stakes, block_ids=block_ids, n_boot=n_boot,
    )
    print(f"\n--- Betting ROI ({profile}) ---")
    print(f"  Active bets: {len(active)}  win% (excl. pushes): {wp:.1%}" if np.isfinite(wp) else f"  Active bets: {len(active)}")
    print(f"  ROI: {roi:+.1%}  total profit (units): {total_profit:+.3f}")
    print(f"  Max drawdown (units): {dd:.3f}")
    if np.isfinite(sharpe):
        print(f"  Sharpe (per bet): {sharpe:.2f}")
    if np.isfinite(mean_lo):
        print(f"  Mean profit/bet 95% CI (NOT ROI): [{mean_lo:+.3f}, {mean_hi:+.3f}]")
    if np.isfinite(roi_lo):
        print(f"  ROI 95% CI (block bootstrap): [{roi_lo:+.1%}, {roi_hi:+.1%}]")
    if "CONFIDENCE_TIER" in active.columns:
        print("  By confidence tier:")
        for tier in sorted(active["CONFIDENCE_TIER"].dropna().unique()):
            sub = active[active["CONFIDENCE_TIER"] == tier]
            if len(sub) >= 5:
                sub_roi = sub[prof_col].sum() / sub[stake_col].sum()
                print(f"    tier {int(tier)}: n={len(sub)} roi={sub_roi:+.1%}")
    return {"roi": float(roi), "n_bets": len(active), "win_pct": float(wp),
            "max_drawdown": dd, "sharpe": sharpe,
            "mean_profit_ci_lo": mean_lo, "mean_profit_ci_hi": mean_hi,
            "roi_ci_lo": roi_lo, "roi_ci_hi": roi_hi,
            "roi_point": roi_point}


def add_all_profile_columns(df):
    """Add stake/profit columns for all three profiles."""
    out = df
    for p in ("conservative", "moderate", "aggressive"):
        out = compute_stake_profits(out, profile=p)
    return out


def portfolio_kelly_fractions(df, max_daily_exposure: float = 0.15, fractional: float = 0.25):
    """Cap correlated same-day Kelly stakes."""
    if df is None or df.empty or "KELLY_FRACTION" not in df.columns:
        return df
    out = df.copy()
    if "DATE" not in out.columns:
        out["ADJ_KELLY"] = out["KELLY_FRACTION"] * fractional
        return out
    out["ADJ_KELLY"] = 0.0
    for _, g in out.groupby("DATE"):
        k = g["KELLY_FRACTION"].fillna(0) * fractional
        total = k.sum()
        if total > max_daily_exposure:
            k = k * (max_daily_exposure / total)
        out.loc[g.index, "ADJ_KELLY"] = k
    return out


def monthly_holdout_metrics(results_df, season_col="simulated_season_window"):
    """Spread MAE and ATS by calendar month within each season."""
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    df = results_df.copy()
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    df["month"] = df["DATE"].dt.to_period("M").astype(str)
    rows = []
    for (season, month), g in df.groupby([season_col, "month"], dropna=False):
        mae = (g["PRED_SPREAD"] - g["ACTUAL_MARGIN"]).abs().mean()
        wins = ats_win_series(g)
        rows.append({
            "season": season, "month": month, "n_games": len(g),
            "spread_mae": mae, "ats_pct": wins.mean() if len(wins) else np.nan,
            "n_bets": len(wins),
        })
    return pd.DataFrame(rows)


def benchmark_results(results_df, n_boot: int = 1000):
    """Print spread MAE, ATS, CLV, bootstrap CIs."""
    if results_df.empty:
        print("No results to benchmark.")
        return
    df = add_clv_columns(results_df)
    mae = (df["PRED_SPREAD"] - df["ACTUAL_MARGIN"]).abs().mean()
    active = _active_spread_bets(df)
    if len(active):
        wins = ats_win_series(df)
        ats = wins.mean()
        _, lo, hi = bootstrap_ci(wins.astype(float), n_boot=n_boot)
    else:
        ats = lo = hi = float("nan")
    ml_acc = df.get("MODEL_ML_CORRECT", pd.Series(dtype=float)).mean()
    total_mae = df.get("TOTAL_ERR", pd.Series(dtype=float)).abs().mean()
    clv_mean = df.get("CLV", pd.Series(dtype=float)).mean()
    close_mae = np.nan
    if "CLOSING_SPREAD" in df.columns and df["CLOSING_SPREAD"].notna().any():
        mkt = df[df["CLOSING_SPREAD"].notna()]
        close_mae = (-mkt["CLOSING_SPREAD"] - mkt["ACTUAL_MARGIN"]).abs().mean()
    print(f"Spread MAE: {mae:.2f}")
    if np.isfinite(close_mae):
        print(f"Closing line MAE: {close_mae:.2f}  (model vs close: {mae - close_mae:+.2f})")
    print(f"ATS win% (active bets): {ats:.1%} ({len(active)} bets)")
    if np.isfinite(lo):
        print(f"  95% CI: [{lo:.1%}, {hi:.1%}]  (breakeven {BREAKEVEN_ATS:.1%})")
    if np.isfinite(clv_mean):
        print(f"Mean CLV (pts): {clv_mean:+.3f}")
    print(f"ML winner accuracy (diag): {ml_acc:.1%}")
    print(f"Total MAE (diag): {total_mae:.2f}")


def print_accuracy_layers(results_df) -> None:
    """Spread MAE vs all-leans ATS vs actionable ATS (model vs policy quality)."""
    from pipeline.bet_selection import actionable_spread_frame, lean_spread_frame

    if results_df is None or results_df.empty:
        print("No results for accuracy layers.")
        return
    spread_mae = (
        pd.to_numeric(results_df["PRED_SPREAD"], errors="coerce")
        - pd.to_numeric(results_df["ACTUAL_MARGIN"], errors="coerce")
    ).abs().mean()
    leans = lean_spread_frame(results_df)
    if not leans.empty:
        leans = leans.copy()
        # Grade on lean side even when confidence_only left DIRECTION=Pass.
        if "_side" in leans.columns:
            leans["DIRECTION"] = leans["_side"]
    act = actionable_spread_frame(results_df)
    lean_w = float(ats_win_series(leans).mean()) if len(leans) else float("nan")
    act_w = float(ats_win_series(act).mean()) if len(act) else float("nan")
    lean_roi = float(lean_w * (100.0 / 110.0) - (1.0 - lean_w)) if np.isfinite(lean_w) else float("nan")
    act_roi = float(act_w * (100.0 / 110.0) - (1.0 - act_w)) if np.isfinite(act_w) else float("nan")
    print("\nAccuracy layers (spread model vs selection policy):")
    print(f"  1. Spread MAE:        {spread_mae:.2f}")
    print(f"  2. All-leans ATS:     {lean_w:.1%}  ({len(leans):,} games)  flat ROI {lean_roi:+.1%}")
    print(f"  3. Actionable ATS:    {act_w:.1%}  ({len(act):,} bets)   flat ROI {act_roi:+.1%}")


def compute_metrics(results_df: pd.DataFrame, season_col="simulated_season_window") -> pd.DataFrame:
    """Per-season + overall metric table for one backtest result frame."""
    if results_df is None or results_df.empty:
        return pd.DataFrame()
    df = results_df.copy()
    df["_spread_ae"] = (df["PRED_SPREAD"] - df["ACTUAL_MARGIN"]).abs()
    if "RAW_PRED_MARGIN" in df.columns:
        df["_raw_spread_ae"] = (df["RAW_PRED_MARGIN"] - df["ACTUAL_MARGIN"]).abs()
    if "PRED_TOTAL" in df and "ACTUAL_HOME" in df:
        df["_total_ae"] = (df["PRED_TOTAL"] - (df["ACTUAL_HOME"] + df["ACTUAL_AWAY"])).abs()
    else:
        df["_total_ae"] = np.nan
    if "PRED_HOME" in df.columns and "ACTUAL_HOME" in df.columns:
        df["_home_ae"] = (
            pd.to_numeric(df["PRED_HOME"], errors="coerce")
            - pd.to_numeric(df["ACTUAL_HOME"], errors="coerce")
        ).abs()
        df["_home_resid"] = (
            pd.to_numeric(df["ACTUAL_HOME"], errors="coerce")
            - pd.to_numeric(df["PRED_HOME"], errors="coerce")
        )
    else:
        df["_home_ae"] = np.nan
        df["_home_resid"] = np.nan
    if "PRED_AWAY" in df.columns and "ACTUAL_AWAY" in df.columns:
        df["_away_ae"] = (
            pd.to_numeric(df["PRED_AWAY"], errors="coerce")
            - pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")
        ).abs()
        df["_away_resid"] = (
            pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")
            - pd.to_numeric(df["PRED_AWAY"], errors="coerce")
        )
    else:
        df["_away_ae"] = np.nan
        df["_away_resid"] = np.nan
    if "_home_ae" in df.columns and "_away_ae" in df.columns:
        df["_paired_ae"] = 0.5 * (df["_home_ae"] + df["_away_ae"])
        df["_paired_se"] = 0.5 * (
            (pd.to_numeric(df.get("PRED_HOME"), errors="coerce")
             - pd.to_numeric(df.get("ACTUAL_HOME"), errors="coerce")) ** 2
            + (pd.to_numeric(df.get("PRED_AWAY"), errors="coerce")
               - pd.to_numeric(df.get("ACTUAL_AWAY"), errors="coerce")) ** 2
        )
    else:
        df["_paired_ae"] = np.nan
        df["_paired_se"] = np.nan
    # Algebra consistency: | (H-A) - PRED_SPREAD | and | (H+A) - PRED_TOTAL |
    if {"PRED_HOME", "PRED_AWAY", "PRED_SPREAD"}.issubset(df.columns):
        df["_margin_algebra_err"] = (
            (pd.to_numeric(df["PRED_HOME"], errors="coerce")
             - pd.to_numeric(df["PRED_AWAY"], errors="coerce"))
            - pd.to_numeric(df["PRED_SPREAD"], errors="coerce")
        ).abs()
    else:
        df["_margin_algebra_err"] = np.nan
    if {"PRED_HOME", "PRED_AWAY", "PRED_TOTAL"}.issubset(df.columns):
        df["_total_algebra_err"] = (
            (pd.to_numeric(df["PRED_HOME"], errors="coerce")
             + pd.to_numeric(df["PRED_AWAY"], errors="coerce"))
            - pd.to_numeric(df["PRED_TOTAL"], errors="coerce")
        ).abs()
    else:
        df["_total_algebra_err"] = np.nan
    home_win = (df["ACTUAL_MARGIN"] > 0).astype(int)
    df["_brier"] = (df.get("WIN_PROB", 0.5) - home_win) ** 2
    try:
        from pipeline.calibration_metrics import compute_ece as _ece
    except ImportError:
        _ece = None

    def _one(g):
        active = g[g["DIRECTION"] != "Pass"]
        if len(active):
            wins = (
                ((active["DIRECTION"] == "Home") & (active["ACTUAL_MARGIN"] + active["MARKET_SPREAD"] > 0))
                | ((active["DIRECTION"] == "Away") & (active["ACTUAL_MARGIN"] + active["MARKET_SPREAD"] < 0))
            )
            ats = wins.mean()
            roi = ats * (100.0 / 110.0) - (1 - ats)
            n_bets = int(len(active))
        else:
            ats = roi = np.nan
            n_bets = 0
        raw_mae = g["_raw_spread_ae"].mean() if "_raw_spread_ae" in g.columns else np.nan
        ece = np.nan
        if _ece is not None and "WIN_PROB" in g.columns:
            try:
                p = pd.to_numeric(g["WIN_PROB"], errors="coerce")
                y = (g["ACTUAL_MARGIN"] > 0).astype(int)
                mask = p.notna()
                if mask.sum() >= 20:
                    ece = float(_ece(y[mask].values, p[mask].values))
            except Exception:
                ece = np.nan
        market_spread_mae = np.nan
        market_total_mae = np.nan
        model_minus_market_spread = np.nan
        if "MARKET_SPREAD" in g.columns:
            mkt_spread_err = (
                -pd.to_numeric(g["MARKET_SPREAD"], errors="coerce")
                - pd.to_numeric(g["ACTUAL_MARGIN"], errors="coerce")
            ).abs()
            market_spread_mae = float(mkt_spread_err.mean())
            if np.isfinite(market_spread_mae) and np.isfinite(g["_spread_ae"].mean()):
                model_minus_market_spread = float(g["_spread_ae"].mean() - market_spread_mae)
        if "MARKET_TOTAL" in g.columns and "ACTUAL_HOME" in g.columns:
            actual_tot = (
                pd.to_numeric(g["ACTUAL_HOME"], errors="coerce")
                + pd.to_numeric(g["ACTUAL_AWAY"], errors="coerce")
            )
            market_total_mae = float(
                (pd.to_numeric(g["MARKET_TOTAL"], errors="coerce") - actual_tot).abs().mean()
            )
        calib_slope = calib_intercept = log_loss = np.nan
        try:
            from pipeline.calibration_metrics import (
                compute_log_loss as _ll,
                calibration_slope_intercept as _csi,
            )
            if "WIN_PROB" in g.columns:
                p = pd.to_numeric(g["WIN_PROB"], errors="coerce")
                y = (g["ACTUAL_MARGIN"] > 0).astype(int)
                mask = p.notna()
                if int(mask.sum()) >= 30:
                    log_loss = float(_ll(y[mask].values, p[mask].values))
                    slope_int = _csi(y[mask].values, p[mask].values)
                    calib_slope = slope_int.get("slope", np.nan)
                    calib_intercept = slope_int.get("intercept", np.nan)
        except Exception:
            pass
        n_clv = 0
        if "CLV" in g.columns:
            n_clv = int(pd.to_numeric(g["CLV"], errors="coerce").notna().sum())
        interval_coverage = np.nan
        mean_conf_width = np.nan
        margin_dispersion_ratio = np.nan
        if {"CONF_LOWER", "CONF_UPPER", "ACTUAL_MARGIN"}.issubset(g.columns):
            lo = pd.to_numeric(g["CONF_LOWER"], errors="coerce")
            hi = pd.to_numeric(g["CONF_UPPER"], errors="coerce")
            act = pd.to_numeric(g["ACTUAL_MARGIN"], errors="coerce")
            m = lo.notna() & hi.notna() & act.notna()
            if int(m.sum()) >= 20:
                interval_coverage = float(((act[m] >= lo[m]) & (act[m] <= hi[m])).mean())
                mean_conf_width = float((hi[m] - lo[m]).mean())
        # Nominal 50/80/95 coverage from quantile columns when present.
        coverage_50 = coverage_80 = coverage_95 = np.nan
        home_cov_50 = home_cov_80 = home_cov_95 = np.nan
        away_cov_50 = away_cov_80 = away_cov_95 = np.nan
        total_cov_50 = total_cov_80 = total_cov_95 = np.nan

        def _cov_from_cols(act_s, lo_name, hi_name, fallback_lo=None, fallback_hi=None):
            lo_c = lo_name if lo_name in g.columns else fallback_lo
            hi_c = hi_name if hi_name in g.columns else fallback_hi
            if lo_c is None or hi_c is None or lo_c not in g.columns or hi_c not in g.columns:
                return np.nan, np.nan
            lo = pd.to_numeric(g[lo_c], errors="coerce")
            hi = pd.to_numeric(g[hi_c], errors="coerce")
            mm = lo.notna() & hi.notna() & act_s.notna()
            if int(mm.sum()) < 20:
                return np.nan, np.nan
            cov = float(((act_s[mm] >= lo[mm]) & (act_s[mm] <= hi[mm])).mean())
            width = float((hi[mm] - lo[mm]).mean())
            return cov, width

        if "ACTUAL_MARGIN" in g.columns:
            act = pd.to_numeric(g["ACTUAL_MARGIN"], errors="coerce")
            for lo_name, hi_name, dest, fb_lo, fb_hi in (
                ("MARGIN_QLO_50", "MARGIN_QHI_50", "coverage_50", "SPREAD_Q25", "SPREAD_Q75"),
                ("MARGIN_QLO_80", "MARGIN_QHI_80", "coverage_80", "SPREAD_Q10", "SPREAD_Q90"),
                ("MARGIN_QLO_95", "MARGIN_QHI_95", "coverage_95", "CONF_LOWER", "CONF_UPPER"),
            ):
                # Also accept lowercase aliases from in-memory preds frames.
                alt = {
                    "MARGIN_QLO_50": "margin_qlo_50", "MARGIN_QHI_50": "margin_qhi_50",
                    "MARGIN_QLO_80": "margin_qlo_80", "MARGIN_QHI_80": "margin_qhi_80",
                    "MARGIN_QLO_95": "margin_qlo_95", "MARGIN_QHI_95": "margin_qhi_95",
                    "SPREAD_Q25": "spread_q25", "SPREAD_Q75": "spread_q75",
                    "SPREAD_Q10": "spread_q10", "SPREAD_Q90": "spread_q90",
                }
                lo_try = lo_name if lo_name in g.columns else alt.get(lo_name)
                hi_try = hi_name if hi_name in g.columns else alt.get(hi_name)
                fb_lo_try = fb_lo if fb_lo in g.columns else alt.get(fb_lo, fb_lo)
                fb_hi_try = fb_hi if fb_hi in g.columns else alt.get(fb_hi, fb_hi)
                cov, width = _cov_from_cols(act, lo_try or lo_name, hi_try or hi_name, fb_lo_try, fb_hi_try)
                if not np.isfinite(cov):
                    continue
                if dest == "coverage_50":
                    coverage_50 = cov
                elif dest == "coverage_80":
                    coverage_80 = cov
                    if not np.isfinite(interval_coverage):
                        interval_coverage = cov
                        mean_conf_width = width
                else:
                    coverage_95 = cov
        if "ACTUAL_HOME" in g.columns:
            ah = pd.to_numeric(g["ACTUAL_HOME"], errors="coerce")
            home_cov_50, _ = _cov_from_cols(ah, "HOME_QLO_50" if "HOME_QLO_50" in g.columns else "home_qlo_50",
                                            "HOME_QHI_50" if "HOME_QHI_50" in g.columns else "home_qhi_50")
            home_cov_80, _ = _cov_from_cols(ah, "HOME_QLO_80" if "HOME_QLO_80" in g.columns else "home_qlo_80",
                                            "HOME_QHI_80" if "HOME_QHI_80" in g.columns else "home_qhi_80")
            home_cov_95, _ = _cov_from_cols(ah, "HOME_QLO_95" if "HOME_QLO_95" in g.columns else "home_qlo_95",
                                            "HOME_QHI_95" if "HOME_QHI_95" in g.columns else "home_qhi_95")
        if "ACTUAL_AWAY" in g.columns:
            aa = pd.to_numeric(g["ACTUAL_AWAY"], errors="coerce")
            away_cov_50, _ = _cov_from_cols(aa, "AWAY_QLO_50" if "AWAY_QLO_50" in g.columns else "away_qlo_50",
                                            "AWAY_QHI_50" if "AWAY_QHI_50" in g.columns else "away_qhi_50")
            away_cov_80, _ = _cov_from_cols(aa, "AWAY_QLO_80" if "AWAY_QLO_80" in g.columns else "away_qlo_80",
                                            "AWAY_QHI_80" if "AWAY_QHI_80" in g.columns else "away_qhi_80")
            away_cov_95, _ = _cov_from_cols(aa, "AWAY_QLO_95" if "AWAY_QLO_95" in g.columns else "away_qlo_95",
                                            "AWAY_QHI_95" if "AWAY_QHI_95" in g.columns else "away_qhi_95")
        if "ACTUAL_TOTAL" in g.columns or (
            "ACTUAL_HOME" in g.columns and "ACTUAL_AWAY" in g.columns
        ):
            if "ACTUAL_TOTAL" in g.columns:
                at = pd.to_numeric(g["ACTUAL_TOTAL"], errors="coerce")
            else:
                at = (
                    pd.to_numeric(g["ACTUAL_HOME"], errors="coerce")
                    + pd.to_numeric(g["ACTUAL_AWAY"], errors="coerce")
                )
            total_cov_50, _ = _cov_from_cols(at, "TOTAL_QLO_50" if "TOTAL_QLO_50" in g.columns else "total_qlo_50",
                                             "TOTAL_QHI_50" if "TOTAL_QHI_50" in g.columns else "total_qhi_50")
            total_cov_80, _ = _cov_from_cols(at, "TOTAL_QLO_80" if "TOTAL_QLO_80" in g.columns else "total_qlo_80",
                                             "TOTAL_QHI_80" if "TOTAL_QHI_80" in g.columns else "total_qhi_80")
            total_cov_95, _ = _cov_from_cols(at, "TOTAL_QLO_95" if "TOTAL_QLO_95" in g.columns else "total_qlo_95",
                                             "TOTAL_QHI_95" if "TOTAL_QHI_95" in g.columns else "total_qhi_95")
        # Prefer 80% margin coverage as the primary interval_coverage target.
        if np.isfinite(coverage_80):
            interval_coverage = coverage_80
        if "PRED_SPREAD" in g.columns and "MARKET_SPREAD" in g.columns:
            pred = pd.to_numeric(g["PRED_SPREAD"], errors="coerce")
            mkt = -pd.to_numeric(g["MARKET_SPREAD"], errors="coerce")
            mm = pred.notna() & mkt.notna()
            if int(mm.sum()) >= 20 and float(mkt[mm].abs().mean()) > 1e-9:
                margin_dispersion_ratio = float(pred[mm].abs().mean() / mkt[mm].abs().mean())
        home_mae = float(g["_home_ae"].mean()) if "_home_ae" in g.columns else np.nan
        away_mae = float(g["_away_ae"].mean()) if "_away_ae" in g.columns else np.nan
        paired_mae = float(g["_paired_ae"].mean()) if "_paired_ae" in g.columns else np.nan
        paired_rmse = float(np.sqrt(g["_paired_se"].mean())) if "_paired_se" in g.columns else np.nan
        home_bias = away_bias = resid_corr = np.nan
        if "_home_resid" in g.columns and "_away_resid" in g.columns:
            hr = pd.to_numeric(g["_home_resid"], errors="coerce")
            ar = pd.to_numeric(g["_away_resid"], errors="coerce")
            rm = hr.notna() & ar.notna()
            if int(rm.sum()) >= 20:
                home_bias = float(hr[rm].mean())
                away_bias = float(ar[rm].mean())
                if float(hr[rm].std()) > 1e-9 and float(ar[rm].std()) > 1e-9:
                    resid_corr = float(hr[rm].corr(ar[rm]))
        market_err_corr = np.nan
        margin_slope = np.nan
        within2 = np.nan
        if "PRED_SPREAD" in g.columns and "MARKET_SPREAD" in g.columns and "ACTUAL_MARGIN" in g.columns:
            pred = pd.to_numeric(g["PRED_SPREAD"], errors="coerce")
            mkt = -pd.to_numeric(g["MARKET_SPREAD"], errors="coerce")
            act = pd.to_numeric(g["ACTUAL_MARGIN"], errors="coerce")
            edge = pred - mkt
            realized = act - mkt
            mm = edge.notna() & realized.notna()
            if int(mm.sum()) >= 30:
                market_err_corr = float(edge[mm].corr(realized[mm]))
            mm2 = pred.notna() & mkt.notna()
            if int(mm2.sum()) >= 30 and float(mkt[mm2].std()) > 1e-9:
                margin_slope = float(np.polyfit(mkt[mm2], pred[mm2], 1)[0])
                within2 = float((pred[mm2].abs() <= 2.0).mean())
        algebra_ok = np.nan
        if "_margin_algebra_err" in g.columns and "_total_algebra_err" in g.columns:
            algebra_ok = float(
                ((g["_margin_algebra_err"] < 1e-3) & (g["_total_algebra_err"] < 1e-3)).mean()
            )
        return pd.Series({
            "n_games": int(len(g)),
            "spread_mae": g["_spread_ae"].mean(),
            "raw_spread_mae": raw_mae,
            "total_mae": g["_total_ae"].mean(),
            "home_mae": home_mae,
            "away_mae": away_mae,
            "paired_score_mae": paired_mae,
            "paired_score_rmse": paired_rmse,
            "home_resid_bias": home_bias,
            "away_resid_bias": away_bias,
            "home_away_resid_corr": resid_corr,
            "market_spread_mae": market_spread_mae,
            "market_total_mae": market_total_mae,
            "model_minus_market_spread_mae": model_minus_market_spread,
            "margin_dispersion_ratio": margin_dispersion_ratio,
            "margin_vs_market_slope": margin_slope,
            "margin_within2_pct": within2,
            "market_error_corr": market_err_corr,
            "score_algebra_ok_pct": algebra_ok,
            "interval_coverage": interval_coverage,
            "interval_coverage_50": coverage_50,
            "interval_coverage_80": coverage_80,
            "interval_coverage_95": coverage_95,
            "home_coverage_50": home_cov_50,
            "home_coverage_80": home_cov_80,
            "home_coverage_95": home_cov_95,
            "away_coverage_50": away_cov_50,
            "away_coverage_80": away_cov_80,
            "away_coverage_95": away_cov_95,
            "total_coverage_50": total_cov_50,
            "total_coverage_80": total_cov_80,
            "total_coverage_95": total_cov_95,
            "mean_conf_width": mean_conf_width,
            "ats_pct": ats,
            "roi": roi,
            "n_bets": n_bets,
            "brier": g["_brier"].mean(),
            "ece": ece,
            "log_loss": log_loss,
            "calib_slope": calib_slope,
            "calib_intercept": calib_intercept,
            "n_finite_clv": n_clv,
        })

    rows = []
    for s, g in df.groupby(season_col):
        rec = {"season": s}
        rec.update(_one(g).to_dict())
        rows.append(rec)
    overall = {"season": "ALL"}
    overall.update(_one(df).to_dict())
    rows.append(overall)
    return pd.DataFrame(rows)
