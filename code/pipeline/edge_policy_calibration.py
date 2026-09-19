"""Post-hoc min-|edge| policy calibration (policy_tuning only).

Re-thresholds walk-forward backtest rows from model lean (EDGE / EDGE_LEAN)
without re-running the suite or mutating live ``MIN_EDGE_BUCKET``.

Scoreboard order for recommendations: n_bets + mean CLV (decision≠close) →
ATS/flat ROI. Never promote a floor solely because it maximizes volume.
"""
from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from pipeline.bet_selection import (
    normalize_edge_avoid_bands,
    passes_confidence_actionable_gates,
    passes_edge_avoid_band,
    spread_side_series,
)
from pipeline.config import (
    EDGE_AVOID_BAND,
    EDGE_AVOID_BAND_MIN_ELO_AGREE,
    MAX_CONFIDENCE_SCORE,
    MIN_CONFIDENCE_SCORE,
    MIN_EDGE_BUCKET,
    WALKFORWARD_EDGE_MIN_FLOOR,
)
from pipeline.metrics import BREAKEVEN_ATS, add_clv_columns, apply_edge_threshold, ats_win_series

DEFAULT_EDGE_GRID = (
    3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5, 7.0,
)


def _season_col(df: pd.DataFrame) -> str | None:
    for c in ("simulated_season_window", "season", "SEASON"):
        if c in df.columns:
            return c
    return None


def policy_lean_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Lined games with a model lean — includes sub-threshold |edge| rows."""
    if df is None or df.empty or "MARKET_SPREAD" not in df.columns or "EDGE" not in df.columns:
        return pd.DataFrame()
    m = df[pd.to_numeric(df["MARKET_SPREAD"], errors="coerce").notna()].copy()
    if m.empty:
        return m
    side = spread_side_series(m)
    m = m[side.isin(("Home", "Away"))].copy()
    m["_side"] = side.loc[m.index]
    m["abs_edge"] = pd.to_numeric(m["EDGE"], errors="coerce").abs()
    m = m[m["abs_edge"].notna() & (m["abs_edge"] > 0)]
    return m


def _row_gate_ok(
    row: pd.Series,
    *,
    min_edge: float,
    use_avoid_band: bool,
    full_gates: bool,
) -> bool:
    """True if this lean would be actionable at ``min_edge``.

    ``full_gates=True`` (default): reuse live confidence / width / trust / phantom
    / avoid-band helpers so only the edge floor is varied.
    ``full_gates=False``: |edge| (+ optional avoid band) only — upper-bound volume.
    """
    edge = float(row["EDGE"])
    lean = str(row.get("_side", "Pass"))
    if lean not in ("Home", "Away"):
        return False
    if abs(edge) < float(min_edge):
        return False
    agree = row.get("ELO_META_AGREEMENT", np.nan)
    agree_f = None if pd.isna(agree) else float(agree)
    if not use_avoid_band:
        # Temporarily ignore dead-zone band by treating agreement as high.
        agree_f = 1.0
    if not full_gates:
        return passes_edge_avoid_band(abs(edge), elo_meta_agreement=agree_f) if use_avoid_band else True

    conf = row.get("CONFIDENCE", row.get("WIN_PCT", np.nan))
    width = row.get("CONF_WIDTH", row.get("SPREAD_QUANTILE_WIDTH", 0.0))
    trust = row.get("DISAGREEMENT_TRUST", 1.0)
    phantom = bool(row.get("PHANTOM_INJURY_FLAG", False))
    mkt = row.get("MARKET_SPREAD", np.nan)
    if pd.isna(conf):
        conf = 0.0
    if pd.isna(width):
        width = 0.0
    if pd.isna(trust):
        trust = 1.0
    return passes_confidence_actionable_gates(
        lean=lean,
        edge_pts=edge,
        conf_score=float(conf),
        conf_width=float(width),
        min_confidence=float(MIN_CONFIDENCE_SCORE),
        max_confidence=MAX_CONFIDENCE_SCORE,
        disagreement_trust=float(trust),
        phantom_injury_flag=phantom,
        market_spread=None if pd.isna(mkt) else float(mkt),
        elo_meta_agreement=agree_f,
        min_edge=float(min_edge),
    )


def _select_bets(
    work: pd.DataFrame,
    *,
    min_edge: float,
    use_avoid_band: bool,
    full_gates: bool,
) -> pd.DataFrame:
    if work.empty:
        return work
    keep = [
        _row_gate_ok(
            row,
            min_edge=min_edge,
            use_avoid_band=use_avoid_band,
            full_gates=full_gates,
        )
        for _, row in work.iterrows()
    ]
    return work.loc[np.asarray(keep)].copy()


def _score_bets(bets: pd.DataFrame) -> dict[str, Any]:
    """Grade one candidate selection set."""
    if bets.empty:
        return {
            "n_bets": 0,
            "n_graded": 0,
            "ats_pct": np.nan,
            "flat_roi": np.nan,
            "mean_clv": np.nan,
            "n_clv": 0,
            "median_abs_edge": np.nan,
        }
    sim = bets.copy()
    sim["DIRECTION"] = sim["_side"].astype(str)
    wins = ats_win_series(sim)
    n_graded = int(len(wins))
    ats = float(wins.mean()) if n_graded else np.nan
    roi = float(ats * (100.0 / 110.0) - (1.0 - ats)) if n_graded else np.nan

    scored = add_clv_columns(sim)
    clv = pd.to_numeric(scored.get("CLV", pd.Series(dtype=float)), errors="coerce")
    # Decision==close → NaN by design; only finite CLV counts.
    finite = clv[np.isfinite(clv)]
    return {
        "n_bets": int(len(bets)),
        "n_graded": n_graded,
        "ats_pct": ats,
        "flat_roi": roi,
        "mean_clv": float(finite.mean()) if len(finite) else np.nan,
        "n_clv": int(len(finite)),
        "median_abs_edge": float(bets["abs_edge"].median()),
    }


def edge_policy_grid(
    df: pd.DataFrame,
    *,
    thresholds: Sequence[float] | None = None,
    use_avoid_band: bool = True,
    full_gates: bool = True,
    min_bets: int = 20,
    seasons: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """Grid min-|edge| floors on lean rows; report volume / ATS / CLV.

    Does not mutate config. Rows already filtered to ``seasons`` when provided.
    Default ``full_gates=True`` keeps confidence/width/trust/phantom fixed so the
    grid isolates the edge-floor knob (production-comparable bet counts).
    """
    work = policy_lean_frame(df)
    if work.empty:
        return pd.DataFrame()
    scol = _season_col(work)
    if seasons is not None and scol is not None:
        want = {int(s) if str(s).isdigit() else s for s in seasons}
        season_vals = pd.to_numeric(work[scol], errors="coerce")
        if season_vals.notna().any():
            work = work[season_vals.isin(want)].copy()
        else:
            work = work[work[scol].isin(want)].copy()
    if work.empty:
        return pd.DataFrame()

    thrs = list(thresholds) if thresholds is not None else list(DEFAULT_EDGE_GRID)
    rows: list[dict[str, Any]] = []
    live = float(MIN_EDGE_BUCKET)
    for thr in thrs:
        thr_f = float(thr)
        bets = _select_bets(
            work,
            min_edge=thr_f,
            use_avoid_band=use_avoid_band,
            full_gates=full_gates,
        )
        if len(bets) < min_bets:
            rows.append({
                "min_edge": thr_f,
                "n_bets": int(len(bets)),
                "n_graded": 0,
                "ats_pct": np.nan,
                "flat_roi": np.nan,
                "mean_clv": np.nan,
                "n_clv": 0,
                "median_abs_edge": np.nan,
                "is_live_floor": abs(thr_f - live) < 1e-9,
                "use_avoid_band": bool(use_avoid_band),
                "full_gates": bool(full_gates),
                "below_min_bets": True,
            })
            continue
        stats = _score_bets(bets)
        stats.update({
            "min_edge": thr_f,
            "is_live_floor": abs(thr_f - live) < 1e-9,
            "use_avoid_band": bool(use_avoid_band),
            "full_gates": bool(full_gates),
            "below_min_bets": False,
        })
        rows.append(stats)
    return pd.DataFrame(rows)


def recommend_edge_floor(
    grid: pd.DataFrame,
    *,
    require_nonneg_clv: bool = True,
    min_clv_n: int = 15,
    min_bets: int = 40,
    prefer_near_live: bool = True,
    live_floor: float | None = None,
) -> dict[str, Any]:
    """Pick a candidate floor from a scored grid.

    Prefer finite mean CLV ≥ 0 with enough CLV samples, then max flat ROI,
    then volume. Optionally break ties toward the current live floor.
    """
    live = float(MIN_EDGE_BUCKET if live_floor is None else live_floor)
    out: dict[str, Any] = {
        "live_floor": live,
        "walkforward_min_floor": float(WALKFORWARD_EDGE_MIN_FLOOR)
        if WALKFORWARD_EDGE_MIN_FLOOR is not None
        else None,
        "recommended_min_edge": None,
        "reason": "insufficient_grid",
        "candidate": None,
    }
    if grid is None or grid.empty:
        return out

    g = grid[~grid.get("below_min_bets", False)].copy() if "below_min_bets" in grid.columns else grid.copy()
    g = g[g["n_bets"] >= int(min_bets)]
    if g.empty:
        out["reason"] = "no_row_meets_min_bets"
        return out

    eligible = g.copy()
    if require_nonneg_clv:
        has_clv = eligible["n_clv"] >= int(min_clv_n)
        if has_clv.any():
            eligible = eligible[has_clv & (eligible["mean_clv"].fillna(-1e9) >= 0)]
            if eligible.empty:
                out["reason"] = "no_row_with_nonneg_clv"
                # Fall back to best ROI among CLV-capable rows for diagnosis.
                clv_rows = g[g["n_clv"] >= int(min_clv_n)]
                if not clv_rows.empty:
                    worst = clv_rows.loc[clv_rows["mean_clv"].idxmax()]
                    out["diagnostic_best_clv_row"] = worst.to_dict()
                return out
        else:
            out["clv_note"] = "insufficient_finite_clv; ranking by flat_roi only"

    # Rank: flat_roi ↓, mean_clv ↓ (nan last), n_bets ↓, distance to live ↑
    rank = eligible.copy()
    rank["_roi"] = rank["flat_roi"].fillna(-1e9)
    rank["_clv"] = rank["mean_clv"].fillna(-1e9)
    rank["_n"] = rank["n_bets"].astype(float)
    if prefer_near_live:
        rank["_dist"] = (rank["min_edge"] - live).abs()
        rank = rank.sort_values(
            by=["_roi", "_clv", "_n", "_dist"],
            ascending=[False, False, False, True],
        )
    else:
        rank = rank.sort_values(by=["_roi", "_clv", "_n"], ascending=[False, False, False])

    best = rank.iloc[0]
    out["recommended_min_edge"] = float(best["min_edge"])
    out["candidate"] = {
        k: (None if isinstance(v, float) and not np.isfinite(v) else v)
        for k, v in best.drop(labels=[c for c in best.index if str(c).startswith("_")], errors="ignore")
        .to_dict()
        .items()
    }
    out["reason"] = "ok"
    if float(best["min_edge"]) + 1e-9 < live:
        out["caution"] = (
            "Recommended floor is below live MIN_EDGE_BUCKET; "
            "promote only after policy_tuning review — do not change config from this script."
        )
    return out


def compare_avoid_band(
    df: pd.DataFrame,
    *,
    thresholds: Sequence[float] | None = None,
    full_gates: bool = True,
    min_bets: int = 20,
    seasons: Iterable[Any] | None = None,
) -> pd.DataFrame:
    """Side-by-side grid with EDGE_AVOID_BAND on vs off."""
    a = edge_policy_grid(
        df,
        thresholds=thresholds,
        use_avoid_band=True,
        full_gates=full_gates,
        min_bets=min_bets,
        seasons=seasons,
    )
    b = edge_policy_grid(
        df,
        thresholds=thresholds,
        use_avoid_band=False,
        full_gates=full_gates,
        min_bets=min_bets,
        seasons=seasons,
    )
    if a.empty and b.empty:
        return pd.DataFrame()
    a = a.add_suffix("_avoid_on") if not a.empty else a
    b = b.add_suffix("_avoid_off") if not b.empty else b
    if a.empty:
        return b
    if b.empty:
        return a
    # Restore join key
    a = a.rename(columns={"min_edge_avoid_on": "min_edge"})
    b = b.rename(columns={"min_edge_avoid_off": "min_edge"})
    return a.merge(b, on="min_edge", how="outer").sort_values("min_edge")


def season_edge_summary(df: pd.DataFrame, *, min_edge: float | None = None) -> pd.DataFrame:
    """Per-season lean |edge| distribution vs a candidate floor."""
    work = policy_lean_frame(df)
    if work.empty:
        return pd.DataFrame()
    scol = _season_col(work)
    if scol is None:
        work["_season"] = "all"
        scol = "_season"
    floor = float(MIN_EDGE_BUCKET if min_edge is None else min_edge)
    rows = []
    for season, sub in work.groupby(scol, sort=True):
        ae = sub["abs_edge"]
        rows.append({
            "season": season,
            "n_lean": int(len(sub)),
            "median_abs_edge": float(ae.median()),
            "p75_abs_edge": float(ae.quantile(0.75)),
            f"n_ge_{floor:g}": int((ae >= floor).sum()),
            "frac_ge_floor": float((ae >= floor).mean()),
            "frac_ge_3.5": float((ae >= 3.5).mean()),
            "frac_ge_5.0": float((ae >= 5.0).mean()),
        })
    return pd.DataFrame(rows)


def live_policy_snapshot() -> dict[str, Any]:
    return {
        "MIN_EDGE_BUCKET": float(MIN_EDGE_BUCKET),
        "WALKFORWARD_EDGE_MIN_FLOOR": (
            float(WALKFORWARD_EDGE_MIN_FLOOR)
            if WALKFORWARD_EDGE_MIN_FLOOR is not None
            else None
        ),
        "EDGE_AVOID_BAND": normalize_edge_avoid_bands(EDGE_AVOID_BAND),
        "EDGE_AVOID_BAND_MIN_ELO_AGREE": float(EDGE_AVOID_BAND_MIN_ELO_AGREE),
        "breakeven_ats": float(BREAKEVEN_ATS),
    }


def run_edge_policy_calibration(
    df: pd.DataFrame,
    *,
    thresholds: Sequence[float] | None = None,
    use_avoid_band: bool = True,
    full_gates: bool = True,
    compare_avoid: bool = False,
    min_bets: int = 20,
    seasons: Iterable[Any] | None = None,
    require_nonneg_clv: bool = True,
    verbose: bool = True,
    metrics: dict | None = None,
    odds_provenance: dict | None = None,
    allow_without_accuracy_gates: bool = False,
) -> dict[str, Any]:
    """Full calibration report dict (+ optional console summary).

    Policy floor recommendations are blocked unless accuracy/uncertainty gates
    pass (or ``allow_without_accuracy_gates`` for research-only grids). Tip-proxy
    provenance always blocks ROI promotion of a new floor.
    """
    from pipeline.promotion_gates import policy_retune_allowed, tip_proxy_roi_blocked

    snap = live_policy_snapshot()
    accuracy_block = None
    if not allow_without_accuracy_gates:
        m = dict(metrics or {})
        if not m and isinstance(df, pd.DataFrame) and len(df):
            # Lightweight signals from the lean frame when pooled metrics absent.
            if "CONF_LOWER" in df.columns and "CONF_UPPER" in df.columns and "ACTUAL_MARGIN" in df.columns:
                lo = pd.to_numeric(df["CONF_LOWER"], errors="coerce")
                hi = pd.to_numeric(df["CONF_UPPER"], errors="coerce")
                act = pd.to_numeric(df["ACTUAL_MARGIN"], errors="coerce")
                mm = lo.notna() & hi.notna() & act.notna()
                if int(mm.sum()) >= 20:
                    m["interval_coverage"] = float(((act[mm] >= lo[mm]) & (act[mm] <= hi[mm])).mean())
            if {"PRED_SPREAD", "MARKET_SPREAD", "ACTUAL_MARGIN"}.issubset(df.columns):
                pred = pd.to_numeric(df["PRED_SPREAD"], errors="coerce")
                mkt = -pd.to_numeric(df["MARKET_SPREAD"], errors="coerce")
                act = pd.to_numeric(df["ACTUAL_MARGIN"], errors="coerce")
                edge = pred - mkt
                realized = act - mkt
                ok = edge.notna() & realized.notna()
                if int(ok.sum()) >= 30:
                    m["market_error_corr"] = float(edge[ok].corr(realized[ok]))
                mm2 = pred.notna() & mkt.notna()
                if int(mm2.sum()) >= 20 and float(mkt[mm2].abs().mean()) > 1e-9:
                    m["margin_dispersion_ratio"] = float(pred[mm2].abs().mean() / mkt[mm2].abs().mean())
        if not policy_retune_allowed(m, odds_provenance=odds_provenance):
            accuracy_block = (
                "Policy retune blocked: accuracy/uncertainty gates not passed "
                "(or tip-proxy / non-promotion_eligible odds)."
            )
    if tip_proxy_roi_blocked(odds_provenance):
        accuracy_block = (accuracy_block or "") + (
            " Tip-proxy ROI claims blocked."
        )

    grid = edge_policy_grid(
        df,
        thresholds=thresholds,
        use_avoid_band=use_avoid_band,
        full_gates=full_gates,
        min_bets=min_bets,
        seasons=seasons,
    )
    rec = recommend_edge_floor(
        grid,
        require_nonneg_clv=require_nonneg_clv,
        min_bets=max(min_bets, 40),
    )
    if accuracy_block:
        rec = {
            **rec,
            "recommended_min_edge": None,
            "reason": accuracy_block,
            "promote_allowed": False,
            "caution": accuracy_block,
        }
    else:
        rec = {**rec, "promote_allowed": True}
    seasons_tbl = season_edge_summary(df)
    avoid_cmp = (
        compare_avoid_band(
            df,
            thresholds=thresholds,
            full_gates=full_gates,
            min_bets=min_bets,
            seasons=seasons,
        )
        if compare_avoid
        else pd.DataFrame()
    )

    # Reference: apply_edge_threshold path at live floor (sanity vs lean grid).
    live_ref = {}
    if "EDGE" in df.columns and "MARKET_SPREAD" in df.columns:
        sim = apply_edge_threshold(df[df["MARKET_SPREAD"].notna()], float(MIN_EDGE_BUCKET))
        bets = sim[sim["DIRECTION"].isin(("Home", "Away"))]
        live_ref = {
            "min_edge": float(MIN_EDGE_BUCKET),
            "n_bets_apply_threshold": int(len(bets)),
        }

    report = {
        "live_policy": snap,
        "grid": grid,
        "recommendation": rec,
        "season_summary": seasons_tbl,
        "avoid_band_compare": avoid_cmp,
        "live_apply_threshold_ref": live_ref,
        "n_lean_rows": int(len(policy_lean_frame(df))),
    }

    if verbose:
        print("\n" + "=" * 64)
        print("EDGE POLICY CALIBRATION (post-hoc; does not change config)")
        print("=" * 64)
        print(
            f"Live floor: MIN_EDGE_BUCKET={snap['MIN_EDGE_BUCKET']}  "
            f"WF_MIN_FLOOR={snap['WALKFORWARD_EDGE_MIN_FLOOR']}"
        )
        print(f"Avoid bands: {snap['EDGE_AVOID_BAND']}  (agree≥{snap['EDGE_AVOID_BAND_MIN_ELO_AGREE']})")
        print(
            f"Gate mode: {'full (edge + confidence/trust/…)' if full_gates else 'edge-only (upper-bound volume)'}"
        )
        print(f"Lean rows with market line: {report['n_lean_rows']:,}")
        if live_ref:
            print(f"edge-only apply_edge_threshold @ live: n={live_ref['n_bets_apply_threshold']}")
        if not seasons_tbl.empty:
            print("\nPer-season lean |edge| vs live floor:")
            print(seasons_tbl.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
        if not grid.empty:
            show = grid.copy()
            for c in ("ats_pct", "flat_roi", "mean_clv"):
                if c in show.columns:
                    show[c] = show[c].map(lambda v: f"{v:.3f}" if pd.notna(v) else "")
            print("\nThreshold grid:")
            print(show.to_string(index=False))
        print(f"\nRecommendation: {rec.get('recommended_min_edge')}  ({rec.get('reason')})")
        if rec.get("caution"):
            print(f"  ⚠ {rec['caution']}")
        if rec.get("candidate"):
            c = rec["candidate"]
            print(
                f"  n={c.get('n_bets')}  ATS={c.get('ats_pct')}  "
                f"ROI={c.get('flat_roi')}  mean_CLV={c.get('mean_clv')}  n_CLV={c.get('n_clv')}"
            )
        print(
            "\nThis script never writes pipeline/config.py. "
            "Promote a new floor only after policy_tuning review."
        )
    return report
