"""Data-driven edge vs accuracy analysis (flat and stake-weighted ROI)."""
from __future__ import annotations

import numpy as np
import pandas as pd

BREAKEVEN_ATS = 110.0 / 210.0


def _analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Spread bets for edge analysis — all leans in confidence_only, else actionable only."""
    if df is None or df.empty:
        return pd.DataFrame()
    from pipeline.config import BET_SELECTION_MODE
    from pipeline.bet_selection import lean_spread_frame, spread_side_series

    if BET_SELECTION_MODE == "confidence_only":
        d = lean_spread_frame(df)
        if d.empty:
            return pd.DataFrame()
        cover = d["ACTUAL_MARGIN"] + d["MARKET_SPREAD"]
        side = d["_side"] if "_side" in d.columns else spread_side_series(d)
        d = d.copy()
        d["won"] = (((side == "Home") & (cover > 0)) | ((side == "Away") & (cover < 0))).astype(int)
        d["abs_edge"] = pd.to_numeric(d["EDGE"], errors="coerce").abs()
        return d

    return _ats_frame(df)


def _ats_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()
    d = df[df.get("DIRECTION", "Pass") != "Pass"].copy()
    if "MARKET_SPREAD" not in d.columns:
        return pd.DataFrame()
    cover = d["ACTUAL_MARGIN"] + d["MARKET_SPREAD"]
    home = d["DIRECTION"] == "Home"
    d["won"] = ((home & (cover > 0)) | (~home & (cover < 0))).astype(int)
    d["abs_edge"] = d["EDGE"].abs()
    return d


def _roi_from_winrate(wp: float) -> float:
    return wp * (100.0 / 110.0) - (1.0 - wp)


def fine_edge_bins(
    df: pd.DataFrame,
    *,
    width: float = 0.5,
    min_edge: float = 2.5,
    max_edge: float = 16.0,
    min_n: int = 25,
) -> pd.DataFrame:
    """ATS/ROI in half-point (or custom width) edge bins."""
    bets = _analysis_frame(df)
    if bets.empty:
        return pd.DataFrame()
    bins = np.arange(min_edge, max_edge + width, width)
    bets["ebin"] = pd.cut(bets["abs_edge"], bins=bins, right=False)
    rows = []
    for iv, sub in bets.groupby("ebin", observed=True):
        if len(sub) < min_n:
            continue
        wp = float(sub["won"].mean())
        rows.append({
            "edge_lo": float(iv.left),
            "edge_hi": float(iv.right),
            "edge_mid": float((iv.left + iv.right) / 2.0),
            "n_bets": len(sub),
            "ats_pct": wp,
            "flat_roi": _roi_from_winrate(wp),
        })
    return pd.DataFrame(rows)


def rolling_edge_ats(
    df: pd.DataFrame,
    *,
    centers: np.ndarray | None = None,
    half_width: float = 0.75,
    min_n: int = 40,
) -> pd.DataFrame:
    """Smoothed ATS curve over |edge| (rolling window in edge space)."""
    bets = _analysis_frame(df)
    if bets.empty:
        return pd.DataFrame()
    if centers is None:
        centers = np.arange(3.0, 13.0, 0.25)
    rows = []
    for c in centers:
        sub = bets[(bets["abs_edge"] >= c - half_width) & (bets["abs_edge"] < c + half_width)]
        if len(sub) < min_n:
            continue
        wp = float(sub["won"].mean())
        rows.append({
            "edge_center": float(c),
            "n_bets": len(sub),
            "ats_pct": wp,
            "flat_roi": _roi_from_winrate(wp),
        })
    return pd.DataFrame(rows)


def fit_parabolic_edge_curve(binned: pd.DataFrame) -> dict:
    """Quadratic ATS ~ edge + edge² on fine bins (detects mid-range valleys)."""
    if binned is None or binned.empty or len(binned) < 4:
        return {}
    x = binned["edge_mid"].to_numpy(dtype=float)
    y = binned["ats_pct"].to_numpy(dtype=float)
    w = np.sqrt(binned["n_bets"].to_numpy(dtype=float))
    X = np.column_stack([np.ones(len(x)), x, x ** 2])
    coef, _, _, _ = np.linalg.lstsq(X * w[:, None], y * w, rcond=None)
    a, b, c = coef
    out = {"intercept": float(a), "linear": float(b), "quadratic": float(c)}
    if abs(c) > 1e-8:
        peak = -b / (2.0 * c)
        out["vertex_edge"] = float(peak)
        out["vertex_ats"] = float(a + b * peak + c * peak ** 2)
    return out


def find_edge_valleys(rolling: pd.DataFrame, *, min_drop: float = 0.025) -> list[dict]:
    """Local ATS dips on the rolling curve (candidate injury/noise zones)."""
    if rolling is None or len(rolling) < 5:
        return []
    r = rolling.sort_values("edge_center").reset_index(drop=True)
    valleys = []
    for i in range(1, len(r) - 1):
        prev_a = r.loc[i - 1, "ats_pct"]
        cur_a = r.loc[i, "ats_pct"]
        next_a = r.loc[i + 1, "ats_pct"]
        local_peak = max(prev_a, next_a)
        if local_peak - cur_a >= min_drop:
            valleys.append({
                "edge_center": float(r.loc[i, "edge_center"]),
                "ats_pct": float(cur_a),
                "drop_from_neighbors": float(local_peak - cur_a),
                "n_bets": int(r.loc[i, "n_bets"]),
            })
    return valleys


def threshold_grid(
    df: pd.DataFrame,
    *,
    thresholds: np.ndarray | None = None,
    min_n: int = 40,
    stake_col: str = "STAKE_MODERATE",
    profit_col: str = "PROFIT_MODERATE",
) -> pd.DataFrame:
    """Flat ATS/ROI and optional Kelly-weighted ROI by minimum |edge|."""
    bets = _analysis_frame(df)
    if bets.empty:
        return pd.DataFrame()
    if thresholds is None:
        thresholds = np.arange(2.5, 10.5, 0.25)
    rows = []
    for thr in thresholds:
        sub = bets[bets["abs_edge"] >= thr]
        if len(sub) < min_n:
            continue
        wp = float(sub["won"].mean())
        row = {
            "min_edge": float(thr),
            "n_bets": len(sub),
            "ats_pct": wp,
            "flat_roi": _roi_from_winrate(wp),
        }
        if stake_col in sub.columns and profit_col in sub.columns:
            st = sub[stake_col].fillna(0)
            if st.sum() > 0:
                row["stake_roi"] = float(sub[profit_col].fillna(0).sum() / st.sum())
        rows.append(row)
    return pd.DataFrame(rows)


def segment_breakpoints(df: pd.DataFrame) -> pd.DataFrame:
    """Hand-tuned segments that match typical NBA backtest shapes."""
    bets = _analysis_frame(df)
    if bets.empty:
        return pd.DataFrame()
    segments = [
        (2.5, 4.0, "dead_zone_2-4"),
        (4.0, 5.5, "build_4-5.5"),
        (5.5, 7.0, "core_5.5-7"),
        (7.0, 9.5, "valley_7-9.5"),
        (9.5, 11.0, "recovery_9.5-11"),
        (11.0, 99.0, "tail_11+"),
    ]
    rows = []
    for lo, hi, name in segments:
        sub = bets[(bets["abs_edge"] >= lo) & (bets["abs_edge"] < hi)]
        if len(sub) < 15:
            continue
        wp = float(sub["won"].mean())
        rows.append({
            "segment": name,
            "edge_lo": lo,
            "edge_hi": hi,
            "n_bets": len(sub),
            "ats_pct": wp,
            "flat_roi": _roi_from_winrate(wp),
            "mean_confidence": float(sub["CONFIDENCE"].mean()) if "CONFIDENCE" in sub.columns else np.nan,
        })
    return pd.DataFrame(rows)


def injury_disagreement_split(df: pd.DataFrame) -> pd.DataFrame:
    """Split ATS by phantom-injury / disagreement flags when present."""
    bets = _analysis_frame(df)
    if bets.empty:
        return pd.DataFrame()
    rows = []
    flags = []
    if "PHANTOM_INJURY_FLAG" in bets.columns:
        flags.append(("phantom_injury", bets["PHANTOM_INJURY_FLAG"].astype(bool)))
    if "DISAGREEMENT_TRUST" in bets.columns:
        flags.append(("low_trust", bets["DISAGREEMENT_TRUST"] < 0.85))
    if "H_STAR_OUT" in bets.columns or "A_STAR_OUT" in bets.columns:
        star = bets.get("H_STAR_OUT", 0).fillna(0) + bets.get("A_STAR_OUT", 0).fillna(0)
        flags.append(("known_star_out", star > 0))
    if "ELO_META_AGREEMENT" in bets.columns:
        flags.append(("elo_meta_disagree", bets["ELO_META_AGREEMENT"] < 0.5))

    if not flags:
        # Legacy proxy: huge edge + high confidence without injury columns
        proxy = (bets["abs_edge"] >= 8.0) & (bets.get("CONFIDENCE", 0) >= 35)
        flags.append(("legacy_phantom_proxy", proxy))

    for name, mask in flags:
        for label, sub in [("yes", bets[mask]), ("no", bets[~mask])]:
            if len(sub) < 20:
                continue
            wp = float(sub["won"].mean())
            rows.append({
                "flag": name,
                "value": label,
                "n_bets": len(sub),
                "ats_pct": wp,
                "flat_roi": _roi_from_winrate(wp),
                "mean_edge": float(sub["abs_edge"].mean()),
            })
    return pd.DataFrame(rows)


def recommend_policy(
    df: pd.DataFrame,
    *,
    min_bets: int = 80,
) -> dict:
    """Suggest min edge, optional cap, and segments to favor/avoid."""
    grid = threshold_grid(df, min_n=min_bets)
    rolling = rolling_edge_ats(df)
    valleys = find_edge_valleys(rolling)
    segments = segment_breakpoints(df)
    injury = injury_disagreement_split(df)

    policy = {
        "n_active_bets": int(len(_analysis_frame(df))),
        "valleys": valleys,
        "injury_split": injury.to_dict(orient="records") if not injury.empty else [],
    }

    if not grid.empty:
        best_flat = grid.loc[grid["flat_roi"].idxmax()]
        policy["best_min_edge_flat"] = {
            "min_edge": float(best_flat["min_edge"]),
            "n_bets": int(best_flat["n_bets"]),
            "ats_pct": float(best_flat["ats_pct"]),
            "flat_roi": float(best_flat["flat_roi"]),
        }
        # Volume-balanced: best ROI with n >= 15% of max n in grid
        max_n = grid["n_bets"].max()
        bal = grid[grid["n_bets"] >= 0.15 * max_n]
        if not bal.empty:
            bb = bal.loc[bal["flat_roi"].idxmax()]
            policy["best_min_edge_balanced"] = {
                "min_edge": float(bb["min_edge"]),
                "n_bets": int(bb["n_bets"]),
                "ats_pct": float(bb["ats_pct"]),
                "flat_roi": float(bb["flat_roi"]),
            }

    if not segments.empty:
        policy["best_segment"] = segments.loc[segments["flat_roi"].idxmax()].to_dict()
        policy["worst_segment"] = segments.loc[segments["flat_roi"].idxmin()].to_dict()

    bins = fine_edge_bins(df)
    policy["parabolic_fit"] = fit_parabolic_edge_curve(bins)

    return policy


def run_edge_analysis(df: pd.DataFrame, *, verbose: bool = True) -> dict:
    """Full report dict + optional console summary."""
    bets = _analysis_frame(df)
    out = {
        "fine_bins": fine_edge_bins(df),
        "rolling": rolling_edge_ats(df),
        "threshold_grid": threshold_grid(df),
        "segments": segment_breakpoints(df),
        "injury_split": injury_disagreement_split(df),
        "policy": recommend_policy(df),
    }
    if verbose and not bets.empty:
        print("\n" + "=" * 60)
        print("EDGE vs ACCURACY ANALYSIS")
        print("=" * 60)
        print(f"Active spread bets (non-push): {len(bets):,}")
        print(f"Pooled ATS: {bets['won'].mean():.1%}  flat ROI: {_roi_from_winrate(bets['won'].mean()):+.1%}")

        seg = out["segments"]
        if not seg.empty:
            print("\nSegment table:")
            print(seg.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        pol = out["policy"]
        if pol.get("best_min_edge_balanced"):
            b = pol["best_min_edge_balanced"]
            print(f"\nRecommended min |edge| (volume-balanced): {b['min_edge']:.2f} pt")
            print(f"  n={b['n_bets']}  ATS={b['ats_pct']:.1%}  flat ROI={b['flat_roi']:+.1%}")
        if pol.get("valleys"):
            print("\nLocal ATS valleys (possible injury/noise zones):")
            for v in pol["valleys"]:
                print(f"  edge~{v['edge_center']:.2f}: ATS={v['ats_pct']:.1%}  drop={v['drop_from_neighbors']:.1%}")

        inj = out["injury_split"]
        if not inj.empty:
            print("\nInjury / disagreement splits:")
            print(inj.to_string(index=False, float_format=lambda x: f"{x:.3f}"))

        pf = pol.get("parabolic_fit", {})
        if pf:
            print(f"\nQuadratic bin fit: ATS ≈ {pf.get('intercept', 0):.3f} + {pf.get('linear', 0):.4f}·edge + {pf.get('quadratic', 0):.5f}·edge²")
    return out
