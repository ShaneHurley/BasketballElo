"""Bet selection helpers: edge-bucket (legacy) vs confidence-only actionable gate."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.config import (
    BET_SELECTION_MODE,
    CONFIDENCE_EDGE_SCALED,
    EDGE_AVOID_BAND,
    EDGE_AVOID_BAND_MIN_ELO_AGREE,
    EDGE_STAKE_TIERS,
    MAX_QUANTILE_WIDTH,
    MIN_CONFIDENCE_SCORE,
    MIN_DISAGREEMENT_TRUST,
    MIN_EDGE_BUCKET,
    SKIP_PHANTOM_INJURY,
    SKIP_TIGHT_SPREAD,
    TIGHT_SPREAD_MAX,
    USE_TIER_STAKE_GATES,
)


def uses_edge_gates(mode: str | None = None) -> bool:
    """True when |edge| bucket / quantile width filter actionable bets."""
    import pipeline.config as cfg
    mode = mode or cfg.BET_SELECTION_MODE
    return mode in ("edge_bucket", "edge_bucket_ats")


def edge_bucket_min_for_stakes(mode: str | None = None) -> float | None:
    import pipeline.config as cfg
    if not uses_edge_gates(mode):
        return None
    return float(cfg.MIN_EDGE_BUCKET)


def analysis_min_edge_pts(effective_threshold: float, mode: str | None = None) -> float:
    """Min edge passed to bet_analysis — 0 in confidence_only so all lines get win%."""
    if not uses_edge_gates(mode):
        return 0.0
    return float(effective_threshold)


def spread_lean_from_edge(edge_pts: float) -> str:
    if edge_pts is None or not np.isfinite(edge_pts) or abs(float(edge_pts)) < 1e-9:
        return "Pass"
    return "Home" if float(edge_pts) > 0 else "Away"


def spread_side_series(df: pd.DataFrame) -> pd.Series:
    """Model lean side for every row (EDGE_LEAN → EDGE sign → DIRECTION)."""
    if "EDGE_LEAN" in df.columns:
        lean = df["EDGE_LEAN"].fillna("Pass").astype(str)
    else:
        lean = df.get("DIRECTION", pd.Series("Pass", index=df.index)).fillna("Pass").astype(str)
    if "EDGE" in df.columns:
        edge = pd.to_numeric(df["EDGE"], errors="coerce").fillna(0)
        need = lean.isin(("Pass", "nan", "")) | lean.isna()
        inferred = np.where(edge > 0, "Home", np.where(edge < 0, "Away", "Pass"))
        lean = lean.where(~need, inferred)
    return lean


def win_pct_column(df: pd.DataFrame) -> str:
    if "WIN_PCT" in df.columns:
        return "WIN_PCT"
    return "CONFIDENCE"


def lean_spread_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Lined games with a model lean (includes sub-threshold edges)."""
    if df is None or df.empty:
        return pd.DataFrame()
    m = df[df["MARKET_SPREAD"].notna()].copy()
    side = spread_side_series(m)
    m = m[side != "Pass"].copy()
    m["_side"] = side.loc[m.index]
    return m


def actionable_spread_frame(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """Rows that count as placed spread bets under the active selection mode."""
    if df is None or df.empty:
        return pd.DataFrame()
    import pipeline.config as cfg
    mode = mode or cfg.BET_SELECTION_MODE
    m = df[df["MARKET_SPREAD"].notna()].copy()
    if uses_edge_gates(mode):
        if "DIRECTION" in m.columns:
            return m[m["DIRECTION"] != "Pass"].copy()
        return lean_spread_frame(m)
    if "ACTIONABLE" in m.columns:
        return m[m["ACTIONABLE"].astype(int) == 1].copy()
    if "DIRECTION" in m.columns:
        return m[m["DIRECTION"] != "Pass"].copy()
    return lean_spread_frame(m)


def spread_bet_frame(df: pd.DataFrame, mode: str | None = None) -> pd.DataFrame:
    """Alias for actionable spread bets (metrics / diagnostics)."""
    return actionable_spread_frame(df, mode)


def passes_edge_bucket(abs_edge: float, min_edge: float | None = None) -> bool:
    if abs_edge is None or not np.isfinite(abs_edge):
        return False
    thr = MIN_EDGE_BUCKET if min_edge is None else min_edge
    return float(abs(abs_edge)) >= float(thr)


def passes_quantile_width(width: float, max_width: float | None = None) -> bool:
    if width is None or not np.isfinite(width):
        return True
    import pipeline.config as cfg
    cap = cfg.MAX_QUANTILE_WIDTH if max_width is None else max_width
    if cap is None:
        return True
    return float(width) <= float(cap)


def edge_stake_multiplier(abs_edge: float, tiers=EDGE_STAKE_TIERS) -> float:
    ae = float(abs(abs_edge)) if abs_edge is not None and np.isfinite(abs_edge) else 0.0
    for lo, hi, mult in tiers:
        if lo <= ae < hi:
            return float(mult)
    return 1.0


def edge_scaled_min_confidence(abs_edge: float, base: float | None = None) -> float:
    """Higher confidence bar for small edges when CONFIDENCE_EDGE_SCALED is enabled."""
    import pipeline.config as cfg
    floor = float(MIN_CONFIDENCE_SCORE if base is None else base)
    if not cfg.CONFIDENCE_EDGE_SCALED:
        return floor
    ae = float(abs(abs_edge)) if abs_edge is not None and np.isfinite(abs_edge) else 0.0
    if ae < 5.0:
        return floor + 4.0
    if ae < 8.0:
        return floor + 2.0
    return floor


def passes_edge_avoid_band(
    abs_edge: float,
    elo_meta_agreement: float | None = None,
) -> bool:
    """False when |edge| is in the configured dead-zone band without Elo agreement."""
    import pipeline.config as cfg
    band = cfg.EDGE_AVOID_BAND
    if band is None:
        return True
    if abs_edge is None or not np.isfinite(abs_edge):
        return True
    lo, hi = band
    ae = float(abs(abs_edge))
    if not (float(lo) <= ae < float(hi)):
        return True
    agree = float(elo_meta_agreement if elo_meta_agreement is not None else 1.0)
    return agree >= float(cfg.EDGE_AVOID_BAND_MIN_ELO_AGREE)


def passes_confidence_actionable_gates(
    *,
    lean: str,
    edge_pts: float,
    conf_score: float,
    conf_width: float,
    min_confidence: float,
    max_confidence: float | None = None,
    disagreement_trust: float = 1.0,
    phantom_injury_flag: bool = False,
    market_spread: float | None = None,
    elo_meta_agreement: float | None = None,
    min_edge: float | None = None,
) -> bool:
    """Shared actionable spread gate for simulate + live predict."""
    import pipeline.config as cfg
    if lean in ("Pass", None, "", "nan"):
        return False
    min_edge_val = float(cfg.CONFIDENCE_MIN_EDGE if min_edge is None else min_edge)
    if abs(float(edge_pts or 0.0)) < min_edge_val:
        return False
    if not passes_quantile_width(conf_width):
        return False
    if float(disagreement_trust or 1.0) < float(MIN_DISAGREEMENT_TRUST):
        return False
    if SKIP_PHANTOM_INJURY and phantom_injury_flag:
        return False
    if SKIP_TIGHT_SPREAD and market_spread is not None and np.isfinite(market_spread):
        if abs(float(market_spread)) <= float(TIGHT_SPREAD_MAX):
            return False
    if not passes_edge_avoid_band(abs(edge_pts), elo_meta_agreement=elo_meta_agreement):
        return False
    if cfg.CONFIDENCE_SELECTION_MODE == "min_score":
        eff_min = edge_scaled_min_confidence(abs(edge_pts), base=min_confidence)
        if float(conf_score) < eff_min:
            return False
        if max_confidence is not None and float(conf_score) > float(max_confidence):
            return False
    return True


def apply_bet_selection_gates(
    direction: str,
    edge_pts: float,
    conf_width: float,
    *,
    mode: str | None = None,
) -> str:
    """Return direction or Pass after edge-bucket / quantile filters (skipped in confidence_only)."""
    import pipeline.config as cfg
    if direction == "Pass":
        return direction
    mode = mode or cfg.BET_SELECTION_MODE
    if not uses_edge_gates(mode):
        return direction
    if mode in ("edge_bucket", "edge_bucket_ats"):
        if not passes_edge_bucket(abs(edge_pts), min_edge=cfg.MIN_EDGE_BUCKET):
            return "Pass"
        if not passes_quantile_width(conf_width, max_width=cfg.MAX_QUANTILE_WIDTH):
            return "Pass"
    return direction


def use_tier_stake_gates() -> bool:
    import pipeline.config as cfg
    return bool(cfg.USE_TIER_STAKE_GATES) and cfg.BET_SELECTION_MODE == "legacy_tiers"
