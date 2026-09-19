"""T-60 decision-line residual targets for leak-free meta-model training.

Closing lines are evaluation-only (CLV). Model training and live
reconstruction must use the decision (T-60) spread available at bet time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def decision_spread_series(df: pd.DataFrame) -> pd.Series | None:
    """T-60 decision line (home perspective); falls back to market_spread.

    Never prefers ``closing_spread`` — close is evaluation-only.
    """
    if df is None or df.empty:
        return None
    if "decision_spread" in df.columns:
        decision = pd.to_numeric(df["decision_spread"], errors="coerce")
        # Fill gaps from market_spread (often the same T-60 bet line).
        if "market_spread" in df.columns:
            mkt = pd.to_numeric(df["market_spread"], errors="coerce")
            decision = decision.fillna(mkt)
    elif "market_spread" in df.columns:
        decision = pd.to_numeric(df["market_spread"], errors="coerce")
    else:
        return None
    if decision.notna().sum() < max(20, int(0.05 * len(decision))):
        return None
    return decision


def closing_spread_series(df: pd.DataFrame) -> pd.Series | None:
    """Closing line (home perspective) for CLV / evaluation only.

    Prefer ``closing_spread``; fall back to market only when no close column
    exists (single-snapshot sources). Do not use this for model targets.
    """
    if df is None or df.empty:
        return None
    if "closing_spread" in df.columns:
        close = pd.to_numeric(df["closing_spread"], errors="coerce")
    elif "market_spread" in df.columns:
        close = pd.to_numeric(df["market_spread"], errors="coerce")
    else:
        return None
    if close.notna().sum() < max(20, int(0.05 * len(close))):
        return None
    return close


def margin_decision_residual(actual_margin, decision_spread) -> np.ndarray:
    """ATS residual vs T-60 decision line: actual_margin + decision_spread."""
    m = np.asarray(actual_margin, dtype=float)
    d = np.asarray(decision_spread, dtype=float)
    out = m.copy()
    mask = np.isfinite(d)
    out[mask] = m[mask] + d[mask]
    return out


def residual_to_margin(residual, decision_spread) -> np.ndarray:
    """Map predicted decision-residual back to predicted home margin."""
    r = np.asarray(residual, dtype=float)
    d = np.asarray(decision_spread, dtype=float)
    out = r.copy()
    mask = np.isfinite(d)
    out[mask] = r[mask] - d[mask]
    return out


# Backward-compatible aliases (close was historically conflated with decision).
margin_close_residual = margin_decision_residual


def market_total_series(df: pd.DataFrame) -> pd.Series | None:
    """Pre-tip market total; NaN when unavailable."""
    if df is None or df.empty:
        return None
    if "market_total" in df.columns:
        mkt = pd.to_numeric(df["market_total"], errors="coerce")
        # Treat legacy 0.0 sentinel as missing (pre-fix substitution).
        mkt = mkt.where(mkt > 0, np.nan)
    else:
        return None
    if mkt.notna().sum() < max(20, int(0.05 * len(mkt))):
        return None
    return mkt


def total_market_residual(actual_total, market_total) -> np.ndarray:
    """O/U residual vs market total: actual_total - market_total."""
    a = np.asarray(actual_total, dtype=float)
    m = np.asarray(market_total, dtype=float)
    out = a.copy()
    mask = np.isfinite(m)
    out[mask] = a[mask] - m[mask]
    return out


def residual_to_total(residual, market_total) -> np.ndarray:
    """Map predicted market-residual back to predicted game total."""
    r = np.asarray(residual, dtype=float)
    m = np.asarray(market_total, dtype=float)
    out = r.copy()
    mask = np.isfinite(m)
    out[mask] = r[mask] + m[mask]
    return out


def valid_market_total_fraction(df: pd.DataFrame) -> float:
    mkt = market_total_series(df)
    if mkt is None:
        return 0.0
    return float(mkt.notna().mean())


def valid_decision_fraction(df: pd.DataFrame) -> float:
    decision = decision_spread_series(df)
    if decision is None:
        return 0.0
    return float(decision.notna().mean())


def valid_closing_fraction(df: pd.DataFrame) -> float:
    """Fraction of rows with a usable *decision* line (train gate).

    Kept name for call-site compatibility; now checks decision/T-60 coverage.
    """
    return valid_decision_fraction(df)
