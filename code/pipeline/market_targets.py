"""Closing-line residual targets for leak-free meta-model training."""
from __future__ import annotations

import numpy as np
import pandas as pd


def closing_spread_series(df: pd.DataFrame) -> pd.Series | None:
    """Pre-tip closing line (home perspective); NaN when unavailable."""
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


def margin_close_residual(actual_margin, closing_spread) -> np.ndarray:
    """ATS residual vs closing line: actual_margin + closing_spread."""
    m = np.asarray(actual_margin, dtype=float)
    c = np.asarray(closing_spread, dtype=float)
    out = m.copy()
    mask = np.isfinite(c)
    out[mask] = m[mask] + c[mask]
    return out


def residual_to_margin(residual, closing_spread) -> np.ndarray:
    """Map predicted close-residual back to predicted home margin."""
    r = np.asarray(residual, dtype=float)
    c = np.asarray(closing_spread, dtype=float)
    out = r.copy()
    mask = np.isfinite(c)
    out[mask] = r[mask] - c[mask]
    return out


def market_total_series(df: pd.DataFrame) -> pd.Series | None:
    """Pre-tip market total; NaN when unavailable."""
    if df is None or df.empty:
        return None
    if "market_total" in df.columns:
        mkt = pd.to_numeric(df["market_total"], errors="coerce")
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


def valid_closing_fraction(df: pd.DataFrame) -> float:
    close = closing_spread_series(df)
    if close is None:
        return 0.0
    return float(close.notna().mean())
