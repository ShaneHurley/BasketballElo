"""Calibration quality metrics for probabilistic betting models."""
from __future__ import annotations

import numpy as np
import pandas as pd


def compute_brier(y_true, y_prob) -> float:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    if mask.sum() == 0:
        return float("nan")
    return float(np.mean((p[mask] - y[mask]) ** 2))


def compute_ece(y_true, y_prob, n_bins: int = 10) -> float:
    """Expected Calibration Error (equal-width bins on predicted probability)."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) < n_bins:
        return float("nan")
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        ece += m.mean() * abs(float(y[m].mean()) - float(p[m].mean()))
    return float(ece)


def reliability_bins(y_true, y_prob, n_bins: int = 10) -> pd.DataFrame:
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return pd.DataFrame()
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        if not m.any():
            continue
        rows.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "pred_mean": float(p[m].mean()),
            "obs_rate": float(y[m].mean()),
            "n": int(m.sum()),
        })
    return pd.DataFrame(rows)
