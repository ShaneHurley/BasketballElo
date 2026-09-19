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


def compute_log_loss(y_true, y_prob, *, eps: float = 1e-15) -> float:
    """Binary log-loss (cross-entropy); lower is better."""
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    if len(y) == 0:
        return float("nan")
    p = np.clip(p, eps, 1.0 - eps)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def murphy_brier_decomposition(y_true, y_prob, n_bins: int = 10) -> dict:
    """Murphy (1973) Brier decomposition: REL − RES + UNC.

    Returns dict with ``brier``, ``reliability``, ``resolution``, ``uncertainty``,
    and ``n``. Read-only diagnostic — does not change bet selection.
    """
    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], p[mask]
    n = int(len(y))
    if n == 0:
        return {
            "brier": float("nan"), "reliability": float("nan"),
            "resolution": float("nan"), "uncertainty": float("nan"), "n": 0,
        }
    base_rate = float(y.mean())
    unc = base_rate * (1.0 - base_rate)
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    rel = 0.0
    res = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            m = (p >= lo) & (p <= hi)
        else:
            m = (p >= lo) & (p < hi)
        nk = int(m.sum())
        if nk == 0:
            continue
        pk = float(p[m].mean())
        ok = float(y[m].mean())
        rel += (nk / n) * (pk - ok) ** 2
        res += (nk / n) * (ok - base_rate) ** 2
    brier = float(np.mean((p - y) ** 2))
    return {
        "brier": brier,
        "reliability": float(rel),
        "resolution": float(res),
        "uncertainty": float(unc),
        "n": n,
    }


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


def calibration_slope_intercept(y_true, y_prob, *, eps: float = 1e-6) -> dict:
    """Logistic calibration slope/intercept of outcomes vs logit(pred).

    Ideal slope ≈ 1 and intercept ≈ 0. Low-variance diagnostic preferred
    over ECE alone for promotion reviews.
    """
    from sklearn.linear_model import LogisticRegression

    y = np.asarray(y_true, dtype=float)
    p = np.asarray(y_prob, dtype=float)
    mask = np.isfinite(y) & np.isfinite(p)
    y, p = y[mask], np.clip(p[mask], eps, 1.0 - eps)
    if len(y) < 30 or len(np.unique(y)) < 2:
        return {"slope": float("nan"), "intercept": float("nan"), "n": int(len(y))}
    logit = np.log(p / (1.0 - p)).reshape(-1, 1)
    clf = LogisticRegression(C=1e6, solver="lbfgs", max_iter=200)
    clf.fit(logit, y.astype(int))
    return {
        "slope": float(clf.coef_[0, 0]),
        "intercept": float(clf.intercept_[0]),
        "n": int(len(y)),
    }
