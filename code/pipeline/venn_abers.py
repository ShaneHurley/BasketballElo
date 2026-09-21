"""Minimal binary Venn-Abers interval from calibration scores."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression

MIN_CALIBRATION_SAMPLES = 10


def scores_to_interval(scores_cal: np.ndarray, labels_cal: np.ndarray, scores_test: np.ndarray):
    """Return (p0, p1, optimal_point, width) arrays for test scores."""
    scores_cal = np.asarray(scores_cal, dtype=float)
    labels_cal = np.asarray(labels_cal, dtype=int)
    scores_test = np.asarray(scores_test, dtype=float)
    if scores_cal.size < MIN_CALIBRATION_SAMPLES:
        raise ValueError(
            f"scores_to_interval requires at least {MIN_CALIBRATION_SAMPLES} "
            f"calibration scores, got {scores_cal.size}"
        )
    p0_list, p1_list = [], []
    for s in scores_test:
        p0 = _isotonic_prob(np.append(scores_cal, s), np.append(labels_cal, 0), s)
        p1 = _isotonic_prob(np.append(scores_cal, s), np.append(labels_cal, 1), s)
        p0_list.append(p0)
        p1_list.append(p1)
    p0 = np.asarray(p0_list)
    p1 = np.asarray(p1_list)
    denom = 1.0 - p0 + p1
    opt = np.where(denom > 1e-9, p1 / denom, 0.5)
    width = p1 - p0
    return p0, p1, opt, width


def _isotonic_prob(scores, labels, query_score):
    if len(np.unique(labels)) < 2:
        return float(labels.mean())
    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(scores, labels)
    return float(iso.predict([query_score])[0])


def passes_venn_abers_filter(width: float, max_width: float) -> bool:
    # Fail closed: a risk-limiting filter that cannot evaluate width must reject.
    if width is None or not np.isfinite(width):
        return False
    return float(width) <= float(max_width)


def market_implied_inside_interval(p0, p1, p_mkt) -> bool:
    """True when market implied probability lies in the Venn-Abers interval.

    Used to gate the ML head: a quote outside ``[p0, p1]`` is rejected.
    Non-finite inputs fail closed (False).
    """
    try:
        a, b, p = float(p0), float(p1), float(p_mkt)
    except (TypeError, ValueError):
        return False
    if not (np.isfinite(a) and np.isfinite(b) and np.isfinite(p)):
        return False
    lo, hi = (a, b) if a <= b else (b, a)
    return lo <= p <= hi
