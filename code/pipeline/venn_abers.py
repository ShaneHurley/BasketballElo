"""Minimal binary Venn-Abers interval from calibration scores."""
from __future__ import annotations

import numpy as np
from sklearn.isotonic import IsotonicRegression


def scores_to_interval(scores_cal: np.ndarray, labels_cal: np.ndarray, scores_test: np.ndarray):
    """Return (p0, p1, optimal_point, width) arrays for test scores."""
    scores_cal = np.asarray(scores_cal, dtype=float)
    labels_cal = np.asarray(labels_cal, dtype=int)
    scores_test = np.asarray(scores_test, dtype=float)
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
