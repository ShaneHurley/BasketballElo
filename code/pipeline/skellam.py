"""Skellam distribution cover probability from predicted scores."""
from __future__ import annotations

import numpy as np


def cover_prob_skellam(
    pred_home: float,
    pred_away: float,
    market_spread: float,
    direction: str = "Home",
) -> float:
    """P(bet side covers) using Poisson means = predicted scores."""
    mu1 = max(float(pred_home), 0.5)
    mu2 = max(float(pred_away), 0.5)
    spread = float(market_spread)
    # Home covers when actual_margin + market_spread > 0  =>  margin > -spread
    threshold = int(np.floor(-spread))
    try:
        from scipy.stats import skellam
        p_home = float(1.0 - skellam.cdf(threshold, mu1, mu2))
    except ImportError:
        diff_mean = mu1 - mu2
        diff_std = max(np.sqrt(mu1 + mu2), 1.0)
        z = (threshold + 0.5 - diff_mean) / diff_std
        p_home = float(1.0 - _norm_cdf(z))
    if direction == "Away":
        return 1.0 - p_home
    return p_home


def _norm_cdf(z: float) -> float:
    return 0.5 * (1.0 + float(np.tanh(z * 0.7978845608)))
