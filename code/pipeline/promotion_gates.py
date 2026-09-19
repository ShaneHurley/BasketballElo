"""Promotion gates for roadmap ablations (MAE / proper scores first).

ROI and ATS are secondary and require authoritative CLV provenance before
any betting-claim promotion. CLV samples without ``promotion_eligible`` odds
must not be treated as evidence of improvement.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np


REQUIRED_BLIND_SEASON_WINS = 3


def market_baseline_beats_model(metrics: dict, *, tol: float = 0.0) -> bool:
    """True when market MAE is better (lower) than model on matched games."""
    model = float(metrics.get("spread_mae", np.nan))
    market = float(metrics.get("market_spread_mae", np.nan))
    if not np.isfinite(model) or not np.isfinite(market):
        return False
    return market + tol < model


def beats_market_and_baseline(
    candidate: dict,
    baseline: dict,
    *,
    mae_tol: float = 0.05,
    total_mae_tol: float = 0.25,
    brier_tol: float = 0.005,
    log_loss_tol: float = 0.01,
) -> bool:
    """Candidate must beat both current baseline and market on primary scores."""
    c_mae = float(candidate.get("spread_mae", np.nan))
    b_mae = float(baseline.get("spread_mae", np.nan))
    mkt = float(candidate.get("market_spread_mae", baseline.get("market_spread_mae", np.nan)))
    if not np.isfinite(c_mae) or not np.isfinite(b_mae):
        return False
    if c_mae > b_mae - mae_tol:
        return False
    if np.isfinite(mkt) and c_mae > mkt - mae_tol:
        return False

    c_tot = candidate.get("total_mae")
    b_tot = baseline.get("total_mae")
    if c_tot is not None and b_tot is not None and np.isfinite(c_tot) and np.isfinite(b_tot):
        if float(c_tot) > float(b_tot) + total_mae_tol:
            return False

    for key, tol in (("brier", brier_tol), ("log_loss", log_loss_tol), ("crps", 0.5)):
        cv = candidate.get(key)
        bv = baseline.get(key)
        if cv is None or bv is None:
            continue
        if np.isfinite(cv) and np.isfinite(bv) and float(cv) > float(bv) + tol:
            return False
    return True


def multi_season_gate(
    per_season: Iterable[dict],
    baseline_per_season: Iterable[dict],
    *,
    min_wins: int = REQUIRED_BLIND_SEASON_WINS,
    metric: str = "spread_mae",
) -> dict:
    """Require improvement on ``min_wins`` blind seasons with no material collapse.

    ``per_season`` / ``baseline_per_season`` are aligned iterables of metric dicts
    (one per season). A season "wins" when candidate metric is strictly better.
    """
    c_list = list(per_season)
    b_list = list(baseline_per_season)
    n = min(len(c_list), len(b_list))
    wins = 0
    collapses = 0
    deltas = []
    for i in range(n):
        c = float(c_list[i].get(metric, np.nan))
        b = float(b_list[i].get(metric, np.nan))
        if not np.isfinite(c) or not np.isfinite(b):
            continue
        delta = c - b  # negative = better for MAE/Brier/log_loss
        deltas.append(delta)
        if delta < 0:
            wins += 1
        # Material degradation: >0.5 pts MAE (or 5% relative for probabilities)
        if metric.endswith("mae") and delta > 0.5:
            collapses += 1
        elif not metric.endswith("mae") and b > 0 and delta > 0.05 * abs(b):
            collapses += 1
    return {
        "wins": wins,
        "n_seasons": n,
        "collapses": collapses,
        "passed": wins >= min_wins and collapses == 0 and n >= min_wins,
        "mean_delta": float(np.mean(deltas)) if deltas else float("nan"),
    }


def clv_promotion_allowed(provenance: dict | None, n_finite_clv: int, *, min_n: int = 200) -> bool:
    """Block ROI/CLV claims without authoritative decision/close provenance."""
    if not provenance or not provenance.get("promotion_eligible"):
        return False
    return int(n_finite_clv or 0) >= int(min_n)


def interval_coverage_credible(
    metrics: dict,
    *,
    target: float = 0.80,
    tol: float = 0.08,
) -> bool:
    """True when reported interval coverage is near the nominal target.

    Do not loosen ``MAX_QUANTILE_WIDTH`` until this passes on locked folds.
    """
    cov = float(metrics.get("interval_coverage", np.nan))
    if not np.isfinite(cov):
        return False
    return abs(cov - float(target)) <= float(tol)


def margin_dispersion_ok(
    metrics: dict,
    *,
    min_ratio: float = 0.45,
    max_ratio: float = 0.95,
) -> bool:
    """Predicted |margin| should not collapse toward zero vs market implied."""
    r = float(metrics.get("margin_dispersion_ratio", np.nan))
    if not np.isfinite(r):
        return False
    return float(min_ratio) <= r <= float(max_ratio)


def market_error_corr_positive(metrics: dict, *, min_corr: float = 0.02) -> bool:
    """True when predicted edges correlate positively with realized market errors."""
    c = float(metrics.get("market_error_corr", np.nan))
    if not np.isfinite(c):
        return False
    return c >= float(min_corr)


def score_algebra_ok(metrics: dict, *, min_pct: float = 0.99) -> bool:
    p = float(metrics.get("score_algebra_ok_pct", np.nan))
    if not np.isfinite(p):
        return False
    return p >= float(min_pct)


def policy_retune_allowed(
    metrics: dict,
    *,
    accuracy_ok: bool | None = None,
    odds_provenance: dict | None = None,
) -> bool:
    """Allow confidence/edge policy retunes only after accuracy/uncertainty gates.

    Never loosens ``MIN_CONFIDENCE_SCORE`` / ``MAX_QUANTILE_WIDTH`` / min edge.
    Tip-proxy / non-promotion_eligible runs remain research-only (ROI blocked).
    """
    if accuracy_ok is None:
        accuracy_ok = bool(
            interval_coverage_credible(metrics, target=0.80)
            and market_error_corr_positive(metrics, min_corr=0.02)
            and margin_dispersion_ok(metrics)
        )
    if not accuracy_ok:
        return False
    if odds_provenance is not None and not odds_provenance.get("promotion_eligible"):
        # Policy search may still run as research, but promotion of floors is blocked.
        return False
    return True


def tip_proxy_roi_blocked(odds_provenance: dict | None) -> bool:
    """True when ROI/CLV claims must not be made from this odds source."""
    if not odds_provenance:
        return True
    if odds_provenance.get("used_tip_proxy_fallback"):
        return True
    if odds_provenance.get("quote_source") == "tip_proxy":
        return True
    return not bool(odds_provenance.get("promotion_eligible"))


