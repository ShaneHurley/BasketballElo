"""Tasks 037-040: centralized bet grading, exact-price profit, and CLV.

Canonical win/loss/push outcomes and exact-price P&L live here so
``pipeline/metrics.py`` and ``pipeline/stake_profiles.py`` cannot drift into
incompatible push/CLV definitions again.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.market import american_to_decimal
from pipeline.market_snapshots import point_clv, price_clv


def grade_spread_bet(actual_margin, market_home_spread, side: str) -> str:
    """Return ``win``, ``loss``, or ``push`` for a spread bet.

    ``actual_margin`` is home_score - away_score. ``market_home_spread`` is the
    home line (negative when home is favored). ``side`` is ``Home`` or ``Away``.
    """
    if pd.isna(actual_margin) or pd.isna(market_home_spread):
        raise ValueError("actual_margin and market_home_spread are required")
    side_l = str(side).strip().lower()
    if side_l not in ("home", "away"):
        raise ValueError(f"unknown spread side: {side!r}")
    cover = float(actual_margin) + float(market_home_spread)
    if cover == 0.0:
        return "push"
    home_covers = cover > 0.0
    if side_l == "home":
        return "win" if home_covers else "loss"
    return "win" if not home_covers else "loss"


def grade_total_bet(actual_total, market_total, side: str) -> str:
    """Return ``win``, ``loss``, or ``push`` for an over/under bet."""
    if pd.isna(actual_total) or pd.isna(market_total):
        raise ValueError("actual_total and market_total are required")
    side_l = str(side).strip().lower()
    if side_l not in ("over", "under"):
        raise ValueError(f"unknown total side: {side!r}")
    diff = float(actual_total) - float(market_total)
    if diff == 0.0:
        return "push"
    went_over = diff > 0.0
    if side_l == "over":
        return "win" if went_over else "loss"
    return "win" if not went_over else "loss"


def bet_side_point_clv(decision_home_spread, close_home_spread, side: str) -> float:
    """Bet-side point CLV (Task 038 / Phase 3).

    Home: ``decision_home_spread - close_home_spread``
    Away: ``close_home_spread - decision_home_spread``
    """
    return point_clv(decision_home_spread, close_home_spread, side)


def bet_side_price_clv(decision_fair_prob, close_fair_prob) -> float:
    """Probability-domain CLV; never compound with point CLV."""
    return price_clv(decision_fair_prob, close_fair_prob)


def _to_decimal_odds(price) -> float:
    """Accept American or decimal odds; return decimal payout multiplier."""
    if pd.isna(price):
        return float("nan")
    p = float(price)
    if p == 0.0:
        return float("nan")
    # Decimal odds are typically in (1, ~20]; American are <= -100 or >= +100.
    if abs(p) >= 100.0 or p <= -100.0:
        return float(american_to_decimal(p))
    if p > 1.0:
        return p
    return float("nan")


def exact_price_profit(stake: float, outcome: str, price) -> float:
    """Profit using the actual accepted side price (Task 039).

    ``outcome`` is ``win`` / ``loss`` / ``push``. Pushes return 0 (Task 040).
    ``price`` may be American (e.g. -110, +150) or decimal (e.g. 1.91).
    """
    stake = float(stake or 0.0)
    if stake <= 0.0:
        return 0.0
    outcome_l = str(outcome).strip().lower()
    if outcome_l == "push":
        return 0.0
    if outcome_l == "loss":
        return -stake
    if outcome_l != "win":
        raise ValueError(f"unknown outcome: {outcome!r}")
    dec = _to_decimal_odds(price)
    if not np.isfinite(dec) or dec <= 1.0:
        raise ValueError(f"invalid price for exact_price_profit: {price!r}")
    return stake * (dec - 1.0)


def hit_rate(outcomes) -> float:
    """Win rate excluding pushes from the denominator (Task 040)."""
    arr = [str(o).lower() for o in outcomes if str(o).lower() != "push"]
    if not arr:
        return float("nan")
    return float(sum(1 for o in arr if o == "win") / len(arr))
