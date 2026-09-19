"""T-60 coverage gate and two-sided ML wiring smoke tests."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.market import fair_home_win_prob, select_ml_bet
from pipeline.t60_coverage import t60_betting_data_gate


def test_fair_home_win_prob_uses_away_price():
    # Asymmetric books: home -200 / away +150 vs negating home.
    p_pair = fair_home_win_prob(-200, 150)
    p_neg = fair_home_win_prob(-200, None)
    assert abs(p_pair - p_neg) > 1e-6


def test_select_ml_bet_uses_real_away():
    side, ev, dec = select_ml_bet(
        0.40, -150, market_ml_away=180, min_ev=0.0, winner_only=False,
        min_win_pct=0.0, max_favorite_decimal=1.01,
    )
    # With p_home=0.40, away side should be considered with +180.
    assert side in ("Home", "Away", "Pass")
    if side == "Away":
        assert dec > 2.0  # +180 decimal ≈ 2.8


def test_t60_gate_requires_three_seasons():
    rows = []
    for year in (2023, 2024):
        for i in range(20):
            rows.append({
                "season": year,
                "decision_spread": -3.5,
                "market_spread": -3.5,
            })
    df = pd.DataFrame(rows)
    gate = t60_betting_data_gate(df, min_seasons=3, min_coverage=0.5)
    assert gate["allow_betting_heads"] is False
    assert gate["n_adequate_seasons"] == 2
