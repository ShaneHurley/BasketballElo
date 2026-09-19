"""Tests for matchup rating invariants and PP100 forecasts."""
from __future__ import annotations

import numpy as np

from pipeline.matchup_rating import (
    assert_defense_lowers_opp_scoring,
    assert_equal_lineups_home_court_only,
    assert_home_away_swap_reverses_margin,
    assert_offense_raises_own_scoring,
    forecast_from_lineup_elos,
    moment_match_scenarios,
    residual_update_signs,
    elo_to_pp100_delta,
    pp100_delta_to_elo,
)


def test_pp100_elo_roundtrip():
    for elo in (1400.0, 1500.0, 1600.0):
        assert abs(pp100_delta_to_elo(elo_to_pp100_delta(elo)) - elo) < 1e-9


def test_synthetic_invariants():
    assert_equal_lineups_home_court_only()
    assert_offense_raises_own_scoring()
    assert_defense_lowers_opp_scoring()
    assert_home_away_swap_reverses_margin()


def test_moment_match_mixture_mean():
    a = forecast_from_lineup_elos(1550, 1500, 1500, 1500, 100.0)
    b = forecast_from_lineup_elos(1450, 1500, 1500, 1500, 100.0)
    mixed = moment_match_scenarios([a, b], weights=[0.5, 0.5])
    assert abs(mixed.margin - 0.5 * (a.margin + b.margin)) < 1e-6
    assert mixed.uncertainty >= 0


def test_residual_update_signs():
    signs = residual_update_signs(1.2, 1.1, 1.0, 1.05)
    assert signs["home_offense_error"] > 0
    assert signs["away_offense_error"] < 0
    assert signs["away_defense_error_vs_home"] == -signs["home_offense_error"]
