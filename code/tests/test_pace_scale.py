"""Possession-scale contracts for PaceTracker and matchup/PPP consumers."""
from __future__ import annotations

import numpy as np

from pipeline.matchup_rating import forecast_from_lineup_elos
from pipeline.model import elo_implied_total_from_row
from pipeline.trackers import PaceTracker


def test_expected_pace_is_shared_hundred_scale():
    pace = PaceTracker(team_window=5, league_window=50, per_team=True)
    for _ in range(8):
        pace.update_pace("BOS", "NYK", 98.0, 102.0)
    exp = pace.get_expected_pace("BOS", "NYK")
    assert 90.0 <= exp <= 110.0, f"expected ~100 shared possessions, got {exp}"


def test_legacy_total_history_converts_to_shared():
    pace = PaceTracker(team_window=5, league_window=50, per_team=False)
    for _ in range(5):
        # legacy stores whole-game totals (~200) for both sides
        pace.update_pace("BOS", "NYK", 100.0, 100.0)
    exp = pace.get_expected_pace("BOS", "NYK")
    assert 90.0 <= exp <= 110.0, f"legacy path should convert to ~100, got {exp}"


def test_matchup_total_realistic_with_shared_poss():
    f = forecast_from_lineup_elos(1550, 1480, 1520, 1490, 100.0, home_court_pp100=2.0)
    assert 180.0 <= f.total <= 260.0
    assert abs(f.margin) < 40.0


def test_elo_implied_total_uses_shared_poss_not_double():
    row = {
        "exp_poss": 100.0,
        "h_elo_off": 1520, "h_elo_def": 1480,
        "a_elo_off": 1500, "a_elo_def": 1500,
    }
    tot = elo_implied_total_from_row(row)
    assert tot is not None
    assert 180.0 <= tot <= 260.0


def test_doubled_poss_would_inflate_total():
    """Regression guard: ~200 possessions must not be treated as shared."""
    row_ok = {
        "exp_poss": 100.0,
        "h_elo_off": 1500, "h_elo_def": 1500,
        "a_elo_off": 1500, "a_elo_def": 1500,
    }
    row_bad = dict(row_ok)
    row_bad["exp_poss"] = 200.0
    ok = elo_implied_total_from_row(row_ok)
    bad = elo_implied_total_from_row(row_bad)
    assert abs(bad - 2.0 * ok) < 1e-6


def test_pace_distribution_emits_uncertainty():
    pace = PaceTracker(team_window=5, league_window=50, per_team=True)
    for i in range(6):
        pace.update_pace("BOS", "NYK", 95.0 + i, 100.0 + i)
    dist = pace.get_expected_pace_distribution("BOS", "NYK")
    assert "pace_mean" in dist and "pace_std" in dist
    assert dist["pace_std"] >= 1.0
    assert dist["pace_q10"] <= dist["pace_q50"] <= dist["pace_q90"]
