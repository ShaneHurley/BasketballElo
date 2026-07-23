"""Pythagorean features (no leakage) and market de-vig helpers."""
from datetime import timedelta

import pandas as pd

from pipeline.market import devig_two_way, fair_home_win_prob, fair_probs_from_home_ml
from pipeline.teamstats import TeamFormTracker, pythagorean_win_pct, team_game_form


def test_devig_two_way_sums_to_one():
    a, b = devig_two_way(0.55, 0.55)
    assert abs(a + b - 1.0) < 1e-9
    assert a == b == 0.5


def test_fair_probs_from_home_ml():
    p_home, p_away = fair_probs_from_home_ml(-110)
    assert p_home > 0.5
    assert abs(p_home + p_away - 1.0) < 1e-9
    assert fair_home_win_prob(-110) == p_home


def test_pythagorean_win_pct_symmetric():
    assert abs(pythagorean_win_pct(110.0, 110.0) - 0.5) < 1e-6
    assert pythagorean_win_pct(120.0, 100.0) > 0.5


def test_team_form_pythagorean_no_leakage():
    """Current game stats must not enter rolling Pythagorean features."""
    tracker = TeamFormTracker(window=10, prev_season_weight=0.4)
    season = 2026
    d0 = pd.Timestamp("2026-01-01")
    gs_win = {"act_h": 120.0, "act_a": 100.0, "home_poss": 100.0, "away_poss": 100.0, "tot_poss": 200.0}
    gs_loss = {"act_h": 90.0, "act_a": 110.0, "home_poss": 100.0, "away_poss": 100.0, "tot_poss": 200.0}

    # First game: no prior history → defaults
    g0 = d0
    assert tracker.get("BOS", season, g0)["pythag_win_pct"] == 0.5

    for i in range(6):
        gdate = d0 + timedelta(days=i)
        gs = gs_win if i % 2 == 0 else gs_loss
        tracker.update("BOS", gdate, season, team_game_form(gs, "home"))

    gdate = d0 + timedelta(days=6)
    pre = tracker.get("BOS", season, gdate)
    assert pre["actual_win_pct"] > 0.4
    assert pre["pythag_win_pct"] > 0.4
    assert abs(pre["pythag_residual"]) < 0.5

    # Same-day update must not affect same-day get (strict date filter gd >= current_date)
    tracker.update("BOS", gdate, season, team_game_form(gs_win, "home"))
    same_day = tracker.get("BOS", season, gdate)
    assert same_day["actual_win_pct"] == pre["actual_win_pct"]
    assert same_day["pythag_win_pct"] == pre["pythag_win_pct"]
