"""Schema / factory smoke tests for Epic 10.1 (synthetic bench)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.synth.factories import (
    make_game,
    make_game_sequence,
    make_odds,
    make_season,
    make_stint,
    make_stint_frame,
)
from tests.synth.presets import BLOWOUT_60, ONE_SIDED_ODDS_FEED, ZERO_POSSESSION_FT_STINT


@pytest.mark.bench
def test_make_stint_required_columns():
    row = make_stint()
    required = {
        "GAME_ID", "HOME_players", "AWAY_players", "possessions",
        "home_xpts", "away_xpts", "PERIOD", "garbage",
        "HOME_SCORE_START", "AWAY_SCORE_START", "HOME_SCORE_END", "AWAY_SCORE_END",
    }
    assert required.issubset(row.keys())
    assert row["HOME_players"].count("-") == 4
    assert row["AWAY_players"].count("-") == 4


@pytest.mark.bench
def test_make_stint_frame_deterministic():
    a = make_stint_frame(n=5, seed=42)
    b = make_stint_frame(n=5, seed=42)
    pd.testing.assert_frame_equal(a, b)
    c = make_stint_frame(n=5, seed=99)
    assert not a.equals(c)


@pytest.mark.bench
def test_presets_blowout_and_zero_poss():
    assert BLOWOUT_60["garbage"] is True
    assert BLOWOUT_60["PERIOD"] == 4
    assert abs(BLOWOUT_60["HOME_SCORE_END"] - BLOWOUT_60["AWAY_SCORE_END"]) >= 60
    assert ZERO_POSSESSION_FT_STINT["possessions"] == 0.0
    assert ZERO_POSSESSION_FT_STINT["home_pts"] == 1.0


@pytest.mark.bench
def test_one_sided_odds_feed_preset():
    assert np.isnan(ONE_SIDED_ODDS_FEED["market_ml_away"])
    assert np.isfinite(ONE_SIDED_ODDS_FEED["market_ml_home"])


@pytest.mark.bench
def test_make_game_and_odds_shapes():
    g = make_game(closing_spread=-4.0, market_spread=-3.5)
    assert g["CLOSING_SPREAD"] != g["MARKET_SPREAD"]
    o = make_odds(one_sided=False, ml_away=130)
    assert o["market_ml_away"] == 130.0


# ── Task 10.1.1: game-sequence and season builders ─────────────────────────

_TEAMS4 = ["GSW", "LAL", "BOS", "NYK"]


@pytest.mark.bench
def test_make_game_sequence_deterministic():
    a = make_game_sequence(12, _TEAMS4, "2025-10-01", 7)
    b = make_game_sequence(12, _TEAMS4, "2025-10-01", 7)
    assert a == b
    c = make_game_sequence(12, _TEAMS4, "2025-10-01", 8)
    assert a != c


@pytest.mark.bench
def test_make_game_sequence_chronology_and_spacing():
    games = make_game_sequence(30, ["GSW", "LAL", "BOS"], "2025-10-15", 3)
    assert len(games) == 30
    dates = [g["game_date"] for g in games]
    assert dates[0] == pd.Timestamp("2025-10-15")
    assert dates == sorted(dates)
    for prev, cur in zip(dates, dates[1:]):
        gap_days = (cur - prev).total_seconds() / 86400.0
        assert 1.0 <= gap_days <= 3.0


@pytest.mark.bench
def test_make_game_sequence_no_self_games_and_schema():
    games = make_game_sequence(25, ["GSW", "LAL"], "2025-10-01", 11)
    assert len(games) == 25
    for g in games:
        assert g["home_team"] != g["away_team"]
        assert {g["home_team"], g["away_team"]} <= {"GSW", "LAL"}
    # Rows carry the full make_game schema and unique game ids.
    assert set(make_game().keys()) <= set(games[0].keys())
    assert len({g["GAME_ID"] for g in games}) == 25


@pytest.mark.bench
def test_make_game_sequence_seed_streams_distinct():
    """Adversarial: nearby seeds must not collide onto the same schedule."""
    sigs = set()
    for s in range(8):
        seq = make_game_sequence(8, _TEAMS4, "2025-10-01", s)
        sig = tuple((g["home_team"], g["away_team"], g["game_date"]) for g in seq)
        sigs.add(sig)
    assert len(sigs) == 8


@pytest.mark.bench
def test_make_game_sequence_edge_cases():
    one = make_game_sequence(1, _TEAMS4, "2025-12-25", 0)
    assert len(one) == 1 and one[0]["game_date"] == pd.Timestamp("2025-12-25")
    assert make_game_sequence(0, _TEAMS4, "2025-10-01", 0) == []
    with pytest.raises(ValueError):
        make_game_sequence(5, ["GSW"], "2025-10-01", 0)
    with pytest.raises(ValueError):
        make_game_sequence(5, ["GSW", "GSW"], "2025-10-01", 0)


@pytest.mark.bench
def test_make_season_team_game_counts_and_no_self_games():
    teams = ["GSW", "LAL", "BOS", "NYK", "DAL", "PHX"]
    df = make_season(teams, 10, 5)
    assert isinstance(df, pd.DataFrame)
    assert len(df) == len(teams) * 10 // 2
    assert (df["home_team"] != df["away_team"]).all()
    for t in teams:
        played = int((df["home_team"] == t).sum() + (df["away_team"] == t).sum())
        assert played == 10, f"{t} played {played}, expected 10"


@pytest.mark.bench
def test_make_season_deterministic():
    a = make_season(_TEAMS4, 6, 42)
    b = make_season(_TEAMS4, 6, 42)
    pd.testing.assert_frame_equal(a, b)
    c = make_season(_TEAMS4, 6, 43)
    assert not a.equals(c)


@pytest.mark.bench
def test_make_season_chronological_spacing():
    df = make_season(_TEAMS4, 6, 9)
    dates = list(df["game_date"])
    assert dates == sorted(dates)
    gaps = [(c - p).total_seconds() / 86400.0 for p, c in zip(dates, dates[1:])]
    assert all(1.0 <= g <= 3.0 for g in gaps)


@pytest.mark.bench
def test_make_season_edge_cases():
    # Two teams: only mutual matchups possible, still exactly n games each.
    df2 = make_season(["GSW", "LAL"], 4, 1)
    assert len(df2) == 4
    assert (df2["home_team"] != df2["away_team"]).all()
    # Odd team count with even games-per-team works.
    df3 = make_season(["GSW", "LAL", "BOS"], 4, 2)
    for t in ["GSW", "LAL", "BOS"]:
        assert int((df3["home_team"] == t).sum() + (df3["away_team"] == t).sum()) == 4
    with pytest.raises(ValueError):
        make_season(["GSW"], 4, 1)
    with pytest.raises(ValueError):
        make_season(["GSW", "GSW", "LAL"], 2, 1)  # duplicate identities
    with pytest.raises(ValueError):
        make_season(["GSW", "LAL", "BOS"], 3, 1)  # odd total team-games
    with pytest.raises(ValueError):
        make_season(_TEAMS4, 0, 1)
