"""Schema / factory smoke tests for Epic 10.1 (synthetic bench)."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from tests.synth.factories import make_game, make_odds, make_stint, make_stint_frame
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
