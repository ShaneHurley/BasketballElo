"""Tests for stint context and Elo context multipliers."""
import pandas as pd

from pipeline.stint_context import build_stint_context, extract_crew_id
from pipeline.ratings import PlayerRatingTracker


def test_build_stint_context_clutch():
    row = {
        "possessions": 10,
        "PERIOD": 4,
        "HOME_SCORE_START": 100,
        "AWAY_SCORE_START": 95,
        "home_tov": 2,
        "away_tov": 1,
        "home_stl": 1,
        "away_stl": 0,
        "home_blks": 0,
        "away_blks": 1,
        "home_tovs_forced": 1,
        "away_tovs_forced": 0,
        "home_fouls_drawn": 2,
        "away_fouls_drawn": 0,
        "home_fta": 4,
        "away_fta": 0,
        "home_fga": 8,
        "away_fga": 6,
        "home_3pa": 4,
        "away_3pa": 2,
        "home_pts": 12,
        "away_pts": 10,
        "home_xpts": 11,
        "away_xpts": 10,
        "garbage": False,
    }
    ctx = build_stint_context(row)
    assert ctx["clutch"] is True
    assert ctx["home_tov_rate"] == 0.2
    assert ctx["home_luck_ppp"] == 0.1


def test_extract_crew_id():
    group = pd.DataFrame({"official": ["A", "A", "B"]})
    assert extract_crew_id(group) == "A"


def test_context_multiplier_clutch_boost():
    tracker = PlayerRatingTracker()
    ctx = {"clutch": True, "garbage": False}
    m = tracker._context_multiplier(ctx, "home")
    assert m >= tracker.cfg["clutch_boost"]
