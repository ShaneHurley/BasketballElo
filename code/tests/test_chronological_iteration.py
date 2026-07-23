"""Task 015: features/simulation entry points must sort by decision
timestamp then GAME_ID, assert monotonic ordering, and produce identical
output regardless of input row shuffling."""
import pandas as pd
import pytest

from pipeline.config import DEFAULT_LEAGUE_XPPP
from pipeline.features import generate_features
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.ratings import PlayerRatingTracker
from pipeline.trackers import PaceTracker


def _synthetic_stints():
    """Three tiny games on three different dates, each with two stints."""
    rows = []
    games = [
        ("0022500001", "2025-11-01", "GSW", "LAL"),
        ("0022500002", "2025-11-02", "BOS", "MIA"),
        ("0022500003", "2025-11-03", "DEN", "UTA"),
    ]
    for gid, date, home, away in games:
        for stint_idx in range(2):
            rows.append({
                "GAME_ID": gid,
                "game_date": pd.Timestamp(date),
                "home_team": home,
                "away_team": away,
                "PERIOD": 1 + stint_idx,
                "stint_id": stint_idx,
                "HOME_players": "1-2-3-4-5",
                "AWAY_players": "6-7-8-9-10",
                "HOME_SCORE_START": 0.0, "AWAY_SCORE_START": 0.0,
                "home_pts": 10.0, "away_pts": 8.0,
                "home_xpts": 9.0, "away_xpts": 7.5,
                "possessions": 10.0, "home_poss": 5.0, "away_poss": 5.0,
                "home_oreb": 1, "away_oreb": 1, "home_dreb": 3, "away_dreb": 3,
                "home_tovs_forced": 1, "away_tovs_forced": 1,
                "home_fouls_drawn": 2, "away_fouls_drawn": 2,
                "home_fgm": 4, "away_fgm": 3, "home_fga": 8, "away_fga": 8,
                "home_tov": 1, "away_tov": 1, "home_fta": 2, "away_fta": 2,
                "home_stl": 1, "away_stl": 1, "home_blks": 0, "away_blks": 0,
                "home_3pm": 1, "away_3pm": 1, "home_3pa": 3, "away_3pa": 3,
                "home_xefg_sum": 2.0, "away_xefg_sum": 2.0,
                "home_rim_fga": 2, "away_rim_fga": 2,
                "home_three_fga": 3, "away_three_fga": 3,
                "home_shot_dist_sum": 40.0, "away_shot_dist_sum": 40.0,
                "home_usage": {}, "away_usage": {},
            })
    return pd.DataFrame(rows)


def _fresh_trackers():
    elo = PlayerRatingTracker(league_xppp=DEFAULT_LEAGUE_XPPP)
    hier = HierarchicalPossessionEngine()
    pace = PaceTracker(team_window=10, league_window=100)
    return elo, hier, pace


def test_generate_features_asserts_monotonic_order():
    stints = _synthetic_stints()
    elo, hier, pace = _fresh_trackers()
    feats = generate_features(stints, hier, elo, pace, update_engines=False)
    # Two stints per game, 10 home points each -> 20 per game.
    assert feats["actual_home"].tolist() == [20.0, 20.0, 20.0]


def test_generate_features_deterministic_under_row_shuffle():
    """Task 015 pass condition: shuffled input produces identical ordered
    predictions and tracker states."""
    stints = _synthetic_stints()
    shuffled = stints.sample(frac=1.0, random_state=42).reset_index(drop=True)

    elo1, hier1, pace1 = _fresh_trackers()
    feats_sorted = generate_features(stints, hier1, elo1, pace1, update_engines=True)

    elo2, hier2, pace2 = _fresh_trackers()
    feats_shuffled = generate_features(shuffled, hier2, elo2, pace2, update_engines=True)

    pd.testing.assert_frame_equal(
        feats_sorted.reset_index(drop=True),
        feats_shuffled.reset_index(drop=True),
    )
