"""Task 031: `pipeline/features.py` and `pipeline/simulate.py` must never use
the *current* game's actual/first-PBP lineup as a T-60 feature. Only prior
rotation history (previously completed games) or timestamped projections may
feed the T-60 feature row. This is checked by proving that changing a game's
own actual starters (holding every earlier game fixed) does not change that
game's stored T-60 feature row.
"""
import pandas as pd

from pipeline.config import DEFAULT_LEAGUE_XPPP
from pipeline.features import generate_features
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.ratings import PlayerRatingTracker
from pipeline.trackers import PaceTracker


def _fresh_trackers():
    elo = PlayerRatingTracker(league_xppp=DEFAULT_LEAGUE_XPPP)
    hier = HierarchicalPossessionEngine()
    pace = PaceTracker(team_window=10, league_window=100)
    return elo, hier, pace


def _stint_row(gid, date, home, away, period, home_players, away_players):
    return {
        "GAME_ID": gid,
        "game_date": pd.Timestamp(date),
        "home_team": home,
        "away_team": away,
        "PERIOD": period,
        "stint_id": period,
        "HOME_players": home_players,
        "AWAY_players": away_players,
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
    }


def _two_game_stints(second_game_home_players="11-12-13-14-15"):
    """Game 1 establishes rotation history for GSW/LAL; game 2 is the game
    under test whose *own* actual starters we perturb."""
    rows = [
        _stint_row("0022500001", "2025-11-01", "GSW", "LAL", 1, "1-2-3-4-5", "6-7-8-9-10"),
        _stint_row("0022500001", "2025-11-01", "GSW", "LAL", 2, "1-2-3-4-5", "6-7-8-9-10"),
        _stint_row("0022500002", "2025-11-03", "GSW", "LAL", 1, second_game_home_players, "6-7-8-9-10"),
        _stint_row("0022500002", "2025-11-03", "GSW", "LAL", 2, second_game_home_players, "6-7-8-9-10"),
    ]
    return pd.DataFrame(rows)


def test_changing_actual_starters_after_t60_does_not_change_t60_feature_row():
    """Task 031 pass condition: two runs identical except for game 2's own
    actual/current-game PBP starters must produce an identical T-60 feature
    row for game 2 (and game 1, which cannot see game 2 at all)."""
    stints_a = _two_game_stints(second_game_home_players="11-12-13-14-15")
    stints_b = _two_game_stints(second_game_home_players="21-22-23-24-25")

    elo_a, hier_a, pace_a = _fresh_trackers()
    feats_a = generate_features(stints_a, hier_a, elo_a, pace_a, update_engines=True)

    elo_b, hier_b, pace_b = _fresh_trackers()
    feats_b = generate_features(stints_b, hier_b, elo_b, pace_b, update_engines=True)

    lineup_sensitive_cols = [
        "h_new_starters", "h_rating_uncertainty", "h_experience",
        "h_elo_off", "h_elo_def", "elo_margin", "hier_margin",
    ]
    game2_a = feats_a[feats_a["GAME_ID"] == "0022500002"].iloc[0]
    game2_b = feats_b[feats_b["GAME_ID"] == "0022500002"].iloc[0]
    for col in lineup_sensitive_cols:
        assert game2_a[col] == game2_b[col], (
            f"T-60 feature '{col}' for game 0022500002 changed when only "
            "that game's own actual starters changed -- actual/current-game "
            "lineup leaked into a T-60 feature row"
        )

    # Game 1 (earlier than the perturbation) must be completely unaffected.
    game1_a = feats_a[feats_a["GAME_ID"] == "0022500001"].iloc[0]
    game1_b = feats_b[feats_b["GAME_ID"] == "0022500001"].iloc[0]
    pd.testing.assert_series_equal(game1_a, game1_b, check_names=False)


def test_first_season_game_has_no_projected_starters_from_own_pbp():
    """A team's very first game of a season has no prior rotation history,
    so its pre-game roster is genuinely unknown (Rule 5) -- it must not be
    silently filled in from that same game's own actual PBP lineup."""
    from pipeline.trackers import RotationLineupTracker

    stints = _two_game_stints()
    elo, hier, pace = _fresh_trackers()
    rotation_tracker = RotationLineupTracker(window_games=10, top_n=8)
    feats = generate_features(
        stints, hier, elo, pace, update_engines=True, rotation_tracker=rotation_tracker,
    )
    game1 = feats[feats["GAME_ID"] == "0022500001"].iloc[0]
    # With no prior history, `h_new_starters`/`a_new_starters` must reflect
    # an *empty* projected roster (0 known players => 0/1 by definition),
    # not the 5 actual players who took the floor in this same game.
    assert game1["h_new_starters"] == 0.0
    assert game1["a_new_starters"] == 0.0
