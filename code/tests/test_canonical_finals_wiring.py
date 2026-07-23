"""Task 014: canonical finals from Task 006 must override stint-summed
act_h/act_a labels wherever precompute_game_team_stats is used (features.py,
simulate.py both consume it directly)."""
import pandas as pd

from pipeline.game_results import attach_canonical_finals
from pipeline.teamstats import precompute_game_team_stats


def _stints_df():
    return pd.DataFrame([
        {"GAME_ID": "0022500001", "home_pts": 40, "away_pts": 38, "possessions": 20,
         "home_poss": 10, "away_poss": 10, "home_xpts": 20, "away_xpts": 19},
        {"GAME_ID": "0022500001", "home_pts": 40, "away_pts": 38, "possessions": 20,
         "home_poss": 10, "away_poss": 10, "home_xpts": 20, "away_xpts": 19},
        {"GAME_ID": "0022500002", "home_pts": 50, "away_pts": 45, "possessions": 22,
         "home_poss": 11, "away_poss": 11, "home_xpts": 24, "away_xpts": 22},
    ])


def test_precompute_game_stats_uses_stint_sum_without_canonical():
    stats = precompute_game_team_stats(_stints_df())
    assert stats["0022500001"]["act_h"] == 80  # 40 + 40, no canonical override
    assert stats["0022500001"]["act_a"] == 76


def test_precompute_game_stats_overrides_with_canonical_finals():
    """This is the Task 014 regression test: the (deliberately corrupted)
    garbage-time-zeroed stint sum for game 1 disagrees with its canonical
    final score, and the canonical value must win."""
    canonical_df = pd.DataFrame([
        {"GAME_ID": "0022500001", "home_team": "GSW", "away_team": "LAL",
         "season_type": "Regular", "raw_date": pd.Timestamp("2025-11-01"), "tip_utc": pd.NaT,
         "final_home_score": 97, "final_away_score": 90,
         "final_home_score_check": 97, "final_away_score_check": 90, "scores_match": True},
        # Game 2 failed verification -> must NOT be used as an override.
        {"GAME_ID": "0022500002", "home_team": "BOS", "away_team": "MIA",
         "season_type": "Regular", "raw_date": pd.Timestamp("2025-11-02"), "tip_utc": pd.NaT,
         "final_home_score": 999, "final_away_score": 1,
         "final_home_score_check": 111, "final_away_score_check": 105, "scores_match": False},
    ])
    stints = attach_canonical_finals(_stints_df(), canonical_df)
    stats = precompute_game_team_stats(stints)

    # Game 1: stint sum (80/76, corrupted by hypothetical garbage-time
    # zeroing) is overridden by the canonical, independently-verified 97/90.
    assert stats["0022500001"]["act_h"] == 97
    assert stats["0022500001"]["act_a"] == 90

    # Game 2 failed the two-method verification, so its unverified canonical
    # numbers must NOT override the stint sum (fail closed, not silently
    # trust an unverified value).
    assert stats["0022500002"]["act_h"] == 50
    assert stats["0022500002"]["act_a"] == 45


def test_no_label_path_resums_filtered_stint_points_when_canonical_present():
    """Structural guard: once canonical_home_pts/canonical_away_pts are
    attached, act_h/act_a must come only from those columns, not from
    re-summing home_pts/away_pts."""
    canonical_df = pd.DataFrame([
        {"GAME_ID": "0022500001", "home_team": "GSW", "away_team": "LAL",
         "season_type": "Regular", "raw_date": pd.Timestamp("2025-11-01"), "tip_utc": pd.NaT,
         "final_home_score": 0, "final_away_score": 0,
         "final_home_score_check": 0, "final_away_score_check": 0, "scores_match": True},
    ])
    stints = attach_canonical_finals(_stints_df(), canonical_df)
    stats = precompute_game_team_stats(stints)
    assert stats["0022500001"]["act_h"] == 0
    assert stats["0022500001"]["act_a"] == 0
