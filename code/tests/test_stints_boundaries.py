"""Task 012/013: stint boundaries preserve GAME_ID separation and never
silently drop point-bearing zero-possession stints (e.g. intermediate free
throws)."""
import pandas as pd

from pipeline.preprocess import preprocess_pbp
from pipeline.stints import build_stints


def _base_row(**overrides):
    row = {
        "GAME_ID": "0022500001",
        "PERIOD": 1,
        "remaining_time": "6:00",
        "HOME_SCORE": 0,
        "AWAY_SCORE": 0,
        "EVENTMSGTYPE": 0,
        "EVENTMSGACTIONTYPE": 0,
        "HOMEDESCRIPTION": None,
        "VISITORDESCRIPTION": None,
        "PLAYER1_ID": 1,
        "PLAYER2_ID": None,
        "PLAYER3_ID": None,
        "home_team": "GSW",
        "away_team": "LAL",
        "game_date": pd.Timestamp("2025-11-01"),
        "HOME_players": "1-2-3-4-5",
        "AWAY_players": "6-7-8-9-10",
    }
    row.update(overrides)
    return row


def test_intermediate_free_throw_stint_preserved_with_points():
    """A made free throw that is NOT the final attempt of its trip (e.g.
    "1 of 2") does not count as a completed possession, but it did score a
    real point and must not be silently dropped from the stint table."""
    rows = [
        # Intermediate FT make: "1 of 2", home team, in its own stint.
        _base_row(EVENTMSGTYPE=3, HOMEDESCRIPTION="Curry Free Throw 1 of 2 (1 PTS)",
                   HOME_SCORE=1, remaining_time="6:00"),
        # A lineup change forces a new stint boundary right after.
        _base_row(EVENTMSGTYPE=0, PERIOD=1, remaining_time="5:50",
                   HOME_players="1-2-3-4-99"),
    ]
    df = pd.DataFrame(rows)
    pre = preprocess_pbp(df, compute_xpoints=False)
    assert pre.loc[0, "is_true_ft_trip"] == 0, "intermediate FT must not count as a full possession"
    assert pre.loc[0, "home_pts_added"] == 1, "intermediate FT must still score its point"

    st = build_stints(pre)
    first_stint = st[st["HOME_players"] == "1-2-3-4-5"]
    assert not first_stint.empty, "zero-possession, point-bearing stint was dropped"
    assert first_stint.iloc[0]["home_pts"] == 1
    assert first_stint.iloc[0]["possessions"] == 0

    # Summed stint points must equal summed preprocessed immutable event points.
    assert st["home_pts"].sum() == pre["home_pts_added"].sum()
    assert st["away_pts"].sum() == pre["away_pts_added"].sum()


def test_stint_never_spans_two_games():
    """Task 013 pass condition: two adjacent games with identical lineup
    strings and PERIOD value must still create separate stints, keyed apart
    by GAME_ID."""
    rows = [
        _base_row(GAME_ID="0022500001", PERIOD=1, remaining_time="6:00",
                   EVENTMSGTYPE=1, HOMEDESCRIPTION="Made Shot", HOME_SCORE=2),
        # Second game: identical lineups and PERIOD, immediately after in the frame.
        _base_row(GAME_ID="0022500002", PERIOD=1, remaining_time="6:00",
                   EVENTMSGTYPE=1, HOMEDESCRIPTION="Made Shot", HOME_SCORE=2,
                   game_date=pd.Timestamp("2025-11-02")),
    ]
    df = pd.DataFrame(rows)
    pre = preprocess_pbp(df, compute_xpoints=False)
    st = build_stints(pre)
    assert st["GAME_ID"].nunique() == 2
    assert st["stint_id"].nunique() == 2, "identical lineup/period rows from two games must not merge into one stint"
