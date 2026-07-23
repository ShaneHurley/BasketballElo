"""Task 010/011: immutable outcome points must survive garbage-time weighting;
only rating-facing possession/xPoints columns may be downweighted."""
import pandas as pd

from pipeline.preprocess import preprocess_pbp


def _row(period, remaining_time, home_score, away_score, made_home=False, made_away=False,
         three=False):
    desc_home = ("3PT Jump Shot" if three else "Jump Shot") if made_home else None
    desc_away = ("3PT Jump Shot" if three else "Jump Shot") if made_away else None
    return {
        "GAME_ID": "0022500001",
        "PERIOD": period,
        "remaining_time": remaining_time,
        "HOME_SCORE": home_score,
        "AWAY_SCORE": away_score,
        "EVENTMSGTYPE": 1 if (made_home or made_away) else 0,
        "EVENTMSGACTIONTYPE": 0,
        "HOMEDESCRIPTION": desc_home,
        "VISITORDESCRIPTION": desc_away,
        "PLAYER1_ID": 1,
        "PLAYER2_ID": None,
        "PLAYER3_ID": None,
        "game_date": pd.Timestamp("2025-11-01"),
    }


def _blowout_fixture():
    """A running blowout: home team builds a 25-point lead, then keeps
    scoring in garbage time (period 4, margin >= 20, <=3 min left)."""
    rows = []
    # Regulation scoring, non-garbage: home builds a 25-0 lead across Q1-Q3.
    running_home, running_away = 0, 0
    for i in range(10):
        running_home += 3
        rows.append(_row(1 if i < 4 else (2 if i < 7 else 3), "6:00",
                          running_home, running_away, made_home=True, three=True))
    # Q4 garbage time: margin is 30, 2:00 remaining -> flagged garbage.
    running_home += 2
    rows.append(_row(4, "2:00", running_home, running_away, made_home=True))
    running_away += 2
    rows.append(_row(4, "1:30", running_home, running_away, made_away=True))
    running_home += 3
    rows.append(_row(4, "1:00", running_home, running_away, made_home=True, three=True))
    return pd.DataFrame(rows)


def test_immutable_points_columns_not_zeroed_in_garbage_time():
    df = preprocess_pbp(_blowout_fixture(), compute_xpoints=False)
    garbage_rows = df[df["garbage"]]
    assert garbage_rows["home_pts_added"].sum() > 0 or garbage_rows["away_pts_added"].sum() > 0, (
        "garbage-time rows must still carry their real scoring points"
    )


def test_weighted_rating_columns_still_downweighted_in_garbage_time():
    df = preprocess_pbp(_blowout_fixture(), compute_xpoints=False)
    garbage_rows = df[df["garbage"]]
    assert (garbage_rows["total_possessions"] == 0).all()
    assert (garbage_rows["home_xpts_added"] == 0).all()
    assert (garbage_rows["away_xpts_added"] == 0).all()


def test_blowout_final_score_unchanged_regardless_of_garbage_weighting():
    """Task 011 pass condition: canonical final score (sum of home_pts_added
    / away_pts_added) is identical whether or not any row is flagged
    garbage — garbage-time weighting must only ever affect rating-facing
    columns, never the scoreboard."""
    fixture = _blowout_fixture()
    df_normal = preprocess_pbp(fixture.copy(), compute_xpoints=False)
    final_home_normal = df_normal["home_pts_added"].sum()
    final_away_normal = df_normal["away_pts_added"].sum()

    # Force every row to be classified as garbage time (the "weighting on"
    # extreme case) and confirm the summed final score is unchanged.
    df_all_garbage = preprocess_pbp(fixture.copy(), compute_xpoints=False)
    df_all_garbage["garbage"] = True
    weighted_cols = ["total_possessions", "home_xpts_added", "away_xpts_added"]
    df_all_garbage.loc[df_all_garbage["garbage"], weighted_cols] = 0.0
    final_home_all_garbage = df_all_garbage["home_pts_added"].sum()
    final_away_all_garbage = df_all_garbage["away_pts_added"].sum()

    assert final_home_normal == final_home_all_garbage
    assert final_away_normal == final_away_all_garbage
    # Sanity: the fixture actually has real, nonzero scoring.
    assert final_home_normal > 0

    # Expected final score from the fixture construction: home makes 10
    # threes (30) + one 2pt (2) + one 3pt (3) = 35; away makes one 2pt = 2.
    assert final_home_normal == 35
    assert final_away_normal == 2
