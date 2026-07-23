"""Lineup stint builder."""
from pipeline.config import ASSIST_SPLIT

import pandas as pd

def build_stints(pbp_df, assist_split=None):
    if assist_split is None:
        assist_split = ASSIST_SPLIT
    # 1. Initialize DF IMMEDIATELY
    df = pbp_df.copy()

    # 2. Check if the dataframe is empty or missing necessary columns
    if df.empty or "total_possessions" not in df.columns:
        print("⚠️ Warning: Dataframe empty or missing columns. Returning empty stints.")
        return pd.DataFrame()

    # 3. Possession Safety Check (Task 012: a slate with zero *possessions*
    # can still have real scoring points, e.g. an isolated intermediate free
    # throw in a tiny fixture/edge case; only bail out when there is truly
    # nothing to build — no possessions AND no points).
    has_pts = ("home_pts_added" in df.columns and df["home_pts_added"].sum() > 0) or (
        "away_pts_added" in df.columns and df["away_pts_added"].sum() > 0
    )
    if df["total_possessions"].sum() == 0 and not has_pts:
        print("⚠️ Warning: Dataframe contains 0 total possessions and 0 points. Check your Preprocessing.")
        return pd.DataFrame()

    # 4. Grouping Logic
    # Task 013: include GAME_ID in the stint-boundary condition. Without it,
    # the last stint of one game and the first stint of the next game could
    # silently merge into a single stint whenever both games happen to share
    # the same lineup strings and PERIOD value (e.g. two period-1 stints with
    # unresolved/empty lineups), corrupting per-game aggregation.
    df["stint_id"] = (
        (df["HOME_players"] != df["HOME_players"].shift()) |
        (df["AWAY_players"] != df["AWAY_players"].shift()) |
        (df["PERIOD"]       != df["PERIOD"].shift()) |
        (df["GAME_ID"]      != df["GAME_ID"].shift())
    ).cumsum()

    stints = df.groupby("stint_id").agg(
        GAME_ID            = ("GAME_ID",           "first"),
        game_date          = ("game_date",         "first"),
        home_team          = ("home_team",         "first"),
        away_team          = ("away_team",         "first"),
        PERIOD             = ("PERIOD",            "first"),
        HOME_players       = ("HOME_players",      "first"),
        AWAY_players       = ("AWAY_players",      "first"),
        HOME_SCORE_START   = ("HOME_SCORE",        "first"),
        AWAY_SCORE_START   = ("AWAY_SCORE",        "first"),
        home_pts           = ("home_pts_added",    "sum"),
        away_pts           = ("away_pts_added",    "sum"),
        home_xpts          = ("home_xpts_added",   "sum"),
        away_xpts          = ("away_xpts_added",   "sum"),
        possessions        = ("total_possessions", "sum"),
        home_oreb          = ("home_oreb",         "sum"),
        away_oreb          = ("away_oreb",         "sum"),
        home_dreb          = ("home_dreb",         "sum"),
        away_dreb          = ("away_dreb",         "sum"),
        home_tovs_forced   = ("home_tovs_forced",  "sum"),
        away_tovs_forced   = ("away_tovs_forced",  "sum"),
        home_fouls_drawn   = ("home_fouls_drawn",  "sum"),
        away_fouls_drawn   = ("away_fouls_drawn",  "sum"),
        home_fgm           = ("home_fgm",          "sum"),
        away_fgm           = ("away_fgm",          "sum"),
        home_fga           = ("home_fga",          "sum"),
        away_fga           = ("away_fga",          "sum"),
        home_tov           = ("home_tov",          "sum"),
        away_tov           = ("away_tov",          "sum"),
        home_fta           = ("home_fta",          "sum"),
        away_fta           = ("away_fta",          "sum"),
        home_stl           = ("home_stl",          "sum"),
        away_stl           = ("away_stl",          "sum"),
        home_blks          = ("home_blks",         "sum"),
        away_blks          = ("away_blks",         "sum"),
        home_3pm           = ("home_3pm",          "sum"),
        away_3pm           = ("away_3pm",          "sum"),
        home_3pa           = ("home_3pa",          "sum"),
        away_3pa           = ("away_3pa",          "sum"),
        home_poss          = ("home_poss",         "sum"),
        away_poss          = ("away_poss",         "sum"),
        home_xefg_sum      = ("home_xefg_added",   "sum"),
        away_xefg_sum      = ("away_xefg_added",   "sum"),
        home_rim_fga       = ("home_rim_fga",      "sum"),
        away_rim_fga       = ("away_rim_fga",      "sum"),
        home_three_fga     = ("home_three_fga",    "sum"),
        away_three_fga     = ("away_three_fga",    "sum"),
        home_shot_dist_sum = ("home_shot_dist_sum","sum"),
        away_shot_dist_sum = ("away_shot_dist_sum","sum"),
    ).reset_index()

    stints["HOME_SCORE_END"] = stints["HOME_SCORE_START"] + stints["home_pts"]
    stints["AWAY_SCORE_END"] = stints["AWAY_SCORE_START"] + stints["away_pts"]

    # 5. Build Usage Dicts (Raw counts for speed)
    home_u = {row["stint_id"]: {p: [0.0, 0.0, 0.0] for p in str(row["HOME_players"]).split("-") if p and p != "nan"}
              for _, row in stints.iterrows()}
    away_u = {row["stint_id"]: {p: [0.0, 0.0, 0.0] for p in str(row["AWAY_players"]).split("-") if p and p != "nan"}
              for _, row in stints.iterrows()}

    records = df.to_dict("records")
    for i, row in enumerate(records):
        sid  = row["stint_id"]
        p1   = str(int(row["PLAYER1_ID"])) if pd.notna(row.get("PLAYER1_ID")) else None
        p2   = str(int(row["PLAYER2_ID"])) if pd.notna(row.get("PLAYER2_ID")) else None
        is_mk = row.get("is_fg_make", 0) == 1
        is_ms = row.get("is_fg_miss", 0) == 1
        is_tv = row.get("is_tov",     0) == 1
        is_ft = row.get("is_true_ft_trip", 0) == 1

        nxt_oreb = False
        if is_ms and i + 1 < len(records):
            nx = records[i + 1]
            if nx.get("EVENTMSGTYPE") == 4 and nx.get("PLAYER1_ID") == row.get("PLAYER1_ID"):
                nxt_oreb = True

        for usage in (home_u.get(sid, {}), away_u.get(sid, {})):
            if is_mk:
                if p2 in usage and p1 in usage:
                    usage[p1][1] += 1.0
                    usage[p2][2] += 1.0
                elif p1 in usage:
                    usage[p1][0] += 1.0
            elif is_ms and not nxt_oreb:
                if p1 in usage: usage[p1][0] += 1.0
            elif is_tv or is_ft:
                if p1 in usage: usage[p1][0] += 1.0

    stints["home_usage"] = stints["stint_id"].map(home_u)
    stints["away_usage"] = stints["stint_id"].map(away_u)

    # Task 012: a stint with zero *possessions* is not necessarily a stint
    # with zero *points* — an intermediate made free throw (e.g. the first of
    # a "2 of 2" trip) scores a point but is not counted as a completed
    # possession by `is_true_ft_trip` until the final FT of the trip. Only
    # drop truly empty stints (no possessions AND no points); any stint that
    # scored real points must survive so those points are never silently
    # dropped from the canonical/summed final score.
    keep = (stints["possessions"] > 0) | (stints["home_pts"] > 0) | (stints["away_pts"] > 0)
    return stints[keep].reset_index(drop=True)