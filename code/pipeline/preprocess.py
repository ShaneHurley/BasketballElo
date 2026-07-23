"""PBP preprocessing and xPoints."""
import re

import numpy as np
import pandas as pd

from pipeline.shot_zones import (
    ZoneCalibration,
    assign_shot_zones_df,
    compute_zone_calibration,
    compute_zone_pps,
    legacy_zone_alias,
)

# Re-export for callers
__all__ = ["preprocess_pbp", "compute_zone_pps", "compute_zone_calibration", "ZoneCalibration"]


THREE_ZONES = frozenset({"corner3", "abovebreak3"})
RIM_ZONES = frozenset({"restricted", "paint"})


def _resolve_zone_calib(zone_pps=None, zone_calib=None) -> ZoneCalibration:
    if zone_calib is not None:
        return zone_calib
    if zone_pps is not None:
        calib = ZoneCalibration()
        for z, pps in zone_pps.items():
            zc = legacy_zone_alias(z)
            pv = 3.0 if zc in THREE_ZONES else 2.0
            fg = float(pps) / pv if pv > 0 else 0.40
            calib.fg_pct[zc] = fg
            calib.point_value[zc] = pv
        return calib
    return ZoneCalibration()


def preprocess_pbp(df, compute_xpoints=True, zone_pps=None, zone_calib=None):
    """
    Preprocess play-by-play data for possession and expected points calculations.

    zone_calib : ZoneCalibration or None
        Walk-forward calibration (preferred): xPoints = fg_pct[zone] × point_value[zone].
    zone_pps : dict or None
        Deprecated flat PPS map; converted to ZoneCalibration internally.
    """
    is_home = df["HOMEDESCRIPTION"].notna() & df["VISITORDESCRIPTION"].isna()
    is_away = df["VISITORDESCRIPTION"].notna() & df["HOMEDESCRIPTION"].isna()
    is_make = (df["EVENTMSGTYPE"] == 1)
    is_miss = (df["EVENTMSGTYPE"] == 2)
    is_tov = (df["EVENTMSGTYPE"] == 5)
    is_foul = (df["EVENTMSGTYPE"] == 6)

    df["is_fg_make"] = is_make.astype(int)
    df["is_fg_miss"] = is_miss.astype(int)
    df["is_tov"] = is_tov.astype(int)

    combo_desc = df["HOMEDESCRIPTION"].fillna("") + " " + df["VISITORDESCRIPTION"].fillna("")

    calib = _resolve_zone_calib(zone_pps=zone_pps, zone_calib=zone_calib)

    if compute_xpoints:
        if "shot_distance" not in df.columns or df["shot_distance"].isna().all():
            df["shot_distance"] = combo_desc.str.extract(r"(\d+)\s*(?:'|-foot| foot| ft)").astype(float)
            df["shot_distance"] = df["shot_distance"].fillna(-1)

        if "type" not in df.columns or df["type"].isna().all():
            df["type"] = combo_desc.str.lower()

        dist = pd.to_numeric(df.get("shot_distance", pd.Series([-1] * len(df))), errors="coerce").fillna(-1)
        combo = df.get("type", pd.Series([""] * len(df))).fillna("").str.lower()

        is_3pt_text = combo.str.contains("3pt|three", na=False)
        is_rim_text = combo.str.contains("layup|dunk|tip", na=False)
        is_floater_text = combo.str.contains("floater|hook", na=False)
        is_jumper_text = combo.str.contains("jump|fadeaway|bank", na=False)

        dist = np.where((dist == -1) & is_3pt_text, 25.0, dist)
        dist = np.where((dist == -1) & is_rim_text, 1.0, dist)
        dist = np.where((dist == -1) & is_floater_text, 7.0, dist)
        dist = np.where((dist == -1) & is_jumper_text & ~is_3pt_text, 15.0, dist)
        df["shot_distance"] = dist

        df["shot_zone"] = assign_shot_zones_df(df, pd.Series(dist, index=df.index), combo)
        is_shot = is_make | is_miss

        df["xPoints"] = df["shot_zone"].map(lambda z: calib.xpoints(z))
        df.loc[~is_shot, "xPoints"] = 0.0

        df["xefg"] = df["shot_zone"].map(lambda z: calib.fg_pct_for_zone(z))
        df.loc[~is_shot, "xefg"] = 0.0

        is_three_shot = df["shot_zone"].isin(list(THREE_ZONES)) | is_3pt_text
        is_rim_shot = df["shot_zone"].isin(list(RIM_ZONES))

        df["home_xefg_added"] = np.where(is_shot & is_home, df["xefg"], 0.0)
        df["away_xefg_added"] = np.where(is_shot & is_away, df["xefg"], 0.0)
        df["home_rim_fga"] = (is_shot & is_home & is_rim_shot).astype(int)
        df["away_rim_fga"] = (is_shot & is_away & is_rim_shot).astype(int)
        df["home_three_fga"] = (is_shot & is_home & is_three_shot).astype(int)
        df["away_three_fga"] = (is_shot & is_away & is_three_shot).astype(int)
        sd = pd.to_numeric(df["shot_distance"], errors="coerce").fillna(0)
        df["home_shot_dist_sum"] = np.where(is_shot & is_home, sd, 0.0)
        df["away_shot_dist_sum"] = np.where(is_shot & is_away, sd, 0.0)
    else:
        df["xPoints"] = 0.0
        df["xefg"] = 0.0
        for c in (
            "home_xefg_added", "away_xefg_added",
            "home_rim_fga", "away_rim_fga",
            "home_three_fga", "away_three_fga",
            "home_shot_dist_sum", "away_shot_dist_sum",
        ):
            df[c] = 0.0

    df["home_pts_added"] = 0
    df["away_pts_added"] = 0

    df.loc[is_make & is_home, "home_pts_added"] = np.where(
        combo_desc[is_make & is_home].str.contains("3PT", flags=re.IGNORECASE, na=False), 3, 2)

    df.loc[is_make & is_away, "away_pts_added"] = np.where(
        combo_desc[is_make & is_away].str.contains("3PT", flags=re.IGNORECASE, na=False), 3, 2)

    ft_made = (df["EVENTMSGTYPE"] == 3) & ~combo_desc.str.contains("MISS", flags=re.IGNORECASE, na=False)

    df.loc[ft_made & is_home, "home_pts_added"] = 1
    df.loc[ft_made & is_away, "away_pts_added"] = 1

    is_shot2 = is_make | is_miss
    df["home_xpts_added"] = 0.0
    df["away_xpts_added"] = 0.0
    df.loc[is_shot2 & is_home, "home_xpts_added"] = df.loc[is_shot2 & is_home, "xPoints"]
    df.loc[is_shot2 & is_away, "away_xpts_added"] = df.loc[is_shot2 & is_away, "xPoints"]
    ft_att = (df["EVENTMSGTYPE"] == 3)
    if "GAME_ID" in df.columns and "game_date" in df.columns:
        sort_cols = ["game_date", "GAME_ID"]
        df = df.sort_values(sort_cols).reset_index(drop=True)
        game_ft = (
            df.groupby("GAME_ID", sort=False)
            .apply(lambda g: pd.Series({
                "gm_made": int(ft_made.loc[g.index].sum()),
                "gm_att": int(ft_att.loc[g.index].sum()),
            }))
            .reset_index()
        )
        game_ft["cum_made"] = game_ft["gm_made"].cumsum().shift(1, fill_value=0)
        game_ft["cum_att"] = game_ft["gm_att"].cumsum().shift(1, fill_value=0)
        game_ft["ft_pct_prior"] = np.where(
            game_ft["cum_att"] > 0,
            game_ft["cum_made"] / game_ft["cum_att"],
            0.77,
        )
        df = df.merge(game_ft[["GAME_ID", "ft_pct_prior"]], on="GAME_ID", how="left")
        df.loc[ft_att & is_home, "home_xpts_added"] = df.loc[ft_att & is_home, "ft_pct_prior"]
        df.loc[ft_att & is_away, "away_xpts_added"] = df.loc[ft_att & is_away, "ft_pct_prior"]
        df.drop(columns=["ft_pct_prior"], inplace=True)
    else:
        ft_pct = 0.77
        df.loc[ft_att & is_home, "home_xpts_added"] = ft_pct
        df.loc[ft_att & is_away, "away_xpts_added"] = ft_pct

    is_final_ft = ft_att & combo_desc.str.contains(r"1 of 1|2 of 2|3 of 3", regex=True, na=False)
    is_tech = ft_att & (df.get("EVENTMSGACTIONTYPE", pd.Series(0, index=df.index)) == 16)
    is_and1 = ft_att & (df["EVENTMSGTYPE"].shift(1) == 1) & (df["PLAYER1_ID"] == df["PLAYER1_ID"].shift(1))
    df["is_true_ft_trip"] = (is_final_ft & ~is_and1 & ~is_tech).astype(int)

    df["shot_team_state"] = pd.Series(np.nan, index=df.index, dtype="object")
    df.loc[is_miss & is_home, "shot_team_state"] = "home"
    df.loc[is_miss & is_away, "shot_team_state"] = "away"
    reset_mask = df["EVENTMSGTYPE"].isin([1, 5, 8]) | (df["PERIOD"] != df["PERIOD"].shift(1))
    df.loc[reset_mask, "shot_team_state"] = "RESET"

    df["last_shot_team"] = df["shot_team_state"].ffill(inplace=False)
    df.loc[df["last_shot_team"] == "RESET", "last_shot_team"] = np.nan

    df["home_oreb"] = ((df["EVENTMSGTYPE"] == 4) & is_home & (df["last_shot_team"] == "home")).astype(int)
    df["away_oreb"] = ((df["EVENTMSGTYPE"] == 4) & is_away & (df["last_shot_team"] == "away")).astype(int)

    df["home_dreb"] = ((df["EVENTMSGTYPE"] == 4) & is_home & (df["last_shot_team"] == "away")).astype(int)
    df["away_dreb"] = ((df["EVENTMSGTYPE"] == 4) & is_away & (df["last_shot_team"] == "home")).astype(int)

    df["home_tovs_forced"] = (is_tov & is_away).astype(int)
    df["away_tovs_forced"] = (is_tov & is_home).astype(int)
    df["home_fouls_drawn"] = (is_foul & is_away).astype(int)
    df["away_fouls_drawn"] = (is_foul & is_home).astype(int)
    if "PLAYER3_ID" in df.columns:
        df["home_blks"] = (is_miss & is_away & df["PLAYER3_ID"].notna()).astype(int)
        df["away_blks"] = (is_miss & is_home & df["PLAYER3_ID"].notna()).astype(int)
    else:
        df["home_blks"] = df["away_blks"] = 0

    df["home_fgm"] = (is_make & is_home).astype(int)
    df["away_fgm"] = (is_make & is_away).astype(int)
    df["home_fga"] = ((is_make | is_miss) & is_home).astype(int)
    df["away_fga"] = ((is_make | is_miss) & is_away).astype(int)
    df["home_tov"] = (is_tov & is_home).astype(int)
    df["away_tov"] = (is_tov & is_away).astype(int)
    _ft_att = (df["EVENTMSGTYPE"] == 3)
    df["home_fta"] = (_ft_att & is_home).astype(int)
    df["away_fta"] = (_ft_att & is_away).astype(int)
    _is_steal = combo_desc.str.contains("STEAL", case=False, na=False)
    df["home_stl"] = (_is_steal & is_home).astype(int)
    df["away_stl"] = (_is_steal & is_away).astype(int)

    if "shot_zone" in df.columns:
        is_three_shot = df["shot_zone"].isin(list(THREE_ZONES))
    else:
        is_three_shot = combo_desc.str.contains("3PT", flags=re.IGNORECASE, na=False)
    df["home_3pm"] = (is_make & is_home & is_three_shot).astype(int)
    df["away_3pm"] = (is_make & is_away & is_three_shot).astype(int)
    df["home_3pa"] = ((is_make | is_miss) & is_home & is_three_shot).astype(int)
    df["away_3pa"] = ((is_make | is_miss) & is_away & is_three_shot).astype(int)

    df["home_poss"] = ((df["is_fg_make"] + df["is_fg_miss"] + df["is_tov"]) * is_home.astype(int)
                       - df["home_oreb"]
                       + df["is_true_ft_trip"] * is_home.astype(int))

    df["away_poss"] = ((df["is_fg_make"] + df["is_fg_miss"] + df["is_tov"]) * is_away.astype(int)
                       - df["away_oreb"]
                       + df["is_true_ft_trip"] * is_away.astype(int))

    df["total_possessions"] = (df["home_poss"] + df["away_poss"])

    def _min_rem(t):
        if pd.isna(t):
            return 12.0
        parts = str(t).split(":")
        return int(parts[0]) + int(parts[1]) / 60.0 if len(parts) == 2 else 12.0

    if "remaining_time" in df.columns:
        df["min_rem"] = df["remaining_time"].apply(_min_rem)
    else:
        df["min_rem"] = 12.0

    df["abs_margin"] = (
        pd.to_numeric(df.get("HOME_SCORE", pd.Series([0] * len(df))), errors="coerce").fillna(0)
        - pd.to_numeric(df.get("AWAY_SCORE", pd.Series([0] * len(df))), errors="coerce").fillna(0)
    ).abs()

    df["garbage"] = (df["PERIOD"] == 4) & (
        (df["abs_margin"] >= 20)
        | ((df["abs_margin"] >= 15) & (df["min_rem"] <= 3.0))
    )

    # Task 010/011: `home_pts_added`/`away_pts_added` are the immutable
    # scoring-outcome columns — every downstream label, canonical game score,
    # and wager settlement must be able to trust them regardless of garbage
    # time. They are NEVER zeroed here. Only the *rating-facing* weighted
    # possession/xPoints columns are downweighted during garbage time, since
    # those exist purely to reduce Elo/form credit for garbage-time events,
    # not to change what actually happened on the scoreboard.
    #
    # A confirmed leak fixed by this change: 1,090/1,307 2025-26 games had
    # their real final score understated by ~17.8 combined points on average
    # because this same zeroing used to also apply to home_pts_added/
    # away_pts_added, which fed directly into ACTUAL_HOME/ACTUAL_AWAY labels.
    weighted_rating_cols = ["total_possessions", "home_xpts_added", "away_xpts_added"]
    df.loc[df["garbage"], weighted_rating_cols] = 0.0

    return df
