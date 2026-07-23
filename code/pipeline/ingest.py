"""PBP format converters for V3 (2022-25) and 2026 combined-stats."""
import numpy as np
import pandas as pd

from pipeline.utils import _map_player_name, _parse_player_string


def _col_series(df, col_map, *names, default=np.nan):
    """Return first matching column (case-insensitive) from raw frame."""
    for name in names:
        key = name.lower()
        if key in col_map:
            return df[col_map[key]]
    if default is None:
        return None
    return pd.Series(default, index=df.index)


def _attach_shot_metadata(df, raw, col_map):
    """Normalize shot location / area columns for preprocess + shot_zones."""
    sd = _col_series(df, col_map, "shot_distance", "shotdistance")
    if sd is not None:
        df["shot_distance"] = pd.to_numeric(sd, errors="coerce")

    ox = _col_series(df, col_map, "original_x", "xlegacy", "x_legacy")
    oy = _col_series(df, col_map, "original_y", "ylegacy", "y_legacy")
    if ox is not None:
        df["original_x"] = pd.to_numeric(ox, errors="coerce")
    if oy is not None:
        df["original_y"] = pd.to_numeric(oy, errors="coerce")

    area = _col_series(df, col_map, "area")
    if area is not None:
        df["area"] = area.fillna("").astype(str)

    ad = _col_series(df, col_map, "area_detail", "areadetail", "subtype")
    if ad is not None:
        df["area_detail"] = ad.fillna("").astype(str)

    sv = _col_series(df, col_map, "shot_value", "shotvalue")
    if sv is not None:
        df["shot_value"] = pd.to_numeric(sv, errors="coerce")

    # V3 uses isFieldGoal; keep for diagnostics
    fg = _col_series(df, col_map, "isfieldgoal", "is_field_goal", default=None)
    if fg is not None:
        df["is_field_goal"] = pd.to_numeric(fg, errors="coerce").fillna(0).astype(int)

    return df


# ─────────────────────────────────────────────────────────────────────
# 2025-26 combined-stats → internal standard
# ─────────────────────────────────────────────────────────────────────
def convert_new_pbp(pbp_df, name_to_id=None):
    """Convert 2026 combined-stats CSV to internal standard columns."""
    if name_to_id is None:
        name_to_id = {}
    df = pbp_df.copy()
    col_map = {c.lower(): c for c in df.columns}

    df["GAME_ID"] = df[col_map["game_id"]]
    df["PERIOD"] = pd.to_numeric(df[col_map["period"]], errors="coerce").fillna(1).astype(int)
    df["HOME_SCORE"] = pd.to_numeric(df[col_map["home_score"]], errors="coerce").fillna(0)
    df["AWAY_SCORE"] = pd.to_numeric(df[col_map["away_score"]], errors="coerce").fillna(0)
    # Task 008: the 2025-26 combined-stats source mixes ISO ("2025-10-21") and
    # M/D/YYYY ("12/17/2025") date strings in the same column. A plain
    # `pd.to_datetime(..., errors="coerce")` call infers one format from the
    # first row and silently turns every row in the *other* format into NaT
    # (confirmed: 7 games / 4,434 rows go NaT this way). `format="mixed"`
    # parses each value independently and must never produce a NaT for a
    # value that is actually parseable.
    df["game_date"] = pd.to_datetime(df[col_map["date"]], format="mixed", errors="coerce")

    if "remaining_time" in col_map:
        df["remaining_time"] = df[col_map["remaining_time"]]

    cond = [
        (df[col_map["event_type"]] == "shot") & (df[col_map["result"]] == "made"),
        (df[col_map["event_type"]] == "shot") & (df[col_map["result"]] == "missed"),
        (df[col_map["event_type"]] == "free throw"),
        (df[col_map["event_type"]] == "rebound"),
        (df[col_map["event_type"]] == "turnover"),
        (df[col_map["event_type"]] == "foul"),
        (df[col_map["event_type"]] == "substitution"),
    ]
    df["EVENTMSGTYPE"] = np.select(cond, [1, 2, 3, 4, 5, 6, 8], default=0)
    df["EVENTMSGACTIONTYPE"] = np.where(
        (df[col_map["event_type"]] == "free throw") & (df[col_map["type"]] == "1 of 1"), 16, 0
    )

    df["points"] = pd.to_numeric(df[col_map["points"]], errors="coerce").fillna(0).astype(int)

    df["HOMEDESCRIPTION"] = np.where(
        df[col_map["team"]] == df[col_map["home_team"]], df[col_map["description"]], np.nan
    )
    df["VISITORDESCRIPTION"] = np.where(
        df[col_map["team"]] == df[col_map["away_team"]], df[col_map["description"]], np.nan
    )

    df["PLAYER1_ID"] = df[col_map["player"]].apply(lambda x: _map_player_name(x, name_to_id))
    df["PLAYER2_ID"] = df[col_map["assist"]].apply(lambda x: _map_player_name(x, name_to_id))
    df["PLAYER3_ID"] = df[col_map["block"]].apply(lambda x: _map_player_name(x, name_to_id))

    h_id_cols = [f"h{i}" for i in range(1, 6)]
    a_id_cols = [f"a{i}" for i in range(1, 6)]

    def _lineup(row, cols):
        ids = [_map_player_name(row[c], name_to_id) for c in cols if c in row.index]
        ids = sorted(int(x) for x in ids if x is not None)
        return "-".join(map(str, ids))

    df["HOME_players"] = df.apply(lambda r: _lineup(r, h_id_cols), axis=1)
    df["AWAY_players"] = df.apply(lambda r: _lineup(r, a_id_cols), axis=1)

    df = _attach_shot_metadata(df, pbp_df, col_map)

    if "official" in col_map:
        df["official"] = df[col_map["official"]]

    df = df[(df["HOME_players"] != "") & (df["AWAY_players"] != "")].reset_index(drop=True)
    return df


# ─────────────────────────────────────────────────────────────────────
# V3-format (22-23 / 23-24 / 24-25) → internal standard
# ─────────────────────────────────────────────────────────────────────
def convert_v3_pbp(pbp_df):
    """
    V3 API PBP: gameId, shotDistance, clock, xLegacy/yLegacy, actionType/subType.
    """
    df = pbp_df.copy()
    col_map = {c.lower(): c for c in df.columns}

    df["GAME_ID"] = _col_series(df, col_map, "gameid", "game_id")
    df["PERIOD"] = pd.to_numeric(_col_series(df, col_map, "period"), errors="coerce").fillna(1).astype(int)
    df["HOME_SCORE"] = pd.to_numeric(_col_series(df, col_map, "scorehome", "home_score"), errors="coerce").fillna(0)
    df["AWAY_SCORE"] = pd.to_numeric(_col_series(df, col_map, "scoreaway", "away_score"), errors="coerce").fillna(0)

    if "game_date" in col_map or "date" in col_map:
        df["game_date"] = pd.to_datetime(_col_series(df, col_map, "game_date", "date"), errors="coerce")
    else:
        df["game_date"] = df["GAME_ID"].astype(str).apply(
            lambda x: pd.to_datetime(f"20{x[1:3]}-01-01") if len(x) >= 8 else pd.NaT
        )

    if "home_team" in col_map and "away_team" in col_map:
        df["home_team"] = _col_series(df, col_map, "home_team")
        df["away_team"] = _col_series(df, col_map, "away_team")
    elif "location" in col_map and "teamtricode" in col_map:
        loc_col = col_map["location"]
        tri_col = col_map["teamtricode"]
        home_map = df[df[loc_col].astype(str).str.lower() == "h"].groupby("GAME_ID")[tri_col].first()
        away_map = df[df[loc_col].astype(str).str.lower() == "v"].groupby("GAME_ID")[tri_col].first()
        df["home_team"] = df["GAME_ID"].map(home_map)
        df["away_team"] = df["GAME_ID"].map(away_map)

    at = _col_series(df, col_map, "actiontype", "event_type").astype(str).str.lower().fillna("")
    st = _col_series(df, col_map, "subtype", "type").astype(str).str.lower().fillna("")
    sr = _col_series(df, col_map, "shotresult", "result").astype(str).str.lower().fillna("")

    is_make = at.str.contains("made shot", na=False) | ((at.str.contains("shot", na=False)) & sr.str.contains("made", na=False))
    is_miss = at.str.contains("missed shot|miss", na=False) | ((at.str.contains("shot", na=False)) & sr.str.contains("miss", na=False))

    df["EVENTMSGTYPE"] = np.select(
        [
            is_make,
            is_miss,
            at.str.contains("free throw", na=False),
            at.str.contains("rebound", na=False),
            at.str.contains("turnover", na=False),
            at.str.contains("foul", na=False),
            at.str.contains("substitution", na=False),
        ],
        [1, 2, 3, 4, 5, 6, 8],
        default=0,
    )
    df["EVENTMSGACTIONTYPE"] = 0

    desc = _col_series(df, col_map, "description").fillna("")
    loc = _col_series(df, col_map, "location").fillna("")
    df["HOMEDESCRIPTION"] = np.where(loc.astype(str).str.lower() == "h", desc, np.nan)
    df["VISITORDESCRIPTION"] = np.where(loc.astype(str).str.lower() == "v", desc, np.nan)

    df["PLAYER1_ID"] = pd.to_numeric(_col_series(df, col_map, "personid", "player1_id"), errors="coerce")
    df["PLAYER2_ID"] = pd.to_numeric(_col_series(df, col_map, "player2_id"), errors="coerce")
    df["PLAYER3_ID"] = pd.to_numeric(_col_series(df, col_map, "player3_id"), errors="coerce")

    if "home_players_on" in col_map and "away_players_on" in col_map:
        hp_col = col_map["home_players_on"]
        ap_col = col_map["away_players_on"]
        df["HOME_players"] = df[hp_col].apply(
            lambda s: "-".join(sorted(str(x) for x in _parse_player_string(s))) if isinstance(s, str) else ""
        )
        df["AWAY_players"] = df[ap_col].apply(
            lambda s: "-".join(sorted(str(x) for x in _parse_player_string(s))) if isinstance(s, str) else ""
        )
    else:
        df["HOME_players"] = ""
        df["AWAY_players"] = ""

    df["points"] = pd.to_numeric(
        _col_series(df, col_map, "pointstotal", "points"), errors="coerce"
    ).fillna(0).astype(int)

    # V3 clock → remaining_time (ISO PT format or MM:SS)
    if "remaining_time" not in df.columns:
        clock = _col_series(df, col_map, "clock", "remaining_time", default=None)
        if clock is not None and not clock.isna().all():
            def _clock_to_minutes(c):
                if pd.isna(c):
                    return 12.0
                s = str(c).strip()
                if s.startswith("PT") and "M" in s:
                    try:
                        part = s.replace("PT", "").replace("S", "")
                        mins, secs = part.split("M") if "M" in part else (part, "0")
                        return float(mins) + float(secs or 0) / 60.0
                    except (ValueError, TypeError):
                        return 12.0
                if ":" in s:
                    try:
                        mins, secs = s.split(":")
                        return float(mins) + float(secs) / 60.0
                    except (ValueError, TypeError):
                        return 12.0
                return 12.0

            df["remaining_time"] = clock.apply(_clock_to_minutes)

    df = _attach_shot_metadata(df, pbp_df, col_map)

    # subType is useful for zone hints on V3 (e.g. "3PT Jump Shot")
    if "area_detail" not in df.columns or df["area_detail"].isna().all():
        if st is not None:
            df["area_detail"] = st.fillna("").astype(str)

    df = df[(df["HOME_players"] != "") & (df["AWAY_players"] != "")].reset_index(drop=True)
    return df
