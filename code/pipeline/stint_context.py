"""Build stint context for context-aware Elo rating updates."""
from __future__ import annotations

import pandas as pd


def _get(row, key, default=0.0):
    if isinstance(row, dict):
        return row.get(key, default)
    return getattr(row, key, default) if hasattr(row, key) else default


def build_stint_context(row) -> dict:
    """Aggregate stint-level box stats into a context dict for process_stint."""
    poss = float(_get(row, "possessions", 1) or 1)
    if poss < 1:
        poss = 1.0

    period = int(_get(row, "PERIOD", 1) or 1)
    start_h = float(_get(row, "HOME_SCORE_START", 0) or 0)
    start_a = float(_get(row, "AWAY_SCORE_START", 0) or 0)
    margin_start = start_h - start_a

    home_pts = float(_get(row, "home_pts", 0) or 0)
    away_pts = float(_get(row, "away_pts", 0) or 0)
    home_xpts = float(_get(row, "home_xpts", 0) or 0)
    away_xpts = float(_get(row, "away_xpts", 0) or 0)

    home_tov = float(_get(row, "home_tov", 0) or 0)
    away_tov = float(_get(row, "away_tov", 0) or 0)
    home_stl = float(_get(row, "home_stl", 0) or 0)
    away_stl = float(_get(row, "away_stl", 0) or 0)
    home_blks = float(_get(row, "home_blks", 0) or 0)
    away_blks = float(_get(row, "away_blks", 0) or 0)
    home_tovs_forced = float(_get(row, "home_tovs_forced", 0) or 0)
    away_tovs_forced = float(_get(row, "away_tovs_forced", 0) or 0)
    home_fouls_drawn = float(_get(row, "home_fouls_drawn", 0) or 0)
    away_fouls_drawn = float(_get(row, "away_fouls_drawn", 0) or 0)
    home_fta = float(_get(row, "home_fta", 0) or 0)
    away_fta = float(_get(row, "away_fta", 0) or 0)
    home_fga = float(_get(row, "home_fga", 0) or 0)
    away_fga = float(_get(row, "away_fga", 0) or 0)
    home_3pa = float(_get(row, "home_3pa", 0) or 0)
    away_3pa = float(_get(row, "away_3pa", 0) or 0)
    home_rim_fga = float(_get(row, "home_rim_fga", 0) or 0)
    away_rim_fga = float(_get(row, "away_rim_fga", 0) or 0)

    clutch = period >= 4 and abs(margin_start) < 10
    garbage_flag = bool(_get(row, "garbage", False))

    return {
        "poss": poss,
        "clutch": clutch,
        "garbage": garbage_flag,
        "margin_start": margin_start,
        "home_tov_rate": home_tov / poss,
        "away_tov_rate": away_tov / poss,
        "home_def_events": (home_stl + home_blks + home_tovs_forced) / poss,
        "away_def_events": (away_stl + away_blks + away_tovs_forced) / poss,
        "home_foul_draw_rate": (home_fouls_drawn + home_fta * 0.5) / poss,
        "away_foul_draw_rate": (away_fouls_drawn + away_fta * 0.5) / poss,
        "home_3pa_rate": home_3pa / max(home_fga, 1.0),
        "away_3pa_rate": away_3pa / max(away_fga, 1.0),
        "home_rim_rate": home_rim_fga / max(home_fga, 1.0),
        "away_rim_rate": away_rim_fga / max(away_fga, 1.0),
        "home_luck_ppp": (home_pts - home_xpts) / poss,
        "away_luck_ppp": (away_pts - away_xpts) / poss,
        "home_pts": home_pts,
        "away_pts": away_pts,
        "home_xpts": home_xpts,
        "away_xpts": away_xpts,
    }


def extract_crew_id(group) -> str | None:
    """First non-null official/crew id from a game stint group."""
    for col in ("official", "crew_id", "CREW_ID"):
        if col in group.columns:
            vals = group[col].dropna()
            if len(vals):
                return str(vals.iloc[0])
    return None
