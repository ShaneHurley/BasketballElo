"""Smoke tests for V3 and 2026 PBP ingest → preprocess → stints."""
from pathlib import Path

import pandas as pd
import pytest

from pipeline.config import PBP_2026_PATH, V3_DATA_PATHS
from pipeline.ingest import convert_new_pbp, convert_v3_pbp
from pipeline.preprocess import preprocess_pbp, compute_zone_calibration
from pipeline.stints import build_stints

ROOT = Path(__file__).resolve().parents[1]
V3_SAMPLE = ROOT / "example of 22-23 same format as 23 24 25 season.csv"


def _read_sample(path, nrows=5000):
    try:
        return pd.read_csv(path, engine="pyarrow", nrows=nrows)
    except Exception:
        return pd.read_csv(path, nrows=nrows, low_memory=False)


@pytest.mark.skipif(not V3_SAMPLE.exists(), reason="V3 sample CSV missing")
def test_v3_convert_preprocess_stints():
    raw = _read_sample(V3_SAMPLE)
    df = convert_v3_pbp(raw)
    assert not df.empty, "V3 convert produced no rows"
    assert "HOME_players" in df.columns
    assert "AWAY_players" in df.columns
    assert (df["HOME_players"] != "").any()
    assert "shot_distance" in df.columns
    assert df["shot_distance"].notna().any()

    pre = preprocess_pbp(df)
    assert "shot_zone" in pre.columns
    assert pre["shot_zone"].notna().any()
    shots = pre[pre["EVENTMSGTYPE"].isin([1, 2])]
    assert not shots.empty
    assert shots["xPoints"].notna().any()

    calib = compute_zone_calibration(pre)
    assert calib.counts

    st = build_stints(pre)
    assert not st.empty
    for col in ("home_xefg_sum", "home_rim_fga", "home_three_fga"):
        assert col in st.columns


@pytest.mark.skipif(not PBP_2026_PATH.exists(), reason="2026 PBP CSV missing")
def test_2026_convert_preprocess_stints():
    raw = _read_sample(PBP_2026_PATH)
    df = convert_new_pbp(raw)
    assert not df.empty
    assert "original_x" in df.columns or "original_y" in df.columns
    assert "area" in df.columns or "area_detail" in df.columns
    assert "shot_distance" in df.columns

    pre = preprocess_pbp(df)
    shots = pre[pre["EVENTMSGTYPE"].isin([1, 2])]
    assert not shots.empty
    zones = shots["shot_zone"].dropna().unique()
    assert len(zones) >= 2

    st = build_stints(pre)
    assert not st.empty


def test_v3_synthetic_minimal():
    """Minimal V3-shaped frame without relying on local CSV."""
    raw = pd.DataFrame([{
        "gameId": "0022200001",
        "period": 1,
        "clock": "PT11M30.00S",
        "scoreHome": 2,
        "scoreAway": 0,
        "actionType": "Made Shot",
        "subType": "Jump Shot",
        "shotResult": "Made",
        "shotDistance": 26.0,
        "shotValue": 3,
        "xLegacy": -127,
        "yLegacy": 230,
        "description": "Curry 26' 3PT Jump Shot (3 PTS)",
        "location": "v",
        "personId": 201939,
        "HOME_PLAYERS_ON": "201939-203110-2544-1628369-203507",
        "AWAY_PLAYERS_ON": "1626164-1627783-203954-1628366-203083",
    }])
    df = convert_v3_pbp(raw)
    assert len(df) == 1
    pre = preprocess_pbp(df)
    assert pre.iloc[0]["shot_zone"] in ("abovebreak3", "corner3", "midrange")
    assert pre.iloc[0]["xPoints"] > 0


def test_mixed_date_formats_parsed_explicitly():
    """Task 008 regression test: ISO and M/D/YYYY dates in the same column
    must both parse to real timestamps, never NaT."""
    raw = pd.DataFrame([
        {"game_id": "0022500001", "period": 1, "remaining_time": "11:30",
         "home_score": 0, "away_score": 0, "date": "2025-10-21",
         "event_type": "shot", "result": "made", "type": "Jump Shot", "points": 2,
         "description": "Player Jump Shot", "team": "GSW", "home_team": "GSW",
         "away_team": "LAL", "player": "A", "assist": "", "block": "",
         "h1": "A", "h2": "B", "h3": "C", "h4": "D", "h5": "E",
         "a1": "F", "a2": "G", "a3": "H", "a4": "I", "a5": "J"},
        {"game_id": "0022500002", "period": 1, "remaining_time": "10:00",
         "home_score": 0, "away_score": 0, "date": "12/17/2025",
         "event_type": "shot", "result": "made", "type": "Jump Shot", "points": 2,
         "description": "Player Jump Shot", "team": "BOS", "home_team": "BOS",
         "away_team": "MIA", "player": "K", "assist": "", "block": "",
         "h1": "K", "h2": "L", "h3": "M", "h4": "N", "h5": "O",
         "a1": "P", "a2": "Q", "a3": "R", "a4": "S", "a5": "T"},
        {"game_id": "0022500003", "period": 1, "remaining_time": "9:00",
         "home_score": 0, "away_score": 0, "date": "1/22/2026",
         "event_type": "shot", "result": "made", "type": "Jump Shot", "points": 2,
         "description": "Player Jump Shot", "team": "DEN", "home_team": "DEN",
         "away_team": "UTA", "player": "U", "assist": "", "block": "",
         "h1": "U", "h2": "V", "h3": "W", "h4": "X", "h5": "Y",
         "a1": "Z", "a2": "AA", "a3": "BB", "a4": "CC", "a5": "DD"},
    ])
    df = convert_new_pbp(raw, name_to_id={})
    assert df["game_date"].isna().sum() == 0, "mixed-format dates must never become NaT"
    by_game = df.set_index("GAME_ID")["game_date"]
    assert by_game["0022500001"] == pd.Timestamp("2025-10-21")
    assert by_game["0022500002"] == pd.Timestamp("2025-12-17")
    assert by_game["0022500003"] == pd.Timestamp("2026-01-22")


KNOWN_CORRUPTED_DATE_GAME_IDS = {
    22500368: "2025-12-17",
    22500375: "2025-12-18",
    22500458: "2025-12-30",
    22500508: "2026-01-05",
    22500521: "2026-01-07",
    22500550: "2026-01-11",
    22500628: "2026-01-22",
}


@pytest.mark.skipif(not PBP_2026_PATH.exists(), reason="2025-26 combined-stats CSV missing")
def test_seven_known_corrupted_games_get_original_dates():
    """Task 008 pass condition: the 7 known M/D/YYYY games must receive their
    original December/January dates, and no row of the full source becomes NaT."""
    raw = pd.read_csv(PBP_2026_PATH, low_memory=False,
                       usecols=lambda c: c.lower() in ("game_id", "date"))
    parsed = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
    assert parsed.isna().sum() == 0, "no row should be unparseable in the full source"
    raw = raw.assign(_parsed=parsed)
    for gid, expected_date in KNOWN_CORRUPTED_DATE_GAME_IDS.items():
        game_dates = raw.loc[raw["game_id"] == gid, "_parsed"].unique()
        assert len(game_dates) == 1
        assert pd.Timestamp(game_dates[0]) == pd.Timestamp(expected_date)


def test_2026_synthetic_minimal():
    raw = pd.DataFrame([{
        "game_id": "0022500001",
        "period": 1,
        "remaining_time": "11:30",
        "home_score": 0,
        "away_score": 0,
        "date": "2025-10-21",
        "event_type": "shot",
        "result": "made",
        "type": "Jump Shot",
        "points": 3,
        "description": "Player 26' 3PT Jump Shot",
        "team": "GSW",
        "home_team": "GSW",
        "away_team": "LAL",
        "player": "Stephen Curry",
        "assist": "",
        "block": "",
        "shot_distance": 26.0,
        "original_x": 85.0,
        "original_y": 10.0,
        "area": "Above the Break 3",
        "area_detail": "24+ ft",
        "h1": "Stephen Curry", "h2": "A", "h3": "B", "h4": "C", "h5": "D",
        "a1": "E", "a2": "F", "a3": "G", "a4": "H", "a5": "I",
    }])
    df = convert_new_pbp(raw, name_to_id={})
    assert len(df) == 1
    pre = preprocess_pbp(df)
    assert pre.iloc[0]["EVENTMSGTYPE"] == 1
    assert pre.iloc[0]["shot_zone"] in ("abovebreak3", "corner3")
