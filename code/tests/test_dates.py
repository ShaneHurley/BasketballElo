"""Date coercion and NaT-safe Elo decay."""
import datetime

import pandas as pd
import pytest

from pipeline.dates import (
    DATE_STATUS_AUTHORITATIVE,
    DATE_STATUS_SCHEDULE_MATCHED,
    DATE_STATUS_UNRESOLVED,
    coerce_game_date,
    date_from_game_id,
    is_valid_timestamp,
    prepare_chronological_stints,
    resolve_game_date,
    to_py_date,
)
from pipeline.ratings import PlayerRatingTracker


def test_is_valid_timestamp_nat():
    assert not is_valid_timestamp(pd.NaT)
    assert not is_valid_timestamp(None)
    assert is_valid_timestamp("2025-10-21")
    assert is_valid_timestamp(pd.Timestamp("2025-10-21"))


def test_date_from_odds_game_id():
    ts = date_from_game_id("nba.g.2025102101")
    assert ts.date() == datetime.date(2025, 10, 21)


def test_resolve_game_date_authoritative():
    res = resolve_game_date("2025-10-21")
    assert res.status == DATE_STATUS_AUTHORITATIVE
    assert res.date.date() == datetime.date(2025, 10, 21)


def test_resolve_game_date_schedule_matched_from_game_id():
    """A NaT/missing raw date recovered from the GAME_ID's own embedded
    YYYYMMDD digits is schedule_matched — it is derived from the game's
    authoritative identity, not invented."""
    res = resolve_game_date(pd.NaT, game_id="nba.g.2025102101")
    assert res.status == DATE_STATUS_SCHEDULE_MATCHED
    assert res.date.date() == datetime.date(2025, 10, 21)


def test_resolve_game_date_unresolved_does_not_fabricate():
    """Task 009 regression test: a date with no valid raw value and a
    GAME_ID that does not encode a plausible date must be `unresolved`,
    never a fabricated `prev_date + 1` or season-start placeholder."""
    res = resolve_game_date(pd.NaT, game_id="not-a-date-id")
    assert res.status == DATE_STATUS_UNRESOLVED
    assert res.date is None


def test_coerce_game_date_no_longer_fabricates_from_prev_date():
    """Task 009 regression test: coerce_game_date must return None (fail
    closed) instead of silently chaining prev_date + 1 day when the game's
    own date/id give no information."""
    prev = pd.Timestamp("2025-05-01")
    ts = coerce_game_date(pd.NaT, game_id="0022500999", prev_date=prev)
    assert ts is None, "must not fabricate prev_date + 1 day when GAME_ID has no plausible embedded date"


def test_coerce_game_date_ignores_season_start_fallback():
    """No more silent season-start-date fabrication when unresolved."""
    ts = coerce_game_date(pd.NaT, game_id="xx", season_start_year=2025)
    assert ts is None


def test_apply_inactivity_decay_rejects_nat():
    tracker = PlayerRatingTracker()
    tracker.apply_inactivity_decay([101], pd.NaT)
    assert tracker._get(101)["last_date"] is None

    tracker.apply_inactivity_decay([101], datetime.date(2025, 10, 21))
    assert tracker._get(101)["last_date"] == datetime.date(2025, 10, 21)

    # Must not raise when last_date exists and incoming date is NaT
    tracker.apply_inactivity_decay([101], pd.NaT)
    assert tracker._get(101)["last_date"] == datetime.date(2025, 10, 21)


def test_to_py_date_never_nat():
    """Task 009: to_py_date must return a real date/date or None — never a
    NaTType, and never a fabricated prev_date + 1 day."""
    result = to_py_date(pd.NaT, prev_date=pd.Timestamp("2025-01-01"))
    assert result is None
    assert not (isinstance(result, float))  # never NaT-as-float either

    resolved = to_py_date("2025-10-21")
    assert resolved == datetime.date(2025, 10, 21)


def test_prepare_chronological_stints_mixed_iso_us_strings():
    """2025-26 combined-stats: ISO + M/D/YYYY in the same object column must
    sort into real chronological order (not lexicographic string order)."""
    df = pd.DataFrame({
        "GAME_ID": ["22500508", "22500001", "22500368", "22500002"],
        "game_date": ["1/5/2026", "2025-10-21", "12/17/2025", "2025-10-22"],
        "stint_id": [0, 0, 0, 0],
    })
    out = prepare_chronological_stints(df, context="test")
    dates = out.drop_duplicates("GAME_ID")["game_date"].tolist()
    assert dates == sorted(dates)
    assert dates[0] == pd.Timestamp("2025-10-21")
    assert dates[-1] == pd.Timestamp("2026-01-05")


def test_prepare_chronological_stints_mixed_timestamp_and_str():
    """Object column mixing Timestamp + str used to sort Timestamps first
    then strings — breaking is_monotonic_increasing after drop_duplicates."""
    df = pd.DataFrame({
        "GAME_ID": ["a", "b", "c"],
        "game_date": [pd.Timestamp("2025-11-02"), "2025-11-01", pd.Timestamp("2025-11-03")],
        "stint_id": [0, 0, 0],
    })
    # Precondition: naive sort+assert would fail (the Colab 2026 crash mode).
    naive = df.sort_values(["game_date", "GAME_ID", "stint_id"])
    assert not naive.drop_duplicates("GAME_ID")["game_date"].is_monotonic_increasing

    out = prepare_chronological_stints(df, context="test")
    assert out.drop_duplicates("GAME_ID")["game_date"].is_monotonic_increasing
    assert out.iloc[0]["GAME_ID"] == "b"


def test_prepare_chronological_stints_plain_to_datetime_nat_recovered_or_dropped():
    """Without format='mixed', US dates become NaT; helper must not leave
    them in a non-monotonic per-game series."""
    raw = pd.Series(["2025-10-21", "12/17/2025", "2025-10-22"])
    broken = pd.to_datetime(raw, errors="coerce")  # may NaT the US row
    df = pd.DataFrame({
        "GAME_ID": ["g0", "g1", "g2"],
        "game_date": broken,
        "stint_id": [0, 0, 0],
    })
    # Re-attach original strings so mixed parse can recover.
    df["game_date"] = raw
    out = prepare_chronological_stints(df, context="test")
    assert out["game_date"].notna().all()
    assert out.drop_duplicates("GAME_ID")["game_date"].is_monotonic_increasing
