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
