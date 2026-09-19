"""P0: seasons with no schedule match must be dropped by default."""
from __future__ import annotations

import pandas as pd

from pipeline.dates import (
    DATE_STATUS_INTERPOLATED,
    DATE_STATUS_SCHEDULE_MATCHED,
    filter_stints_by_date_quality,
    season_date_coverage,
)


def test_high_interp_season_excluded():
    df = pd.DataFrame({
        "season": [2019] * 10 + [2022] * 10,
        "GAME_ID": [f"g{i}" for i in range(20)],
        "date_status": [DATE_STATUS_INTERPOLATED] * 10 + [DATE_STATUS_SCHEDULE_MATCHED] * 10,
        "game_date": pd.date_range("2019-10-01", periods=20, freq="D"),
    })
    cov = season_date_coverage(df)
    assert not bool(cov.loc[cov["season"] == 2019, "ok"].iloc[0])
    keep, _, dropped = filter_stints_by_date_quality(df, allow_interpolated=False)
    assert 2019 in dropped
    assert set(keep["season"].unique()) == {2022}


def test_authoritative_combined_stats_season_kept():
    """2025–26 CSV dates are authoritative without odds-schedule match."""
    from pipeline.dates import DATE_STATUS_AUTHORITATIVE

    df = pd.DataFrame({
        "season": [2026] * 8,
        "GAME_ID": [f"g{i}" for i in range(8)],
        "date_status": [DATE_STATUS_AUTHORITATIVE] * 8,
        "game_date": pd.date_range("2025-10-21", periods=8, freq="D"),
    })
    keep, cov, dropped = filter_stints_by_date_quality(df)
    assert dropped == []
    assert len(keep) == 8
    assert bool(cov["ok"].all())


def test_full_schedule_match_kept():
    df = pd.DataFrame({
        "season": [2023] * 8,
        "GAME_ID": [f"g{i}" for i in range(8)],
        "date_status": [DATE_STATUS_SCHEDULE_MATCHED] * 8,
        "game_date": pd.date_range("2023-11-01", periods=8, freq="D"),
    })
    keep, cov, dropped = filter_stints_by_date_quality(df)
    assert dropped == []
    assert len(keep) == 8
    assert bool(cov["ok"].all())


def test_allow_interpolated_keeps_broken_season():
    df = pd.DataFrame({
        "season": [2020] * 5,
        "GAME_ID": [f"g{i}" for i in range(5)],
        "date_status": [DATE_STATUS_INTERPOLATED] * 5,
        "game_date": pd.date_range("2020-10-01", periods=5, freq="D"),
    })
    keep, _, dropped = filter_stints_by_date_quality(df, allow_interpolated=True)
    assert dropped == []
    assert len(keep) == 5
