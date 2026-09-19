"""Preflight tests for 2025–26 CLV retest blockers."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from pipeline.diagnostics import warn_pinnacle_clv_gap
from pipeline.experiment_exports import collect_artifacts
from pipeline.metrics import add_clv_columns


def test_collect_artifacts_same_path_noop(tmp_path: Path):
    """Stints already under run_dir/artifacts must not SameFileError."""
    arts = tmp_path / "artifacts"
    arts.mkdir()
    cache = arts / "stints_cache_abc.pkl"
    cache.write_bytes(b"stints")
    collect_artifacts(tmp_path, stints_cache_path=cache)
    assert cache.exists()
    assert cache.read_bytes() == b"stints"


def test_add_clv_columns_decision_ne_close_finite():
    df = pd.DataFrame({
        "DIRECTION": ["Home", "Away", "Pass"],
        "DECISION_SPREAD": [-4.5, 3.0, -2.0],
        "CLOSING_SPREAD": [-5.0, 2.5, -2.0],  # third equal → NaN CLV
        "MARKET_SPREAD": [-4.5, 3.0, -2.0],
    })
    out = add_clv_columns(df)
    assert pd.notna(out.loc[0, "CLV"])
    assert pd.notna(out.loc[1, "CLV"])
    assert pd.isna(out.loc[2, "CLV"])


def test_add_clv_columns_single_snapshot_nan():
    df = pd.DataFrame({
        "DIRECTION": ["Home", "Away"],
        "DECISION_SPREAD": [-4.5, 3.0],
        "CLOSING_SPREAD": [-4.5, 3.0],
        "MARKET_SPREAD": [-4.5, 3.0],
    })
    out = add_clv_columns(df)
    assert out["CLV"].isna().all()


def test_warn_pinnacle_clv_gap_flags_overlap(tmp_path: Path):
    pin = tmp_path / "nba_main_lines.csv"
    pin.write_text(
        "timestamp\n"
        "2025-10-21T12:00:00\n"
        "2026-05-01T12:00:00\n"
    )
    cov = pd.DataFrame([{
        "season": 2026,
        "date_min": pd.Timestamp("2025-10-21"),
        "date_max": pd.Timestamp("2026-05-18"),
        "n_distinct_decision_close": 0,
        "pct_odds": 0.99,
    }])
    flagged = warn_pinnacle_clv_gap(cov, pinnacle_path=pin)
    assert flagged == [2026]


def test_warn_pinnacle_clv_gap_skips_pre_pin_seasons(tmp_path: Path):
    pin = tmp_path / "nba_main_lines.csv"
    pin.write_text("timestamp\n2025-10-21T12:00:00\n2026-05-01T12:00:00\n")
    cov = pd.DataFrame([{
        "season": 2024,
        "date_min": pd.Timestamp("2023-10-24"),
        "date_max": pd.Timestamp("2024-06-17"),
        "n_distinct_decision_close": 0,
        "pct_odds": 0.99,
    }])
    assert warn_pinnacle_clv_gap(cov, pinnacle_path=pin) == []


def test_stage7_season_col_fallback():
    """Results frames use simulated_season_window, not season."""
    df = pd.DataFrame({
        "EDGE": [2.0, 6.0, 2.5],
        "DIRECTION": ["Pass", "Home", "Pass"],
        "simulated_season_window": ["2024-2025"] * 3,
    })
    season_col = next(
        (c for c in ("simulated_season_window", "season") if c in df.columns),
        None,
    )
    assert season_col == "simulated_season_window"
    g = df.groupby(season_col)
    assert len(list(g)) == 1
