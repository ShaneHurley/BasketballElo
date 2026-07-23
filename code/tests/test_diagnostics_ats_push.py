"""Regression: empty actionable ATS frames must still expose ats_push."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.diagnostics import _ats_frame, _confidence_tier_table, _non_push_ats, run_backtest_diagnostics


def _season_with_zero_actionable_bets(n: int = 20) -> pd.DataFrame:
    """Lined games with DIRECTION=Pass / ACTIONABLE=0 (confidence-only zero-bet season)."""
    rng = np.random.default_rng(0)
    return pd.DataFrame({
        "simulated_season_window": ["2022-2023"] * n,
        "GAME_ID": np.arange(n),
        "MARKET_SPREAD": rng.normal(-3, 4, n),
        "ACTUAL_MARGIN": rng.normal(0, 12, n),
        "PRED_SPREAD": rng.normal(0, 8, n),
        "RAW_PRED_SPREAD": rng.normal(0, 8, n),
        "PRED_TOTAL": rng.normal(220, 10, n),
        "ACTUAL_HOME": 110,
        "ACTUAL_AWAY": 108,
        "EDGE": rng.normal(0, 5, n),
        "DIRECTION": ["Pass"] * n,
        "ACTIONABLE": [0] * n,
        "CONFIDENCE_TIER": [2] * n,
        "CONFIDENCE": [55.0] * n,
        "WIN_PROB": [0.55] * n,
    })


def test_empty_ats_frame_still_has_ats_push_column():
    df = _season_with_zero_actionable_bets()
    bets = _ats_frame(df)
    assert bets.empty
    assert "ats_push" in bets.columns
    assert "ats_win" in bets.columns
    assert _non_push_ats(bets).empty


def test_confidence_tier_table_with_zero_bets_does_not_keyerror():
    df = _season_with_zero_actionable_bets()
    out = _confidence_tier_table(df)
    assert out.empty


def test_run_backtest_diagnostics_zero_bet_season_completes():
    df = _season_with_zero_actionable_bets(40)
    summary, edges = run_backtest_diagnostics(df, show_graphs=False)
    assert not summary.empty
    assert int(summary.iloc[0]["n_bets"]) == 0
    assert "2022-2023" in edges
