"""P0: odds join uses canonical (date, tricode) keys."""
from __future__ import annotations

import datetime as dt

from pipeline.diagnostics import assert_odds_coverage
from pipeline.market import canonicalize_odds_dict, get_game_odds, team_to_abbr
import pandas as pd
import pytest


def test_team_to_abbr_short_and_full():
    assert team_to_abbr("Boston") == "BOS"
    assert team_to_abbr("LA Lakers") == "LAL"
    assert team_to_abbr("BOS") == "BOS"


def test_canonicalize_and_get_game_odds_match():
    d = dt.date(2023, 11, 15)
    raw = {
        (d, "Boston"): {
            "spread": -4.5, "ml": -180,
            "decision_spread": -4.5, "closing_spread": -4.5,
        },
        (d, "BOS"): {
            "spread": -4.0, "ml": -170,
            "decision_spread": -5.0, "closing_spread": -4.0,
        },
    }
    odds = canonicalize_odds_dict(raw)
    assert (d, "BOS") in odds
    go = get_game_odds(d, "BOS", odds)
    assert go["spread"] == -5.0  # richer decision≠close preferred
    assert go["decision_spread"] == -5.0
    assert go["closing_spread"] == -4.0
    # Stints may store short name or tricode — both must hit.
    go2 = get_game_odds(d, "Boston", odds)
    assert go2["decision_spread"] == -5.0


def test_assert_odds_coverage_fail_closed():
    cov = pd.DataFrame([
        {"season": 2022, "pct_odds": 0.1, "n_games": 100, "n_odds_matched": 10},
    ])
    with pytest.raises(SystemExit):
        assert_odds_coverage(cov, min_rate=0.5)
