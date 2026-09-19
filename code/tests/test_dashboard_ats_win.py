"""Tests for ATS_WIN derivation and pipeline python resolution."""
from __future__ import annotations

import pandas as pd

from dashboard.jobs.util import ensure_ats_win


def test_ensure_ats_win_derives_from_decision_spread():
    df = pd.DataFrame({
        "ACTUAL_MARGIN": [5.0, -3.0, 0.0, 2.0],
        "DECISION_SPREAD": [-4.0, 2.0, -1.0, -5.0],
        "DIRECTION": ["Home", "Away", "Home", "Away"],
    })
    # Home +5 vs -4 → cover (+1); Away -3 vs +2 → away cover margin+spread=-1 → away covers → win
    # Home 0 vs -1 → cover +(-1)=-1 → away covers → home loss; Away +2 vs -5 → 2-5=-3 → away covers → win
    out = ensure_ats_win(df)
    assert "ATS_WIN" in out.columns
    assert list(out["ATS_WIN"]) == [1.0, 1.0, 0.0, 1.0]


def test_ensure_ats_win_noop_when_present():
    df = pd.DataFrame({"ATS_WIN": [1, 0], "EDGE": [1.0, -1.0]})
    out = ensure_ats_win(df)
    assert list(out["ATS_WIN"]) == [1, 0]
