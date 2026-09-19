"""Epic 11.1 / P0.7 — Remove stale HOME_PPP_BOOST = 0.024 landmine defaults.

GitHub: P0.7 · Registry ID: ``bench_p0_home_boost``

``LineupEloTracker``'s constructor default and ``engine_implied_margins``'s
config lookup must not silently fall back to the old, stale 0.024 value.
The single source of truth is ``pipeline/config.py::HOME_PPP_BOOST`` (0.002).
A missing key must fail loud (KeyError), not silently default.
"""
from __future__ import annotations

import pytest

from pipeline.feature_utils import engine_implied_margins
from pipeline.lineup_elo import LineupEloTracker


@pytest.mark.bench
def test_p0_7_lineup_elo_tracker_default_home_boost_matches_config():
    """Acceptance: default home_boost == 0.002 (config.py HOME_PPP_BOOST), not 0.024."""
    tracker = LineupEloTracker()
    assert tracker.home_boost == 0.002
    assert tracker.home_boost != 0.024


@pytest.mark.bench
def test_p0_7_engine_implied_margins_fails_loud_on_missing_key():
    """Acceptance: missing HOME_PPP_BOOST in tracker cfg must raise KeyError, not
    silently substitute the stale 0.024 landmine default."""

    class _StubTracker:
        cfg = {"ELO_SCALING_FACTOR": 1000}  # deliberately missing HOME_PPP_BOOST
        league_xppp = 1.10

    with pytest.raises(KeyError):
        engine_implied_margins(
            _StubTracker(),
            hier_engine=None,
            ho_off=1500.0,
            ho_def=1500.0,
            ao_off=1500.0,
            ao_def=1500.0,
            home_starters=[],
            away_starters=[],
            exp_poss=100.0,
        )


@pytest.mark.bench
def test_p0_7_engine_implied_margins_works_when_key_present():
    """Witness: with the key present, no exception and margin is finite."""

    class _StubTracker:
        cfg = {"ELO_SCALING_FACTOR": 1000, "HOME_PPP_BOOST": 0.002}
        league_xppp = 1.10

    elo_margin, hier_margin = engine_implied_margins(
        _StubTracker(),
        hier_engine=None,
        ho_off=1500.0,
        ho_def=1500.0,
        ao_off=1500.0,
        ao_def=1500.0,
        home_starters=[],
        away_starters=[],
        exp_poss=100.0,
    )
    assert elo_margin == elo_margin  # not NaN
    assert hier_margin == 0.0  # hier_engine=None ⇒ caught exception path
