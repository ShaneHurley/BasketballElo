"""Documented open gaps as strict expected-fails.

Each test asserts the *desired* future contract. Today the assertion is red,
so ``xfail(strict=True)`` records it as XFAIL (suite still exits 0). If the
gap is closed, the test XPASSes and the suite fails until the marker is
removed — same pattern as P0.5 RD.

These are not production fixes. Do not tune ``config.py`` from them.
"""
from __future__ import annotations

import inspect

import numpy as np
import pandas as pd
import pytest

from pipeline.ratings import PlayerRatingTracker
from tests.synth.factories import make_stint


def _feed(tracker: PlayerRatingTracker, stint: dict) -> None:
    ids_a = [p for p in str(stint["HOME_players"]).split("-") if p]
    ids_b = [p for p in str(stint["AWAY_players"]).split("-") if p]
    tracker.process_stint(
        ids_a,
        ids_b,
        float(stint["possessions"]),
        float(stint["home_xpts"]),
        float(stint["away_xpts"]),
        period=int(stint.get("period", 1)),
        season_progress=1.0,
        stint_ctx={"garbage": False, "clutch": False},
    )


@pytest.mark.bench
def test_pinnacle_without_authoritative_tip_is_refused(monkeypatch, tmp_path):
    """Desired: no quote-level tip UTC ⇒ raise, do not emit tip_proxy odds."""
    pin = tmp_path / "pinnacle.csv"
    pin.write_text("dummy\n")
    monkeypatch.setattr(
        "pipeline.odds_loader.load_pinnacle_lines",
        lambda *a, **k: {"dummy": {"spread": -3.5}},
    )
    monkeypatch.setattr("pipeline.odds_loader.canonicalize_odds_dict", lambda d: d)
    from pipeline.odds_loader import load_odds_dict

    with pytest.raises((ValueError, RuntimeError)):
        load_odds_dict(pinnacle_path=pin)


@pytest.mark.bench
def test_live_market_uses_fold_selected_shin_or_power():
    """Desired: Shin/Power selection is wired into the live ML de-vig path."""
    from pipeline import market

    src = inspect.getsource(market)
    assert "select_devig_method_from_folds" in src


@pytest.mark.bench
@pytest.mark.xfail(
    strict=True,
    reason="Epic 7.2 — compute_stake still uses ad-hoc Kelly, not robust_fractional_kelly",
)
def test_compute_stake_uses_robust_fractional_kelly():
    """Desired: live stake sizing calls the slate/robust Kelly helpers that already exist."""
    from pipeline.stake_profiles import compute_stake

    src = inspect.getsource(compute_stake)
    assert "robust_fractional_kelly" in src or "optimize_slate_stakes" in src


@pytest.mark.bench
def test_venn_abers_interval_gates_market_implied_ml():
    """Desired: market implied p outside [p0, p1] is rejected on the ML head."""
    from pipeline.venn_abers import market_implied_inside_interval

    assert market_implied_inside_interval(0.40, 0.60, 0.52) is True
    assert market_implied_inside_interval(0.40, 0.60, 0.90) is False


@pytest.mark.bench
def test_epm_blend_weight_tracks_minutes_forecast():
    """Desired: a 36-minute star blends more external EPM than an 8-minute bench player."""
    from pipeline.epm_priors import EpmPriorTracker

    tracker = EpmPriorTracker(csv_path="__no_such_epm_file__.csv")
    tracker.snapshots = {"star": [(pd.Timestamp("2024-01-01"), 8.0)]}
    low = tracker.blend_lineup_off(
        ["star"], 1500.0, as_of="2025-01-01", minutes={"star": 8.0},
    )
    high = tracker.blend_lineup_off(
        ["star"], 1500.0, as_of="2025-01-01", minutes={"star": 36.0},
    )
    assert high != low


@pytest.mark.bench
@pytest.mark.xfail(
    strict=True,
    reason="Epic 8 — chemistry, HAPM, and lineup-Elo synergy features still all enter the stack",
)
def test_stacker_has_one_synergy_family_not_three():
    """Desired: one composite synergy source, not chem + HAPM + 5-man lineup in parallel."""
    from pipeline.model import SAFE_FEATURE_COLS

    families = []
    if any("chem" in c for c in SAFE_FEATURE_COLS):
        families.append("chem")
    if any("hapm" in c for c in SAFE_FEATURE_COLS):
        families.append("hapm")
    if any("lineup5" in c for c in SAFE_FEATURE_COLS):
        families.append("lineup5")
    assert len(families) <= 1, families


@pytest.mark.bench
def test_rd_shrinks_more_after_high_possession_observation():
    """Kalman: 200-poss shock is more informative than a 1-poss shock."""
    t_lo = PlayerRatingTracker(config={"HOME_PPP_BOOST": 0.0})
    t_hi = PlayerRatingTracker(config={"HOME_PPP_BOOST": 0.0})
    shock_lo = make_stint(possessions=1.0, home_xpts=3.0, away_xpts=0.2, period=1)
    shock_hi = make_stint(possessions=200.0, home_xpts=600.0, away_xpts=40.0, period=1)
    _feed(t_lo, shock_lo)
    _feed(t_hi, shock_hi)
    pid = str(shock_lo["HOME_players"]).split("-")[0]
    assert t_hi.players[pid]["O_rd"] < t_lo.players[pid]["O_rd"]
