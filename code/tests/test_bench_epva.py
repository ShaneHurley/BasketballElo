"""Bench: Epic 12.1 zone EPVA (decision vs execution; no tracking MDP)."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.epva import (
    EpvaTracker,
    decision_epva,
    epva_from_gs_side,
    epva_from_shot_rows,
    execution_epva,
)
from pipeline.shot_zones import CANONICAL_ZONES

pytestmark = pytest.mark.bench


def test_decision_epva_rewards_restricted_mix_over_midrange():
    """Shooter taking only restricted attempts has higher decision EPVA than midrange-only."""
    rim = {z: 0 for z in CANONICAL_ZONES}
    rim["restricted"] = 20
    mid = {z: 0 for z in CANONICAL_ZONES}
    mid["midrange"] = 20
    assert decision_epva(rim) > decision_epva(mid)


def test_execution_epva_negative_on_misses_of_high_xpts():
    # 10 rim attempts expected ~6.4 pts; all miss → strongly negative execution
    exe = execution_epva(makes_points=0.0, attempts=10, zone_xpts_sum=6.4)
    assert exe < -5.0


def test_epva_from_shot_rows_and_tracker_features():
    shots = pd.DataFrame({
        "shot_zone": ["restricted"] * 8 + ["midrange"] * 8,
        "shot_made": [1, 1, 1, 0, 0, 0, 0, 0] + [0] * 8,
        "shot_points": [2, 2, 2, 0, 0, 0, 0, 0] + [0] * 8,
    })
    agg = epva_from_shot_rows(shots)
    assert agg["n_attempts"] == 16
    assert "decision" in agg and "execution" in agg

    tr = EpvaTracker(window_games=5, days_since_prior=0.0)
    tr.update_game("GSW", decision=0.05, execution=-0.02, n_attempts=16)
    tr.update_game("LAL", decision=-0.01, execution=0.01, n_attempts=12)
    feats = tr.feature_dict("GSW", "LAL")
    assert feats["epva_diff"] != 0.0
    assert feats["epva_sample_min"] >= 12


def test_epva_prior_shrinks_game1_shock():
    """Large Game-1 residual must not dominate Game-2 when k is large (stale prior)."""
    tr = EpvaTracker(
        window_games=5,
        shrink_k=50.0,
        prior_season_epva={"BOS": {"decision": 0.0, "execution": 0.0}},
        days_since_prior=180.0,  # stale → large effective k
    )
    tr.update_game("BOS", decision=5.0, execution=5.0, n_attempts=3)
    dec, exe, n = tr._agg("BOS")
    assert n == 3
    # Heavily shrunk toward prior 0
    assert abs(dec) < 1.0
    assert abs(exe) < 1.0


def test_epva_from_gs_side_counts_attempts():
    gs = {
        "home_rim_fga": 10,
        "home_three_fga": 15,
        "home_fga": 40,
        "home_fgm": 18,
        "home_3pm": 5,
    }
    agg = epva_from_gs_side(gs, "home")
    assert agg["n_attempts"] == 40
