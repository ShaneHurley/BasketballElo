"""Negative controls for roadmap upgrades (leak / overfit guards)."""
from __future__ import annotations

import numpy as np

from pipeline.promotion_gates import (
    beats_market_and_baseline,
    clv_promotion_allowed,
    multi_season_gate,
)
from pipeline.shot_hierarchy import HierarchicalZoneRates
from pipeline.target_calibration import TeamPooledCalibrator


def test_shot_hierarchy_no_future_team_bleed():
    """Updating team B must not mutate team A's stored make rates."""
    rates = HierarchicalZoneRates()
    for _ in range(50):
        rates.update_shot(zone="restricted", made=True, team="BOS")
    bos_fg_before = rates.team_fg["BOS"]["restricted"]
    bos_n_before = rates.team_n["BOS"]["restricted"]
    for _ in range(200):
        rates.update_shot(zone="restricted", made=False, team="LAL")
    assert rates.team_fg["BOS"]["restricted"] == bos_fg_before
    assert rates.team_n["BOS"]["restricted"] == bos_n_before
    # League prior may move; team-specific store must stay isolated.
    assert "restricted" in rates.team_fg["LAL"]


def test_target_calibrator_no_pooling_overfit_single_team():
    """A single-team spike should be heavily shrunk."""
    cal = TeamPooledCalibrator(window=300, min_samples=40, team_shrink_k=30.0)
    rng = np.random.default_rng(1)
    # Global well-calibrated stream
    for _ in range(100):
        p = 0.55
        y = int(rng.random() < 0.55)
        cal.update(p, y, team="LEA")
    # Tiny team with perfect outcomes at high raw prob
    for _ in range(3):
        cal.update(0.9, 1, team="TINY")
    global_p = cal.predict(0.9, team=None)
    tiny_p = cal.predict(0.9, team="TINY")
    # Partial pooling: tiny team adjustment must be small vs raw 1.0 identity
    assert abs(tiny_p - global_p) < 0.15


def test_closing_line_not_in_structured_features():
    from pipeline.structured_score import STRUCTURED_SCORE_COLS
    assert "closing_spread" not in STRUCTURED_SCORE_COLS
    assert "close_residual" not in STRUCTURED_SCORE_COLS


def test_clv_blocked_without_promotion_eligible():
    assert clv_promotion_allowed({"promotion_eligible": False}, 500) is False
    assert clv_promotion_allowed({"promotion_eligible": True}, 50) is False
    assert clv_promotion_allowed({"promotion_eligible": True}, 250) is True


def test_multi_season_gate_requires_three_wins():
    base = [{"spread_mae": 11.0} for _ in range(4)]
    good = [{"spread_mae": 10.5}, {"spread_mae": 10.6}, {"spread_mae": 10.4}, {"spread_mae": 11.2}]
    info = multi_season_gate(good, base)
    assert info["wins"] == 3
    assert info["passed"] is True
    collapse = [{"spread_mae": 10.5}, {"spread_mae": 10.5}, {"spread_mae": 10.5}, {"spread_mae": 12.0}]
    bad = multi_season_gate(collapse, base)
    assert bad["passed"] is False


def test_beats_market_and_baseline():
    baseline = {"spread_mae": 11.2, "market_spread_mae": 10.8, "brier": 0.22}
    cand = {"spread_mae": 10.7, "market_spread_mae": 10.8, "brier": 0.21}
    assert beats_market_and_baseline(cand, baseline) is True
    weak = {"spread_mae": 11.0, "market_spread_mae": 10.8, "brier": 0.21}
    assert beats_market_and_baseline(weak, baseline) is False


def test_shuffled_chronology_negative_control_flag():
    """Ablation configs must expose a neg-control that disables all roadmap upgrades."""
    from pipeline.ablation import default_configs
    cfgs = default_configs()
    assert "neg_control_no_hier_all" in cfgs
    ov = cfgs["neg_control_no_hier_all"]["config_overrides"]
    assert ov["USE_HIERARCHICAL_PACE"] is False
    assert ov["USE_TARGET_FAMILY_CALIBRATION"] is False
