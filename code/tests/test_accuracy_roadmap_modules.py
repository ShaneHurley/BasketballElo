"""Tests for hierarchical minutes, shot mix, structured scores, target calibration."""
from __future__ import annotations

import numpy as np

from pipeline.lineup_composite import composite_lineup_rating, lineup5_reliability
from pipeline.minutes_hierarchy import HierarchicalMinutesModel
from pipeline.shot_hierarchy import HierarchicalZoneRates
from pipeline.structured_score import hybrid_blend, structured_score_from_pace_pps
from pipeline.target_calibration import TargetFamilyCalibrationRegistry, TeamPooledCalibrator


def test_hierarchical_minutes_sum_to_240():
    m = HierarchicalMinutesModel()
    for i in range(10):
        m.record_game("p1", 34.0, True, team="BOS")
        m.record_game("p2", 28.0, True, team="BOS")
        m.record_game("p3", 22.0, True, team="BOS")
        m.record_game("p4", 18.0, True, team="BOS")
        m.record_game("p5", 12.0, True, team="BOS")
    out = m.forecast_team(["p1", "p2", "p3", "p4", "p5", "p6"])
    assert abs(sum(out.values()) - 240.0) < 1e-6


def test_minutes_factor_probs_learn():
    m = HierarchicalMinutesModel()
    for _ in range(30):
        m.record_team_rest_night(3, roster_size=8)
    for _ in range(70):
        m.record_team_rest_night(0, roster_size=8)
    fp = m.factor_probs()
    assert abs(sum(fp.values()) - 1.0) < 1e-9
    assert fp["rest_night"] > 0.05


def test_shot_hierarchy_mix_and_pps():
    rates = HierarchicalZoneRates()
    # BOS: heavy rim mix + high make; SAS: heavy three mix
    for _ in range(300):
        rates.update_shot(zone="restricted", made=True, team="BOS")
    for _ in range(50):
        rates.update_shot(zone="abovebreak3", made=False, team="BOS")
    for _ in range(50):
        rates.update_shot(zone="restricted", made=False, team="SAS")
    for _ in range(300):
        rates.update_shot(zone="abovebreak3", made=True, team="SAS")
    feats = rates.feature_dict("BOS", "SAS")
    assert feats["h_rim_share"] > feats["a_rim_share"]
    assert feats["a_three_share"] > feats["h_three_share"]
    assert "h_hier_shot_pps" in feats


def test_structured_score_realistic_totals():
    fc = structured_score_from_pace_pps(
        exp_poss=100.0, home_pps=1.12, away_pps=1.08, pace_var=9.0,
    )
    assert 180.0 <= fc.total <= 260.0
    assert fc.sigma_total > 0
    assert -0.95 <= fc.corr <= 0.95
    hyb = hybrid_blend(220.0, 3.0, fc)
    assert "hybrid_total" in hyb


def test_lineup_evidence_pooling_reduces_sparse_weight():
    sparse = composite_lineup_rating(
        player_off_delta=100, player_def_delta=0,
        lineup5_net=50, chem_duo_net=10, chem_trio_net=5,
        lineup5_sample_poss=5.0, evidence_pooling=True,
    )
    rich = composite_lineup_rating(
        player_off_delta=100, player_def_delta=0,
        lineup5_net=50, chem_duo_net=10, chem_trio_net=5,
        lineup5_sample_poss=500.0, evidence_pooling=True,
    )
    # Both finite; reliability increases with sample
    assert np.isfinite(sparse) and np.isfinite(rich)
    assert lineup5_reliability(500) > lineup5_reliability(5)


def test_target_family_calibrator_partial_pool():
    cal = TeamPooledCalibrator(window=200, min_samples=40)
    rng = np.random.default_rng(0)
    for i in range(80):
        p = 0.4 + 0.2 * (i % 2)
        y = int(rng.random() < p)
        cal.update(p, y, team="BOS" if i % 2 == 0 else "NYK")
    assert cal.fitted
    p0 = cal.predict(0.6, team=None)
    p1 = cal.predict(0.6, team="BOS")
    assert 0.01 <= p0 <= 0.99
    assert 0.01 <= p1 <= 0.99
    reg = TargetFamilyCalibrationRegistry()
    assert set(reg.families()) == {"spread_cover", "total_over", "moneyline"}
