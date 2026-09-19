"""Tests for upset / blowout / calibration plan changes."""
import numpy as np
import pandas as pd
import pytest

import pipeline.config as cfg
from pipeline.bet_selection import (
    normalize_edge_avoid_bands,
    passes_confidence_actionable_gates,
    passes_edge_avoid_band,
)
from pipeline.market import SpreadCalibrator, spread_cover_prob
from pipeline.metrics import add_clv_columns
from pipeline.ml_calibration import WalkForwardMLCalibrator
from pipeline.upset_classifier import UpsetClassifier, favorite_lost_label


def test_normalize_edge_avoid_bands_single_and_multi():
    assert normalize_edge_avoid_bands(None) == []
    assert normalize_edge_avoid_bands((3.0, 4.0)) == [(3.0, 4.0)]
    assert normalize_edge_avoid_bands([(3.0, 4.0), (5.5, 6.0)]) == [(3.0, 4.0), (5.5, 6.0)]


def test_passes_edge_avoid_band_multi(monkeypatch):
    monkeypatch.setattr(cfg, "EDGE_AVOID_BAND", [(3.0, 4.0), (5.5, 6.0)])
    monkeypatch.setattr(cfg, "EDGE_AVOID_BAND_MIN_ELO_AGREE", 0.5)
    assert passes_edge_avoid_band(3.5, elo_meta_agreement=0.0) is False
    assert passes_edge_avoid_band(5.75, elo_meta_agreement=0.0) is False
    assert passes_edge_avoid_band(3.5, elo_meta_agreement=1.0) is True
    assert passes_edge_avoid_band(6.5, elo_meta_agreement=0.0) is True


def test_spread_cover_prob_heteroscedastic():
    lo = spread_cover_prob(10.0, -6.0, conf_width=24.0, matchup_vol_sigma=8.0)
    hi = spread_cover_prob(10.0, -6.0, conf_width=24.0, matchup_vol_sigma=20.0)
    # Higher vol should pull cover prob toward 0.5
    assert abs(hi - 0.5) <= abs(lo - 0.5) + 1e-9


def test_binned_spread_calibrator_large_favorite():
    cal = SpreadCalibrator(window=80, min_samples=20, binned=True)
    rng = np.random.default_rng(0)
    for _ in range(60):
        pred = float(rng.uniform(-3, 3))
        actual = 0.9 * pred + float(rng.normal(0, 1))
        cal.update(pred, actual)
    for _ in range(40):
        pred = float(rng.uniform(8, 14))
        actual = 0.6 * pred + float(rng.normal(0, 2))  # shrink big favorites
        cal.update(pred, actual)
    small = cal.correct(2.0)
    large = cal.correct(12.0)
    # Large favorite should be shrunk more aggressively toward smaller margin.
    assert abs(large) < abs(12.0)
    assert np.isfinite(small) and np.isfinite(large)


def test_clv_identical_decision_close_is_nan():
    df = pd.DataFrame({
        "MARKET_SPREAD": [-5.0, -3.0],
        "DECISION_SPREAD": [-5.0, -4.0],
        "CLOSING_SPREAD": [-5.0, -3.5],
        "PRED_SPREAD": [1.0, 2.0],
        "ACTUAL_MARGIN": [4.0, -1.0],
        "DIRECTION": ["Home", "Away"],
    })
    out = add_clv_columns(df)
    assert pd.isna(out.loc[0, "CLV"])
    assert pd.notna(out.loc[1, "CLV"])


def test_favorite_lost_label():
    assert favorite_lost_label({"MARKET_SPREAD": -7.0, "ACTUAL_MARGIN": -3.0}) == 1.0
    assert favorite_lost_label({"MARKET_SPREAD": -7.0, "ACTUAL_MARGIN": 5.0}) == 0.0
    assert favorite_lost_label({"MARKET_SPREAD": 4.0, "ACTUAL_MARGIN": 2.0}) == 1.0  # away fav lost


def test_upset_classifier_fit_predict():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(200):
        mkt = float(rng.choice([-8, -5, -3, 3, 5, 8]))
        # Higher vol → more upsets
        vol = float(rng.uniform(8, 18))
        upset = rng.random() < (0.15 + 0.02 * (vol - 8))
        home_fav = mkt < 0
        if home_fav:
            margin = -5.0 if upset else 8.0
        else:
            margin = 5.0 if upset else -8.0
        rows.append({
            "MARKET_SPREAD": mkt,
            "ACTUAL_MARGIN": margin,
            "PRED_SPREAD": -mkt * 0.8,
            "MATCHUP_VOL_SIGMA": vol,
            "SPREAD_QUANTILE_WIDTH": 24.0,
            "ELO_META_AGREEMENT": 1.0,
            "RATING_UNCERTAINTY": 700.0,
            "H_STAR_OUT": 0,
            "A_STAR_OUT": 0,
            "DISAGREEMENT_TRUST": 1.0,
            "WIN_PROB": 0.65 if home_fav else 0.35,
            "h_rest": 1 if upset else 3,
            "a_rest": 3 if upset else 1,
            "h_b2b": 1 if upset and home_fav else 0,
            "a_b2b": 1 if upset and not home_fav else 0,
            "h_travel_miles_7d": 2000 if upset and home_fav else 100,
            "a_travel_miles_7d": 2000 if upset and not home_fav else 100,
            "h_pts_std_roll_10": vol,
            "a_pts_std_roll_10": vol,
            "h_pythag_residual": -0.1 if upset and home_fav else 0.05,
            "a_pythag_residual": -0.1 if upset and not home_fav else 0.05,
            "h_missing_rotation": 0.2 if upset else 0.0,
            "a_missing_rotation": 0.0,
        })
    df = pd.DataFrame(rows)
    clf = UpsetClassifier()
    clf.fit(df.iloc[:150], calib_df=df.iloc[150:])
    assert clf.fitted
    p = clf.predict_upset_prob(df.iloc[0].to_dict())
    assert 0.01 <= p <= 0.99


def test_scenario_mix_sigma_proxy_and_upset_dampen():
    from pipeline.market import dampen_cover_for_upset, scenario_mix_sigma_proxy

    base = scenario_mix_sigma_proxy(matchup_vol_sigma=12.0, conf_width=24.0)
    high = scenario_mix_sigma_proxy(
        matchup_vol_sigma=12.0, conf_width=24.0,
        h_missing_rotation=0.4, a_missing_rotation=0.3, h_star_out=1,
    )
    assert high >= base
    raw = 0.70
    damp = dampen_cover_for_upset(
        raw, direction="Home", market_spread=-7.0, upset_prob=0.75,
    )
    assert 0.5 <= damp < raw
    # Dog lean should not dampen
    assert dampen_cover_for_upset(
        raw, direction="Away", market_spread=-7.0, upset_prob=0.75,
    ) == raw


def test_ml_split_calibrator_ece_gate():
    rng = np.random.default_rng(2)
    n = 300
    # Raw probs already well calibrated → split calib should often disable itself
    p = rng.uniform(0.2, 0.8, size=n)
    y = (rng.random(n) < p).astype(int)
    df = pd.DataFrame({
        "WIN_PROB_RAW": p,
        "ACTUAL_HOME": y * 110 + (1 - y) * 100,
        "ACTUAL_AWAY": (1 - y) * 110 + y * 100,
        "MARKET_ML": np.where(p > 0.5, -150, 140),
        "simulated_season_window": ["2024-2025"] * n,
    })
    cal = WalkForwardMLCalibrator(method="isotonic")
    cal.fit(df, min_samples=80)
    assert cal._fitted
    # transform always returns a valid probability
    out = cal.transform(0.7, market_ml=-150)
    assert 0.01 <= out <= 0.99


def test_plan_gates_require_higher_edge_and_conf(monkeypatch):
    monkeypatch.setattr(cfg, "BET_SELECTION_MODE", "edge_bucket")
    monkeypatch.setattr(cfg, "CONFIDENCE_MIN_EDGE", 5.5)
    monkeypatch.setattr(cfg, "MIN_EDGE_BUCKET", 5.5)
    monkeypatch.setattr(cfg, "MIN_CONFIDENCE_SCORE", 55)
    monkeypatch.setattr(cfg, "CONFIDENCE_EDGE_SCALED", False)
    monkeypatch.setattr(cfg, "EDGE_AVOID_BAND", [(3.0, 4.0), (5.5, 6.0)])
    monkeypatch.setattr(cfg, "MIN_DISAGREEMENT_TRUST", 0.7)
    monkeypatch.setattr(cfg, "SKIP_PHANTOM_INJURY", True)
    monkeypatch.setattr(cfg, "SKIP_TIGHT_SPREAD", False)
    monkeypatch.setattr(cfg, "CONFIDENCE_SELECTION_MODE", "min_score")
    assert passes_confidence_actionable_gates(
        lean="Home", edge_pts=4.0, conf_score=60, conf_width=22,
        min_confidence=55,
    ) is False  # below min edge
    assert passes_confidence_actionable_gates(
        lean="Home", edge_pts=7.0, conf_score=50, conf_width=22,
        min_confidence=55,
    ) is False  # below min conf
    assert passes_confidence_actionable_gates(
        lean="Home", edge_pts=7.0, conf_score=60, conf_width=22,
        min_confidence=55, elo_meta_agreement=1.0,
    ) is True


def test_clamp_confidence_weights_cover_scale():
    from pipeline.bet_confidence import clamp_confidence_weights

    w = clamp_confidence_weights({"cover_scale": 312.0, "vol_penalty_scale": 200.0})
    assert w["cover_scale"] <= 80.0
    assert w["vol_penalty_scale"] <= 40.0


def test_tier_2_stake_mult_zero(monkeypatch):
    from pipeline.bet_selection import confidence_tier_stake_mult

    monkeypatch.setattr(cfg, "CONFIDENCE_TIER_2_STAKE_MULT", 0.0)
    assert confidence_tier_stake_mult(2) == 0.0
    assert confidence_tier_stake_mult(3) == 1.0


def test_ml_favorite_extra_shrink(monkeypatch):
    from pipeline.market import ml_prob_for_side

    monkeypatch.setattr(cfg, "ML_MARKET_SHRINK", 0.0)
    monkeypatch.setattr(cfg, "ML_UNDERDOG_EXTRA_SHRINK", 0.0)
    monkeypatch.setattr(cfg, "ML_FAVORITE_EXTRA_SHRINK", 0.20)
    monkeypatch.setattr(cfg, "ML_FAV_ABS_SPREAD_MIN", 8.0)
    monkeypatch.setattr(cfg, "ML_FAV_WIN_PROB_MIN", 0.75)
    monkeypatch.setattr(cfg, "ML_MARKET_DEVIG", False)
    p = ml_prob_for_side(0.90, -150.0, "Home", market_spread=-10.0)
    assert p < 0.90
