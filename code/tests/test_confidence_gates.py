"""Tests for confidence-only actionable gates and calib-split frame."""
import numpy as np
import pandas as pd

import pipeline.config as cfg
from pipeline.bet_confidence import build_calib_split_ats_frame
from pipeline.bet_selection import (
    edge_scaled_min_confidence,
    passes_confidence_actionable_gates,
    passes_edge_avoid_band,
)
from pipeline.ablation import calibration_ablation_configs


def test_edge_scaled_min_confidence(monkeypatch):
    monkeypatch.setattr(cfg, "CONFIDENCE_EDGE_SCALED", True)
    assert edge_scaled_min_confidence(3.0, base=55) == 59
    assert edge_scaled_min_confidence(6.0, base=55) == 57
    assert edge_scaled_min_confidence(9.0, base=55) == 55
    monkeypatch.setattr(cfg, "CONFIDENCE_EDGE_SCALED", False)
    assert edge_scaled_min_confidence(3.0, base=55) == 55


def test_passes_edge_avoid_band(monkeypatch):
    monkeypatch.setattr(cfg, "EDGE_AVOID_BAND", (7.0, 9.5))
    monkeypatch.setattr(cfg, "EDGE_AVOID_BAND_MIN_ELO_AGREE", 0.5)
    assert passes_edge_avoid_band(8.0, elo_meta_agreement=0.6)
    assert not passes_edge_avoid_band(8.0, elo_meta_agreement=0.3)
    assert passes_edge_avoid_band(10.0, elo_meta_agreement=0.0)


def test_passes_confidence_actionable_gates(monkeypatch):
    monkeypatch.setattr(cfg, "CONFIDENCE_SELECTION_MODE", "min_score")
    monkeypatch.setattr(cfg, "CONFIDENCE_EDGE_SCALED", False)
    monkeypatch.setattr(cfg, "MIN_DISAGREEMENT_TRUST", 0.70)
    monkeypatch.setattr(cfg, "SKIP_PHANTOM_INJURY", True)
    monkeypatch.setattr(cfg, "SKIP_TIGHT_SPREAD", True)  # exercise gate even if default is off
    monkeypatch.setattr(cfg, "EDGE_AVOID_BAND", None)
    monkeypatch.setattr(cfg, "CONFIDENCE_MIN_EDGE", 3.0)
    monkeypatch.setattr(cfg, "MAX_QUANTILE_WIDTH", 22.0)
    assert passes_confidence_actionable_gates(
        lean="Home",
        edge_pts=6.0,
        conf_score=58,
        conf_width=20.0,
        min_confidence=55,
        disagreement_trust=0.9,
        phantom_injury_flag=False,
        market_spread=-5.0,
    )
    assert not passes_confidence_actionable_gates(
        lean="Home",
        edge_pts=6.0,
        conf_score=58,
        conf_width=20.0,
        min_confidence=55,
        disagreement_trust=0.8,
        phantom_injury_flag=False,
        market_spread=-5.0,
    )
    assert not passes_confidence_actionable_gates(
        lean="Home",
        edge_pts=6.0,
        conf_score=58,
        conf_width=20.0,
        min_confidence=55,
        disagreement_trust=0.9,
        phantom_injury_flag=True,
        market_spread=-5.0,
    )
    assert not passes_confidence_actionable_gates(
        lean="Home",
        edge_pts=6.0,
        conf_score=58,
        conf_width=30.0,
        min_confidence=55,
        disagreement_trust=0.9,
        phantom_injury_flag=False,
        market_spread=-5.0,
    )
    assert not passes_confidence_actionable_gates(
        lean="Home",
        edge_pts=6.0,
        conf_score=58,
        conf_width=20.0,
        min_confidence=55,
        disagreement_trust=0.9,
        phantom_injury_flag=False,
        market_spread=-2.0,
    )


def test_build_calib_split_ats_frame():
    df = pd.DataFrame({
        "market_spread": [-3.0, 2.0, np.nan],
        "pred_margin": [1.0, -4.0, 0.0],
        "actual_margin": [5.0, -6.0, 0.0],
        "actual_home": [60, 50, 55],
        "actual_away": [55, 56, 55],
        "elo_margin_calibrated": [0.5, -1.0, 0.0],
    })
    out = build_calib_split_ats_frame(df, pred_margin_col="pred_margin")
    assert len(out) == 2
    assert "EDGE" in out.columns
    assert out.iloc[0]["DIRECTION"] == "Away"
    assert out.iloc[1]["DIRECTION"] == "Away"


def test_calibration_ablation_configs():
    configs = calibration_ablation_configs()
    assert "simplified_calib" in configs
    assert "beta_ats_calib" in configs
    assert "skellam_cover" in configs
    assert configs["beta_ats_calib"]["config_overrides"]["CALIBRATION_MODE"] == "beta_ats"
