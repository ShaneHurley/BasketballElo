"""Tests for shared forecast schema and OOF helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.shared_forecast import (
    SHARED_FORECAST_COLS,
    attach_oof_shared_forecasts,
    build_oof_shared_frame,
    emit_shared_forecast_from_feature_row,
    row_from_scores,
)


def test_row_from_scores_schema():
    row = row_from_scores(112.0, 108.0, decision_spread=-4.0, market_total=220.0)
    d = row.as_dict()
    assert set(SHARED_FORECAST_COLS) <= set(d)
    assert abs(d["sf_fair_margin"] - 4.0) < 1e-9
    assert abs(d["sf_fair_total"] - 220.0) < 1e-9
    assert abs(d["sf_decision_residual"] - 0.0) < 1e-9
    assert 0.0 < d["sf_home_win_prob"] < 1.0


def test_emit_from_feature_row():
    feat = {
        "decision_spread": -3.5,
        "market_total": 225.0,
        "h_rating_uncertainty": 200.0,
        "a_rating_uncertainty": 180.0,
    }
    preds = {"pred_home": 114.0, "pred_away": 110.0, "sigma_margin": 11.0}
    out = emit_shared_forecast_from_feature_row(feat, preds)
    assert out["sf_fair_margin"] == 4.0
    assert out["sf_dq_missing_decision"] == 0.0


def test_build_oof_shared_frame_marks_uncovered():
    df = pd.DataFrame({
        "decision_spread": [-4.0, -2.0],
        "market_total": [220.0, 218.0],
        "h_rating_uncertainty": [200.0, 200.0],
        "a_rating_uncertainty": [200.0, 200.0],
    })
    oof_h = np.array([110.0, np.nan])
    oof_a = np.array([108.0, np.nan])
    covered = np.array([True, False])
    out = build_oof_shared_frame(df, oof_h, oof_a, covered)
    assert out.loc[0, "sf_oof_covered"] == 1.0
    assert out.loc[1, "sf_oof_covered"] == 0.0


def test_attach_oof_shared_forecasts_past_only():
    n = 80
    rng = np.random.default_rng(0)
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    df = pd.DataFrame({
        "game_date": dates,
        "elo_margin": rng.normal(0, 5, n),
        "elo_net": rng.normal(0, 2, n),
        "hier_net": rng.normal(0, 2, n),
        "exp_poss": np.full(n, 100.0),
        "matchup_margin": rng.normal(0, 5, n),
        "matchup_total": rng.normal(220, 8, n),
        "h_rating_uncertainty": np.full(n, 200.0),
        "a_rating_uncertainty": np.full(n, 200.0),
        "pace_diff": rng.normal(0, 1, n),
        "decision_spread": rng.normal(-3, 2, n),
        "market_total": rng.normal(220, 5, n),
        "actual_home": rng.normal(112, 10, n),
        "actual_away": rng.normal(108, 10, n),
    })
    out = attach_oof_shared_forecasts(df, meta_model=None, n_splits=4)
    assert "sf_fair_margin" in out.columns
    # Burn-in / uncovered rows must be marked; covered rows finite.
    covered = out["sf_oof_covered"].fillna(0).astype(bool)
    assert covered.any()
    assert out.loc[covered, "sf_fair_margin"].notna().all()


def test_serve_emit_includes_sf_keys():
    feat = {
        "decision_spread": -4.0,
        "market_total": 220.0,
        "matchup_home_pts": 112.0,
        "matchup_away_pts": 108.0,
        "h_rating_uncertainty": 200.0,
        "a_rating_uncertainty": 180.0,
    }
    out = emit_shared_forecast_from_feature_row(
        feat,
        {"pred_home": 112.0, "pred_away": 108.0},
    )
    for k in ("sf_pred_home", "sf_pred_away", "sf_fair_margin", "sf_fair_total"):
        assert k in out
