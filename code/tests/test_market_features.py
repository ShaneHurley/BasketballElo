"""Tests for close-residual training, quantile heads, and market disagreement."""
import numpy as np
import pandas as pd
import pytest

from pipeline.market_targets import (
    margin_close_residual,
    residual_to_margin,
    closing_spread_series,
)
from pipeline.market_disagreement import (
    WalkForwardMarketDisagreementModel,
    phantom_injury_flag,
)
from pipeline.metrics import variance_aware_edge_threshold
from pipeline.model import MetaScoreModel, SAFE_FEATURE_COLS
from pipeline.ratings import PlayerRatingTracker


def test_margin_close_residual_roundtrip():
    actual = np.array([7.0, -3.0, 10.0])
    close = np.array([-4.0, 6.0, -7.0])
    resid = margin_close_residual(actual, close)
    assert np.allclose(resid, actual + close)
    assert np.allclose(residual_to_margin(resid, close), actual)


def test_closing_spread_series_prefers_close():
    df = pd.DataFrame({
        "closing_spread": np.full(25, -3.0),
        "market_spread": np.full(25, -2.5),
    })
    s = closing_spread_series(df)
    assert s is not None
    assert s.iloc[0] == -3.0


def test_variance_aware_edge_threshold():
    assert variance_aware_edge_threshold(3.5, 18.0) == 3.5
    bumped = variance_aware_edge_threshold(3.5, 30.0)
    assert bumped > 3.5
    assert bumped <= 3.5 + 2.5


def test_phantom_injury_flag():
    assert phantom_injury_flag(5.0, 1.0, 0, 0) is True
    assert phantom_injury_flag(5.0, 1.0, 1, 0) is False


def test_meta_model_close_residual_fit():
    n = 80
    rng = np.random.default_rng(0)
    decision = rng.normal(-4, 2, n)
    margin = rng.normal(0, 8, n)
    rows = {c: np.zeros(n) for c in SAFE_FEATURE_COLS}
    rows["elo_margin"] = margin + rng.normal(0, 2, n)
    rows["elo_net"] = rows["elo_margin"] / 10.0
    rows["exp_poss"] = np.full(n, 100.0)
    rows["decision_spread"] = decision
    rows["closing_spread"] = decision
    rows["market_spread"] = decision
    df = pd.DataFrame(rows)
    model = MetaScoreModel(
        ridge_alpha=5.0,
        use_quantile_heads=True,
        train_target="decision_residual",
        cb_params={"iterations": 50, "depth": 2, "verbose": 0, "allow_writing_files": False},
    )
    y_h = (margin + 220) / 2.0
    y_a = (220 - margin) / 2.0
    model.fit(df, y_h, y_a)
    assert model.fitted
    assert model._active_train_mode == "decision_residual"
    assert "q10" in model.quantile_models or model.quantile_models == {}
    pred = model.predict({**{c: float(df[c].iloc[0]) for c in SAFE_FEATURE_COLS},
                          "decision_spread": decision[0], "market_spread": decision[0],
                          "closing_spread": decision[0]})
    assert "pred_margin" in pred
    assert np.isfinite(pred["pred_margin"])


def test_rim_peri_def_split():
    t = PlayerRatingTracker(config={"K_DEF": 0.9, "ELO_SCALING_FACTOR": 1000})
    p1, p2 = ["201", "202"]
    stint_ctx = {
        "home_rim_rate": 0.40,
        "home_3pa_rate": 0.35,
        "away_rim_rate": 0.25,
        "away_3pa_rate": 0.42,
        "garbage": False,
    }
    t.process_stint([p1], [p2], 50, 55.0, 50.0, period=2, start_A=0, start_B=0, end_A=5, end_B=3,
                    stint_ctx=stint_ctx)
    pl = t._get(p2)
    assert "D_rim_mu" in pl
    assert "D_peri_mu" in pl
    rim, peri, _ = t._lineup_rim_peri([p2])
    blended = t.matchup_def_rating([p2], opp_rim_rate=0.40, opp_three_rate=0.35)
    assert np.isfinite(blended)


def test_disagreement_model_fit_predict():
    rng = np.random.default_rng(1)
    n = 120
    df = pd.DataFrame({
        "ACTUAL_MARGIN": rng.normal(0, 10, n),
        "MARKET_SPREAD": rng.normal(-3, 2, n),
        "CLOSING_SPREAD": rng.normal(-3, 2, n),
        "PRED_SPREAD": rng.normal(0, 8, n),
        "ELO_MARGIN_CALIBRATED": rng.normal(0, 9, n),
        "EDGE": rng.normal(0, 4, n),
        "RATING_UNCERTAINTY": np.full(n, 300.0),
        "H_STAR_OUT": np.zeros(n),
        "A_STAR_OUT": np.zeros(n),
        "SPREAD_MOVE": rng.normal(0, 0.5, n),
    })
    m = WalkForwardMarketDisagreementModel(min_train=50)
    m.fit(df)
    out = m.predict(
        elo_margin_calibrated=8.0,
        model_spread=2.0,
        closing_spread=-3.0,
        market_spread=-3.0,
    )
    assert "disagreement_trust" in out
    assert 0.2 <= out["disagreement_trust"] <= 1.2
