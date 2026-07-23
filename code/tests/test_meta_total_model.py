"""Tests for MetaTotalModel and related tuning helpers."""
import numpy as np
import pandas as pd

from pipeline.model import (
    MetaTotalModel,
    MetaWinModel,
    total_feature_cols,
    win_feature_cols,
    tune_meta_win_model,
)


def _sample_features(n=120):
    rng = np.random.default_rng(42)
    rows = []
    for i in range(n):
        market_total = 220 + rng.normal(0, 4)
        actual_total = market_total + rng.normal(0, 10)
        actual_home = actual_total / 2 + rng.normal(0, 3)
        actual_away = actual_total - actual_home
        row = {
            "actual_total": actual_total,
            "actual_home": actual_home,
            "actual_away": actual_away,
            "actual_margin": actual_home - actual_away,
            "market_total": market_total,
            "exp_poss": 98 + rng.normal(0, 2),
            "pace_diff": rng.normal(0, 2),
            "pace_abs_diff": abs(rng.normal(0, 2)),
            "pace_interaction": rng.normal(0, 1),
            "h_rest": rng.integers(0, 4),
            "a_rest": rng.integers(0, 4),
            "h_b2b": rng.integers(0, 2),
            "a_b2b": rng.integers(0, 2),
            "market_total_minus_league": market_total - 225,
            "ref_pace_bias": rng.normal(0, 0.5),
            "ref_foul_bias": rng.normal(0, 0.5),
            "team_elo_total_adj": rng.normal(0, 3),
            "team_elo_net": rng.normal(0, 5),
            "rtg_pace_interaction": rng.normal(0, 1),
            "form_pace_interaction": rng.normal(0, 1),
            "elo_pace_interaction": rng.normal(0, 1),
            "h_fatigue_index": rng.uniform(0, 1),
            "a_fatigue_index": rng.uniform(0, 1),
            "fatigue_diff": rng.normal(0, 0.2),
            "h_roll_off_xppp": 1.1 + rng.normal(0, 0.05),
            "a_roll_off_xppp": 1.1 + rng.normal(0, 0.05),
            "h_roll_def_xppp": 1.1 + rng.normal(0, 0.05),
            "a_roll_def_xppp": 1.1 + rng.normal(0, 0.05),
            "roll_net_xppp": rng.normal(0, 0.05),
            "h_off_rating": 110 + rng.normal(0, 5),
            "a_off_rating": 110 + rng.normal(0, 5),
            "h_def_rating": 110 + rng.normal(0, 5),
            "a_def_rating": 110 + rng.normal(0, 5),
            "off_rating_diff": rng.normal(0, 4),
            "def_rating_diff": rng.normal(0, 4),
            "h_3pt_rate": 0.35 + rng.normal(0, 0.03),
            "a_3pt_rate": 0.35 + rng.normal(0, 0.03),
            "three_rate_diff": rng.normal(0, 0.03),
            "h_sos": rng.normal(0, 1),
            "a_sos": rng.normal(0, 1),
            "sos_diff": rng.normal(0, 1),
            "elo_net": rng.normal(0, 5),
            "pred_margin": rng.normal(0, 4),
            "market_ml": -110 if rng.random() > 0.5 else 120,
        }
        rows.append(row)
    return pd.DataFrame(rows)


def test_meta_total_model_fit_predict():
    df = _sample_features()
    cols = total_feature_cols(df)
    model = MetaTotalModel(feature_cols=cols, train_target="absolute")
    model.fit(df)
    feat = df.iloc[0].to_dict()
    pred = model.predict_total(feat)
    assert np.isfinite(pred)
    assert 180 < pred < 260


def test_tune_meta_win_smoke():
    df = _sample_features(80)
    params = tune_meta_win_model(df, n_trials=2, pred_margin_col="pred_margin", fast_mode=True)
    assert "C" in params
    win = MetaWinModel(
        C=params["C"],
        feature_cols=params.get("feature_cols") or win_feature_cols(df),
        elo_win_blend=params.get("elo_win_blend"),
        calib_method=params.get("calib_method", "isotonic"),
    )
    calib = df.iloc[-20:]
    train = df.iloc[:-20]
    win.fit(
        train,
        (train["actual_home"] > train["actual_away"]).astype(int),
        calib_df=calib,
    )
    p = win.predict_proba(train.iloc[0].to_dict(), pred_margin=float(train.iloc[0]["pred_margin"]))
    assert 0.01 <= p <= 0.99


def test_market_targets_helpers():
    from pipeline.market_targets import total_market_residual, residual_to_total

    r = total_market_residual(np.array([220.0]), np.array([215.0]))
    assert r[0] == 5.0
    t = residual_to_total(np.array([5.0]), np.array([215.0]))
    assert t[0] == 220.0
