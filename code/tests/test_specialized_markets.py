"""Tests for Gaussian O/U probs, CRPS, both-sides ML explain, score-pair, SHAP prune."""
import numpy as np
import pandas as pd
import pytest

from pipeline.market import (
    total_over_prob_gaussian,
    total_under_prob_gaussian,
    crps_gaussian,
    select_ou_bet,
    explain_ml_bet,
)
from pipeline.shap_prune import prune_bottom_fraction, stable_top_features
from pipeline.feature_utils import rest_bucket_flags, schedule_density_extended
from pipeline.model import WIN_FEATURE_COLS, TOTAL_FEATURE_COLS, MetaScorePairModel


def test_total_over_prob_symmetric():
    # When pred == line, P(Over) ≈ 0.5
    p = total_over_prob_gaussian(220.0, 220.0, 10.0)
    assert abs(p - 0.5) < 0.02


def test_total_over_prob_high_when_mu_above_line():
    p = total_over_prob_gaussian(230.0, 220.0, 10.0)
    assert p > 0.7
    assert abs(total_under_prob_gaussian(230.0, 220.0, 10.0) - (1 - p)) < 1e-9


def test_crps_gaussian_nonnegative():
    c = crps_gaussian(225.0, 220.0, 12.0)
    assert np.isfinite(c) and c >= 0


def test_select_ou_bet_pass_on_small_edge():
    side, p_over, edge = select_ou_bet(221.0, 220.0, 12.0, min_edge=4.5, min_prob=0.55)
    assert side == "Pass"


def test_select_ou_bet_over_with_edge_and_prob():
    side, p_over, edge = select_ou_bet(232.0, 220.0, 10.0, min_edge=4.5, min_prob=0.55)
    assert side == "Over"
    assert p_over >= 0.55
    assert edge > 4.5


def test_explain_ml_bet_both_sides():
    out = explain_ml_bet(0.62, -150, min_ev=0.03, min_win_pct=50, winner_only=False)
    assert out["predicted_winner"] == "Home"
    assert "ev_home" in out and "ev_away" in out
    assert out["ml_bet"] in ("Home", "Away", "Pass")
    assert out["ml_reason"]


def test_explain_ml_bet_pass_heavy_favorite():
    # Very short favorite — model may predict home but EV fails → Pass
    out = explain_ml_bet(0.70, -400, min_ev=0.08, min_win_pct=55, winner_only=False,
                         max_favorite_decimal=1.45)
    assert out["predicted_winner"] == "Home"
    # Either Pass or Away value; not forced Home solely from prediction
    if out["ml_bet"] == "Home":
        assert out["ev_home"] > 0.08


def test_rest_buckets():
    assert rest_bucket_flags(0)["rest_0"] == 1
    assert rest_bucket_flags(3)["rest_3plus"] == 1
    assert rest_bucket_flags(1)["rest_1"] == 1


def test_schedule_density_extended():
    dates = [pd.Timestamp("2024-01-01"), pd.Timestamp("2024-01-02"), pd.Timestamp("2024-01-03")]
    last7, three_in_4, four_in_6 = schedule_density_extended(dates, pd.Timestamp("2024-01-04"))
    assert three_in_4 == 1
    assert last7 >= 2


def test_feature_matrices_differ():
    # Moneyline and totals must not share identical matrices
    assert set(WIN_FEATURE_COLS) != set(TOTAL_FEATURE_COLS)
    # Elo heavily present in WIN; de-emphasized in TOTAL
    assert "elo_net" in WIN_FEATURE_COLS
    assert "elo_net" not in TOTAL_FEATURE_COLS
    assert "exp_poss" in TOTAL_FEATURE_COLS


def test_shap_prune_helpers():
    imp = pd.Series({"a": 0.5, "b": 0.3, "c": 0.1, "d": 0.05, "e": 0.02})
    kept = prune_bottom_fraction(imp, drop_frac=0.3)
    assert "a" in kept and "e" not in kept
    fold1 = pd.Series({"a": 1, "b": 0.8, "c": 0.2, "d": 0.1})
    fold2 = pd.Series({"a": 0.9, "b": 0.7, "c": 0.05, "d": 0.01})
    stable = stable_top_features([fold1, fold2], top_frac=0.5, min_fold_frac=0.8)
    assert "a" in stable


def test_meta_score_pair_fit_predict():
    rng = np.random.default_rng(0)
    n = 80
    df = pd.DataFrame({
        "exp_poss": rng.normal(100, 3, n),
        "pace_diff": rng.normal(0, 2, n),
        "h_off_rtg": rng.normal(113, 4, n),
        "a_off_rtg": rng.normal(113, 4, n),
        "h_def_rtg": rng.normal(113, 4, n),
        "a_def_rtg": rng.normal(113, 4, n),
        "market_total": rng.normal(225, 8, n),
        "h_rest": rng.integers(0, 4, n),
        "a_rest": rng.integers(0, 4, n),
    })
    df["actual_home"] = 110 + 0.1 * df["h_off_rtg"] + rng.normal(0, 5, n)
    df["actual_away"] = 108 + 0.1 * df["a_off_rtg"] + rng.normal(0, 5, n)
    model = MetaScorePairModel(
        feature_cols=["exp_poss", "pace_diff", "h_off_rtg", "a_off_rtg", "h_def_rtg", "a_def_rtg", "market_total"],
        cb_params={"depth": 2, "iterations": 30, "learning_rate": 0.1, "verbose": 0,
                   "random_seed": 0, "thread_count": 1, "allow_writing_files": False},
    )
    model.fit(df)
    out = model.predict_scores(df.iloc[0].to_dict())
    assert "pred_home" in out and "pred_away" in out
    assert abs(out["pred_total"] - (out["pred_home"] + out["pred_away"])) < 1e-6
    assert out["sigma_total"] > 0
