"""Regression tests for direct actual-score (home/away) training plan."""
from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


def _toy_score_frame(n: int = 80, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    elo = rng.normal(0, 8, n)
    pace = rng.normal(100, 3, n)
    home = 110 + 0.35 * elo + 0.1 * (pace - 100) + rng.normal(0, 8, n)
    away = 108 - 0.30 * elo + 0.1 * (pace - 100) + rng.normal(0, 8, n)
    return pd.DataFrame({
        "elo_net": elo,
        "elo_margin": elo * 0.9,
        "hier_net": elo * 0.8,
        "exp_poss": pace,
        "pace_diff": rng.normal(0, 2, n),
        "h_rating_uncertainty": rng.uniform(200, 500, n),
        "a_rating_uncertainty": rng.uniform(200, 500, n),
        "matchup_margin": elo * 0.7,
        "matchup_total": 2 * pace,
        "h_rest": rng.integers(0, 4, n),
        "a_rest": rng.integers(0, 4, n),
        "game_date": pd.date_range("2023-11-01", periods=n, freq="D"),
        "actual_home": home,
        "actual_away": away,
        "actual_margin": home - away,
        "actual_total": home + away,
        # Market columns present in frame but must be ignored by score-safe fit.
        "market_total": 220 + rng.normal(0, 5, n),
        "decision_spread": -elo * 0.5,
        "market_spread": -elo * 0.5,
        "elo_vs_market": rng.normal(0, 1, n),
    })


def test_canonical_labels_are_actual_home_away():
    from pipeline.score_targets import SCORE_LABEL_COLS, DERIVED_SCORE_TARGETS
    assert SCORE_LABEL_COLS == ("actual_home", "actual_away")
    assert "actual_margin" in DERIVED_SCORE_TARGETS
    assert "actual_total" in DERIVED_SCORE_TARGETS


def test_score_safe_features_exclude_market():
    from pipeline.score_targets import (
        assert_score_safe_features,
        filter_score_safe_features,
        score_safe_feature_cols,
    )
    cols = score_safe_feature_cols()
    assert "market_total" not in cols
    assert "decision_spread" not in cols
    assert "elo_vs_market" not in cols
    assert "fair_spread_vigfree" not in cols
    with pytest.raises(ValueError):
        assert_score_safe_features(["elo_net", "market_total"])
    cleaned = filter_score_safe_features(
        ["elo_net", "market_total", "decision_spread", "pace_diff"],
        enforce=True,
    )
    assert cleaned == ["elo_net", "pace_diff"]


def test_meta_score_pair_rejects_market_features_and_predicts_algebra():
    from pipeline.model import MetaScorePairModel
    from pipeline.score_targets import score_safe_feature_cols

    df = _toy_score_frame()
    cols = score_safe_feature_cols(df)
    assert "market_total" not in cols
    model = MetaScorePairModel(
        feature_cols=cols + ["market_total"],  # sneaky market col
        cb_params={"depth": 3, "iterations": 40, "learning_rate": 0.08,
                   "verbose": 0, "random_seed": 0, "thread_count": 1,
                   "allow_writing_files": False},
        enforce_score_safe_features=True,
    )
    model.fit(df)
    assert "market_total" not in model._fit_cols_
    out = model.predict_scores(df.iloc[0].to_dict())
    assert abs((out["pred_home"] - out["pred_away"]) - out["pred_margin"]) < 1e-6
    assert abs((out["pred_home"] + out["pred_away"]) - out["pred_total"]) < 1e-6
    assert out["forecast_source"] == "score_pair"
    assert out["sigma_margin"] > 0 and out["sigma_total"] > 0
    assert "home_qlo_50" in out and "margin_qhi_95" in out
    assert -1.0 <= out["score_residual_corr"] <= 1.0


def test_meta_score_pair_forbids_non_absolute_target():
    from pipeline.model import MetaScorePairModel
    with pytest.raises(ValueError):
        MetaScorePairModel(train_target="decision_residual")


def test_joint_intervals_and_cover_prob():
    from pipeline.score_targets import (
        joint_score_moments,
        margin_home_cover_prob,
        margin_win_prob,
        score_predictive_intervals,
        total_over_prob,
    )
    m = joint_score_moments(10, 10, 0.4)
    assert m["sigma_total"] > m["sigma_margin"]  # positive corr → total wider
    iv = score_predictive_intervals(112, 108, sigma_home=10, sigma_away=10, corr=0.3)
    assert iv["CONF_WIDTH"] > 0
    assert 0.01 <= margin_win_prob(4.0, 12.0) <= 0.99
    assert 0.01 <= margin_home_cover_prob(2.0, -3.5, 12.0) <= 0.99
    assert 0.01 <= total_over_prob(225, 220, 14) <= 0.99


def test_canonical_apply_overwrites_legacy_margin():
    from pipeline.canonical_scores import apply_canonical_score_pair, prefer_canonical_margin
    from pipeline.model import MetaScorePairModel
    from pipeline.score_targets import score_safe_feature_cols

    df = _toy_score_frame(60)
    model = MetaScorePairModel(
        feature_cols=score_safe_feature_cols(df),
        cb_params={"depth": 3, "iterations": 30, "learning_rate": 0.1,
                   "verbose": 0, "random_seed": 1, "thread_count": 1,
                   "allow_writing_files": False},
    )
    model.fit(df)
    preds = {"pred_home": 100, "pred_away": 100, "pred_margin": 0.0, "pred_total": 200}
    feat = df.iloc[-1].to_dict()
    feat["struct_home_pts"] = feat.get("actual_home", 110)
    feat["struct_away_pts"] = feat.get("actual_away", 108)
    out = apply_canonical_score_pair(preds, feat, model, legacy_margin=1.0)
    assert out["forecast_source"] == "score_pair"
    assert abs((out["pred_home"] - out["pred_away"]) - out["pred_margin"]) < 1e-6
    assert prefer_canonical_margin(out, 99.0) == pytest.approx(out["pred_margin"])


def test_score_pair_persistence_roundtrip(tmp_path: Path):
    from pipeline.model import MetaScorePairModel
    from pipeline.score_targets import score_safe_feature_cols

    df = _toy_score_frame(50)
    model = MetaScorePairModel(
        feature_cols=score_safe_feature_cols(df),
        cb_params={"depth": 3, "iterations": 25, "learning_rate": 0.1,
                   "verbose": 0, "random_seed": 2, "thread_count": 1,
                   "allow_writing_files": False},
    )
    model.fit(df)
    path = tmp_path / "score_pair.pkl"
    model.save(path)
    loaded = MetaScorePairModel.load(path)
    a = model.predict_scores(df.iloc[0].to_dict())
    b = loaded.predict_scores(df.iloc[0].to_dict())
    assert a["pred_home"] == pytest.approx(b["pred_home"], abs=1e-6)
    meta = model.metadata()
    assert meta["schema_version"] == MetaScorePairModel.MODEL_SCHEMA_VERSION
    assert meta["train_target"] == "absolute"


def test_direct_score_ablations_and_gates():
    from pipeline.ablation import (
        direct_score_ablation_configs,
        passes_direct_score_promotion_gate,
    )
    cfgs = direct_score_ablation_configs()
    assert "direct_score_pair" in cfgs
    assert "canonical_score_combined" in cfgs
    assert "legacy_residual_stack" in cfgs
    gate = passes_direct_score_promotion_gate(
        {"spread_mae": 12.5, "market_spread_mae": 11.0, "brier": 0.25,
         "paired_score_mae": 9.5, "total_mae": 14.0},
        {"spread_mae": 12.0, "market_spread_mae": 11.0, "brier": 0.24,
         "paired_score_mae": 9.0, "total_mae": 13.5,
         "score_algebra_ok_pct": 1.0, "interval_coverage": 0.81,
         "market_error_corr": 0.05, "n_finite_clv": 3},
        odds_provenance={"promotion_eligible": False},
    )
    assert gate["paired_score_ok"] is True
    assert gate["algebra_ok"] is True
    assert gate["betting_edge_claim_allowed"] is False


def test_negative_controls_score_swap_and_folds():
    from pipeline.negative_controls import (
        locked_fold_definitions,
        market_features_forbidden_in_score_matrix,
        score_swap_destroys_pair_advantage,
    )
    rng = np.random.default_rng(0)
    ph = rng.normal(110, 5, 40)
    pa = rng.normal(108, 5, 40)
    ah = ph + rng.normal(0, 2, 40)
    aa = pa + rng.normal(0, 2, 40)
    assert score_swap_destroys_pair_advantage(ph, pa, ah, aa)["passed"]
    assert market_features_forbidden_in_score_matrix(["elo_net"])["passed"]
    assert not market_features_forbidden_in_score_matrix(["market_total"])["passed"]
    folds = locked_fold_definitions()
    assert folds["immutable"] is True
    assert len(folds["folds"]) >= 3
    assert "direct_score_pair" in folds["configs"]


def test_compute_metrics_includes_paired_scores():
    from pipeline.metrics import compute_metrics

    rng = np.random.default_rng(1)
    n = 40
    df = pd.DataFrame({
        "PRED_HOME": 110 + rng.normal(0, 5, n),
        "PRED_AWAY": 108 + rng.normal(0, 5, n),
        "ACTUAL_HOME": 110 + rng.normal(0, 8, n),
        "ACTUAL_AWAY": 108 + rng.normal(0, 8, n),
        "PRED_SPREAD": rng.normal(0, 6, n),
        "PRED_TOTAL": 218 + rng.normal(0, 8, n),
        "ACTUAL_MARGIN": rng.normal(0, 12, n),
        "MARKET_SPREAD": rng.normal(0, 6, n),
        "DIRECTION": ["Pass"] * n,
        "WIN_PROB": rng.uniform(0.3, 0.7, n),
        "simulated_season_window": ["2024-2025"] * n,
        "CONF_LOWER": -20.0,
        "CONF_UPPER": 20.0,
    })
    # Enforce algebra for a subset
    df["PRED_SPREAD"] = df["PRED_HOME"] - df["PRED_AWAY"]
    df["PRED_TOTAL"] = df["PRED_HOME"] + df["PRED_AWAY"]
    m = compute_metrics(df)
    row = m[m["season"] == "ALL"].iloc[0]
    assert np.isfinite(row["paired_score_mae"])
    assert np.isfinite(row["home_mae"])
    assert row["score_algebra_ok_pct"] == pytest.approx(1.0)


def test_feature_provenance_buckets():
    from pipeline.score_targets import assert_score_feature_provenance, score_safe_feature_cols
    cols = score_safe_feature_cols()[:40]
    buckets = assert_score_feature_provenance(cols)
    assert isinstance(buckets, dict)
    assert "ratings" in buckets


def test_config_canonical_score_pair_enabled():
    from pipeline import config
    assert config.USE_CANONICAL_SCORE_PAIR is True
    assert config.USE_SCORE_PAIR_TOTAL is True
