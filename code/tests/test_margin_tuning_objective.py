"""Task 055 — margin-model Optuna loss must be margin/probability-first:
no validation-fold ROI/edge-grid/CLV terms, correct overfit-penalty sign,
and tuning must run without any betting columns present.
"""
import inspect

import numpy as np
import optuna
import pandas as pd
import pytest

from pipeline.model import SAFE_FEATURE_COLS, _margin_tuning_objective, tune_margin_model


def _synthetic(n=120, seed=11, with_market=True):
    rng = np.random.RandomState(seed)
    dates = pd.date_range("2025-10-01", periods=n, freq="D")
    df = pd.DataFrame({c: rng.normal(size=n) for c in SAFE_FEATURE_COLS if c != "elo_stack_pred"})
    df["game_date"] = dates
    df["GAME_ID"] = np.arange(n)
    home = 100 + rng.normal(scale=8, size=n) + df["elo_net"] * 2
    away = 100 + rng.normal(scale=8, size=n)
    df["actual_home"] = home
    df["actual_away"] = away
    df["actual_margin"] = home - away
    if with_market:
        df["market_spread"] = -(home - away) + rng.normal(scale=2, size=n)
    return df


_FAST_CB = {"depth": 2, "iterations": 15, "learning_rate": 0.3, "thread_count": 1}


class TestOverfitPenaltySign:
    def test_penalizes_val_worse_than_train_not_the_reverse(self):
        src = inspect.getsource(_margin_tuning_objective)
        assert "(val_mae - train_mae)" in src
        assert "(train_mae - val_mae)" not in src


class TestNoRoiClvTermsInLoss:
    def test_source_has_no_roi_or_clv_terms(self):
        src = inspect.getsource(_margin_tuning_objective)
        for banned in ("val_ats_roi", "clv_roi", "clv_weighted_roi", "edge_grid", "best_roi"):
            assert banned not in src, f"Task 055: found banned ROI/CLV term {banned!r} in tuning loss"


class TestTuningRunsWithoutBettingColumns:
    def test_tune_margin_model_runs_without_market_spread(self):
        df = _synthetic(with_market=False)
        assert "market_spread" not in df.columns
        best = tune_margin_model(
            df, n_trials=2, fast_mode=True,
        )
        assert "ridge_alpha" in best
        assert "cb_params" in best

    def test_objective_score_is_finite_without_market_spread(self):
        df = _synthetic(with_market=False)
        cols = [c for c in SAFE_FEATURE_COLS if c in df.columns and c != "elo_stack_pred"]
        X_base = df[cols].fillna(0)

        study = optuna.create_study(direction="minimize")

        def objective(trial):
            return _margin_tuning_objective(
                df, X_base, trial, None, 5, use_elo_stack=True, fast_mode=True,
            )

        study.optimize(objective, n_trials=1)
        assert np.isfinite(study.best_value)
        assert study.best_value < 9999.0
