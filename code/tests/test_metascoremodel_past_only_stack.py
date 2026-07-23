"""Tasks 049/050/051/052 wiring — MetaScoreModel/MetaTotalModel/
MetaScorePairModel must fit and predict using ManualOOFStacker (no
StackingRegressor+ChronologicalPartitionCV in the production path), and
every fitted stacking/Elo component must carry ``fit_max_timestamp``.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.model import (
    MetaScoreModel,
    MetaTotalModel,
    MetaScorePairModel,
    SAFE_FEATURE_COLS,
    total_feature_cols,
)
from pipeline.oof import ManualOOFStacker


_FAST_CB = {
    "depth": 2, "iterations": 15, "learning_rate": 0.3, "verbose": 0,
    "random_seed": 42, "thread_count": 1, "allow_writing_files": False,
}


def _synthetic_games(n=90, seed=7):
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
    df["market_spread"] = -(home - away) + rng.normal(scale=2, size=n)
    return df


class TestMetaScoreModelUsesManualOOFStacker:
    def test_stack_is_manual_oof_stacker_not_sklearn(self):
        m = MetaScoreModel()
        assert isinstance(m.stack, ManualOOFStacker)
        assert isinstance(m.total_stack, ManualOOFStacker)

    def test_fit_predict_round_trip_and_lineage(self):
        df = _synthetic_games()
        m = MetaScoreModel(margin_cap=30.0, cb_params=_FAST_CB, use_quantile_heads=False)
        m.fit(df, df["actual_home"], df["actual_away"], calib_df=df.iloc[:30])
        assert m.fitted

        # Task 050/058: every fitted stacking component carries fit_max_timestamp.
        for _, est in m.stack.estimators_:
            assert hasattr(est, "fit_max_timestamp")
        assert hasattr(m.stack.final_estimator_, "fit_max_timestamp")
        assert hasattr(m.elo_ridge, "fit_max_timestamp")

        # Task 051: burn-in rows were actually dropped from meta-training,
        # not silently predicted by a future-trained base learner.
        assert m.stack.n_burn_in_dropped_ > 0
        assert m.stack.oof_rows_used_ > 0

        row = df.iloc[-1].to_dict()
        out = m.predict(row)
        assert np.isfinite(out["pred_margin"])
        assert 0.0 <= out["win_prob"] <= 1.0

    def test_elo_stack_pred_is_cross_fit_not_in_sample(self):
        """Task 052: elo_stack_pred fed into the margin stack's training
        rows must come from a past-only cross-fit, not a ridge fit on the
        full training set (which would leak the target into a feature)."""
        df = _synthetic_games(n=80)
        m = MetaScoreModel(use_elo_stack=True, cb_params=_FAST_CB, use_quantile_heads=False)
        m.fit(df, df["actual_home"], df["actual_away"])
        assert hasattr(m, "_elo_stack_covered")
        # Burn-in rows (no strictly-earlier fold) must not be marked covered.
        assert (~m._elo_stack_covered).sum() > 0
        assert m._elo_stack_covered.sum() > 0


class TestMetaTotalModelUsesManualOOFStacker:
    def test_fit_predict(self):
        df = _synthetic_games(n=80)
        df["actual_total"] = df["actual_home"] + df["actual_away"]
        m = MetaTotalModel(feature_cols=total_feature_cols(df), cb_params=_FAST_CB)
        m.fit(df)
        assert isinstance(m.stack, ManualOOFStacker)
        pred = m.predict_total(df.iloc[-1].to_dict())
        assert np.isfinite(pred)


class TestMetaScorePairModelUsesManualOOFStacker:
    def test_fit_predict(self):
        df = _synthetic_games(n=80)
        m = MetaScorePairModel(feature_cols=total_feature_cols(df), cb_params=_FAST_CB)
        m.fit(df)
        assert isinstance(m.home_stack, ManualOOFStacker)
        assert isinstance(m.away_stack, ManualOOFStacker)
        out = m.predict_scores(df.iloc[-1].to_dict())
        assert np.isfinite(out["pred_total"])
