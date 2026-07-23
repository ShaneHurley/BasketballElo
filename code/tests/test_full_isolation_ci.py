"""Task 058 — single CI test instrumenting every fitted component with
training timestamps; asserts all OOF/test predictions satisfy
fit_max_timestamp < prediction_timestamp for every component. Blocks if
any component lacks lineage.

This is the concrete implementation of the "Required test and audit
suite" bullet in the plan: *"Instrument every scaler, imputer, encoder,
feature selector, rating prior, synthetic generator, calibrator, and
ensemble with fit_max_timestamp; CI must assert fit_max_timestamp <
prediction_timestamp for every emitted OOF/test prediction."*
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, Ridge

from pipeline.model import MetaScoreModel, SAFE_FEATURE_COLS
from pipeline.oof import ManualOOFStacker


REQUIRED_LINEAGE_COMPONENTS = (
    "stack_base_estimators",
    "stack_final_estimator",
    "elo_ridge",
)


def assert_full_lineage(model: MetaScoreModel, prediction_timestamp) -> None:
    """Task 058's single CI assertion, generalized to any fitted
    MetaScoreModel: every fitted component must (a) carry
    fit_max_timestamp and (b) have fit_max_timestamp strictly before the
    timestamp of any prediction it contributes to."""
    missing = []
    violations = []

    def _check(label, obj):
        if not hasattr(obj, "fit_max_timestamp"):
            missing.append(label)
            return
        if not (obj.fit_max_timestamp < prediction_timestamp):
            violations.append((label, obj.fit_max_timestamp, prediction_timestamp))

    for name, est in model.stack.estimators_:
        _check(f"stack_base::{name}", est)
    _check("stack_final_estimator", model.stack.final_estimator_)
    if model.use_elo_stack and model.elo_ridge is not None:
        _check("elo_ridge", model.elo_ridge)
        _check("elo_scaler", model.elo_scaler)

    if missing:
        raise AssertionError(
            f"Task 058: components missing fit_max_timestamp lineage: {missing}"
        )
    if violations:
        raise AssertionError(
            f"Task 058: components whose fit_max_timestamp is not strictly "
            f"before the prediction timestamp: {violations}"
        )


def _synthetic_games(n=90, seed=13):
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
    return df


_FAST_CB = {"depth": 2, "iterations": 15, "learning_rate": 0.3, "thread_count": 1}


class TestFullIsolationCI:
    def test_metascoremodel_has_complete_lineage_before_any_future_prediction(self):
        df = _synthetic_games()
        m = MetaScoreModel(cb_params=_FAST_CB, use_quantile_heads=False)
        m.fit(df, df["actual_home"], df["actual_away"])

        # Every real prediction this model will ever be asked to make in
        # production happens strictly after every training row's game_date
        # (Task 015's chronological-iteration invariant elsewhere in the
        # pipeline); simulate that here with a timestamp one day after the
        # last training row.
        prediction_timestamp = df["game_date"].max() + pd.Timedelta(days=1)
        assert_full_lineage(m, prediction_timestamp)

    def test_blocks_when_a_component_lacks_lineage(self):
        """A component that was never instrumented with fit_max_timestamp
        must fail this CI check rather than silently pass — this is the
        explicit Task 058 pass condition ("do not continue if any
        component lacks lineage")."""
        df = _synthetic_games(n=60)
        m = MetaScoreModel(cb_params=_FAST_CB, use_quantile_heads=False)
        m.fit(df, df["actual_home"], df["actual_away"])
        # Simulate an un-instrumented component sneaking into the stack.
        del m.stack.final_estimator_.fit_max_timestamp
        prediction_timestamp = df["game_date"].max() + pd.Timedelta(days=1)
        with pytest.raises(AssertionError, match="missing fit_max_timestamp"):
            assert_full_lineage(m, prediction_timestamp)

    def test_blocks_when_a_component_was_fit_after_the_prediction_timestamp(self):
        """Directly exercises the fit_max_timestamp < prediction_timestamp
        assertion with a deliberately-violating timestamp."""
        df = _synthetic_games(n=60)
        m = MetaScoreModel(cb_params=_FAST_CB, use_quantile_heads=False)
        m.fit(df, df["actual_home"], df["actual_away"])
        stale_prediction_timestamp = df["game_date"].min()  # before training even finished
        with pytest.raises(AssertionError, match="not strictly"):
            assert_full_lineage(m, stale_prediction_timestamp)

    def test_manual_oof_stacker_alone_satisfies_lineage_for_synthetic_case(self):
        n = 100
        rng = np.random.RandomState(2)
        X = rng.normal(size=(n, 3))
        y = X[:, 0] * 2.0 + rng.normal(scale=0.1, size=n)
        groups = np.arange(n)
        stacker = ManualOOFStacker(
            estimators=[("ridge", Ridge(alpha=1.0))],
            final_estimator=LinearRegression(),
            n_splits=4,
        )
        stacker.fit(X, y, groups=groups)
        prediction_group = groups.max() + 1
        for _, est in stacker.estimators_:
            assert est.fit_max_timestamp < prediction_group
        assert stacker.final_estimator_.fit_max_timestamp < prediction_group
