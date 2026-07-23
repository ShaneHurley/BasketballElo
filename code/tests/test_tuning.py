"""Unit tests for tuning bounds, season-fold logic, and adaptive shrinking."""
import numpy as np
import pandas as pd
import pytest

from pipeline.model import AdaptiveParameterBounds, meta_params_for_bounds, _outer_game_cv_splits
from pipeline.tuning import (
    ELO_PARAM_BOUNDS,
    HIER_PARAM_BOUNDS,
    _blend_mae,
    _fold_mae_from_predictions,
    _group_records_by_season,
    _params_at_bounds,
    _prepare_stint_records,
    _season_walkforward_mae,
)


class TestAdaptiveParameterBounds:
    def test_no_shrink_before_first_season(self):
        bounds = AdaptiveParameterBounds()
        lo, hi = bounds.suggest_bounds_float("k_off", 0.2, 1.0)
        assert lo == 0.2 and hi == 1.0

    def test_shrink_after_record_best(self):
        bounds = AdaptiveParameterBounds()
        bounds.record_best(2023, {"k_off": 0.9})
        lo, hi = bounds.suggest_bounds_float("k_off", 0.2, 1.0)
        assert lo < 0.9 < hi
        assert hi - lo < 1.0 - 0.2

    def test_plateau_after_five_seasons(self):
        bounds = AdaptiveParameterBounds()
        for yr in range(2020, 2026):
            bounds.record_best(yr, {"k_off": 0.5})
        f5 = bounds._shrink_factor()
        bounds.record_best(2026, {"k_off": 0.5})
        f6 = bounds._shrink_factor()
        assert f5 == f6 == AdaptiveParameterBounds.MIN_RANGE_FRACTION

    def test_param_alias_k_off(self):
        bounds = AdaptiveParameterBounds()
        bounds.record_best(2023, {"K_OFF": 0.85})
        lo, hi = bounds.suggest_bounds_float("k_off", 0.2, 1.0)
        assert lo < 0.85 < hi

    def test_log_scale_shrink(self):
        bounds = AdaptiveParameterBounds()
        bounds.record_best(2023, {"ridge_alpha": 10.0})
        lo, hi = bounds.suggest_bounds_float("ridge_alpha", 0.1, 100.0, log=True)
        assert lo < 10.0 < hi
        assert lo >= 0.1 and hi <= 100.0

    def test_none_param_does_not_crash_lookup(self):
        bounds = AdaptiveParameterBounds()
        bounds.record_best(2023, {"cb_l2": None, "ridge_alpha": 5.0})
        lo, hi = bounds.suggest_bounds_float("cb_l2", 1.0, 10.0, log=True)
        assert lo == 1.0 and hi == 10.0

    def test_merge_same_season_preserves_prior_params(self):
        bounds = AdaptiveParameterBounds()
        bounds.record_best(2023, {"cb_l2": 2.0, "ridge_alpha": 10.0})
        bounds.record_best(2023, {"cb_depth": 5})
        assert bounds._running_best()["cb_l2"] == 2.0
        assert bounds._running_best()["cb_depth"] == 5
        assert bounds.seasons_seen == 1

    def test_multiple_records_same_season_count_once(self):
        bounds = AdaptiveParameterBounds()
        bounds.record_best(2023, {"k_off": 0.5})
        bounds.record_best(2023, {"k_def": 0.8})
        bounds.record_best(2024, {"k_off": 0.6})
        assert bounds.seasons_seen == 2


class TestBoundEdgeDetection:
    def test_detects_lower_edge(self):
        at = _params_at_bounds({"usage_floor": 0.201}, {"usage_floor": (0.20, 0.32)})
        assert "usage_floor" in at

    def test_stable_param_not_flagged(self):
        at = _params_at_bounds({"k_off": 0.5}, {"k_off": (0.2, 1.0)})
        assert "k_off" not in at


class TestSeasonWalkforward:
    def _make_records(self, seasons):
        recs = []
        for s in seasons:
            for _ in range(5):
                recs.append({
                    "season": s,
                    "hp": ["1"], "ap": ["2"],
                    "poss": 10.0,
                    "home_xpts": 11.0, "away_xpts": 10.0,
                    "home_pts": 11.0, "away_pts": 10.0,
                    "home_usage": {}, "away_usage": {},
                    "period": 1,
                    "start_A": 0.0, "start_B": 0.0,
                    "end_A": 11.0, "end_B": 10.0,
                })
        return recs

    def test_single_season_too_few_records_returns_invalid(self):
        by_season = _group_records_by_season(self._make_records([2022]))
        score = _season_walkforward_mae([2022], by_season, lambda t, v: 1.0)
        assert score == 9999.0

    def test_single_season_holdout_when_enough_records(self):
        recs = []
        for _ in range(12):
            recs.extend(self._make_records([2022]))
        by_season = _group_records_by_season(recs)
        calls = []

        def fold_fn(train, val):
            calls.append((len(train), len(val)))
            return 3.0

        score = _season_walkforward_mae([2022], by_season, fold_fn)
        assert len(calls) == 1
        assert calls[0][0] == 42  # 70% of 60
        assert calls[0][1] == 18
        assert score == pytest.approx(3.0)

    def test_two_seasons_runs_one_fold(self):
        by_season = _group_records_by_season(self._make_records([2022, 2023]))
        calls = []

        def fold_fn(train, val):
            calls.append((len(train), len(val)))
            return 2.5

        score = _season_walkforward_mae([2022, 2023], by_season, fold_fn)
        assert len(calls) == 1
        assert score == pytest.approx(2.5)


class TestMetaParamsForBounds:
    def test_flattens_cb_params(self):
        flat = meta_params_for_bounds({
            "ridge_alpha": 7.0,
            "cb_params": {"depth": 2, "iterations": 500, "learning_rate": 0.01, "l2_leaf_reg": 5.0},
        })
        assert flat["cb_depth"] == 2
        assert flat["cb_iter"] == 500
        assert flat["cb_lr"] == 0.01
        assert flat["cb_l2"] == 5.0

    def test_total_head_partial_cb_does_not_set_none(self):
        flat = meta_params_for_bounds({
            "ridge_alpha": 12.0,
            "cb_params": {"depth": 5, "iterations": 300},
        })
        assert "cb_l2" not in flat
        assert flat["cb_depth"] == 5


class TestOuterGameCV:
    def test_splits_do_not_share_game_dates(self):
        df = pd.DataFrame({
            "game_date": pd.date_range("2023-01-01", periods=90, freq="D").repeat(2),
            "GAME_ID": np.repeat(np.arange(90), 2),
            "x": np.arange(180),
        })
        splits = list(_outer_game_cv_splits(df, n_splits=3, embargo=5))
        assert len(splits) >= 1
        for train_idx, val_idx in splits:
            train_dates = set(df.iloc[train_idx]["game_date"])
            val_dates = set(df.iloc[val_idx]["game_date"])
            assert train_dates.isdisjoint(val_dates)


class TestBlendMae:
    def test_weighted_blend(self):
        assert _blend_mae(10.0, 20.0) == pytest.approx(0.3 * 10 + 0.7 * 20)

    def test_fold_mae_invalid_on_empty(self):
        assert _fold_mae_from_predictions([], [], [], []) == 9999.0


class TestParamBoundsWidened:
    def test_elo_bounds_include_widened_ranges(self):
        assert ELO_PARAM_BOUNDS["usage_floor"][1] == 0.32
        assert ELO_PARAM_BOUNDS["assist_split"][0] == 0.60
        assert ELO_PARAM_BOUNDS["k_mult_half_life"][1] == 35.0
        assert ELO_PARAM_BOUNDS["garbage_time_weight"][0] == 0.0
        assert ELO_PARAM_BOUNDS["elo_scaling"][0] == 200

    def test_hier_bounds_widened(self):
        assert HIER_PARAM_BOUNDS["k_def"][0] == 0.01
        assert HIER_PARAM_BOUNDS["league_rtg"][0] == 109.5
