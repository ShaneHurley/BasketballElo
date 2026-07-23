"""Tasks 037-047: bet grading, CLV, exact-price profit, caps, ROI CI, Kelly, risk."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.bet_grading import (
    exact_price_profit,
    grade_spread_bet,
    grade_total_bet,
    hit_rate,
)
from pipeline.metrics import (
    add_clv_columns,
    assert_actionable_frame,
    bootstrap_ci,
    bootstrap_roi_ci,
    compute_clv,
    compute_stake_profits,
)
from pipeline.stake_profiles import (
    apply_daily_caps,
    kelly_fraction,
    optimize_slate_stakes,
    recommend_fraction_under_risk,
    robust_fractional_kelly,
    simulate_bankroll,
)


class TestGradeSpreadAndTotal:
    def test_home_away_push_symmetry(self):
        assert grade_spread_bet(5, -5, "Home") == "push"
        assert grade_spread_bet(5, -5, "Away") == "push"
        assert grade_spread_bet(10, -5, "Home") == "win"
        assert grade_spread_bet(10, -5, "Away") == "loss"
        assert grade_spread_bet(0, -5, "Home") == "loss"
        assert grade_spread_bet(0, -5, "Away") == "win"

    def test_over_under_push_symmetry(self):
        assert grade_total_bet(220, 220, "Over") == "push"
        assert grade_total_bet(220, 220, "Under") == "push"
        assert grade_total_bet(230, 220, "Over") == "win"
        assert grade_total_bet(230, 220, "Under") == "loss"

    def test_hit_rate_excludes_pushes(self):
        assert hit_rate(["win", "push", "loss", "push", "win"]) == pytest.approx(2 / 3)


class TestPointAndPriceClv:
    def test_home_line_moves_toward_home_is_positive(self):
        # Home bet at -3, closes -5 → home got a better number → +2 point CLV
        assert compute_clv(None, -3, -5, side="Home") == pytest.approx(2.0)
        assert compute_clv(4.0, -3, -5) == pytest.approx(2.0)  # inferred Home via edge

    def test_away_line_moves_toward_home_is_negative_for_away(self):
        # Away bet at -3, closes -5 → away got a worse number → -2
        assert compute_clv(None, -3, -5, side="Away") == pytest.approx(-2.0)

    def test_add_clv_keeps_point_and_price_separate(self):
        df = pd.DataFrame({
            "DIRECTION": ["Home", "Away"],
            "MARKET_SPREAD": [-3.0, -3.0],
            "CLOSING_SPREAD": [-5.0, -5.0],
            "PRED_SPREAD": [4.0, -4.0],
            "ACTUAL_MARGIN": [8.0, -2.0],
            "DECISION_FAIR_PROB": [0.55, 0.45],
            "CLOSE_FAIR_PROB": [0.60, 0.40],
        })
        out = add_clv_columns(df)
        assert "POINT_CLV" in out.columns and "PRICE_CLV" in out.columns
        assert out.loc[0, "POINT_CLV"] == pytest.approx(2.0)
        assert out.loc[1, "POINT_CLV"] == pytest.approx(-2.0)
        assert out.loc[0, "PRICE_CLV"] == pytest.approx(0.05)


class TestExactPriceProfit:
    def test_american_favorite_and_dog(self):
        assert exact_price_profit(10, "win", -110) == pytest.approx(10 * 100 / 110)
        assert exact_price_profit(10, "win", +150) == pytest.approx(15.0)
        assert exact_price_profit(10, "loss", -110) == pytest.approx(-10.0)

    def test_decimal_odds(self):
        assert exact_price_profit(10, "win", 1.91) == pytest.approx(9.1)

    def test_push_returns_zero(self):
        assert exact_price_profit(10, "push", -110) == 0.0


class TestCapsBeforeProfit:
    def test_profit_scales_with_capped_stake(self):
        # Two same-day bets with raw stakes that exceed the moderate daily cap (0.12)
        df = pd.DataFrame({
            "DATE": ["2025-01-01", "2025-01-01"],
            "DIRECTION": ["Home", "Away"],
            "EDGE": [5.0, -5.0],
            "EDGE_THRESHOLD": [2.5, 2.5],
            "MARKET_SPREAD": [-3.0, -3.0],
            "ACTUAL_MARGIN": [10.0, -10.0],  # both win
            "COVER_PROB_CALIBRATED": [0.7, 0.7],
            "CONFIDENCE_TIER": [3, 3],
            "CONF_WIDTH": [12.0, 12.0],
            "RATING_UNCERTAINTY": [150.0, 150.0],
            "JUICE": [-110, -110],
            "STAKE_MODERATE": [0.10, 0.10],  # raw 0.20 > 0.12 cap
        })
        out = compute_stake_profits(df, profile="moderate")
        capped = out["STAKE_MODERATE"].sum()
        assert capped == pytest.approx(0.12, rel=1e-6)
        # Both win at -110 → profit = stake * (100/110)
        expected = out["STAKE_MODERATE"] * (100.0 / 110.0)
        assert out["PROFIT_MODERATE"].tolist() == pytest.approx(expected.tolist(), rel=1e-6)

    def test_apply_daily_caps_alone_reduces_exposure(self):
        df = pd.DataFrame({
            "DATE": ["2025-01-01", "2025-01-01", "2025-01-01"],
            "STAKE_MODERATE": [0.08, 0.08, 0.08],
        })
        out = apply_daily_caps(df, "STAKE_MODERATE", profile="moderate")
        assert out["STAKE_MODERATE"].sum() == pytest.approx(0.12, rel=1e-6)


class TestRoiConfidenceIntervals:
    def test_bootstrap_roi_ci_deterministic_synthetic(self):
        # 10 bets of stake 1, alternating +0.91 / -1 → ROI known
        profits = np.array([0.91, -1.0] * 10)
        stakes = np.ones(20)
        blocks = np.array([i // 2 for i in range(20)])  # 10 date blocks
        roi, lo, hi = bootstrap_roi_ci(profits, stakes, block_ids=blocks, n_boot=500, seed=0)
        assert roi == pytest.approx(profits.sum() / stakes.sum())
        assert lo <= roi <= hi

    def test_mean_profit_ci_is_not_labeled_as_roi(self):
        profits = np.array([0.91, -1.0] * 10)
        mean_est, mean_lo, mean_hi = bootstrap_ci(profits, n_boot=200, seed=0)
        roi, roi_lo, roi_hi = bootstrap_roi_ci(profits, np.ones(20), n_boot=200, seed=0)
        assert np.isfinite(mean_est) and np.isfinite(roi)
        # Distinct APIs: mean-profit CI vs ROI CI (even when unit stakes make
        # the point estimates close, the function names/docs must stay separate).
        assert bootstrap_ci.__name__ != bootstrap_roi_ci.__name__
        assert "NOT ROI" in (bootstrap_ci.__doc__ or "").upper() or "mean-profit" in (bootstrap_ci.__doc__ or "")
        assert "sum(profit)/sum(stake)" in (bootstrap_roi_ci.__doc__ or "")


class TestActionableFrame:
    def test_rejects_inconsistent_rows(self):
        df = pd.DataFrame({
            "DIRECTION": ["Pass"],
            "MARKET_SPREAD": [np.nan],
            "ACTIONABLE": [True],
            "STAKE_MODERATE": [0.05],
        })
        with pytest.raises(ValueError):
            assert_actionable_frame(df)

    def test_accepts_consistent_rows(self):
        df = pd.DataFrame({
            "DIRECTION": ["Home"],
            "MARKET_SPREAD": [-3.0],
            "ACTIONABLE": [True],
            "STAKE_MODERATE": [0.05],
        })
        out = assert_actionable_frame(df)
        assert len(out) == 1


class TestFractionalKelly:
    def test_hand_calculated_kelly(self):
        # p=0.55, decimal=1.91 → b=0.91; f*=(0.91*0.55 - 0.45)/0.91
        b = 0.91
        p = 0.55
        expected = (b * p - (1 - p)) / b
        assert kelly_fraction(p, 1.91) == pytest.approx(expected)
        assert kelly_fraction(0.4, 1.91) == 0.0  # negative edge → 0

    def test_greater_uncertainty_never_increases_stake(self):
        base = robust_fractional_kelly(0.58, 1.91, market_fair_p=0.52, uncertainty=0.0, fraction=0.25)
        worse = robust_fractional_kelly(0.58, 1.91, market_fair_p=0.52, uncertainty=0.10, fraction=0.25)
        assert worse <= base + 1e-12


class TestSlateOptimization:
    def test_exposure_caps_and_psd(self):
        raw = np.array([0.08, 0.08, 0.08])
        cov = np.array([
            [1.0, 0.5, 0.5],
            [0.5, 1.0, 0.5],
            [0.5, 0.5, 1.0],
        ])
        sized = optimize_slate_stakes(raw, cov, shrink=0.85, per_bet_cap=0.05, slate_cap=0.12)
        assert sized.max() <= 0.05 + 1e-12
        assert sized.sum() <= 0.12 + 1e-12
        eig = np.linalg.eigvalsh(cov)
        assert (eig >= -1e-10).all()


class TestBankrollSimulation:
    def test_worse_calibration_never_less_conservative(self):
        p = np.full(20, 0.55)
        dec = np.full(20, 1.91)
        good = recommend_fraction_under_risk(p, dec, miscalibration=0.0, correlation=0.0, seed=1)
        bad = recommend_fraction_under_risk(p, dec, miscalibration=0.08, correlation=0.0, seed=1)
        corr = recommend_fraction_under_risk(p, dec, miscalibration=0.0, correlation=0.5, seed=1)
        assert bad <= good
        assert corr <= good

    def test_simulate_bankroll_keys(self):
        stats = simulate_bankroll(
            np.full(10, 0.55), np.full(10, 1.91), np.full(10, 0.1), n_paths=500, seed=0,
        )
        for k in ("expected_log_growth", "prob_loss", "p05_return", "p10_return",
                  "max_drawdown", "time_under_water"):
            assert k in stats
