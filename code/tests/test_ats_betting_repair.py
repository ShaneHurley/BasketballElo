"""Regression tests for ATS betting repair plan."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.ats_ev import cover_prob_for_side
from pipeline.model import LINE_DERIVED_FEATURE_COLS, MetaScoreModel, SAFE_FEATURE_COLS
from pipeline.t60_coverage import nba_season_start_year_series, season_decision_coverage, t60_betting_data_gate


def test_line_derived_features_stripped_under_decision_residual():
    feats = [c for c in SAFE_FEATURE_COLS if c not in ("elo_stack_pred",)][:40]
    # Ensure banned cols are in the requested feature list.
    for c in ("elo_vs_market", "elo_edge_pts", "fair_spread_vigfree"):
        if c not in feats:
            feats.append(c)
    m = MetaScoreModel(feature_cols=feats, train_target="decision_residual", use_elo_stack=False)
    m._active_train_mode = "decision_residual"
    active = m._active_feature_list("decision_residual")
    for c in LINE_DERIVED_FEATURE_COLS:
        assert c not in active


def test_absolute_target_keeps_line_features():
    feats = list(SAFE_FEATURE_COLS[:30]) + ["elo_vs_market", "elo_edge_pts"]
    m = MetaScoreModel(feature_cols=feats, train_target="absolute", use_elo_stack=False)
    m._active_train_mode = "absolute"
    active = m._active_feature_list("absolute")
    assert "elo_vs_market" in active


def test_nba_season_start_from_game_id_not_calendar_year():
    # Oct 2024 and Jan 2025 are the same NBA season (start 2024).
    df = pd.DataFrame({
        "GAME_ID": ["0022400001", "0022400500", "0022500001"],
        "game_date": ["2024-10-22", "2025-01-15", "2025-10-21"],
        "decision_spread": [-3.5, -2.0, 1.5],
        "market_spread": [-3.5, -2.0, 1.5],
    })
    starts = nba_season_start_year_series(df)
    assert list(starts.astype(int)) == [2024, 2024, 2025]
    cov = season_decision_coverage(df)
    # Two NBA seasons, not three calendar years.
    assert len(cov) == 2
    assert set(cov["season"].astype(int)) == {2024, 2025}


def test_t60_gate_counts_nba_seasons():
    rows = []
    for season_start, n in ((2022, 100), (2023, 100), (2024, 100)):
        yy = season_start % 100
        for i in range(n):
            rows.append({
                "GAME_ID": f"002{yy:02d}{i:05d}",
                "game_date": f"{season_start}-11-01" if i < n // 2 else f"{season_start + 1}-02-01",
                "decision_spread": -3.0,
                "market_spread": -3.0,
                "closing_spread": -3.0,
            })
    df = pd.DataFrame(rows)
    gate = t60_betting_data_gate(df, min_seasons=3, min_coverage=0.5)
    assert gate["n_adequate_seasons"] >= 3
    assert gate["allow_betting_heads"] is True


def test_ats_classifier_returns_home_cover_only():
    from pipeline.ats_classifier import ATSClassifier
    # Unfitted → constant home prior; Away must NOT flip.
    clf = ATSClassifier()
    p_home = clf.predict_cover_prob({"pred_margin": 2.0}, "Home", -3.5)
    p_away_call = clf.predict_cover_prob({"pred_margin": 2.0}, "Away", -3.5)
    assert abs(p_home - p_away_call) < 1e-9
    # Side conversion is explicit and once.
    assert abs(cover_prob_for_side(p_home, "Away") - (1.0 - p_home)) < 1e-9


def test_bet_confidence_ats_outcome_prefers_decision_spread():
    from pipeline.bet_confidence import WalkForwardBetCalibrator
    row = {
        "DECISION_SPREAD": -5.0,
        "MARKET_SPREAD": -2.0,  # different closing/legacy
        "ACTUAL_MARGIN": 3.0,
        "DIRECTION": "Home",
        "EDGE": 1.0,
    }
    # Home covers vs decision: 3 + (-5) = -2 → loss
    assert WalkForwardBetCalibrator._ats_outcome(row) == 0.0
    row2 = dict(row)
    row2["ACTUAL_MARGIN"] = 6.0  # 6 - 5 > 0 → win vs decision
    assert WalkForwardBetCalibrator._ats_outcome(row2) == 1.0


def test_ats_scorecard_separates_populations():
    from dashboard.services import ats as ats_svc
    import tempfile
    from pathlib import Path

    df = pd.DataFrame({
        "GAME_ID": [1, 2, 3, 4],
        "ACTUAL_MARGIN": [5, -5, 2, -2],
        "DECISION_SPREAD": [-3, 3, -1, 1],
        "MARKET_SPREAD": [-3, 3, -1, 1],
        "EDGE": [4, 4, 0.5, 0.5],
        "EDGE_LEAN": ["Home", "Away", "Home", "Away"],
        "DIRECTION": ["Home", "Pass", "Pass", "Pass"],
        "ACTIONABLE": [1, 0, 0, 0],
        "PRED_SPREAD": [1.0, -1.0, 0.2, -0.2],
        "CALIBRATED_COVER_PROB": [0.6, 0.55, 0.52, 0.51],
        "CONF_LOWER": [-10, -10, -10, -10],
        "CONF_UPPER": [10, 10, 10, 10],
    })
    with tempfile.TemporaryDirectory() as td:
        run_id = "test_run_ats_pop"
        run_dir = Path(td) / run_id
        bet = run_dir / "betting"
        bet.mkdir(parents=True)
        csv_path = bet / "backtest_results.csv"
        df.to_csv(csv_path, index=False)
        orig = ats_svc.results_csv_path
        ats_svc.results_csv_path = lambda rid: csv_path if rid == run_id else None
        try:
            sc = ats_svc.ats_scorecard(run_id)
        finally:
            ats_svc.results_csv_path = orig
    assert sc["metrics"]["n_actionable"] == 1
    assert sc["metrics"]["population"] == "actionable"
    assert sc["metrics"]["n_lean_graded"] >= 1


def test_residual_reconstruction_identity():
    """r̂ − decision_spread must recover margin when r̂ = margin + decision_spread."""
    from pipeline.market_targets import margin_decision_residual, residual_to_margin
    margin = np.array([5.0, -3.0, 10.0])
    decision = np.array([-4.0, 2.5, -7.0])
    resid = margin_decision_residual(margin, decision)
    recovered = residual_to_margin(resid, decision)
    assert np.allclose(recovered, margin)


def test_promotion_dispersion_and_coverage_gates():
    from pipeline.promotion_gates import interval_coverage_credible, margin_dispersion_ok
    assert margin_dispersion_ok({"margin_dispersion_ratio": 0.6})
    assert not margin_dispersion_ok({"margin_dispersion_ratio": 0.17})
    assert interval_coverage_credible({"interval_coverage": 0.82}, target=0.80)
    assert not interval_coverage_credible({"interval_coverage": 0.40}, target=0.80)


def test_chart_cover_calibration_prefers_actionable():
    from dashboard.services.charts import chart_cover_calibration
    from dashboard.jobs.util import ensure_ats_win
    df = pd.DataFrame({
        "ACTUAL_MARGIN": np.random.default_rng(0).normal(0, 12, 60),
        "DECISION_SPREAD": -3.0,
        "MARKET_SPREAD": -3.0,
        "EDGE_LEAN": ["Home"] * 60,
        "ACTIONABLE": [1] * 30 + [0] * 30,
        "CALIBRATED_COVER_PROB": np.linspace(0.4, 0.7, 60),
    })
    df = ensure_ats_win(df)
    # ATS_WIN from ensure
    chart = chart_cover_calibration(df)
    assert chart.get("meta", {}).get("population") == "actionable"


def test_bet_analysis_defers_confidence_calibrator():
    """bet_analysis must not invoke WalkForwardBetCalibrator when deferred."""
    from pipeline.market import bet_analysis

    class _Spy:
        def __init__(self):
            self.n = 0

        def predict(self, *a, **k):
            self.n += 1
            return {
                "confidence_score": 50,
                "stars": "**",
                "calibrated_prob": 0.55,
                "confidence_tier": 2,
                "confidence_score_raw": 50.0,
            }

    spy = _Spy()
    bet_analysis(
        model_spread=2.0,
        market_spread=-3.5,
        min_edge_pts=1.0,
        confidence_calibrator=spy,
        defer_confidence_calib=True,
    )
    assert spy.n == 0


def test_home_side_cover_invariants():
    from pipeline.ats_ev import cover_prob_for_side
    from pipeline.calibration_policy import select_cover_prob

    home = 0.62
    assert abs(cover_prob_for_side(home, "Home") - home) < 1e-9
    assert abs(cover_prob_for_side(home, "Away") - (1.0 - home)) < 1e-9
    # beta_ats fails closed to raw when classifier missing
    p = select_cover_prob(
        "beta_ats",
        raw_cover=0.58,
        calibrated_cover=0.70,
        ats_classifier_prob=None,
    )
    assert abs(p - 0.58) < 1e-9


def test_margin_wp_zero_winner_sign_conflict():
    """ATS confidence win_prob must share sign with predicted margin."""
    from pipeline.bet_confidence import lean_side_win_prob

    for margin in (8.0, -6.0, 0.5, -0.5):
        margin_wp = float(np.clip(1.0 / (1.0 + np.exp(-float(margin) / 12.0)), 0.01, 0.99))
        if margin > 0:
            assert margin_wp > 0.5
        elif margin < 0:
            assert margin_wp < 0.5
        lean = "Home" if margin > 0 else "Away"
        side_wp = lean_side_win_prob(margin_wp, lean)
        assert side_wp >= 0.5


def test_ats_repair_ablation_configs_present():
    from pipeline.ablation import ats_repair_ablation_configs, passes_ats_repair_promotion_gate

    cfgs = ats_repair_ablation_configs()
    assert "decision_residual_no_line_feats" in cfgs
    assert "absolute_margin_target" in cfgs
    assert "ats_repair_combined" in cfgs
    gate = passes_ats_repair_promotion_gate(
        {"spread_mae": 12.0, "market_spread_mae": 11.0, "brier": 0.25},
        {
            "spread_mae": 11.2,
            "market_spread_mae": 11.0,
            "brier": 0.24,
            "margin_dispersion_ratio": 0.6,
            "interval_coverage": 0.82,
            "n_finite_clv": 3,
        },
        odds_provenance={"promotion_eligible": False},
    )
    assert gate["dispersion_ok"] is True
    assert gate["interval_coverage_ok"] is True
    assert gate["clv_roi_allowed"] is False


def test_tip_proxy_blocks_clv_promotion():
    from pipeline.promotion_gates import clv_promotion_allowed

    assert clv_promotion_allowed({"promotion_eligible": False}, 500) is False
    assert clv_promotion_allowed({"promotion_eligible": True}, 250) is True
