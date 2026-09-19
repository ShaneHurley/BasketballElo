"""ATS recovery: OOF uncertainty, cover calibration, scorecard honesty, gates."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


def test_oof_score_residual_moments_and_apply():
    from pipeline.score_uncertainty import (
        apply_oof_uncertainty_to_score_pair,
        oof_score_residual_moments,
    )

    rng = np.random.default_rng(0)
    n = 120
    ph = 110 + rng.normal(0, 5, n)
    pa = 108 + rng.normal(0, 5, n)
    # Correlated residuals
    z = rng.normal(0, 1, n)
    ah = ph + 9 * z + rng.normal(0, 3, n)
    aa = pa + 9 * z + rng.normal(0, 3, n)
    m = oof_score_residual_moments(ph, pa, ah, aa, sigma_floor=5.0, min_n=40)
    assert m["source"] == "oof_residuals"
    assert m["sigma_home"] >= 5.0
    assert m["sigma_away"] >= 5.0
    assert m["residual_corr"] > 0.3

    class _Pair:
        fitted = True
        def set_walkforward_rmse(self, rmse_home=None, rmse_away=None, residual_corr=None, **_):
            self.h, self.a, self.c = rmse_home, rmse_away, residual_corr

    pair = _Pair()
    df = pd.DataFrame({
        "sf_pred_home": ph, "sf_pred_away": pa,
        "actual_home": ah, "actual_away": aa,
    })
    out = apply_oof_uncertainty_to_score_pair(pair, df, sigma_floor=5.0)
    assert out["source"] == "oof_residuals"
    assert pair.h == pytest.approx(out["sigma_home"])


def test_cover_calibrator_reduces_phantom_ev():
    from pipeline.score_targets import margin_home_cover_prob
    from pipeline.score_uncertainty import (
        CoverProbCalibrator,
        fit_cover_calibrator_from_oof,
        phantom_ev_gap,
    )

    rng = np.random.default_rng(1)
    n = 200
    pred_m = rng.normal(0, 6, n)
    line = rng.normal(0, 5, n)
    # Reality less extreme than raw Gaussian claims
    act = pred_m * 0.3 + rng.normal(0, 12, n)
    raw = np.array([
        margin_home_cover_prob(float(pm), float(ln), 10.0)
        for pm, ln in zip(pred_m, line)
    ])
    y = np.where(act + line > 0, 1.0, 0.0)
    cal = CoverProbCalibrator(method="platt").fit(raw, y, min_n=80)
    assert cal.fitted
    cal_p = cal.transform(raw)
    # Calibrated probs should be closer to base rate than raw extremes on avg.
    assert float(np.mean(np.abs(cal_p - y.mean()))) < float(np.mean(np.abs(raw - y.mean()))) + 0.05
    assert phantom_ev_gap(0.25, -0.02) is True
    assert phantom_ev_gap(0.01, 0.00) is False

    df = pd.DataFrame({
        "sf_fair_margin": pred_m,
        "decision_spread": line,
        "actual_margin": act,
    })
    cal2 = fit_cover_calibrator_from_oof(df, sigma_margin=10.0)
    assert cal2.fitted


def test_canonical_cover_uses_calibrator():
    from pipeline.canonical_scores import apply_canonical_score_pair
    from pipeline.model import MetaScorePairModel
    from pipeline.score_targets import score_safe_feature_cols
    from pipeline.score_uncertainty import CoverProbCalibrator

    rng = np.random.default_rng(2)
    n = 60
    elo = rng.normal(0, 8, n)
    df = pd.DataFrame({
        "elo_net": elo,
        "elo_margin": elo * 0.9,
        "hier_net": elo * 0.8,
        "exp_poss": 100 + rng.normal(0, 2, n),
        "pace_diff": rng.normal(0, 1, n),
        "h_rating_uncertainty": 300.0,
        "a_rating_uncertainty": 300.0,
        "matchup_margin": elo * 0.7,
        "matchup_total": 220.0,
        "h_rest": 1, "a_rest": 1,
        "game_date": pd.date_range("2023-11-01", periods=n, freq="D"),
        "actual_home": 110 + 0.3 * elo + rng.normal(0, 6, n),
        "actual_away": 108 - 0.3 * elo + rng.normal(0, 6, n),
    })
    df["actual_margin"] = df["actual_home"] - df["actual_away"]
    df["actual_total"] = df["actual_home"] + df["actual_away"]
    model = MetaScorePairModel(
        feature_cols=score_safe_feature_cols(df),
        cb_params={"depth": 3, "iterations": 25, "learning_rate": 0.1,
                   "verbose": 0, "random_seed": 0, "thread_count": 1,
                   "allow_writing_files": False},
    )
    model.fit(df)
    cal = CoverProbCalibrator()
    cal.fitted = True
    cal._model = ("platt", type("M", (), {
        "predict_proba": staticmethod(lambda X: np.column_stack([1 - 0.55 * np.ones(len(X)), 0.55 * np.ones(len(X))])),
    })())
    feat = df.iloc[-1].to_dict()
    feat["decision_spread"] = -3.5
    out = apply_canonical_score_pair({}, feat, model, cover_calibrator=cal)
    assert out["cover_prob_source"] == "pair_margin_oof_calibrated"
    assert out["home_cover_prob"] == pytest.approx(0.55, abs=0.02)
    assert "home_cover_prob_raw" in out


def test_ats_scorecard_separates_lean_from_actionable(tmp_path, monkeypatch):
    from dashboard.services import ats as ats_mod

    path = tmp_path / "results.csv"
    rng = np.random.default_rng(0)
    n = 40
    pred = rng.normal(0, 5, n)
    act = pred + rng.normal(0, 10, n)
    line = -pred * 0.5
    df = pd.DataFrame({
        "PRED_SPREAD": pred,
        "ACTUAL_MARGIN": act,
        "DECISION_SPREAD": line,
        "MARKET_SPREAD": line,
        "PRED_HOME": 110 + pred / 2,
        "PRED_AWAY": 110 - pred / 2,
        "ACTUAL_HOME": 110 + act / 2,
        "ACTUAL_AWAY": 110 - act / 2,
        "FORECAST_SOURCE": "score_pair",
        "CONF_LOWER": pred - 10,
        "CONF_UPPER": pred + 10,
        "ACTIONABLE": [0] * (n - 1) + [1],
        "ATS_WIN": [1, 0] * (n // 2),
        "ATS_EV": [0.2] * n,
        "EDGE": pred - (-line),
        "DIRECTION": ["Pass"] * (n - 1) + ["Home"],
    })
    df.to_csv(path, index=False)
    monkeypatch.setattr(ats_mod, "results_csv_path", lambda run_id: path)
    sc = ats_mod.ats_scorecard("dummy")
    assert sc["metrics"]["n_actionable"] == 1
    assert sc["metrics"]["population"] == "actionable"
    assert sc["metrics"].get("mean_ats_ev") is not None or sc["metrics"].get("ev_population") == "actionable"
    assert "market_error_corr" in sc["metrics"]
    assert "paired_score_mae" in sc["metrics"]
    assert sc["metrics"].get("clv_claim_allowed") is False


def test_direct_score_gate_requires_edge_corr_and_policy_block():
    from pipeline.ablation import (
        direct_score_ablation_configs,
        passes_direct_score_promotion_gate,
    )
    from pipeline.promotion_gates import policy_retune_allowed, tip_proxy_roi_blocked

    cfgs = direct_score_ablation_configs()
    assert "direct_score_pair_elo_prior" in cfgs
    assert "direct_score_pair_elo_agree" in cfgs

    baseline = {
        "spread_mae": 12.5, "market_spread_mae": 11.5, "brier": 0.25,
        "paired_score_mae": 9.5, "total_mae": 14.0,
        "model_minus_market_spread_mae": 1.0,
        "margin_dispersion_ratio": 0.6,
    }
    good = {
        "spread_mae": 11.2, "market_spread_mae": 11.5, "brier": 0.24,
        "paired_score_mae": 9.0, "total_mae": 13.5,
        "score_algebra_ok_pct": 1.0, "interval_coverage": 0.81,
        "market_error_corr": 0.08, "n_finite_clv": 250,
        "model_minus_market_spread_mae": 0.4,
        "margin_dispersion_ratio": 0.65,
    }
    no_edge = {**good, "market_error_corr": 0.0}
    gate_bad = passes_direct_score_promotion_gate(
        baseline, no_edge, odds_provenance={"promotion_eligible": True},
    )
    assert gate_bad["edge_evidence"] is False
    assert gate_bad["accuracy_gates_ok"] is False

    gate_ok = passes_direct_score_promotion_gate(
        baseline, good, odds_provenance={"promotion_eligible": True},
    )
    assert gate_ok["edge_evidence"] is True
    assert gate_ok["accuracy_gates_ok"] is True
    assert gate_ok["promote"] is True
    assert gate_ok["betting_edge_claim_allowed"] is True
    assert tip_proxy_roi_blocked({"quote_source": "tip_proxy"}) is True
    assert policy_retune_allowed(
        good, accuracy_ok=True, odds_provenance={"promotion_eligible": False},
    ) is False
    assert policy_retune_allowed(
        good, accuracy_ok=True, odds_provenance={"promotion_eligible": True},
    ) is True


def test_score_intervals_use_80pct_for_conf_width():
    from pipeline.score_targets import score_predictive_intervals

    iv = score_predictive_intervals(112, 108, sigma_home=10, sigma_away=10, corr=0.3)
    assert iv["CONF_WIDTH"] == pytest.approx(iv["margin_qhi_80"] - iv["margin_qlo_80"], abs=1e-6)
    assert iv["CONF_WIDTH"] < (iv["margin_qhi_95"] - iv["margin_qlo_95"])
    assert "spread_q10" in iv and "spread_q90" in iv
