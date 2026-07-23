"""Tests for ATS confidence scoring and walk-forward calibrator."""
import numpy as np
import pandas as pd
import pytest

from pipeline.bet_confidence import (
    WalkForwardBetCalibrator,
    ats_confidence_score,
    bucket_lift_map,
    empirical_bucket_ats_rates,
)
from pipeline.market import spread_cover_prob, elo_meta_agreement
from pipeline.metrics import walkforward_edge_threshold


class TestSpreadCoverProb:
    def test_home_covers_more_when_positive_edge(self):
        # model margin +8, home line -7 → edge +1 pt
        p = spread_cover_prob(8.0, -7.0, conf_width=20.0, direction="Home")
        assert p > 0.52

    def test_away_inverts(self):
        p_h = spread_cover_prob(8.0, -7.0, conf_width=20.0, direction="Home")
        p_a = spread_cover_prob(8.0, -7.0, conf_width=20.0, direction="Away")
        assert pytest.approx(p_h + p_a, abs=1e-6) == 1.0


class TestAtsConfidenceScore:
    def test_four_to_six_beats_two_to_four(self):
        lifts = bucket_lift_map({"2-4": 0.51, "4-6": 0.56})
        low = ats_confidence_score(
            {"spread_edge_pts": 3.0, "conf_width": 24, "elo_meta_agreement": 1.0,
             "spread_cover_prob": 0.54, "rating_uncertainty": 300},
            lifts,
        )
        high = ats_confidence_score(
            {"spread_edge_pts": 5.0, "conf_width": 24, "elo_meta_agreement": 1.0,
             "spread_cover_prob": 0.58, "rating_uncertainty": 300},
            lifts,
        )
        assert high > low

    def test_elo_disagreement_penalized(self):
        base = {"spread_edge_pts": 4.5, "conf_width": 20, "spread_cover_prob": 0.56,
                "rating_uncertainty": 280}
        agree = ats_confidence_score({**base, "elo_meta_agreement": 1.0})
        disagree = ats_confidence_score({**base, "elo_meta_agreement": 0.0})
        assert agree > disagree


class TestWalkForwardCalibrator:
    def _synthetic_prior(self, n=200):
        rng = np.random.default_rng(0)
        rows = []
        for i in range(n):
            edge = rng.uniform(2, 8)
            win = float(rng.random() < (0.50 + 0.04 * (edge - 2)))
            rows.append({
                "MARKET_SPREAD": -3.0,
                "EDGE": edge,
                "DIRECTION": "Home",
                "ACTUAL_MARGIN": 5.0 if win else -10.0,
                "ACTUAL_HOME": 110,
                "ACTUAL_AWAY": 105 if win else 120,
                "CONF_WIDTH": 22.0,
                "WIN_PROB": 0.55,
                "RATING_UNCERTAINTY": 320.0,
                "PRED_SPREAD": edge - 3.0,
            })
        return pd.DataFrame(rows)

    def test_fit_and_predict_monotonic_in_edge(self):
        cal = WalkForwardBetCalibrator()
        cal.fit(self._synthetic_prior(250))
        low = cal.predict("ats", {
            "spread_edge_pts": 3.0, "conf_width": 24, "win_prob": 0.52,
            "rating_uncertainty": 320, "elo_meta_agreement": 1.0,
            "spread_cover_prob": 0.53,
        })
        high = cal.predict("ats", {
            "spread_edge_pts": 6.0, "conf_width": 20, "win_prob": 0.58,
            "rating_uncertainty": 280, "elo_meta_agreement": 1.0,
            "spread_cover_prob": 0.60,
        })
        assert high["calibrated_prob"] >= low["calibrated_prob"]


class TestEdgeThreshold:
    def test_bucket_roi_prefers_higher_threshold(self):
        rows = []
        for edge, win in [(3.0, False)] * 80 + [(5.0, True)] * 60 + [(5.0, False)] * 40:
            rows.append({
                "MARKET_SPREAD": -3.0, "EDGE": edge, "DIRECTION": "Home",
                "ACTUAL_MARGIN": 5.0 if win else -10.0,
                "PRED_SPREAD": edge - 3.0, "CLOSING_SPREAD": -3.0,
            })
        prior = pd.DataFrame(rows)
        thr = walkforward_edge_threshold(
            prior, default=3.5, optimize="bucket_roi", min_bets=30,
            require_positive_ci=False,
        )
        assert thr >= 4.0


class TestEloMetaAgreement:
    def test_agreement_when_same_sign(self):
        # model (+8) and elo (+9) both favor home vs market -7
        assert elo_meta_agreement(8.0, -7.0, 9.0) == 1.0
        assert elo_meta_agreement(-2.0, -7.0, 9.0) == 0.0
