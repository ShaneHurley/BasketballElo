"""Promotion gate unit tests against the live ``pipeline.promotion_gates`` API.

The deleted ``ats_promote`` / ``ml_promote`` / ``total_promote`` /
``evaluate_negative_controls`` wrappers are not restored — callers must use
``beats_market_and_baseline``, ``multi_season_gate``, ``clv_promotion_allowed``,
and ``policy_retune_allowed``.
"""
from __future__ import annotations

from pipeline.promotion_gates import (
    beats_market_and_baseline,
    clv_promotion_allowed,
    multi_season_gate,
    policy_retune_allowed,
    tip_proxy_roi_blocked,
)


def test_beats_market_and_baseline_requires_mae_improvement():
    baseline = {"spread_mae": 11.5, "market_spread_mae": 11.4, "brier": 0.25}
    worse = {"spread_mae": 12.0, "market_spread_mae": 11.4, "brier": 0.24}
    better = {"spread_mae": 11.2, "market_spread_mae": 11.4, "brier": 0.24}
    assert beats_market_and_baseline(worse, baseline) is False
    assert beats_market_and_baseline(better, baseline) is True


def test_multi_season_gate_rejects_insufficient_blind_wins():
    candidate = [{"spread_mae": 11.0}, {"spread_mae": 12.0}]
    baseline = [{"spread_mae": 11.4}, {"spread_mae": 11.5}]
    res = multi_season_gate(candidate, baseline, min_wins=3)
    assert res["passed"] is False
    assert res["n_seasons"] == 2
    assert res["wins"] == 1


def test_multi_season_gate_passes_three_strict_wins():
    candidate = [{"spread_mae": 11.0}, {"spread_mae": 11.1}, {"spread_mae": 11.2}]
    baseline = [{"spread_mae": 11.4}, {"spread_mae": 11.5}, {"spread_mae": 11.6}]
    res = multi_season_gate(candidate, baseline, min_wins=3)
    assert res["passed"] is True
    assert res["wins"] == 3
    assert res["collapses"] == 0


def test_clv_promotion_blocked_without_eligible_provenance():
    assert clv_promotion_allowed(None, 500) is False
    assert clv_promotion_allowed({"promotion_eligible": False}, 500) is False
    assert clv_promotion_allowed({"promotion_eligible": True}, 50) is False
    assert clv_promotion_allowed({"promotion_eligible": True}, 200) is True


def test_policy_retune_and_tip_proxy_roi_block():
    honest = {
        "interval_coverage": 0.80,
        "market_error_corr": 0.05,
        "margin_dispersion_ratio": 0.70,
    }
    assert policy_retune_allowed(honest, odds_provenance={"promotion_eligible": True}) is True
    assert policy_retune_allowed(honest, odds_provenance={"promotion_eligible": False}) is False
    assert tip_proxy_roi_blocked({"quote_source": "tip_proxy"}) is True
    assert tip_proxy_roi_blocked({"promotion_eligible": True, "quote_source": "pinnacle"}) is False
