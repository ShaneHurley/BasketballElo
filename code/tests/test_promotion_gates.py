"""Promotion gate unit tests."""
from __future__ import annotations

from pipeline.promotion_gates import ats_promote, evaluate_negative_controls, ml_promote, total_promote


def test_ats_promote_rejects_insufficient_seasons():
    res = ats_promote(
        seasons=[2024, 2025],
        n_bets=200,
        brier_model=0.24,
        brier_market=0.25,
        point_clv=[0.1] * 50,
        roi_by_bet=[0.02] * 50,
    )
    assert res.promote is False
    assert res.status == "research_only"
    assert any("blind_seasons" in r for r in res.reasons)


def test_ats_promote_passes_when_criteria_met():
    res = ats_promote(
        seasons=[2023, 2024, 2025],
        n_bets=200,
        brier_model=0.23,
        brier_market=0.25,
        point_clv=[0.15] * 80,
        roi_by_bet=[0.03] * 80,
        min_bets=100,
    )
    assert res.promote is True
    assert res.status == "promote"


def test_ml_and_total_gates():
    ml = ml_promote(
        seasons=[1, 2, 3],
        brier_model=0.20,
        brier_novig_market=0.22,
        logloss_model=0.55,
        logloss_novig_market=0.58,
        price_clv=[0.01] * 20,
    )
    assert ml.promote is True
    tot = total_promote(
        seasons=[1, 2],
        mae_model=14.0,
        mae_market=15.0,
    )
    assert tot.promote is False


def test_negative_controls():
    ok = evaluate_negative_controls({
        "shuffled_outcomes": {"passed": True},
        "future_line_injection": {"passed": True},
    })
    assert ok.promote is True
    bad = evaluate_negative_controls({
        "shuffled_outcomes": {"passed": False, "detail": "still profitable"},
    })
    assert bad.promote is False
    assert bad.status == "reject"
