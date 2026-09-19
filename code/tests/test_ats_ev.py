"""ATS exact-price EV selection tests."""
from __future__ import annotations

from pipeline.ats_ev import (
    ats_ev,
    ats_decision_record,
    break_even_prob,
    canonical_home_cover_prob,
    home_cover_prob_from_margin,
    select_ats_bet_ev,
)


def test_break_even_minus_110():
    assert abs(break_even_prob(-110) - (110 / 210)) < 1e-9


def test_select_ats_prefers_positive_ev_side():
    # Strong home cover signal with -110 both sides.
    decision = select_ats_bet_ev(
        0.62,
        decision_spread=-3.5,
        spread_home_price=-110,
        spread_away_price=-110,
        min_ev=0.01,
    )
    assert decision["side"] == "Home"
    assert decision["expected_value"] > 0
    assert decision["fair_cover_prob"] == 0.62


def test_select_ats_pass_when_no_edge():
    decision = select_ats_bet_ev(
        0.52,
        decision_spread=-3.5,
        spread_home_price=-110,
        spread_away_price=-110,
        min_ev=0.05,
    )
    assert decision["side"] == "Pass"
    assert decision["pass_reason"] == "no_side_clears_ev"


def test_ats_ev_uses_real_away_price():
    # Away at +150 with P(away cover)=0.55 should have positive EV.
    ev = ats_ev(0.45, "Away", 150)  # home_cover=0.45 → away=0.55
    assert ev > 0


def test_away_lean_must_not_invert_home_cover_for_ev():
    """Regression: direction-specific P(away covers)=0.70 must NOT be fed as home_cover.

    If incorrectly passed as home_cover, select_ats would prefer Home; with the
    correct un-flip (p_home=0.30) Away must be the EV side.
    """
    # Simulated bug input: lean=Away, cover_prob_calibrated = P(away covers) = 0.70
    # Do not pass pred_margin here — we are testing the un-flip path used when
    # only a direction-specific cover is available.
    direction_specific = 0.70
    p_home = canonical_home_cover_prob(
        direction_specific_cover=direction_specific,
        lean="Away",
    )
    assert abs(p_home - 0.30) < 1e-9
    rec = ats_decision_record(
        p_home,
        {
            "decision_spread": 4.0,
            "spread_home_price": -110,
            "spread_away_price": -110,
        },
        min_ev=0.0,
    )
    # p_home=0.30 → Away cover 0.70 clears -110 break-even (~0.524)
    assert rec["side"] == "Away"
    assert abs(rec["fair_cover_prob"] - 0.70) < 1e-9
    # And EV(Home) using p_home must be negative; EV(Away) positive.
    assert ats_ev(p_home, "Home", -110) < 0
    assert ats_ev(p_home, "Away", -110) > 0


def test_buggy_pass_of_direction_cover_as_home_would_pick_wrong_side():
    """Document the polarity bug: feeding P(away)=0.70 as home_cover selects Home."""
    wrong = select_ats_bet_ev(
        0.70,  # wrongly treated as home_cover
        decision_spread=4.0,
        spread_home_price=-110,
        spread_away_price=-110,
        min_ev=0.0,
    )
    assert wrong["side"] == "Home"  # inverted decision — this is the bug pattern



def test_home_cover_prob_from_margin_monotonic():
    low = home_cover_prob_from_margin(-8.0, -3.0, sigma=12.0)
    high = home_cover_prob_from_margin(8.0, -3.0, sigma=12.0)
    assert high > low
    assert 0.01 <= low <= 0.99
