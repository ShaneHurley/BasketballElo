"""Epic 10.6 / P0.2 — Kelly payout identity under extreme juice.

GitHub: P0.2 · Planned registry ID: ``bench_kelly_payout_inverted``

``spread_kelly_fraction`` must use ``b = american_to_decimal(juice) - 1`` for
every American price. Bug: ``market.py`` uses ``abs(juice)/100`` when
``juice != -110``, which *inflates* stake for favorites (e.g. −120 → b=1.2
instead of 0.833). Positive juices accidentally match the buggy branch, so
the acceptance xfail targets negative juices other than −110.
"""
from __future__ import annotations

import pytest

from pipeline.market import american_to_decimal, spread_kelly_fraction
from tests.synth.presets import JUICE_EXTREMES

# Negative juices where the bug manifests:
# wrong formula abs(j)/100 equals correct 100/abs(j) only when |j| == 100.
_BUGGY_NEGATIVE_JUICES = [
    j for j in JUICE_EXTREMES if j < 0 and j not in (-110, -100)
]


def _kelly_b_from_function(cover_prob: float, juice: float) -> float:
    """Recover the payout ratio ``b`` implied by ``spread_kelly_fraction``.

    Kelly: ``f = (p*b - (1-p)) / b`` ⇒ ``b = (1-p) / (p - f)`` when ``p != f``.
    """
    p = float(cover_prob)
    f = spread_kelly_fraction(p, juice=juice)
    if abs(p - f) < 1e-12:
        return float("nan")
    return (1.0 - p) / (p - f)


@pytest.mark.bench
@pytest.mark.xfail(
    strict=True,
    reason=(
        "P0.2: spread_kelly_fraction inverts b for juice != -110 "
        "(market.py:575 uses abs(juice)/100 instead of 100/abs(juice))"
    ),
)
@pytest.mark.parametrize("juice", _BUGGY_NEGATIVE_JUICES)
def test_p0_2_kelly_payout_matches_american_to_decimal(juice: float):
    """Acceptance: for buggy negative juices, implied Kelly ``b`` equals ``dec - 1``."""
    cover_prob = 0.55
    expected_b = float(american_to_decimal(juice) - 1.0)
    assert expected_b > 0.0
    recovered_b = _kelly_b_from_function(cover_prob, juice)
    assert recovered_b == pytest.approx(expected_b, rel=1e-9, abs=1e-9), (
        f"juice={juice}: spread_kelly_fraction implies b={recovered_b}, "
        f"but american_to_decimal-1 = {expected_b}"
    )


@pytest.mark.bench
@pytest.mark.parametrize("juice", [-110, 100, 120, 5000])
def test_p0_2_kelly_payout_correct_for_known_good_cases(juice: float):
    """−110 and positive juices are correct today (positive luckily matches abs/100)."""
    cover_prob = 0.55
    expected_b = float(american_to_decimal(juice) - 1.0)
    recovered_b = _kelly_b_from_function(cover_prob, juice)
    assert recovered_b == pytest.approx(expected_b, rel=1e-9, abs=1e-9)


@pytest.mark.bench
def test_p0_2_documents_bug_at_minus_120():
    """Regression witness (always green while bug exists): −120 is wrong today.

    Mirrors ``test_cv_past_only``'s 'prove the leak exists' pattern so nobody
    silently 'fixes' the witness without fixing the root cause.
    """
    cover_prob = 0.55
    correct_b = 100.0 / 120.0  # ≈ 0.833
    wrong_b = 120.0 / 100.0  # ≈ 1.2 — what market.py currently computes
    recovered = _kelly_b_from_function(cover_prob, -120)
    if recovered == pytest.approx(wrong_b, rel=1e-6):
        assert recovered != pytest.approx(correct_b, rel=1e-3)
        return
    assert recovered == pytest.approx(correct_b, rel=1e-6)
