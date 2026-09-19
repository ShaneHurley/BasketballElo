"""Epic 10.6 / P0.2 — Kelly payout identity under extreme juice.

GitHub: P0.2 · Registry ID: ``bench_kelly_payout_inverted`` (fixed)

``spread_kelly_fraction`` must use ``b = american_to_decimal(juice) - 1`` for
every American price.
"""
from __future__ import annotations

import pytest

from pipeline.market import american_to_decimal, spread_kelly_fraction
from tests.synth.presets import JUICE_EXTREMES


def _expected_kelly(cover_prob: float, juice: float) -> float:
    """Reference Kelly using the correct American payout ratio."""
    b = float(american_to_decimal(juice) - 1.0)
    if b <= 0:
        return 0.0
    return float(max(0.0, (cover_prob * b - (1.0 - cover_prob)) / b))


@pytest.mark.bench
@pytest.mark.parametrize("juice", list(JUICE_EXTREMES))
@pytest.mark.parametrize("cover_prob", [0.45, 0.55, 0.90])
def test_p0_2_kelly_payout_matches_american_to_decimal(juice: float, cover_prob: float):
    """Acceptance: Kelly matches american_to_decimal-based payout for all juices."""
    if juice == 0:
        pytest.skip("American odds of 0 are undefined")
    expected_b = float(american_to_decimal(juice) - 1.0)
    assert expected_b > 0.0
    got = spread_kelly_fraction(cover_prob, juice=juice)
    want = _expected_kelly(cover_prob, juice)
    assert got == pytest.approx(want, rel=1e-9, abs=1e-12), (
        f"juice={juice} p={cover_prob}: got kelly={got}, want={want} (b={expected_b})"
    )


@pytest.mark.bench
def test_p0_2_minus_120_uses_correct_payout():
    """Witness: −120 must use b=100/120 ≈ 0.833 (not the old inverted 1.2)."""
    correct_b = 100.0 / 120.0
    wrong_b = 120.0 / 100.0
    assert float(american_to_decimal(-120) - 1.0) == pytest.approx(correct_b, rel=1e-9)

    p = 0.90
    correct_kelly = max(0.0, (p * correct_b - (1.0 - p)) / correct_b)
    wrong_kelly = max(0.0, (p * wrong_b - (1.0 - p)) / wrong_b)
    got = spread_kelly_fraction(p, juice=-120)
    assert got == pytest.approx(correct_kelly, rel=1e-9)
    assert got != pytest.approx(wrong_kelly, rel=1e-3)
