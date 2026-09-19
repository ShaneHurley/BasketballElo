"""Epic 10.6 / P0.3 — Moneyline de-vig must not be a no-op on single-sided feeds.

GitHub: P0.3 · Registry ID: ``bench_ml_devig_no_op``

Before the fix, ``fair_probs_from_ml_pair`` fell back to negating the home
American price to synthesize an away price whenever ``market_ml_away`` was
missing. Because ``implied_probability(x) + implied_probability(-x) == 1``
for any American price ``x``, that negated pair already summed to 1, so
``devig_two_way`` on it was a mathematical no-op: the function's "fair"
output was bit-for-bit identical to the *raw vigged* implied probabilities
from `implied_probability`, silently mislabeled as de-vigged.

The fix keeps the negation fallback (still the best information available
without a real two-sided quote) but shrinks the result toward 0.5/0.5 by a
documented, conservative ``ASSUMED_SINGLE_SIDED_VIG_SHRINK`` factor so the
single-sided estimate is never presented as genuinely vig-free, and adds
``is_single_sided_ml_quote`` so callers can detect the fallback explicitly.
"""
from __future__ import annotations

import pytest

from pipeline.market import (
    ASSUMED_SINGLE_SIDED_VIG_SHRINK,
    devig_two_way,
    fair_probs_from_ml_pair,
    implied_probability,
    is_single_sided_ml_quote,
)


@pytest.mark.bench
def test_p0_3_two_sided_quotes_sum_to_one():
    """Real two-sided quotes still de-vig correctly (unaffected by the fix)."""
    p_home, p_away = fair_probs_from_ml_pair(-200, 160)
    assert p_home + p_away == pytest.approx(1.0, abs=1e-9)
    # Real away price must differ from a negated-home price.
    p_neg_away = implied_probability(200)
    assert abs(p_away - p_neg_away) > 1e-6


@pytest.mark.bench
@pytest.mark.parametrize("ml_home", [-110, -150, -200, -300, 120, 180])
def test_p0_3_single_sided_is_not_raw_negated_implied(ml_home):
    """Acceptance: single-sided output must differ from the raw negation-devig no-op.

    Witnesses the P0.3 bug directly: the *old* behavior returned exactly
    ``devig_two_way(implied_probability(ml_home), implied_probability(-ml_home))``,
    which — because that pair already summed to 1 — equalled the raw
    ``implied_probability`` values verbatim. The fixed function must not.
    """
    raw_home = implied_probability(ml_home)
    raw_away = implied_probability(-ml_home)
    old_buggy_home, old_buggy_away = devig_two_way(raw_home, raw_away)
    # Sanity: the old computation really was a no-op (documents the bug).
    assert old_buggy_home == pytest.approx(raw_home, abs=1e-9)
    assert old_buggy_away == pytest.approx(raw_away, abs=1e-9)

    fixed_home, fixed_away = fair_probs_from_ml_pair(ml_home)
    assert fixed_home != pytest.approx(old_buggy_home, abs=1e-9)
    assert fixed_away != pytest.approx(old_buggy_away, abs=1e-9)
    # Still must sum to 1 (a valid probability pair), just not vig-free.
    assert fixed_home + fixed_away == pytest.approx(1.0, abs=1e-9)


@pytest.mark.bench
def test_p0_3_single_sided_shrinks_toward_half_by_documented_factor():
    """The single-sided fallback matches the documented shrink factor exactly."""
    ml_home = -200
    raw_home = implied_probability(ml_home)
    raw_away = implied_probability(-ml_home)
    no_op_home, no_op_away = devig_two_way(raw_home, raw_away)

    expected_home = (1.0 - ASSUMED_SINGLE_SIDED_VIG_SHRINK) * no_op_home + \
        ASSUMED_SINGLE_SIDED_VIG_SHRINK * 0.5
    expected_away = (1.0 - ASSUMED_SINGLE_SIDED_VIG_SHRINK) * no_op_away + \
        ASSUMED_SINGLE_SIDED_VIG_SHRINK * 0.5

    got_home, got_away = fair_probs_from_ml_pair(ml_home)
    assert got_home == pytest.approx(expected_home, abs=1e-9)
    assert got_away == pytest.approx(expected_away, abs=1e-9)
    # Shrunk result must be strictly closer to 0.5 than the no-op raw split.
    assert abs(got_home - 0.5) < abs(no_op_home - 0.5)


@pytest.mark.bench
def test_p0_3_is_single_sided_ml_quote_flag():
    assert is_single_sided_ml_quote(None) is True
    assert is_single_sided_ml_quote(float("nan")) is True
    assert is_single_sided_ml_quote(160) is False
