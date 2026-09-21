"""Epic 10.2.3 — synthetic formula test bench: market / devig invariants.

Four invariants of ``pipeline.market`` / ``pipeline.devig``:

1. **P0.3 — de-vig sums to 1.** ``devig_two_way`` output sums to 1.0 ± 1e-9
   for random and extreme-juice two-sided ML pairs. Single-sided feeds (only
   the home moneyline known) must be *flagged* (``is_single_sided_ml_quote``)
   and *shrunk toward 0.5* — never returned as the raw vigged negation
   (``implied_probability(x) + implied_probability(-x) == 1`` makes a plain
   de-vig of the negated pair a mathematical no-op, silently mislabeling a
   vigged number as vig-free; the P0.3 fix shrinks by
   ``ASSUMED_SINGLE_SIDED_VIG_SHRINK`` instead).
2. **P0.2 — Kelly payout identity** is covered by ``test_bench_p0_kelly.py``
   and deliberately not duplicated here (nothing is imported from it).
3. **P0.10 — vig-free honesty.** ``market_microstructure_features`` exposes
   the raw single-book spread under ``market_spread_raw`` and never under the
   dishonest ``fair_spread_vigfree`` label.
4. **Gaussian/Skellam agreement.** ``spread_cover_prob`` (Gaussian margin
   model) and ``cover_prob_skellam`` (Skellam margin model with Poisson means
   matched so ``mu_home + mu_away == sigma**2``) agree within 0.02 for sigma
   in [5, 50].

Adversarial findings baked into this bench (see ADVERSARIAL notes inline):

* **Integer market lines break the 0.02 Gaussian/Skellam tolerance.** The
  Skellam strict-cover event excludes the push (integer margin ==
  ``-market_spread``) while the Gaussian integrates straight through it; at
  sigma=5 the gap reaches ~0.044. Real NBA books deal half-integer lines
  precisely to avoid pushes, so the agreement invariant is scoped to
  half-integer lines and the integer-line gap is pinned by a witness test.
* **A single-sided price of exactly ±100 is a vacuous shrink boundary.** The
  raw negated split there is already exactly 0.5/0.5, so "output differs from
  the raw vigged negation" cannot hold; strict-shrink assertions use prices
  away from ±100 and the boundary is pinned by its own test.
* **Sigma below 4 breaks Gaussian/Skellam agreement** (diff up to ~0.37 at
  sigma=1): ``spread_cover_prob`` floors its sigma at 4.0 while the matched
  Skellam keeps the true variance. The [5, 50] scope of the invariant is
  therefore necessary, not merely sufficient.
* **Symmetric (−x, +x) ML pairs carry zero overround** (implied(−x) +
  implied(+x) == 1 exactly), so "raw implied sum > 1" is only asserted for
  genuinely juiced pairs — a counterexample caught while tightening
  ``test_bench_devig_two_way_sums_to_one_extreme_juice``.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.devig import multiplicative_devig
from pipeline.market import (
    ASSUMED_SINGLE_SIDED_VIG_SHRINK,
    devig_two_way,
    fair_probs_from_ml_pair,
    implied_probability,
    is_single_sided_ml_quote,
    market_microstructure_features,
    spread_cover_prob,
)
from pipeline.skellam import cover_prob_skellam

# Deterministic seed for every randomized case in this file (Task 10.2.3).
SEED = 100203

# Adversarial juice extremes, including the required −10000/+5000 pair.
# Third element flags whether the pair carries a real overround: symmetric
# (−x, +x) pairs sum to exactly 1 by construction (implied(−x) = x/(x+100),
# implied(+x) = 100/(x+100)), so "raw implied sum > 1" must not be asserted
# for them — a counterexample caught while tightening this very test.
EXTREME_ML_PAIRS = [
    (-10000.0, 5000.0, True),
    (5000.0, -10000.0, True),
    (-10000.0, 100.0, True),
    (-110.0, -110.0, True),   # both-negative pick'em pricing
    (-101.0, 101.0, False),   # symmetric: zero overround by construction
    (-100.0, 100.0, False),   # exactly even: zero overround boundary
]

# Single-sided home prices, kept away from the vacuous ±100 boundary.
SINGLE_SIDED_ML_HOME = [-10000.0, -500.0, -200.0, -150.0, -110.0, -101.0,
                        101.0, 120.0, 180.0, 5000.0]

GAUSSIAN_SKELLAM_TOL = 0.02


def _gaussian_skellam_pair(sigma: float, market_spread: float, edge: float):
    """Matched Gaussian/Skellam cover probabilities for one synthetic game.

    ``edge = model_spread + market_spread``; Poisson means are chosen so the
    Skellam margin variance ``mu_home + mu_away`` equals ``sigma**2`` and the
    Skellam margin mean equals the model spread.
    """
    total = float(sigma) ** 2
    model_spread = float(edge) - float(market_spread)
    mu_home = (total + model_spread) / 2.0
    mu_away = (total - model_spread) / 2.0
    p_gauss = spread_cover_prob(model_spread, market_spread, sigma=sigma)
    p_skellam = cover_prob_skellam(mu_home, mu_away, market_spread)
    return p_gauss, p_skellam


# ──────────────────────────────────────────────────────────────────────────────
# 1. P0.3 — de-vig sums to 1; single-sided feeds flagged + shrunk.
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.bench
def test_bench_devig_two_way_sums_to_one_random_ml_pairs():
    """100 random two-sided ML pairs: fair probs sum to 1.0 ± 1e-9."""
    rng = np.random.default_rng(SEED)
    for i in range(100):
        if i < 80:
            # One favorite (negative) and one underdog (positive), random side.
            fav = -rng.uniform(100.0, 10000.0)
            dog = rng.uniform(100.0, 5000.0)
            ml_home, ml_away = (fav, dog) if rng.random() < 0.5 else (dog, fav)
        else:
            # Both-negative near-even pricing (e.g. −105/−115 pick'em).
            ml_home = -rng.uniform(101.0, 120.0)
            ml_away = -rng.uniform(101.0, 120.0)
        p_home = implied_probability(ml_home)
        p_away = implied_probability(ml_away)

        fair_home, fair_away = devig_two_way(p_home, p_away)
        assert fair_home + fair_away == pytest.approx(1.0, abs=1e-9)
        assert 0.0 <= fair_home <= 1.0
        assert 0.0 <= fair_away <= 1.0

        # The pipeline entry point routes through the same de-vig.
        fh, fa = fair_probs_from_ml_pair(ml_home, ml_away)
        assert fh + fa == pytest.approx(1.0, abs=1e-9)
        assert fh == pytest.approx(fair_home, abs=1e-12)
        assert fa == pytest.approx(fair_away, abs=1e-12)

        # pipeline.devig's guarded multiplicative transform agrees exactly.
        p = multiplicative_devig([p_home, p_away])
        assert float(p.sum()) == pytest.approx(1.0, abs=1e-9)
        assert float(p[0]) == pytest.approx(fair_home, abs=1e-9)
        assert float(p[1]) == pytest.approx(fair_away, abs=1e-9)


@pytest.mark.bench
@pytest.mark.parametrize("ml_home,ml_away,expect_overround", EXTREME_ML_PAIRS)
def test_bench_devig_two_way_sums_to_one_extreme_juice(ml_home, ml_away, expect_overround):
    """Adversarial: extreme juice pairs (−10000/+5000 etc.) still sum to 1."""
    p_home = implied_probability(ml_home)
    p_away = implied_probability(ml_away)
    fair_home, fair_away = devig_two_way(p_home, p_away)
    assert fair_home + fair_away == pytest.approx(1.0, abs=1e-9)
    assert 0.0 <= fair_home <= 1.0
    assert 0.0 <= fair_away <= 1.0
    # Vig must actually be present to remove: raw implied sum exceeds 1 for
    # genuinely juiced pairs (symmetric pairs carry none — see above).
    if expect_overround:
        assert p_home + p_away > 1.0
    else:
        assert p_home + p_away == pytest.approx(1.0, abs=1e-9)
    # Favorite/underdog ordering is preserved by the de-vig.
    assert (fair_home > fair_away) == (p_home > p_away)


@pytest.mark.bench
def test_bench_devig_two_way_degenerate_inputs_still_sum_to_one():
    """Degenerate guards (None/NaN/tiny totals) return the 0.5/0.5 coin flip."""
    assert devig_two_way(None, 0.6) == (0.5, 0.5)
    assert devig_two_way(0.6, None) == (0.5, 0.5)
    assert devig_two_way(float("nan"), 0.6) == (0.5, 0.5)
    fair_a, fair_b = devig_two_way(1e-12, 1e-12)  # total <= 1e-9 guard
    assert fair_a + fair_b == pytest.approx(1.0, abs=1e-9)


@pytest.mark.bench
@pytest.mark.parametrize("ml_home", SINGLE_SIDED_ML_HOME)
def test_bench_single_sided_ml_flagged_and_shrunk_never_raw_negation(ml_home):
    """P0.3 acceptance: single-sided output is flagged and shrunk toward 0.5."""
    # The fallback must be detectable by callers.
    assert is_single_sided_ml_quote(None) is True
    assert is_single_sided_ml_quote(float("nan")) is True

    raw_home = implied_probability(ml_home)
    raw_away = implied_probability(-ml_home)  # the legacy negation fallback
    # Sanity: the negated pair sums to 1 by construction, so a plain de-vig
    # of it is a no-op — this is exactly the P0.3 bug shape.
    noop_home, noop_away = devig_two_way(raw_home, raw_away)
    assert noop_home == pytest.approx(raw_home, abs=1e-9)
    assert noop_away == pytest.approx(raw_away, abs=1e-9)

    fair_home, fair_away = fair_probs_from_ml_pair(ml_home)  # away ML missing
    # Still a valid probability pair…
    assert fair_home + fair_away == pytest.approx(1.0, abs=1e-9)
    # …but never the raw vigged negation…
    assert fair_home != pytest.approx(raw_home, abs=1e-9)
    assert fair_away != pytest.approx(raw_away, abs=1e-9)
    # …and strictly shrunk toward 0.5…
    assert abs(fair_home - 0.5) < abs(raw_home - 0.5)
    assert abs(fair_away - 0.5) < abs(raw_away - 0.5)
    # …by exactly the documented conservative factor.
    shrink = float(ASSUMED_SINGLE_SIDED_VIG_SHRINK)
    assert fair_home == pytest.approx((1.0 - shrink) * raw_home + shrink * 0.5, abs=1e-9)
    assert fair_away == pytest.approx((1.0 - shrink) * raw_away + shrink * 0.5, abs=1e-9)


@pytest.mark.bench
@pytest.mark.parametrize("ml_home", [-100.0, 100.0])
def test_bench_single_sided_even_price_shrink_is_vacuous(ml_home):
    """ADVERSARIAL boundary: at exactly ±100 the raw negation *is* 0.5/0.5.

    Shrinking toward 0.5 cannot change (or be distinguished from) the raw
    split there, so the "never the raw negation" assertion above is scoped to
    prices off ±100. The output must still be the valid 0.5/0.5 pair and the
    quote must still be flaggable as single-sided.
    """
    assert is_single_sided_ml_quote(None) is True
    fair_home, fair_away = fair_probs_from_ml_pair(ml_home)
    assert fair_home + fair_away == pytest.approx(1.0, abs=1e-9)
    assert fair_home == pytest.approx(0.5, abs=1e-12)
    assert fair_away == pytest.approx(0.5, abs=1e-12)


@pytest.mark.bench
def test_bench_two_sided_quotes_take_exact_devig_path_no_shrink():
    """Real two-sided quotes get the exact de-vig with no assumed shrink."""
    ml_home, ml_away = -200.0, 160.0
    assert is_single_sided_ml_quote(ml_away) is False
    fair_home, fair_away = fair_probs_from_ml_pair(ml_home, ml_away)
    exp_home, exp_away = devig_two_way(
        implied_probability(ml_home), implied_probability(ml_away)
    )
    assert fair_home == pytest.approx(exp_home, abs=1e-12)
    assert fair_away == pytest.approx(exp_away, abs=1e-12)
    # Must NOT carry the single-sided assumed-vig shrink.
    shrink = float(ASSUMED_SINGLE_SIDED_VIG_SHRINK)
    assert fair_home != pytest.approx((1.0 - shrink) * exp_home + shrink * 0.5, abs=1e-9)


# ──────────────────────────────────────────────────────────────────────────────
# 3. P0.10 — market_spread_raw, never fair_spread_vigfree.
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.bench
def test_bench_microstructure_market_spread_raw_not_vigfree():
    """P0.10 acceptance: raw spread honestly labeled, vigfree key absent."""
    result = market_microstructure_features(
        spread_move=0, public_home_pct=0.5, market_spread=-3.5
    )
    assert "market_spread_raw" in result
    assert "fair_spread_vigfree" not in result
    assert result["market_spread_raw"] == pytest.approx(-3.5)


# ──────────────────────────────────────────────────────────────────────────────
# 4. Gaussian/Skellam agreement (sigma in [5, 50], half-integer lines).
# ──────────────────────────────────────────────────────────────────────────────

@pytest.mark.bench
def test_bench_gaussian_skellam_agreement_sigma_grid():
    """Fixed grid: endpoints sigma=5 and sigma=50 included, tol 0.02."""
    for sigma in [5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0]:
        total = sigma ** 2
        for market_spread in [-9.5, -6.5, -3.5, 0.5, 2.5, 4.5, 7.5]:
            for edge in [-8.0, -5.0, -3.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]:
                model_spread = edge - market_spread
                if (total - abs(model_spread)) / 2.0 < 0.5:
                    continue  # Poisson means must stay positive
                p_gauss, p_skellam = _gaussian_skellam_pair(
                    sigma, market_spread, edge
                )
                assert abs(p_gauss - p_skellam) <= GAUSSIAN_SKELLAM_TOL, (
                    f"sigma={sigma} spread={market_spread} edge={edge}: "
                    f"gauss={p_gauss} skellam={p_skellam}"
                )


@pytest.mark.bench
def test_bench_gaussian_skellam_agreement_random_half_integer_lines():
    """200 deterministic random (sigma, line, edge) combos, tol 0.02."""
    rng = np.random.default_rng(SEED)
    checked = 0
    while checked < 200:
        sigma = rng.uniform(5.0, 50.0)
        total = sigma ** 2
        market_spread = float(rng.integers(-24, 20)) + 0.5  # half-integer line
        edge = rng.uniform(-3.0 * sigma, 3.0 * sigma)  # stay inside z-clip
        model_spread = edge - market_spread
        if (total - abs(model_spread)) / 2.0 < 0.5:
            continue
        p_gauss, p_skellam = _gaussian_skellam_pair(sigma, market_spread, edge)
        assert abs(p_gauss - p_skellam) <= GAUSSIAN_SKELLAM_TOL, (
            f"sigma={sigma} spread={market_spread} edge={edge}: "
            f"gauss={p_gauss} skellam={p_skellam}"
        )
        # Away direction is the exact complement on both models.
        p_gauss_away = spread_cover_prob(
            model_spread, market_spread, sigma=sigma, direction="Away"
        )
        p_skellam_away = cover_prob_skellam(
            (total + model_spread) / 2.0, (total - model_spread) / 2.0,
            market_spread, direction="Away",
        )
        assert p_gauss_away == pytest.approx(1.0 - p_gauss, abs=1e-12)
        assert p_skellam_away == pytest.approx(1.0 - p_skellam, abs=1e-12)
        checked += 1


@pytest.mark.bench
def test_bench_gaussian_skellam_integer_line_push_gap_witness():
    """ADVERSARIAL witness: integer lines can exceed the 0.02 tolerance.

    Counterexample found by the adversarial protocol: at sigma=5 on the
    integer line −7 with zero edge, the Gaussian returns 0.5 while the
    Skellam strict-cover probability is ≈0.456 — a ≈0.044 gap from the push
    mass / half-unit continuity offset. The agreement invariant is therefore
    asserted only on half-integer lines (which real books use to avoid
    pushes); the neighboring half-integer line on the same matchup agrees.
    """
    p_gauss, p_skellam = _gaussian_skellam_pair(5.0, -7.0, 0.0)
    assert p_gauss == pytest.approx(0.5, abs=1e-12)
    assert abs(p_gauss - p_skellam) > GAUSSIAN_SKELLAM_TOL  # documented gap

    p_gauss_half, p_skellam_half = _gaussian_skellam_pair(5.0, -6.5, 0.0)
    assert abs(p_gauss_half - p_skellam_half) <= GAUSSIAN_SKELLAM_TOL


@pytest.mark.bench
@pytest.mark.parametrize("market_spread", [-3.5, 0.5, 3.5])
@pytest.mark.parametrize("direction", ["Home", "Away"])
def test_bench_skellam_near_degenerate_sigma_point_one_no_crash(market_spread, direction):
    """Adversarial: sigma=0.1 (Poisson means 0.005, clamped to 0.5 internally).

    Must not crash and must return a sensible probability in [0, 1].
    """
    p = cover_prob_skellam(0.005, 0.005, market_spread, direction=direction)
    assert np.isfinite(p)
    assert 0.0 <= p <= 1.0


@pytest.mark.bench
def test_bench_skellam_degenerate_scores_stay_in_unit_interval():
    """Zero/negative predicted scores are clamped; Home/Away complement holds."""
    for pred_home, pred_away in [(0.0, 0.0), (-5.0, 120.0), (120.0, -5.0)]:
        p = cover_prob_skellam(pred_home, pred_away, -3.5)
        assert np.isfinite(p)
        assert 0.0 <= p <= 1.0
    p_home = cover_prob_skellam(0.005, 0.005, -3.5, direction="Home")
    p_away = cover_prob_skellam(0.005, 0.005, -3.5, direction="Away")
    assert p_home + p_away == pytest.approx(1.0, abs=1e-12)
