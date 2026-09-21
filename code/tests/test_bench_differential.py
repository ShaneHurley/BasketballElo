"""Epic 10.5.2 — synthetic formula test bench: differential oracles.

Three cross-formula differential checks, building on (not repeating) the
market bench of ``test_bench_market.py`` (Epic 10.2.3), which established:
Gaussian/Skellam agreement within 0.02 for half-integer lines with sigma in
[5, 50]; integer lines carry a push-mass offset up to ~0.044 at sigma=5; and
``spread_cover_prob`` floors sigma at 4.0. Nothing here re-asserts those
facts; the integer-line and sigma-floor boundaries are *used* as scoping
inputs.

1. **De-vig method agreement.** All four ``pipeline.devig`` methods
   (multiplicative, power, odds_ratio, shin) agree pairwise within 0.01 on
   two-way books — but NOT universally. Documented boundary (measured on a
   seeded sweep, see witness test): the pairwise divergence grows ~linearly
   with ``overround x log-skew`` and exceeds 0.01 outside two scopes:

   * **Scope A (sharp books, any skew):** overround <= 1%. Worst observed
     over an 8,000-sample sweep: 0.0093 (at q_fav=0.9992 / q_dog=0.0105).
     Includes the required extreme pair -10000/+5000 (overround 0.97%,
     measured divergence 0.0081) — extreme *price* alone does not break
     agreement; extreme price *with* a wide book does.
   * **Scope B (typical near-even books):** both sides' implied probabilities
     in [1/3, 2/3] (American odds inside [-200, +200]) with overround <= 4%.
     Worst observed: 0.0085 at the -200 / 4%-overround corner. At 5% the
     corner reaches 0.0105 — the 4% cap is the documented edge.

   Outside both scopes the methods legitimately disagree (they encode
   different favorite-longshot corrections): -1000/+700 (a realistic heavy-
   favorite playoff book, 3.4% overround) diverges by 0.0208; -200/+100
   (16.7% overround) by 0.0205; -10000/+100 (49% overround) by 0.2917. On
   every skewed witness the favorite-side fair probs are strictly ordered
   multiplicative < shin < odds_ratio < power — pinned as a structural
   invariant so a refactor cannot silently swap method implementations.

2. **Skellam vs Gaussian convergence.** On a fixed half-integer grid, the
   max |Gaussian - Skellam| cover-probability gap decays monotonically from
   0.00732 (sigma=5) to 7.2e-6 (sigma=50) — a >1000x reduction, strictly
   decreasing at every grid step (min step ratio 1.73x). Scoped to
   half-integer lines per the market bench's integer-line push-mass finding
   and to sigma >= 5 per its sigma-floor finding.

3. **Static vs rolling spread calibration.** ``SpreadCalibrator.fit_static``
   (blanket-correct with end state) vs ``replay_rolling_spread_calibration``
   (leak-free predict-then-update; parity with live inference is already
   pinned by ``test_spread_calibration_train_live_parity.py`` and not
   duplicated here). Documented bounds on a synthetic 400-game stream
   (window=50, min_samples=30, binned=False, residual noise sd=8):

   * **Stationary stream** (mu = 1.5 + 0.85*pred): their corrected
     predictions agree on the settled final 100 rows within max 3.64 points
     / mean 1.32 (asserted <= 5.0 / <= 2.0). Static is the better estimator
     there (mu-error 0.36 vs rolling's 1.54) because it pools all 400 rows.
     The divergence is warmup-dominated by construction: rolling row 0 is
     exactly the identity (bias starts at 0, additive fallback until
     min_samples accrue) while static corrects row 0 with the end-state
     (a, b) — so agreement is scoped to the settled window, and the warmup
     mechanics are pinned by an exact witness.
   * **Conditional-mean drift** (bias +3 -> -4 and slope 0.9 -> 1.2 at
     t=240): rolling tracks better — final-100 mu-MAE 1.50 vs static 3.70
     (asserted margin >= 1.0), also on realized actuals (6.24 vs 7.02).
     Static never forgets: ``fit_static``'s window spans the whole slice, so
     the dead regime contaminates its end state permanently.
   * **Variance-only drift control** (noise sd 4 -> 16 at t=240, mean fixed):
     rolling does NOT win (static 0.67 <= rolling 1.50) — the drift-test
     advantage is specific to conditional-mean drift, not a uniform
     "rolling is better" artifact.

ADVERSARIAL PROTOCOL NOTES (things tried that would pass these tests while
violating the invariants, and the counter-measures):

* **Symmetric books are a vacuous agreement trap.** Any (-x, +x) pair has
  zero overround, so all four methods return the input exactly (measured
  0.0000 at -110/-110); -105/-115 differs by only 0.0008. A grid of
  near-symmetric books would pass with a broken power/shin implementation.
  Both seeded grids therefore sample up to their scope's skew/overround
  corner, and the scope-A sweep worst case (0.0093) sits within 8% of the
  0.01 tolerance — the test has teeth at the boundary.
* **Two methods can coincide while a third diverges.** At low skew,
  odds_ratio and shin track each other within ~5e-4 while power already
  diverges from both (e.g. -150/+100: odds-vs-shin 0.0005, power-vs-mult
  0.0073). All assertions therefore use the max over all six method pairs,
  not agreement with a single reference method.
* **Constant-prediction streams are a vacuous calibration trap.** With
  var(pred) <= 1e-6 both paths fall back to identity corrections (a=0, b=1)
  and "agree" trivially outside warmup. Every stream below uses sd-8 varied
  predictions, and the variance-only control proves the drift assertion
  credits rolling only when the conditional mean actually moves.
* **Rolling can "win for the wrong reason" against noisy actuals.** With
  residual sd=8 the realized-actual MAE difference is small (0.78) and could
  flip under an unlucky noise draw; the primary drift assertion is therefore
  against the true conditional mean mu_t (margin 2.2), with realized actuals
  as a secondary check.
* **Integer lines would poison the convergence signal.** At sigma=5 an
  integer line carries a ~0.044 push-mass gap (market bench witness) that
  would dominate gap(5) and conflate "lattice/push offset" with
  "distributional convergence"; the grid uses half-integer lines only.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.devig import DEVIG_METHODS
from pipeline.market import (
    SpreadCalibrator,
    implied_probability,
    replay_rolling_spread_calibration,
    spread_cover_prob,
)
from pipeline.skellam import cover_prob_skellam

# Deterministic seed for every randomized case in this file (Task 10.5.2).
SEED = 100502

DEVIG_TOL = 0.01
METHODS = list(DEVIG_METHODS)  # multiplicative, power, odds_ratio, shin

# Calibration stream configuration (documented in the module docstring).
CAL_WINDOW = 50
CAL_MIN_SAMPLES = 30
CAL_N = 400
CAL_NOISE_SD = 8.0
SETTLED_TAIL = 100  # final-window size for settled (post-warmup) comparisons


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _max_pairwise_gap(q: np.ndarray) -> float:
    """Max |p_a - p_b| over all six method pairs for one two-way book."""
    probs = {m: DEVIG_METHODS[m](q) for m in METHODS}
    return max(
        float(np.max(np.abs(probs[a] - probs[b])))
        for i, a in enumerate(METHODS)
        for b in METHODS[i + 1:]
    )


def _method_probs(q: np.ndarray) -> dict:
    return {m: DEVIG_METHODS[m](q) for m in METHODS}


def _static_vs_rolling(preds, actuals):
    """Blanket static correction vs leak-free rolling replay (binned off)."""
    static_cal = SpreadCalibrator.fit_static(
        preds, actuals, window=CAL_WINDOW, min_samples=CAL_MIN_SAMPLES, binned=False
    )
    static_corr = np.array([static_cal.correct(float(p)) for p in preds])
    rolling_corr, _ = replay_rolling_spread_calibration(
        preds,
        actuals,
        window=CAL_WINDOW,
        min_samples=CAL_MIN_SAMPLES,
        calibrator=SpreadCalibrator(
            window=CAL_WINDOW, min_samples=CAL_MIN_SAMPLES, binned=False
        ),
    )
    return static_corr, rolling_corr, static_cal


# ---------------------------------------------------------------------------
# 1. De-vig method agreement
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_bench_devig_agreement_sharp_books_seeded_grid():
    """Scope A: overround <= 1% (sharp books), ANY skew up to q_fav=0.99999.

    150 seeded books plus the required extreme pair -10000/+5000. Dense
    8,000-sample sweep worst case is 0.0093 (< 0.01), so the scope is a
    verified sufficient condition, not a fitted one.
    """
    rng = np.random.default_rng(SEED)
    books = []
    for _ in range(150):
        q_fav = rng.uniform(0.5001, 0.99999)
        overround = rng.uniform(0.001, 0.010)
        q_dog = 1.0 + overround - q_fav
        assert 0.0 < q_dog < 1.0
        books.append((np.array([q_fav, q_dog]), f"q=({q_fav:.5f},{q_dog:.5f})"))
    # Required extreme pair: -10000/+5000 (overround 0.97%, inside scope A).
    books.append((
        np.array([implied_probability(-10000.0), implied_probability(5000.0)]),
        "-10000/+5000",
    ))
    # More extreme-price sharp books.
    for fav, dog in [(-5000.0, 4000.0), (-20000.0, 8000.0)]:
        books.append((
            np.array([implied_probability(fav), implied_probability(dog)]),
            f"{fav:+.0f}/{dog:+.0f}",
        ))

    for q, label in books:
        assert q.sum() > 1.0  # vig present to remove
        probs = _method_probs(q)
        for m in METHODS:
            assert probs[m].sum() == pytest.approx(1.0, abs=1e-9)
        gap = _max_pairwise_gap(q)
        assert gap <= DEVIG_TOL, f"sharp-book scope violated at {label}: gap={gap:.5f}"


@pytest.mark.bench
def test_bench_devig_agreement_typical_books_seeded_grid():
    """Scope B: both sides inside [-200, +200] American, overround <= 4%.

    Covers the bulk of NBA two-way books, including both-negative pick'em
    pricing. The 4% cap is the documented edge: at 5% overround the -200
    corner reaches 0.0105 (witness test below).
    """
    rng = np.random.default_rng(SEED + 1)
    checked = 0
    while checked < 150:
        q_fav = rng.uniform(0.5001, 2.0 / 3.0)  # favorite implied, up to -200
        lo = 1.0001 - q_fav  # dog dear enough to keep a positive overround
        hi = min(2.0 / 3.0, 1.04 - q_fav)  # dog cheap enough to keep ovr <= 4%
        if hi <= lo:
            continue
        q_dog = rng.uniform(lo, hi)
        q = np.array([q_fav, q_dog])
        overround = q.sum() - 1.0
        assert 0.0 < overround <= 0.04 + 1e-12
        gap = _max_pairwise_gap(q)
        assert gap <= DEVIG_TOL, (
            f"typical-book scope violated at q=({q_fav:.4f},{q_dog:.4f}) "
            f"ovr={overround:.4f}: gap={gap:.5f}"
        )
        checked += 1


@pytest.mark.bench
@pytest.mark.parametrize(
    "ml_fav,ml_dog",
    [
        (-110.0, -110.0),   # symmetric pick'em: zero skew, exact agreement
        (-105.0, -115.0),   # both-negative pick'em, 4.7% overround
        (-150.0, 130.0),    # standard moderate book, 3.5% overround
        (-200.0, 170.0),    # scope-B corner odds, 3.7% overround
        (-10000.0, 5000.0), # required extreme pair (also in scope A grid)
    ],
)
def test_bench_devig_agreement_fixed_named_books(ml_fav, ml_dog):
    """Named books that must agree within 0.01 (each individually verified)."""
    q = np.array([implied_probability(ml_fav), implied_probability(ml_dog)])
    gap = _max_pairwise_gap(q)
    assert gap <= DEVIG_TOL, f"{ml_fav}/{ml_dog}: gap={gap:.5f}"
    if ml_fav == -110.0 and ml_dog == -110.0:
        assert gap == pytest.approx(0.0, abs=1e-12)


@pytest.mark.bench
@pytest.mark.parametrize(
    "ml_fav,ml_dog",
    [
        (-200.0, 160.0),    # realistic boundary: 5.1% overround at -200
        (-1000.0, 700.0),   # realistic heavy-favorite playoff book
        (-200.0, 100.0),    # wide soft book, 16.7% overround
        (-10000.0, 100.0),  # extreme price AND wide book, 49% overround
    ],
)
def test_bench_devig_divergence_witnesses_and_method_ordering(ml_fav, ml_dog):
    """ADVERSARIAL witness: outside scopes A/B the four methods legitimately
    diverge beyond 0.01 — they encode different favorite-longshot corrections
    and Task 027's leakage-free selector exists precisely because the choice
    matters there. Pins the boundary the agreement tests are scoped against,
    and the structural ordering multiplicative < shin < odds_ratio < power on
    the favorite's fair probability (so a refactor cannot swap methods).

    Exact decimal gaps are not pinned — they drift with library float noise;
    the invariant is divergence past DEVIG_TOL plus method ordering.
    """
    q = np.array([implied_probability(ml_fav), implied_probability(ml_dog)])
    gap = _max_pairwise_gap(q)
    assert gap > DEVIG_TOL

    probs = _method_probs(q)
    fav_side = (
        probs["multiplicative"][0]
        < probs["shin"][0]
        < probs["odds_ratio"][0]
        < probs["power"][0]
    )
    assert fav_side, f"method ordering broken at {ml_fav}/{ml_dog}: {probs}"
    assert gap == pytest.approx(
        float(abs(probs["multiplicative"][0] - probs["power"][0])), abs=1e-12
    )


@pytest.mark.bench
def test_bench_devig_swap_invariance_all_methods():
    """Side-swap invariance: de-vigging (home, away) must equal the reverse of
    de-vigging (away, home) exactly — every transform is a symmetric function
    of the book, so a home/away feed-order bug must show up here."""
    rng = np.random.default_rng(SEED + 2)
    for _ in range(30):
        q_fav = rng.uniform(0.5001, 0.999)
        q_dog = rng.uniform(1.001 - q_fav, min(0.999, 1.06 - q_fav))
        q = np.array([q_fav, q_dog])
        for m in METHODS:
            p_fwd = DEVIG_METHODS[m](q)
            p_rev = DEVIG_METHODS[m](q[::-1])[::-1]
            assert np.allclose(p_fwd, p_rev, atol=1e-12), m


# ---------------------------------------------------------------------------
# 2. Skellam vs Gaussian convergence in sigma
# ---------------------------------------------------------------------------

# Fixed half-integer line/edge grid (integer lines excluded: push-mass offset
# documented by test_bench_market.py). |edge| <= 6 keeps z = edge/sigma <= 1.2,
# far from spread_cover_prob's +/-4 z-clip, and keeps matched Poisson means
# positive at sigma=5 ((25 - 19.5)/2 > 0.5).
_HALF_LINES = [-13.5, -9.5, -6.5, -3.5, -0.5, 2.5, 5.5, 8.5, 12.5]
_EDGES = [-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0]
_SIGMA_GRID = [5.0, 8.0, 12.0, 16.0, 20.0, 25.0, 30.0, 40.0, 50.0]


def _max_gaussian_skellam_gap(sigma: float) -> tuple[float, list]:
    total = float(sigma) ** 2
    gaps, p_gauss_all = [], []
    for line in _HALF_LINES:
        for edge in _EDGES:
            model_spread = edge - line
            if (total - abs(model_spread)) / 2.0 < 0.5:
                continue  # matched Poisson means must stay positive
            p_gauss = spread_cover_prob(model_spread, line, sigma=sigma)
            p_skellam = cover_prob_skellam(
                (total + model_spread) / 2.0, (total - model_spread) / 2.0, line
            )
            gaps.append(abs(p_gauss - p_skellam))
            p_gauss_all.append(p_gauss)
    return max(gaps), p_gauss_all


@pytest.mark.bench
def test_bench_skellam_gaussian_convergence_monotone_in_sigma():
    """Cover-probability gap converges as sigma grows (CLT: Skellam with
    matched variance sigma^2 approaches the Gaussian margin model).

    Documented measurements on this grid: 0.00732 (sigma=5) -> 7.2e-6
    (sigma=50), strictly decreasing at every step (min step ratio 1.73x).
    Asserted: endpoint gap(50) < gap(5) with a >=100x reduction, monotone-ish
    decrease (5% slack per step for solver jitter), and non-vacuity anchors
    so a degenerate constant-probability regression cannot pass.
    """
    gaps = []
    for sigma in _SIGMA_GRID:
        gap, _ = _max_gaussian_skellam_gap(sigma)
        gaps.append(gap)

    # Task assertion: the gap at sigma=50 is smaller than at sigma=5.
    assert gaps[-1] < gaps[0]
    # Documented convergence rate: >1000x measured; assert a conservative 100x.
    assert gaps[-1] <= gaps[0] / 100.0
    # Monotone-ish convergence across the fixed grid (min measured step ratio
    # is 1.73x; 5% relative slack plus a 1e-6 absolute floor for tail jitter).
    for prev, curr in zip(gaps, gaps[1:]):
        assert curr <= prev * 1.05 + 1e-6, f"non-monotone: {gaps}"
    # Non-vacuity: the sigma=5 gap is real (not a degenerate zero) and the
    # grid exercises a wide range of cover probabilities (not all ~0.5/clipped).
    assert gaps[0] >= 1e-3
    _, p_gauss_grid = _max_gaussian_skellam_gap(_SIGMA_GRID[0])
    assert float(np.std(p_gauss_grid)) > 0.1
    assert min(p_gauss_grid) < 0.2 and max(p_gauss_grid) > 0.8


# ---------------------------------------------------------------------------
# 3. fit_static vs replay_rolling_spread_calibration
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_bench_static_vs_rolling_stationary_bounded_divergence():
    """Stationary stream (mu = 1.5 + 0.85*pred, noise sd=8): the two
    correction paths agree on the settled final 100 rows within the
    documented bound (measured max 3.64 / mean 1.32 points; asserted <= 5.0 /
    <= 2.0), and static is the better estimator there because it pools all
    400 rows (mu-MAE 0.36 <= rolling 1.54).

    The bound is scoped to the settled window because the full-series
    divergence is warmup-dominated *by construction*: rolling row 0 is
    exactly the identity (bias starts at 0; additive fallback until
    min_samples accrue) while static corrects row 0 with the end-state
    (a, b) — pinned exactly below.
    """
    rng = np.random.default_rng(SEED + 10)
    preds = rng.normal(0.0, 8.0, CAL_N)
    mu = 1.5 + 0.85 * preds
    actuals = mu + rng.normal(0.0, CAL_NOISE_SD, CAL_N)

    static_corr, rolling_corr, static_cal = _static_vs_rolling(preds, actuals)

    # Warmup mechanics witness (exact): rolling row 0 is the identity…
    assert rolling_corr[0] == pytest.approx(float(preds[0]), abs=1e-12)
    # …while static row 0 carries the end-state linear fit (a ~ 1.5, b ~ 0.85).
    assert static_corr[0] == pytest.approx(
        static_cal.a + static_cal.b * float(preds[0]), abs=1e-9
    )
    assert static_corr[0] != pytest.approx(float(preds[0]), abs=1e-6)

    # Documented settled-window bound.
    tail = np.s_[-SETTLED_TAIL:]
    div = np.abs(static_corr[tail] - rolling_corr[tail])
    assert float(div.max()) <= 5.0, f"max divergence {div.max():.3f}"
    assert float(div.mean()) <= 2.0, f"mean divergence {div.mean():.3f}"

    # On a stationary stream static must not be the worse estimator.
    err_static = np.abs(static_corr[tail] - mu[tail]).mean()
    err_rolling = np.abs(rolling_corr[tail] - mu[tail]).mean()
    assert err_static <= err_rolling


@pytest.mark.bench
def test_bench_static_vs_rolling_mean_drift_rolling_tracks_better():
    """Conditional-mean drift (bias +3 -> -4, slope 0.9 -> 1.2 at t=240):
    rolling tracks the new regime; static's whole-slice window never forgets
    the dead one. Documented: final-100 mu-MAE rolling 1.50 vs static 3.70
    (asserted margin >= 1.0 so a both-equally-bad regression cannot pass);
    secondary check on realized actuals (6.24 vs 7.02).
    """
    rng = np.random.default_rng(SEED + 11)
    t = np.arange(CAL_N)
    preds = rng.normal(0.0, 8.0, CAL_N)
    a_t = np.where(t < 240, 3.0, -4.0)
    b_t = np.where(t < 240, 0.9, 1.2)
    mu = a_t + b_t * preds
    actuals = mu + rng.normal(0.0, CAL_NOISE_SD, CAL_N)

    static_corr, rolling_corr, _ = _static_vs_rolling(preds, actuals)

    tail = np.s_[-SETTLED_TAIL:]
    err_static = np.abs(static_corr[tail] - mu[tail]).mean()
    err_rolling = np.abs(rolling_corr[tail] - mu[tail]).mean()
    assert err_rolling <= err_static - 1.0, (
        f"rolling should track mean drift clearly better: "
        f"rolling={err_rolling:.3f} static={err_static:.3f}"
    )
    # Non-vacuity: static must be genuinely wrong on the settled window.
    assert err_static >= 2.5
    # Secondary: the advantage survives realized-noise evaluation too.
    mae_static = np.abs(static_corr[tail] - actuals[tail]).mean()
    mae_rolling = np.abs(rolling_corr[tail] - actuals[tail]).mean()
    assert mae_rolling < mae_static


@pytest.mark.bench
def test_bench_static_vs_rolling_variance_only_drift_control():
    """ADVERSARIAL control: variance-only drift (noise sd 4 -> 16 at t=240,
    conditional mean fixed at mu = 1.0 + 0.9*pred). Rolling must NOT
    spuriously "track" anything — its window estimate just gets noisier, so
    static stays the better estimator (measured mu-MAE static 0.67 <= rolling
    1.50). Proves the drift test above credits rolling specifically for
    conditional-mean tracking, not a uniform advantage."""
    rng = np.random.default_rng(SEED + 12)
    t = np.arange(CAL_N)
    preds = rng.normal(0.0, 8.0, CAL_N)
    mu = 1.0 + 0.9 * preds
    sd_t = np.where(t < 240, 4.0, 16.0)
    actuals = mu + rng.normal(0.0, 1.0, CAL_N) * sd_t

    static_corr, rolling_corr, _ = _static_vs_rolling(preds, actuals)

    tail = np.s_[-SETTLED_TAIL:]
    err_static = np.abs(static_corr[tail] - mu[tail]).mean()
    err_rolling = np.abs(rolling_corr[tail] - mu[tail]).mean()
    assert err_static <= err_rolling, (
        f"variance-only drift: rolling should not beat static "
        f"(static={err_static:.3f}, rolling={err_rolling:.3f})"
    )
