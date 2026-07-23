"""Tasks 026-027: de-vig candidate transforms and leakage-free selection."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.devig import (
    DEVIG_METHODS,
    DevigError,
    brier_score,
    calibration_error,
    devig,
    log_loss_score,
    multiplicative_devig,
    odds_ratio_devig,
    power_devig,
    select_devig_method_from_folds,
    shin_devig,
    store_devig_choice,
)


def _implied(american):
    if american > 0:
        return 100.0 / (american + 100.0)
    return abs(american) / (abs(american) + 100.0)


# ---------------------------------------------------------------------------
# Task 026: every method finite, monotone, in [0,1], sums to 1.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("method_name", list(DEVIG_METHODS))
@pytest.mark.parametrize("american_pair", [(-150, 130), (-110, -110), (250, -350), (-1000, 600)])
def test_every_method_returns_valid_probabilities(method_name, american_pair):
    q = np.array([_implied(a) for a in american_pair])
    p = devig(q, method=method_name)
    assert np.all(np.isfinite(p))
    assert np.all(p >= 0.0) and np.all(p <= 1.0)
    assert abs(p.sum() - 1.0) < 1e-6
    # Favorite (higher implied prob) stays favorite after de-vigging.
    assert (p[0] >= p[1]) == (q[0] >= q[1])


def test_multiplicative_devig_matches_simple_ratio():
    q = np.array([0.55, 0.55])
    p = multiplicative_devig(q)
    assert p[0] == pytest.approx(0.5)
    assert p[1] == pytest.approx(0.5)


def test_power_devig_reduces_to_multiplicative_when_two_sided_symmetric():
    q = np.array([0.5238, 0.5238])  # -110/-110, symmetric vig
    p = power_devig(q)
    assert p[0] == pytest.approx(0.5, abs=1e-3)


def test_devig_rejects_out_of_range_probability():
    with pytest.raises(DevigError):
        multiplicative_devig(np.array([1.2, 0.3]))
    with pytest.raises(DevigError):
        multiplicative_devig(np.array([0.5, -0.1]))


def test_devig_rejects_nonfinite_input():
    with pytest.raises(DevigError):
        multiplicative_devig(np.array([np.nan, 0.5]))
    with pytest.raises(DevigError):
        multiplicative_devig(np.array([np.inf, 0.5]))


def test_unknown_method_raises():
    with pytest.raises(DevigError):
        devig(np.array([0.5, 0.5]), method="not_a_method")


def test_odds_ratio_and_shin_agree_with_multiplicative_at_zero_vig():
    q = np.array([0.5, 0.5])
    for fn in (multiplicative_devig, power_devig, odds_ratio_devig, shin_devig):
        p = fn(q)
        assert p[0] == pytest.approx(0.5, abs=1e-3)
        assert p[1] == pytest.approx(0.5, abs=1e-3)


def test_favorite_longshot_methods_differ_from_multiplicative_on_skewed_book():
    # A heavily skewed favorite/dog line is exactly where the plan says the
    # power/Shin favorite-longshot correction should diverge from naive
    # proportional normalization.
    q = np.array([_implied(-900), _implied(650)])
    p_mult = multiplicative_devig(q)
    p_power = power_devig(q)
    assert not np.allclose(p_mult, p_power, atol=1e-4)


# ---------------------------------------------------------------------------
# Task 027: selection uses only prior folds.
# ---------------------------------------------------------------------------

def _make_folds(n_folds, n_per_fold, true_method, seed=0):
    """Synthetic folds where outcomes are generated from `true_method`'s fair
    probabilities, so that method should look best OOS once enough data has
    accrued."""
    rng = np.random.default_rng(seed)
    folds = []
    for _ in range(n_folds):
        qs, outs = [], []
        for _ in range(n_per_fold):
            fav_am = rng.uniform(-900, -105)
            dog_am = rng.uniform(105, 700)
            q = np.array([_implied(fav_am), _implied(dog_am)])
            p_true = devig(q, method=true_method)
            outcome = rng.random() < p_true[0]
            qs.append(q)
            outs.append(float(outcome))
        folds.append({"q": qs, "outcome": outs})
    return folds


def test_selection_never_uses_current_fold_outcomes():
    folds = _make_folds(6, 150, true_method="shin", seed=1)
    chosen_a, _ = select_devig_method_from_folds(folds, shrink_min_n=100)

    # Corrupt only the LAST fold's outcomes; the method chosen for that same
    # last fold must be unchanged, because it can only depend on folds[:-1].
    folds_corrupted = [dict(f) for f in folds]
    last = dict(folds_corrupted[-1])
    last["outcome"] = [1.0 - o for o in last["outcome"]]
    folds_corrupted[-1] = last
    chosen_b, _ = select_devig_method_from_folds(folds_corrupted, shrink_min_n=100)

    assert chosen_a[-1] == chosen_b[-1]
    assert chosen_a[:-1] == chosen_b[:-1]


def test_selection_does_not_hardcode_power_as_best():
    """When the *true* generating process is Shin (not Power), and there is
    enough prior-fold history, the selector must be able to pick something
    other than 'power'."""
    folds = _make_folds(8, 300, true_method="shin", seed=7)
    chosen, _ = select_devig_method_from_folds(folds, shrink_min_n=300)
    # After the burn-in fold(s), later folds should be able to select shin
    # (or at least not be stuck on power forever).
    assert any(c != "power" for c in chosen[2:])


def test_selection_shrinks_to_global_default_with_little_data():
    folds = _make_folds(3, 5, true_method="power", seed=2)  # tiny folds
    chosen, _ = select_devig_method_from_folds(folds, shrink_min_n=10_000, global_default="multiplicative")
    assert all(c == "multiplicative" for c in chosen)


def test_store_devig_choice_shrinks_by_book_market_horizon():
    store: dict = {}
    key = ("pinnacle", "spread", "T-60")
    picked = store_devig_choice(store, key, method="power", n_obs=5, shrink_min_n=200,
                                 global_default="multiplicative")
    assert picked == "multiplicative"
    assert store[key]["method"] == "multiplicative"

    picked2 = store_devig_choice(store, key, method="power", n_obs=5000, shrink_min_n=200,
                                  global_default="multiplicative")
    assert picked2 == "power"


# ---------------------------------------------------------------------------
# Loss functions used by the selector.
# ---------------------------------------------------------------------------

def test_brier_and_log_loss_perfect_prediction_is_zero_ish():
    assert brier_score([1.0], [1.0]) == pytest.approx(0.0)
    assert log_loss_score([0.999999999999], [1.0]) == pytest.approx(0.0, abs=1e-6)


def test_calibration_error_perfectly_calibrated_bucket():
    preds = [0.5] * 100
    outs = [1.0] * 50 + [0.0] * 50
    assert calibration_error(preds, outs) == pytest.approx(0.0)
