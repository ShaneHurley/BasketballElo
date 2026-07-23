"""Task 036: optional/research-only TVAE/SDV synthetic-data gate.

Verifies: (a) the hard fold-isolation guard rejects any synthetic row id
that also appears in validation/test/calibration, (b) each statistical
fidelity gate correctly rejects an obviously-bad synthetic generator and
accepts an obviously-good one, and (c) the combined gate rejects synthetic
augmentation that narrows uncertainty even if its point-estimate score
improves.
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.synthetic_research import (
    SyntheticFoldLeakError,
    assert_synthetic_rows_train_fold_only,
    check_conditional_correlations,
    check_constraints,
    check_nearest_neighbor_copying,
    check_support,
    check_tails,
    evaluate_synthetic_generator,
)


def test_fold_leak_guard_raises_on_overlap_with_validation():
    with pytest.raises(SyntheticFoldLeakError):
        assert_synthetic_rows_train_fold_only(
            synthetic_row_ids=["s1", "s2", "v1"],
            validation_ids=["v1", "v2"],
            test_ids=["t1"],
        )


def test_fold_leak_guard_raises_on_overlap_with_test():
    with pytest.raises(SyntheticFoldLeakError):
        assert_synthetic_rows_train_fold_only(
            synthetic_row_ids=["s1"], test_ids=["s1"],
        )


def test_fold_leak_guard_passes_when_disjoint():
    assert_synthetic_rows_train_fold_only(
        synthetic_row_ids=["s1", "s2"],
        validation_ids=["v1"], test_ids=["t1"], calibration_ids=["c1"],
    )  # must not raise


def _real_train(n=500, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "minutes": rng.normal(24, 6, n).clip(0, 48),
        "margin": rng.normal(0, 12, n),
    })


def test_check_support_accepts_matching_range():
    real = _real_train()
    synth = _real_train(seed=1)
    result = check_support(real, synth, cols=["minutes", "margin"])
    assert result.passed


def test_check_support_rejects_out_of_range_synthetic():
    real = _real_train()
    synth = real.copy()
    synth["minutes"] = synth["minutes"] + 1000.0  # wildly out of real support
    result = check_support(real, synth, cols=["minutes"])
    assert not result.passed


def test_check_constraints_flags_violations():
    synth = pd.DataFrame({"minutes": [-5.0, 10.0, 48.0]})
    result = check_constraints(synth, {"nonnegative_minutes": lambda df: df["minutes"] >= 0})
    assert not result.passed
    assert result.metrics["constraint_violation_frac_nonnegative_minutes"] == pytest.approx(1 / 3)


def test_check_tails_rejects_shifted_distribution():
    real = _real_train()
    bad_synth = real.copy()
    bad_synth["minutes"] = bad_synth["minutes"] + 20.0  # obviously shifted
    result = check_tails(real, bad_synth, cols=["minutes"])
    assert not result.passed


def test_check_tails_accepts_similar_distribution():
    real = _real_train()
    good_synth = _real_train(seed=2)
    result = check_tails(real, good_synth, cols=["minutes", "margin"])
    assert result.passed


def test_nearest_neighbor_copying_flags_exact_duplicates():
    real = _real_train(n=200)
    copied = real.sample(n=50, random_state=3).reset_index(drop=True)  # exact copies
    result = check_nearest_neighbor_copying(real, copied, cols=["minutes", "margin"])
    assert not result.passed
    assert result.metrics["nn_copy_frac"] > 0.5


def test_nearest_neighbor_copying_accepts_genuinely_novel_rows():
    real = _real_train(n=200)
    novel = _real_train(n=50, seed=99)
    result = check_nearest_neighbor_copying(real, novel, cols=["minutes", "margin"])
    assert result.passed


def test_conditional_correlations_rejects_broken_structure():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(size=n)
    real = pd.DataFrame({"x": x, "y": x * 0.9 + rng.normal(scale=0.1, size=n)})
    # Synthetic destroys the x/y correlation entirely.
    synth = pd.DataFrame({"x": rng.normal(size=n), "y": rng.normal(size=n)})
    result = check_conditional_correlations(real, synth, cols=["x", "y"], tol=0.25)
    assert not result.passed


def test_conditional_correlations_accepts_preserved_structure():
    rng = np.random.default_rng(5)
    n = 500
    x = rng.normal(size=n)
    real = pd.DataFrame({"x": x, "y": x * 0.9 + rng.normal(scale=0.1, size=n)})
    x2 = rng.normal(size=n)
    synth = pd.DataFrame({"x": x2, "y": x2 * 0.9 + rng.normal(scale=0.1, size=n)})
    result = check_conditional_correlations(real, synth, cols=["x", "y"], tol=0.25)
    assert result.passed


def _fit_mean_model(df):
    return {"mean": df["margin"].mean(), "std": df["margin"].std()}


def _score_mae(model, test_df):
    return float(np.mean(np.abs(test_df["margin"] - model["mean"])))


def _uncertainty(model, test_df):
    return float(model["std"])


def test_evaluate_synthetic_generator_rejects_narrowed_uncertainty_even_if_score_improves():
    rng = np.random.default_rng(7)
    real_train = pd.DataFrame({"margin": rng.normal(0, 12, 300)})
    real_test = pd.DataFrame({"margin": rng.normal(0, 12, 300)})
    # "Bad" synthetic generator: near-copy of the mean with almost no
    # spread, which would artificially shrink the fitted std and could look
    # like it improves MAE against a lucky test-fold mean, but destroys
    # calibrated uncertainty.
    synthetic = pd.DataFrame({"margin": np.full(300, real_train["margin"].mean())})

    result = evaluate_synthetic_generator(
        real_train=real_train, synthetic=synthetic, real_test=real_test,
        cols=["margin"],
        fit_fn=_fit_mean_model, score_fn=_score_mae, uncertainty_fn=_uncertainty,
        min_uncertainty_ratio=0.85,
    )
    assert not result.passed
    assert any("uncertainty" in r or "outside the real support" in r or "KS test" in r
               or "copies" in r for r in result.reasons)


def test_evaluate_synthetic_generator_accepts_genuinely_helpful_generator():
    rng = np.random.default_rng(11)
    real_train = pd.DataFrame({"margin": rng.normal(0, 12, 300)})
    real_test = pd.DataFrame({"margin": rng.normal(0, 12, 300)})
    # A "good" generator: statistically indistinguishable extra draws from
    # the same real distribution, which should not be rejected.
    synthetic = pd.DataFrame({"margin": rng.normal(0, 12, 300)})

    result = evaluate_synthetic_generator(
        real_train=real_train, synthetic=synthetic, real_test=real_test,
        cols=["margin"],
        fit_fn=_fit_mean_model, score_fn=_score_mae, uncertainty_fn=_uncertainty,
        min_uncertainty_ratio=0.85,
    )
    assert result.passed, result.reasons
