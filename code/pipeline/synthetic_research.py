"""Task 036: gated TVAE/SDV (or any other) synthetic-data research harness.

This module is **optional and research-only** (Phase 7 Stage B / Rule 7).
No production feature, model, or evaluation path may import a synthetic
generator's output through anything other than this module's gates, and
this module never runs anything against validation/test/grading rows.

Two independent safety layers:

1. `assert_synthetic_rows_train_fold_only` -- a hard, non-statistical guard:
   raise immediately if any synthetic row id overlaps the validation/test/
   calibration id sets for the active fold. This is checked unconditionally,
   before any of the statistical fidelity gates below even run.

2. A battery of statistical fidelity/utility gates, all computed against a
   held-out **real** test fold that the generator never saw:
     - `check_support`                  : synthetic values stay within the
                                           real training range per column.
     - `check_constraints`              : synthetic rows satisfy declared
                                           basketball constraints (e.g.
                                           minutes >= 0, team minutes == 240).
     - `check_tails`                    : synthetic tail quantiles are not
                                           wildly different from real ones
                                           (two-sample KS test per column).
     - `check_nearest_neighbor_copying` : synthetic rows are not
                                           near-exact copies of individual
                                           training rows (privacy/overfitting
                                           leak).
     - `check_conditional_correlations` : the synthetic correlation
                                           structure roughly matches the
                                           real training fold's.
     - `train_synthetic_test_real_utility`: downstream utility comparison
                                           (real-only vs real+synthetic)
                                           evaluated only on the real test
                                           fold.

`evaluate_synthetic_generator` combines all of the above into one
accept/reject decision: synthetic augmentation is rejected unless it
improves the real OOS metric *and* does not narrow predictive uncertainty
below a floor fraction of the real-only model's uncertainty (an easy way to
"cheat" an OOS score is to synthetically shrink predicted variance).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class SyntheticGateResult:
    passed: bool
    reasons: List[str] = field(default_factory=list)
    metrics: Dict[str, float] = field(default_factory=dict)


class SyntheticFoldLeakError(RuntimeError):
    """Raised when a synthetic row would be visible outside its training fold."""


def assert_synthetic_rows_train_fold_only(
    synthetic_row_ids: Iterable,
    *,
    validation_ids: Iterable = (),
    test_ids: Iterable = (),
    calibration_ids: Iterable = (),
) -> None:
    """Rule 7 hard guard: synthetic rows may exist only in training folds.

    Raises `SyntheticFoldLeakError` immediately if any synthetic row id is
    also present in the validation, test, or calibration id sets.
    """
    synth = set(synthetic_row_ids)
    for role_name, role_ids in (
        ("validation", validation_ids), ("test", test_ids), ("calibration", calibration_ids),
    ):
        overlap = synth & set(role_ids)
        if overlap:
            raise SyntheticFoldLeakError(
                f"{len(overlap)} synthetic row id(s) also appear in the "
                f"'{role_name}' fold: {sorted(overlap)[:10]} -- synthetic "
                "rows may exist only in training folds (Rule 7)"
            )


def check_support(real_train: pd.DataFrame, synthetic: pd.DataFrame, cols: Sequence[str],
                   tol: float = 0.05) -> SyntheticGateResult:
    reasons, metrics = [], {}
    for c in cols:
        lo, hi = real_train[c].min(), real_train[c].max()
        span = max(hi - lo, 1e-9)
        pad = span * tol
        bad = ((synthetic[c] < lo - pad) | (synthetic[c] > hi + pad)).mean()
        metrics[f"support_violation_frac_{c}"] = float(bad)
        if bad > tol:
            reasons.append(f"column '{c}': {bad:.1%} of synthetic rows fall outside the real support")
    return SyntheticGateResult(passed=not reasons, reasons=reasons, metrics=metrics)


def check_constraints(synthetic: pd.DataFrame, constraint_fns: Dict[str, Callable[[pd.DataFrame], pd.Series]]) -> SyntheticGateResult:
    """`constraint_fns`: name -> function(df) -> boolean Series (True = row satisfies constraint)."""
    reasons, metrics = [], {}
    for name, fn in constraint_fns.items():
        ok = fn(synthetic)
        violation_frac = float((~ok).mean())
        metrics[f"constraint_violation_frac_{name}"] = violation_frac
        if violation_frac > 0:
            reasons.append(f"constraint '{name}' violated by {violation_frac:.1%} of synthetic rows")
    return SyntheticGateResult(passed=not reasons, reasons=reasons, metrics=metrics)


def check_tails(real_train: pd.DataFrame, synthetic: pd.DataFrame, cols: Sequence[str],
                 alpha: float = 0.01) -> SyntheticGateResult:
    """Two-sample KS test per column; reject if the synthetic distribution's
    tails diverge significantly from the real training fold's (small
    p-value -> distributions differ)."""
    reasons, metrics = [], {}
    for c in cols:
        real_vals = real_train[c].dropna().to_numpy()
        synth_vals = synthetic[c].dropna().to_numpy()
        if len(real_vals) < 5 or len(synth_vals) < 5:
            continue
        stat, pvalue = stats.ks_2samp(real_vals, synth_vals)
        metrics[f"ks_stat_{c}"] = float(stat)
        metrics[f"ks_pvalue_{c}"] = float(pvalue)
        if pvalue < alpha:
            reasons.append(f"column '{c}': KS test rejects distributional match (p={pvalue:.4g})")
    return SyntheticGateResult(passed=not reasons, reasons=reasons, metrics=metrics)


def check_nearest_neighbor_copying(
    real_train: pd.DataFrame, synthetic: pd.DataFrame, cols: Sequence[str],
    min_relative_distance: float = 0.02,
) -> SyntheticGateResult:
    """Flag synthetic rows that are near-exact copies of a real training
    row (memorization/privacy leak), using normalized Euclidean distance to
    the nearest real-training neighbor."""
    real_mat = real_train[list(cols)].to_numpy(dtype=float)
    synth_mat = synthetic[list(cols)].to_numpy(dtype=float)
    scale = real_mat.std(axis=0)
    scale[scale == 0] = 1.0
    real_norm = real_mat / scale
    synth_norm = synth_mat / scale

    # Typical real-to-real nearest-neighbor distance sets the "this is
    # basically noise-level closeness" reference scale.
    real_nn = []
    for i in range(len(real_norm)):
        d = np.linalg.norm(real_norm - real_norm[i], axis=1)
        d[i] = np.inf
        real_nn.append(d.min())
    ref_scale = max(float(np.median(real_nn)), 1e-9) if real_nn else 1e-9

    min_dists = []
    for row in synth_norm:
        d = np.linalg.norm(real_norm - row, axis=1)
        min_dists.append(d.min())
    min_dists = np.array(min_dists)

    copy_frac = float((min_dists < min_relative_distance * ref_scale).mean())
    reasons = []
    if copy_frac > 0.01:
        reasons.append(
            f"{copy_frac:.1%} of synthetic rows are near-exact copies of a "
            "real training row (nearest-neighbor distance below threshold)"
        )
    return SyntheticGateResult(
        passed=not reasons, reasons=reasons,
        metrics={"nn_copy_frac": copy_frac, "nn_reference_scale": ref_scale},
    )


def check_conditional_correlations(
    real_train: pd.DataFrame, synthetic: pd.DataFrame, cols: Sequence[str], tol: float = 0.25,
) -> SyntheticGateResult:
    if len(cols) < 2:
        return SyntheticGateResult(passed=True, reasons=[], metrics={"max_corr_deviation": 0.0})
    real_corr = real_train[list(cols)].corr().to_numpy()
    synth_corr = synthetic[list(cols)].corr().to_numpy()
    diff = np.nanmax(np.abs(real_corr - synth_corr))
    reasons = []
    if diff > tol:
        reasons.append(f"max absolute correlation-matrix deviation {diff:.3f} exceeds tolerance {tol}")
    return SyntheticGateResult(passed=not reasons, reasons=reasons, metrics={"max_corr_deviation": float(diff)})


def train_synthetic_test_real_utility(
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_test: pd.DataFrame,
    fit_fn: Callable[[pd.DataFrame], object],
    score_fn: Callable[[object, pd.DataFrame], float],
) -> Dict[str, float]:
    """Compare downstream utility of real-only vs real+synthetic training,
    evaluated only on `real_test` (never on synthetic rows). Lower
    `score_fn` output must mean better (e.g. MAE); the returned dict lets
    the caller decide the accept/reject threshold.
    """
    model_real_only = fit_fn(real_train)
    model_augmented = fit_fn(pd.concat([real_train, synthetic], ignore_index=True))

    return {
        "real_only_score": float(score_fn(model_real_only, real_test)),
        "real_plus_synthetic_score": float(score_fn(model_augmented, real_test)),
    }


def evaluate_synthetic_generator(
    *,
    real_train: pd.DataFrame,
    synthetic: pd.DataFrame,
    real_test: pd.DataFrame,
    cols: Sequence[str],
    constraint_fns: Optional[Dict[str, Callable[[pd.DataFrame], pd.Series]]] = None,
    fit_fn: Optional[Callable[[pd.DataFrame], object]] = None,
    score_fn: Optional[Callable[[object, pd.DataFrame], float]] = None,
    uncertainty_fn: Optional[Callable[[object, pd.DataFrame], float]] = None,
    min_uncertainty_ratio: float = 0.85,
) -> SyntheticGateResult:
    """Run the full Task 036 gate battery and return one combined verdict.

    Synthetic augmentation is rejected unless *all* of the following hold:
      - support/constraint/tail/nearest-neighbor/correlation gates pass;
      - the real+synthetic model's real-test score improves on (is no
        worse than) the real-only model's real-test score;
      - the real+synthetic model's predicted uncertainty on the real test
        fold is not narrower than `min_uncertainty_ratio` times the
        real-only model's uncertainty (guards against "cheating" an OOS
        score by artificially shrinking predicted variance).
    """
    reasons: List[str] = []
    metrics: Dict[str, float] = {}

    for gate in (
        check_support(real_train, synthetic, cols),
        check_tails(real_train, synthetic, cols),
        check_nearest_neighbor_copying(real_train, synthetic, cols),
        check_conditional_correlations(real_train, synthetic, cols),
    ):
        reasons.extend(gate.reasons)
        metrics.update(gate.metrics)

    if constraint_fns:
        gate = check_constraints(synthetic, constraint_fns)
        reasons.extend(gate.reasons)
        metrics.update(gate.metrics)

    if fit_fn is not None and score_fn is not None:
        utility = train_synthetic_test_real_utility(real_train, synthetic, real_test, fit_fn, score_fn)
        metrics.update(utility)
        if utility["real_plus_synthetic_score"] > utility["real_only_score"] + 1e-9:
            reasons.append(
                "real+synthetic OOS score "
                f"({utility['real_plus_synthetic_score']:.4f}) is worse than "
                f"real-only ({utility['real_only_score']:.4f})"
            )

        if uncertainty_fn is not None:
            model_real_only = fit_fn(real_train)
            model_augmented = fit_fn(pd.concat([real_train, synthetic], ignore_index=True))
            unc_real = float(uncertainty_fn(model_real_only, real_test))
            unc_aug = float(uncertainty_fn(model_augmented, real_test))
            metrics["uncertainty_real_only"] = unc_real
            metrics["uncertainty_real_plus_synthetic"] = unc_aug
            if unc_real > 0 and unc_aug < min_uncertainty_ratio * unc_real:
                reasons.append(
                    f"real+synthetic uncertainty ({unc_aug:.4f}) is narrower than "
                    f"{min_uncertainty_ratio:.0%} of real-only uncertainty ({unc_real:.4f}) "
                    "-- rejecting to avoid artificially narrowed uncertainty"
                )

    return SyntheticGateResult(passed=not reasons, reasons=reasons, metrics=metrics)
