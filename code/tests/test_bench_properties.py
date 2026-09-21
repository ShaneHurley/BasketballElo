"""Task 10.3.2 — Hypothesis property-based tests (Epic 10.3, synthetic bench).

Three properties, all ``@pytest.mark.bench``, all run with
``@settings(max_examples=200, deadline=None, derandomize=True)``:

1. **PastOnlyGroupCV fold chronology** — for random group arrays (varying
   cardinality, heavy duplicates, tiny ``n_groups``, single-date blocks),
   every fold satisfies ``max(train_groups) < min(val_groups)``.
2. **Residual round-trip** — for random (margin, decision-spread) float
   pairs, ``residual_to_margin(margin_decision_residual(m, d), d) == m``
   within float tolerance (exact when the decision line is non-finite,
   since both functions pass the margin through untouched there).
3. **ManualOOFStacker** — for random feature matrices and group arrays,
   every OOF base prediction is produced by a model whose maximum training
   group is strictly earlier than the minimum group it predicts on.

Adversarial protocol: any failure Hypothesis discovers here must be frozen
into a permanent explicit example test (shrunk input hardcoded). If the
failure is a genuine pipeline bug (not a test bug), do NOT fix pipeline
source from this file — mark the frozen example ``xfail(strict=True)``
with a reason citing the module:line and report it.
"""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("hypothesis")
from hypothesis import given, settings, strategies as st
from hypothesis.extra.numpy import arrays
from sklearn.linear_model import LinearRegression

from pipeline.cv import PastOnlyGroupCV
from pipeline.market_targets import margin_decision_residual, residual_to_margin
from pipeline.oof import LeakageError, ManualOOFStacker

pytestmark = pytest.mark.bench


# ---------------------------------------------------------------------------
# Strategies
# ---------------------------------------------------------------------------

@st.composite
def _group_arrays(draw, min_size=0, max_size=200, max_pool=30):
    """Random integer 'game date' group arrays.

    ``pool`` caps the number of distinct date values, so draws cover
    high-cardinality arrays (pool >= n → all unique), heavy duplicates,
    tiny n_groups (pool 2-3), and single-date blocks (pool == 1). Row
    order is arbitrary — groups are NOT sorted by row position.
    """
    pool = draw(st.integers(1, max_pool))
    n = draw(st.integers(min_size, max_size))
    vals = draw(st.lists(st.integers(0, pool - 1), min_size=n, max_size=n))
    return np.array(vals, dtype=np.int64)


# Magnitude-capped finite floats: |m|, |d| <= 1e150 keeps m + d far from
# float64 overflow (~1.8e308) while still spanning an enormous dynamic
# range (subnormal-adjacent values near 0 included).
_finite_margin = st.floats(
    min_value=-1e150, max_value=1e150, allow_nan=False, allow_infinity=False
)
_decision_spread = st.one_of(
    st.floats(min_value=-1e150, max_value=1e150, allow_nan=False, allow_infinity=False),
    st.sampled_from([np.nan, np.inf, -np.inf]),  # exercise the passthrough mask
)


@st.composite
def _stacker_dataset(draw):
    """Random (X, y, groups, n_splits) for ManualOOFStacker.

    Column 0 of X mirrors the group value as a float so a sentinel base
    learner can observe exactly which rows it was fit on vs asked to
    predict. pool == 1 draws exercise the single-date-block refusal path.
    """
    n = draw(st.integers(2, 120))
    k = draw(st.integers(1, 5))
    pool = draw(st.integers(1, 20))
    groups = np.array(
        draw(st.lists(st.integers(0, pool - 1), min_size=n, max_size=n)),
        dtype=np.int64,
    )
    features = draw(
        arrays(
            np.float64,
            (n, k),
            elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
        )
    )
    X = np.column_stack([groups.astype(np.float64), features])
    y = draw(
        arrays(
            np.float64,
            n,
            elements=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
        )
    )
    n_splits = draw(st.integers(1, 6))
    return X, y, groups, n_splits


class _ChronologySentinel:
    """Base learner that records (max train group, min predict group).

    ``log`` is a class attribute, so the per-fold ``copy.deepcopy`` clones
    inside ``ManualOOFStacker.fit`` all append to the same shared list.
    The group value is read from column 0 of X (mirrored there by
    ``_stacker_dataset``). Predictions are constant zeros — the property
    under test is *when* the model was trained, not what it predicts.
    """

    log: list = []

    def fit(self, X, y, sample_weight=None):
        self._max_train_group = float(np.max(X[:, 0]))
        return self

    def predict(self, X):
        type(self).log.append((self._max_train_group, float(np.min(X[:, 0]))))
        return np.zeros(len(X), dtype=float)


# ---------------------------------------------------------------------------
# Property 1 — PastOnlyGroupCV fold chronology
# ---------------------------------------------------------------------------

@given(groups=_group_arrays(), n_splits=st.integers(1, 8))
@settings(max_examples=200, deadline=None, derandomize=True)
def test_past_only_group_cv_fold_chronology(groups, n_splits):
    cv = PastOnlyGroupCV(n_splits=n_splits)
    X = np.zeros((len(groups), 1), dtype=float)
    folds = list(cv.split(X, groups=groups))

    n_unique = len(np.unique(groups))
    if n_unique < 2:
        # No strictly-earlier training block exists → must refuse to split
        # rather than fall back to a leaky fold.
        assert folds == []
        return

    assert 1 <= len(folds) <= n_splits
    for train_idx, val_idx in folds:
        assert len(train_idx) > 0 and len(val_idx) > 0
        # Core invariant: max(train_groups) < min(val_groups).
        assert groups[train_idx].max() < groups[val_idx].min()
        # A single date is never split across train/val.
        assert set(groups[train_idx].tolist()).isdisjoint(groups[val_idx].tolist())

    # Burn-in: the earliest block of dates is never a validation fold.
    val_groups = np.concatenate([groups[val_idx] for _, val_idx in folds])
    assert val_groups.min() > groups.min()


# ---------------------------------------------------------------------------
# Property 2 — residual ↔ margin round-trip
# ---------------------------------------------------------------------------

@given(m=_finite_margin, d=_decision_spread)
@settings(max_examples=200, deadline=None, derandomize=True)
def test_residual_margin_round_trip(m, d):
    resid = margin_decision_residual(np.array([m]), np.array([d]))
    recovered = residual_to_margin(resid, np.array([d]))[0]
    if np.isfinite(d):
        # IEEE-754 error bound for fl(fl(m + d) - d): a few ULPs scaled by
        # the largest operand involved (catastrophic cancellation when
        # |d| >> |m| is inherent to floating point, not a pipeline bug).
        scale = max(1.0, abs(m), abs(d), abs(m + d))
        tol = 16 * np.finfo(float).eps * scale
        assert abs(recovered - m) <= tol
    else:
        # Non-finite decision line → both functions pass the value through
        # untouched (mask is False), so the round-trip is exact.
        assert recovered == m


# ---------------------------------------------------------------------------
# Property 3 — ManualOOFStacker OOF predictions are past-only
# ---------------------------------------------------------------------------

@given(dataset=_stacker_dataset())
@settings(max_examples=200, deadline=None, derandomize=True)
def test_manual_oof_stacker_oof_never_trained_on_future_rows(dataset):
    X, y, groups, n_splits = dataset
    stacker = ManualOOFStacker(
        estimators=[("sentinel", _ChronologySentinel())],
        final_estimator=LinearRegression(),
        n_splits=n_splits,
    )

    sentinel_log: list = []
    _ChronologySentinel.log = sentinel_log
    try:
        n_unique = len(np.unique(groups))
        if n_unique < 2:
            # Single-date block: must raise rather than train on an unsafe CV.
            with pytest.raises(LeakageError):
                stacker.fit(X, y, groups=groups)
            return

        stacker.fit(X, y, groups=groups)

        # Every OOF prediction logged during fit came from a fold clone whose
        # max training group is strictly earlier than the min predicted group.
        assert len(sentinel_log) >= 1
        for max_train_group, min_val_group in sentinel_log:
            assert max_train_group < min_val_group

        # The stacker's own fold lineage bookkeeping agrees.
        assert len(stacker.fold_lineage_) >= 1
        for fold in stacker.fold_lineage_:
            assert fold["train_max_t"] < fold["val_min_t"]

        # Burn-in rows are dropped from meta-training, never predicted by a
        # future-trained model; accounting covers every row exactly once.
        assert stacker.n_burn_in_dropped_ > 0
        assert stacker.oof_rows_used_ + stacker.n_burn_in_dropped_ == len(X)
    finally:
        _ChronologySentinel.log = []


# ---------------------------------------------------------------------------
# Deterministic edge-case examples (document and pin the boundary behavior
# the property tests explore; add frozen Hypothesis failures here per the
# adversarial protocol if any are ever discovered).
# ---------------------------------------------------------------------------

def test_single_date_block_produces_zero_folds():
    groups = np.full(50, 7, dtype=np.int64)
    folds = list(PastOnlyGroupCV(n_splits=5).split(np.zeros((50, 1)), groups=groups))
    assert folds == []


def test_two_groups_unsorted_rows_single_chronological_fold():
    # Two distinct dates, rows interleaved out of order.
    groups = np.array([9, 3, 9, 3], dtype=np.int64)
    folds = list(PastOnlyGroupCV(n_splits=5).split(np.zeros((4, 1)), groups=groups))
    assert len(folds) == 1
    train_idx, val_idx = folds[0]
    assert groups[train_idx].max() < groups[val_idx].min()
    assert set(train_idx.tolist()) == {1, 3}
    assert set(val_idx.tolist()) == {0, 2}


def test_round_trip_exact_for_non_finite_decision():
    m = np.array([-25.0, 0.0, 13.5])
    for d in (np.nan, np.inf, -np.inf):
        dd = np.full(3, d)
        recovered = residual_to_margin(margin_decision_residual(m, dd), dd)
        assert np.array_equal(recovered, m)


def test_round_trip_vectorized_typical_nba_ranges():
    rng = np.random.RandomState(0)
    m = rng.uniform(-60.0, 60.0, size=1000)
    d = rng.uniform(-30.0, 30.0, size=1000)
    recovered = residual_to_margin(margin_decision_residual(m, d), d)
    assert np.allclose(recovered, m, rtol=1e-12, atol=1e-12)


def test_stacker_refuses_single_date_block_instead_of_leaking():
    n = 30
    groups = np.full(n, 4, dtype=np.int64)
    X = np.column_stack([groups.astype(float), np.zeros(n)])
    stacker = ManualOOFStacker(
        estimators=[("sentinel", _ChronologySentinel())],
        final_estimator=LinearRegression(),
        n_splits=5,
    )
    with pytest.raises(LeakageError):
        stacker.fit(X, np.zeros(n), groups=groups)
