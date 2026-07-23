"""Time-series cross-validation utilities (purged / embargoed)."""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import BaseCrossValidator


class PurgedGroupTimeSeriesSplit(BaseCrossValidator):
    """Chronological CV with optional purge gap between train and test.

    Groups (e.g. game dates) keep all samples on the same day together.
    Suitable for StackingRegressor internal CV when dates are available.
    """

    def __init__(self, n_splits: int = 5, embargo: int = 0):
        self.n_splits = n_splits
        self.embargo = embargo

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        if groups is None:
            order = np.arange(n)
        else:
            groups = np.asarray(groups)
            uniq = np.unique(groups)
            order = np.argsort([np.where(groups == g)[0][0] for g in uniq])
            # map sample indices to sorted group order
            group_order = {g: i for i, g in enumerate(uniq[order])}
            order = np.argsort([group_order[g] for g in groups])

        indices = np.arange(n)[order]
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        current = 0
        for fold_size in fold_sizes:
            start, stop = current, current + fold_size
            test_idx = indices[start:stop]
            train_end = max(0, start - self.embargo)
            train_idx = indices[:train_end]
            current = stop
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx


class ChronologicalPartitionCV(BaseCrossValidator):
    """Date-ordered KFold partition for StackingRegressor internal CV.

    .. deprecated:: Task 049 (spread-accuracy-roadmap Stage 5)
        **Do not use this splitter in production.** Its train fold is
        ``concatenate(order[:cursor], order[cursor + fold_size:])`` — i.e.
        every fold except the first trains on rows *after* the held-out
        test block (``stack_cv_future_leakage`` in ``LEAK_REGISTRY.md``).
        It is kept only so the regression test that caught the leak
        (``tests/test_cv_past_only.py::test_chronological_partition_cv_is_documented_as_leaky``)
        has something concrete to assert against, and so any historical
        artifact fit with it can still be unpickled/inspected. Production
        code must use :class:`PastOnlyGroupCV` (via
        :class:`pipeline.oof.ManualOOFStacker`) instead, which never trains
        on a row at or after the validation block's earliest timestamp.

    StackingRegressor requires every training row to appear in exactly one
    held-out fold (``cross_val_predict`` partition constraint). Strict
    past-only purged splits cannot satisfy that on early folds, so this
    splitter sorts by ``game_date`` groups and assigns contiguous blocks
    as test folds (no shuffle). Train folds include non-test rows only —
    which, for every fold but the last, includes future rows.
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        if groups is not None:
            order = np.argsort(np.asarray(groups), kind="stable")
        else:
            order = np.arange(n)
        fold_sizes = np.full(self.n_splits, n // self.n_splits, dtype=int)
        fold_sizes[: n % self.n_splits] += 1
        cursor = 0
        for fs in fold_sizes:
            test_idx = order[cursor:cursor + fs]
            train_idx = np.concatenate([order[:cursor], order[cursor + fs:]])
            cursor += fs
            yield train_idx, test_idx


class PastOnlyGroupCV(BaseCrossValidator):
    """Expanding-window, group-aware CV where every fold satisfies
    ``max(train_time) < min(validation_time))`` (Task 049).

    Replaces :class:`ChronologicalPartitionCV` for any production
    stacking/meta-learning use. Groups (e.g. ``game_date``) are sorted
    ascending and partitioned into ``n_splits + 1`` contiguous blocks of
    *unique group values* (never splitting a single date across blocks).
    Block 0 is reserved as burn-in history that is never itself a
    validation fold (there is no strictly-earlier data to train it on).
    Fold ``i`` (``1..n_splits``) trains on every row in blocks ``[0, i)``
    and validates on block ``i``.

    Because block boundaries are snapped to unique group values, the
    training block's maximum group value is strictly less than the
    validation block's minimum group value by construction — this is
    verified by ``tests/test_cv_past_only.py``.

    Rows belonging to burn-in block 0 are never covered by any validation
    fold and must be dropped from any OOF meta-training set that consumes
    this splitter (Task 051 burn-in omission) — trying to "recover" them
    with a same-block or future-trained model would reintroduce the exact
    leak this class exists to remove (Rule 5/6).
    """

    def __init__(self, n_splits: int = 5):
        self.n_splits = n_splits

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

    def split(self, X, y=None, groups=None):
        n = len(X)
        if groups is None:
            group_vals = np.arange(n)
        else:
            group_vals = np.asarray(groups)

        uniq_sorted = np.sort(np.unique(group_vals))
        n_groups = len(uniq_sorted)
        n_blocks = min(self.n_splits + 1, n_groups) if n_groups > 0 else 0
        n_blocks = max(n_blocks, 2) if n_groups >= 2 else n_groups
        if n_blocks < 2:
            return

        block_sizes = np.full(n_blocks, n_groups // n_blocks, dtype=int)
        block_sizes[: n_groups % n_blocks] += 1
        boundaries = np.concatenate([[0], np.cumsum(block_sizes)])
        group_blocks = [
            set(uniq_sorted[boundaries[i]:boundaries[i + 1]].tolist())
            for i in range(n_blocks)
        ]

        produced = 0
        max_yield = self.n_splits
        for i in range(1, n_blocks):
            if produced >= max_yield:
                break
            train_groups = np.array(sorted(set().union(*group_blocks[:i])), dtype=group_vals.dtype)
            test_groups = np.array(sorted(group_blocks[i]), dtype=group_vals.dtype)
            train_idx = np.flatnonzero(np.isin(group_vals, train_groups))
            test_idx = np.flatnonzero(np.isin(group_vals, test_groups))
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            yield train_idx, test_idx
            produced += 1


def bind_cv_groups(cv_splitter, groups):
    """Fix group labels on a splitter (StackingRegressor cannot take groups= in older sklearn)."""
    if groups is None:
        return cv_splitter
    bound_groups = np.asarray(groups)
    inner = cv_splitter

    class _BoundSplit(BaseCrossValidator):
        def __init__(self, base, grp):
            self._base = base
            self._groups = grp
            if hasattr(base, "n_splits"):
                self.n_splits = base.n_splits
            if hasattr(base, "embargo"):
                self.embargo = base.embargo

        def get_n_splits(self, X=None, y=None, groups=None):
            return self._base.get_n_splits(X, y, self._groups)

        def split(self, X, y=None, groups=None):
            yield from self._base.split(X, y, self._groups)

    if isinstance(cv_splitter, (PurgedGroupTimeSeriesSplit, ChronologicalPartitionCV, PastOnlyGroupCV)):
        return _BoundSplit(inner, bound_groups)
    return cv_splitter
