"""Task 049 — remove production use of ChronologicalPartitionCV for
stacking; every fold of its replacement must satisfy
max(train_time) < min(validation_time).
"""
import numpy as np
import pandas as pd
import pytest

from pipeline.cv import ChronologicalPartitionCV, PastOnlyGroupCV, bind_cv_groups


class TestChronologicalPartitionCVIsDocumentedAsLeaky:
    def test_chronological_partition_cv_trains_on_future_rows(self):
        """Regression test for LEAK_REGISTRY.md::stack_cv_future_leakage.

        This *proves* the leak exists so nobody accidentally "fixes" this
        splitter into looking safe without noticing — it must simply not
        be used for stacking OOF anymore (see PastOnlyGroupCV below and
        pipeline.oof.ManualOOFStacker).
        """
        n = 100
        groups = np.arange(n)  # each row its own "date", strictly increasing
        cv = ChronologicalPartitionCV(n_splits=5)
        found_future_leak = False
        for train_idx, test_idx in cv.split(np.zeros((n, 1)), groups=groups):
            if len(train_idx) == 0 or len(test_idx) == 0:
                continue
            if train_idx.max() > test_idx.min():
                found_future_leak = True
        assert found_future_leak, (
            "Expected ChronologicalPartitionCV to leak future rows into an "
            "earlier fold's training set (that is the confirmed defect); "
            "if this now fails, the splitter changed behavior and "
            "LEAK_REGISTRY.md must be re-audited, not this test relaxed."
        )


class TestPastOnlyGroupCV:
    def test_every_fold_satisfies_max_train_lt_min_val(self):
        n = 200
        groups = np.arange(n)
        cv = PastOnlyGroupCV(n_splits=5)
        folds = list(cv.split(np.zeros((n, 1)), groups=groups))
        assert len(folds) >= 1
        for train_idx, val_idx in folds:
            assert train_idx.max() < val_idx.min()

    def test_grouped_dates_never_split_within_a_block(self):
        # Two rows share every date; a leaky splitter could put one row of
        # a date in train and the other in val for the *same* date.
        dates = pd.date_range("2025-01-01", periods=50, freq="D").repeat(2)
        groups = dates.values
        cv = PastOnlyGroupCV(n_splits=4)
        folds = list(cv.split(np.zeros((len(groups), 1)), groups=groups))
        assert len(folds) >= 1
        for train_idx, val_idx in folds:
            train_dates = set(groups[train_idx])
            val_dates = set(groups[val_idx])
            assert train_dates.isdisjoint(val_dates)
            assert max(train_dates) < min(val_dates)

    def test_burn_in_block_never_appears_as_validation(self):
        n = 60
        groups = np.arange(n)
        cv = PastOnlyGroupCV(n_splits=5)
        val_indices_seen = set()
        for _, val_idx in cv.split(np.zeros((n, 1)), groups=groups):
            val_indices_seen.update(val_idx.tolist())
        # Roughly the first block (~1/6 of rows) is never validated on.
        assert 0 not in val_indices_seen
        assert min(val_indices_seen) > 0

    def test_no_groups_falls_back_to_row_order(self):
        n = 30
        cv = PastOnlyGroupCV(n_splits=3)
        folds = list(cv.split(np.zeros((n, 1))))
        for train_idx, val_idx in folds:
            assert train_idx.max() < val_idx.min()

    def test_bind_cv_groups_supports_past_only_group_cv(self):
        n = 40
        groups = np.arange(n)
        cv = PastOnlyGroupCV(n_splits=4)
        bound = bind_cv_groups(cv, groups)
        folds_direct = list(cv.split(np.zeros((n, 1)), groups=groups))
        folds_bound = list(bound.split(np.zeros((n, 1))))
        assert len(folds_direct) == len(folds_bound)
        for (t1, v1), (t2, v2) in zip(folds_direct, folds_bound):
            assert np.array_equal(t1, t2)
            assert np.array_equal(v1, v2)
