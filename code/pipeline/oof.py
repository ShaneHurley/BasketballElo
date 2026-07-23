"""Manual past-only OOF stacking (Tasks 049, 050, 051).

Replaces sklearn's ``StackingRegressor`` + ``ChronologicalPartitionCV`` for
production meta-model fitting. sklearn's ``StackingRegressor`` calls
``cross_val_predict`` internally, which *requires* every training row to be
covered by exactly one held-out fold (a partition). A strictly past-only
splitter (``PastOnlyGroupCV``) cannot satisfy that for its earliest block —
there is no strictly-earlier data to train on — so sklearn's mechanism
cannot be used at all for a leak-free stack; this module implements the OOF
generation manually instead:

- Every held-out (OOF) base-learner prediction comes from a base learner
  trained only on strictly earlier rows (Task 051; ``max(train_time) <
  min(validation_time)`` for every fold, enforced by ``PastOnlyGroupCV``
  and re-checked here defensively).
- Rows in the first (burn-in) block are never covered by any fold and are
  *dropped* from the meta-learner's training set rather than imputed or
  predicted by a future-trained model (Rule 5/6; Task 051's explicit
  "omit early burn-in rows" allowance).
- Every fitted component (each base estimator's final refit, and the
  meta/final estimator) is instrumented with ``fit_max_timestamp`` — the
  maximum group/timestamp value it was trained on (Task 050/058).
- Fold-local preprocessing: callers are expected to fit any scaler/imputer
  *inside* each fold (see ``fit_transform_fn`` hook) rather than on the
  full frame beforehand; this module fits base learners fold-locally by
  construction, and exposes the same fold boundaries for callers that also
  need to cross-fit a feature (see ``cross_fit_derived_feature``).
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field

import numpy as np

from pipeline.cv import PastOnlyGroupCV


class LeakageError(RuntimeError):
    """Raised when a fold or lineage check proves a leak is present."""


def _check_fold_chronology(groups, train_idx, val_idx):
    if len(train_idx) == 0 or len(val_idx) == 0:
        return
    max_train_t = groups[train_idx].max()
    min_val_t = groups[val_idx].min()
    if not (max_train_t < min_val_t):
        raise LeakageError(
            "Fold violates max(train_time) < min(validation_time): "
            f"{max_train_t!r} >= {min_val_t!r}"
        )


@dataclass
class ManualOOFStacker:
    """Past-only manual OOF stacking regressor.

    Parameters
    ----------
    estimators : list[(name, estimator)]
        Base learners with sklearn-like ``fit``/``predict``.
    final_estimator : sklearn-like estimator
        Meta-learner fit on the OOF base predictions.
    n_splits : int
        Number of past-only folds requested from ``PastOnlyGroupCV``.
    cv : splitter or None
        Defaults to ``PastOnlyGroupCV(n_splits=n_splits)``.
    """

    estimators: list
    final_estimator: object
    n_splits: int = 5
    cv: object = None

    def __post_init__(self):
        if self.cv is None:
            self.cv = PastOnlyGroupCV(n_splits=self.n_splits)
        self.estimators_: list = []
        self.final_estimator_ = None
        self.fit_max_timestamp = None
        self.oof_rows_used_ = 0
        self.n_burn_in_dropped_ = 0
        self.fold_lineage_: list = []
        self.fitted = False

    def fit(self, X, y, groups=None):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y, dtype=float)
        n = len(X)
        groups = np.arange(n) if groups is None else np.asarray(groups)
        if len(groups) != n:
            raise ValueError("groups must be the same length as X")

        n_est = len(self.estimators)
        oof = np.full((n, n_est), np.nan, dtype=float)
        covered = np.zeros(n, dtype=bool)

        folds = list(self.cv.split(X, y, groups))
        if not folds:
            raise LeakageError(
                "ManualOOFStacker: PastOnlyGroupCV produced zero folds — "
                "not enough distinct groups/timestamps to guarantee a "
                "past-only split. Refusing to fall back to an unsafe CV."
            )

        for train_idx, val_idx in folds:
            _check_fold_chronology(groups, train_idx, val_idx)
            self.fold_lineage_.append({
                "train_max_t": groups[train_idx].max(),
                "val_min_t": groups[val_idx].min(),
                "n_train": len(train_idx),
                "n_val": len(val_idx),
            })
            for j, (_, est) in enumerate(self.estimators):
                fold_est = copy.deepcopy(est)
                fold_est.fit(X[train_idx], y[train_idx])
                oof[val_idx, j] = fold_est.predict(X[val_idx])
            covered[val_idx] = True

        self.oof_rows_used_ = int(covered.sum())
        self.n_burn_in_dropped_ = int((~covered).sum())
        if self.oof_rows_used_ == 0:
            raise LeakageError(
                "ManualOOFStacker: no row was ever validated past-only; "
                "cannot train the meta-learner without leaking."
            )
        if np.isnan(oof[covered]).any():
            raise LeakageError(
                "ManualOOFStacker: covered rows have NaN OOF predictions — "
                "internal fold bookkeeping bug, refusing to train on gaps."
            )

        meta_X = oof[covered]
        meta_y = y[covered]
        self.final_estimator_ = copy.deepcopy(self.final_estimator)
        self.final_estimator_.fit(meta_X, meta_y)

        overall_max_t = groups.max()
        self.estimators_ = []
        for name, est in self.estimators:
            fitted = copy.deepcopy(est)
            fitted.fit(X, y)
            setattr(fitted, "fit_max_timestamp", overall_max_t)
            self.estimators_.append((name, fitted))
        setattr(self.final_estimator_, "fit_max_timestamp", overall_max_t)
        self.fit_max_timestamp = overall_max_t
        self.fitted = True
        return self

    def _base_predictions(self, X):
        X = np.asarray(X, dtype=float)
        return np.column_stack([est.predict(X) for _, est in self.estimators_])

    def predict(self, X):
        if not self.fitted:
            raise RuntimeError("ManualOOFStacker must be fit() before predict().")
        return self.final_estimator_.predict(self._base_predictions(X))


def cross_fit_derived_feature(fit_fn, predict_fn, X_feat, y_target, groups, cv=None, n_splits=5):
    """Past-only cross-fit a single derived feature (Task 052).

    Used for target-fitted engine outputs such as ``elo_stack_pred`` that
    are *features* fed into a downstream stack rather than base learners
    inside it. Returns ``(oof_values, covered_mask, fit_final)`` where
    ``fit_final`` is the artifact refit on all rows (for predict-time use,
    with ``fit_max_timestamp`` set) and ``oof_values``/``covered_mask``
    give a leak-free training column (burn-in rows are NaN/uncovered and
    must be dropped from meta-training, matching ManualOOFStacker).

    Parameters
    ----------
    fit_fn : callable(X_train, y_train) -> artifact
    predict_fn : callable(artifact, X_apply) -> np.ndarray
    """
    X_feat = np.asarray(X_feat, dtype=float)
    y_target = np.asarray(y_target, dtype=float)
    n = len(X_feat)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    cv = cv or PastOnlyGroupCV(n_splits=n_splits)

    oof = np.full(n, np.nan, dtype=float)
    covered = np.zeros(n, dtype=bool)
    folds = list(cv.split(X_feat, y_target, groups))
    if not folds:
        raise LeakageError(
            "cross_fit_derived_feature: PastOnlyGroupCV produced zero folds."
        )
    for train_idx, val_idx in folds:
        _check_fold_chronology(groups, train_idx, val_idx)
        artifact = fit_fn(X_feat[train_idx], y_target[train_idx])
        oof[val_idx] = predict_fn(artifact, X_feat[val_idx])
        covered[val_idx] = True

    fit_final = fit_fn(X_feat, y_target)
    setattr(fit_final, "fit_max_timestamp", groups.max())
    return oof, covered, fit_final
