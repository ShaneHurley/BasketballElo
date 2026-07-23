"""SHAP-guided feature pruning with walk-forward stability gate.

Workflow per market (spread / ML / total):
  1. Fit a baseline tree model
  2. Rank features by mean |SHAP|
  3. Drop bottom ~30%
  4. Retrain
  5. Keep only features in the top 50% of importance in >80% of folds
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd


def _mean_abs_shap(model, X: pd.DataFrame) -> pd.Series:
    """Return mean |SHAP| per feature. Falls back to tree feature_importances_."""
    try:
        import shap
        explainer = shap.TreeExplainer(model)
        values = explainer.shap_values(X)
        if isinstance(values, list):
            values = values[1] if len(values) > 1 else values[0]
        vals = np.abs(np.asarray(values)).mean(axis=0)
        return pd.Series(vals, index=list(X.columns))
    except Exception:
        if hasattr(model, "feature_importances_"):
            return pd.Series(model.feature_importances_, index=list(X.columns))
        raise RuntimeError("Could not compute SHAP or feature_importances_")


def prune_bottom_fraction(importance: pd.Series, drop_frac: float = 0.30) -> list[str]:
    """Keep top (1 - drop_frac) features by importance."""
    if importance.empty:
        return []
    keep_n = max(1, int(np.ceil(len(importance) * (1.0 - drop_frac))))
    return list(importance.sort_values(ascending=False).head(keep_n).index)


def stable_top_features(
    fold_rankings: Iterable[pd.Series],
    *,
    top_frac: float = 0.50,
    min_fold_frac: float = 0.80,
) -> list[str]:
    """Keep features that appear in the top_frac of importance in >= min_fold_frac folds."""
    fold_sets = []
    for imp in fold_rankings:
        if imp is None or len(imp) == 0:
            continue
        k = max(1, int(np.ceil(len(imp) * top_frac)))
        fold_sets.append(set(imp.sort_values(ascending=False).head(k).index))
    if not fold_sets:
        return []
    n = len(fold_sets)
    threshold = min_fold_frac * n
    counts: dict[str, int] = {}
    for s in fold_sets:
        for f in s:
            counts[f] = counts.get(f, 0) + 1
    return [f for f, c in counts.items() if c >= threshold]


def shap_prune_loop(
    model_factory,
    X: pd.DataFrame,
    y,
    *,
    drop_frac: float = 0.30,
    n_rounds: int = 2,
    target_size: tuple[int, int] = (40, 60),
    fold_importances: list[pd.Series] | None = None,
) -> list[str]:
    """Iterative SHAP prune toward target feature count, with optional stability gate.

    model_factory(X_cols) -> fitted model with predict / feature_importances_ or TreeExplainer support.
    """
    cols = list(X.columns)
    for _ in range(max(1, n_rounds)):
        if len(cols) <= target_size[0]:
            break
        model = model_factory(X[cols])
        imp = _mean_abs_shap(model, X[cols])
        cols = prune_bottom_fraction(imp, drop_frac=drop_frac)
        if fold_importances:
            stable = stable_top_features(fold_importances + [imp])
            if stable:
                cols = [c for c in cols if c in stable] or cols
        if len(cols) <= target_size[1]:
            break
    # Cap at upper target if still too many
    if len(cols) > target_size[1]:
        model = model_factory(X[cols])
        imp = _mean_abs_shap(model, X[cols])
        cols = list(imp.sort_values(ascending=False).head(target_size[1]).index)
    return cols
