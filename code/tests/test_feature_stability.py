"""Multi-year SHAP / importance stability: keep features helping in ≥2 of last 3 seasons."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.shap_prune import (
    features_helping_in_seasons,
    multi_year_shap_stable_features,
    stable_top_features,
)


def test_features_helping_in_seasons_requires_two_of_three():
    a = pd.Series({"f1": 1.0, "f2": 0.9, "f3": 0.1, "f4": 0.05})
    b = pd.Series({"f1": 0.8, "f2": 0.05, "f3": 0.7, "f4": 0.02})
    c = pd.Series({"f1": 0.9, "f2": 0.1, "f3": 0.85, "f4": 0.01})
    # top 50% of 4 = 2 features each season
    kept = features_helping_in_seasons(
        {"2022": a, "2023": b, "2024": c},
        min_seasons=2,
        last_n=3,
        top_frac=0.50,
    )
    assert "f1" in kept  # top in all three
    assert "f3" in kept  # top in b and c
    assert "f4" not in kept


def test_features_helping_drops_one_season_fluke():
    a = pd.Series({"good": 1.0, "fluke": 0.01, "mid": 0.5})
    b = pd.Series({"good": 0.9, "fluke": 0.02, "mid": 0.4})
    c = pd.Series({"good": 0.2, "fluke": 1.0, "mid": 0.3})  # fluke only here
    kept = features_helping_in_seasons(
        [a, b, c],
        min_seasons=2,
        last_n=3,
        top_frac=0.50,
    )
    assert "good" in kept or "mid" in kept
    # fluke only tops one season → should not survive min_seasons=2
    # (top_frac 0.5 of 3 → 2 features; fluke is #1 only in c)
    assert "fluke" not in kept or kept.count("fluke") == 0


class _FakeTree:
    def __init__(self, importances, cols):
        self.feature_importances_ = np.asarray(importances, dtype=float)
        self._cols = cols

    def predict(self, X):
        return np.zeros(len(X))


def test_multi_year_shap_stable_features_intersection():
    rng = np.random.RandomState(0)
    cols = ["a", "b", "c", "d"]
    frames = {}
    for i, season in enumerate(["2022", "2023", "2024"]):
        X = pd.DataFrame(rng.normal(size=(40, 4)), columns=cols)
        y = X["a"] + 0.1 * rng.normal(size=40)
        frames[season] = (X, y)

    def factory(X, y=None):
        # Always rank a > b > c > d
        return _FakeTree([0.5, 0.3, 0.15, 0.05], list(X.columns))

    kept = multi_year_shap_stable_features(
        factory, frames, feature_cols=cols, min_seasons=2, last_n=3, top_frac=0.50,
    )
    assert "a" in kept
    assert "b" in kept
    assert "d" not in kept
