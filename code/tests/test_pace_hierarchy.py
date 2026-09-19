"""Tests for hierarchical pace forecasting."""
from __future__ import annotations

import numpy as np

from pipeline.pace_hierarchy import HierarchicalPaceModel


def test_hierarchical_pace_near_league_cold_start():
    m = HierarchicalPaceModel()
    fc = m.predict("BOS", "NYK")
    assert 90.0 <= fc.mean <= 110.0
    assert fc.variance >= 1.0
    assert fc.q10 <= fc.q50 <= fc.q90


def test_hierarchical_pace_learns_fast_teams():
    m = HierarchicalPaceModel()
    for _ in range(25):
        m.update("BOS", "NYK", 108.0, 107.0)
        m.update("BOS", "CHI", 109.0, 106.0)
    for _ in range(25):
        m.update("SAS", "UTA", 92.0, 91.0)
    fast = m.predict("BOS", "NYK")
    slow = m.predict("SAS", "UTA")
    assert fast.mean > slow.mean + 2.0


def test_pace_forecast_features_keys():
    m = HierarchicalPaceModel()
    fc = m.predict("BOS", "NYK")
    d = fc.as_feature_dict()
    for k in ("pace_mean", "pace_var", "pace_std", "pace_baseline", "pace_eff_n"):
        assert k in d
