"""Recency sample weights for multi-year meta fit."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.model import recency_sample_weights


def test_recency_weights_newer_games_heavier():
    dates = pd.date_range("2023-10-01", periods=5, freq="180D")
    w = recency_sample_weights(dates, half_life_seasons=1.0, season_days=180.0)
    assert len(w) == 5
    assert w[-1] == pytest.approx(1.0)
    assert w[-1] > w[0]
    assert w[-2] == pytest.approx(0.5, abs=0.05)