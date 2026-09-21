"""Additional green bench invariants (sign, calibration, promotion).

These are not expected-fails. They pin fail-closed and ranking contracts that
the sign/ranking pass left implicit.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.calibration_registry import (
    BACKTEST_CALIBRATION_TARGETS,
    chronological_game_id_partition,
)
from pipeline.promotion_gates import clv_promotion_allowed, tip_proxy_roi_blocked
from pipeline.venn_abers import scores_to_interval


@pytest.mark.bench
def test_empty_calibration_partition_does_not_raise():
    """No games → empty id arrays, not a min_slice_n error."""
    df = pd.DataFrame(columns=["GAME_ID", "game_date"])
    part = chronological_game_id_partition(df)
    assert set(part) == set(BACKTEST_CALIBRATION_TARGETS)
    assert all(len(ids) == 0 for ids in part.values())


@pytest.mark.bench
def test_same_label_venn_abers_stays_in_unit_interval():
    """All-negative labels (early-return path) after the min-sample floor."""
    scores = np.linspace(0.1, 0.9, 10)
    labels = np.zeros(10, dtype=int)
    p0, p1, opt, width = scores_to_interval(scores, labels, np.array([0.4, 0.8]))
    for arr in (p0, p1, opt):
        assert np.all(np.isfinite(arr))
        assert np.all((0.0 <= arr) & (arr <= 1.0))
    assert np.all(width >= 0.0)


@pytest.mark.bench
def test_promotion_gates_block_tip_proxy_and_tiny_clv():
    """ROI promotion stays blocked without promotion_eligible + adequate N."""
    assert clv_promotion_allowed({"promotion_eligible": False}, 500) is False
    assert clv_promotion_allowed({"promotion_eligible": True}, 50) is False
    assert clv_promotion_allowed({"promotion_eligible": True}, 200) is True
    assert tip_proxy_roi_blocked({"promotion_eligible": False, "quote_source": "tip_proxy"}) is True
    assert tip_proxy_roi_blocked({"promotion_eligible": True, "quote_source": "pinnacle"}) is False
