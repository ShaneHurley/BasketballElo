"""Epic 10.6 / P0.4 & Epic 11.4 — Frozen Elo calibration interval.

GitHub: P0.4 · Registry ID: ``bench_frozen_elo_interval``

``WalkForwardEloCalibrator.update_residuals`` existed but was never called
anywhere in the live/backtest path, so ``predict_interval`` always returned
the hardcoded default ``half = 12.0`` width forever, regardless of how
well- or poorly-calibrated the model actually was. The fix wires
``update_residuals`` into ``pipeline/backtest.py``'s per-season
walk-forward loop, once per graded season, from that season's own
already-realized ``|ELO_MARGIN_CALIBRATED − ACTUAL_MARGIN|`` residuals.
"""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.elo_calibration import WalkForwardEloCalibrator


@pytest.mark.bench
def test_p0_4_default_interval_width_is_24():
    """Witness: before any residuals are observed, width is the hardcoded default."""
    cal = WalkForwardEloCalibrator()
    lo, hi, width = cal.predict_interval({"elo_margin": 3.0})
    assert width == pytest.approx(24.0)
    assert lo == pytest.approx(3.0 - 12.0)
    assert hi == pytest.approx(3.0 + 12.0)


@pytest.mark.bench
def test_p0_4_update_residuals_changes_interval_width():
    """Acceptance: after update_residuals, width must differ from the ±12 default."""
    cal = WalkForwardEloCalibrator()
    _, _, default_width = cal.predict_interval({"elo_margin": 0.0})
    assert default_width == pytest.approx(24.0)

    cal.update_residuals([5.0] * 30)
    _, _, new_width = cal.predict_interval({"elo_margin": 0.0})

    assert new_width != pytest.approx(default_width)
    # With all-constant residuals of 5.0, the 90th-percentile quantile must be 5.0.
    assert new_width == pytest.approx(10.0)


@pytest.mark.bench
def test_p0_4_update_residuals_requires_min_sample():
    """Fewer than 20 residuals must not overwrite the interval (documented threshold)."""
    cal = WalkForwardEloCalibrator()
    cal.update_residuals([100.0] * 5)
    _, _, width = cal.predict_interval({"elo_margin": 0.0})
    assert width == pytest.approx(24.0)


@pytest.mark.bench
def test_p0_4_backtest_wires_update_residuals_after_graded_season():
    """Static check: pipeline/backtest.py calls update_residuals in the walk-forward loop."""
    import inspect

    from pipeline import backtest

    src = inspect.getsource(backtest)
    assert "elo_calibrator.update_residuals(" in src


@pytest.mark.bench
def test_p0_4_interval_reflects_larger_residuals_with_wider_band():
    cal_tight = WalkForwardEloCalibrator()
    cal_tight.update_residuals(list(np.full(30, 2.0)))
    _, _, width_tight = cal_tight.predict_interval({"elo_margin": 0.0})

    cal_wide = WalkForwardEloCalibrator()
    cal_wide.update_residuals(list(np.full(30, 20.0)))
    _, _, width_wide = cal_wide.predict_interval({"elo_margin": 0.0})

    assert width_wide > width_tight
