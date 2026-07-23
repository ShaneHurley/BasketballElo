"""Task 053 — each target gets exactly one calibration path; the same row
slice must not calibrate more than one target.

Regression test for the confirmed defect: ``pipeline/backtest.py`` passes
the identical ``calib_features`` tail slice to the Elo calibrator, the
MetaScore isotonic calibrator, and the MetaWin isotonic/ATS calibrators.
"""
import numpy as np
import pytest

import pandas as pd

from pipeline.calibration_registry import (
    BACKTEST_CALIBRATION_TARGETS,
    CalibrationRoleError,
    CalibrationSliceRegistry,
    chronological_game_id_partition,
    slice_by_game_ids,
)


class TestCalibrationSliceRegistry:
    def test_distinct_slices_per_target_is_allowed(self):
        reg = CalibrationSliceRegistry()
        reg.register("engine_margin", np.arange(0, 20))
        reg.register("final_margin", np.arange(20, 40))
        reg.register("cover_probability", np.arange(40, 60))
        reg.register("ml_win_probability", np.arange(60, 80))
        reg.register("interval_width", np.arange(80, 100))
        assert set(reg.registered_targets()) == {
            "engine_margin", "final_margin", "cover_probability",
            "ml_win_probability", "interval_width",
        }
        reg.assert_all_targets_registered()

    def test_reusing_the_same_slice_for_a_second_target_raises(self):
        """This is the exact production defect: the same tail slice used
        for Elo isotonic, MetaScore isotonic, MetaWin isotonic, and ATS
        isotonic calibration."""
        shared_slice = np.arange(100, 130)
        reg = CalibrationSliceRegistry()
        reg.register("engine_margin", shared_slice)  # e.g. Elo isotonic
        with pytest.raises(CalibrationRoleError):
            reg.register("cover_probability", shared_slice)  # e.g. ATS isotonic

    def test_reusing_the_same_slice_reordered_still_detected(self):
        reg = CalibrationSliceRegistry()
        reg.register("final_margin", np.array([5, 1, 3, 2, 4]))
        with pytest.raises(CalibrationRoleError):
            reg.register("ml_win_probability", np.array([1, 2, 3, 4, 5]))

    def test_registering_the_same_target_twice_raises(self):
        reg = CalibrationSliceRegistry()
        reg.register("final_margin", np.arange(10))
        with pytest.raises(CalibrationRoleError):
            reg.register("final_margin", np.arange(10, 20))

    def test_missing_target_detected(self):
        reg = CalibrationSliceRegistry()
        reg.register("engine_margin", np.arange(10))
        with pytest.raises(CalibrationRoleError):
            reg.assert_all_targets_registered()


class TestBacktestDisjointCalibrationPartition:
    """Mirrors the live `pipeline/backtest.py` wiring: partition the
    calibration-tail GAME_IDs into five disjoint chronological blocks, register
    each, and prove no two targets share a fingerprint."""

    def _calib_frame(self, n=50):
        return pd.DataFrame({
            "GAME_ID": np.arange(n),
            "game_date": pd.date_range("2025-01-01", periods=n, freq="D"),
            "elo_margin": np.linspace(-10, 10, n),
        })

    def test_partition_is_mutually_disjoint_and_covers_all_games(self):
        calib = self._calib_frame(50)
        parts = chronological_game_id_partition(calib, targets=BACKTEST_CALIBRATION_TARGETS)
        assert set(parts) == set(BACKTEST_CALIBRATION_TARGETS)
        all_ids = []
        for ids in parts.values():
            all_ids.extend(list(ids))
        assert sorted(all_ids) == list(range(50))
        # pairwise disjoint
        for i, t1 in enumerate(BACKTEST_CALIBRATION_TARGETS):
            for t2 in BACKTEST_CALIBRATION_TARGETS[i + 1:]:
                assert set(parts[t1]).isdisjoint(set(parts[t2]))

    def test_registering_all_five_backtest_targets_succeeds_when_disjoint(self):
        calib = self._calib_frame(50)
        reg = CalibrationSliceRegistry()
        parts = chronological_game_id_partition(calib, targets=BACKTEST_CALIBRATION_TARGETS)
        for target, ids in parts.items():
            reg.register(target, ids)
            slice_df = slice_by_game_ids(calib, ids)
            assert set(slice_df["GAME_ID"]) == set(ids)
        reg.assert_all_targets_registered(BACKTEST_CALIBRATION_TARGETS)

    def test_identical_full_slice_for_all_five_would_still_raise(self):
        """Documents the pre-fix defect the partition prevents."""
        shared = np.arange(50)
        reg = CalibrationSliceRegistry()
        reg.register(BACKTEST_CALIBRATION_TARGETS[0], shared)
        with pytest.raises(CalibrationRoleError):
            reg.register(BACKTEST_CALIBRATION_TARGETS[1], shared)
