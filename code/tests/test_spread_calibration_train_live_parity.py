"""Task 054 — replay rolling spread calibration identically in training
simulation and live inference; remove the static-calibration-to-rolling-
inference mismatch.
"""
import numpy as np
import pytest

from pipeline.market import SpreadCalibrator, replay_rolling_spread_calibration


def _synthetic_preds_actuals(n=80, seed=3):
    rng = np.random.RandomState(seed)
    preds = rng.normal(scale=8, size=n)
    actuals = preds * 0.8 + 1.5 + rng.normal(scale=2, size=n)  # true bias+slope
    return preds, actuals


class TestReplayMatchesLiveSequentialPattern:
    def test_replay_matches_manual_predict_then_update_loop(self):
        """This is exactly what pipeline/simulate.py does at live inference
        time (correct() before update()) — the replay helper must produce
        byte-identical output to that manual loop."""
        preds, actuals = _synthetic_preds_actuals()
        cal_manual = SpreadCalibrator(window=30, min_samples=10)
        manual_corrected = []
        for p, a in zip(preds, actuals):
            manual_corrected.append(cal_manual.correct(float(p)))
            cal_manual.update(float(p), float(a))
        manual_corrected = np.array(manual_corrected)

        replayed, cal_replay = replay_rolling_spread_calibration(
            preds, actuals, window=30, min_samples=10,
        )
        assert np.allclose(replayed, manual_corrected)
        assert cal_replay.a == cal_manual.a
        assert cal_replay.b == cal_manual.b

    def test_early_rows_are_not_corrected_using_later_rows_state(self):
        """Regression test for the static-fit defect: the corrected value
        for row 0 must be identical whether or not later rows exist, since
        row 0 has no strictly-earlier history to calibrate from."""
        preds, actuals = _synthetic_preds_actuals(n=50)
        full_replay, _ = replay_rolling_spread_calibration(preds, actuals, window=30, min_samples=10)
        truncated_replay, _ = replay_rolling_spread_calibration(
            preds[:5], actuals[:5], window=30, min_samples=10,
        )
        assert np.allclose(full_replay[:5], truncated_replay)


class TestStaticFitThenBlanketCorrectIsTheConfirmedMismatch:
    def test_fit_static_blanket_correct_differs_from_rolling_replay(self):
        """Regression test documenting the defect Task 054 removes:
        SpreadCalibrator.fit_static() + a blanket list-comprehension
        .correct() over the same slice (the current pipeline/backtest.py
        ``static_cal`` pattern) uses each row's *own* future data to
        correct itself, so it disagrees with the leak-free rolling replay
        that live inference actually performs.
        """
        preds, actuals = _synthetic_preds_actuals(n=60)
        static_cal = SpreadCalibrator.fit_static(preds, actuals, window=60, min_samples=10)
        blanket_corrected = np.array([static_cal.correct(float(p)) for p in preds])

        replayed, _ = replay_rolling_spread_calibration(preds, actuals, window=60, min_samples=10)

        assert not np.allclose(blanket_corrected, replayed), (
            "Expected the static blanket-correct pattern to disagree with "
            "the leak-free rolling replay (that disagreement is exactly "
            "the train/live mismatch Task 054 must remove from production "
            "call sites); if this now passes, re-audit before assuming the "
            "mismatch is gone rather than relaxing this test."
        )
