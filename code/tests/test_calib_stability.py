"""Multi-year calib ablation: lock winner by MAE/ECE stability (not ROI)."""
from __future__ import annotations

import pandas as pd

from pipeline.ablation import (
    calibration_ablation_configs,
    calib_stability_score,
    lock_calib_winner,
)


def test_calibration_ablation_configs_include_mode_and_windows():
    cfgs = calibration_ablation_configs()
    assert "simplified_calib" in cfgs
    assert "beta_ats_calib" in cfgs
    assert "legacy_stack_calib" in cfgs
    assert "spread_window_30" in cfgs
    assert "spread_window_50" in cfgs
    assert "spread_window_80" in cfgs


def test_calib_stability_prefers_lower_mae_variance():
    summary = pd.DataFrame([
        {"config": "volatile", "season": "a", "spread_mae": 10.0, "brier": 0.2},
        {"config": "volatile", "season": "b", "spread_mae": 14.0, "brier": 0.3},
        {"config": "stable", "season": "a", "spread_mae": 11.5, "brier": 0.22},
        {"config": "stable", "season": "b", "spread_mae": 11.7, "brier": 0.23},
        {"config": "volatile", "season": "ALL", "spread_mae": 12.0, "brier": 0.25},
        {"config": "stable", "season": "ALL", "spread_mae": 11.6, "brier": 0.225},
    ])
    stab = calib_stability_score(summary)
    assert stab.iloc[0]["config"] == "stable"
    locked = lock_calib_winner(summary)
    assert locked["winner"] == "stable"


def test_lock_spread_window_winner_maps_config_override():
    summary = pd.DataFrame([
        {"config": "spread_window_30", "season": "a", "spread_mae": 11.0, "brier": 0.2},
        {"config": "spread_window_50", "season": "a", "spread_mae": 12.0, "brier": 0.2},
        {"config": "spread_window_30", "season": "ALL", "spread_mae": 11.0, "brier": 0.2},
        {"config": "spread_window_50", "season": "ALL", "spread_mae": 12.0, "brier": 0.2},
    ])
    locked = lock_calib_winner(summary)
    assert locked["winner"] == "spread_window_30"
    assert locked["promotions"]["SPREAD_CALIB_WINDOW"] == 30
