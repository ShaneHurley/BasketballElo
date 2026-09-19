"""Phase 1–3 / 6 unit tests for calibration + selection + meta-label."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.calibration_metrics import (
    compute_log_loss,
    murphy_brier_decomposition,
)
from pipeline.dataset_roles import RoleAssignmentError
from pipeline.meta_labeling import apply_meta_labels, assert_policy_role, meta_label_pass
from pipeline.selection_bias import compare_to_baseline, season_metric_paths
from pipeline.trackers import LeagueRollingStats
from pipeline.tuning import ELO_PARAM_BOUNDS
import inspect
import pipeline.tuning as tuning_mod


def test_murphy_brier_perfect_is_zero_rel():
    y = np.array([0.0, 1.0, 0.0, 1.0] * 20)
    p = y.copy()
    d = murphy_brier_decomposition(y, p, n_bins=10)
    assert d["n"] == 80
    assert d["reliability"] < 1e-9
    assert d["brier"] < 1e-9


def test_log_loss_bounds():
    y = np.array([1.0, 0.0, 1.0, 0.0])
    p = np.array([0.9, 0.1, 0.8, 0.2])
    ll = compute_log_loss(y, p)
    assert 0.0 < ll < 1.0


def test_elo_optuna_is_mae_first():
    src2 = inspect.getsource(tuning_mod)
    assert "return float(mae)" in src2
    assert "ats_miss * 30" not in src2
    assert ELO_PARAM_BOUNDS["k_def"][0] <= 0.05
    assert ELO_PARAM_BOUNDS["k_def"][1] >= 1.5
    assert ELO_PARAM_BOUNDS["clutch_boost"][1] >= 1.6


def test_league_rolling_z_past_only():
    tr = LeagueRollingStats(window_games=50)
    for i in range(40):
        tr.update(pace_diff=float(i), elo_net=0.0, roll_net_xppp=0.0, elo_margin=0.0)
    feats = tr.feature_dict({"pace_diff": 39.0, "elo_net": 0.0, "roll_net_xppp": 0.0, "elo_margin": 0.0})
    assert "pace_diff_lz" in feats
    assert feats["pace_diff_lz"] != 0.0


def test_selection_bias_paths():
    df = pd.DataFrame({
        "season": [2022, 2022, 2023, 2023],
        "PRED_SPREAD": [1.0, 2.0, 3.0, 4.0],
        "ACTUAL_MARGIN": [0.0, 0.0, 0.0, 0.0],
        "DIRECTION": ["Home", "Pass", "Away", "Home"],
        "HIT": [1, 0, 1, 0],
        "CLV": [0.5, np.nan, 0.2, -0.1],
    })
    paths = season_metric_paths(df)
    assert len(paths) == 2
    ok, msg = compare_to_baseline({"mae_mean": 1.0}, {"mae_mean": 1.0})
    assert ok
    ok2, _ = compare_to_baseline({"mae_mean": 2.0}, {"mae_mean": 1.0}, mae_tol=0.25)
    assert not ok2


def test_meta_label_role_enforcement():
    with pytest.raises(RoleAssignmentError):
        assert_policy_role(["train", "policy_tuning"])
    assert meta_label_pass(direction="Home", ats_cover_prob=0.6)
    assert not meta_label_pass(direction="Home", ats_cover_prob=0.4)
    df = pd.DataFrame({
        "DIRECTION": ["Home", "Away", "Pass"],
        "ATS_COVER_PROB": [0.6, 0.4, 0.9],
        "UPSET_PROB": [0.2, 0.2, 0.2],
        "MARKET_SPREAD": [-3.0, 4.0, -1.0],
    })
    out = apply_meta_labels(df, min_ats_prob=0.52)
    assert bool(out.loc[0, "META_LABEL_PASS"])
    assert not bool(out.loc[1, "META_LABEL_PASS"])
    assert not bool(out.loc[2, "META_LABEL_PASS"])
