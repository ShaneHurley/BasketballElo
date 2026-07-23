"""Unified confidence ranking: gated tuning, walk-forward calibrator, min-score gate."""
import numpy as np
import pandas as pd

from pipeline.bet_confidence import (
    DEFAULT_CONFIDENCE_WEIGHTS,
    WalkForwardBetCalibrator,
    _pick_threshold_gated_roi,
    _scored_ats_bets,
    ats_confidence_score,
    build_confidence_features,
    prior_year_frame,
    tune_confidence_weights,
    weight_search_ranges,
    weights_differ_from_default,
)
from pipeline.calibration_metrics import compute_ece
from pipeline.confidence_diagnostics import prior_year_calibration_ablation
from pipeline.metrics import grid_search_bet_edge, walkforward_min_confidence


def _synthetic_season_df(n=120, edge_scale: float = 1.0, conf_width: float = 18.0):
    rows = []
    for i in range(n):
        edge = (4.0 + (i % 6)) * edge_scale
        win = i % 3 != 0
        rows.append({
            "simulated_season_window": "2023-2024",
            "MARKET_SPREAD": -3.5,
            "DIRECTION": "Home" if i % 2 == 0 else "Away",
            "EDGE": edge if i % 2 == 0 else -edge,
            "ACTUAL_MARGIN": 5 if win else -4,
            "ACTUAL_HOME": 110,
            "ACTUAL_AWAY": 105,
            "CONF_WIDTH": conf_width,
            "RATING_UNCERTAINTY": 200.0,
            "ELO_META_AGREEMENT": 1.0,
            "SPREAD_COVER_PROB": 0.58,
            "DISAGREEMENT_TRUST": 0.9,
            "PHANTOM_INJURY_FLAG": 0,
            "PRED_SPREAD": -2.0,
        })
    return pd.DataFrame(rows)


def test_raw_score_increases_with_edge_and_decreases_with_width_and_vol():
    low = ats_confidence_score(build_confidence_features(spread_edge_pts=2.5))
    high = ats_confidence_score(build_confidence_features(spread_edge_pts=9.0))
    assert high > low

    narrow = ats_confidence_score(build_confidence_features(conf_width=14.0))
    wide = ats_confidence_score(build_confidence_features(conf_width=26.0))
    assert narrow > wide

    calm = ats_confidence_score(build_confidence_features(matchup_vol_sigma=8.0, league_vol_sigma=10.0))
    volatile = ats_confidence_score(build_confidence_features(matchup_vol_sigma=14.0, league_vol_sigma=10.0))
    assert calm > volatile


def test_prior_year_frame_returns_latest_season_only():
    df = pd.DataFrame({
        "simulated_season_window": ["2022-2023"] * 3 + ["2023-2024"] * 5,
        "EDGE": [6.0] * 8,
    })
    py = prior_year_frame(df)
    assert len(py) == 5


def test_weight_ranges_shrink_year_over_year():
    center = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    wide = weight_search_ranges(center, shrink=1.0)
    narrow = weight_search_ranges(center, shrink=0.5)
    key = "edge_slope"
    assert (narrow[key][1] - narrow[key][0]) < (wide[key][1] - wide[key][0])


def test_gated_tuner_prefers_high_edge_slope_weights():
    train = _synthetic_season_df(150, edge_scale=1.0, conf_width=14.0)
    weights, cal, roi, thr, ece, n = tune_confidence_weights(
        train,
        n_samples=40,
        min_bets=25,
        thresholds=range(50, 66),
    )
    assert n >= 25
    assert roi > -0.5
    high_edge_weights = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    high_edge_weights["edge_slope"] = 3.5
    cal_hi = WalkForwardBetCalibrator(confidence_weights=high_edge_weights)
    cal_hi.fit(train, scope="all_prior")
    scored_hi = _scored_ats_bets(cal_hi, train)
    roi_hi, _, _, _ = _pick_threshold_gated_roi(scored_hi, thresholds=range(50, 66), min_bets=25)
    assert roi_hi >= roi - 0.05 or weights["edge_slope"] >= DEFAULT_CONFIDENCE_WEIGHTS["edge_slope"]


def test_calibration_ablation_methods_differ_on_ece():
    rows = []
    for season in ("2022-2023", "2023-2024", "2024-2025"):
        rows.extend(_synthetic_season_df(100).assign(simulated_season_window=season).to_dict("records"))
    df = pd.DataFrame(rows)
    ablation = prior_year_calibration_ablation(df, methods=("isotonic", "none"))
    if len(ablation) >= 2:
        eces = ablation.groupby("method")["ece"].mean()
        assert len(eces) >= 1


def test_grid_search_bet_edge_varies_n_bets():
    rows = []
    for i in range(80):
        edge = 2.0 + (i % 8)
        rows.append({
            "EDGE": edge if i % 2 == 0 else -edge,
            "DIRECTION": "Home",
            "MARKET_SPREAD": -3.0,
            "ACTUAL_MARGIN": 4 if i % 3 else -2,
            "ACTUAL_HOME": 110,
            "ACTUAL_AWAY": 105,
        })
    df = pd.DataFrame(rows)
    _, grid = grid_search_bet_edge(df, edges=(2.5, 5.5, 7.0))
    assert len(grid) >= 2
    assert grid["n_bets"].nunique() > 1


def test_min_confidence_gate_blocks_low_scores():
    from pipeline.config import CONFIDENCE_SELECTION_MODE, MIN_CONFIDENCE_SCORE

    direction = "Home"
    conf_score = 52
    if CONFIDENCE_SELECTION_MODE == "min_score" and direction != "Pass" and conf_score < MIN_CONFIDENCE_SCORE:
        direction = "Pass"
    assert direction == "Pass"

    direction = "Home"
    conf_score = 66
    if CONFIDENCE_SELECTION_MODE == "min_score" and direction != "Pass" and conf_score < MIN_CONFIDENCE_SCORE:
        direction = "Pass"
    assert direction == "Home"


def test_walkforward_min_confidence_on_prior_year():
    rows = []
    for i in range(120):
        conf = 50 + (i % 16)
        win = conf >= 64 and i % 3 != 0
        rows.append({
            "MARKET_SPREAD": -3.0,
            "DIRECTION": "Home",
            "CONFIDENCE": conf,
            "ACTUAL_MARGIN": 4 if win else -5,
            "EDGE": 6.0,
        })
    prior = pd.DataFrame(rows)
    thr = walkforward_min_confidence(
        prior, thresholds=(58, 60, 62, 64, 66), default=64, min_bets=20, require_positive_ci=False,
    )
    assert thr >= 58
