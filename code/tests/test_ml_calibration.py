"""Walk-forward ML win-probability calibration and selection gates."""
import numpy as np
import pandas as pd

from pipeline.market import select_ml_bet
from pipeline.ml_calibration import WalkForwardMLCalibrator


def _synthetic_ml_df(n=200, bias=0.15):
    rng = np.random.default_rng(42)
    raw_p = rng.uniform(0.25, 0.75, n)
    home_win = (rng.random(n) < raw_p + bias).astype(int)
    return pd.DataFrame({
        "simulated_season_window": ["2024-2025"] * n,
        "WIN_PROB_RAW": raw_p,
        "WIN_PROB": raw_p,
        "ACTUAL_HOME": np.where(home_win, 110, 100),
        "ACTUAL_AWAY": np.where(home_win, 100, 110),
        "ACTUAL_MARGIN": np.where(home_win, 10, -10),
        "MARKET_ML": -110.0,
    })


def test_ml_calibrator_improves_ece():
    from pipeline.calibration_metrics import compute_ece

    df = _synthetic_ml_df(300, bias=0.12)
    cal = WalkForwardMLCalibrator(method="platt")
    cal.fit(df, prob_col="WIN_PROB_RAW", scope="all_prior")
    assert cal._fitted
    hw = (df["ACTUAL_MARGIN"] > 0).astype(int).values
    raw_ece = compute_ece(hw, df["WIN_PROB_RAW"].values)
    cal_p = np.array([cal.transform(p) for p in df["WIN_PROB_RAW"]])
    cal_ece = compute_ece(hw, cal_p)
    assert cal_ece <= raw_ece + 0.02


def test_select_ml_bet_respects_min_win_pct():
  side, ev, dec = select_ml_bet(
      0.52, -110.0, min_ev=0.01, max_favorite_decimal=1.35, min_win_pct=60,
  )
  assert side == "Pass"


def test_select_ml_bet_not_all_underdogs_when_calibrated():
    """High min_win_pct + min_ev should block most fake underdog flags."""
    flagged = 0
    for p in np.linspace(0.35, 0.65, 40):
        side, _, dec = select_ml_bet(
            float(p), 150.0, min_ev=0.08, max_favorite_decimal=1.45, min_win_pct=55,
        )
        if side != "Pass":
            flagged += 1
    assert flagged < 25
