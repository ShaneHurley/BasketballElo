"""Model monitoring and drift detection for live deployment."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.metrics import add_clv_columns, bootstrap_ci, ats_win_series


def load_prediction_log(path) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p)


def weekly_report(results_df: pd.DataFrame, window: int = 30) -> dict:
    """Summary metrics for the last ``window`` bets/games."""
    if results_df is None or results_df.empty:
        return {"status": "no_data"}
    df = add_clv_columns(results_df.tail(window))
    wins = ats_win_series(df)
    mae = (df["PRED_SPREAD"] - df["ACTUAL_MARGIN"]).abs().mean() if "ACTUAL_MARGIN" in df else np.nan
    _, lo, hi = bootstrap_ci(wins.astype(float)) if len(wins) else (np.nan, np.nan, np.nan)
    clv = df["CLV"].mean() if "CLV" in df else np.nan
    alert = bool(len(wins) >= 20 and wins.mean() < 0.48)
    return {
        "n_games": len(df),
        "n_bets": len(wins),
        "ats_pct": wins.mean() if len(wins) else np.nan,
        "ats_ci_lo": lo,
        "ats_ci_hi": hi,
        "spread_mae": mae,
        "mean_clv": clv,
        "alert_underperformance": alert,
    }


def feature_drift(train_features: pd.DataFrame, live_features: pd.DataFrame,
                  cols=None, threshold: float = 2.0) -> pd.DataFrame:
    """Flag features whose live mean deviates > threshold std from training."""
    cols = cols or [c for c in train_features.columns if c in live_features.columns]
    rows = []
    for c in cols:
        if not np.issubdtype(train_features[c].dtype, np.number):
            continue
        tm, ts = train_features[c].mean(), train_features[c].std() or 1e-6
        lm = live_features[c].mean()
        z = abs(lm - tm) / ts
        if z >= threshold:
            rows.append({"feature": c, "train_mean": tm, "live_mean": lm, "z_score": z})
    return pd.DataFrame(rows)
