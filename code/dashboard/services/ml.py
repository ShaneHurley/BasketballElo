"""ML tab helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.paths import results_csv_path


ML_PRESETS = {
    "win_prob_source": ["model", "market_blend", "upset_blend"],
    "ml_calib_method": ["none", "isotonic", "platt", "beta"],
}


def ml_scorecard(run_id: str | None) -> dict[str, Any]:
    path = results_csv_path(run_id)
    out: dict[str, Any] = {"run_id": run_id, "metrics": {}}
    if path is None or not path.exists():
        return out
    df = pd.read_csv(path)
    wp = df.get("WIN_PROB", df.get("P_HOME"))
    if wp is None or "ACTUAL_HOME" not in df.columns:
        return out
    p = pd.to_numeric(wp, errors="coerce").clip(1e-6, 1 - 1e-6)
    y = (pd.to_numeric(df["ACTUAL_HOME"], errors="coerce")
         > pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")).astype(float)
    m = p.notna() & y.notna()
    if not m.any():
        return out
    pp, yy = p[m].to_numpy(), y[m].to_numpy()
    out["metrics"]["brier"] = float(np.mean((pp - yy) ** 2))
    out["metrics"]["logloss"] = float(-np.mean(yy * np.log(pp) + (1 - yy) * np.log(1 - pp)))
    out["metrics"]["n"] = int(m.sum())
    if "MARKET_ML" in df.columns:
        out["metrics"]["has_market_ml"] = True
    return out
