"""Totals tab helpers."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.paths import results_csv_path


TOTALS_PRESETS = {
    "total_train_target": ["actual", "market_residual"],
    "sigma_source": ["model", "market", "blend"],
}


def totals_scorecard(run_id: str | None) -> dict[str, Any]:
    path = results_csv_path(run_id)
    out: dict[str, Any] = {"run_id": run_id, "metrics": {}}
    if path is None or not path.exists():
        return out
    df = pd.read_csv(path)
    if "PRED_TOTAL" not in df.columns or "ACTUAL_HOME" not in df.columns:
        return out
    pred = pd.to_numeric(df["PRED_TOTAL"], errors="coerce")
    act = pd.to_numeric(df["ACTUAL_HOME"], errors="coerce") + pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")
    err = (pred - act).dropna()
    if len(err):
        out["metrics"]["mae"] = float(err.abs().mean())
        out["metrics"]["rmse"] = float(np.sqrt((err ** 2).mean()))
        out["metrics"]["bias"] = float(err.mean())
        out["metrics"]["n"] = int(len(err))
    if "OU_HIT" in df.columns:
        out["metrics"]["ou_hit"] = float(pd.to_numeric(df["OU_HIT"], errors="coerce").mean())
    return out
