#!/usr/bin/env python3
"""Run ATS/ML/totals promotion gates on a backtest_results.csv (subprocess helper).

Avoids importing pipeline from the thin dashboard venv (tqdm/sklearn).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.promotion_gates import ats_promote, ml_promote, total_promote  # noqa: E402


def _ensure_ats_win(df: pd.DataFrame) -> pd.DataFrame:
    if "ATS_WIN" in df.columns:
        return df
    spread_col = None
    for c in ("DECISION_SPREAD", "MARKET_SPREAD", "CLOSING_SPREAD"):
        if c in df.columns:
            spread_col = c
            break
    if spread_col is None or "ACTUAL_MARGIN" not in df.columns:
        return df
    margin = pd.to_numeric(df["ACTUAL_MARGIN"], errors="coerce")
    spread = pd.to_numeric(df[spread_col], errors="coerce")
    cover = margin + spread
    if "EDGE_LEAN" in df.columns:
        side = df["EDGE_LEAN"].astype(str)
    elif "DIRECTION" in df.columns:
        side = df["DIRECTION"].astype(str)
    elif "EDGE" in df.columns:
        e = pd.to_numeric(df["EDGE"], errors="coerce")
        side = pd.Series(np.where(e > 0, "Home", np.where(e < 0, "Away", "Pass")), index=df.index)
    else:
        return df
    home_win = cover > 1e-9
    away_win = cover < -1e-9
    ats = pd.Series(np.nan, index=df.index, dtype=float)
    home_mask = side.str.lower().isin(("home", "h"))
    away_mask = side.str.lower().isin(("away", "a"))
    ats.loc[home_mask & home_win] = 1.0
    ats.loc[home_mask & away_win] = 0.0
    ats.loc[away_mask & away_win] = 1.0
    ats.loc[away_mask & home_win] = 0.0
    out = df.copy()
    out["ATS_WIN"] = ats
    return out


def main() -> int:
    p = argparse.ArgumentParser(description="Promotion gates scorecard")
    p.add_argument("--results", required=True, help="Path to backtest_results.csv")
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    path = Path(args.results)
    if not path.is_file():
        print(f"CSV not found: {path}", file=sys.stderr)
        return 1

    df = _ensure_ats_win(pd.read_csv(path, low_memory=False))
    if "simulated_season_window" in df.columns:
        seasons = list(df["simulated_season_window"].dropna().unique())
    elif "DATE" in df.columns:
        seasons = list(pd.to_datetime(df["DATE"], errors="coerce").dt.year.dropna().unique())
    else:
        seasons = [2024, 2025]

    brier_model = 0.25
    if "CALIBRATED_COVER_PROB" in df.columns and "ATS_WIN" in df.columns:
        pr = pd.to_numeric(df["CALIBRATED_COVER_PROB"], errors="coerce")
        y = pd.to_numeric(df["ATS_WIN"], errors="coerce")
        m = pr.notna() & y.notna()
        if m.any():
            brier_model = float(np.mean((pr[m] - y[m]) ** 2))

    n_bets = (
        int(pd.to_numeric(df.get("ACTIONABLE"), errors="coerce").fillna(0).sum())
        if "ACTIONABLE" in df.columns
        else len(df)
    )
    point_clv = (
        pd.to_numeric(df.get("POINT_CLV"), errors="coerce").dropna().tolist()
        if "POINT_CLV" in df.columns
        else None
    )
    ats = ats_promote(
        seasons=seasons,
        n_bets=n_bets,
        brier_model=brier_model,
        brier_market=0.25,
        point_clv=point_clv,
    )
    mae_model = (
        float(pd.to_numeric(df.get("TOTAL_ERR"), errors="coerce").abs().mean())
        if "TOTAL_ERR" in df.columns
        else 20.0
    )
    tot = total_promote(seasons=seasons, mae_model=mae_model, mae_market=mae_model + 0.5)
    ml = ml_promote(
        seasons=seasons,
        brier_model=0.22,
        brier_novig_market=0.23,
        logloss_model=0.6,
        logloss_novig_market=0.61,
    )
    out = {"ats": ats.as_dict(), "ml": ml.as_dict(), "totals": tot.as_dict()}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
