#!/usr/bin/env python3
"""Rolling monitoring report from backtest results CSV."""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.calibration_metrics import compute_brier, compute_ece
from pipeline.metrics import max_drawdown, sharpe_ratio


def _ats_profit_row(row, stake_col="STAKE_MODERATE", profit_col="PROFIT_MODERATE"):
    return float(row.get(profit_col, 0) or 0)


def rolling_report(df: pd.DataFrame, window: int = 30) -> pd.DataFrame:
    df = df.sort_values("DATE") if "DATE" in df.columns else df
    bets = df[df.get("DIRECTION", "Pass") != "Pass"].copy()
    if bets.empty:
        return pd.DataFrame()
    if "STAKE_MODERATE" not in bets.columns:
        return pd.DataFrame({"note": ["Run add_all_profile_columns first"]})

    profits = bets["PROFIT_MODERATE"].fillna(0).values
    stakes = bets["STAKE_MODERATE"].fillna(0).values
    wins = profits > 0
    rows = []
    for i in range(window, len(bets) + 1):
        sl = slice(i - window, i)
        p = profits[sl]
        s = stakes[sl]
        w = wins[sl]
        staked = s.sum()
        roi = p.sum() / staked if staked > 0 else np.nan
        cover_p = bets.iloc[sl]["COVER_PROB_CALIBRATED"] if "COVER_PROB_CALIBRATED" in bets.columns else None
        ece = np.nan
        if cover_p is not None and "ACTUAL_MARGIN" in bets.columns and "MARKET_SPREAD" in bets.columns:
            sub = bets.iloc[sl]
            cover = sub["ACTUAL_MARGIN"] + sub["MARKET_SPREAD"]
            home = sub["DIRECTION"] == "Home"
            won = (home & (cover > 0)) | (~home & (cover < 0))
            ece = compute_ece(won.astype(int).values, cover_p.values)
        rows.append({
            "end_idx": i,
            "date": bets.iloc[i - 1].get("DATE"),
            "ats_pct": w.mean(),
            "roi": roi,
            "ece": ece,
            "sharpe": sharpe_ratio(p),
            "drawdown": max_drawdown(np.cumsum(p)),
        })
    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("results_csv", type=Path, nargs="?", default=Path("backtest_results.csv"))
    parser.add_argument("--window", type=int, default=30)
    parser.add_argument("--out", type=Path, default=Path("monitoring_report.csv"))
    args = parser.parse_args()

    if not args.results_csv.exists():
        raise SystemExit(f"Missing {args.results_csv}")

    df = pd.read_csv(args.results_csv, low_memory=False)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    from pipeline.metrics import add_all_profile_columns
    if "PROFIT_MODERATE" not in df.columns:
        df = add_all_profile_columns(df)

    report = rolling_report(df, window=args.window)
    if report.empty:
        print("No rolling report generated (insufficient bets or missing columns).")
        return
    args.out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(args.out, index=False)
    last = report.iloc[-1]
    print(f"Rolling {args.window}-bet snapshot (last row):")
    print(f"  ATS%={last['ats_pct']:.1%}  ROI={last['roi']:+.1%}  ECE={last['ece']:.3f}")
    print(f"  Sharpe={last['sharpe']:.2f}  drawdown={last['drawdown']:.3f}")
    print(f"Full report → {args.out}")


if __name__ == "__main__":
    main()
