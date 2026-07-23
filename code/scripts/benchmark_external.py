#!/usr/bin/env python3
"""Compare model predictions against external benchmarks and closing lines."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.metrics import add_clv_columns, ats_win_series, benchmark_results


def load_external_predictions(path: Path) -> pd.DataFrame:
    """CSV with columns: date, home, away, pred_margin (optional: source)."""
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date
    return df


def merge_results(backtest_csv: Path, external_csv: Path = None) -> pd.DataFrame:
    df = pd.read_csv(backtest_csv)
    df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce").dt.date
    df = add_clv_columns(df)
    if external_csv and Path(external_csv).exists():
        ext = load_external_predictions(Path(external_csv))
        if not ext.empty:
            df = df.merge(
                ext.rename(columns={"pred_margin": "ext_pred_margin"}),
                left_on=["DATE", "HOME", "AWAY"],
                right_on=["date", "home", "away"],
                how="left",
            )
            if "ext_pred_margin" in df.columns:
                df["ext_spread_mae"] = (df["ext_pred_margin"] - df["ACTUAL_MARGIN"]).abs()
    return df


def head_to_head(df: pd.DataFrame) -> dict:
    out = {}
    if "PRED_SPREAD" in df.columns and "ACTUAL_MARGIN" in df.columns:
        out["model_spread_mae"] = (df["PRED_SPREAD"] - df["ACTUAL_MARGIN"]).abs().mean()
    if "CLOSING_SPREAD" in df.columns:
        m = df[df["CLOSING_SPREAD"].notna()]
        out["closing_line_mae"] = (-m["CLOSING_SPREAD"] - m["ACTUAL_MARGIN"]).abs().mean()
    if "ext_pred_margin" in df.columns:
        valid = df[df["ext_pred_margin"].notna()]
        out["external_mae"] = (valid["ext_pred_margin"] - valid["ACTUAL_MARGIN"]).abs().mean()
    wins = ats_win_series(df)
    out["model_ats_pct"] = wins.mean() if len(wins) else np.nan
    out["mean_clv"] = df["CLV"].mean() if "CLV" in df else np.nan
    return out


def main():
    parser = argparse.ArgumentParser(description="Benchmark vs closing line and external models")
    parser.add_argument("--backtest", default=str(ROOT / "backtest_results.csv"))
    parser.add_argument("--external", default=str(ROOT / "state" / "external_predictions.csv"))
    args = parser.parse_args()

    df = merge_results(Path(args.backtest), Path(args.external) if args.external else None)
    if df.empty:
        print("No backtest results found.")
        return 1

    print("=== Head-to-Head Benchmark ===")
    for k, v in head_to_head(df).items():
        print(f"  {k}: {v:.3f}" if isinstance(v, float) and np.isfinite(v) else f"  {k}: {v}")

    benchmark_results(df)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
