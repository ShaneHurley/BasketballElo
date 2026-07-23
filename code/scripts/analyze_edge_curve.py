#!/usr/bin/env python3
"""Run data-driven edge vs accuracy analysis on walk-forward backtest CSV."""
import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.edge_analysis import run_edge_analysis
from pipeline.metrics import add_all_profile_columns


def main():
    p = argparse.ArgumentParser(description="Analyze edge vs ATS/ROI from backtest results")
    p.add_argument("csv", nargs="?", default=str(ROOT / "backtest_results.csv"))
    p.add_argument("--save-dir", type=Path, default=None, help="Write CSV tables here")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    if "STAKE_MODERATE" not in df.columns:
        df = add_all_profile_columns(df)

    report = run_edge_analysis(df, verbose=True)

    if args.save_dir:
        args.save_dir.mkdir(parents=True, exist_ok=True)
        for key in ("fine_bins", "rolling", "threshold_grid", "segments", "injury_split"):
            tbl = report.get(key)
            if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                tbl.to_csv(args.save_dir / f"edge_{key}.csv", index=False)
        print(f"\nSaved tables → {args.save_dir}")


if __name__ == "__main__":
    main()
