#!/usr/bin/env python3
"""Compare smoke vs full backtest for lean-|edge| compression + conf grid.

Example:
  python scripts/diagnose_edge_compression.py \\
    --smoke-dir ../output/20260731_110433_calib_2026_smoke \\
    --full-dir  ../output/20260731_120002_calib_2026
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.edge_compression import (  # noqa: E402
    compare_runs,
    confidence_floor_grid,
    print_report,
)


def _load_csv(run_dir: Path | None, csv: Path | None) -> tuple[pd.DataFrame, Path | None]:
    if csv is not None:
        path = Path(csv)
        return pd.read_csv(path, low_memory=False), path.parent.parent if path.parent.name == "betting" else path.parent
    if run_dir is None:
        raise SystemExit("Provide --smoke-dir/--full-dir or --smoke-csv/--full-csv")
    run_dir = Path(run_dir)
    path = run_dir / "betting" / "backtest_results.csv"
    if not path.is_file():
        raise SystemExit(f"Missing {path}")
    return pd.read_csv(path, low_memory=False), run_dir


def main() -> int:
    p = argparse.ArgumentParser(description="Diagnose edge compression (smoke vs full)")
    p.add_argument("--smoke-dir", type=Path, default=None)
    p.add_argument("--full-dir", type=Path, default=None)
    p.add_argument("--smoke-csv", type=Path, default=None)
    p.add_argument("--full-csv", type=Path, default=None)
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write diagnosis tables/JSON (default: <full-dir>/betting/edge_compression_diag)",
    )
    p.add_argument(
        "--conf-floors",
        default="40,45,50,55",
        help="Comma-separated confidence floors for CSV ablation at min_edge=5.5",
    )
    args = p.parse_args()

    smoke_df, smoke_dir = _load_csv(args.smoke_dir, args.smoke_csv)
    full_df, full_dir = _load_csv(args.full_dir, args.full_csv)

    report = compare_runs(smoke_df, full_df, smoke_dir=smoke_dir, full_dir=full_dir)
    floors = tuple(float(x.strip()) for x in args.conf_floors.split(",") if x.strip())
    conf_grid = confidence_floor_grid(full_df, floors=floors, min_edge=5.5)
    print_report(report, conf_grid=conf_grid)

    out_dir = args.out_dir
    if out_dir is None and full_dir is not None:
        out_dir = Path(full_dir) / "betting" / "edge_compression_diag"
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        for key, name in (
            ("smoke_seasons", "smoke_season_edge.csv"),
            ("full_seasons", "full_season_edge.csv"),
        ):
            tbl = report.get(key)
            if isinstance(tbl, pd.DataFrame) and not tbl.empty:
                tbl.to_csv(out_dir / name, index=False)
        if not conf_grid.empty:
            conf_grid.to_csv(out_dir / "confidence_floor_grid.csv", index=False)
        payload = {
            "smoke_elo_blend_alpha": report.get("smoke_elo_blend_alpha"),
            "full_elo_blend_alpha": report.get("full_elo_blend_alpha"),
            "overlap": report.get("overlap"),
            "smoke_attrition": report.get("smoke_attrition"),
            "full_attrition": report.get("full_attrition"),
            "ranked_causes": report.get("ranked_causes"),
            "note": "Do not lower MIN_EDGE_BUCKET; prefer capping elo_blend_alpha search/hi.",
        }
        (out_dir / "diagnosis.json").write_text(json.dumps(payload, indent=2, default=str))
        print(f"\nWrote → {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
