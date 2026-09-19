#!/usr/bin/env python3
"""Calibrate min-|edge| policy from a walk-forward backtest CSV.

Post-hoc only: re-thresholds EDGE / EDGE_LEAN to show how many bets each
floor would produce, plus ATS and decision≠close CLV. Does **not** change
``MIN_EDGE_BUCKET`` / ``WALKFORWARD_EDGE_MIN_FLOOR``.

Example:
  python scripts/calibrate_edge_policy.py \\
    ../output/20260731_110433_calib_2026_smoke/betting/backtest_results.csv \\
    --out-dir ../output/20260731_110433_calib_2026_smoke/betting/edge_policy_calib \\
    --compare-avoid
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.edge_policy_calibration import (  # noqa: E402
    DEFAULT_EDGE_GRID,
    run_edge_policy_calibration,
)


def _parse_floats(raw: str) -> tuple[float, ...]:
    parts = [p.strip() for p in raw.replace(";", ",").split(",") if p.strip()]
    return tuple(float(p) for p in parts)


def _parse_seasons(raw: str | None) -> list[int] | None:
    if not raw:
        return None
    return [int(p.strip()) for p in raw.replace(";", ",").split(",") if p.strip()]


def main() -> int:
    p = argparse.ArgumentParser(
        description="Post-hoc edge-floor calibration (policy_tuning; no config writes)",
    )
    p.add_argument(
        "csv",
        nargs="?",
        default=str(ROOT / "backtest_results.csv"),
        help="Walk-forward backtest_results.csv (default: ./backtest_results.csv)",
    )
    p.add_argument(
        "--edges",
        default=",".join(str(x) for x in DEFAULT_EDGE_GRID),
        help="Comma-separated min-|edge| candidates",
    )
    p.add_argument(
        "--seasons",
        default=None,
        help="Optional season filter (e.g. 2024,2025) using simulated_season_window",
    )
    p.add_argument("--min-bets", type=int, default=20, help="Skip scoring below this n")
    p.add_argument(
        "--no-avoid-band",
        action="store_true",
        help="Ignore EDGE_AVOID_BAND when selecting bets",
    )
    p.add_argument(
        "--edge-only",
        action="store_true",
        help="Ignore confidence/trust/phantom gates (upper-bound volume only)",
    )
    p.add_argument(
        "--compare-avoid",
        action="store_true",
        help="Also write avoid-band on vs off comparison table",
    )
    p.add_argument(
        "--allow-neg-clv",
        action="store_true",
        help="Do not require mean CLV ≥ 0 for recommendation",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Write grid CSV + recommendation JSON here",
    )
    args = p.parse_args()

    path = Path(args.csv)
    if not path.is_file():
        print(f"CSV not found: {path}", file=sys.stderr)
        return 1

    df = pd.read_csv(path, low_memory=False)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")

    report = run_edge_policy_calibration(
        df,
        thresholds=_parse_floats(args.edges),
        use_avoid_band=not args.no_avoid_band,
        full_gates=not args.edge_only,
        compare_avoid=args.compare_avoid,
        min_bets=int(args.min_bets),
        seasons=_parse_seasons(args.seasons),
        require_nonneg_clv=not args.allow_neg_clv,
        verbose=True,
    )

    out_dir = args.out_dir
    if out_dir is None and path.parent.name == "betting":
        out_dir = path.parent / "edge_policy_calib"
    if out_dir is not None:
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        grid = report.get("grid")
        if isinstance(grid, pd.DataFrame) and not grid.empty:
            grid.to_csv(out_dir / "edge_policy_grid.csv", index=False)
        seasons = report.get("season_summary")
        if isinstance(seasons, pd.DataFrame) and not seasons.empty:
            seasons.to_csv(out_dir / "edge_season_summary.csv", index=False)
        avoid = report.get("avoid_band_compare")
        if isinstance(avoid, pd.DataFrame) and not avoid.empty:
            avoid.to_csv(out_dir / "edge_avoid_band_compare.csv", index=False)
        payload = {
            "source_csv": str(path.resolve()),
            "live_policy": report.get("live_policy"),
            "recommendation": report.get("recommendation"),
            "live_apply_threshold_ref": report.get("live_apply_threshold_ref"),
            "n_lean_rows": report.get("n_lean_rows"),
            "note": (
                "Advisory only. Do not edit MIN_EDGE_BUCKET from this file; "
                "promote via explicit policy_tuning decision."
            ),
        }
        (out_dir / "edge_policy_recommendation.json").write_text(
            json.dumps(payload, indent=2, default=str)
        )
        print(f"\nWrote → {out_dir}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
