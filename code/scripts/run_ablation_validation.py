#!/usr/bin/env python3
"""Run post-hoc ablation on backtest_results.csv; optionally flip config defaults."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import numpy as np

from pipeline.ablation import passes_ablation_gate, recommend_default_flips
from pipeline.ablation_posthoc import (
    passes_flip_gate,
    posthoc_ablation_summary,
    simulate_selection_mode,
)
from pipeline.config import STATE_DIR


def _load_results(path: Path):
    import pandas as pd
    df = pd.read_csv(path, low_memory=False)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df


def flip_config_defaults(reason: str) -> bool:
    """Update pipeline/config.py when ablation flip gate passes."""
    cfg_path = ROOT / "pipeline" / "config.py"
    text = cfg_path.read_text()
    replacements = {
        'BET_SELECTION_MODE = "legacy_tiers"': 'BET_SELECTION_MODE = "edge_bucket"',
        "USE_TIER_STAKE_GATES = True": "USE_TIER_STAKE_GATES = False",
        "WALKFORWARD_EDGE_MIN_FLOOR = None": "WALKFORWARD_EDGE_MIN_FLOOR = 5.5",
    }
    new_text = text
    for old, new in replacements.items():
        if old not in new_text:
            print(f"  skip flip (pattern not found): {old}")
            return False
        new_text = new_text.replace(old, new, 1)
    if new_text == text:
        return False
    cfg_path.write_text(new_text)
    manifest = {
        "flipped": True,
        "reason": reason,
        "changes": list(replacements.values()),
    }
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    (STATE_DIR / "config_flip_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"Updated defaults in {cfg_path}")
    return True


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-hoc ablation validation")
    parser.add_argument(
        "--results",
        type=Path,
        default=ROOT / "backtest_results.csv",
        help="Backtest CSV export",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="Flip config.py defaults if edge_bucket passes flip gate",
    )
    parser.add_argument("--out", type=Path, default=STATE_DIR / "ablation_posthoc.json")
    args = parser.parse_args()

    if not args.results.exists():
        print(f"Missing results file: {args.results}")
        print("Run: python run_backtest.py  (or use existing backtest_results.csv)")
        return 1

    df = _load_results(args.results)
    summary = posthoc_ablation_summary(df)
    if summary.empty:
        print("No metrics computed.")
        return 1

    print("\n=== POST-HOC ABLATION (saved backtest) ===")
    print(summary[["config", "spread_mae", "ats_pct", "roi", "n_bets", "brier"]].to_string(
        index=False, float_format=lambda v: f"{v:.4f}" if isinstance(v, float) else str(v),
    ))

    baseline = summary[summary["config"] == "baseline_current"].iloc[0].to_dict()
    edge = summary[summary["config"] == "edge_bucket"]
    edge_row = edge.iloc[0].to_dict() if not edge.empty else {}

    passed_gate = passes_ablation_gate(baseline, edge_row) if edge_row else False
    passed_flip = passes_flip_gate(edge_row, roi_gain=baseline.get("roi", 0) + 0.02) if edge_row else False

    # Flip gate: candidate ROI must beat baseline by 2% AND meet ATS/bet count floors
    if edge_row:
        roi_delta = float(edge_row.get("roi", 0)) - float(baseline.get("roi", 0))
        passed_flip = (
            passes_flip_gate(edge_row)
            and roi_delta >= 0.02
            and passes_ablation_gate(baseline, edge_row)
        )

    rec = recommend_default_flips(
        summary.assign(season="ALL"),
        baseline_name="baseline_current",
    )

    report = {
        "source": str(args.results),
        "summary": summary.to_dict(orient="records"),
        "passes_ablation_gate": passed_gate,
        "passes_flip_gate": passed_flip,
        "recommend_configs": rec,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, default=str))
    print(f"\nSaved report → {args.out}")
    print(f"Ablation gate (edge_bucket vs baseline): {'PASS' if passed_gate else 'FAIL'}")
    print(f"Flip gate (+2% ROI, ATS≥55%, n≥400): {'PASS' if passed_flip else 'FAIL'}")

    if args.flip:
        if passed_flip:
            flip_config_defaults("edge_bucket passed post-hoc flip gate")
        else:
            print("Defaults NOT flipped (gate failed). Legacy config preserved.")
    elif passed_flip:
        print("Flip gate passed — re-run with --flip to update config.py defaults.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
