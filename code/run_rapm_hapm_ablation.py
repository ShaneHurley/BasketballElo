#!/usr/bin/env python3
"""Ablation 8.4 — RAPM vs HAPM/chemistry multicollinearity study.

Runs ``rapm_only``, ``hapm_chem``, ``all``, ``rapm_only_retuned``, then applies
``should_prune_hapm_chem``. Writes a report under ``output/<stamp>_ablation_8_4/``.

Example
-------
  cd code
  python run_rapm_hapm_ablation.py \\
    --stints ../output/20260920_215558_baseline_rapm_v7/artifacts/suite_stints.pkl \\
    --elo-trials 8 --hier-trials 8 --meta-trials 12 --allow-tip-proxy
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Ablation 8.4 RAPM vs HAPM/chem")
    p.add_argument("--stints", type=str, required=True, help="Path to suite_stints.pkl")
    p.add_argument("--elo-trials", type=int, default=8)
    p.add_argument("--hier-trials", type=int, default=8)
    p.add_argument("--meta-trials", type=int, default=12)
    p.add_argument("--window", type=int, default=4)
    p.add_argument("--allow-tip-proxy", action="store_true")
    p.add_argument("--output-root", type=str, default=None)
    p.add_argument("--label", type=str, default="ablation_8_4")
    args = p.parse_args(argv)

    from pipeline.ablation import (
        hapm_chem_prune_cols,
        run_rapm_hapm_ablation,
    )
    from pipeline.data_paths import resolve_odds_path, resolve_data_dir, resolve_pinnacle_path
    from pipeline.odds_loader import load_odds_dict

    stints_path = Path(args.stints).expanduser().resolve()
    if not stints_path.exists():
        raise SystemExit(f"stints not found: {stints_path}")
    stints = pd.read_pickle(stints_path)
    print(f"Loaded stints: {stints_path} rows={len(stints)}")

    data_dir = resolve_data_dir(None)
    odds_path = resolve_odds_path(data_dir, REPO_ROOT)
    if odds_path is None or not Path(odds_path).exists():
        raise SystemExit("all_odds.csv not found")
    pinnacle_path = resolve_pinnacle_path(data_dir, REPO_ROOT)
    odds_dict, prov = load_odds_dict(
        modern_odds_path=odds_path,
        pinnacle_path=pinnacle_path,
        allow_tip_proxy=bool(args.allow_tip_proxy),
    )
    print(f"Odds: keys={len(odds_dict)} provenance={prov.get('quote_source')}")

    out_root = Path(args.output_root) if args.output_root else (REPO_ROOT / "output")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = out_root / f"{stamp}_{args.label}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "artifacts").mkdir(exist_ok=True)
    print(f"Run dir: {run_dir}")

    summary_df, detail, prune = run_rapm_hapm_ablation(
        stints,
        odds_dict=odds_dict,
        rolling_window_size=int(args.window),
        n_tuning_trials_elo=int(args.elo_trials),
        n_tuning_trials_hier=int(args.hier_trials),
        n_tuning_trials_meta=int(args.meta_trials),
    )

    summary_path = run_dir / "ablation_summary.csv"
    if summary_df is not None and not summary_df.empty:
        summary_df.to_csv(summary_path, index=False)
    for name, df in (detail or {}).items():
        if df is not None and hasattr(df, "to_pickle"):
            df.to_pickle(run_dir / "artifacts" / f"{name}_results.pkl")

    report = {
        "finished_at": _utc_now(),
        "stints": str(stints_path),
        "trials": {
            "elo": args.elo_trials,
            "hier": args.hier_trials,
            "meta": args.meta_trials,
        },
        "window": args.window,
        "odds_provenance": prov,
        "prune": prune,
        "prune_cols_if_applied": hapm_chem_prune_cols() if prune.get("prune") else [],
        "summary_csv": str(summary_path) if summary_path.exists() else None,
    }
    report_path = run_dir / "ablation_8_4_report.json"
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"Wrote {report_path}")

    # Document prune decision next to locked baselines (do not auto-edit SAFE).
    baselines = REPO_ROOT / "output" / "baselines"
    baselines.mkdir(parents=True, exist_ok=True)
    note = {
        "source_run": str(run_dir),
        "prune": prune,
        "note": (
            "Apply hapm_chem_prune_cols() to SAFE_FEATURE_COLS only when prune=true; "
            "do not auto-flip config.py from tip_proxy Optuna."
        ),
        "at": _utc_now(),
    }
    (baselines / "ablation_8_4_prune_decision.json").write_text(
        json.dumps(note, indent=2, default=str)
    )
    print(f"Prune decision: prune={prune.get('prune')} — {prune.get('reason')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
