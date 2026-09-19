#!/usr/bin/env python3
"""
Run walk-forward backtest with leak fixes applied.
Usage: python run_backtest.py [--quick]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.config import MODERN_ODDS_PATH
from pipeline.config import OPTIMAL_BET_EDGE, SPREAD_CALIB_WINDOW, BET_SELECTION_MODE, OU_MIN_EDGE
from pipeline.bet_selection import uses_edge_gates
from pipeline.market import load_modern_odds
from pipeline.backtest import run_multi_year_backtest_walkforward
from pipeline.metrics import grid_search_bet_edge, benchmark_results, benchmark_betting_roi, add_all_profile_columns
from pipeline.diagnostics import diagnose_loaded_data, run_backtest_diagnostics
from pipeline.stake_profiles import PROFILES
from pipeline.stint_loader import load_stints as _load_stints_impl, read_csv_fast  # noqa: F401


def load_stints(quick: bool = False, walkforward_zone_pps: bool = True,
                use_cache: bool = True) -> pd.DataFrame:
    """Load and stitch historical stint data from available PBP files."""
    odds_path = MODERN_ODDS_PATH if Path(MODERN_ODDS_PATH).exists() else ROOT / "all_odds.csv"
    paths = None
    try:
        from pipeline.data_paths import discover_pbp_paths, resolve_data_dir, select_season_keys

        data_dir = resolve_data_dir(None)
        available = discover_pbp_paths(data_dir)
        if available:
            keys = select_season_keys(available, mini=quick, mini_n=2) if quick else list(available.keys())
            paths = [(y, available[y]) for y in keys]
    except Exception:
        paths = None
    stints, _meta = _load_stints_impl(
        paths=paths,
        quick=quick,
        walkforward_zone_pps=walkforward_zone_pps,
        use_cache=use_cache,
        odds_path=odds_path,
    )
    return stints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use only 2 seasons for smoke test")
    parser.add_argument("--elo-trials", type=int, default=30)
    parser.add_argument("--hier-trials", type=int, default=30)
    parser.add_argument("--meta-trials", type=int, default=40)
    parser.add_argument("--window", type=int, default=4)
    parser.add_argument(
        "--fast-tuning", action="store_true",
        help="Lighter inner CV during meta tuning (~3-5× faster per trial)",
    )
    args = parser.parse_args()

    print("Loading stints...")
    stints = load_stints(quick=args.quick)

    odds_path = MODERN_ODDS_PATH if MODERN_ODDS_PATH.exists() else ROOT / "all_odds.csv"
    print(f"Loading odds from {odds_path}...")
    odds_dict = load_modern_odds(str(odds_path))

    # Supplement with full-season Pinnacle lines (matched to schedule so
    # decision T-60 vs close are distinct when multi-timestamp quotes exist).
    from pipeline.config import PINNACLE_LINES_PATH
    from pipeline.market import load_pinnacle_lines
    if Path(PINNACLE_LINES_PATH).exists():
        sched = (
            stints[["GAME_ID", "game_date", "home_team", "away_team"]]
            .drop_duplicates("GAME_ID")
            .rename(columns={"game_date": "date", "home_team": "home", "away_team": "away"})
        )
        pin = load_pinnacle_lines(
            str(PINNACLE_LINES_PATH),
            schedule_df=sched if not sched.empty else None,
        )
        print(f"  merged {len(pin)} Pinnacle line entries from {PINNACLE_LINES_PATH}")
        odds_dict.update(pin)

    diagnose_loaded_data(stints, odds_dict)

    print("Running walk-forward backtest...")
    results = run_multi_year_backtest_walkforward(
        stints,
        odds_dict=odds_dict,
        n_tuning_trials_elo=args.elo_trials,
        n_tuning_trials_hier=args.hier_trials,
        n_tuning_trials_meta=args.meta_trials,
        rolling_window_size=args.window,
        fast_tuning=args.fast_tuning,
    )

    if results.empty:
        print("No backtest results produced.")
        return 1

    benchmark_results(results)

    if "STAKE_MODERATE" not in results.columns:
        results = add_all_profile_columns(results)

    profile_stats = {}
    recommended = "moderate"
    best_roi = -1e9
    for profile in ("conservative", "moderate", "aggressive"):
        stats = benchmark_betting_roi(results, profile=profile)
        profile_stats[profile] = stats
        roi = stats.get("roi", float("nan"))
        ci_lo = stats.get("ci_lo", float("nan"))
        if np.isfinite(roi) and np.isfinite(ci_lo) and ci_lo > 0 and roi > best_roi:
            best_roi = roi
            recommended = profile

    print(f"\nRecommended stake profile (blind walk-forward): {recommended}")

    diag_dir = ROOT / "analysis_plots"
    run_backtest_diagnostics(results, save_dir=diag_dir, show_graphs=True)

    try:
        from pipeline.diagnostics import generate_betting_plots
        generate_betting_plots(results, save_dir=diag_dir)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ Betting plots skipped: {e}")

    best_edge, grid = OPTIMAL_BET_EDGE, pd.DataFrame()
    if uses_edge_gates():
        best_edge, grid = grid_search_bet_edge(results)
        print(f"\nGrid search GOOD_BET_EDGE (current={OPTIMAL_BET_EDGE}):")
        print(grid.to_string(index=False) if not grid.empty else "  insufficient bets")
        print(f"Recommended OPTIMAL_BET_EDGE: {best_edge}")
    else:
        print(f"\nSkipping edge grid search (BET_SELECTION_MODE={BET_SELECTION_MODE})")

    from pipeline.artifacts import save_backtest_metadata, validate_backtest_df
    stale = validate_backtest_df(results)
    if stale:
        print(f"  ⚠️ Backtest schema notes: {'; '.join(stale)}")

    tuning_out = ROOT / "state" / "tuning_results.json"
    tuning_out.parent.mkdir(parents=True, exist_ok=True)
    blind_roi = {
        p: profile_stats.get(p, {}).get("roi")
        for p in ("conservative", "moderate", "aggressive")
    }
    blind_ci = {
        p: {"lo": profile_stats.get(p, {}).get("ci_lo"), "hi": profile_stats.get(p, {}).get("ci_hi")}
        for p in ("conservative", "moderate", "aggressive")
    }
    edge_thr = float(results["EDGE_THRESHOLD"].iloc[-1]) if "EDGE_THRESHOLD" in results.columns else best_edge
    fav_dec = float(results["MAX_FAVORITE_DECIMAL"].iloc[-1]) if "MAX_FAVORITE_DECIMAL" in results.columns else 1.45
    ou_thr = float(results["OU_MIN_EDGE"].iloc[-1]) if "OU_MIN_EDGE" in results.columns else float(OU_MIN_EDGE)
    last_season = results["simulated_season_window"].iloc[-1] if "simulated_season_window" in results.columns else None
    from pipeline.config import (
        CONFIDENCE_SELECTION_MODE,
        MIN_CONFIDENCE_SCORE,
        MIN_ML_WIN_PCT,
        ARTIFACT_SCHEMA_VERSION,
    )
    tuning_out.write_text(json.dumps({
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "optimal_bet_edge": best_edge,
        "walkforward_edge_threshold": edge_thr,
        "max_favorite_decimal": fav_dec,
        "ou_min_edge": ou_thr,
        "min_confidence_score": float(results["MIN_CONFIDENCE_SCORE"].iloc[-1]) if "MIN_CONFIDENCE_SCORE" in results.columns else float(MIN_CONFIDENCE_SCORE),
        "min_ml_win_pct": float(results["MIN_ML_WIN_PCT"].iloc[-1]) if "MIN_ML_WIN_PCT" in results.columns else float(MIN_ML_WIN_PCT),
        "recommended_profile": recommended,
        "spread_calib_window": SPREAD_CALIB_WINDOW,
        "last_walkforward_season": last_season,
        "config_flags": {
            "BET_SELECTION_MODE": BET_SELECTION_MODE,
            "CONFIDENCE_SELECTION_MODE": CONFIDENCE_SELECTION_MODE,
        },
        "blind_ats_roi": blind_roi,
        "blind_roi_ci": blind_ci,
        "stake_profiles": {
            p: PROFILES[p] for p in PROFILES
        },
        "confidence_calibrator_paths": {"ats": str(ROOT / "state" / "bet_calibrator.pkl")},
        "grid": grid.to_dict(orient="records") if not grid.empty else [],
    }, indent=2))
    save_backtest_metadata(extra={"n_games": int(len(results)), "last_walkforward_season": last_season})
    print(f"Saved tuning results to {tuning_out}")

    out = ROOT / "backtest_results.csv"
    results.to_csv(out, index=False)
    print(f"\nSaved {len(results)} rows to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
