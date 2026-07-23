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

from pipeline.config import ROOT as DATA_ROOT, MODERN_ODDS_PATH, PBP_2026_PATH, V3_DATA_PATHS
from pipeline.config import OPTIMAL_BET_EDGE, SPREAD_CALIB_WINDOW, BET_SELECTION_MODE, OU_MIN_EDGE
from pipeline.bet_selection import uses_edge_gates
from pipeline.game_results import attach_canonical_finals, build_canonical_games
from pipeline.ingest import convert_v3_pbp, convert_new_pbp
from pipeline.preprocess import preprocess_pbp, compute_zone_calibration
from pipeline.stints import build_stints
from pipeline.market import load_modern_odds
from pipeline.backtest import run_multi_year_backtest_walkforward
from pipeline.metrics import grid_search_bet_edge, benchmark_results, benchmark_betting_roi, add_all_profile_columns
from pipeline.diagnostics import diagnose_loaded_data, run_backtest_diagnostics
from pipeline.stake_profiles import PROFILES


def read_csv_fast(path):
    """Read a (potentially huge) CSV as fast as available.

    Tries the pyarrow CSV engine first (multi-threaded, ~3-6x faster on the
    300MB combined-stats file), then falls back to the C engine. Output dtypes
    match the default reader closely enough for the converters downstream.
    """
    try:
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, low_memory=False)


def _stints_cache_key(paths, quick, walkforward_zone_pps, assist_split):
    """Signature over source files (mtime+size) + params + schema versions.

    Task 004: including the preprocessing/feature/market-snapshot/validation
    schema versions means any change to preprocessing, stint-building, or
    date/score-repair logic (which bumps a version in pipeline/config.py)
    automatically invalidates every stale ``stints_cache_*`` file instead of
    silently reusing one built under the old (leaky) semantics.
    """
    import hashlib
    from pipeline.config import (
        FEATURE_SCHEMA_VERSION,
        MARKET_SNAPSHOT_SCHEMA_VERSION,
        PREPROCESSING_SCHEMA_VERSION,
        VALIDATION_SCHEMA_VERSION,
    )
    h = hashlib.md5()
    for season, path in sorted(paths, key=lambda kv: kv[0]):
        p = Path(path)
        if p.exists():
            stt = p.stat()
            h.update(f"{season}:{p.name}:{int(stt.st_mtime)}:{stt.st_size}".encode())
    h.update(f"q={quick};wzpps={walkforward_zone_pps};as={assist_split};zones=v2".encode())
    h.update(
        f"prep={PREPROCESSING_SCHEMA_VERSION};feat={FEATURE_SCHEMA_VERSION};"
        f"mkt={MARKET_SNAPSHOT_SCHEMA_VERSION};val={VALIDATION_SCHEMA_VERSION}".encode()
    )
    return h.hexdigest()[:16]


def load_stints(quick: bool = False, walkforward_zone_pps: bool = True,
                use_cache: bool = True) -> pd.DataFrame:
    """Load and stitch historical stint data from available PBP files.

    V3 PBP files carry no date column, so we recover real per-game dates from
    the odds/schedule file (see pipeline.dates) before assembling the stints.
    Without this, every V3 season collapses onto a single ``YYYY-01-01`` date,
    which breaks odds matching and all date-derived features.
    """
    from pipeline.config import name_to_id, ASSIST_SPLIT, STATE_DIR
    from pipeline.dates import load_odds_schedule, attach_real_dates

    paths_all = list(V3_DATA_PATHS.items())
    if PBP_2026_PATH.exists():
        paths_all.append((2025, PBP_2026_PATH))

    # ── Fast path: reuse a cached, fully-assembled stints DataFrame ──
    cache_path = None
    if use_cache:
        key = _stints_cache_key(paths_all, quick, walkforward_zone_pps, ASSIST_SPLIT)
        cache_path = Path(STATE_DIR) / f"stints_cache_{key}.pkl"
        if cache_path.exists():
            try:
                print(f"  ⚡ loading cached stints from {cache_path.name} ...")
                cached = pd.read_pickle(cache_path)
                print(f"  ✅ cache hit: {len(cached):,} stint rows")
                return cached
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ cache read failed ({e}); rebuilding from source")

    # Load the schedule once (used to recover dates for V3 seasons).
    odds_path = MODERN_ODDS_PATH if MODERN_ODDS_PATH.exists() else ROOT / "all_odds.csv"
    sched = None
    try:
        if Path(odds_path).exists():
            sched = load_odds_schedule(str(odds_path))
            print(f"  loaded schedule for date recovery: {len(sched)} games "
                  f"({sched['date'].min().date()} -> {sched['date'].max().date()})")
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ could not load schedule for date recovery: {e}")
        sched = None

    frames = []
    paths = paths_all

    # Walk-forward zone PPS: each season's xPoints uses the *prior* season's
    # realized points-per-shot per zone (leak-free), instead of fixed constants.
    prev_zone_calib = None
    for season, path in sorted(paths, key=lambda kv: kv[0]):
        if not Path(path).exists():
            print(f"  skip missing {path}")
            continue
        print(f"  loading {path} ...")
        raw = read_csv_fast(path)
        canonical_games = None
        if season >= 2025:
            # Task 006/014: build the canonical one-row-per-game outcome
            # table from the *raw* PBP before any conversion/filtering, so
            # canonical finals can override stint-summed labels below.
            try:
                canonical_games = build_canonical_games(raw)
            except ValueError as e:  # raw columns not in the expected shape
                print(f"  ⚠️ could not build canonical games for {path}: {e}")
                canonical_games = None
            df = convert_new_pbp(raw, name_to_id=name_to_id)
        else:
            df = convert_v3_pbp(raw)
        df = preprocess_pbp(df, compute_xpoints=True,
                            zone_calib=prev_zone_calib if walkforward_zone_pps else None)
        if walkforward_zone_pps:
            prev_zone_calib = compute_zone_calibration(df)
        st = build_stints(df, assist_split=ASSIST_SPLIT)
        st["season"] = season + 1
        if canonical_games is not None:
            st = attach_canonical_finals(st, canonical_games)

        # Recover real dates for date-less V3 seasons (season < 2025).
        if season < 2025 and sched is not None and not st.empty:
            n_dates = st["game_date"].dropna().dt.normalize().nunique() if "game_date" in st else 0
            if n_dates <= 1:  # the broken constant-date case
                st = attach_real_dates(st, sched, season_start_year=season)

        frames.append(st)
        if quick and len(frames) >= 2:
            break

    if not frames:
        raise FileNotFoundError("No PBP data found under basketballData/")

    all_stints = pd.concat(frames, ignore_index=True)
    all_stints["game_date"] = pd.to_datetime(all_stints["game_date"], errors="coerce")
    all_stints = all_stints.sort_values(["game_date", "GAME_ID", "stint_id"]).reset_index(drop=True)

    if use_cache and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            all_stints.to_pickle(cache_path)
            print(f"  💾 cached stints -> {cache_path.name} ({len(all_stints):,} rows)")
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ cache write failed: {e}")
    return all_stints


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use only 2 seasons for smoke test")
    parser.add_argument("--elo-trials", type=int, default=30)
    parser.add_argument("--hier-trials", type=int, default=30)
    parser.add_argument("--meta-trials", type=int, default=40)
    parser.add_argument("--window", type=int, default=3)
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

    # Supplement with full-season 2026 Pinnacle lines (matched to the schedule
    # we just loaded, so home/away + dates are authoritative).
    from pipeline.config import PINNACLE_LINES_PATH
    from pipeline.market import load_pinnacle_lines
    if Path(PINNACLE_LINES_PATH).exists():
        sched_2026 = (
            stints[stints["season"] == 2026][["GAME_ID", "game_date", "home_team", "away_team"]]
            .drop_duplicates("GAME_ID")
            .rename(columns={"game_date": "date", "home_team": "home", "away_team": "away"})
        )
        pin = load_pinnacle_lines(str(PINNACLE_LINES_PATH),
                                  schedule_df=sched_2026 if not sched_2026.empty else None)
        print(f"  merged {len(pin)} Pinnacle 2026 line entries from {PINNACLE_LINES_PATH}")
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
