#!/usr/bin/env python3
"""
Training / testing experiment harness (not for daily live use).

Default (no flags) = mini run: newest 2 seasons, 5/5/5 Optuna trials,
auto rolling window biased high (window=5).

Usage examples:
  python run_training_experiment.py
  python run_training_experiment.py --years 2021,2022,2023,2024 --elo-trials 30 --hier-trials 30 --meta-trials 40
  python run_training_experiment.py --artifacts-dir BasketballElo/output/.../artifacts --years 2022,2023
  python run_training_experiment.py --data-dir /path/to/data --years 2023,2024
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.data_paths import (
    auto_rolling_window,
    discover_pbp_paths,
    load_player_names,
    parse_years_arg,
    resolve_data_dir,
    resolve_odds_path,
    resolve_output_root,
    resolve_pinnacle_path,
    resolve_player_list_path,
    select_season_keys,
)
from pipeline.experiment_exports import (
    collect_artifacts,
    create_run_dirs,
    env_info,
    export_betting_csvs,
    export_elo_and_combos,
    write_readme,
)
from pipeline.stint_loader import load_stints


def _hash_file(path: Path) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _apply_artifacts_reuse(artifacts_dir: Path | None) -> dict:
    """Copy prior calibrations / pickles into STATE_DIR when reusing a run."""
    from pipeline.config import STATE_DIR

    meta = {"reused": False, "copied": []}
    if artifacts_dir is None:
        return meta
    src = Path(artifacts_dir)
    if not src.is_dir():
        print(f"  ⚠️ --artifacts-dir not found: {src}")
        return meta

    STATE_DIR.mkdir(parents=True, exist_ok=True)
    for f in src.iterdir():
        if not f.is_file():
            continue
        if f.suffix.lower() in {".pkl", ".json"} or f.name.startswith("stints_cache"):
            dest = STATE_DIR / f.name
            try:
                shutil.copy2(f, dest)
                meta["copied"].append(f.name)
            except Exception as e:  # noqa: BLE001
                print(f"  ⚠️ could not restore artifact {f.name}: {e}")
    meta["reused"] = bool(meta["copied"])
    if meta["reused"]:
        print(f"  Restored {len(meta['copied'])} artifacts from {src}")
    return meta


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Walk-forward training experiment with timestamped output bundle.",
    )
    p.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="PBP/odds directory (required on non-Mac; Mac defaults to basketballData/data).",
    )
    p.add_argument(
        "--output-root",
        type=str,
        default=None,
        help="Override output root (default BasketballElo/output).",
    )
    p.add_argument(
        "--years",
        type=str,
        default=None,
        help="Comma-separated season start years, e.g. 2021,2022,2023. "
             "Omit with no --start/--end for mini (newest 2).",
    )
    p.add_argument("--start-season", type=int, default=None, help="Inclusive season start year.")
    p.add_argument("--end-season", type=int, default=None, help="Inclusive season start year.")
    p.add_argument("--elo-trials", type=int, default=None, help="Optuna trials for Elo K tuning.")
    p.add_argument("--hier-trials", type=int, default=None, help="Optuna trials for hierarchical.")
    p.add_argument("--meta-trials", type=int, default=None, help="Optuna trials for meta model.")
    p.add_argument(
        "--window",
        type=int,
        default=None,
        help="Rolling window seasons override. If omitted, auto from trial budget "
             "(higher window when trials are low).",
    )
    p.add_argument("--fast-tuning", action="store_true", help="Lighter inner CV during meta tuning.")
    p.add_argument("--label", type=str, default=None, help="Suffix for run folder name.")
    p.add_argument(
        "--stints-cache",
        type=str,
        default=None,
        help="Path to an existing stints_cache_*.pkl to load.",
    )
    p.add_argument(
        "--artifacts-dir",
        type=str,
        default=None,
        help="Prior run artifacts/ folder (restores calibrations/stints pickles).",
    )
    p.add_argument("--elo-calib", type=str, default=None, help="Path to elo_calibrator.pkl to restore.")
    p.add_argument("--bet-calib", type=str, default=None, help="Path to bet_calibrator.pkl to restore.")
    p.add_argument("--tuning-json", type=str, default=None, help="Path to tuning_results.json to restore.")
    p.add_argument("--no-cache", action="store_true", help="Disable stints pickle cache.")
    p.add_argument("--skip-elo-export", action="store_true", help="Skip Elo/combo CSV rebuild (faster).")
    p.add_argument("--skip-plots", action="store_true", help="Skip graph generation.")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()

    # ── Data dir ──
    try:
        data_dir = resolve_data_dir(args.data_dir)
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    output_root = resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    available = discover_pbp_paths(data_dir)
    if not available:
        print(f"ERROR: No PBP files found under {data_dir}", file=sys.stderr)
        return 2
    print(f"Data dir: {data_dir}")
    print(f"Available seasons (start years): {list(available.keys())}")

    years = parse_years_arg(args.years)
    mini = years is None and args.start_season is None and args.end_season is None
    try:
        season_keys = select_season_keys(
            available,
            years=years,
            start_season=args.start_season,
            end_season=args.end_season,
            mini=mini,
        )
    except ValueError as e:
        print(f"ERROR: {e}", file=sys.stderr)
        return 2

    # Mini defaults for trials
    elo_trials = 5 if args.elo_trials is None and mini else (args.elo_trials if args.elo_trials is not None else 30)
    hier_trials = 5 if args.hier_trials is None and mini else (args.hier_trials if args.hier_trials is not None else 30)
    meta_trials = 5 if args.meta_trials is None and mini else (args.meta_trials if args.meta_trials is not None else 40)

    if args.window is not None:
        window = args.window
        window_source = "cli"
    else:
        window = auto_rolling_window(elo_trials, hier_trials, meta_trials)
        window_source = "auto_from_trials"

    label = args.label or ("mini" if mini else "full")
    run_dir = create_run_dirs(output_root, label=label)
    print(f"Run output → {run_dir}")
    print(
        f"Seasons={season_keys}  trials elo/hier/meta={elo_trials}/{hier_trials}/{meta_trials}  "
        f"window={window} ({window_source})"
    )

    # ── Restore prior artifacts / explicit calib paths ──
    arts_reuse = _apply_artifacts_reuse(Path(args.artifacts_dir) if args.artifacts_dir else None)
    from pipeline.config import STATE_DIR
    for src_arg, dest_name in (
        (args.elo_calib, "elo_calibrator.pkl"),
        (args.bet_calib, "bet_calibrator.pkl"),
        (args.tuning_json, "tuning_results.json"),
    ):
        if src_arg:
            src = Path(src_arg)
            if src.exists():
                shutil.copy2(src, STATE_DIR / dest_name)
                print(f"  Restored {dest_name} from {src}")
            else:
                print(f"  ⚠️ missing {src_arg}")

    # Player names from data dir (fixes empty name_to_id when list lives under data/)
    names_dict, name_to_id = load_player_names(data_dir)
    import pipeline.config as cfg
    if names_dict:
        cfg.names_dict = names_dict
        cfg.name_to_id = name_to_id

    odds_path = resolve_odds_path(data_dir, ROOT)
    pinnacle_path = resolve_pinnacle_path(data_dir, ROOT)
    player_list_path = resolve_player_list_path(data_dir, ROOT)

    paths = [(y, available[y]) for y in season_keys]

    # Prefer stints cache inside --artifacts-dir if not explicitly set
    stints_cache = Path(args.stints_cache) if args.stints_cache else None
    if stints_cache is None and args.artifacts_dir:
        arts = Path(args.artifacts_dir)
        caches = sorted(arts.glob("stints_cache_*.pkl"))
        if caches:
            stints_cache = caches[-1]

    print("Loading stints...")
    stints, stints_meta = load_stints(
        paths=paths,
        quick=False,  # year selection already limits seasons
        use_cache=not args.no_cache,
        cache_dir=run_dir / "artifacts",
        stints_cache=stints_cache,
        odds_path=odds_path,
        name_to_id=name_to_id,
    )
    print(f"  stints: {len(stints):,} rows  seasons={sorted(stints['season'].dropna().unique().tolist())}")

    if odds_path is None or not Path(odds_path).exists():
        print("ERROR: all_odds.csv not found (needed for market + date recovery).", file=sys.stderr)
        return 2

    from pipeline.market import load_modern_odds, load_pinnacle_lines
    from pipeline.diagnostics import diagnose_loaded_data, run_backtest_diagnostics, generate_betting_plots
    from pipeline.backtest import run_multi_year_backtest_walkforward
    from pipeline.metrics import (
        grid_search_bet_edge,
        benchmark_results,
        benchmark_betting_roi,
        add_all_profile_columns,
    )
    from pipeline.bet_selection import uses_edge_gates
    from pipeline.config import (
        OPTIMAL_BET_EDGE,
        SPREAD_CALIB_WINDOW,
        BET_SELECTION_MODE,
        OU_MIN_EDGE,
        CONFIDENCE_SELECTION_MODE,
        MIN_CONFIDENCE_SCORE,
        MIN_ML_WIN_PCT,
        ARTIFACT_SCHEMA_VERSION,
    )
    from pipeline.stake_profiles import PROFILES
    from pipeline.artifacts import (
        backtest_config_snapshot,
        save_backtest_metadata,
        validate_backtest_df,
    )

    print(f"Loading odds from {odds_path}...")
    odds_dict = load_modern_odds(str(odds_path))
    if pinnacle_path is not None and Path(pinnacle_path).exists():
        sched = (
            stints[["GAME_ID", "game_date", "home_team", "away_team"]]
            .drop_duplicates("GAME_ID")
            .rename(columns={"game_date": "date", "home_team": "home", "away_team": "away"})
        )
        pin = load_pinnacle_lines(
            str(pinnacle_path),
            schedule_df=sched if not sched.empty else None,
        )
        print(f"  merged {len(pin)} Pinnacle line entries from {pinnacle_path}")
        odds_dict.update(pin)

    diagnose_loaded_data(stints, odds_dict)

    print("Running walk-forward backtest (tune K → Elo calib → blind season eval)...")
    results = run_multi_year_backtest_walkforward(
        stints,
        odds_dict=odds_dict,
        n_tuning_trials_elo=elo_trials,
        n_tuning_trials_hier=hier_trials,
        n_tuning_trials_meta=meta_trials,
        rolling_window_size=window,
        fast_tuning=args.fast_tuning,
    )

    if results is None or results.empty:
        print("No backtest results produced.")
        write_readme(run_dir, {
            **env_info(),
            "label": label,
            "data_dir": str(data_dir),
            "odds_path": str(odds_path),
            "pinnacle_path": str(pinnacle_path) if pinnacle_path else "",
            "player_list_path": str(player_list_path) if player_list_path else "",
            "season_keys": season_keys,
            "stint_seasons": sorted(stints["season"].dropna().unique().tolist()),
            "mini": mini,
            "cache_hit": stints_meta.get("cache_hit"),
            "cache_path": stints_meta.get("cache_path"),
            "n_stints": len(stints),
            "n_games": 0,
            "paths_used": stints_meta.get("paths_used", []),
            "elo_trials": elo_trials,
            "hier_trials": hier_trials,
            "meta_trials": meta_trials,
            "window": window,
            "window_source": window_source,
            "fast_tuning": args.fast_tuning,
            "runtime_sec": round(time.time() - t0, 1),
            "config_snapshot": backtest_config_snapshot(),
        })
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

    # Graphs
    graphs_dir = run_dir / "graphs"
    if not args.skip_plots:
        try:
            run_backtest_diagnostics(results, save_dir=graphs_dir, show_graphs=True)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ diagnostics plots skipped: {e}")
        try:
            generate_betting_plots(results, save_dir=graphs_dir)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ betting plots skipped: {e}")

    # Betting CSVs
    print("Exporting betting CSVs...")
    export_betting_csvs(run_dir, results)

    # Elo / combo CSVs
    if not args.skip_elo_export:
        print("Exporting Elo / combo leaderboards (post-K recalibration replay)...")
        try:
            export_elo_and_combos(
                run_dir,
                stints,
                elo_pkl=STATE_DIR / "latest_elo.pkl",
                hier_pkl=STATE_DIR / "latest_hier.pkl",
                team_elo_pkl=STATE_DIR / "latest_team_elo.pkl",
                names=names_dict,
            )
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ Elo/combo export failed: {e}")

    # Tuning JSON + artifacts
    best_edge, grid = OPTIMAL_BET_EDGE, pd.DataFrame()
    if uses_edge_gates():
        try:
            best_edge, grid = grid_search_bet_edge(results)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ edge grid search skipped: {e}")

    stale = validate_backtest_df(results)
    if stale:
        print(f"  ⚠️ Backtest schema notes: {'; '.join(stale)}")

    edge_thr = float(results["EDGE_THRESHOLD"].iloc[-1]) if "EDGE_THRESHOLD" in results.columns else best_edge
    fav_dec = float(results["MAX_FAVORITE_DECIMAL"].iloc[-1]) if "MAX_FAVORITE_DECIMAL" in results.columns else 1.45
    ou_thr = float(results["OU_MIN_EDGE"].iloc[-1]) if "OU_MIN_EDGE" in results.columns else float(OU_MIN_EDGE)
    last_season = (
        results["simulated_season_window"].iloc[-1]
        if "simulated_season_window" in results.columns
        else None
    )
    tuning_payload = {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "optimal_bet_edge": best_edge,
        "walkforward_edge_threshold": edge_thr,
        "max_favorite_decimal": fav_dec,
        "ou_min_edge": ou_thr,
        "min_confidence_score": (
            float(results["MIN_CONFIDENCE_SCORE"].iloc[-1])
            if "MIN_CONFIDENCE_SCORE" in results.columns
            else float(MIN_CONFIDENCE_SCORE)
        ),
        "min_ml_win_pct": (
            float(results["MIN_ML_WIN_PCT"].iloc[-1])
            if "MIN_ML_WIN_PCT" in results.columns
            else float(MIN_ML_WIN_PCT)
        ),
        "recommended_profile": recommended,
        "spread_calib_window": SPREAD_CALIB_WINDOW,
        "last_walkforward_season": last_season,
        "config_flags": {
            "BET_SELECTION_MODE": BET_SELECTION_MODE,
            "CONFIDENCE_SELECTION_MODE": CONFIDENCE_SELECTION_MODE,
        },
        "blind_ats_roi": {
            p: profile_stats.get(p, {}).get("roi")
            for p in ("conservative", "moderate", "aggressive")
        },
        "blind_roi_ci": {
            p: {
                "lo": profile_stats.get(p, {}).get("ci_lo"),
                "hi": profile_stats.get(p, {}).get("ci_hi"),
            }
            for p in ("conservative", "moderate", "aggressive")
        },
        "stake_profiles": {p: PROFILES[p] for p in PROFILES},
        "experiment": {
            "season_keys": season_keys,
            "elo_trials": elo_trials,
            "hier_trials": hier_trials,
            "meta_trials": meta_trials,
            "window": window,
            "window_source": window_source,
            "mini": mini,
            "run_dir": str(run_dir),
        },
        "grid": grid.to_dict(orient="records") if not grid.empty else [],
        "artifacts_reused": arts_reuse,
    }

    input_manifest = []
    for season, path in paths:
        p = Path(path)
        entry = {"season_start": season, "path": str(p), "exists": p.exists()}
        if p.exists():
            stt = p.stat()
            entry.update({
                "size": stt.st_size,
                "mtime": int(stt.st_mtime),
                "md5": _hash_file(p),
            })
        input_manifest.append(entry)

    config_snap = backtest_config_snapshot()
    config_snap.update({
        "data_dir": str(data_dir),
        "season_keys": season_keys,
        "elo_trials": elo_trials,
        "hier_trials": hier_trials,
        "meta_trials": meta_trials,
        "window": window,
        "window_source": window_source,
        "mini": mini,
        "fast_tuning": args.fast_tuning,
    })

    collect_artifacts(
        run_dir,
        stints_cache_path=stints_meta.get("cache_path"),
        tuning_json=tuning_payload,
        config_snapshot=config_snap,
        input_manifest=input_manifest,
        extra_copy_from=Path(args.artifacts_dir) if args.artifacts_dir else None,
    )
    # Also keep a copy of tuning under state for daily scripts
    (STATE_DIR / "tuning_results.json").write_text(
        json.dumps(tuning_payload, indent=2, default=str), encoding="utf-8",
    )
    save_backtest_metadata(extra={
        "n_games": int(len(results)),
        "last_walkforward_season": last_season,
        "experiment_run_dir": str(run_dir),
    })

    runtime = round(time.time() - t0, 1)
    write_readme(run_dir, {
        **env_info(),
        "label": label,
        "data_dir": str(data_dir),
        "odds_path": str(odds_path),
        "pinnacle_path": str(pinnacle_path) if pinnacle_path else "",
        "player_list_path": str(player_list_path) if player_list_path else "",
        "season_keys": season_keys,
        "stint_seasons": sorted(stints["season"].dropna().unique().tolist()),
        "mini": mini,
        "cache_hit": stints_meta.get("cache_hit"),
        "cache_path": stints_meta.get("cache_path"),
        "n_stints": len(stints),
        "n_games": int(len(results)),
        "paths_used": stints_meta.get("paths_used", []),
        "elo_trials": elo_trials,
        "hier_trials": hier_trials,
        "meta_trials": meta_trials,
        "window": window,
        "window_source": window_source,
        "fast_tuning": args.fast_tuning,
        "runtime_sec": runtime,
        "config_snapshot": config_snap,
    })

    print(f"\nDone in {runtime}s → {run_dir}")
    print(f"  README:   {run_dir / 'README.md'}")
    print(f"  graphs:   {run_dir / 'graphs'}")
    print(f"  data:     {run_dir / 'data'}")
    print(f"  betting:  {run_dir / 'betting'}")
    print(f"  artifacts:{run_dir / 'artifacts'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
