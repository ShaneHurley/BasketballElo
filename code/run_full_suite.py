#!/usr/bin/env python3
"""Ordered full-run suite for BasketballElo/code (stages 0–8).

Walk through calibration with checkpoints and anti-overfit stops.
Never trains from quarantined ``newest data/`` exports — fresh engines only.

Examples
--------
  # Mini interactive walkthrough (pause at review)
  python run_full_suite.py --pause

  # Non-interactive mini
  python run_full_suite.py --yes --stints auto

  # Resume from stage 5 using a prior run dir
  python run_full_suite.py --run-dir ../output/20260730_120000_suite --from-stage 5 --yes

  # Force rebuild stints after changing basketballData/data
  python run_full_suite.py --stints rebuild --yes
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = CODE_ROOT.parent.parent  # BasketballElo/code → BasketballElo → repo
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

STAGE_NAMES = {
    0: "toggles",
    1: "stints",
    2: "integrity",
    3: "engines_walkforward",  # Elo/Hier/meta/calib inside walk-forward
    4: "calibrators",          # alias note: produced inside stage 3/5 fold loop
    5: "walkforward",
    6: "review",
    7: "policy",
    8: "persist",
}


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text()) if path.exists() else {}


def _checkpoint_dir(run_dir: Path) -> Path:
    d = run_dir / "checkpoints"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _mark_stage(run_dir: Path, stage: int, status: str, extra: dict | None = None) -> None:
    ck = _checkpoint_dir(run_dir)
    payload = {
        "stage": stage,
        "name": STAGE_NAMES.get(stage, str(stage)),
        "status": status,
        "finished_at": _utc_now(),
        **(extra or {}),
    }
    _write_json(ck / f"stage_{stage:02d}_{STAGE_NAMES.get(stage, stage)}.json", payload)
    manifest = _read_json(ck / "manifest.json")
    manifest.setdefault("stages", {})[str(stage)] = payload
    manifest["updated_at"] = _utc_now()
    _write_json(ck / "manifest.json", manifest)


def _print_scoreboard(results: pd.DataFrame) -> dict:
    """Primary: MAE → Brier/ECE → CLV → ATS (secondary)."""
    from pipeline.metrics import compute_metrics, add_clv_columns
    from pipeline.config import LOG_LOSS_IN_REVIEW
    from pipeline.calibration_metrics import murphy_brier_decomposition, compute_log_loss

    if results is None or results.empty:
        print("  (no results)")
        return {}
    df = results.copy()
    try:
        df = add_clv_columns(df)
    except Exception:
        pass
    m = compute_metrics(df)
    print("\n=== SCOREBOARD (locked folds) — MAE / ECE / CLV first; ATS secondary ===")
    if not m.empty:
        cols = [c for c in ("season", "spread_mae", "ece", "brier", "ats_pct", "roi", "n_bets") if c in m.columns]
        print(m[cols].to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    clv = np.nan
    n_act = 0
    n_clv_capable = 0
    n_finite_clv = 0
    if "DIRECTION" in df.columns:
        active = df[df["DIRECTION"].astype(str).ne("Pass")]
    else:
        active = df
    n_act = len(active)
    if n_act and "DECISION_SPREAD" in active.columns and "CLOSING_SPREAD" in active.columns:
        dec = pd.to_numeric(active["DECISION_SPREAD"], errors="coerce")
        clo = pd.to_numeric(active["CLOSING_SPREAD"], errors="coerce")
        capable = (dec.notna()) & (clo.notna()) & (dec != clo)
        n_clv_capable = int(capable.sum())
        if "CLV" in active.columns:
            clv_s = pd.to_numeric(active.loc[capable, "CLV"], errors="coerce")
            n_finite_clv = int(clv_s.notna().sum())
            if n_finite_clv:
                clv = float(clv_s.mean())
        print(
            f"  CLV (decision≠close only): mean={clv:.4f}  "
            f"finite={n_finite_clv}/{n_clv_capable} capable  "
            f"(actionable={n_act})"
        )
    elif "CLV" in df.columns:
        clv_all = pd.to_numeric(active["CLV"], errors="coerce")
        n_finite_clv = int(clv_all.notna().sum())
        clv = float(clv_all.mean()) if n_finite_clv else np.nan
        print(f"  mean CLV (actionable): {clv:.4f}")

    murphy: dict = {}
    logloss = np.nan
    y_col = next((c for c in ("HIT", "COVERED", "ATS_HIT") if c in df.columns), None)
    p_col = next(
        (c for c in ("SPREAD_COVER_PROB", "COVER_PROB", "P_COVER") if c in df.columns),
        None,
    )
    if y_col and p_col:
        act = df[df["DIRECTION"].astype(str).ne("Pass")] if "DIRECTION" in df.columns else df
        y = pd.to_numeric(act[y_col], errors="coerce")
        p = pd.to_numeric(act[p_col], errors="coerce")
        mask = y.notna() & p.notna()
        if int(mask.sum()) >= 10:
            murphy = murphy_brier_decomposition(y[mask].values, p[mask].values)
            print(
                f"  Murphy Brier: REL={murphy['reliability']:.4f} "
                f"RES={murphy['resolution']:.4f} UNC={murphy['uncertainty']:.4f} "
                f"(Brier={murphy['brier']:.4f}, n={murphy['n']})"
            )
            if LOG_LOSS_IN_REVIEW:
                logloss = compute_log_loss(y[mask].values, p[mask].values)
                print(f"  log-loss (actionable): {logloss:.4f}")

    seasons = m[m["season"] != "ALL"] if not m.empty and "season" in m.columns else m
    stab = {}
    if seasons is not None and not seasons.empty and "spread_mae" in seasons.columns:
        mae = pd.to_numeric(seasons["spread_mae"], errors="coerce")
        ece = pd.to_numeric(seasons["ece"], errors="coerce") if "ece" in seasons.columns else pd.Series(dtype=float)
        stab = {
            "spread_mae_mean": float(mae.mean()),
            "spread_mae_std": float(mae.std(ddof=0)) if len(mae) else np.nan,
            "ece_mean": float(ece.mean()) if ece.notna().any() else np.nan,
            "ece_std": float(ece.std(ddof=0)) if ece.notna().any() and len(ece) > 1 else 0.0,
            "mean_clv": clv,
            "n_actionable": n_act,
            "n_clv_capable": n_clv_capable,
            "n_finite_clv": n_finite_clv,
            "murphy": murphy,
            "log_loss": logloss,
        }
        score = stab["spread_mae_mean"] + 0.5 * (stab["spread_mae_std"] or 0.0)
        if np.isfinite(stab.get("ece_mean", np.nan)):
            score += stab["ece_mean"] + 0.5 * (stab.get("ece_std") or 0.0)
        stab["stability_score"] = score
        print(
            f"  stability: MAE {stab['spread_mae_mean']:.3f}±{stab['spread_mae_std']:.3f}  "
            f"score={score:.3f}"
        )
    return {"metrics": m.to_dict(orient="records") if not m.empty else [], "stability": stab, "murphy": murphy}


def _anti_overfit_gate(results: pd.DataFrame, run_dir: Path | None = None) -> tuple[bool, str]:
    """Compare path MAE/ECE to persisted review_baseline when available."""
    from pipeline.metrics import compute_metrics
    from pipeline.selection_bias import (
        write_path_variance,
        compare_to_baseline,
        path_variance_summary,
        season_metric_paths,
    )

    if results is None or results.empty:
        return False, "no results to review"
    m = compute_metrics(results)
    seasons = m[m["season"] != "ALL"] if "season" in m.columns else m
    if seasons.empty or "spread_mae" not in seasons.columns:
        return True, "insufficient season rows — proceed with caution"
    mae = pd.to_numeric(seasons["spread_mae"], errors="coerce")
    msg_parts = []
    if mae.std(ddof=0) > 3.0:
        msg_parts.append(
            f"WARNING: high MAE variance across seasons (std={mae.std(ddof=0):.2f})"
        )

    path_sum = path_variance_summary(season_metric_paths(results))
    if "ece" in seasons.columns:
        path_sum["ece_mean"] = float(pd.to_numeric(seasons["ece"], errors="coerce").mean())
    path_sum["mae_mean"] = float(mae.mean()) if mae.notna().any() else np.nan
    if run_dir is not None:
        write_path_variance(run_dir, results)
        base_path = _checkpoint_dir(run_dir) / "review_baseline.json"
        baseline = None
        # Look for a prior sibling baseline under output root (optional).
        if base_path.exists():
            try:
                loaded = json.loads(base_path.read_text())
                # Only compare if this looks like a *prior* run baseline with stability.
                if loaded.get("stability") or loaded.get("mae_mean"):
                    baseline = loaded
            except Exception:
                baseline = None
        ok, cmp_msg = compare_to_baseline(
            {"mae_mean": path_sum.get("mae_mean"), "ece_mean": path_sum.get("ece_mean")},
            baseline.get("stability") if isinstance(baseline, dict) and "stability" in baseline else baseline,
        )
        msg_parts.append(cmp_msg)
        if not ok:
            return False, "; ".join(msg_parts)
    if msg_parts:
        return True, "; ".join(msg_parts) + " — primary metrics are MAE/ECE/CLV; ATS secondary"
    return True, "review OK — primary metrics are MAE/ECE/CLV; ATS is secondary"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="BasketballElo/code ordered full-run suite (stages 0–8)")
    p.add_argument("--data-dir", type=str, default=None)
    p.add_argument("--output-root", type=str, default=None,
                   help="Default: BasketballElo/output")
    p.add_argument("--years", type=str, default=None, help="Comma start years, e.g. 2023,2024,2025")
    p.add_argument("--start-season", type=int, default=None)
    p.add_argument("--end-season", type=int, default=None)
    p.add_argument("--elo-trials", type=int, default=None)
    p.add_argument("--hier-trials", type=int, default=None)
    p.add_argument("--meta-trials", type=int, default=None)
    p.add_argument("--window", type=int, default=4, help="Model-train rolling window (2–5)")
    p.add_argument("--fast-tuning", action="store_true")
    p.add_argument("--label", type=str, default="suite")
    p.add_argument("--stage", type=int, default=None, help="Run only this stage")
    p.add_argument("--from-stage", type=int, default=0, help="Start from this stage (default 0)")
    p.add_argument("--to-stage", type=int, default=8, help="Stop after this stage (default 8)")
    p.add_argument("--run-dir", type=str, default=None, help="Resume into existing run directory")
    p.add_argument("--pause", action="store_true", help="Pause after stage 6 review (Enter to continue)")
    p.add_argument("--yes", action="store_true", help="Non-interactive (skip pause)")
    p.add_argument(
        "--stints",
        type=str,
        default="auto",
        help="auto | rebuild | use:<path>",
    )
    p.add_argument("--stints-cache", type=str, default=None)
    p.add_argument("--force-stints-cache", action="store_true")
    p.add_argument("--no-cache", action="store_true")
    p.add_argument("--skip-integrity-tests", action="store_true",
                   help="Skip pytest leak/integrity smoke (not recommended)")
    p.add_argument(
        "--allow-interpolated-dates",
        action="store_true",
        help="Keep seasons with >5%% interpolated PBP dates (not recommended; poisons odds/CLV)",
    )
    p.add_argument("--venn-abers", action="store_true", help="Enable Venn-Abers filter for this run")
    p.add_argument("--rolling-z", action="store_true", help="Enable rolling league Z features")
    p.add_argument("--skip-odds-gate", action="store_true", help="Skip odds-match fail-closed (debug only)")
    return p


def stage0_toggles(args, run_dir: Path) -> dict:
    from pipeline.data_paths import auto_rolling_window, parse_years_arg

    mini = args.years is None and args.start_season is None and args.end_season is None
    elo = 5 if args.elo_trials is None and mini else (args.elo_trials if args.elo_trials is not None else 30)
    hier = 5 if args.hier_trials is None and mini else (args.hier_trials if args.hier_trials is not None else 30)
    meta = 5 if args.meta_trials is None and mini else (args.meta_trials if args.meta_trials is not None else 40)
    window = int(args.window) if args.window is not None else auto_rolling_window(elo, hier, meta)
    years = parse_years_arg(args.years)
    toggles = {
        "mini": mini,
        "years": years,
        "start_season": args.start_season,
        "end_season": args.end_season,
        "elo_trials": elo,
        "hier_trials": hier,
        "meta_trials": meta,
        "window": window,
        "fast_tuning": bool(args.fast_tuning),
        "stints_mode": args.stints,
        "rating_history_use_all": True,
        "never_use_newest_data_baseline": True,
        "allow_interpolated_dates": bool(getattr(args, "allow_interpolated_dates", False)),
        "use_venn_abers": bool(getattr(args, "venn_abers", False)),
        "use_rolling_league_z": bool(getattr(args, "rolling_z", False)),
        "created_at": _utc_now(),
    }
    _write_json(_checkpoint_dir(run_dir) / "runtime_toggles.json", toggles)
    print("Stage 0 toggles:", json.dumps({k: toggles[k] for k in (
        "mini", "years", "elo_trials", "hier_trials", "meta_trials", "window", "stints_mode"
    )}, indent=2))
    _mark_stage(run_dir, 0, "ok", toggles)
    return toggles


def stage1_stints(args, run_dir: Path, toggles: dict) -> tuple[pd.DataFrame, dict]:
    from pipeline.data_paths import (
        discover_pbp_paths,
        load_player_names,
        resolve_data_dir,
        resolve_odds_path,
        resolve_pinnacle_path,
        select_season_keys,
    )
    from pipeline.stint_loader import (
        StintsCacheMismatchError,
        describe_stints_cache,
        load_stints,
        stints_cache_key,
        resolve_paths_for_stints,
    )
    import pipeline.config as cfg

    data_dir = resolve_data_dir(args.data_dir)
    available = discover_pbp_paths(data_dir)
    if not available:
        raise SystemExit(f"No PBP files under {data_dir}")

    mini = toggles.get("mini", True)
    season_keys = select_season_keys(
        available,
        years=toggles.get("years"),
        start_season=toggles.get("start_season"),
        end_season=toggles.get("end_season"),
        mini=mini,
    )
    paths = [(y, available[y]) for y in season_keys]
    print(f"Data dir: {data_dir}")
    print(f"Seasons (start years): {season_keys}")

    # Parse --stints auto|rebuild|use:path
    stints_arg = (args.stints or "auto").strip()
    stints_cache = Path(args.stints_cache) if args.stints_cache else None
    mode = "auto"
    if stints_arg == "rebuild":
        mode = "rebuild"
    elif stints_arg.startswith("use:"):
        mode = "use"
        stints_cache = Path(stints_arg.split(":", 1)[1])
    elif stints_arg == "use":
        mode = "use"
        if stints_cache is None:
            raise SystemExit("--stints use requires --stints-cache PATH or --stints use:PATH")
    else:
        mode = "auto"

    names_dict, name_to_id = load_player_names(data_dir)
    if names_dict:
        cfg.names_dict = names_dict
        cfg.name_to_id = name_to_id

    odds_path = resolve_odds_path(data_dir, REPO_ROOT)
    paths_all, _ = resolve_paths_for_stints(paths)
    key = stints_cache_key(
        paths_all, quick=False, walkforward_zone_pps=True, assist_split=float(cfg.ASSIST_SPLIT),
    )
    artifacts = run_dir / "artifacts"
    artifacts.mkdir(parents=True, exist_ok=True)
    preview = artifacts / f"stints_cache_{key}.pkl"
    print(describe_stints_cache(preview, current_key=key))

    try:
        stints, stints_meta = load_stints(
            paths=paths,
            quick=False,
            use_cache=not args.no_cache,
            cache_dir=artifacts,
            stints_cache=stints_cache,
            odds_path=odds_path,
            name_to_id=name_to_id,
            force_stints_cache=args.force_stints_cache,
            stints_mode=mode,
        )
    except StintsCacheMismatchError as e:
        raise SystemExit(str(e)) from e

    from pipeline.dates import filter_stints_by_date_quality, season_date_coverage

    allow_interp = bool(toggles.get("allow_interpolated_dates") or getattr(args, "allow_interpolated_dates", False))
    cov = season_date_coverage(stints)
    if not cov.empty:
        print("Date coverage:\n", cov.to_string(index=False))
    stints, cov, dropped = filter_stints_by_date_quality(stints, allow_interpolated=allow_interp)
    if dropped:
        print(
            f"  ⚠️ dropped seasons with poor schedule match (interp): {dropped}. "
            "Pass --allow-interpolated-dates to keep (not recommended)."
        )
        if stints.empty:
            raise SystemExit("All seasons dropped for interpolated dates — fix schedule or pass --allow-interpolated-dates")
    toggles["dropped_interp_seasons"] = dropped
    _write_json(_checkpoint_dir(run_dir) / "date_coverage.json", cov.to_dict(orient="records"))

    payload = {
        "season_keys": season_keys,
        "data_dir": str(data_dir),
        "odds_path": str(odds_path) if odds_path else None,
        "pinnacle_path": str(resolve_pinnacle_path(data_dir, REPO_ROOT) or ""),
        "stints_meta": {k: v for k, v in stints_meta.items() if k != "sidecar"},
        "built_at": stints_meta.get("built_at"),
        "cache_key": stints_meta.get("cache_key"),
        "n_rows": len(stints),
        "stint_seasons": sorted(stints["season"].dropna().unique().tolist()),
    }
    _write_json(_checkpoint_dir(run_dir) / "stints_info.json", payload)
    # Persist stints frame for later stages
    stints_path = artifacts / "suite_stints.pkl"
    stints.to_pickle(stints_path)
    payload["suite_stints_path"] = str(stints_path)
    _write_json(_checkpoint_dir(run_dir) / "stints_info.json", payload)
    _mark_stage(run_dir, 1, "ok", {"n_rows": len(stints), "built_at": payload.get("built_at")})
    return stints, payload


def stage2_integrity(run_dir: Path, skip_tests: bool) -> None:
    """Fail closed on critical leak/integrity tests before tuning."""
    print("Stage 2 — integrity / leak gates (never use newest data/ as baseline)")
    warn_path = REPO_ROOT / "newest data"
    if warn_path.exists():
        print(f"  NOTE: {warn_path} is quarantined — suite will not train from it.")

    if skip_tests:
        print("  ⚠️ --skip-integrity-tests: skipping pytest smoke")
        _mark_stage(run_dir, 2, "skipped")
        return

    tests = [
        "tests/test_leak_registry.py",
        "tests/test_score_date_integrity_gate.py",
        "tests/test_calibration_slice_registry.py",
        "tests/test_spread_calibration_train_live_parity.py",
        "tests/test_two_sided_prices.py",
        "tests/test_data_paths_window.py",
    ]
    # Prefer CODE_ROOT tests (synced)
    cmd = [
        sys.executable, "-m", "pytest", "-q", "--tb=line",
        *[str(CODE_ROOT / t) for t in tests if (CODE_ROOT / t).exists()],
    ]
    env = {
        **dict(**{k: v for k, v in __import__("os").environ.items()}),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "PYTHONPATH": str(CODE_ROOT),
    }
    print("  running:", " ".join(cmd[-6:]), "...")
    proc = subprocess.run(cmd, cwd=str(CODE_ROOT), env=env, capture_output=True, text=True)
    log_path = _checkpoint_dir(run_dir) / "integrity_pytest.log"
    log_path.write_text((proc.stdout or "") + "\n" + (proc.stderr or ""))
    if proc.returncode != 0:
        print(proc.stdout)
        print(proc.stderr)
        _mark_stage(run_dir, 2, "failed", {"returncode": proc.returncode})
        raise SystemExit(
            f"Stage 2 integrity tests failed (exit {proc.returncode}). "
            f"See {log_path}. Fix leaks before calibrating."
        )
    print("  ✅ integrity smoke passed")
    _mark_stage(run_dir, 2, "ok")


def stage3_to_5_walkforward(
    args,
    run_dir: Path,
    toggles: dict,
    stints: pd.DataFrame,
    stints_info: dict,
) -> pd.DataFrame:
    """Stages 3–5: engines + disjoint calibrators + blind walk-forward (one call)."""
    from pipeline.data_paths import resolve_odds_path, resolve_pinnacle_path, resolve_data_dir
    from pipeline.odds_loader import load_odds_dict
    from pipeline.backtest import run_multi_year_backtest_walkforward
    from pipeline.diagnostics import diagnose_loaded_data, assert_odds_coverage, warn_pinnacle_clv_gap
    import pipeline.config as cfg

    if toggles.get("use_venn_abers"):
        cfg.USE_VENN_ABERS_FILTER = True
    if toggles.get("use_rolling_league_z"):
        cfg.USE_ROLLING_LEAGUE_Z = True

    data_dir = resolve_data_dir(args.data_dir or stints_info.get("data_dir"))
    odds_path = resolve_odds_path(data_dir, REPO_ROOT)
    if odds_path is None or not Path(odds_path).exists():
        raise SystemExit("all_odds.csv not found")
    print(f"Loading odds from {odds_path}...")
    pinnacle_path = resolve_pinnacle_path(data_dir, REPO_ROOT)
    sched = (
        stints[["GAME_ID", "game_date", "home_team", "away_team"]]
        .drop_duplicates("GAME_ID")
        .rename(columns={"game_date": "date", "home_team": "home", "away_team": "away"})
    )
    odds_dict, odds_prov = load_odds_dict(
        modern_odds_path=odds_path,
        pinnacle_path=pinnacle_path,
        schedule_df=sched if not sched.empty else None,
        # Quote-level + tip map supplied by callers when available; otherwise tip-proxy.
        quotes_df=toggles.get("quotes_df"),
        tip_utc_map=toggles.get("tip_utc_map"),
    )
    print(
        f"  odds quote_source={odds_prov.get('quote_source')} "
        f"keys={len(odds_dict):,} tip_proxy={odds_prov.get('used_tip_proxy_fallback')}"
    )
    _write_json(_checkpoint_dir(run_dir) / "odds_provenance.json", odds_prov)

    cov = diagnose_loaded_data(stints, odds_dict)
    if not getattr(args, "skip_odds_gate", False):
        assert_odds_coverage(cov)
    pin_gap = warn_pinnacle_clv_gap(cov, pinnacle_path=pinnacle_path)
    _write_json(_checkpoint_dir(run_dir) / "odds_coverage.json", cov.to_dict(orient="records"))
    if pin_gap:
        _write_json(
            _checkpoint_dir(run_dir) / "pinnacle_clv_gap.json",
            {"seasons": pin_gap, "pinnacle_path": str(pinnacle_path) if pinnacle_path else None},
        )

    _mark_stage(run_dir, 3, "started", {"note": "Elo/Hier tune inside walk-forward"})
    _mark_stage(run_dir, 4, "started", {"note": "Disjoint calibrators inside walk-forward folds"})
    _mark_stage(run_dir, 5, "started", {"note": "Blind season simulation"})

    print(
        f"Running walk-forward: window={toggles['window']}  "
        f"trials={toggles['elo_trials']}/{toggles['hier_trials']}/{toggles['meta_trials']}  "
        f"rating_history=all prior"
    )
    results = run_multi_year_backtest_walkforward(
        stints,
        odds_dict=odds_dict,
        n_tuning_trials_elo=toggles["elo_trials"],
        n_tuning_trials_hier=toggles["hier_trials"],
        n_tuning_trials_meta=toggles["meta_trials"],
        rolling_window_size=toggles["window"],
        fast_tuning=toggles.get("fast_tuning", False),
        rating_history_use_all=True,
        use_venn_abers_filter=bool(toggles.get("use_venn_abers")),
    )
    if results is None or results.empty:
        raise SystemExit("Walk-forward produced no results")

    from pipeline.metrics import add_clv_columns
    results = add_clv_columns(results)

    out_csv = run_dir / "betting" / "backtest_results.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(out_csv, index=False)
    results.to_pickle(run_dir / "artifacts" / "backtest_results.pkl")
    _mark_stage(run_dir, 3, "ok")
    _mark_stage(run_dir, 4, "ok")
    _mark_stage(run_dir, 5, "ok", {"n_games": len(results), "results_csv": str(out_csv)})
    return results


def stage6_review(run_dir: Path, results: pd.DataFrame, pause: bool, yes: bool) -> dict:
    print("\nStage 6 — REVIEW (anti-overfit stop)")
    from pipeline.config import REQUIRE_CLV_ON_REVIEW
    from pipeline.metrics import add_clv_columns

    df = add_clv_columns(results.copy())
    summary = _print_scoreboard(df)
    ok, msg = _anti_overfit_gate(df, run_dir=run_dir)
    print(f"  gate: {msg}")

    # CLV integrity: if decision≠close exists among actionable, CLV must be finite.
    act = df
    if "DIRECTION" in df.columns:
        act = df[df["DIRECTION"].astype(str).ne("Pass")]
    n_act = len(act)
    n_dist = 0
    n_clv = 0
    if n_act and "DECISION_SPREAD" in act.columns and "CLOSING_SPREAD" in act.columns:
        dec = pd.to_numeric(act["DECISION_SPREAD"], errors="coerce")
        clo = pd.to_numeric(act["CLOSING_SPREAD"], errors="coerce")
        n_dist = int(((dec.notna()) & (clo.notna()) & (dec != clo)).sum())
        if "CLV" in act.columns:
            n_clv = int(pd.to_numeric(act["CLV"], errors="coerce").notna().sum())
    clv_note = (
        f"CLV: finite={n_clv}/{n_act} actionable; distinct decision≠close={n_dist}"
    )
    print(f"  {clv_note}")
    if REQUIRE_CLV_ON_REVIEW and n_dist > 0 and n_clv == 0:
        ok = False
        msg = (
            "CLV wiring failure: distinct decision/close present but CLV all-NaN. "
            "Refuse promoting ROI/ATS."
        )
        print(f"  ❌ {msg}")
    elif n_dist == 0 and n_act > 0:
        print(
            "  ⚠️ CLV unavailable (single-snapshot odds; Pinnacle T-60 only covers "
            "recent seasons). Do not treat ROI as proven edge."
        )

    review = {
        "ok": ok, "message": msg, "stability": summary.get("stability", {}),
        "clv": {"n_actionable": n_act, "n_distinct_decision_close": n_dist, "n_finite_clv": n_clv},
        "at": _utc_now(),
    }
    _write_json(_checkpoint_dir(run_dir) / "review.json", review)
    # Persist baseline for next-run anti-overfit compare
    baseline = {
        "seasons": summary.get("metrics", []),
        "stability": summary.get("stability", {}),
        "at": _utc_now(),
    }
    _write_json(_checkpoint_dir(run_dir) / "review_baseline.json", baseline)
    _mark_stage(run_dir, 6, "ok" if ok else "warn", review)

    # Emit plots (soft-fail)
    try:
        _emit_suite_plots(run_dir, df)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ plot emission failed: {e}")

    if pause and not yes:
        print("\n*** PAUSE *** Review MAE/ECE/CLV above before policy gates.")
        print("Reject a change that lifts ATS but worsens MAE+ECE on ≥2 seasons.")
        print("Press Enter to continue to stage 7 (policy), or Ctrl-C to stop.")
        try:
            input()
        except EOFError:
            pass
    return review


def _emit_suite_plots(run_dir: Path, results: pd.DataFrame) -> None:
    """Write analysis PNGs under run_dir/plots via analyze_backtest / diagnostics."""
    plots = run_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    try:
        from scripts.analyze_backtest import generate_betting_plots
        generate_betting_plots(results, save_dir=plots)
    except Exception:
        # Fallback: diagnostics module
        from pipeline.diagnostics import run_backtest_diagnostics
        run_backtest_diagnostics(results, save_dir=plots)
    n = len(list(plots.glob("*.png")))
    print(f"  plots: {n} PNG(s) → {plots}")
    if n == 0:
        print("  ⚠️ no plot PNGs written")


def stage7_policy(run_dir: Path, results: pd.DataFrame) -> dict:
    """Policy gates only — never retune models."""
    from pipeline.metrics import (
        grid_search_bet_edge,
        walkforward_edge_threshold,
        add_all_profile_columns,
        benchmark_betting_roi,
    )
    from pipeline.bet_selection import uses_edge_gates
    from pipeline.config import GOOD_BET_EDGE, OPTIMAL_BET_EDGE
    from pipeline.meta_labeling import meta_label_grid, apply_meta_labels

    print("Stage 7 — policy gates (prior graded seasons only; no model retune)")
    df = results.copy()
    if "STAKE_MODERATE" not in df.columns:
        df = add_all_profile_columns(df)

    # P1b: diagnose |edge| collapse — do NOT loosen gates for volume.
    season_col = next(
        (c for c in ("simulated_season_window", "season") if c in df.columns),
        None,
    )
    if "EDGE" in df.columns and season_col is not None:
        print("  |edge| by season (diagnosis only; gates unchanged):")
        for season, g in df.groupby(season_col):
            e = pd.to_numeric(g["EDGE"], errors="coerce").abs()
            n_act = int((g["DIRECTION"].astype(str).ne("Pass")).sum()) if "DIRECTION" in g.columns else 0
            print(
                f"    {season}: median|edge|={e.median():.2f}  "
                f"p90={e.quantile(0.9):.2f}  n_actionable={n_act}"
            )
        print(
            "  NOTE: do not drop min edge to 3.5 globally for volume; "
            "fix predictive edge / CLV first (anti-roadmap)."
        )

    from pipeline.config import WALKFORWARD_EDGE_MIN_FLOOR

    policy = {
        "edge_threshold": float(GOOD_BET_EDGE),
        "optimal_bet_edge": float(OPTIMAL_BET_EDGE),
        "live_edge_floor": float(WALKFORWARD_EDGE_MIN_FLOOR),
    }
    if uses_edge_gates():
        try:
            thr = walkforward_edge_threshold(
                df,
                default=GOOD_BET_EDGE,
                min_floor=WALKFORWARD_EDGE_MIN_FLOOR,
            )
            policy["walkforward_edge_threshold"] = float(thr)
            best_edge, grid = grid_search_bet_edge(df)
            policy["grid_best_edge"] = float(best_edge)
            if grid is not None and not grid.empty:
                grid.to_csv(run_dir / "betting" / "edge_grid.csv", index=False)
        except Exception as e:  # noqa: BLE001
            policy["edge_error"] = str(e)

    # Meta-label grid (policy layer only).
    try:
        grid = meta_label_grid(df)
        if not grid.empty:
            grid.to_csv(run_dir / "betting" / "meta_label_grid.csv", index=False)
            best = grid.sort_values(["ats_pct", "mean_clv"], ascending=False).iloc[0]
            policy["meta_label_best"] = best.to_dict()
            labeled = apply_meta_labels(
                df,
                min_ats_prob=float(best["min_ats_prob"]),
                max_upset_when_fav=float(best["max_upset_when_fav"]),
            )
            labeled.to_csv(run_dir / "betting" / "meta_labeled_results.csv", index=False)
            print(
                f"  meta-label best: ats_prob>={best['min_ats_prob']} "
                f"upset_cap={best['max_upset_when_fav']} n={int(best['n_pass'])}"
            )
    except Exception as e:  # noqa: BLE001
        policy["meta_label_error"] = str(e)

    for profile in ("conservative", "moderate", "aggressive"):
        try:
            policy[f"roi_{profile}"] = benchmark_betting_roi(df, profile=profile)
        except Exception:
            pass
    _write_json(_checkpoint_dir(run_dir) / "policy.json", policy)
    print(json.dumps({k: policy[k] for k in policy if not k.startswith("roi_")}, indent=2, default=str))
    _mark_stage(run_dir, 7, "ok", {k: policy[k] for k in policy if k != "grid"})
    return policy


def stage8_persist(run_dir: Path, results: pd.DataFrame, toggles: dict, stints_info: dict) -> None:
    from pipeline.config import STATE_DIR, ARTIFACT_SCHEMA_VERSION
    from pipeline.experiment_exports import collect_artifacts, write_readme, env_info
    from pipeline.artifacts import save_backtest_metadata

    print("Stage 8 — persist engines / calibrators / README")
    stints_meta = stints_info.get("stints_meta") or {}
    collect_artifacts(
        run_dir,
        stints_cache_path=stints_meta.get("cache_path"),
        config_snapshot={
            "label": "suite",
            "elo_trials": toggles.get("elo_trials"),
            "hier_trials": toggles.get("hier_trials"),
            "meta_trials": toggles.get("meta_trials"),
            "window": toggles.get("window"),
            "fast_tuning": toggles.get("fast_tuning"),
            "years": toggles.get("years"),
            "stints_mode": toggles.get("stints_mode"),
        },
    )
    arts_dir = run_dir / "artifacts"
    collected = sorted(p.name for p in arts_dir.iterdir() if p.is_file()) if arts_dir.exists() else []
    info = {
        **env_info(),
        "timestamp": _utc_now(),
        "label": "suite",
        "data_dir": stints_info.get("data_dir"),
        "odds_path": stints_info.get("odds_path"),
        "pinnacle_path": stints_info.get("pinnacle_path"),
        "season_keys": stints_info.get("season_keys"),
        "stint_seasons": stints_info.get("stint_seasons"),
        "cache_hit": stints_meta.get("cache_hit"),
        "cache_path": stints_meta.get("cache_path"),
        "built_at": stints_info.get("built_at"),
        "n_stints": stints_info.get("n_rows"),
        "n_games": len(results),
        "elo_trials": toggles.get("elo_trials"),
        "hier_trials": toggles.get("hier_trials"),
        "meta_trials": toggles.get("meta_trials"),
        "window": toggles.get("window"),
        "window_source": "suite",
        "fast_tuning": toggles.get("fast_tuning"),
        "paths_used": [(s, p) for s, p in stints_meta.get("paths_used", [])]
        if isinstance(stints_meta.get("paths_used"), list)
        else [],
        "artifact_schema": ARTIFACT_SCHEMA_VERSION,
        "note": (
            "Fresh Elo/engine state from this suite only. "
            "Do not train from quarantined newest data/ baselines."
        ),
        "collected_artifacts": collected,
    }
    write_readme(run_dir, info)
    try:
        save_backtest_metadata(extra={
            "suite_run_dir": str(run_dir),
            "window": toggles.get("window"),
            "built_at_stints": stints_info.get("built_at"),
        })
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ metadata save: {e}")
    # Copy key results into STATE_DIR for daily use
    STATE_DIR.mkdir(parents=True, exist_ok=True)
    src = run_dir / "betting" / "backtest_results.csv"
    if src.exists():
        (STATE_DIR / "suite_backtest_results.csv").write_bytes(src.read_bytes())
    plots = run_dir / "plots"
    n_png = len(list(plots.glob("*.png"))) if plots.exists() else 0
    if n_png == 0 and (run_dir / "betting" / "backtest_results.csv").exists():
        try:
            _emit_suite_plots(run_dir, results)
            n_png = len(list(plots.glob("*.png"))) if plots.exists() else 0
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ stage8 plot retry failed: {e}")
    if n_png == 0:
        print("  ⚠️ plots/ empty after suite — soft-fail checklist")
    _mark_stage(run_dir, 8, "ok", {"run_dir": str(run_dir), "n_plots": n_png})
    print(f"\n✅ Suite complete → {run_dir}")
    print("   Engines/calibrators: artifacts/ and state/")
    print("   Next: review checkpoints/review.json then run_daily.py")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    t0 = time.time()

    from pipeline.data_paths import resolve_output_root
    from pipeline.experiment_exports import create_run_dirs

    output_root = resolve_output_root(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    if args.run_dir:
        run_dir = Path(args.run_dir)
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "artifacts").mkdir(exist_ok=True)
        (run_dir / "betting").mkdir(exist_ok=True)
        (run_dir / "checkpoints").mkdir(exist_ok=True)
    else:
        # create_run_dirs uses graphs/data/...; add checkpoints
        run_dir = create_run_dirs(output_root, label=args.label or "suite")
        _checkpoint_dir(run_dir)

    print(f"Suite run dir: {run_dir}")
    print("Stages:", STAGE_NAMES)
    print("NEVER use quarantined newest data/ as training baseline — fresh walk-forward only.\n")

    from_stage = int(args.from_stage)
    to_stage = int(args.to_stage)
    only = args.stage
    stages = [only] if only is not None else list(range(from_stage, to_stage + 1))

    toggles = _read_json(_checkpoint_dir(run_dir) / "runtime_toggles.json")
    stints_info = _read_json(_checkpoint_dir(run_dir) / "stints_info.json")
    stints = None
    results = None

    def need(s: int) -> bool:
        return s in stages

    if need(0) or not toggles:
        toggles = stage0_toggles(args, run_dir)

    if need(1):
        stints, stints_info = stage1_stints(args, run_dir, toggles)
    elif any(need(s) for s in (3, 4, 5, 6, 7, 8)):
        sp = stints_info.get("suite_stints_path")
        if sp and Path(sp).exists():
            stints = pd.read_pickle(sp)
            print(f"  loaded stints from checkpoint ({len(stints):,} rows)")
        else:
            raise SystemExit("Missing suite_stints.pkl — run stage 1 first")

    if need(2):
        stage2_integrity(run_dir, skip_tests=args.skip_integrity_tests)

    if any(need(s) for s in (3, 4, 5)):
        if stints is None:
            raise SystemExit("stints required for walk-forward")
        results = stage3_to_5_walkforward(args, run_dir, toggles, stints, stints_info)
    elif any(need(s) for s in (6, 7, 8)):
        rp = run_dir / "artifacts" / "backtest_results.pkl"
        if rp.exists():
            results = pd.read_pickle(rp)
        else:
            csvp = run_dir / "betting" / "backtest_results.csv"
            if csvp.exists():
                results = pd.read_csv(csvp)
            else:
                raise SystemExit("No backtest results — run stages 3–5 first")

    if need(6):
        stage6_review(run_dir, results, pause=args.pause, yes=args.yes)

    if need(7):
        stage7_policy(run_dir, results)

    if need(8):
        stage8_persist(run_dir, results, toggles, stints_info)

    print(f"\nWall time: {time.time() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
