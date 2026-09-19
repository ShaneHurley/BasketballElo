"""Export Elo/combo/betting CSVs and experiment README for training runs."""
from __future__ import annotations

import json
import platform
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import DEFAULT_LEAGUE_XPPP, STATE_DIR
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.ratings import PlayerRatingTracker
from pipeline.stint_context import build_stint_context
from pipeline.team_elo import TeamEloTracker
from pipeline.utils import _parse_player_string


def create_run_dirs(output_root: Path, label: str = "mini") -> Path:
    """Create timestamped experiment bundle directories; return run root."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_root / f"{stamp}_{label}"
    for sub in ("graphs", "data/elo", "data/combos", "betting", "artifacts"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    return run_dir


def write_readme(run_dir: Path, info: dict[str, Any]) -> Path:
    """Write experiment README with full environment / data / trial spec."""
    lines = [
        "# Training Experiment Run",
        "",
        f"Generated: {info.get('timestamp', '')}",
        f"Label: `{info.get('label', '')}`",
        f"Run directory: `{run_dir}`",
        "",
        "## Environment",
        "",
        f"- Platform: `{info.get('platform', '')}`",
        f"- Python: `{info.get('python', '')}`",
        f"- Hostname: `{info.get('hostname', '')}`",
        f"- Repo: `{info.get('repo_root', '')}`",
        f"- Git revision: `{info.get('git_rev', 'n/a')}`",
        f"- Wall-clock seconds: `{info.get('runtime_sec', 'n/a')}`",
        "",
        "## Data",
        "",
        f"- Data directory: `{info.get('data_dir', '')}`",
        f"- Odds file: `{info.get('odds_path', '')}`",
        f"- Pinnacle file: `{info.get('pinnacle_path', '')}`",
        f"- Player list: `{info.get('player_list_path', '')}`",
        f"- Season start years requested: `{info.get('season_keys', [])}`",
        f"- Season end-year labels in stints: `{info.get('stint_seasons', [])}`",
        f"- Mini run: `{info.get('mini', False)}`",
        f"- Stints cache hit: `{info.get('cache_hit', False)}`",
        f"- Stints cache path: `{info.get('cache_path', '')}`",
        f"- Stint rows: `{info.get('n_stints', 0)}`",
        f"- Backtest games: `{info.get('n_games', 0)}`",
        "",
        "### Input files used",
        "",
    ]
    for season, path in info.get("paths_used", []):
        lines.append(f"- season_start={season}: `{path}`")
    lines.extend([
        "",
        "## Trials & window",
        "",
        f"- Elo Optuna trials: `{info.get('elo_trials')}`",
        f"- Hierarchical Optuna trials: `{info.get('hier_trials')}`",
        f"- Meta Optuna trials: `{info.get('meta_trials')}`",
        f"- Rolling window: `{info.get('window')}` "
        f"(source: `{info.get('window_source', 'auto')}`)",
        f"- Fast tuning: `{info.get('fast_tuning', False)}`",
        "",
        "## Blind calibration contract",
        "",
        "Walk-forward only: for each held-out season, K/Elo/hier tuning, Elo "
        "margin recalibration, meta/spread/bet calibrators, and gates are fit "
        "on **prior** seasons only; the evaluation season is scored blind "
        "(no leakage into calibrators).",
        "",
        "## Config snapshot",
        "",
        "```json",
        json.dumps(info.get("config_snapshot", {}), indent=2, default=str),
        "```",
        "",
        "## Artifacts for re-run",
        "",
        f"Reuse this run with:",
        "",
        "```bash",
        f"python run_training_experiment.py --artifacts-dir {run_dir / 'artifacts'} \\",
        f"  --years {','.join(str(y) for y in info.get('season_keys', []))}",
        "```",
        "",
        "Subfolders: `graphs/`, `data/elo/`, `data/combos/`, `betting/`, `artifacts/`.",
        "",
    ])
    path = run_dir / "README.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _name_lookup(names: dict | None, pid) -> str:
    if not names:
        return ""
    if pid in names:
        return str(names[pid])
    try:
        ip = int(pid)
        if ip in names:
            return str(names[ip])
    except (TypeError, ValueError):
        pass
    sp = str(pid)
    if sp in names:
        return str(names[sp])
    return ""


def _player_frame(tracker: PlayerRatingTracker, names: dict | None = None) -> pd.DataFrame:
    rows = []
    for pid, pl in tracker.players.items():
        o = float(pl.get("O_mu", 1500))
        d = float(pl.get("D_mu", 1500))
        rows.append({
            "player_id": pid,
            "player_name": _name_lookup(names, pid),
            "O_mu": o,
            "D_mu": d,
            "net": o - d,
            "O_rd": float(pl.get("O_rd", np.nan)),
            "D_rd": float(pl.get("D_rd", np.nan)),
            "D_rim_mu": float(pl.get("D_rim_mu", np.nan)),
            "D_peri_mu": float(pl.get("D_peri_mu", np.nan)),
            "possessions": float(pl.get("Possessions", 0)),
            "games": int(tracker.player_games.get(str(pid), 0)),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("net", ascending=False).reset_index(drop=True)


def _team_frame(tracker: TeamEloTracker) -> pd.DataFrame:
    teams = set(tracker.spread_mu) | set(tracker.total_mu) | set(tracker.games)
    rows = []
    for t in teams:
        rows.append({
            "team": t,
            "spread_mu": float(tracker.spread_mu[t]),
            "total_mu": float(tracker.total_mu[t]),
            "games": int(tracker.games[t]),
        })
    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows).sort_values("spread_mu", ascending=False).reset_index(drop=True)


def _combo_frames(hier: HierarchicalPossessionEngine, names: dict | None = None) -> dict[int, pd.DataFrame]:
    def _name(pid) -> str:
        return _name_lookup(names, pid) or str(pid)

    out: dict[int, pd.DataFrame] = {}
    for size in (1, 2, 3, 5):
        keys = [k for k in hier.off.keys() if len(k) == size]
        keys = list(set(keys) | {k for k in hier.dff.keys() if len(k) == size})
        rows = []
        for key in keys:
            off = float(hier.off.get(key, 0.0))
            dff = float(hier.dff.get(key, 0.0))
            poss = float(hier._combo_poss.get(key, 0.0))
            rows.append({
                "combo_size": size,
                "player_ids": "-".join(str(x) for x in key),
                "player_names": " | ".join(_name(x) for x in key),
                "off": off,
                "def": dff,
                "net": off - dff,
                "possessions": poss,
            })
        df = pd.DataFrame(rows)
        if not df.empty:
            df = df.sort_values(["net", "possessions"], ascending=[False, False]).reset_index(drop=True)
        out[size] = df
    return out


def replay_ratings_by_season(
    stints: pd.DataFrame,
    *,
    elo_cfg: dict | None = None,
    names: dict | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, pd.DataFrame], PlayerRatingTracker, HierarchicalPossessionEngine, TeamEloTracker]:
    """Chronologically update trackers; snapshot after each season.

    Uses post-calibration K config from the saved Elo tracker when provided.
    This rebuild is for export/leaderboards only; backtest scoring remains blind.
    """
    elo = PlayerRatingTracker(config=elo_cfg, league_xppp=DEFAULT_LEAGUE_XPPP)
    hier = HierarchicalPossessionEngine()
    team = TeamEloTracker()

    player_snaps: list[pd.DataFrame] = []
    team_snaps: list[pd.DataFrame] = []
    combo_snaps: dict[int, list[pd.DataFrame]] = {1: [], 2: [], 3: [], 5: []}

    if stints.empty or "season" not in stints.columns:
        return pd.DataFrame(), pd.DataFrame(), {1: pd.DataFrame(), 2: pd.DataFrame(), 3: pd.DataFrame(), 5: pd.DataFrame()}, elo, hier, team

    for season, season_df in stints.groupby("season", sort=True):
        for game_id, group in season_df.groupby("GAME_ID", sort=False):
            home = group["home_team"].iloc[0] if "home_team" in group.columns else None
            away = group["away_team"].iloc[0] if "away_team" in group.columns else None
            act_h = float(group.get("HOME_SCORE_END", pd.Series([np.nan])).max())
            act_a = float(group.get("AWAY_SCORE_END", pd.Series([np.nan])).max())
            if "canonical_home_pts" in group.columns and pd.notna(group["canonical_home_pts"].iloc[0]):
                act_h = float(group["canonical_home_pts"].iloc[0])
                act_a = float(group["canonical_away_pts"].iloc[0])

            for _, row in group.iterrows():
                hp = _parse_player_string(row.get("HOME_players", ""))
                ap = _parse_player_string(row.get("AWAY_players", ""))
                p = float(row.get("possessions", 1) or 0)
                xh = float(row.get("home_xpts", 0) or 0)
                xa = float(row.get("away_xpts", 0) or 0)
                uh = row.get("home_usage", {}) or {}
                ua = row.get("away_usage", {}) or {}
                stint_ctx = build_stint_context(row)
                elo.process_stint(
                    ids_A=hp, ids_B=ap, poss=max(p, 1e-6),
                    xpts_A=xh, xpts_B=xa,
                    usage_A=uh, usage_B=ua,
                    period=int(row.get("PERIOD", 1) or 1),
                    start_A=float(row.get("HOME_SCORE_START", 0) or 0),
                    start_B=float(row.get("AWAY_SCORE_START", 0) or 0),
                    end_A=float(row.get("HOME_SCORE_END", 0) or 0),
                    end_B=float(row.get("AWAY_SCORE_END", 0) or 0),
                    stint_ctx=stint_ctx,
                )
                if p > 0:
                    hier.update(
                        hp, ap,
                        float(row.get("home_pts", 0) or 0),
                        float(row.get("away_pts", 0) or 0),
                        p, xpts_off=xh, xpts_def=xa,
                    )

            if home and away and np.isfinite(act_h) and np.isfinite(act_a):
                team.update(str(home), str(away), act_h - act_a, act_h + act_a)

        pf = _player_frame(elo, names)
        if not pf.empty:
            pf = pf.assign(season=season)
            player_snaps.append(pf)
        tf = _team_frame(team)
        if not tf.empty:
            tf = tf.assign(season=season)
            team_snaps.append(tf)
        for size, cdf in _combo_frames(hier, names).items():
            if not cdf.empty:
                combo_snaps[size].append(cdf.assign(season=season))

    players_by_season = pd.concat(player_snaps, ignore_index=True) if player_snaps else pd.DataFrame()
    teams_by_season = pd.concat(team_snaps, ignore_index=True) if team_snaps else pd.DataFrame()
    combos_by_season = {
        size: (pd.concat(lst, ignore_index=True) if lst else pd.DataFrame())
        for size, lst in combo_snaps.items()
    }
    return players_by_season, teams_by_season, combos_by_season, elo, hier, team


def export_elo_and_combos(
    run_dir: Path,
    stints: pd.DataFrame,
    *,
    elo_pkl: Path | None = None,
    hier_pkl: Path | None = None,
    team_elo_pkl: Path | None = None,
    names: dict | None = None,
    top_n: int = 50,
) -> dict[str, Path]:
    """Write Elo / combo CSVs under data/elo and data/combos."""
    elo_cfg = None
    if elo_pkl is not None and Path(elo_pkl).exists():
        try:
            saved = PlayerRatingTracker.load_state(elo_pkl, league_xppp=DEFAULT_LEAGUE_XPPP)
            elo_cfg = dict(saved.cfg)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ could not load elo cfg from {elo_pkl}: {e}")

    players_by_season, teams_by_season, combos_by_season, elo, hier, team = replay_ratings_by_season(
        stints, elo_cfg=elo_cfg, names=names,
    )

    # Prefer end-of-run pickles when present (walk-forward final state)
    if elo_pkl is not None and Path(elo_pkl).exists():
        try:
            elo = PlayerRatingTracker.load_state(elo_pkl, league_xppp=DEFAULT_LEAGUE_XPPP)
        except Exception:
            pass
    if hier_pkl is not None and Path(hier_pkl).exists():
        try:
            hier = HierarchicalPossessionEngine.load_state(hier_pkl)
        except Exception:
            pass
    if team_elo_pkl is not None and Path(team_elo_pkl).exists():
        try:
            team = TeamEloTracker.load_state(team_elo_pkl)
        except Exception:
            pass

    written: dict[str, Path] = {}
    elo_dir = run_dir / "data" / "elo"
    combo_dir = run_dir / "data" / "combos"

    def _write(df: pd.DataFrame, path: Path):
        if df is None or df.empty:
            return
        df.to_csv(path, index=False)
        written[path.name] = path

    _write(players_by_season, elo_dir / "elo_players_by_season.csv")
    best_players = _player_frame(elo, names)
    _write(best_players, elo_dir / "elo_players_final.csv")
    if not best_players.empty:
        _write(best_players.head(top_n), elo_dir / "elo_best_players.csv")
        # also by offense / defense
        _write(
            best_players.sort_values("O_mu", ascending=False).head(top_n),
            elo_dir / "elo_best_offense.csv",
        )
        _write(
            best_players.sort_values("D_mu", ascending=True).head(top_n),
            elo_dir / "elo_best_defense.csv",
        )

    _write(teams_by_season, elo_dir / "elo_teams_by_season.csv")
    best_teams = _team_frame(team)
    _write(best_teams, elo_dir / "elo_teams_final.csv")
    if not best_teams.empty:
        _write(best_teams.head(top_n), elo_dir / "elo_best_teams.csv")

    final_combos = _combo_frames(hier, names)
    for size, label in ((1, "1man"), (2, "2man"), (3, "3man"), (5, "5man")):
        by_s = combos_by_season.get(size, pd.DataFrame())
        _write(by_s, combo_dir / f"combo_{label}_by_season.csv")
        final = final_combos.get(size, pd.DataFrame())
        _write(final, combo_dir / f"combo_{label}.csv")
        if not final.empty:
            # min possession filter for "best"
            filt = final[final["possessions"] >= (20 if size == 1 else 10)].copy()
            if filt.empty:
                filt = final
            _write(filt.head(top_n), combo_dir / f"combo_best_{label}.csv")

    return written


def _ats_won(df: pd.DataFrame) -> pd.Series:
    """Boolean ATS win for non-Pass rows (pushes excluded as NaN)."""
    g = df.copy()
    if "DIRECTION" not in g.columns:
        return pd.Series(dtype=float)
    active = g["DIRECTION"].astype(str) != "Pass"
    cover = pd.to_numeric(g.get("ACTUAL_MARGIN"), errors="coerce") + pd.to_numeric(
        g.get("MARKET_SPREAD"), errors="coerce"
    )
    home = g["DIRECTION"].astype(str) == "Home"
    won = np.where(home, cover > 0, cover < 0).astype(float)
    push = cover == 0
    out = pd.Series(np.nan, index=g.index, dtype=float)
    out.loc[active & ~push] = won[active & ~push]
    return out


def export_betting_csvs(run_dir: Path, results: pd.DataFrame) -> dict[str, Path]:
    """Write backtest + derived betting idea tables under betting/."""
    betting = run_dir / "betting"
    written: dict[str, Path] = {}

    def _save(name: str, df: pd.DataFrame):
        if df is None or (isinstance(df, pd.DataFrame) and df.empty):
            return
        path = betting / name
        df.to_csv(path, index=False)
        written[name] = path

    _save("backtest_results.csv", results)

    # Edge analysis
    try:
        from pipeline.edge_analysis import run_edge_analysis
        edge = run_edge_analysis(results, verbose=False)
        for key, fname in (
            ("fine_bins", "edge_fine_bins.csv"),
            ("rolling", "edge_rolling.csv"),
            ("threshold_grid", "edge_threshold_grid.csv"),
            ("segments", "edge_segments.csv"),
            ("injury_split", "edge_injury_split.csv"),
        ):
            _save(fname, edge.get(key, pd.DataFrame()))
        pol = edge.get("policy")
        if pol:
            (betting / "edge_policy.json").write_text(json.dumps(pol, indent=2, default=str))
            written["edge_policy.json"] = betting / "edge_policy.json"
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ edge analysis export skipped: {e}")

    # Confidence diagnostics
    try:
        from pipeline.confidence_diagnostics import (
            confidence_threshold_grid,
            prior_year_calibration_ablation,
        )
        _save("confidence_threshold_grid.csv", confidence_threshold_grid(results))
        # Ablation is expensive; keep for training runs but silence progress spam.
        _save(
            "confidence_calibration_ablation.csv",
            prior_year_calibration_ablation(results, show_progress=False),
        )
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ confidence diagnostics export skipped: {e}")

    # ML diagnostics
    try:
        from pipeline.ml_diagnostics import run_ml_diagnostics
        ml_summary = run_ml_diagnostics(results, save_dir=None, show_plots=False)
        if isinstance(ml_summary, pd.DataFrame):
            _save("ml_diagnostics_summary.csv", ml_summary)
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ ML diagnostics summary skipped: {e}")
    try:
        from pipeline.ml_diagnostics import export_ml_tracking, ml_value_plays
        plays = ml_value_plays(results)
        _save("ml_value_plays.csv", plays)
        export_ml_tracking(results, betting / "ml_tracking_value_plays.csv")
        written["ml_tracking_value_plays.csv"] = betting / "ml_tracking_value_plays.csv"
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ ML value plays export skipped: {e}")
    # Stake / bankroll equity
    for profile in ("conservative", "moderate", "aggressive"):
        profit_col = f"PROFIT_{profile.upper()}"
        if profit_col in results.columns:
            eq = results[["DATE", "GAME_ID", "HOME", "AWAY", profit_col]].copy() if "DATE" in results.columns else results[[profit_col]].copy()
            eq["cum_profit"] = pd.to_numeric(eq[profit_col], errors="coerce").fillna(0).cumsum()
            _save(f"equity_{profile}.csv", eq)

    # CLV summary
    if "CLV" in results.columns or "POINT_CLV" in results.columns:
        rows = []
        active = results[results.get("DIRECTION", pd.Series(dtype=str)).astype(str) != "Pass"] if "DIRECTION" in results.columns else results
        for col in ("CLV", "POINT_CLV"):
            if col in active.columns:
                s = pd.to_numeric(active[col], errors="coerce").dropna()
                if not s.empty:
                    rows.append({
                        "metric": col,
                        "n": int(len(s)),
                        "mean": float(s.mean()),
                        "median": float(s.median()),
                        "positive_rate": float((s > 0).mean()),
                    })
        _save("clv_summary.csv", pd.DataFrame(rows))

    # ── New idea CSVs (derived from scored rows only) ──
    g = results.copy()
    g["_won"] = _ats_won(g)
    active = g[g.get("DIRECTION", pd.Series(dtype=str)).astype(str) != "Pass"].copy() if "DIRECTION" in g.columns else g.copy()

    # Closing-line movement vs ATS (MODEL_VS_CLOSE / POINT_CLV buckets)
    for col, name in (("POINT_CLV", "clv_buckets_vs_ats.csv"), ("MODEL_VS_CLOSE", "model_vs_close_buckets.csv")):
        if col in active.columns and active["_won"].notna().any():
            vals = pd.to_numeric(active[col], errors="coerce")
            bins = [-np.inf, -2, -1, -0.5, 0, 0.5, 1, 2, np.inf]
            labels = ["<-2", "-2:-1", "-1:-0.5", "-0.5:0", "0:0.5", "0.5:1", "1:2", ">2"]
            tmp = active.assign(_bin=pd.cut(vals, bins=bins, labels=labels))
            agg = (
                tmp.dropna(subset=["_won", "_bin"])
                .groupby("_bin", observed=True)
                .agg(n=("_won", "size"), ats=("_won", "mean"))
                .reset_index()
                .rename(columns={"_bin": "bucket"})
            )
            _save(name, agg)

    # Favorite vs dog / home vs away
    if "MARKET_SPREAD" in active.columns and active["_won"].notna().any():
        mkt = pd.to_numeric(active["MARKET_SPREAD"], errors="coerce")
        side = active["DIRECTION"].astype(str)
        is_home_bet = side == "Home"
        betting_fav = ((is_home_bet & (mkt < 0)) | ((~is_home_bet) & (mkt > 0)))
        split = pd.DataFrame({
            "segment": ["favorite", "dog", "home_bet", "away_bet"],
            "n": [
                int(betting_fav.sum()),
                int((~betting_fav).sum()),
                int(is_home_bet.sum()),
                int((~is_home_bet).sum()),
            ],
            "ats": [
                float(active.loc[betting_fav, "_won"].mean()) if betting_fav.any() else np.nan,
                float(active.loc[~betting_fav, "_won"].mean()) if (~betting_fav).any() else np.nan,
                float(active.loc[is_home_bet, "_won"].mean()) if is_home_bet.any() else np.nan,
                float(active.loc[~is_home_bet, "_won"].mean()) if (~is_home_bet).any() else np.nan,
            ],
        })
        _save("favorite_dog_home_away_ats.csv", split)

    # Edge × confidence heat grid
    if {"EDGE", "CONFIDENCE"}.issubset(active.columns) and active["_won"].notna().any():
        edge = pd.to_numeric(active["EDGE"], errors="coerce").abs()
        conf = pd.to_numeric(active["CONFIDENCE"], errors="coerce")
        e_bins = [0, 3, 5, 7, 10, 99]
        c_bins = [0, 40, 55, 70, 85, 101]
        tmp = active.assign(
            edge_bin=pd.cut(edge, bins=e_bins, right=False),
            conf_bin=pd.cut(conf, bins=c_bins, right=False),
        )
        heat = (
            tmp.dropna(subset=["_won", "edge_bin", "conf_bin"])
            .groupby(["edge_bin", "conf_bin"], observed=True)
            .agg(n=("_won", "size"), ats=("_won", "mean"), mean_edge=("EDGE", lambda s: pd.to_numeric(s, errors="coerce").abs().mean()))
            .reset_index()
        )
        _save("edge_confidence_heat.csv", heat)

    # Season-to-date cumulative bankroll (moderate)
    if "PROFIT_MODERATE" in g.columns and "simulated_season_window" in g.columns:
        rows = []
        for season, sub in g.groupby("simulated_season_window", sort=True):
            cum = pd.to_numeric(sub["PROFIT_MODERATE"], errors="coerce").fillna(0).cumsum()
            for i, (idx, val) in enumerate(cum.items()):
                rows.append({
                    "season": season,
                    "game_idx": i + 1,
                    "cum_profit_moderate": float(val),
                    "DATE": sub.loc[idx, "DATE"] if "DATE" in sub.columns else None,
                })
        _save("season_to_date_bankroll.csv", pd.DataFrame(rows))

    # Push / no-vig sensitivity (cover == 0 rate + ATS excluding pushes already)
    if {"ACTUAL_MARGIN", "MARKET_SPREAD", "DIRECTION"}.issubset(g.columns):
        cover = pd.to_numeric(g["ACTUAL_MARGIN"], errors="coerce") + pd.to_numeric(g["MARKET_SPREAD"], errors="coerce")
        directed = g["DIRECTION"].astype(str) != "Pass"
        push_rate = float((cover[directed] == 0).mean()) if directed.any() else np.nan
        _save("push_sensitivity.csv", pd.DataFrame([{
            "n_directed": int(directed.sum()),
            "push_rate": push_rate,
            "ats_ex_push": float(g.loc[directed & (cover != 0), "_won"].mean()) if directed.any() else np.nan,
        }]))

    # Model vs market residual distribution
    if "MARKET_RESIDUAL" in g.columns or "SPREAD_ERR" in g.columns:
        col = "MARKET_RESIDUAL" if "MARKET_RESIDUAL" in g.columns else "SPREAD_ERR"
        s = pd.to_numeric(g[col], errors="coerce").dropna()
        if not s.empty:
            quant = s.quantile([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99])
            dist = pd.DataFrame({
                "stat": ["n", "mean", "std", "mae"] + [f"p{int(q*100)}" for q in quant.index],
                "value": [len(s), float(s.mean()), float(s.std()), float(s.abs().mean())]
                + [float(v) for v in quant.values],
            })
            _save("model_vs_market_residual_dist.csv", dist)

    # Rest/travel split if columns exist
    for col in ("HOME_REST", "AWAY_REST", "TRAVEL_MILES", "B2B_HOME", "B2B_AWAY", "home_rest_days", "away_rest_days"):
        if col in active.columns and active["_won"].notna().any():
            vals = pd.to_numeric(active[col], errors="coerce")
            if vals.notna().sum() < 20:
                continue
            tmp = active.assign(_bin=pd.qcut(vals, q=min(5, vals.nunique()), duplicates="drop"))
            agg = (
                tmp.dropna(subset=["_won", "_bin"])
                .groupby("_bin", observed=True)
                .agg(n=("_won", "size"), ats=("_won", "mean"), mean_val=(col, "mean"))
                .reset_index()
            )
            _save(f"rest_travel_split_{col}.csv", agg)

    return written


def collect_artifacts(
    run_dir: Path,
    *,
    stints_cache_path: Path | str | None = None,
    tuning_json: dict | None = None,
    config_snapshot: dict | None = None,
    input_manifest: list[dict] | None = None,
    extra_copy_from: Path | None = None,
) -> None:
    """Copy pickles / knobs / caches into artifacts/ for re-run."""
    arts = run_dir / "artifacts"
    arts.mkdir(parents=True, exist_ok=True)

    # Copy latest_* and calibrators from STATE_DIR (and optional prior artifacts)
    sources = [Path(STATE_DIR)]
    if extra_copy_from is not None:
        sources.insert(0, Path(extra_copy_from))

    patterns = [
        "latest_*.pkl",
        "*calibrator*.pkl",
        "elo_calibration_knobs.json",
        "tuning_results.json",
        "bet_calibrator.pkl",
        "elo_calibrator.pkl",
        "disagreement*.pkl",
        "total.pkl",
        "score_pair.pkl",
    ]
    for src_dir in sources:
        if not src_dir.exists():
            continue
        for pat in patterns:
            for f in src_dir.glob(pat):
                if f.is_file():
                    dest = arts / f.name
                    if not dest.exists():
                        try:
                            shutil.copy2(f, dest)
                        except Exception as e:  # noqa: BLE001
                            print(f"  ⚠️ artifact copy failed {f.name}: {e}")

    if stints_cache_path is not None and Path(stints_cache_path).exists():
        try:
            src = Path(stints_cache_path).resolve()
            dest = (arts / Path(stints_cache_path).name).resolve()
            if src != dest:
                shutil.copy2(src, dest)
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ stints cache copy failed: {e}")

    if tuning_json is not None:
        (arts / "tuning_results.json").write_text(
            json.dumps(tuning_json, indent=2, default=str), encoding="utf-8",
        )
    if config_snapshot is not None:
        (arts / "config_snapshot.json").write_text(
            json.dumps(config_snapshot, indent=2, default=str), encoding="utf-8",
        )
    if input_manifest is not None:
        (arts / "input_manifest.json").write_text(
            json.dumps(input_manifest, indent=2, default=str), encoding="utf-8",
        )


def env_info() -> dict[str, Any]:
    import socket
    git_rev = "n/a"
    try:
        import subprocess
        git_rev = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=str(Path(__file__).resolve().parent.parent),
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except Exception:
        pass
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "platform": f"{platform.system()} {platform.release()} ({platform.machine()})",
        "python": sys.version.replace("\n", " "),
        "hostname": socket.gethostname(),
        "repo_root": str(Path(__file__).resolve().parent.parent),
        "git_rev": git_rev,
    }
