#!/usr/bin/env python3
"""
Daily paper-trading workflow for 2026-27 season.
Load saved engine state, predict today's games, append to prediction log.
"""
from __future__ import annotations

import argparse
import sys
from datetime import date
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.config import STATE_DIR, MODERN_ODDS_PATH, PINNACLE_LINES_PATH, SPREAD_CALIB_WINDOW
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.market import get_odds, load_modern_odds, load_pinnacle_lines, SpreadCalibrator
from pipeline.model import MetaScoreModel, MetaWinModel
from pipeline.predict import predict_game, PredictionContext, load_tuning_config, load_live_calibrator, load_elo_calibrator
from pipeline.ratings import PlayerRatingTracker
from pipeline.trackers import PaceTracker, TeamXpppTracker, RotationLineupTracker
from pipeline.teamstats import TeamFormTracker
from pipeline.lineup_elo import LineupEloTracker
from pipeline.chemistry import ChemistryTracker
from pipeline.team_elo import TeamEloTracker
from pipeline.travel import TravelTracker
from pipeline.epm_priors import EpmPriorTracker
from pipeline.availability import fetch_espn_injuries
from pipeline.monitoring import weekly_report


def _load_optional(loader, path):
    p = Path(path)
    return loader(p) if p.exists() else None


def load_engines(state_prefix: str = "latest"):
    prefix = STATE_DIR / state_prefix
    elo = PlayerRatingTracker.load_state(prefix.with_name(prefix.name + "_elo.pkl"))
    hier = HierarchicalPossessionEngine.load_state(prefix.with_name(prefix.name + "_hier.pkl"))
    pace = _load_optional(PaceTracker.load_state, prefix.with_name(prefix.name + "_pace.pkl"))
    if pace is None:
        pace = PaceTracker(team_window=10, league_window=100)
    meta = _load_optional(MetaScoreModel.load, prefix.with_name(prefix.name + "_meta.pkl"))
    win = _load_optional(MetaWinModel.load, prefix.with_name(prefix.name + "_win.pkl"))
    spread_cal = SpreadCalibrator(window=SPREAD_CALIB_WINDOW)
    rotation = _load_optional(RotationLineupTracker.load_state, prefix.with_name(prefix.name + "_rotation.pkl"))
    lineup_elo = _load_optional(LineupEloTracker.load_state, prefix.with_name(prefix.name + "_lineup_elo.pkl"))
    chemistry = _load_optional(ChemistryTracker.load_state, prefix.with_name(prefix.name + "_chemistry.pkl"))
    team_elo = _load_optional(TeamEloTracker.load_state, prefix.with_name(prefix.name + "_team_elo.pkl"))
    travel = _load_optional(TravelTracker.load_state, prefix.with_name(prefix.name + "_travel.pkl"))
    ctx = _load_optional(PredictionContext.load_state, prefix.with_name(prefix.name + "_ctx.pkl"))
    return {
        "elo": elo, "hier": hier, "pace": pace, "meta": meta,
        "win": win, "spread_calibrator": spread_cal,
        "rotation": rotation or RotationLineupTracker(),
        "lineup_elo": lineup_elo or LineupEloTracker(),
        "chemistry": chemistry or ChemistryTracker(),
        "team_elo": team_elo or TeamEloTracker(),
        "travel": travel or TravelTracker(),
        "xppp": TeamXpppTracker(),
        "form": TeamFormTracker(),
        "epm": EpmPriorTracker(),
        "ctx": ctx or PredictionContext(),
    }


def save_engines(engines: dict, state_prefix: str = "latest"):
    prefix = STATE_DIR / state_prefix
    prefix.parent.mkdir(parents=True, exist_ok=True)
    engines["elo"].save_state(prefix.with_name(prefix.name + "_elo.pkl"))
    engines["hier"].save_state(prefix.with_name(prefix.name + "_hier.pkl"))
    if hasattr(engines["pace"], "save_state"):
        engines["pace"].save_state(prefix.with_name(prefix.name + "_pace.pkl"))
    if engines["meta"] is not None:
        engines["meta"].save(prefix.with_name(prefix.name + "_meta.pkl"))
    engines["rotation"].save_state(prefix.with_name(prefix.name + "_rotation.pkl"))
    engines["lineup_elo"].save_state(prefix.with_name(prefix.name + "_lineup_elo.pkl"))
    engines["chemistry"].save_state(prefix.with_name(prefix.name + "_chemistry.pkl"))
    engines["team_elo"].save_state(prefix.with_name(prefix.name + "_team_elo.pkl"))
    engines["travel"].save_state(prefix.with_name(prefix.name + "_travel.pkl"))
    engines["ctx"].save_state(prefix.with_name(prefix.name + "_ctx.pkl"))


def load_odds():
    odds_dict = load_modern_odds(str(MODERN_ODDS_PATH))
    if Path(PINNACLE_LINES_PATH).exists():
        pin = load_pinnacle_lines(str(PINNACLE_LINES_PATH))
        odds_dict.update(pin)
    return odds_dict


def main():
    parser = argparse.ArgumentParser(description="Predict today's NBA spreads")
    parser.add_argument("--date", default=str(date.today()))
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--home-starters", nargs="*", default=[])
    parser.add_argument("--away-starters", nargs="*", default=[])
    parser.add_argument("--inactive", nargs="*", default=[], help="Player IDs confirmed out")
    parser.add_argument("--fetch-injuries", action="store_true")
    parser.add_argument("--monitor", action="store_true", help="Print weekly monitoring report")
    args = parser.parse_args()

    engines = load_engines()
    if engines["meta"] is None:
        print("Meta model state not found. Train via run_backtest.py first.")
        return 1

    odds_dict = load_odds()
    spread, ml = get_odds(args.date, args.home, odds_dict)

    inactive = list(args.inactive)
    if args.fetch_injuries:
        injuries = fetch_espn_injuries()
        print(f"Fetched {len(injuries)} injury entries from ESPN")

    tuning_cfg = load_tuning_config()
    bet_cal = load_live_calibrator()
    elo_cal = load_elo_calibrator()

    result = predict_game(
        home_abbr=args.home,
        away_abbr=args.away,
        game_date=args.date,
        hier_engine=engines["hier"],
        elo_tracker=engines["elo"],
        meta_model=engines["meta"],
        pace_tracker=engines["pace"],
        team_xppp_tracker=engines["xppp"],
        team_form_tracker=engines["form"],
        rotation_tracker=engines["rotation"],
        lineup_elo_tracker=engines["lineup_elo"],
        chemistry_tracker=engines["chemistry"],
        team_elo_tracker=engines["team_elo"],
        travel_tracker=engines["travel"],
        epm_tracker=engines["epm"],
        ctx=engines["ctx"],
        spread_calibrator=engines.get("spread_calibrator"),
        win_model=engines.get("win"),
        confidence_calibrator=bet_cal,
        elo_calibrator=elo_cal,
        tuning_config=tuning_cfg,
        home_starters=args.home_starters or None,
        away_starters=args.away_starters or None,
        inactive_ids=inactive or None,
        live_market_spread=spread,
        live_market_ml=ml,
        odds_dict=odds_dict,
    )

    log_path = STATE_DIR / "prediction_log.csv"
    row = pd.DataFrame([{k: v for k, v in result.items() if k != "features"}])
    if log_path.exists():
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row.to_csv(log_path, index=False)

    if args.monitor and log_path.exists():
        report = weekly_report(pd.read_csv(log_path))
        print("Monitoring:", report)

    print(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
