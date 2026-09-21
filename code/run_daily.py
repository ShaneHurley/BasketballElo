#!/usr/bin/env python3
"""
Daily paper-trading workflow for T-60 inference.

Loads versioned artifacts, validates schema/cutoff compatibility, scores ATS /
ML / totals heads, and refuses recommendations when required inputs are stale
or missing. Default mode is paper trading until promotion gates pass.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import date, datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from pipeline.config import (
    STATE_DIR,
    MODERN_ODDS_PATH,
    PINNACLE_LINES_PATH,
    SPREAD_CALIB_WINDOW,
    ARTIFACT_SCHEMA_VERSION,
    FEATURE_SCHEMA_VERSION,
    DECISION_CUTOFF_MINUTES_BEFORE_TIP,
)
from pipeline.artifacts import build_run_manifest
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.market import get_odds, load_modern_odds, load_pinnacle_lines, SpreadCalibrator
from pipeline.model import MetaScoreModel, MetaWinModel
from pipeline.predict import (
    predict_game,
    PredictionContext,
    load_tuning_config,
    load_live_calibrator,
    load_elo_calibrator,
    load_total_model,
    load_score_pair_model,
    load_ats_classifier,
)
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
from pipeline.ats_ev import ats_decision_record
from pipeline.shared_forecast import emit_shared_forecast_from_feature_row


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
    total = _load_optional(load_total_model, prefix.with_name(prefix.name + "_total.pkl"))
    score_pair = _load_optional(load_score_pair_model, prefix.with_name(prefix.name + "_score_pair.pkl"))
    ats = _load_optional(load_ats_classifier, prefix.with_name(prefix.name + "_ats.pkl"))
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
        "total": total,
        "score_pair": score_pair,
        "ats": ats,
        "state_prefix": state_prefix,
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


def load_odds(schedule_df=None, quotes_df=None, tip_utc_map=None):
    """Load odds with provenance via ``odds_loader`` (snapshots preferred)."""
    from pipeline.odds_loader import load_odds_dict
    return load_odds_dict(
        modern_odds_path=MODERN_ODDS_PATH,
        pinnacle_path=PINNACLE_LINES_PATH,
        schedule_df=schedule_df,
        quotes_df=quotes_df,
        tip_utc_map=tip_utc_map,
        allow_tip_proxy=True,
    )


def build_t60_snapshot(
    *,
    game_date: str,
    home: str,
    away: str,
    engines: dict,
    odds_dict: dict,
    quote_age_minutes: float | None = None,
) -> dict:
    """Versioned inference snapshot for auditability."""
    manifest = build_run_manifest(
        model_version=engines.get("state_prefix", "latest"),
        feature_list=["t60_daily"],
        extra={
            "paper_trading": True,
            "heads": {
                "meta": engines.get("meta") is not None,
                "win": engines.get("win") is not None,
                "ats": engines.get("ats") is not None,
                "total": engines.get("total") is not None,
                "score_pair": engines.get("score_pair") is not None,
            },
        },
    )
    tip_proxy_note = (
        "If tip UTC is unknown, refuse actionable bets. "
        f"Decision cutoff = tip - {DECISION_CUTOFF_MINUTES_BEFORE_TIP}m."
    )
    return {
        "schema": {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "decision_cutoff_minutes": DECISION_CUTOFF_MINUTES_BEFORE_TIP,
        },
        "game": {"date": game_date, "home": home, "away": away},
        "quote_age_minutes": quote_age_minutes,
        "tip_note": tip_proxy_note,
        "manifest": manifest,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def _input_hash(payload: dict) -> str:
    """Hash feature *values* + model version + quote age (not just keys)."""
    feat = payload.get("feat") or {}
    canonical = {
        "model_version": payload.get("model_version"),
        "quote_age_minutes": payload.get("quote_age_minutes"),
        "feat": {
            k: (round(float(v), 6) if isinstance(v, (int, float)) and v == v else v)
            for k, v in sorted(feat.items())
        },
    }
    blob = json.dumps(canonical, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def refuse_if_stale(snapshot: dict, *, max_quote_age_minutes: float = 90.0) -> str | None:
    """Return pass reason if recommendation must be refused."""
    age = snapshot.get("quote_age_minutes")
    if age is not None and float(age) > float(max_quote_age_minutes):
        return f"quote_stale_age_{age:.0f}m"
    schema = snapshot.get("schema") or {}
    if int(schema.get("artifact_schema_version", -1)) != int(ARTIFACT_SCHEMA_VERSION):
        return "artifact_schema_mismatch"
    if int(schema.get("feature_schema_version", -1)) != int(FEATURE_SCHEMA_VERSION):
        return "feature_schema_mismatch"
    return None


def main():
    parser = argparse.ArgumentParser(description="T-60 paper-trading NBA predictions")
    parser.add_argument("--date", default=str(date.today()))
    parser.add_argument("--home", required=True)
    parser.add_argument("--away", required=True)
    parser.add_argument("--home-starters", nargs="*", default=[])
    parser.add_argument("--away-starters", nargs="*", default=[])
    parser.add_argument("--inactive", nargs="*", default=[], help="Player IDs confirmed out")
    parser.add_argument("--fetch-injuries", action="store_true")
    parser.add_argument("--monitor", action="store_true", help="Print weekly monitoring report")
    parser.add_argument("--quote-age-minutes", type=float, default=None)
    parser.add_argument("--max-quote-age", type=float, default=90.0)
    parser.add_argument(
        "--live",
        action="store_true",
        default=False,
        help="Opt into live staking mode (default remains paper trading)",
    )
    args = parser.parse_args()

    engines = load_engines()
    if engines["meta"] is None:
        print("Meta model state not found. Train via run_backtest.py first.")
        return 1

    odds_dict, odds_prov = load_odds()
    spread, ml = get_odds(args.date, args.home, odds_dict)

    snapshot = build_t60_snapshot(
        game_date=args.date,
        home=args.home,
        away=args.away,
        engines=engines,
        odds_dict=odds_dict,
        quote_age_minutes=args.quote_age_minutes,
    )
    refuse_reason = refuse_if_stale(snapshot, max_quote_age_minutes=args.max_quote_age)

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
        total_model=engines.get("total"),
        score_pair_model=engines.get("score_pair"),
        ats_classifier=engines.get("ats"),
    )

    feat = result.get("features") or {}
    preds = {
        "pred_home": result.get("pred_home"),
        "pred_away": result.get("pred_away"),
        "pred_margin": result.get("pred_spread") or result.get("model_spread"),
        "pred_total": result.get("pred_total"),
        "sigma_margin": result.get("conf_width"),
        "sigma_total": result.get("sigma_total"),
        "win_prob": result.get("win_prob"),
    }
    shared = emit_shared_forecast_from_feature_row(feat, preds)
    ats_rec = None
    ats_engine = engines.get("ats")
    ats_fitted = ats_engine is not None and getattr(ats_engine, "fitted", False)
    if not ats_fitted:
        # Finding 6: refuse silent fallback to direction-specific cover for EV.
        ats_rec = {
            "side": "Pass",
            "pass_reason": "ats_artifact_missing_or_unfitted",
            "fair_cover_prob": float("nan"),
            "expected_value": float("nan"),
        }
    else:
        from pipeline.ats_ev import canonical_home_cover_prob
        p_home = canonical_home_cover_prob(
            ats_classifier=ats_engine,
            feat={**feat, "pred_margin": preds.get("pred_margin")},
            decision_spread=feat.get("decision_spread", feat.get("market_spread")),
            pred_margin=preds.get("pred_margin"),
            sigma=float(result.get("conf_width") or 24.0) / 2.5,
            lean=result.get("direction") or result.get("edge_lean"),
            direction_specific_cover=result.get("cover_prob_calibrated"),
        )
        ats_rec = ats_decision_record(
            p_home,
            feat,
            quote_age_minutes=args.quote_age_minutes,
        )

    audit = {
        "paper_trading": not bool(args.live),
        "input_hash": _input_hash({
            "feat": feat,
            "model_version": engines.get("state_prefix", "latest"),
            "quote_age_minutes": args.quote_age_minutes,
        }),
        "snapshot": snapshot,
        "shared_forecast": shared,
        "ats_decision": ats_rec,
        "odds_provenance": odds_prov,
        "refuse_reason": refuse_reason,
        "pass_reasons": [refuse_reason] if refuse_reason else [],
    }
    if refuse_reason:
        # Force pass on actionable fields when stale/mismatched.
        result["direction"] = "Pass"
        result["ml_bet"] = "Pass"
        result["ou_direction"] = "Pass"
        if ats_rec is not None:
            ats_rec["side"] = "Pass"
            ats_rec["pass_reason"] = refuse_reason

    result["t60_audit"] = audit

    log_path = STATE_DIR / "prediction_log.csv"
    row = pd.DataFrame([{k: v for k, v in result.items() if k not in ("features", "t60_audit")}])
    row["input_hash"] = audit["input_hash"]
    row["refuse_reason"] = refuse_reason
    row["paper_trading"] = not bool(args.live)
    if log_path.exists():
        row.to_csv(log_path, mode="a", header=False, index=False)
    else:
        row.to_csv(log_path, index=False)

    audit_dir = STATE_DIR / "t60_snapshots"
    audit_dir.mkdir(parents=True, exist_ok=True)
    snap_path = audit_dir / f"{args.date}_{args.home}_{args.away}_{audit['input_hash']}.json"
    snap_path.write_text(json.dumps(audit, indent=2, default=str))

    if args.monitor and log_path.exists():
        report = weekly_report(pd.read_csv(log_path))
        print("Monitoring:", report)

    print(result)
    print(f"T-60 audit written: {snap_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
