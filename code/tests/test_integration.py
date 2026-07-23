"""End-to-end integration test on the 2025-26 sample (single-season split)."""
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import PBP_2026_PATH, MODERN_ODDS_PATH, name_to_id, ASSIST_SPLIT, DEFAULT_LEAGUE_XPPP
from pipeline.ingest import convert_new_pbp
from pipeline.preprocess import preprocess_pbp
from pipeline.stints import build_stints
from pipeline.market import load_modern_odds
from pipeline.ratings import PlayerRatingTracker
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.trackers import PaceTracker, TeamXpppTracker, RotationLineupTracker
from pipeline.teamstats import TeamFormTracker, FORM_FEATURE_COLS
from pipeline.lineup_elo import LineupEloTracker
from pipeline.chemistry import ChemistryTracker
from pipeline.team_elo import TeamEloTracker
from pipeline.travel import TravelTracker
from pipeline.epm_priors import EpmPriorTracker
from pipeline.elo_calibration import apply_elo_calibration_df, WalkForwardEloCalibrator
from pipeline.features import generate_features
from pipeline.model import MetaScoreModel, engineer_interaction_features, SAFE_FEATURE_COLS
from pipeline.simulate import run_simulation
from pipeline.metrics import benchmark_results

# Model-derived at fit time; not produced by generate_features.
MODEL_DERIVED_FEATURES = {"elo_stack_pred"}


def main():
    raw = pd.read_csv(PBP_2026_PATH, nrows=120000, low_memory=False)
    df = convert_new_pbp(raw, name_to_id=name_to_id)
    df = preprocess_pbp(df, compute_xpoints=True)
    st = build_stints(df, assist_split=ASSIST_SPLIT)
    st["season"] = 2026
    st["game_date"] = pd.to_datetime(st["game_date"], errors="coerce")
    st = st.sort_values(["game_date", "GAME_ID", "stint_id"]).reset_index(drop=True)
    print(f"stints={len(st)} games={st['GAME_ID'].nunique()}")

    for c in ["home_3pm", "away_3pm", "home_blks", "home_tovs_forced", "home_poss"]:
        assert c in st.columns, f"missing stint column {c}"

    odds = load_modern_odds(str(MODERN_ODDS_PATH))

    game_ids = st.drop_duplicates("GAME_ID").sort_values("game_date")["GAME_ID"].values
    cut = int(len(game_ids) * 0.7)
    train_ids, test_ids = set(game_ids[:cut]), set(game_ids[cut:])
    train = st[st["GAME_ID"].isin(train_ids)].copy()
    test = st[st["GAME_ID"].isin(test_ids)].copy()

    elo = PlayerRatingTracker(league_xppp=DEFAULT_LEAGUE_XPPP)
    hier = HierarchicalPossessionEngine()
    pace = PaceTracker(team_window=10, league_window=100)
    xppp = TeamXpppTracker(window_size=40, prev_season_weight=0.5)
    form = TeamFormTracker(window=15, prev_season_weight=0.4)
    rotation = RotationLineupTracker(window_games=10, top_n=8)
    lineup_elo = LineupEloTracker()
    chemistry = ChemistryTracker()
    team_elo = TeamEloTracker()
    travel = TravelTracker()
    epm = EpmPriorTracker()

    feats = generate_features(
        train, hier, elo, pace, odds_dict=odds, update_engines=True,
        team_xppp_tracker=xppp, team_form_tracker=form,
        rotation_tracker=rotation, lineup_elo_tracker=lineup_elo,
        chemistry_tracker=chemistry, team_elo_tracker=team_elo,
        travel_tracker=travel, epm_tracker=epm,
    )
    feats = engineer_interaction_features(feats)

    elo_cal = WalkForwardEloCalibrator()
    elo_cal.fit(feats)
    feats = apply_elo_calibration_df(feats, elo_cal)

    required = [c for c in SAFE_FEATURE_COLS if c not in MODEL_DERIVED_FEATURES]
    missing = [c for c in required if c not in feats.columns]
    assert not missing, f"missing model features: {missing}"
    assert all(c in feats.columns for c in FORM_FEATURE_COLS)
    print(f"features rows={len(feats)} cols={len(feats.columns)} (form feats wired: {len(FORM_FEATURE_COLS)})")

    meta = MetaScoreModel(use_isotonic_calibration=True)
    meta.fit(feats, feats["actual_home"], feats["actual_away"], calib_df=feats)
    assert meta.use_purged_cv is True

    results = run_simulation(
        season_df=test, hier_engine=hier, elo_tracker=elo, meta_model=meta,
        pace_tracker=copy.deepcopy(pace), odds_dict=odds,
        team_xppp_tracker=copy.deepcopy(xppp), team_form_tracker=copy.deepcopy(form),
        rotation_tracker=copy.deepcopy(rotation), lineup_elo_tracker=copy.deepcopy(lineup_elo),
        chemistry_tracker=copy.deepcopy(chemistry), team_elo_tracker=copy.deepcopy(team_elo),
        travel_tracker=copy.deepcopy(travel), epm_tracker=copy.deepcopy(epm),
        elo_calibrator=elo_cal if elo_cal.fitted else None,
    )
    assert not results.empty, "simulation produced no rows"
    print(f"simulation rows={len(results)}")
    benchmark_results(results)

    assert feats["h_off_rtg"].std() > 0, "off_rtg has no variance"
    assert feats["h_fg3pct"].std() > 0, "fg3pct has no variance"
    print("Integration test passed.")


if __name__ == "__main__":
    main()
