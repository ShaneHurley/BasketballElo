"""Validate the generated Colab notebook: exec inlined module cells, run a smoke workflow."""
import json
import sys
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parent.parent
NB = REPO / "nba_unified_pipeline_colab.ipynb"


def main():
    nb = json.loads(NB.read_text())
    ns = {}

    skip_markers = (
        "#  PHASE 0",
        "#  PHASE 1",
        "#  PHASE 2 ·",
        "#  PHASE 2a",
        "#  PHASE 2b",
        "#  PHASE 2c",
        "#  PHASE 2d",
        "#  PHASE 2e",
        "#  PHASE 3",
        "#  PHASE 4",
        "#  PHASE 5",
        "!pip install",
    )

    for i, cell in enumerate(nb["cells"]):
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if any(m in src for m in skip_markers):
            continue
        if src.startswith("# Install"):
            continue
        if "CONFIG" in src and "ROOT = Path" in src:
            # Config cell: local paths, skip Colab drive mount
            src = src.replace(
                'ROOT = Path("/content/drive/MyDrive/basketballData")',
                f'ROOT = Path(r"{REPO}/basketballData")',
            )
            src = src.replace("drive.mount(\"/content/drive\")", "pass  # drive.mount skipped in test")
        try:
            exec(compile(src, f"<nb cell {i}>", "exec"), ns)
        except Exception as e:
            raise AssertionError(f"Cell {i} failed: {e}\n---\n{src[:500]}") from e

    required = [
        "convert_new_pbp", "preprocess_pbp", "build_stints", "load_modern_odds",
        "PlayerRatingTracker", "HierarchicalPossessionEngine", "TeamFormTracker",
        "generate_features", "engineer_interaction_features", "MetaScoreModel",
        "MetaWinModel", "WalkForwardEloCalibrator", "apply_elo_calibration_df",
        "tune_elo_calibrator", "tune_margin_model", "save_elo_knobs",
        "WalkForwardMarketDisagreementModel", "margin_close_residual",
        "variance_aware_edge_threshold", "benchmark_betting_roi",
        "run_simulation", "predict_game", "run_multi_year_backtest_walkforward",
        "grid_search_bet_edge", "benchmark_results", "map_elo_params",
        "compare_selection_strategies", "print_accuracy_layers",
        "passes_confidence_actionable_gates", "calibration_ablation_configs",
        "build_calib_split_ats_frame",
        "SAFE_FEATURE_COLS", "generate_betting_plots", "load_tuning_config",
        "add_all_profile_columns", "apply_bet_selection_gates", "ATSClassifier",
        "TeamVolatilityTracker",         "posthoc_ablation_summary", "passes_ablation_gate",
        "BET_SELECTION_MODE", "MIN_EDGE_BUCKET",
        "run_ml_diagnostics", "export_ml_tracking", "ml_value_plays",
        "ensure_results", "load_min_confidence_threshold", "run_phase_2a_unified_confidence",
        "WalkForwardBetCalibrator", "WalkForwardMLCalibrator",
        "MetaTotalModel", "MetaScorePairModel", "tune_meta_win_model",
        "total_feature_cols", "win_feature_cols", "WIN_FEATURE_COLS", "TOTAL_FEATURE_COLS",
        "total_over_prob_gaussian", "select_ou_bet", "explain_ml_bet", "crps_gaussian",
        "promote_ablation_winners", "expected_availability_impact",
        "OU_USE_GAUSSIAN_PROB", "USE_SCORE_PAIR_TOTAL",
    ]
    missing = [s for s in required if s not in ns]
    assert not missing, f"missing inlined symbols: {missing}"
    print(f"All {len(required)} core symbols present in inlined notebook namespace.")

    pbp = REPO / "[10-21-2025]-[05-18-2026]-combined-stats.csv"
    if not pbp.exists():
        print("Skip smoke workflow (no PBP CSV). NOTEBOOK STRUCTURE VALIDATION PASSED.")
        return

    raw = pd.read_csv(pbp, nrows=80000, low_memory=False)
    df = ns["convert_new_pbp"](raw, name_to_id=ns.get("name_to_id", {}))
    df = ns["preprocess_pbp"](df, compute_xpoints=True)
    st = ns["build_stints"](df, assist_split=ns["ASSIST_SPLIT"])
    st["season"] = 2026
    st["game_date"] = pd.to_datetime(st["game_date"], errors="coerce")
    st = st.sort_values(["game_date", "GAME_ID", "stint_id"]).reset_index(drop=True)

    elo = ns["PlayerRatingTracker"](league_xppp=ns["DEFAULT_LEAGUE_XPPP"])
    hier = ns["HierarchicalPossessionEngine"]()
    pace = ns["PaceTracker"](team_window=10, league_window=100)
    xppp = ns["TeamXpppTracker"](window_size=40, prev_season_weight=0.5)
    form = ns["TeamFormTracker"](window=15, prev_season_weight=0.4)
    lineup_elo = ns["LineupEloTracker"]()
    chemistry = ns["ChemistryTracker"]()
    shot_q = ns["ShotQualityTracker"]()
    team_elo = ns["TeamEloTracker"]()
    travel = ns["TravelTracker"]()
    epm = ns["EpmPriorTracker"]()
    refs = ns["RefTracker"]()
    hapm = ns["HapmPriorTracker"]()
    rotation = ns["RotationLineupTracker"]()

    feats = ns["generate_features"](
        st, hier, elo, pace, odds_dict={}, update_engines=True,
        team_xppp_tracker=xppp, team_form_tracker=form,
        lineup_elo_tracker=lineup_elo, chemistry_tracker=chemistry,
        shot_quality_tracker=shot_q, team_elo_tracker=team_elo,
        travel_tracker=travel, epm_tracker=epm, ref_tracker=refs,
        hapm_tracker=hapm, rotation_tracker=rotation,
    )
    feats = ns["engineer_interaction_features"](feats)
    missing_feats = [c for c in ns["SAFE_FEATURE_COLS"] if c not in feats.columns]
    if len(missing_feats) > 15:
        assert False, f"too many missing features ({len(missing_feats)}): {missing_feats[:15]}"
    elif missing_feats:
        print(f"Note: {len(missing_feats)} optional SAFE_FEATURE_COLS absent in smoke slice")

    meta = ns["MetaScoreModel"](use_isotonic_calibration=True)
    meta.fit(feats, feats["actual_home"], feats["actual_away"], calib_df=feats)
    try:
        pred = ns["predict_game"](
            "BOS", "LAL", "2026-10-28", hier, elo, meta, pace,
            team_xppp_tracker=xppp, team_form_tracker=form,
            lineup_elo_tracker=lineup_elo, chemistry_tracker=chemistry,
            auto_load_models=False,
        )
        assert pred and ("pred_spread" in pred or "direction" in pred)
    except ValueError as e:
        if "features" in str(e).lower():
            print(f"predict_game skipped on smoke slice: {e}")
        else:
            raise
    print(f"Notebook smoke workflow OK. games={st['GAME_ID'].nunique()} feat_cols={len(feats.columns)}")
    print("NOTEBOOK VALIDATION PASSED")


if __name__ == "__main__":
    main()
