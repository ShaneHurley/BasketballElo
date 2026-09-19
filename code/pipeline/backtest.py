"""Walk-forward backtest orchestration."""
from __future__ import annotations

import copy
import json

import numpy as np
import pandas as pd

from pipeline.config import (
    DEFAULT_LEAGUE_XPPP,
    TUNING_INVALID_SCORE,
    GOOD_BET_EDGE,
    STATE_DIR,
    SPREAD_CALIB_WINDOW,
    SPREAD_CALIB_WINDOW_CANDIDATES,
    SPREAD_CALIB_ADAPTIVE_WINDOW,
    BET_SELECTION_MODE,
    WALKFORWARD_EDGE_MIN_FLOOR,
    CALIBRATION_MODE,
    STAKE_SIZING_MODE,
    WIN_PROB_SOURCE,
    MIN_CONFIDENCE_SCORE,
    MAX_CONFIDENCE_SCORE,
    CONFIDENCE_CALIB_METHOD,
    CONFIDENCE_CALIB_SCOPE,
    CONFIDENCE_MODE,
    CONFIDENCE_SELECTION_MODE,
    CONFIDENCE_EDGE_SCALED,
    ML_CALIB_METHOD,
    ML_CALIB_SCOPE,
    MIN_ML_WIN_PCT,
    ML_MAX_FAVORITE_DECIMAL_DEFAULT,
    OU_MIN_EDGE,
    CONFIDENCE_MIN_EDGE,
    MAX_QUANTILE_WIDTH,
    MIN_DISAGREEMENT_TRUST,
    SKIP_TIGHT_SPREAD,
    EDGE_AVOID_BAND,
    MIN_EDGE_BUCKET,
    CONFIDENCE_GATE_MAX_LIFT,
    ARTIFACT_SCHEMA_VERSION,
    TOTAL_HEAD_CV_SANITY_CAP,
    RATING_HISTORY_USE_ALL,
    MODEL_TRAIN_WINDOW_DEFAULT,
)
from pipeline.ml_calibration import WalkForwardMLCalibrator, default_ml_calibrator_path
from pipeline.calibration_policy import should_use_season1_calibrator
from pipeline.bet_selection import uses_edge_gates
from pipeline.team_volatility import TeamVolatilityTracker
from pipeline.ratings import PlayerRatingTracker
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.trackers import PaceTracker, TeamXpppTracker, RotationLineupTracker
from pipeline.teamstats import TeamFormTracker
from pipeline.lineup_elo import LineupEloTracker
from pipeline.chemistry import ChemistryTracker
from pipeline.team_elo import TeamEloTracker
from pipeline.travel import TravelTracker
from pipeline.refs import RefTracker
from pipeline.shot_quality import ShotQualityTracker
from pipeline.hapm import HapmPriorTracker, hapm_lookup_for_training
from pipeline.epm_priors import EpmPriorTracker
from pipeline.features import generate_features
from pipeline.model import (
    MetaScoreModel,
    MetaWinModel,
    MetaTotalModel,
    MetaScorePairModel,
    engineer_interaction_features,
    tune_margin_model,
    tune_total_model,
    tune_meta_win_model,
    win_feature_cols,
    total_feature_cols,
    AdaptiveParameterBounds,
    meta_params_for_bounds,
)
from pipeline.tuning import tune_elo_tracker, tune_hierarchical
from pipeline.tuning_cache import load_cached, save_cached
from pipeline.market import SpreadCalibrator, replay_rolling_spread_calibration
from pipeline.calibration_registry import (
    CalibrationSliceRegistry,
    BACKTEST_CALIBRATION_TARGETS,
    chronological_game_id_partition,
    slice_by_game_ids,
)
from pipeline.calibrators import RollingPlattCalibrator
from pipeline.bet_confidence import (
    WalkForwardBetCalibrator,
    DEFAULT_CONFIDENCE_WEIGHTS,
    default_calibrator_path,
    fit_walkforward_confidence_calibrator,
    build_calib_split_ats_frame,
    save_phase2a_walkforward_state,
    shrink_weight_center,
    logistic_feature_importance,
)
from pipeline.market_disagreement import WalkForwardMarketDisagreementModel, default_disagreement_path
from pipeline.ats_classifier import ATSClassifier
from pipeline.upset_classifier import UpsetClassifier
from pipeline.data_paths import clamp_rolling_window, split_rating_and_model_seasons


def _series_last_optional_float(df: pd.DataFrame, col: str, default=None):
    if col not in df.columns or df.empty:
        return default
    val = df[col].iloc[-1]
    if val is None or (isinstance(val, float) and (pd.isna(val) or not np.isfinite(val))):
        return default
    try:
        return float(val)
    except (TypeError, ValueError):
        return default


def select_spread_calib_window(
    prior_df: pd.DataFrame,
    *,
    candidates: tuple[int, ...] | list[int] | None = None,
    default: int | None = None,
    pred_col: str = "RAW_PRED_MARGIN",
    actual_col: str = "ACTUAL_MARGIN",
) -> int:
    """Pick rolling spread window by prior-season corrected MAE (walk-forward)."""
    from pipeline.market import replay_rolling_spread_calibration

    default_w = int(default if default is not None else SPREAD_CALIB_WINDOW)
    cands = tuple(candidates or SPREAD_CALIB_WINDOW_CANDIDATES or (default_w,))
    if prior_df is None or prior_df.empty:
        return default_w
    if pred_col not in prior_df.columns or actual_col not in prior_df.columns:
        return default_w
    pred = pd.to_numeric(prior_df[pred_col], errors="coerce").values
    actual = pd.to_numeric(prior_df[actual_col], errors="coerce").values
    mask = np.isfinite(pred) & np.isfinite(actual)
    if mask.sum() < 40:
        return default_w
    pred, actual = pred[mask], actual[mask]
    best_w, best_mae = default_w, float("inf")
    for w in cands:
        w = int(w)
        if w < 10:
            continue
        corrected, _ = replay_rolling_spread_calibration(
            pred, actual,
            window=w,
            min_samples=min(30, max(10, w // 3)),
            mode="linear",
        )
        mae = float(np.mean(np.abs(np.asarray(corrected, dtype=float) - actual)))
        if np.isfinite(mae) and mae < best_mae - 1e-9:
            best_mae, best_w = mae, w
    return int(best_w)


from pipeline.elo_calibration import (
    WalkForwardEloCalibrator,
    apply_elo_calibration_df,
    default_elo_calibrator_path,
    save_elo_knobs,
    tune_elo_calibrator,
)
from pipeline.metrics import (
    benchmark_results,
    grid_search_bet_edge,
    walkforward_edge_threshold,
    walkforward_min_confidence,
    walkforward_confidence_gate,
    walkforward_favorite_decimal,
    walkforward_ou_edge,
    apply_edge_threshold,
    add_all_profile_columns,
    benchmark_betting_roi,
    compute_metrics,
)
from pipeline.simulate import run_simulation

__all__ = [
    "run_multi_year_backtest_walkforward",
    "select_spread_calib_window",
    "benchmark_results",
    "grid_search_bet_edge",
    "walkforward_edge_threshold",
    "walkforward_min_confidence",
    "apply_edge_threshold",
]


def _persist_backtest_metadata(extra: dict) -> None:
    """Save run metadata; works in package and flat Colab namespace."""
    fn = globals().get("save_backtest_metadata")
    if callable(fn):
        fn(extra=extra)
        return
    try:
        import importlib
        mod = importlib.import_module("pipeline.artifacts")
        mod.save_backtest_metadata(extra=extra)
    except (ImportError, AttributeError):
        pass


def run_multi_year_backtest_walkforward(
    stints_df: pd.DataFrame,
    odds_dict: dict = None,
    n_tuning_trials_elo: int = 30,
    n_tuning_trials_hier: int = 30,
    n_tuning_trials_meta: int = 40,
    n_tuning_trials_total: int = 15,
    n_tuning_trials_meta_win: int | None = None,
    rolling_window_size: int = MODEL_TRAIN_WINDOW_DEFAULT,
    model_kwargs: dict = None,
    feature_cols: list = None,
    walkforward_edge: bool = True,
    pace_per_team: bool = True,
    spread_calib_mode: str = "linear",
    variance_aware_winprob: bool = True,
    use_tuning_cache: bool = True,
    train_win_model: bool = True,
    tune_total_head: bool = True,
    fast_tuning: bool = False,
    require_elo_agreement: bool = False,
    use_hapm_priors: bool = False,
    rating_history_use_all: bool | None = None,
    use_venn_abers_filter: bool | None = None,
) -> pd.DataFrame:
    """
    True walk-forward backtest with NO data leakage.

    ``rolling_window_size`` controls *model-train* seasons (clamped 2–5).
    When ``rating_history_use_all`` is True (default from config), Elo/hier/
    pace engines are warm-started on every season before the test fold, then
    meta models train only on the rolling model window.
    """
    rolling_window_size = clamp_rolling_window(rolling_window_size)
    use_all_rating = (
        RATING_HISTORY_USE_ALL if rating_history_use_all is None else bool(rating_history_use_all)
    )
    if n_tuning_trials_meta_win is None:
        n_tuning_trials_meta_win = min(n_tuning_trials_meta, 25)
    stints_df = stints_df.copy()
    stints_df["game_date"] = pd.to_datetime(stints_df["game_date"], errors="coerce")
    if "season" not in stints_df.columns:
        stints_df['season'] = stints_df['game_date'].dt.year + (stints_df['game_date'].dt.month >= 9).astype(int)

    all_seasons = sorted(stints_df['season'].dropna().unique())
    print(f"\n🚀 Walk-forward over {len(all_seasons)} seasons: {all_seasons}")
    print(
        f"   Model-train window = {rolling_window_size} season(s); "
        f"rating history = {'all prior' if use_all_rating else 'model window only'}"
    )
    print(
        f"   Bet gates: mode={BET_SELECTION_MODE}  min_edge={CONFIDENCE_MIN_EDGE}  "
        f"bucket_min={MIN_EDGE_BUCKET}  min_conf={MIN_CONFIDENCE_SCORE}  "
        f"gate_lift≤{CONFIDENCE_GATE_MAX_LIFT}  max_width={MAX_QUANTILE_WIDTH}  "
        f"trust>={MIN_DISAGREEMENT_TRUST}  tight_spread={'on' if SKIP_TIGHT_SPREAD else 'off'}  "
        f"avoid_band={EDGE_AVOID_BAND}  edge_scaled={CONFIDENCE_EDGE_SCALED}"
    )

    compiled_results = []
    phase2a_rows: list[dict] = []
    confidence_weight_center = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    param_bounds = AdaptiveParameterBounds()
    elo_bounds = AdaptiveParameterBounds()
    hier_bounds = AdaptiveParameterBounds()
    bet_calibrator = WalkForwardBetCalibrator(method=CONFIDENCE_CALIB_METHOD)
    ml_calibrator = WalkForwardMLCalibrator(method=ML_CALIB_METHOD)
    disagreement_model = WalkForwardMarketDisagreementModel()

    if model_kwargs is None:
        model_kwargs = {
            "total_mode": "external",
            "use_elo_stack": True,
            "train_target": "decision_residual",
            "use_quantile_heads": True,
            "prediction_mode": "absolute",
        }

    full_elo = full_hier = meta_model = win_model = total_model = score_pair_model = ats_classifier = upset_classifier = None
    sim_pace = sim_xppp = sim_form = sim_rotation = None
    sim_lineup_elo = sim_chemistry = sim_team_elo = sim_travel = sim_epm = None
    sim_vol = None
    last_good_total_params = None
    ats_status_records: list = []
    cover_prob_calibrator = None

    active_spread_window = int(SPREAD_CALIB_WINDOW)

    for i, test_season in enumerate(all_seasons):
        cover_prob_calibrator = None
        print(f"\n{'='*65}")
        print(f"🚀 SIMULATING SEASON {int(test_season)-1}-{int(test_season)}")
        print('='*65)

        rating_history_seasons, model_train_seasons = split_rating_and_model_seasons(
            all_seasons, i, rolling_window_size, rating_history_use_all=use_all_rating,
        )
        train_seasons = model_train_seasons
        if not train_seasons:
            print("  [!] No previous seasons – skipping (no training data).")
            continue

        warm_seasons = [s for s in rating_history_seasons if s not in set(model_train_seasons)]
        print(f"  Model-train seasons: {train_seasons}")
        if warm_seasons:
            print(f"  Rating warm-start seasons: {warm_seasons}")
        train_stints = stints_df[stints_df['season'].isin(train_seasons)].copy()
        test_stints = stints_df[stints_df['season'] == test_season].copy()
        warm_stints = (
            stints_df[stints_df['season'].isin(warm_seasons)].copy()
            if warm_seasons else stints_df.iloc[0:0].copy()
        )

        # Tune Elo / Hier (with cache)
        best_elo = load_cached("elo", train_seasons, train_stints) if use_tuning_cache else None
        if best_elo is None:
            print("  ⏳ Tuning Elo tracker on past data...")
            best_elo, elo_cv = tune_elo_tracker(
                train_stints, DEFAULT_LEAGUE_XPPP,
                n_trials=n_tuning_trials_elo, bounds=elo_bounds,
            )
            if use_tuning_cache:
                save_cached("elo", train_seasons, best_elo, train_stints)
            if elo_cv < TUNING_INVALID_SCORE:
                elo_bounds.record_best(int(test_season), best_elo)
                elo_bounds.report()
            else:
                print("  ⚠ Skipping Elo adaptive bounds (CV did not run).")
        else:
            print("  ⚡ Using cached Elo params")
            elo_bounds.record_best(int(test_season), best_elo)

        best_hier = load_cached("hier", train_seasons, train_stints) if use_tuning_cache else None
        if best_hier is None:
            print("  ⏳ Tuning Hierarchical engine on past data...")
            best_hier, hier_cv = tune_hierarchical(
                train_stints, n_trials=n_tuning_trials_hier, bounds=hier_bounds,
            )
            if use_tuning_cache:
                save_cached("hier", train_seasons, best_hier, train_stints)
            if hier_cv < TUNING_INVALID_SCORE:
                hier_bounds.record_best(int(test_season), best_hier)
                hier_bounds.report()
            else:
                print("  ⚠ Skipping Hierarchical adaptive bounds (CV did not run).")
        else:
            print("  ⚡ Using cached Hierarchical params")
            hier_bounds.record_best(int(test_season), best_hier)

        game_dates = (
            train_stints[['game_date', 'GAME_ID']]
            .drop_duplicates('GAME_ID')
            .sort_values('game_date')
        )
        split_idx = int(len(game_dates) * 0.8)
        base_game_ids = game_dates.iloc[:split_idx]['GAME_ID'].values
        calib_game_ids = game_dates.iloc[split_idx:]['GAME_ID'].values
        base_stints = train_stints[train_stints['GAME_ID'].isin(base_game_ids)].copy()
        calib_stints = train_stints[train_stints['GAME_ID'].isin(calib_game_ids)].copy()
        print(f"  Base games: {len(base_game_ids)} | Calibration games: {len(calib_game_ids)}")

        base_hier = HierarchicalPossessionEngine(**best_hier)
        base_elo = PlayerRatingTracker(config=best_elo, league_xppp=DEFAULT_LEAGUE_XPPP)
        base_pace = PaceTracker(team_window=10, league_window=100, per_team=pace_per_team)
        base_xppp = TeamXpppTracker(window_size=40, prev_season_weight=0.5)
        base_form = TeamFormTracker(window=15, prev_season_weight=0.4)
        base_rotation = RotationLineupTracker(window_games=10, top_n=8)
        base_lineup_elo = LineupEloTracker()
        base_chemistry = ChemistryTracker()
        base_team_elo = TeamEloTracker()
        base_travel = TravelTracker()
        base_ref = RefTracker()
        base_shot_quality = ShotQualityTracker()
        base_hapm = HapmPriorTracker()
        if use_hapm_priors:
            # Warm-start + model-train span for post-train test-season simulation only.
            hist_for_hapm = (
                stints_df[stints_df["season"].isin(rating_history_seasons)].copy()
                if rating_history_seasons else train_stints
            )
            base_hapm.fit(hist_for_hapm)
        hapm_train_lookup = hapm_lookup_for_training(train_stints) if use_hapm_priors else None
        base_epm = EpmPriorTracker()

        # Warm-start engines on earlier seasons (ratings only; features discarded).
        if not warm_stints.empty:
            print(f"  🔥 Warm-starting engines on {len(warm_seasons)} prior season(s)...")
            _ = generate_features(
                warm_stints, base_hier, base_elo, base_pace,
                odds_dict=odds_dict, update_engines=True,
                team_xppp_tracker=base_xppp, team_form_tracker=base_form,
                rotation_tracker=base_rotation, lineup_elo_tracker=base_lineup_elo,
                chemistry_tracker=base_chemistry, team_elo_tracker=base_team_elo,
                travel_tracker=base_travel, epm_tracker=base_epm,
                ref_tracker=base_ref,
                shot_quality_tracker=base_shot_quality,
                hapm_tracker=None,  # past-only lookup is for model-train rows only
            )

        base_features = generate_features(
            base_stints, base_hier, base_elo, base_pace,
            odds_dict=odds_dict, update_engines=True,
            team_xppp_tracker=base_xppp, team_form_tracker=base_form,
            rotation_tracker=base_rotation, lineup_elo_tracker=base_lineup_elo,
            chemistry_tracker=base_chemistry, team_elo_tracker=base_team_elo,
            travel_tracker=base_travel, epm_tracker=base_epm,
            ref_tracker=base_ref,
            shot_quality_tracker=base_shot_quality,
            hapm_tracker=hapm_train_lookup,
        )
        base_features = engineer_interaction_features(base_features)

        calib_hier = copy.deepcopy(base_hier)
        calib_elo = copy.deepcopy(base_elo)
        calib_pace = copy.deepcopy(base_pace)
        calib_xppp = copy.deepcopy(base_xppp)
        calib_form = copy.deepcopy(base_form)
        calib_rotation = copy.deepcopy(base_rotation)
        calib_lineup_elo = copy.deepcopy(base_lineup_elo)
        calib_chemistry = copy.deepcopy(base_chemistry)
        calib_team_elo = copy.deepcopy(base_team_elo)
        calib_travel = copy.deepcopy(base_travel)
        calib_ref = copy.deepcopy(base_ref)
        calib_shot_quality = copy.deepcopy(base_shot_quality)
        calib_hapm = copy.deepcopy(base_hapm)
        calib_epm = copy.deepcopy(base_epm)

        calib_features = generate_features(
            calib_stints, calib_hier, calib_elo, calib_pace,
            odds_dict=odds_dict, update_engines=True,
            team_xppp_tracker=calib_xppp, team_form_tracker=calib_form,
            rotation_tracker=calib_rotation, lineup_elo_tracker=calib_lineup_elo,
            chemistry_tracker=calib_chemistry, team_elo_tracker=calib_team_elo,
            travel_tracker=calib_travel, epm_tracker=calib_epm,
            ref_tracker=calib_ref,
            shot_quality_tracker=calib_shot_quality,
            # Same past-only lookup as base_features (Task hapm_before_split_leak
            # fix): calib_stints is a later slice of the same train_stints span,
            # so `hapm_train_lookup` still resolves every game to a tracker
            # fit only on strictly-earlier blocks.
            hapm_tracker=hapm_train_lookup,
        )
        calib_features = engineer_interaction_features(calib_features)

        # Task calibration_slice_reuse fix: Elo isotonic, MetaScore isotonic,
        # static spread calibration, MetaWin isotonic, and ATS isotonic
        # previously all fit on the identical tail `calib_features` slice.
        # `CalibrationSliceRegistry` raises immediately if that ever
        # regresses; `chronological_game_id_partition` gives each of the
        # five its own disjoint, chronologically-contiguous sub-slice of the
        # same calibration-tail span (still strictly before test_season, so
        # no chronological leakage — only the cross-target row-reuse is
        # fixed). `_calib_slice(target)` below re-derives each target's rows
        # from the *current* `calib_features` (before/after Elo calibration
        # is applied) using this same fixed GAME_ID partition.
        calib_slice_registry = CalibrationSliceRegistry()
        calib_id_partition = chronological_game_id_partition(
            calib_features, targets=BACKTEST_CALIBRATION_TARGETS,
        )
        for _target, _ids in calib_id_partition.items():
            calib_slice_registry.register(_target, _ids)
        calib_slice_registry.assert_all_targets_registered(BACKTEST_CALIBRATION_TARGETS)

        def _calib_slice(target: str) -> pd.DataFrame:
            """The disjoint calibration sub-slice for `target`, taken from
            the *current* (possibly Elo-calibrated) `calib_features` frame —
            same GAME_ID partition throughout, so re-slicing after
            `apply_elo_calibration_df` picks up its new/overwritten columns
            without re-registering (and therefore without risking a second
            registry entry for the same target)."""
            return slice_by_game_ids(calib_features, calib_id_partition[target])

        elo_knobs = tune_elo_calibrator(base_features, calib_df=_calib_slice("elo_isotonic"))
        elo_calibrator = WalkForwardEloCalibrator(knobs=elo_knobs)
        elo_calibrator.fit(base_features, calib_df=_calib_slice("elo_isotonic"))
        if elo_calibrator.fitted:
            print(
                f"  📐 ELO 3-group calibrator: eps={elo_knobs.huber_epsilon} "
                f"blend=({elo_knobs.hier_blend_elo:.2f},{elo_knobs.hier_blend_hier:.2f}) "
                f"MAE={elo_calibrator.knobs.group_train_mae.get('combined', 0):.2f}"
            )
        base_features = apply_elo_calibration_df(base_features, elo_calibrator)
        calib_features = apply_elo_calibration_df(calib_features, elo_calibrator)

        best_params = load_cached("meta", train_seasons, train_stints) if use_tuning_cache else None
        if best_params is None:
            print("  ⏳ Tuning MetaScoreModel hyperparameters...")
            best_params = tune_margin_model(
                base_features, n_trials=n_tuning_trials_meta, feature_cols=feature_cols,
                bounds=param_bounds, season=int(test_season),
                fast_mode=fast_tuning or n_tuning_trials_meta <= 5,
                train_target=(model_kwargs or {}).get("train_target", "decision_residual"),
            )
            if use_tuning_cache:
                save_cached("meta", train_seasons, best_params, train_stints)
            param_bounds.record_best(int(test_season), meta_params_for_bounds(best_params))
        else:
            print("  ⚡ Using cached Meta params")
            param_bounds.record_best(int(test_season), meta_params_for_bounds(best_params))

        elo_knobs.elo_blend_alpha = best_params.get(
            "elo_blend_alpha", elo_knobs.elo_blend_alpha,
        )
        from pipeline.config import ELO_BLEND_ALPHA_MAX
        elo_knobs.elo_blend_alpha = float(min(float(elo_knobs.elo_blend_alpha), float(ELO_BLEND_ALPHA_MAX)))
        elo_knobs.elo_ridge_alpha = best_params.get(
            "elo_ridge_alpha", elo_knobs.elo_ridge_alpha,
        )

        mk = dict(model_kwargs)
        meta_model = MetaScoreModel(
            ridge_alpha=best_params['ridge_alpha'],
            cb_params=best_params['cb_params'],
            huber_epsilon=best_params['huber_epsilon'],
            margin_cap=best_params.get('margin_cap'),
            use_isotonic_calibration=True,
            feature_cols=feature_cols,
            elo_blend_alpha=best_params.get(
                'elo_blend_alpha', elo_knobs.elo_blend_alpha,
            ),
            elo_ridge_alpha=best_params.get(
                'elo_ridge_alpha', elo_knobs.elo_ridge_alpha,
            ),
            **mk,
        )
        meta_model.fit(
            base_features, base_features['actual_home'], base_features['actual_away'],
            calib_df=_calib_slice("metascore_isotonic"),
        )

        if tune_total_head:
            total_params = load_cached("total", train_seasons, train_stints) if use_tuning_cache else None
            if total_params is None:
                print("  ⏳ Tuning total model...")
                total_params = tune_total_model(
                    base_features,
                    n_trials=n_tuning_trials_total,
                    feature_cols=total_feature_cols(base_features),
                    bounds=param_bounds,
                    season=int(test_season),
                    fast_mode=fast_tuning or n_tuning_trials_total <= 5,
                )
                if use_tuning_cache:
                    save_cached("total", train_seasons, total_params, train_stints)
                param_bounds.record_best(int(test_season), meta_params_for_bounds(total_params))
            else:
                print("  ⚡ Using cached total params")
                param_bounds.record_best(int(test_season), meta_params_for_bounds(total_params))
            total_params = dict(total_params)
            cv_score = float(total_params.pop("_cv_score", total_params.pop("_cv_mae", np.nan)))
            apply_total = True
            if np.isfinite(cv_score) and cv_score > float(TOTAL_HEAD_CV_SANITY_CAP):
                print(
                    f"  ⚠ Total-model CV score {cv_score:.1f} exceeds cap "
                    f"({TOTAL_HEAD_CV_SANITY_CAP}) — skipping total refit"
                )
                apply_total = False
                if last_good_total_params is not None:
                    total_params = dict(last_good_total_params)
                    apply_total = True
            elif np.isfinite(cv_score):
                last_good_total_params = dict(total_params)
            if apply_total and total_params:
                total_model = MetaTotalModel(
                    ridge_alpha=total_params.get("ridge_alpha", 10.0),
                    cb_params=total_params.get("cb_params"),
                    huber_epsilon=total_params.get("huber_epsilon", 1.35),
                    feature_cols=total_params.get("feature_cols") or total_feature_cols(base_features),
                    total_elo_beta=total_params.get("total_elo_beta", 0.25),
                    train_target=total_params.get("train_target", "absolute"),
                )
                total_model.fit(base_features)

        # Always fit the canonical paired score model (actual home/away points).
        from pipeline.config import USE_CANONICAL_SCORE_PAIR
        from pipeline.score_targets import score_safe_feature_cols
        from pipeline.model import tune_score_pair_model

        if score_pair_model is None and (tune_total_head or USE_CANONICAL_SCORE_PAIR):
            pair_cols = score_safe_feature_cols(base_features)
            score_pair_params = None
            if n_tuning_trials_meta > 0:
                try:
                    score_pair_params = tune_score_pair_model(
                        base_features,
                        n_trials=min(12, max(4, n_tuning_trials_meta // 2)),
                        season=int(test_season),
                        feature_cols=pair_cols,
                        fast_mode=fast_tuning or n_tuning_trials_meta <= 8,
                    )
                except Exception as e:
                    print(f"  ⚠ tune_score_pair_model skipped: {e}")
            score_pair_model = MetaScorePairModel(
                ridge_alpha=(score_pair_params or {}).get("ridge_alpha", 10.0),
                cb_params=(score_pair_params or {}).get("cb_params"),
                huber_epsilon=(score_pair_params or {}).get("huber_epsilon", 1.35),
                feature_cols=(score_pair_params or {}).get("feature_cols") or pair_cols,
                train_target="absolute",
                enforce_score_safe_features=True,
            )
            try:
                score_pair_model.fit(base_features)
            except Exception as e:
                print(f"  ⚠ MetaScorePairModel fit skipped: {e}")
                score_pair_model = None

            cover_prob_calibrator = None
            if score_pair_model is not None and getattr(score_pair_model, "fitted", False):
                try:
                    from pipeline.shared_forecast import attach_oof_shared_forecasts
                    base_features = attach_oof_shared_forecasts(
                        base_features, meta_model, score_pair_model=score_pair_model,
                    )
                    from pipeline.score_uncertainty import (
                        apply_oof_uncertainty_to_score_pair,
                        fit_cover_calibrator_from_oof,
                    )
                    unc = apply_oof_uncertainty_to_score_pair(score_pair_model, base_features)
                    print(
                        f"  📐 Score-pair OOF σ_h={unc.get('sigma_home', float('nan')):.2f} "
                        f"σ_a={unc.get('sigma_away', float('nan')):.2f} "
                        f"corr={unc.get('residual_corr', float('nan')):.2f} "
                        f"({unc.get('source')})"
                    )
                    cover_prob_calibrator = fit_cover_calibrator_from_oof(
                        base_features,
                        sigma_margin=float(getattr(score_pair_model, "_rmse_margin", 12.0)),
                    )
                    if getattr(cover_prob_calibrator, "fitted", False):
                        print(
                            f"  📐 Cover calibrator fitted on n={cover_prob_calibrator.n_fit} OOF rows"
                        )
                except Exception as e:
                    print(f"  ⚠ OOF score-pair attach skipped: {e}")
                    cover_prob_calibrator = None

        win_model = None
        ats_classifier = None
        upset_classifier = None
        if train_win_model:
            train_feats = base_features.copy()
            # Task calibration_slice_reuse fix: `static_cal` (the "static
            # spread calibration" target) is fit only on its own disjoint
            # slice — not the same rows used below for the MetaWin/ATS
            # isotonic calibrators.
            calib_feats = _calib_slice("static_spread_calibration")
            from pipeline.config import USE_CANONICAL_SCORE_PAIR
            # Prefer OOF pair-derived margins for downstream probability heads.
            if (
                USE_CANONICAL_SCORE_PAIR
                and "sf_fair_margin" in train_feats.columns
                and train_feats["sf_fair_margin"].notna().sum() >= 30
            ):
                train_raw = train_feats["sf_fair_margin"].to_numpy(dtype=float)
                if "sf_fair_margin" in calib_feats.columns:
                    calib_raw = calib_feats["sf_fair_margin"].to_numpy(dtype=float)
                else:
                    calib_raw = meta_model._predict_raw(calib_feats)["pred_margin"]
            else:
                train_raw = meta_model._predict_raw(train_feats)["pred_margin"]
                calib_raw = meta_model._predict_raw(calib_feats)["pred_margin"]
            # Task 054: replay the rolling calibrator row-by-row over the
            # calib slice (predict, then observe) instead of fitting once on
            # the whole slice and blanket-correcting it with the *final*
            # state — the blanket pattern lets each row's correction see
            # later rows in the same slice, which disagrees with how
            # pipeline/simulate.py calibrates at live inference time.
            calib_actual = (
                calib_feats["actual_margin"].values
                if "actual_margin" in calib_feats.columns
                else np.full(len(calib_feats), np.nan)
            )
            if USE_CANONICAL_SCORE_PAIR and "sf_fair_margin" in train_feats.columns:
                # Pair-derived margins already OOF — skip spread calibrator replay
                # that was fit to the legacy margin residual head.
                calib_corrected = np.asarray(calib_raw, dtype=float)
                static_cal = None
                train_feats["pred_margin"] = np.asarray(train_raw, dtype=float)
                calib_feats["pred_margin"] = calib_corrected
            else:
                calib_corrected, static_cal = replay_rolling_spread_calibration(
                    calib_raw, calib_actual,
                    window=max(SPREAD_CALIB_WINDOW, len(calib_feats)),
                    min_samples=min(30, max(10, len(calib_feats) // 3)),
                    mode=spread_calib_mode,
                )
                calib_feats["pred_margin"] = calib_corrected
                # train_feats predates calib_feats and was never used to fit
                # static_cal, so applying its fully-replayed end state here is
                # not self-referential (no row corrects itself with its own
                # future information).
                train_feats["pred_margin"] = [static_cal.correct(float(p)) for p in train_raw]

            # OOF shared-forecast columns only (never in-sample meta predictions).
            if "sf_fair_margin" not in train_feats.columns:
                try:
                    from pipeline.shared_forecast import attach_oof_shared_forecasts
                    train_feats = attach_oof_shared_forecasts(
                        train_feats, meta_model, score_pair_model=score_pair_model,
                    )
                except Exception as e:  # noqa: BLE001
                    print(f"  ⚠ OOF shared forecast attach skipped: {e}")

            win_params = load_cached("meta_win", train_seasons, train_stints) if use_tuning_cache else None
            if win_params is None and n_tuning_trials_meta_win > 0:
                print("  ⏳ Tuning MetaWin model...")
                win_params = tune_meta_win_model(
                    train_feats,
                    n_trials=n_tuning_trials_meta_win,
                    season=int(test_season),
                    pred_margin_col="pred_margin",
                    fast_mode=fast_tuning or n_tuning_trials_meta_win <= 5,
                )
                if use_tuning_cache:
                    save_cached("meta_win", train_seasons, win_params, train_stints)
            elif win_params is None:
                win_params = {
                    "C": 0.1,
                    "elo_win_blend": None,
                    "calib_method": "isotonic",
                    "feature_cols": win_feature_cols(train_feats),
                }
            else:
                print("  ⚡ Using cached MetaWin params")

            win_model = MetaWinModel(
                C=win_params.get("C", 0.1),
                feature_cols=win_params.get("feature_cols") or win_feature_cols(train_feats),
                elo_win_blend=win_params.get("elo_win_blend"),
                calib_method=win_params.get("calib_method", "isotonic"),
            )
            # Task calibration_slice_reuse fix: MetaWin isotonic gets its own
            # disjoint slice (not `calib_feats`, which fit `static_cal`
            # above). Its `pred_margin` is produced by *applying* (not
            # re-fitting) the already-fit `static_cal` correction — this is
            # not self-referential, since `static_cal` was fit on a
            # completely different set of rows.
            win_calib_df = _calib_slice("metawin_isotonic").copy()
            if not win_calib_df.empty:
                if (
                    USE_CANONICAL_SCORE_PAIR
                    and "sf_fair_margin" in win_calib_df.columns
                    and win_calib_df["sf_fair_margin"].notna().any()
                ):
                    win_calib_df["pred_margin"] = win_calib_df["sf_fair_margin"].to_numpy(dtype=float)
                elif static_cal is not None:
                    win_raw = meta_model._predict_raw(win_calib_df)["pred_margin"]
                    win_calib_df["pred_margin"] = [static_cal.correct(float(p)) for p in win_raw]
                else:
                    win_calib_df["pred_margin"] = meta_model._predict_raw(win_calib_df)["pred_margin"]
            win_model.fit(
                train_feats,
                (train_feats["actual_home"] > train_feats["actual_away"]).astype(int),
                calib_df=win_calib_df,
            )

            from pipeline.t60_coverage import t60_betting_data_gate
            from pipeline.config import ENFORCE_T60_BETTING_GATE
            t60_gate = t60_betting_data_gate(train_feats)
            ats_status = {
                "season": int(test_season),
                "gate": t60_gate,
                "status": "ok",
            }
            if ENFORCE_T60_BETTING_GATE and not t60_gate["allow_betting_heads"]:
                print(
                    f"  ⚠ T-60 betting data gate blocked ATS head: "
                    f"{t60_gate.get('reason')} "
                    f"(adequate_seasons={t60_gate['n_adequate_seasons']}/"
                    f"{t60_gate['min_seasons_required']}, "
                    f"coverage={t60_gate['overall_decision_coverage']:.3f})"
                )
                ats_classifier = None
                ats_status["status"] = "blocked_research_only"
                ats_status_records.append(ats_status)
            else:
                if not t60_gate["allow_betting_heads"]:
                    print(
                        f"  ⚠ T-60 coverage below promote threshold "
                        f"({t60_gate['n_adequate_seasons']} seasons); "
                        f"fitting ATS for research only (ENFORCE_T60_BETTING_GATE=False)."
                    )
                    ats_status["status"] = "research_only_fitted"
                ats_classifier = ATSClassifier(feature_cols=feature_cols)
                ats_train = train_feats.copy()
                # Task calibration_slice_reuse fix: ATS isotonic gets its own
                # disjoint slice, distinct from both `calib_feats`
                # (static_spread_calibration) and `win_calib_df` (metawin_isotonic)
                # above.
                ats_calib = _calib_slice("ats_isotonic").copy()
                ats_train["pred_margin"] = train_feats["pred_margin"]
                if not ats_calib.empty:
                    if (
                        USE_CANONICAL_SCORE_PAIR
                        and "sf_fair_margin" in ats_calib.columns
                        and ats_calib["sf_fair_margin"].notna().any()
                    ):
                        ats_calib["pred_margin"] = ats_calib["sf_fair_margin"].to_numpy(dtype=float)
                    elif static_cal is not None:
                        ats_raw = meta_model._predict_raw(ats_calib)["pred_margin"]
                        ats_calib["pred_margin"] = [static_cal.correct(float(p)) for p in ats_raw]
                    else:
                        ats_calib["pred_margin"] = meta_model._predict_raw(ats_calib)["pred_margin"]
                if "market_spread" not in ats_train.columns:
                    if "decision_spread" in ats_train.columns:
                        ats_train["market_spread"] = ats_train["decision_spread"]
                        if "decision_spread" in ats_calib.columns:
                            ats_calib["market_spread"] = ats_calib["decision_spread"]
                    # Never fall back to closing_spread (evaluation-only / leak).
                # Drop rows without a usable decision/market line for ATS.
                for frame_name, frame in (("ats_train", ats_train), ("ats_calib", ats_calib)):
                    if frame is None or frame.empty:
                        continue
                    line = frame["market_spread"] if "market_spread" in frame.columns else None
                    if line is None and "decision_spread" in frame.columns:
                        line = frame["decision_spread"]
                    if line is not None:
                        keep = line.notna()
                        if frame_name == "ats_train":
                            ats_train = frame.loc[keep].copy()
                        else:
                            ats_calib = frame.loc[keep].copy()
                ats_classifier.fit(ats_train, calib_df=ats_calib)
                ats_status["fitted"] = bool(getattr(ats_classifier, "fitted", False))
                ats_status_records.append(ats_status)

            if ats_classifier is not None:
                from pipeline.config import USE_UPSET_CLASSIFIER
                if USE_UPSET_CLASSIFIER:
                    upset_classifier = UpsetClassifier()
                    upset_train = ats_train.copy()
                    upset_calib = ats_calib.copy() if ats_calib is not None else None
                    upset_classifier.fit(upset_train, calib_df=upset_calib)

        # Reuse post-calibration engines (skip redundant third feature pass)
        full_hier = calib_hier
        full_elo = calib_elo
        sim_pace = copy.deepcopy(calib_pace)
        sim_xppp = copy.deepcopy(calib_xppp)
        sim_form = copy.deepcopy(calib_form)
        sim_rotation = copy.deepcopy(calib_rotation)
        sim_lineup_elo = copy.deepcopy(calib_lineup_elo)
        sim_chemistry = copy.deepcopy(calib_chemistry)
        sim_team_elo = copy.deepcopy(calib_team_elo)
        sim_travel = copy.deepcopy(calib_travel)
        sim_ref = copy.deepcopy(calib_ref)
        sim_shot_quality = copy.deepcopy(calib_shot_quality)
        sim_hapm = copy.deepcopy(calib_hapm)
        sim_vol = TeamVolatilityTracker()
        sim_epm = copy.deepcopy(calib_epm)

        if i > 0:
            returning_ids = set(full_elo.players.keys()) if full_elo.players else None
            full_hier.offseason_revert()
            full_elo.offseason_revert(returning_player_ids=returning_ids)
            # New test season: empty prior_rosters → high turnover warm-start
            from pipeline.trackers import refresh_team_priors_for_season
            refresh_team_priors_for_season(
                sim_xppp,
                elo_tracker=full_elo,
                pace_tracker=sim_pace,
                rotation_tracker=sim_rotation,
                prior_rosters={},
            )

        # Walk-forward thresholds from prior seasons
        edge_thr = 0.0 if not uses_edge_gates() else GOOD_BET_EDGE
        fav_dec = ML_MAX_FAVORITE_DECIMAL_DEFAULT
        ou_thr = float(OU_MIN_EDGE)
        conf_thr = float(MIN_CONFIDENCE_SCORE)
        conf_max = MAX_CONFIDENCE_SCORE
        min_ml_thr = float(MIN_ML_WIN_PCT)
        if not compiled_results and should_use_season1_calibrator():
            calib_for_conf = calib_features.copy()
            calib_raw = meta_model._predict_raw(calib_for_conf)["pred_margin"]
            # Task 054: same rolling replay fix as above — no blanket
            # self-referential correction of the calibration slice.
            calib_for_conf_actual = (
                calib_for_conf["actual_margin"].values
                if "actual_margin" in calib_for_conf.columns
                else np.full(len(calib_for_conf), np.nan)
            )
            calib_for_conf_corrected, _ = replay_rolling_spread_calibration(
                calib_raw, calib_for_conf_actual,
                window=max(SPREAD_CALIB_WINDOW, len(calib_for_conf)),
                min_samples=min(30, max(10, len(calib_for_conf) // 3)),
                mode=spread_calib_mode,
            )
            calib_for_conf["pred_margin"] = calib_for_conf_corrected
            season_lbl = f"{int(test_season)-1}-{int(test_season)}-calib"
            calib_ats_df = build_calib_split_ats_frame(
                calib_for_conf,
                pred_margin_col="pred_margin",
                season_label=season_lbl,
            )
            if len(calib_ats_df) >= 30:
                if str(CONFIDENCE_CALIB_METHOD).lower() == "auto":
                    bet_calibrator, s1_meta = fit_walkforward_confidence_calibrator(
                        calib_ats_df,
                        train_season_label=season_lbl,
                        test_season_label=f"{int(test_season)-1}-{int(test_season)}",
                        season_index=0,
                        weight_center=confidence_weight_center,
                        method="auto",
                    )
                    if s1_meta:
                        phase2a_rows.append(s1_meta)
                        confidence_weight_center = shrink_weight_center(
                            confidence_weight_center,
                            s1_meta.get("suggested_weights", confidence_weight_center),
                        )
                    conf_thr = float(s1_meta.get("train_min_confidence", MIN_CONFIDENCE_SCORE)) if s1_meta else float(MIN_CONFIDENCE_SCORE)
                    conf_max = s1_meta.get("train_max_confidence", MAX_CONFIDENCE_SCORE) if s1_meta else MAX_CONFIDENCE_SCORE
                    if conf_max is not None:
                        conf_max = float(conf_max)
                    print(
                        f"  📐 Season-1 auto calibrator "
                        f"({s1_meta.get('calib_method', 'auto') if s1_meta else 'auto'}; "
                        f"{len(calib_ats_df)} games, min_confidence={conf_thr:.0f}"
                        f"{f'-{conf_max:.0f}' if conf_max is not None else ''})"
                    )
                else:
                    bet_calibrator.fit(
                        calib_ats_df,
                        method=CONFIDENCE_CALIB_METHOD,
                        scope=CONFIDENCE_CALIB_SCOPE,
                    )
                    conf_thr, conf_max = walkforward_confidence_gate(
                        calib_ats_df,
                        default_min=MIN_CONFIDENCE_SCORE,
                        default_max=MAX_CONFIDENCE_SCORE,
                        bet_calibrator=bet_calibrator if bet_calibrator._fitted else None,
                    )
                    print(
                        f"  📐 Season-1 calibrator fit on calib split "
                        f"({len(calib_ats_df)} games, min_confidence={conf_thr:.0f}"
                        f"{f'-{conf_max:.0f}' if conf_max is not None else ''})"
                    )
        if i > 0 and compiled_results:
            prior = pd.concat(compiled_results, ignore_index=True)
            prior_year = compiled_results[-1]
            if uses_edge_gates():
                wf_floor = WALKFORWARD_EDGE_MIN_FLOOR
                if wf_floor is None and BET_SELECTION_MODE != "legacy_tiers":
                    wf_floor = 5.5
                edge_thr = walkforward_edge_threshold(
                    prior, default=GOOD_BET_EDGE, optimize="bucket_roi", min_floor=wf_floor,
                )
            else:
                edge_thr = 0.0
            fav_dec = walkforward_favorite_decimal(
                prior, default=ML_MAX_FAVORITE_DECIMAL_DEFAULT,
            )
            ou_thr = walkforward_ou_edge(prior, default=float(OU_MIN_EDGE))
            calib_source = prior_year
            train_lbl = (
                prior_year["simulated_season_window"].iloc[0]
                if "simulated_season_window" in prior_year.columns and not prior_year.empty
                else "prior-year"
            )
            test_lbl = f"{int(test_season)-1}-{int(test_season)}"
            if CONFIDENCE_MODE == "unified":
                bet_calibrator, p2a_meta = fit_walkforward_confidence_calibrator(
                    calib_source,
                    train_season_label=str(train_lbl),
                    test_season_label=test_lbl,
                    season_index=len(compiled_results),
                    weight_center=confidence_weight_center,
                )
                phase2a_rows.append(p2a_meta)
                confidence_weight_center = shrink_weight_center(
                    confidence_weight_center,
                    p2a_meta.get("suggested_weights", confidence_weight_center),
                )
                conf_thr = float(p2a_meta.get("train_min_confidence", MIN_CONFIDENCE_SCORE))
                conf_max = p2a_meta.get("train_max_confidence", MAX_CONFIDENCE_SCORE)
                if conf_max is not None:
                    conf_max = float(conf_max)
                feat_w = p2a_meta.get("feature_weights") or logistic_feature_importance(bet_calibrator)
                if feat_w:
                    top = sorted(feat_w.items(), key=lambda kv: abs(kv[1]), reverse=True)[:5]
                    coefs = ", ".join(f"{k}={v:+.2f}" for k, v in top)
                    print(
                        f"  📐 Phase 2a inline ({p2a_meta.get('calib_method', CONFIDENCE_CALIB_METHOD)}): "
                        f"{coefs}  C={p2a_meta.get('logistic_c', bet_calibrator.logistic_c):.2f}"
                    )
            else:
                bet_calibrator.fit(
                    calib_source,
                    method=CONFIDENCE_CALIB_METHOD,
                    scope=CONFIDENCE_CALIB_SCOPE,
                )
                conf_thr, conf_max = walkforward_confidence_gate(
                    prior_year,
                    default_min=MIN_CONFIDENCE_SCORE,
                    default_max=MAX_CONFIDENCE_SCORE,
                    bet_calibrator=bet_calibrator if bet_calibrator._fitted else None,
                )
            ml_calibrator.fit(prior_year, scope=ML_CALIB_SCOPE)
            disagreement_model.fit(prior)
            if SPREAD_CALIB_ADAPTIVE_WINDOW:
                active_spread_window = select_spread_calib_window(
                    prior_year,
                    candidates=SPREAD_CALIB_WINDOW_CANDIDATES,
                    default=active_spread_window,
                )
            ml_tag = f"ml={ML_CALIB_METHOD}" if ml_calibrator._fitted else "ml=raw"
            print(
                f"  📊 Walk-forward edge={edge_thr}  max_fav_dec={fav_dec:.2f}  "
                f"ou_edge={ou_thr}  min_confidence={conf_thr:.0f}"
                f"{f'-{conf_max:.0f}' if conf_max is not None else ''}  "
                f"min_ml_win%={min_ml_thr:.0f}  "
                f"spread_window={active_spread_window}  "
                f"confidence={CONFIDENCE_MODE}/{CONFIDENCE_CALIB_METHOD}  {ml_tag} (fit {train_lbl})"
            )

        rolling_calibrator = RollingPlattCalibrator(
            window_size=300, min_samples=20,
            cold_start_fn=meta_model.calibrate_prob,
            variance_aware=variance_aware_winprob,
        )
        spread_calibrator = SpreadCalibrator(window=active_spread_window, mode=spread_calib_mode)

        results = run_simulation(
            season_df=test_stints,
            hier_engine=full_hier,
            elo_tracker=full_elo,
            meta_model=meta_model,
            pace_tracker=sim_pace,
            odds_dict=odds_dict,
            calibrator=rolling_calibrator,
            team_xppp_tracker=sim_xppp,
            team_form_tracker=sim_form,
            spread_calibrator=spread_calibrator,
            rotation_tracker=sim_rotation,
            lineup_elo_tracker=sim_lineup_elo,
            chemistry_tracker=sim_chemistry,
            team_elo_tracker=sim_team_elo,
            travel_tracker=sim_travel,
            epm_tracker=sim_epm,
            ref_tracker=sim_ref,
            shot_quality_tracker=sim_shot_quality,
            hapm_tracker=sim_hapm if use_hapm_priors else None,
            min_edge_pts=edge_thr,
            max_favorite_decimal=fav_dec,
            ou_min_edge=ou_thr,
            confidence_calibrator=bet_calibrator if bet_calibrator._fitted else None,
            win_model=win_model,
            total_model=total_model,
            score_pair_model=score_pair_model,
            cover_prob_calibrator=cover_prob_calibrator,
            ats_classifier=ats_classifier,
            upset_classifier=upset_classifier,
            vol_tracker=sim_vol,
            elo_calibrator=elo_calibrator if elo_calibrator.fitted else None,
            require_elo_agreement=require_elo_agreement,
            disagreement_model=disagreement_model if disagreement_model.fitted else None,
            min_confidence_threshold=conf_thr,
            max_confidence_threshold=conf_max,
            ml_calibrator=ml_calibrator if ml_calibrator._fitted else None,
            min_ml_win_pct=min_ml_thr,
            use_venn_abers_filter=use_venn_abers_filter,
        )

        print(f"  Season {test_season} produced {len(results)} rows." if results is not None else "  Season returned None.")
        if results is not None and not results.empty:
            results['simulated_season_window'] = f"{int(test_season)-1}-{int(test_season)}"
            results['EDGE_THRESHOLD'] = edge_thr
            results['MAX_FAVORITE_DECIMAL'] = fav_dec
            results['OU_MIN_EDGE'] = ou_thr
            results['MIN_CONFIDENCE_SCORE'] = conf_thr
            results['MAX_CONFIDENCE_SCORE'] = conf_max
            results['MIN_ML_WIN_PCT'] = min_ml_thr
            results['PHASE2A_INLINE'] = int(
                CONFIDENCE_MODE == "unified" and i > 0 and len(compiled_results) > 0
            )
            chosen_method = CONFIDENCE_CALIB_METHOD
            if phase2a_rows and phase2a_rows[-1].get("calib_method"):
                chosen_method = phase2a_rows[-1]["calib_method"]
            results['CONFIDENCE_CALIB_METHOD'] = chosen_method

            # P0.4 / Epic 11.4 (bench_frozen_elo_interval registry entry):
            # `WalkForwardEloCalibrator.update_residuals` was never called
            # anywhere in the live/backtest path, so `predict_interval`
            # always fell back to its hardcoded default `half = 12.0` width
            # forever, regardless of how well- or poorly-calibrated the
            # model actually was in a given season. Wire it here, once per
            # graded walk-forward season, using this season's own graded
            # |prediction − actual| residuals — the natural "after graded
            # games" loop for this calibrator (each `test_season` iteration
            # is itself the walk-forward step). Only uses this season's
            # already-realized outcomes (no future leakage into interval
            # widths used for *this* season's own bets).
            if (
                elo_calibrator.fitted
                and "ELO_MARGIN_CALIBRATED" in results.columns
                and "ACTUAL_MARGIN" in results.columns
            ):
                _pred = pd.to_numeric(results["ELO_MARGIN_CALIBRATED"], errors="coerce")
                _actual = pd.to_numeric(results["ACTUAL_MARGIN"], errors="coerce")
                _mask = np.isfinite(_pred) & np.isfinite(_actual)
                if int(_mask.sum()) >= 20:
                    season_residuals = (_pred[_mask] - _actual[_mask]).abs().tolist()
                    elo_calibrator.update_residuals(season_residuals)
            if bet_calibrator._fitted:
                results['PHASE2A_LOGISTIC_C'] = float(bet_calibrator.logistic_c)
            compiled_results.append(results)
        else:
            print(f"  ⚠️ No results for season {test_season}")

    if not compiled_results:
        return pd.DataFrame()

    final = pd.concat(compiled_results, ignore_index=True)

    if walkforward_edge and uses_edge_gates():
        print("\n  (Diagnostic) Post-hoc edge threshold comparison:")
        for idx, season_res in enumerate(compiled_results):
            if idx == 0:
                thr = GOOD_BET_EDGE
            else:
                prior = pd.concat(compiled_results[:idx], ignore_index=True)
                thr = walkforward_edge_threshold(prior, default=GOOD_BET_EDGE, optimize="clv_roi")
            win = season_res.get("simulated_season_window", pd.Series(["?"])).iloc[0]
            print(f"    Season {win}: post-hoc threshold = {thr}")

    final = add_all_profile_columns(final)

    if compiled_results and not final.empty:
        prefix = STATE_DIR / "latest"
        try:
            full_elo.save_state(prefix.with_name(prefix.name + "_elo.pkl"))
            full_hier.save_state(prefix.with_name(prefix.name + "_hier.pkl"))
            if hasattr(sim_pace, "save_state"):
                sim_pace.save_state(prefix.with_name(prefix.name + "_pace.pkl"))
            meta_model.save(prefix.with_name(prefix.name + "_meta.pkl"))
            if win_model is not None:
                win_model.save(prefix.with_name(prefix.name + "_win.pkl"))
            if total_model is not None and getattr(total_model, "fitted", False):
                total_model.save(prefix.with_name(prefix.name + "_total.pkl"))
                total_model.save(STATE_DIR / "total.pkl")
            if score_pair_model is not None and getattr(score_pair_model, "fitted", False):
                score_pair_model.save(prefix.with_name(prefix.name + "_score_pair.pkl"))
                score_pair_model.save(STATE_DIR / "score_pair.pkl")
            if ats_classifier is not None and getattr(ats_classifier, "fitted", False):
                ats_classifier.save(prefix.with_name(prefix.name + "_ats.pkl"))
            # Persist ATS gate status even when blocked (Finding 6 visibility).
            if ats_status_records:
                (STATE_DIR / "ats_status.json").write_text(
                    json.dumps({"records": ats_status_records}, indent=2, default=str)
                )
            if upset_classifier is not None and getattr(upset_classifier, "fitted", False):
                upset_classifier.save(prefix.with_name(prefix.name + "_upset.pkl"))
                from pipeline.upset_classifier import default_upset_classifier_path
                upset_classifier.save(default_upset_classifier_path())
            sim_vol.save_state(prefix.with_name(prefix.name + "_vol.pkl"))
            sim_rotation.save_state(prefix.with_name(prefix.name + "_rotation.pkl"))
            sim_lineup_elo.save_state(prefix.with_name(prefix.name + "_lineup_elo.pkl"))
            sim_chemistry.save_state(prefix.with_name(prefix.name + "_chemistry.pkl"))
            sim_team_elo.save_state(prefix.with_name(prefix.name + "_team_elo.pkl"))
            sim_travel.save_state(prefix.with_name(prefix.name + "_travel.pkl"))
            bet_calibrator.save(default_calibrator_path())
            if ml_calibrator._fitted:
                ml_calibrator.save(default_ml_calibrator_path())
            if disagreement_model.fitted:
                disagreement_model.save(default_disagreement_path())
            if elo_calibrator.fitted:
                elo_calibrator.save(default_elo_calibrator_path())
            save_elo_knobs(elo_knobs)
            elo_knobs_merged = elo_knobs.to_dict()
            elo_knobs_merged["elo_blend_alpha"] = float(meta_model.elo_blend_alpha)
            elo_knobs_merged["elo_ridge_alpha"] = float(meta_model.elo_ridge_alpha)
            edge_val = float(final["EDGE_THRESHOLD"].iloc[-1]) if "EDGE_THRESHOLD" in final.columns else GOOD_BET_EDGE
            import subprocess
            git_hash = ""
            try:
                git_hash = subprocess.check_output(
                    ["git", "rev-parse", "--short", "HEAD"],
                    cwd=str(STATE_DIR.parent),
                    stderr=subprocess.DEVNULL,
                    text=True,
                ).strip()
            except Exception:
                pass
            om = {}
            if not final.empty:
                m = compute_metrics(final)
                overall = m[m["season"] == "ALL"]
                if not overall.empty:
                    om = overall.iloc[0].to_dict()
            tuning_state = {
                "walkforward_edge_threshold": edge_val,
                "optimal_bet_edge": edge_val,
                "max_favorite_decimal": float(final["MAX_FAVORITE_DECIMAL"].iloc[-1]) if "MAX_FAVORITE_DECIMAL" in final.columns else 1.45,
                "ou_min_edge": float(final["OU_MIN_EDGE"].iloc[-1]) if "OU_MIN_EDGE" in final.columns else float(OU_MIN_EDGE),
                "min_confidence_score": _series_last_optional_float(
                    final, "MIN_CONFIDENCE_SCORE", float(MIN_CONFIDENCE_SCORE),
                ),
                "max_confidence_score": _series_last_optional_float(
                    final, "MAX_CONFIDENCE_SCORE", MAX_CONFIDENCE_SCORE,
                ),
                "min_ml_win_pct": float(final["MIN_ML_WIN_PCT"].iloc[-1]) if "MIN_ML_WIN_PCT" in final.columns else float(MIN_ML_WIN_PCT),
                "last_walkforward_season": final["simulated_season_window"].iloc[-1],
                "phase2a_inline": bool(phase2a_rows),
                "confidence_calib_method": CONFIDENCE_CALIB_METHOD,
                "elo_calibration": elo_knobs_merged,
                "config_flags": {
                    "BET_SELECTION_MODE": BET_SELECTION_MODE,
                    "CONFIDENCE_SELECTION_MODE": CONFIDENCE_SELECTION_MODE,
                    "CONFIDENCE_MIN_EDGE": CONFIDENCE_MIN_EDGE,
                    "MIN_EDGE_BUCKET": MIN_EDGE_BUCKET,
                    "EDGE_AVOID_BAND": EDGE_AVOID_BAND,
                    "CONFIDENCE_GATE_MAX_LIFT": CONFIDENCE_GATE_MAX_LIFT,
                    "CALIBRATION_MODE": CALIBRATION_MODE,
                    "STAKE_SIZING_MODE": STAKE_SIZING_MODE,
                    "WIN_PROB_SOURCE": WIN_PROB_SOURCE,
                    "ARTIFACT_SCHEMA_VERSION": ARTIFACT_SCHEMA_VERSION,
                },
                "suggested_min_edge": float(MIN_EDGE_BUCKET),
                "bet_selection_mode": BET_SELECTION_MODE,
                "avoid_bands": EDGE_AVOID_BAND,
                "git_hash": git_hash,
                "seasons": list(final["simulated_season_window"].unique()) if "simulated_season_window" in final.columns else [],
                "overall_metrics": om,
            }
            if "MIN_CONFIDENCE_SCORE" in final.columns:
                tuning_state["min_confidence_by_season"] = (
                    final.groupby("simulated_season_window", dropna=False)["MIN_CONFIDENCE_SCORE"]
                    .first()
                    .astype(float)
                    .to_dict()
                )
            if phase2a_rows:
                tuning_state["unified_weight_tuning"] = phase2a_rows
                latest_p2a = phase2a_rows[-1]
                tuning_state["suggested_weights_latest"] = latest_p2a.get(
                    "suggested_weights", dict(DEFAULT_CONFIDENCE_WEIGHTS),
                )
                tuning_state["feature_weights_latest"] = latest_p2a.get("feature_weights", {})
                tuning_state["suggested_logistic_c"] = latest_p2a.get("logistic_c")
                if latest_p2a.get("train_min_confidence") is not None:
                    tuning_state["suggested_min_confidence"] = float(
                        latest_p2a["train_min_confidence"],
                    )
                save_phase2a_walkforward_state(phase2a_rows, latest_center=confidence_weight_center)
            (STATE_DIR / "tuning_results.json").write_text(json.dumps(tuning_state, indent=2, default=str))
            (STATE_DIR / "run_manifest.json").write_text(json.dumps(tuning_state, indent=2, default=str))
            print(f"  💾 Saved engine state to {STATE_DIR}/latest_*.pkl")
            _persist_backtest_metadata({
                "n_games": int(len(final)),
                "last_walkforward_season": final["simulated_season_window"].iloc[-1],
            })
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ Could not save engine state: {e}")

    return final
