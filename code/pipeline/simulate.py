# Cell 12 – run_simulation (rewritten with SOS, proper Kelly, Platt rolling calibration)
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pipeline.config import (
    DEFAULT_LEAGUE_XPPP,
    ELO_AGREEMENT_EXTRA_EDGE,
    GOOD_BET_EDGE,
    ML_MIN_EV,
    SOS_WINDOW,
    TEAM_MAP,
    BET_SELECTION_MODE,
    MIN_EDGE_BUCKET,
    STAKE_SIZING_MODE,
    ATS_CLASSIFIER_BLEND,
    ATS_CLASSIFIER_MIN_PROB,
    WIN_PROB_SOURCE,
    REQUIRE_META_WIN_WHEN_FITTED,
    CONFIDENCE_SELECTION_MODE,
    MIN_CONFIDENCE_SCORE,
    MAX_CONFIDENCE_SCORE,
    CONFIDENCE_MIN_EDGE,
)
from pipeline.bet_confidence import build_confidence_features
from pipeline.bet_selection import (
    analysis_min_edge_pts,
    apply_bet_selection_gates,
    edge_bucket_min_for_stakes,
    edge_stake_multiplier,
    passes_confidence_actionable_gates,
    spread_lean_from_edge,
    uses_edge_gates,
)
from pipeline.calibration_policy import (
    blend_ats_classifier_cover,
    select_cover_prob,
    should_use_bet_calibrator,
    should_use_rolling_platt_when_win_model,
)
from pipeline.dates import coerce_game_date, is_valid_timestamp
from pipeline.feature_utils import _top_lineup
from pipeline.game_features import build_game_features
from pipeline.game_updates import update_trackers_after_game
from pipeline.market import (
    american_to_decimal,
    bet_analysis,
    elo_meta_agreement,
    fair_probs_from_ml_pair,
    spread_kelly_fraction,
)
from pipeline.venn_abers import market_implied_inside_interval
from pipeline.metrics import add_clv_columns, variance_aware_edge_threshold
from pipeline.model import build_feature_row
from pipeline.stake_profiles import PROFILES, compute_ml_stake, compute_stake
from pipeline.teamstats import precompute_game_team_stats
from pipeline.utils import _parse_player_string


def run_simulation(
    season_df,
    hier_engine,
    elo_tracker,
    meta_model,
    pace_tracker,
    odds_dict,
    calibrator=None,               # should be a RollingPlattCalibrator instance
    team_xppp_tracker=None,
    team_form_tracker=None,
    spread_calibrator=None,
    conformal=None,
    rotation_tracker=None,
    inactive_map: dict = None,
    sos_window: int = None,
    lineup_elo_tracker=None,
    chemistry_tracker=None,
    team_elo_tracker=None,
    travel_tracker=None,
    epm_tracker=None,
    ref_tracker=None,
    shot_quality_tracker=None,
    hapm_tracker=None,
    rapm_tracker=None,
    epva_tracker=None,
    require_rlm: bool = False,
    ou_bet: bool = True,
    min_edge_pts: float = None,
    max_favorite_decimal: float = 1.45,
    ou_min_edge: float = 3.0,
    confidence_calibrator=None,
    win_model=None,
    total_model=None,
    score_pair_model=None,
    cover_prob_calibrator=None,
    ats_classifier=None,
    upset_classifier=None,
    vol_tracker=None,
    edge_threshold: float = None,
    elo_calibrator=None,
    require_elo_agreement: bool = False,
    elo_agreement_extra_edge: float = None,
    disagreement_model=None,
    min_confidence_threshold: float | None = None,
    max_confidence_threshold: float | None = None,
    ml_calibrator=None,
    min_ml_win_pct: float | None = None,
    use_venn_abers_filter: bool | None = None,
    league_rolling_stats=None,
    hierarchical_pace=None,
    hier_shot_rates=None,
    minutes_model=None,
    target_family_calibrator=None,
):
    """
    Walk‑forward simulation for a single season.
    """
    if "game_date" in season_df.columns:
        # Task 015 + Task 008: coerce mixed ISO/US dates, canonicalize per
        # GAME_ID, sort, then assert monotonic decision timestamps.
        from pipeline.dates import prepare_chronological_stints
        season_df = prepare_chronological_stints(season_df, context="run_simulation")

    if min_edge_pts is None:
        min_edge_pts = edge_threshold if edge_threshold is not None else GOOD_BET_EDGE
    if elo_agreement_extra_edge is None:
        elo_agreement_extra_edge = ELO_AGREEMENT_EXTRA_EDGE
    from pipeline.config import (
        MIN_ML_WIN_PCT, USE_VENN_ABERS_FILTER,
        USE_HIERARCHICAL_PACE, USE_HIERARCHICAL_SHOT_ZONES, USE_ROTATION_SCENARIOS,
        USE_TARGET_FAMILY_CALIBRATION, USE_HYBRID_STRUCTURED_BLEND,
        HYBRID_TOTAL_BETA, HYBRID_MARGIN_BETA,
    )
    if min_ml_win_pct is None:
        min_ml_win_pct = float(MIN_ML_WIN_PCT)
    if use_venn_abers_filter is None:
        use_venn_abers_filter = bool(USE_VENN_ABERS_FILTER)

    if hierarchical_pace is None and USE_HIERARCHICAL_PACE:
        try:
            from pipeline.pace_hierarchy import HierarchicalPaceModel
            hierarchical_pace = HierarchicalPaceModel(baseline=pace_tracker)
        except Exception:
            hierarchical_pace = None
    if hier_shot_rates is None and USE_HIERARCHICAL_SHOT_ZONES:
        try:
            from pipeline.shot_hierarchy import HierarchicalZoneRates
            hier_shot_rates = HierarchicalZoneRates()
        except Exception:
            hier_shot_rates = None
    if minutes_model is None and USE_ROTATION_SCENARIOS:
        try:
            from pipeline.minutes_hierarchy import HierarchicalMinutesModel
            minutes_model = HierarchicalMinutesModel()
        except Exception:
            minutes_model = None
    if target_family_calibrator is None and USE_TARGET_FAMILY_CALIBRATION:
        try:
            from pipeline.target_calibration import TargetFamilyCalibrationRegistry
            target_family_calibrator = TargetFamilyCalibrationRegistry()
        except Exception:
            target_family_calibrator = None

    if league_rolling_stats is None:
        from pipeline.config import USE_ROLLING_LEAGUE_Z
        if USE_ROLLING_LEAGUE_Z:
            from pipeline.trackers import LeagueRollingStats
            league_rolling_stats = LeagueRollingStats()

    results = []
    resid_hist = []
    last_date = {}
    team_game_dates = defaultdict(lambda: deque(maxlen=20))   # schedule density

    season_start_date = None
    team_games_played = defaultdict(int)
    team_rosters_seen = defaultdict(set)
    lineup_cache = {}
    season_player_ids = set()
    season_year = None

    # Recent form deques (net rating)
    RECENT_WINDOW = 10
    team_recent_net = defaultdict(lambda: deque(maxlen=RECENT_WINDOW))
    # Team-specific home-court: rolling margins split by venue.
    team_home_margin = defaultdict(lambda: deque(maxlen=20))
    team_road_margin = defaultdict(lambda: deque(maxlen=20))

    # --- Strength of Schedule (opponent net rating history) ---
    if sos_window is None:
        sos_window = SOS_WINDOW
    opponent_history = defaultdict(lambda: deque(maxlen=sos_window))

    # Vectorized per-game aggregation (replaces inner per-stint Python loop)
    game_stats_map = precompute_game_team_stats(season_df)

    last_valid_date = None
    skipped_games = 0

    for game_id, group in tqdm(season_df.groupby("GAME_ID", sort=False), desc="Walk-forward simulation"):
        first_row = group.iloc[0]
        raw_gdate = first_row.get("game_date", pd.NaT)
        home_team = TEAM_MAP.get(first_row.get("home_team", "HOME"), first_row.get("home_team", "HOME"))
        away_team = TEAM_MAP.get(first_row.get("away_team", "AWAY"), first_row.get("away_team", "AWAY"))

        season_yr = None
        if is_valid_timestamp(season_start_date):
            season_yr = season_start_date.year
        gdate = coerce_game_date(
            raw_gdate, game_id=game_id, prev_date=last_valid_date, season_start_year=season_yr,
        )
        if gdate is None:
            skipped_games += 1
            continue
        last_valid_date = gdate

        if isinstance(gdate, pd.Timestamp):
            current_season = gdate.year + (1 if gdate.month >= 9 else 0)
            if season_year is None:
                season_year = gdate.year
        else:
            current_season = 2025

        if season_start_date is None or (
            is_valid_timestamp(gdate) and gdate.month > 8
            and is_valid_timestamp(season_start_date)
            and (gdate - season_start_date).days > 150
        ):
            if isinstance(gdate, pd.Timestamp) and season_year is not None and gdate.year > season_year:
                returning_ids = set(season_player_ids) if season_player_ids else None
                prior_rosters = {t: set(ids) for t, ids in team_rosters_seen.items()}
                hier_engine.offseason_revert()
                elo_tracker.offseason_revert(returning_player_ids=returning_ids)
                from pipeline.trackers import refresh_team_priors_for_season
                refresh_team_priors_for_season(
                    team_xppp_tracker,
                    elo_tracker=elo_tracker,
                    pace_tracker=pace_tracker,
                    rotation_tracker=rotation_tracker,
                    prior_rosters=prior_rosters,
                    lineup_cache=lineup_cache,
                )
                season_player_ids = set()
            season_start_date = gdate if is_valid_timestamp(gdate) else pd.Timestamp.now()
            if isinstance(gdate, pd.Timestamp):
                season_year = gdate.year
            team_games_played.clear()
            team_rosters_seen.clear()
            lineup_cache.clear()
            team_recent_net.clear()
            opponent_history.clear()   # also reset SOS at season boundary

        # Task 031: same fix as `pipeline/features.py::generate_features` —
        # `home_starters`/`away_starters` feed T-60 feature generation below
        # and must never come from this game's own actual PBP lineup
        # (Rule 4). `lineup_cache` only ever holds the roster observed in a
        # previously *completed* game (updated post-game below); an unseen
        # team this season has a genuinely unknown pre-game roster (Rule 5),
        # not a fabricated one.
        home_starters = lineup_cache.get(home_team, [])
        away_starters = lineup_cache.get(away_team, [])
        # Actual PBP-observed lineup for *this* game — postgame-only, used
        # to update rosters-seen bookkeeping and `lineup_cache` for future
        # games, never as this game's own T-60 feature input.
        first_stint = group.iloc[0]
        actual_home_starters = _parse_player_string(first_stint.get("HOME_players", ""))
        actual_away_starters = _parse_player_string(first_stint.get("AWAY_players", ""))

        # Aggregate actual outcomes (precomputed, vectorized).
        # Task 014: act_h/act_a prefer the canonical, independently-verified
        # final score (Task 006/007) via precompute_game_team_stats; never
        # re-sum home_pts/away_pts from filtered stints in this loop.
        gs = game_stats_map.get(game_id, {})
        act_h = float(gs.get("act_h", 0.0))
        act_a = float(gs.get("act_a", 0.0))
        tot_poss = float(gs.get("tot_poss", 0.0))
        home_tot_poss = float(gs.get("home_poss", 0.0))
        away_tot_poss = float(gs.get("away_poss", 0.0))
        home_tot_xpts = float(gs.get("home_xpts", 0.0))
        away_tot_xpts = float(gs.get("away_xpts", 0.0))

        if home_tot_poss == 0:
            home_tot_poss = tot_poss / 2
        if away_tot_poss == 0:
            away_tot_poss = tot_poss / 2
        if act_h == 0 and act_a == 0:
            act_h = group.get("HOME_FINAL_SCORE", pd.Series([110.0])).max()
            act_a = group.get("AWAY_FINAL_SCORE", pd.Series([107.0])).max()
        if tot_poss == 0:
            tot_poss = 100.0

        home_xppp_game = home_tot_xpts / home_tot_poss if home_tot_poss > 0 else DEFAULT_LEAGUE_XPPP
        away_xppp_game = away_tot_xpts / away_tot_poss if away_tot_poss > 0 else DEFAULT_LEAGUE_XPPP

        # Rotation-weighted expected lineups
        h_weights = rotation_tracker.expected_weights(home_team, home_starters) if rotation_tracker else []
        a_weights = rotation_tracker.expected_weights(away_team, away_starters) if rotation_tracker else []
        home_lineup = _top_lineup(h_weights) or home_starters
        away_lineup = _top_lineup(a_weights) or away_starters

        inactive_ids = set()
        if inactive_map is not None:
            inactive_ids = {str(x) for x in inactive_map.get(game_id, [])}

        feat = build_game_features(
            game_id=game_id,
            gdate=gdate,
            home_team=home_team,
            away_team=away_team,
            home_starters=home_starters,
            away_starters=away_starters,
            elo_tracker=elo_tracker,
            hier_engine=hier_engine,
            pace_tracker=pace_tracker,
            odds_dict=odds_dict,
            team_xppp_tracker=team_xppp_tracker,
            team_form_tracker=team_form_tracker,
            rotation_tracker=rotation_tracker,
            lineup_elo_tracker=lineup_elo_tracker,
            chemistry_tracker=chemistry_tracker,
            team_elo_tracker=team_elo_tracker,
            travel_tracker=travel_tracker,
            epm_tracker=epm_tracker,
            ref_tracker=ref_tracker,
            shot_quality_tracker=shot_quality_tracker,
            hapm_tracker=hapm_tracker,
            rapm_tracker=rapm_tracker,
            epva_tracker=epva_tracker,
            hierarchical_pace=hierarchical_pace,
            hier_shot_rates=hier_shot_rates,
            minutes_model=minutes_model,
            last_date=last_date,
            team_game_dates=team_game_dates,
            team_recent_net=team_recent_net,
            opponent_history=opponent_history,
            team_home_margin=team_home_margin,
            team_road_margin=team_road_margin,
            team_games_played=team_games_played,
            team_rosters_seen=team_rosters_seen,
            season_start_date=season_start_date,
            inactive_ids=inactive_ids,
            elo_calibrator=elo_calibrator,
            league_rolling_stats=league_rolling_stats,
        )
        market_spread = feat.get("market_spread", np.nan)
        market_ml = feat.get("market_ml", np.nan)
        market_total = feat.get("market_total", np.nan)
        closing_spread = feat.get("closing_spread", market_spread)
        decision_spread = feat.get("decision_spread", market_spread)
        spread_move = feat.get("spread_move", 0.0)
        public_home_pct = feat.get("public_home_pct", 0.5)

        # Prediction (with interaction features for train/serve parity)
        feat_model = build_feature_row(feat)
        if hasattr(meta_model, 'fitted') and meta_model.fitted:
            preds = meta_model.predict(feat_model)
        else:
            preds = {"pred_home": 110, "pred_away": 110, "pred_margin": 0.0,
                     "pred_total": 220, "win_prob": 0.5}

        raw_pred_margin = preds["pred_margin"]

        # Spread bias correction (legacy margin head; superseded when pair is canonical)
        if spread_calibrator is not None:
            corrected_spread = spread_calibrator.correct(raw_pred_margin)
        else:
            corrected_spread = raw_pred_margin

        from pipeline.config import USE_CANONICAL_SCORE_PAIR
        used_canonical_pair = False
        if (
            score_pair_model is not None
            and getattr(score_pair_model, "fitted", False)
            and USE_CANONICAL_SCORE_PAIR
        ):
            from pipeline.canonical_scores import apply_canonical_score_pair, prefer_canonical_margin
            preds = apply_canonical_score_pair(
                preds, feat_model, score_pair_model,
                legacy_margin=corrected_spread,
                cover_calibrator=cover_prob_calibrator,
            )
            corrected_spread = prefer_canonical_margin(preds, corrected_spread)
            used_canonical_pair = True
        elif (
            score_pair_model is not None
            and getattr(score_pair_model, "fitted", False)
        ):
            from pipeline.config import USE_SCORE_PAIR_TOTAL
            pair_out = score_pair_model.predict_scores(feat_model)
            preds["pred_home"] = pair_out["pred_home"]
            preds["pred_away"] = pair_out["pred_away"]
            preds["sigma_total"] = pair_out.get("sigma_total")
            preds["p_over"] = pair_out.get("p_over")
            if USE_SCORE_PAIR_TOTAL:
                preds["pred_total"] = pair_out["pred_total"]
        elif (
            total_model is not None
            and getattr(total_model, "fitted", False)
        ):
            total_out = total_model._predict_raw_total(feat_model)
            preds["pred_total"] = total_out["pred_total"]
            if "total_quantile_width" in total_out:
                preds["total_quantile_width"] = total_out["total_quantile_width"]
                preds["sigma_total"] = float(total_out["total_quantile_width"]) / 2.0
            if "total_q10" in total_out:
                preds["total_q10"] = total_out["total_q10"]
                preds["total_q90"] = total_out["total_q90"]
            preds["pred_home"] = (preds["pred_total"] + corrected_spread) / 2.0
            preds["pred_away"] = (preds["pred_total"] - corrected_spread) / 2.0
        else:
            from pipeline.config import OU_DEFAULT_SIGMA
            preds.setdefault("sigma_total", float(OU_DEFAULT_SIGMA))

        if (
            not used_canonical_pair
            and USE_HYBRID_STRUCTURED_BLEND
            and feat.get("struct_total") is not None
        ):
            try:
                from pipeline.structured_score import StructuredScoreForecast, hybrid_blend
                structured = StructuredScoreForecast(
                    home_pts=float(feat.get("struct_home_pts", 110)),
                    away_pts=float(feat.get("struct_away_pts", 110)),
                    total=float(feat["struct_total"]),
                    margin=float(feat.get("struct_margin", 0)),
                    sigma_total=float(feat.get("struct_sigma_total", preds.get("sigma_total", 12))),
                    sigma_margin=float(feat.get("struct_sigma_margin", 12)),
                    corr=float(feat.get("struct_score_corr", 0.35)),
                )
                hyb = hybrid_blend(
                    float(preds.get("pred_total", structured.total)),
                    float(corrected_spread),
                    structured,
                    total_beta=float(HYBRID_TOTAL_BETA),
                    margin_beta=float(HYBRID_MARGIN_BETA),
                )
                preds["pred_total"] = hyb["hybrid_total"]
                corrected_spread = hyb["hybrid_margin"]
                preds["pred_margin"] = corrected_spread
                preds["sigma_total"] = hyb["hybrid_sigma_total"]
                preds["hybrid_total"] = hyb["hybrid_total"]
                preds["hybrid_margin"] = hyb["hybrid_margin"]
            except Exception:
                pass

        # Keep score algebra consistent after any non-canonical adjustments.
        if used_canonical_pair:
            preds["pred_margin"] = float(preds["pred_home"]) - float(preds["pred_away"])
            preds["pred_total"] = float(preds["pred_home"]) + float(preds["pred_away"])
            corrected_spread = float(preds["pred_margin"])

        margin_for_win = corrected_spread
        if hasattr(meta_model, 'fitted') and meta_model.fitted:
            use_meta_win = (
                win_model is not None and getattr(win_model, "fitted", False)
                and (
                    WIN_PROB_SOURCE == "meta_win"
                    or (WIN_PROB_SOURCE == "auto" and REQUIRE_META_WIN_WHEN_FITTED)
                )
            )
            use_rolling = (
                calibrator is not None
                and (
                    WIN_PROB_SOURCE == "rolling_platt"
                    or (
                        WIN_PROB_SOURCE == "auto"
                        and not use_meta_win
                        and should_use_rolling_platt_when_win_model()
                    )
                )
            )
            if use_meta_win:
                raw_wp = win_model.predict_proba(
                    feat_model, pred_margin=margin_for_win,
                )
                preds["win_prob_raw"] = raw_wp
                if (
                    ml_calibrator is not None
                    and getattr(ml_calibrator, "_fitted", False)
                ):
                    preds["win_prob"] = ml_calibrator.transform(
                        raw_wp,
                        market_ml=market_ml,
                        market_spread=market_spread,
                    )
                else:
                    preds["win_prob"] = raw_wp
            elif use_rolling:
                try:
                    preds["win_prob"] = calibrator.predict(
                        margin_for_win, total=preds.get("pred_total"),
                    )
                except TypeError:
                    preds["win_prob"] = calibrator.predict(margin_for_win)

        # Keep predicted scores consistent with the bias-corrected spread and the
        # predicted total: home - away == corrected_spread, home + away == total.
        pred_total = preds.get("pred_total", preds.get("pred_home", 0) + preds.get("pred_away", 0))
        cons_home = (pred_total + corrected_spread) / 2.0
        cons_away = (pred_total - corrected_spread) / 2.0
        preds["pred_home"] = cons_home
        preds["pred_away"] = cons_away

        # Conformal interval: prefer joint score-pair intervals when canonical.
        if used_canonical_pair and preds.get("CONF_LOWER") is not None:
            conf_lower = float(preds["CONF_LOWER"])
            conf_upper = float(preds["CONF_UPPER"])
            conf_width = float(preds.get("CONF_WIDTH", conf_upper - conf_lower))
        elif conformal is not None:
            conf_lower, conf_upper, conf_width = conformal.predict_interval(corrected_spread, alpha=0.10)
        elif spread_calibrator is not None and hasattr(spread_calibrator, "predict_interval"):
            conf_lower, conf_upper, conf_width = spread_calibrator.predict_interval(corrected_spread, alpha=0.10)
        else:
            conf_lower, conf_upper, conf_width = corrected_spread - 12.0, corrected_spread + 12.0, 24.0

        q_width = preds.get("spread_quantile_width")
        if q_width is not None and np.isfinite(q_width):
            conf_width = float(q_width)
            if preds.get("spread_q10") is not None and preds.get("spread_q90") is not None:
                conf_lower = float(preds["spread_q10"])
                conf_upper = float(preds["spread_q90"])

        effective_edge_thr = variance_aware_edge_threshold(min_edge_pts, conf_width)
        disagreement_info = {}
        if disagreement_model is not None and getattr(disagreement_model, "fitted", False):
            disagreement_info = disagreement_model.predict(
                elo_margin_calibrated=feat.get("elo_margin_calibrated"),
                model_spread=corrected_spread,
                closing_spread=closing_spread,
                market_spread=market_spread,
                uncertainty=float(feat.get("h_rating_uncertainty", 350) + feat.get("a_rating_uncertainty", 350)),
                spread_move=spread_move,
                h_star_out=float(feat.get("h_star_out", 0) or 0),
                a_star_out=float(feat.get("a_star_out", 0) or 0),
            )
            effective_edge_thr += float(disagreement_info.get("edge_bump", 0.0))

        ho_off = feat.get("h_elo_off", 1500)
        ho_def = feat.get("h_elo_def", 1500)
        ao_off = feat.get("a_elo_off", 1500)
        ao_def = feat.get("a_elo_def", 1500)
        h_o_rd = feat.get("h_rating_uncertainty", 350) / 2
        h_d_rd = h_o_rd
        a_o_rd = feat.get("a_rating_uncertainty", 350) / 2
        a_d_rd = a_o_rd
        pred_poss = feat.get("exp_poss", 100)

        # Rotation-scenario mixture variance (prior-only projected minutes).
        if minutes_model is not None and USE_ROTATION_SCENARIOS:
            try:
                from pipeline.rotation_scenarios import RotationScenarioGenerator
                from pipeline.scenario_mixer import predict_and_mix_scenarios
                roster_h = list(home_lineup) if home_lineup else list(home_starters)
                roster_a = list(away_lineup) if away_lineup else list(away_starters)
                gen = RotationScenarioGenerator(
                    minutes_model,
                    factor_probs=(
                        minutes_model.factor_probs()
                        if hasattr(minutes_model, "factor_probs") else None
                    ),
                )
                # Branch on home roster uncertainty; keep away fixed for tractability.
                scenarios = gen.generate(roster_h) if roster_h else []
                if scenarios:
                    base_margin = float(corrected_spread)
                    base_var = max(float(feat.get("pace_var", 9.0)) * 0.25, 4.0)

                    def _pred(sc):
                        # Soft penalty when replacement minutes dominate.
                        repl = float(sc.minutes.get("_replacement", 0.0))
                        pen = 0.02 * repl
                        return base_margin - pen, base_var

                    mixed = predict_and_mix_scenarios(scenarios, _pred)
                    feat["scenario_mix_sigma"] = mixed.std
                    feat["scenario_mix_mean"] = mixed.mean
                    feat["scenario_n"] = float(len(scenarios))
                    # Nudge corrected spread toward mixture mean lightly.
                    corrected_spread = 0.85 * corrected_spread + 0.15 * mixed.mean
            except Exception:
                pass

        # --- Proper Kelly Fraction (moneyline) ---
        kelly_fraction = 0.0
        market_ml_away = feat.get("market_ml_away", np.nan)
        if not pd.isna(market_ml) and "win_prob" in preds:
            from pipeline.market import ml_prob_for_side, select_ml_bet

            side, ev_side, dec = select_ml_bet(
                preds["win_prob"], market_ml, min_ev=ML_MIN_EV,
                max_favorite_decimal=max_favorite_decimal,
                min_win_pct=min_ml_win_pct,
                market_ml_away=market_ml_away,
                market_spread=market_spread,
            )
            if side != "Pass":
                p = ml_prob_for_side(
                    preds["win_prob"], market_ml, side,
                    market_ml_away=market_ml_away,
                    market_spread=market_spread,
                )
                b = dec - 1
                kelly = (p * b - (1 - p)) / b
            else:
                kelly = 0.0

            # Fractional Kelly (25%) and uncertainty penalty
            FRACTIONAL_K = 0.25
            kelly = max(0.0, min(1.0, kelly * FRACTIONAL_K))

            rating_uncertainty = (h_o_rd + h_d_rd + a_o_rd + a_d_rd) / 4.0
            uncertainty_penalty = max(0.0, 1.0 - (rating_uncertainty - 150) / 400)
            kelly_fraction = kelly * uncertainty_penalty
        else:
            kelly_fraction = 0.0

        # Betting analysis
        rating_uncertainty = float(feat.get("h_rating_uncertainty", 350) + feat.get("a_rating_uncertainty", 350))

        matchup_vol_sigma = (
            float(vol_tracker.matchup_sigma(home_team, away_team))
            if vol_tracker is not None else np.nan
        )

        draft_direction = (
            "Home" if corrected_spread + float(market_spread or 0) > 0 else "Away"
        ) if not pd.isna(market_spread) else "Home"
        ats_prob_preview = None
        home_cover_ats = None
        if (
            ats_classifier is not None and getattr(ats_classifier, "fitted", False)
            and not pd.isna(market_spread)
        ):
            from pipeline.ats_ev import cover_prob_for_side
            ats_feat = {**feat, "pred_margin": corrected_spread, "market_spread": market_spread}
            home_cover_ats = ats_classifier.predict_home_cover_prob(ats_feat, market_spread)
            ats_prob_preview = cover_prob_for_side(home_cover_ats, draft_direction)

        upset_prob = None
        if (
            upset_classifier is not None and getattr(upset_classifier, "fitted", False)
            and not pd.isna(market_spread)
        ):
            upset_feat = {
                **feat,
                "PRED_SPREAD": corrected_spread,
                "pred_margin": corrected_spread,
                "MARKET_SPREAD": market_spread,
                "MATCHUP_VOL_SIGMA": matchup_vol_sigma,
                "WIN_PROB": preds.get("win_prob", 0.5),
                "SPREAD_QUANTILE_WIDTH": conf_width,
                "ELO_META_AGREEMENT": elo_meta_agreement(
                    corrected_spread, market_spread, feat.get("elo_margin_calibrated"),
                ) if not pd.isna(market_spread) else 1.0,
                "RATING_UNCERTAINTY": rating_uncertainty,
                "DISAGREEMENT_TRUST": disagreement_info.get("disagreement_trust", 1.0),
                "H_STAR_OUT": feat.get("h_star_out", 0),
                "A_STAR_OUT": feat.get("a_star_out", 0),
            }
            upset_prob = upset_classifier.predict_upset_prob(upset_feat)

        blowout_probs = preds.get("blowout_probs") if isinstance(preds.get("blowout_probs"), dict) else None
        if blowout_probs is None and hasattr(meta_model, "predict_blowout_probs"):
            try:
                blowout_probs = meta_model.predict_blowout_probs(feat_model)
            except Exception:
                blowout_probs = None

        scenario_mix_sigma = None
        try:
            from pipeline.market import scenario_mix_sigma_proxy
            scenario_mix_sigma = scenario_mix_sigma_proxy(
                matchup_vol_sigma=matchup_vol_sigma,
                conf_width=conf_width,
                h_missing_rotation=feat.get("h_missing_rotation"),
                a_missing_rotation=feat.get("a_missing_rotation"),
                h_star_out=feat.get("h_star_out"),
                a_star_out=feat.get("a_star_out"),
                explicit_sigma=feat.get("scenario_mix_sigma"),
            )
        except Exception:
            scenario_mix_sigma = None

        ba = bet_analysis(
            model_spread=corrected_spread,
            market_spread=market_spread,
            model_win_prob=preds.get("win_prob"),
            market_ml=market_ml,
            min_edge_pts=analysis_min_edge_pts(effective_edge_thr),
            min_ev=ML_MIN_EV,
            conf_width=conf_width,
            spread_move=spread_move,
            public_home_pct=public_home_pct,
            require_rlm=require_rlm,
            # Single nested calibrator lives in simulate/predict — defer here.
            confidence_calibrator=None,
            defer_confidence_calib=True,
            rating_uncertainty=rating_uncertainty,
            max_favorite_decimal=max_favorite_decimal,
            require_elo_agreement=require_elo_agreement,
            elo_margin_calibrated=feat.get("elo_margin_calibrated"),
            elo_agreement_extra_edge=elo_agreement_extra_edge,
            disagreement_trust=disagreement_info.get("disagreement_trust", 1.0),
            phantom_injury_flag=disagreement_info.get("phantom_injury_flag", False),
            pred_home=cons_home,
            pred_away=cons_away,
            matchup_vol_sigma=matchup_vol_sigma,
            ats_classifier_prob=ats_prob_preview,
            scenario_mix_sigma=scenario_mix_sigma,
            upset_prob=upset_prob,
            blowout_probs=blowout_probs,
            market_ml_away=market_ml_away,
        )
        edge_pts = float(ba.get("spread_edge_pts", 0.0) or 0.0)
        if not np.isfinite(edge_pts):
            edge_pts = 0.0
        model_lean = ba.get("spread_direction", "Pass")
        if model_lean == "Pass":
            model_lean = spread_lean_from_edge(edge_pts)
        direction = model_lean
        conf_score = ba.get("spread_confidence", 0)
        conf_raw = ba.get("confidence_score_raw", np.nan)
        stars = ba.get("spread_stars", "No market")
        ml_ev = ba.get("ml_ev", np.nan)
        ml_dir = ba.get("ml_direction", "Pass")

        if direction != "Pass":
            if (edge_pts > 0 and direction != "Home") or (edge_pts < 0 and direction != "Away"):
                direction = spread_lean_from_edge(edge_pts)

        direction = apply_bet_selection_gates(
            direction, edge_pts, conf_width, elo_meta_agreement=elo_meta_agreement(
                corrected_spread, market_spread, feat.get("elo_margin_calibrated"),
            ),
        )

        cover_prob = ba.get("cover_prob_calibrated", 0.524)
        raw_cover = ba.get("spread_cover_prob_raw", cover_prob)
        ats_prob = None
        lean_for_conf = model_lean if model_lean != "Pass" else spread_lean_from_edge(edge_pts)
        if (
            ats_classifier is not None and getattr(ats_classifier, "fitted", False)
            and lean_for_conf != "Pass" and not pd.isna(market_spread)
        ):
            from pipeline.ats_ev import cover_prob_for_side
            ats_feat = {**feat, "pred_margin": corrected_spread, "market_spread": market_spread}
            home_cover_ats = ats_classifier.predict_home_cover_prob(ats_feat, market_spread)
            ats_prob = cover_prob_for_side(home_cover_ats, lean_for_conf)
            if BET_SELECTION_MODE == "edge_bucket_ats" and ats_prob < ATS_CLASSIFIER_MIN_PROB:
                direction = "Pass"
        blended_cover = raw_cover
        if ats_prob is not None:
            blended_cover = blend_ats_classifier_cover(raw_cover, ats_prob, ATS_CLASSIFIER_BLEND)
        cover_prob = select_cover_prob(
            None,
            raw_cover=raw_cover,
            calibrated_cover=blended_cover,
            ats_classifier_prob=ats_prob,
        )
        # Prefer joint score-pair cover when canonical (calibrated on OOF residuals).
        from pipeline.config import USE_CANONICAL_SCORE_PAIR
        if (
            USE_CANONICAL_SCORE_PAIR
            and preds.get("forecast_source") == "score_pair"
            and preds.get("home_cover_prob") is not None
            and np.isfinite(float(preds.get("home_cover_prob")))
        ):
            from pipeline.ats_ev import cover_prob_for_side
            home_cover_pair = float(preds["home_cover_prob"])
            home_cover_ats = home_cover_pair
            if lean_for_conf in ("Home", "Away"):
                cover_prob = cover_prob_for_side(home_cover_pair, lean_for_conf)
            feat["home_cover_prob"] = home_cover_pair
            feat["cover_prob_source"] = preds.get("cover_prob_source", "pair_margin_gaussian")
            raw_cover = float(preds.get("home_cover_prob_raw", home_cover_pair))
        # Margin-consistent win prob for ATS confidence (avoid MetaWin sign conflicts).
        margin_wp = float(1.0 / (1.0 + np.exp(-float(corrected_spread) / 12.0)))
        margin_wp = float(np.clip(margin_wp, 0.01, 0.99))
        if home_cover_ats is not None and feat.get("home_cover_prob") is None:
            feat["home_cover_prob"] = float(home_cover_ats)
        feat["side_cover_prob_raw"] = float(raw_cover) if raw_cover is not None else np.nan
        if "cover_prob_source" not in feat:
            feat["cover_prob_source"] = "margin_gaussian"

        if target_family_calibrator is not None and USE_TARGET_FAMILY_CALIBRATION:
            try:
                family_cover_raw = float(cover_prob)
                cover_prob = target_family_calibrator.get("spread_cover").predict(
                    family_cover_raw, team=home_team,
                )
                if "win_prob" in preds:
                    preds["win_prob_family_raw"] = float(preds["win_prob"])
                    preds["win_prob"] = target_family_calibrator.get("moneyline").predict(
                        float(preds["win_prob"]), team=home_team,
                    )
                if preds.get("p_over") is not None:
                    preds["p_over_family_raw"] = float(preds["p_over"])
                    preds["p_over"] = target_family_calibrator.get("total_over").predict(
                        float(preds["p_over"]), team=home_team,
                    )
                feat["family_cover_raw"] = family_cover_raw
            except Exception:
                feat["family_cover_raw"] = float(cover_prob)

        if (
            lean_for_conf != "Pass"
            and confidence_calibrator is not None
            and should_use_bet_calibrator()
        ):
            conf_feats = build_confidence_features(
                spread_edge_pts=edge_pts,
                conf_width=conf_width,
                rating_uncertainty=rating_uncertainty,
                elo_meta_agreement=elo_meta_agreement(
                    corrected_spread, market_spread, feat.get("elo_margin_calibrated"),
                ),
                spread_cover_prob=blended_cover,
                matchup_vol_sigma=matchup_vol_sigma,
                disagreement_trust=disagreement_info.get("disagreement_trust", 1.0),
                phantom_injury_flag=disagreement_info.get("phantom_injury_flag", False),
                ats_classifier_prob=ats_prob if ats_prob is not None else ats_prob_preview,
                win_prob=margin_wp,
                elo_margin=float(feat.get("elo_margin_calibrated", 0) or 0),
                lean=lean_for_conf,
            )
            conf_feats["upset_prob"] = upset_prob
            conf_feats["market_spread"] = market_spread
            cal = confidence_calibrator.predict("ats", conf_feats, direction=lean_for_conf)
            conf_score = int(round(cal["confidence_score"]))
            conf_raw = cal["confidence_score_raw"]
            from pipeline.config import COVER_PROB_CAP_HI, COVER_PROB_CAP_LO
            cover_prob = float(np.clip(
                cal["calibrated_prob"],
                float(COVER_PROB_CAP_LO),
                float(COVER_PROB_CAP_HI),
            ))
            feat["cover_prob_source"] = "bet_calibrator"
            conf_tier = cal["confidence_tier"]
            stars = cal["stars"]
        else:
            conf_tier = ba.get("confidence_tier", 2)

        conf_thr = (
            float(min_confidence_threshold)
            if min_confidence_threshold is not None
            else float(MIN_CONFIDENCE_SCORE)
        )
        conf_max = (
            float(max_confidence_threshold)
            if max_confidence_threshold is not None
            else MAX_CONFIDENCE_SCORE
        )
        elo_agree = elo_meta_agreement(
            corrected_spread, market_spread, feat.get("elo_margin_calibrated"),
        )
        # Edge-bucket primary + confidence secondary (same helper for all modes).
        actionable = passes_confidence_actionable_gates(
            lean=lean_for_conf,
            edge_pts=edge_pts,
            conf_score=conf_score,
            conf_width=conf_width,
            min_confidence=conf_thr,
            max_confidence=conf_max,
            disagreement_trust=disagreement_info.get("disagreement_trust", 1.0),
            phantom_injury_flag=disagreement_info.get("phantom_injury_flag", False),
            market_spread=market_spread,
            elo_meta_agreement=elo_agree,
        )
        if lean_for_conf == "Pass" or (
            uses_edge_gates() and direction == "Pass"
        ):
            actionable = False

        from pipeline.config import REQUIRE_NONNEGATIVE_CLV, MIN_CLV_POINTS
        from pipeline.metrics import compute_clv

        clv_pre = np.nan
        if pd.notna(market_spread) and pd.notna(closing_spread):
            clv_pre = compute_clv(corrected_spread, market_spread, closing_spread)
        if (
            REQUIRE_NONNEGATIVE_CLV
            and actionable
            and pd.notna(clv_pre)
            and float(clv_pre) < float(MIN_CLV_POINTS)
        ):
            actionable = False

        if CONFIDENCE_SELECTION_MODE == "min_score" and lean_for_conf != "Pass" and not actionable:
            direction = "Pass"
        elif lean_for_conf != "Pass" and actionable:
            direction = lean_for_conf
        elif lean_for_conf != "Pass" and not actionable:
            direction = "Pass"

        from pipeline.config import VENN_ABERS_MAX_WIDTH
        va_ml_interval = None
        if (
            use_venn_abers_filter
            and confidence_calibrator is not None
            and (
                hasattr(confidence_calibrator, "venn_abers_interval")
                or hasattr(confidence_calibrator, "venn_abers_width")
            )
        ):
            va_score = conf_score
            if isinstance(va_score, (int, float)):
                va_w = None
                if hasattr(confidence_calibrator, "venn_abers_interval"):
                    va_iv = confidence_calibrator.venn_abers_interval(float(va_score))
                    if va_iv is not None:
                        va_ml_interval = (va_iv[0], va_iv[1])
                        va_w = va_iv[2]
                elif hasattr(confidence_calibrator, "venn_abers_width"):
                    va_w = confidence_calibrator.venn_abers_width(float(va_score))
                if (
                    direction != "Pass"
                    and va_w is not None
                    and va_w > VENN_ABERS_MAX_WIDTH
                ):
                    direction = "Pass"

        def _gate_ml_va_interval(side: str) -> str:
            if (
                not use_venn_abers_filter
                or side == "Pass"
                or va_ml_interval is None
                or pd.isna(market_ml)
            ):
                return side
            p0, p1 = va_ml_interval
            fh, fa = fair_probs_from_ml_pair(market_ml, market_ml_away)
            p_mkt = fh if side == "Home" else fa
            if not market_implied_inside_interval(p0, p1, p_mkt):
                return "Pass"
            return side

        ml_dir = _gate_ml_va_interval(ml_dir)
        actionable = int(direction != "Pass")

        elo_meta_agreement_val = elo_meta_agreement(
            corrected_spread, market_spread, feat.get("elo_margin_calibrated"),
        )

        edge_mult = edge_stake_multiplier(abs(edge_pts)) if uses_edge_gates() else 1.0
        edge_bucket_min = edge_bucket_min_for_stakes()
        vol_mult = 1.0
        if STAKE_SIZING_MODE == "volatility_adjusted" and vol_tracker is not None:
            vol_mult = vol_tracker.stake_multiplier(home_team, away_team)

        from pipeline.config import BLOWOUT_FAV_STAKE_BOOST, USE_BLOWOUT_STAKE_ADJUST
        blowout_mult = 1.0
        if USE_BLOWOUT_STAKE_ADJUST and blowout_probs and direction != "Pass" and pd.notna(market_spread):
            p15 = float(blowout_probs.get(15, blowout_probs.get("15", 0.0)) or 0.0)
            lean_fav = (
                (direction == "Home" and float(market_spread) < 0)
                or (direction == "Away" and float(market_spread) > 0)
            )
            if lean_fav and p15 >= 0.30:
                blowout_mult = float(BLOWOUT_FAV_STAKE_BOOST)

        # Spread Kelly + profile stakes
        spread_kelly = spread_kelly_fraction(cover_prob)
        stakes = {}
        for profile in PROFILES:
            stakes[profile] = compute_stake(
                profile,
                direction=direction,
                edge_pts=edge_pts,
                edge_threshold=effective_edge_thr,
                cover_prob=cover_prob,
                confidence_tier=conf_tier,
                conf_width=conf_width,
                rating_uncertainty=rating_uncertainty,
                edge_stake_mult=edge_mult,
                edge_bucket_min=edge_bucket_min,
                volatility_mult=vol_mult * blowout_mult,
                confidence_score=conf_score,
            )

        ml_kelly = compute_ml_stake(
            "moderate",
            ml_direction=ml_dir,
            win_prob=preds.get("win_prob", 0.5),
            market_ml=market_ml,
            confidence_tier=ba.get("ml_confidence_tier", conf_tier),
            max_favorite_decimal=max_favorite_decimal,
        )

        pred_total_val = preds.get("pred_total", 0)
        total_edge = pred_total_val - float(market_total) if pd.notna(market_total) and market_total else 0.0
        ou_dir = "Pass"
        p_over = 0.5
        sigma_total = preds.get("sigma_total")
        if sigma_total is None or not np.isfinite(float(sigma_total or np.nan)):
            t_width = preds.get("total_quantile_width")
            if t_width is not None and np.isfinite(float(t_width)):
                sigma_total = float(t_width) / 2.0
            else:
                from pipeline.config import OU_DEFAULT_SIGMA
                sigma_total = float(OU_DEFAULT_SIGMA)
        if ou_bet:
            from pipeline.config import (
                OU_BET_ENABLED,
                OU_REQUIRE_POSITIVE_CI,
                OU_CALIBRATOR_MIN_PROB,
            )
            from pipeline.market import select_ou_bet
            if OU_BET_ENABLED and pd.notna(market_total) and market_total > 0:
                ou_dir, p_over, total_edge = select_ou_bet(
                    pred_total_val,
                    market_total,
                    sigma_total,
                    min_edge=ou_min_edge,
                )
                if OU_REQUIRE_POSITIVE_CI and ou_dir != "Pass":
                    t_width = preds.get("total_quantile_width")
                    if t_width is not None and np.isfinite(float(t_width)) and float(t_width) > 18.0:
                        ou_dir = "Pass"
                if (
                    ou_dir != "Pass"
                    and confidence_calibrator is not None
                    and getattr(confidence_calibrator, "_fitted", False)
                    and confidence_calibrator.models.get("ou") is not None
                ):
                    ou_feat = {
                        "total_edge": total_edge,
                        "PRED_TOTAL": pred_total_val,
                        "MARKET_TOTAL": float(market_total),
                        "OU_DIRECTION": ou_dir,
                        "WIN_PROB": preds.get("win_prob", 0.5),
                        "CONF_WIDTH": conf_width,
                        "RATING_UNCERTAINTY": rating_uncertainty,
                        "P_OVER": p_over,
                    }
                    ou_cal = confidence_calibrator.predict("ou", ou_feat, direction=ou_dir)
                    if ou_cal.get("calibrated_prob", 0.5) < OU_CALIBRATOR_MIN_PROB:
                        ou_dir = "Pass"
        ou_win = np.nan
        if ou_dir != "Pass" and pd.notna(market_total) and market_total > 0:
            actual_total = act_h + act_a
            ou_win = int((ou_dir == "Over" and actual_total > market_total) or
                         (ou_dir == "Under" and actual_total < market_total))

        from pipeline.market import explain_ml_bet
        ml_explain = explain_ml_bet(
            preds.get("win_prob", 0.5), market_ml,
            market_ml_away=market_ml_away,
            market_spread=market_spread,
        )
        # Prefer explained side when EV mode selected bet; keep ba ml_dir if already set
        if ml_dir == "Pass" and ml_explain.get("ml_bet") != "Pass":
            ml_dir = ml_explain["ml_bet"]
            ml_ev = ml_explain.get("ml_ev", ml_ev)
        ml_dir = _gate_ml_va_interval(ml_dir)
        if ml_dir == "Pass" and isinstance(ml_explain, dict):
            ml_explain["ml_bet"] = "Pass"

        from pipeline.shared_forecast import emit_shared_forecast_from_feature_row
        shared_fc = emit_shared_forecast_from_feature_row(feat, preds)

        from pipeline.ats_ev import ats_decision_record, canonical_home_cover_prob
        # Cover_prob above is P(chosen lean covers) — never pass it as home_cover.
        p_home_cover = canonical_home_cover_prob(
            ats_classifier=ats_classifier,
            feat={**feat, "pred_margin": corrected_spread},
            decision_spread=decision_spread if pd.notna(decision_spread) else market_spread,
            pred_margin=corrected_spread,
            sigma=float(preds["sigma_margin"]) if preds.get("sigma_margin") is not None
            else max(float(conf_width or 24.0) / 2.5, 4.0),
            direction_specific_cover=float(cover_prob) if cover_prob is not None else None,
            lean=lean_for_conf if lean_for_conf in ("Home", "Away") else None,
        )
        # Prefer pair-calibrated home cover when present.
        if preds.get("home_cover_prob") is not None and np.isfinite(float(preds.get("home_cover_prob"))):
            p_home_cover = float(preds["home_cover_prob"])
        ats_ev_rec = ats_decision_record(
            p_home_cover,
            {
                **feat,
                "decision_spread": decision_spread if pd.notna(decision_spread) else market_spread,
                "market_spread": market_spread,
            },
        )

        row_out = {
            "GAME_ID": game_id,
            "DATE": gdate.date() if is_valid_timestamp(gdate) else None,
            "HOME": home_team, "AWAY": away_team,
            "PRED_HOME": round(preds.get("pred_home", 0), 1),
            "PRED_AWAY": round(preds.get("pred_away", 0), 1),
            "RAW_PRED_HOME": round(float(preds.get("raw_pred_home", preds.get("pred_home", 0))), 1),
            "RAW_PRED_AWAY": round(float(preds.get("raw_pred_away", preds.get("pred_away", 0))), 1),
            "PRED_SPREAD": round(corrected_spread, 2),
            "RAW_PRED_MARGIN": round(raw_pred_margin, 2),
            "PRED_TOTAL": round(preds.get("pred_total", 0), 2),
            "FORECAST_SOURCE": preds.get("forecast_source", "meta_margin"),
            "SIGMA_HOME": round(float(preds["sigma_home"]), 2) if preds.get("sigma_home") is not None else np.nan,
            "SIGMA_AWAY": round(float(preds["sigma_away"]), 2) if preds.get("sigma_away") is not None else np.nan,
            "SIGMA_MARGIN": round(float(preds["sigma_margin"]), 2) if preds.get("sigma_margin") is not None else np.nan,
            "SCORE_RESIDUAL_CORR": round(float(preds["score_residual_corr"]), 3) if preds.get("score_residual_corr") is not None else np.nan,
            "WIN_PROB": round(preds.get("win_prob", 0), 3),
            "WIN_PROB_RAW": round(preds.get("win_prob_raw", preds.get("win_prob", 0)), 3),
            "WIN_PROB_MARGIN": round(float(preds["win_prob_margin"]), 3) if preds.get("win_prob_margin") is not None else np.nan,
            "ACTUAL_HOME": act_h, "ACTUAL_AWAY": act_a,
            "ACTUAL_MARGIN": act_h - act_a,
            "MARKET_SPREAD": market_spread,
            "DECISION_SPREAD": decision_spread,
            "CLOSING_SPREAD": closing_spread,
            "MARKET_ML": market_ml,
            "MARKET_ML_AWAY": market_ml_away,
            "MARKET_TOTAL": market_total,
            "EDGE": round(edge_pts, 2),
            "EDGE_LEAN": model_lean,
            "WIN_PCT": conf_score,
            "ACTIONABLE": int(actionable),
            "DIRECTION": direction,
            "CONFIDENCE": conf_score,
            "CONFIDENCE_RAW": round(float(conf_raw), 3) if pd.notna(conf_raw) else np.nan,
            "MATCHUP_VOL_SIGMA": round(float(matchup_vol_sigma), 3) if pd.notna(matchup_vol_sigma) else np.nan,
            "CALIBRATED_COVER_PROB": round(cover_prob, 3),
            "STARS": stars,
            "ML_EV": ml_ev,
            "ML_DIRECTION": ml_dir,
            "OU_DIRECTION": ou_dir,
            "OU_WIN": ou_win,
            "P_OVER": round(float(p_over), 3),
            "SIGMA_TOTAL": round(float(sigma_total), 2) if sigma_total is not None else np.nan,
            "PREDICTED_WINNER": ml_explain.get("predicted_winner", "Pass"),
            "EV_HOME": round(float(ml_explain["ev_home"]), 4) if pd.notna(ml_explain.get("ev_home")) else np.nan,
            "EV_AWAY": round(float(ml_explain["ev_away"]), 4) if pd.notna(ml_explain.get("ev_away")) else np.nan,
            "ML_BET": ml_explain.get("ml_bet", ml_dir),
            "ML_REASON": ml_explain.get("ml_reason", ""),
            "P_HOME": round(float(ml_explain.get("p_home", preds.get("win_prob", 0.5))), 3),
            "P_AWAY": round(float(ml_explain.get("p_away", 1.0 - preds.get("win_prob", 0.5))), 3),
            "CONF_LOWER": round(conf_lower, 2),
            "CONF_UPPER": round(conf_upper, 2),
            "CONF_WIDTH": round(conf_width, 2),
            "MARGIN_QLO_50": round(float(preds["margin_qlo_50"]), 2) if preds.get("margin_qlo_50") is not None else np.nan,
            "MARGIN_QHI_50": round(float(preds["margin_qhi_50"]), 2) if preds.get("margin_qhi_50") is not None else np.nan,
            "MARGIN_QLO_80": round(float(preds["margin_qlo_80"]), 2) if preds.get("margin_qlo_80") is not None else np.nan,
            "MARGIN_QHI_80": round(float(preds["margin_qhi_80"]), 2) if preds.get("margin_qhi_80") is not None else np.nan,
            "MARGIN_QLO_95": round(float(preds["margin_qlo_95"]), 2) if preds.get("margin_qlo_95") is not None else np.nan,
            "MARGIN_QHI_95": round(float(preds["margin_qhi_95"]), 2) if preds.get("margin_qhi_95") is not None else np.nan,
            "HOME_QLO_80": round(float(preds["home_qlo_80"]), 2) if preds.get("home_qlo_80") is not None else np.nan,
            "HOME_QHI_80": round(float(preds["home_qhi_80"]), 2) if preds.get("home_qhi_80") is not None else np.nan,
            "AWAY_QLO_80": round(float(preds["away_qlo_80"]), 2) if preds.get("away_qlo_80") is not None else np.nan,
            "AWAY_QHI_80": round(float(preds["away_qhi_80"]), 2) if preds.get("away_qhi_80") is not None else np.nan,
            "TOTAL_QLO_80": round(float(preds["total_qlo_80"]), 2) if preds.get("total_qlo_80") is not None else np.nan,
            "TOTAL_QHI_80": round(float(preds["total_qhi_80"]), 2) if preds.get("total_qhi_80") is not None else np.nan,
            "SPREAD_Q10": round(float(preds["spread_q10"]), 2) if preds.get("spread_q10") is not None else np.nan,
            "SPREAD_Q90": round(float(preds["spread_q90"]), 2) if preds.get("spread_q90") is not None else np.nan,
            "SPREAD_Q25": round(float(preds["spread_q25"]), 2) if preds.get("spread_q25") is not None else np.nan,
            "SPREAD_Q75": round(float(preds["spread_q75"]), 2) if preds.get("spread_q75") is not None else np.nan,
            "KELLY_FRACTION": round(kelly_fraction, 4),
            "SPREAD_KELLY": round(spread_kelly, 4),
            "COVER_PROB_CALIBRATED": round(cover_prob, 3),
            "SPREAD_COVER_PROB_RAW": round(ba.get("spread_cover_prob_raw", cover_prob), 3),
            "HOME_COVER_PROB": round(float(home_cover_ats), 3) if home_cover_ats is not None else np.nan,
            "SIDE_COVER_PROB": round(float(cover_prob), 3),
            "SIDE_COVER_PROB_RAW": round(float(raw_cover), 3) if raw_cover is not None else np.nan,
            "COVER_PROB_SOURCE": feat.get("cover_prob_source", "margin_gaussian"),
            "CONFIDENCE_TIER": conf_tier,
            "ELO_META_AGREEMENT": elo_meta_agreement_val,
            "RATING_UNCERTAINTY": round(rating_uncertainty, 1),
            "TOTAL_EDGE": round(total_edge, 2),
            "STAKE_CONSERVATIVE": round(stakes.get("conservative", 0), 4),
            "STAKE_MODERATE": round(stakes.get("moderate", 0), 4),
            "STAKE_AGGRESSIVE": round(stakes.get("aggressive", 0), 4),
            "ML_KELLY": round(ml_kelly, 4),
            "EDGE_THRESHOLD": effective_edge_thr,
            "BASE_EDGE_THRESHOLD": min_edge_pts,
            "SPREAD_QUANTILE_WIDTH": round(float(q_width), 2) if q_width is not None and np.isfinite(q_width) else np.nan,
            "PHANTOM_INJURY_FLAG": int(disagreement_info.get("phantom_injury_flag", False)),
            "DISAGREEMENT_TRUST": round(float(disagreement_info.get("disagreement_trust", 1.0)), 3),
            "ELO_MARGIN_CALIBRATED": round(float(feat.get("elo_margin_calibrated", 0) or 0), 2),
            "H_STAR_OUT": feat.get("h_star_out", 0),
            "A_STAR_OUT": feat.get("a_star_out", 0),
            "ATS_CLASSIFIER_PROB": round(float(ats_prob), 3) if ats_prob is not None else np.nan,
            "ATS_EV_SIDE": ats_ev_rec.get("side", "Pass"),
            "ATS_EV": round(float(ats_ev_rec["expected_value"]), 4) if pd.notna(ats_ev_rec.get("expected_value")) else np.nan,
            "ATS_FAIR_COVER_PROB": round(float(ats_ev_rec["fair_cover_prob"]), 3) if pd.notna(ats_ev_rec.get("fair_cover_prob")) else np.nan,
            "ATS_PASS_REASON": ats_ev_rec.get("pass_reason"),
            "SF_FAIR_MARGIN": round(float(shared_fc.get("sf_fair_margin", np.nan)), 3) if pd.notna(shared_fc.get("sf_fair_margin")) else np.nan,
            "SF_FAIR_TOTAL": round(float(shared_fc.get("sf_fair_total", np.nan)), 3) if pd.notna(shared_fc.get("sf_fair_total")) else np.nan,
            "SF_HOME_WIN_PROB": round(float(shared_fc.get("sf_home_win_prob", np.nan)), 3) if pd.notna(shared_fc.get("sf_home_win_prob")) else np.nan,
            "UPSET_PROB": round(float(upset_prob), 3) if upset_prob is not None else np.nan,
            "P_BLOWOUT_10": round(float(blowout_probs.get(10, blowout_probs.get("10", np.nan))), 3) if blowout_probs else np.nan,
            "P_BLOWOUT_15": round(float(blowout_probs.get(15, blowout_probs.get("15", np.nan))), 3) if blowout_probs else np.nan,
            "P_BLOWOUT_20": round(float(blowout_probs.get(20, blowout_probs.get("20", np.nan))), 3) if blowout_probs else np.nan,
            "VOL_STAKE_MULT": round(vol_mult, 3),
            "MODEL_ML_CORRECT": int((preds.get("win_prob", 0.5) > 0.5) == (act_h - act_a > 0)),
            "TOTAL_ERR": round(preds.get("pred_total", 0) - (act_h + act_a), 2),
        }
        results.append(row_out)

        # --- Post‑game updates ---
        seas_prog = 0.5
        h_form = (
            team_form_tracker.get(home_team, current_season, gdate)
            if team_form_tracker is not None else None
        )
        a_form = (
            team_form_tracker.get(away_team, current_season, gdate)
            if team_form_tracker is not None else None
        )
        update_trackers_after_game(
            group=group,
            gs=gs,
            game_id=game_id,
            gdate=gdate,
            home=home_team,
            away=away_team,
            home_starters=actual_home_starters,
            away_starters=actual_away_starters,
            act_h=act_h,
            act_a=act_a,
            home_tot_poss=home_tot_poss,
            away_tot_poss=away_tot_poss,
            home_xppp_game=home_xppp_game,
            away_xppp_game=away_xppp_game,
            current_season=current_season,
            market_spread=market_spread,
            elo_tracker=elo_tracker,
            hier_engine=hier_engine,
            pace_tracker=pace_tracker,
            team_xppp_tracker=team_xppp_tracker,
            team_form_tracker=team_form_tracker,
            rotation_tracker=rotation_tracker,
            lineup_elo_tracker=lineup_elo_tracker,
            chemistry_tracker=chemistry_tracker,
            team_elo_tracker=team_elo_tracker,
            travel_tracker=travel_tracker,
            ref_tracker=ref_tracker,
            shot_quality_tracker=shot_quality_tracker,
            epva_tracker=epva_tracker,
            hierarchical_pace=hierarchical_pace,
            hier_shot_rates=hier_shot_rates,
            minutes_model=minutes_model,
            last_game_date=last_date,
            team_game_dates=team_game_dates,
            team_recent_net=team_recent_net,
            opponent_history=opponent_history,
            team_home_margin=team_home_margin,
            team_road_margin=team_road_margin,
            team_games_played=team_games_played,
            team_rosters_seen=team_rosters_seen,
            lineup_cache=lineup_cache,
            season_player_ids=season_player_ids,
            seas_prog=seas_prog,
            ho_off=ho_off,
            ho_def=ho_def,
            ao_off=ao_off,
            ao_def=ao_def,
            h_form=h_form,
            a_form=a_form,
        )

        if league_rolling_stats is not None:
            league_rolling_stats.update(
                pace_diff=feat.get("pace_diff"),
                elo_net=feat.get("elo_net"),
                roll_net_xppp=feat.get("roll_net_xppp"),
                elo_margin=feat.get("elo_margin"),
            )

        if calibrator is not None:
            try:
                calibrator.update(corrected_spread, 1 if act_h > act_a else 0,
                                  total=preds.get("pred_total"))
            except TypeError:
                calibrator.update(corrected_spread, 1 if act_h > act_a else 0)

        if spread_calibrator is not None:
            spread_calibrator.update(raw_pred_margin, act_h - act_a)

        if target_family_calibrator is not None and USE_TARGET_FAMILY_CALIBRATION:
            try:
                # Past-only updates after the game (nested / walk-forward).
                covered = int(
                    (act_h - act_a + float(market_spread if pd.notna(market_spread) else 0)) > 0
                ) if pd.notna(market_spread) else None
                if covered is not None:
                    raw_c = float(feat.get("family_cover_raw", cover_prob))
                    target_family_calibrator.get("spread_cover").update(
                        raw_c, covered, team=home_team,
                    )
                ml_y = int(act_h > act_a)
                ml_raw = float(preds.get("win_prob_family_raw", preds.get("win_prob", 0.5)))
                target_family_calibrator.get("moneyline").update(
                    ml_raw, ml_y, team=home_team,
                )
                if pd.notna(market_total) and preds.get("p_over") is not None:
                    went_over = int((act_h + act_a) > float(market_total))
                    ou_raw = float(preds.get("p_over_family_raw", preds["p_over"]))
                    target_family_calibrator.get("total_over").update(
                        ou_raw, went_over, team=home_team,
                    )
            except Exception:
                pass

        if vol_tracker is not None:
            vol_tracker.update_matchup(home_team, away_team, corrected_spread, act_h - act_a)

        resid_hist.append(abs(corrected_spread - (act_h - act_a)))

    if skipped_games:
        print(f"  ℹ️ simulation: skipped {skipped_games} games with unrecoverable dates")

    out = pd.DataFrame(results)
    if not out.empty:
        out["SPREAD_ERR"] = (out["PRED_SPREAD"] - out["ACTUAL_MARGIN"]).abs()
        out["TOTAL_ERR"] = (out["PRED_TOTAL"] - (out["ACTUAL_HOME"] + out["ACTUAL_AWAY"])).abs()
        out = add_clv_columns(out)
    return out