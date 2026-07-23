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
    spread_kelly_fraction,
)
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
    require_rlm: bool = False,
    ou_bet: bool = True,
    min_edge_pts: float = None,
    max_favorite_decimal: float = 1.45,
    ou_min_edge: float = 3.0,
    confidence_calibrator=None,
    win_model=None,
    total_model=None,
    score_pair_model=None,
    ats_classifier=None,
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
):
    """
    Walk‑forward simulation for a single season.
    """
    if "game_date" in season_df.columns:
        # Task 015: enforce chronological iteration explicitly by decision
        # timestamp (game_date) then GAME_ID, regardless of input row order.
        season_df = season_df.sort_values(["game_date", "GAME_ID", "stint_id"]).reset_index(drop=True)
        _per_game_dates = season_df.drop_duplicates("GAME_ID")["game_date"]
        assert _per_game_dates.is_monotonic_increasing, (
            "run_simulation: per-game decision timestamps are not monotonic "
            "nondecreasing after sort — chronological iteration is violated"
        )

    if min_edge_pts is None:
        min_edge_pts = edge_threshold if edge_threshold is not None else GOOD_BET_EDGE
    if elo_agreement_extra_edge is None:
        elo_agreement_extra_edge = ELO_AGREEMENT_EXTRA_EDGE
    from pipeline.config import MIN_ML_WIN_PCT
    if min_ml_win_pct is None:
        min_ml_win_pct = float(MIN_ML_WIN_PCT)

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
                hier_engine.offseason_revert()
                elo_tracker.offseason_revert(returning_player_ids=returning_ids)
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
        )
        market_spread = feat.get("market_spread", np.nan)
        market_ml = feat.get("market_ml", np.nan)
        market_total = feat.get("market_total", np.nan)
        closing_spread = feat.get("closing_spread", market_spread)
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

        # Spread bias correction (before win-prob so probabilities match betting spread)
        if spread_calibrator is not None:
            corrected_spread = spread_calibrator.correct(raw_pred_margin)
        else:
            corrected_spread = raw_pred_margin

        if (
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
                    preds["win_prob"] = ml_calibrator.transform(raw_wp)
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

        # Conformal interval from spread calibrator residuals (fallback ±12)
        if conformal is not None:
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

        # --- Proper Kelly Fraction (moneyline) ---
        kelly_fraction = 0.0
        if not pd.isna(market_ml) and "win_prob" in preds:
            from pipeline.market import ml_prob_for_side, select_ml_bet

            side, ev_side, dec = select_ml_bet(
                preds["win_prob"], market_ml, min_ev=ML_MIN_EV,
                max_favorite_decimal=max_favorite_decimal,
                min_win_pct=min_ml_win_pct,
            )
            if side != "Pass":
                p = ml_prob_for_side(preds["win_prob"], market_ml, side)
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
        if (
            ats_classifier is not None and getattr(ats_classifier, "fitted", False)
            and not pd.isna(market_spread)
        ):
            ats_feat = {**feat, "pred_margin": corrected_spread, "market_spread": market_spread}
            ats_prob_preview = ats_classifier.predict_cover_prob(
                ats_feat, draft_direction, market_spread,
            )

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
            confidence_calibrator=confidence_calibrator if should_use_bet_calibrator() else None,
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

        direction = apply_bet_selection_gates(direction, edge_pts, conf_width)

        cover_prob = ba.get("cover_prob_calibrated", 0.524)
        raw_cover = ba.get("spread_cover_prob_raw", cover_prob)
        ats_prob = None
        lean_for_conf = model_lean if model_lean != "Pass" else spread_lean_from_edge(edge_pts)
        if (
            ats_classifier is not None and getattr(ats_classifier, "fitted", False)
            and lean_for_conf != "Pass" and not pd.isna(market_spread)
        ):
            ats_feat = {**feat, "pred_margin": corrected_spread, "market_spread": market_spread}
            ats_prob = ats_classifier.predict_cover_prob(ats_feat, lean_for_conf, market_spread)
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
                win_prob=float(preds.get("win_prob", 0.5) or 0.5),
                elo_margin=float(feat.get("elo_margin_calibrated", 0) or 0),
                lean=lean_for_conf,
            )
            cal = confidence_calibrator.predict("ats", conf_feats, direction=lean_for_conf)
            conf_score = int(round(cal["confidence_score"]))
            conf_raw = cal["confidence_score_raw"]
            cover_prob = float(np.clip(cal["calibrated_prob"], 0.01, 0.99))
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
        if uses_edge_gates():
            actionable = lean_for_conf != "Pass" and direction != "Pass"
        else:
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

        from pipeline.config import USE_VENN_ABERS_FILTER, VENN_ABERS_MAX_WIDTH
        if (
            USE_VENN_ABERS_FILTER
            and direction != "Pass"
            and confidence_calibrator is not None
            and hasattr(confidence_calibrator, "venn_abers_width")
        ):
            va_score = conf_score
            if isinstance(va_score, (int, float)):
                va_w = confidence_calibrator.venn_abers_width(float(va_score))
                if va_w is not None and va_w > VENN_ABERS_MAX_WIDTH:
                    direction = "Pass"
        actionable = int(direction != "Pass")

        elo_meta_agreement_val = elo_meta_agreement(
            corrected_spread, market_spread, feat.get("elo_margin_calibrated"),
        )

        edge_mult = edge_stake_multiplier(abs(edge_pts)) if uses_edge_gates() else 1.0
        edge_bucket_min = edge_bucket_min_for_stakes()
        vol_mult = 1.0
        if STAKE_SIZING_MODE == "volatility_adjusted" and vol_tracker is not None:
            vol_mult = vol_tracker.stake_multiplier(home_team, away_team)

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
                volatility_mult=vol_mult,
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
        ml_explain = explain_ml_bet(preds.get("win_prob", 0.5), market_ml)
        # Prefer explained side when EV mode selected bet; keep ba ml_dir if already set
        if ml_dir == "Pass" and ml_explain.get("ml_bet") != "Pass":
            ml_dir = ml_explain["ml_bet"]
            ml_ev = ml_explain.get("ml_ev", ml_ev)

        row_out = {
            "GAME_ID": game_id,
            "DATE": gdate.date() if is_valid_timestamp(gdate) else None,
            "HOME": home_team, "AWAY": away_team,
            "PRED_HOME": round(preds.get("pred_home", 0), 1),
            "PRED_AWAY": round(preds.get("pred_away", 0), 1),
            "PRED_SPREAD": round(corrected_spread, 2),
            "RAW_PRED_MARGIN": round(raw_pred_margin, 2),
            "PRED_TOTAL": round(preds.get("pred_total", 0), 2),
            "WIN_PROB": round(preds.get("win_prob", 0), 3),
            "WIN_PROB_RAW": round(preds.get("win_prob_raw", preds.get("win_prob", 0)), 3),
            "ACTUAL_HOME": act_h, "ACTUAL_AWAY": act_a,
            "ACTUAL_MARGIN": act_h - act_a,
            "MARKET_SPREAD": market_spread,
            "CLOSING_SPREAD": closing_spread,
            "MARKET_ML": market_ml,
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
            "KELLY_FRACTION": round(kelly_fraction, 4),
            "SPREAD_KELLY": round(spread_kelly, 4),
            "COVER_PROB_CALIBRATED": round(cover_prob, 3),
            "SPREAD_COVER_PROB_RAW": round(ba.get("spread_cover_prob_raw", cover_prob), 3),
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

        if calibrator is not None:
            try:
                calibrator.update(corrected_spread, 1 if act_h > act_a else 0,
                                  total=preds.get("pred_total"))
            except TypeError:
                calibrator.update(corrected_spread, 1 if act_h > act_a else 0)

        if spread_calibrator is not None:
            spread_calibrator.update(raw_pred_margin, act_h - act_a)

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