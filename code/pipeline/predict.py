"""Single-game live prediction using shared build_game_features path."""
from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.config import TEAM_MAP, GOOD_BET_EDGE, STATE_DIR, ELO_BLEND_ALPHA, ELO_RIDGE_ALPHA
from pipeline.game_features import build_game_features
from pipeline.model import build_feature_row, MetaWinModel
from pipeline.market import bet_analysis, spread_edge, SpreadCalibrator
from pipeline.metrics import variance_aware_edge_threshold
from pipeline.bet_confidence import WalkForwardBetCalibrator, default_calibrator_path
from pipeline.market_disagreement import (
    WalkForwardMarketDisagreementModel,
    default_disagreement_path,
)
from pipeline.elo_calibration import (
    default_elo_calibrator_path,
    WalkForwardEloCalibrator,
    load_elo_knobs,
)
from pipeline.stake_profiles import compute_stake, PROFILES


class PredictionContext:
    """Mutable walk-forward state for live predictions (persist via save/load)."""

    def __init__(self):
        self.last_date = {}
        self.team_game_dates = defaultdict(lambda: deque(maxlen=20))
        self.team_recent_net = defaultdict(lambda: deque(maxlen=10))
        self.opponent_history = defaultdict(lambda: deque(maxlen=15))
        self.team_home_margin = defaultdict(lambda: deque(maxlen=20))
        self.team_road_margin = defaultdict(lambda: deque(maxlen=20))
        self.team_games_played = defaultdict(int)
        self.team_rosters_seen = defaultdict(set)
        self.season_start_date = None
        self.lineup_cache = {}

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        for k, v in data.items():
            if k.endswith("_history") or k in ("team_game_dates", "team_recent_net",
                                                "team_home_margin", "team_road_margin"):
                setattr(obj, k, defaultdict(lambda: deque(maxlen=20), {
                    tk: deque(tv) for tk, tv in v.items()
                }))
            else:
                setattr(obj, k, v)
        return obj


def load_tuning_config(path: Path | None = None) -> dict:
    from pipeline.config import OU_MIN_EDGE, ML_MAX_FAVORITE_DECIMAL_DEFAULT

    path = path or (STATE_DIR / "tuning_results.json")
    if not path.exists():
        return {
            "walkforward_edge_threshold": GOOD_BET_EDGE,
            "max_favorite_decimal": ML_MAX_FAVORITE_DECIMAL_DEFAULT,
            "ou_min_edge": float(OU_MIN_EDGE),
            "recommended_profile": "moderate",
            "elo_calibration": load_elo_knobs().to_dict(),
        }
    cfg = json.loads(path.read_text())
    if "elo_calibration" not in cfg:
        cfg["elo_calibration"] = load_elo_knobs().to_dict()
    return cfg


def _ou_direction_for_predict(preds: dict, feat: dict, cfg: dict) -> str:
    from pipeline.config import OU_BET_ENABLED, OU_MIN_EDGE, OU_DEFAULT_SIGMA
    from pipeline.market import select_ou_bet

    if not OU_BET_ENABLED:
        return "Pass"
    market_total = feat.get("market_total")
    if market_total is None or not np.isfinite(float(market_total or np.nan)) or float(market_total) <= 0:
        return "Pass"
    ou_min = float(cfg.get("ou_min_edge", OU_MIN_EDGE))
    pred_total = float(preds.get("pred_total", 0) or 0)
    sigma = preds.get("sigma_total", OU_DEFAULT_SIGMA)
    direction, _, _ = select_ou_bet(pred_total, market_total, sigma, min_edge=ou_min)
    return direction


def load_score_pair_model(path: Path | None = None):
    path = path or (STATE_DIR / "score_pair.pkl")
    if not path.exists():
        return None
    from pipeline.model import MetaScorePairModel
    return MetaScorePairModel.load(path)


def load_live_calibrator(path: Path | None = None):
    from pipeline.bet_confidence import load_latest_confidence_weights
    from pipeline.config import CONFIDENCE_MODE

    path = path or default_calibrator_path()
    cal = WalkForwardBetCalibrator.load(path) if path.exists() else None
    if cal is None and CONFIDENCE_MODE == "unified":
        cal = WalkForwardBetCalibrator()
    if cal is not None and CONFIDENCE_MODE == "unified":
        cal.confidence_weights = load_latest_confidence_weights()
    return cal


def load_min_confidence_threshold(default: float | None = None) -> float:
    from pipeline.bet_confidence import confidence_weights_path
    from pipeline.config import MIN_CONFIDENCE_SCORE, STATE_DIR

    default = float(MIN_CONFIDENCE_SCORE if default is None else default)
    tuning_path = STATE_DIR / "tuning_results.json"
    if tuning_path.exists():
        try:
            payload = json.loads(tuning_path.read_text())
            if "suggested_min_confidence" in payload:
                return float(payload["suggested_min_confidence"])
            if "min_confidence_score" in payload:
                return float(payload["min_confidence_score"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    path = confidence_weights_path()
    if not path.exists():
        return default
    try:
        with open(path) as f:
            payload = json.load(f)
        return float(payload.get("suggested_min_confidence", default))
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        return default


def load_max_confidence_threshold(default: float | None = None) -> float | None:
    from pipeline.config import MAX_CONFIDENCE_SCORE, STATE_DIR

    if default is None:
        default = MAX_CONFIDENCE_SCORE
    tuning_path = STATE_DIR / "tuning_results.json"
    if tuning_path.exists():
        try:
            payload = json.loads(tuning_path.read_text())
            if "max_confidence_score" in payload and payload["max_confidence_score"] is not None:
                return float(payload["max_confidence_score"])
        except (OSError, json.JSONDecodeError, TypeError, ValueError):
            pass
    return default


def load_elo_calibrator(path: Path | None = None):
    path = path or default_elo_calibrator_path()
    if path.exists():
        return WalkForwardEloCalibrator.load(path)
    return None


def load_disagreement_model(path: Path | None = None):
    path = path or default_disagreement_path()
    if path.exists():
        return WalkForwardMarketDisagreementModel.load(path)
    return None


def load_total_model(path: Path | None = None):
    from pipeline.model import MetaTotalModel
    path = path or (STATE_DIR / "latest_total.pkl")
    if not path.exists():
        path = STATE_DIR / "total.pkl"
    if path.exists():
        return MetaTotalModel.load(path)
    return None


def load_win_model(path: Path | None = None):
    path = path or (STATE_DIR / "latest_win.pkl")
    if path.exists():
        return MetaWinModel.load(path)
    return None


def load_ats_classifier(path: Path | None = None):
    path = path or (STATE_DIR / "latest_ats.pkl")
    if not path.exists():
        return None
    from pipeline.ats_classifier import ATSClassifier
    return ATSClassifier.load(path)


def load_upset_classifier(path: Path | None = None):
    from pipeline.upset_classifier import UpsetClassifier, default_upset_classifier_path

    path = path or default_upset_classifier_path()
    if not path.exists():
        path = STATE_DIR / "latest_upset.pkl"
    if not path.exists():
        return None
    return UpsetClassifier.load(path)


def load_vol_tracker(path: Path | None = None):
    path = path or (STATE_DIR / "latest_vol.pkl")
    if not path.exists():
        return None
    from pipeline.team_volatility import TeamVolatilityTracker
    return TeamVolatilityTracker.load_state(path)


def load_ml_calibrator(path: Path | None = None):
    from pipeline.ml_calibration import WalkForwardMLCalibrator, default_ml_calibrator_path

    path = path or default_ml_calibrator_path()
    if path.exists():
        return WalkForwardMLCalibrator.load(path)
    return None


def predict_game(
    home_abbr: str,
    away_abbr: str,
    game_date,
    hier_engine,
    elo_tracker,
    meta_model,
    pace_tracker,
    team_xppp_tracker=None,
    team_form_tracker=None,
    spread_calibrator=None,
    rotation_tracker=None,
    lineup_elo_tracker=None,
    chemistry_tracker=None,
    team_elo_tracker=None,
    travel_tracker=None,
    epm_tracker=None,
    ref_tracker=None,
    shot_quality_tracker=None,
    hapm_tracker=None,
    hierarchical_pace=None,
    hier_shot_rates=None,
    minutes_model=None,
    target_family_calibrator=None,
    ctx: PredictionContext | None = None,
    last_game_dates: dict | None = None,
    home_starters: list | None = None,
    away_starters: list | None = None,
    inactive_ids: list | None = None,
    live_market_spread=None,
    live_market_ml=None,
    live_market_total=None,
    odds_dict=None,
    win_model=None,
    total_model=None,
    score_pair_model=None,
    cover_prob_calibrator=None,
    ats_classifier=None,
    upset_classifier=None,
    vol_tracker=None,
    confidence_calibrator=None,
    tuning_config: dict | None = None,
    elo_calibrator=None,
    auto_load_models: bool = True,
):
    """Predict a single upcoming game with full train/serve feature parity."""
    if not getattr(meta_model, "fitted", False):
        raise RuntimeError("MetaScoreModel must be fitted before predicting games.")

    if auto_load_models:
        if win_model is None:
            win_model = load_win_model()
        if total_model is None:
            total_model = load_total_model()
        if score_pair_model is None:
            score_pair_model = load_score_pair_model()
        if ats_classifier is None:
            ats_classifier = load_ats_classifier()
        if vol_tracker is None:
            vol_tracker = load_vol_tracker()

    cfg = tuning_config or load_tuning_config()
    edge_thr = float(
        cfg.get("walkforward_edge_threshold")
        or cfg.get("optimal_bet_edge")
        or GOOD_BET_EDGE
    )
    max_fav = float(cfg.get("max_favorite_decimal", 1.45))
    recommended = cfg.get("recommended_profile", "moderate")
    if elo_calibrator is None:
        elo_calibrator = load_elo_calibrator()

    from pipeline.config import (
        USE_HIERARCHICAL_PACE, USE_HIERARCHICAL_SHOT_ZONES, USE_ROTATION_SCENARIOS,
        USE_TARGET_FAMILY_CALIBRATION, USE_HYBRID_STRUCTURED_BLEND,
        HYBRID_TOTAL_BETA, HYBRID_MARGIN_BETA,
    )
    if hierarchical_pace is None and USE_HIERARCHICAL_PACE:
        try:
            from pipeline.pace_hierarchy import HierarchicalPaceModel
            hierarchical_pace = getattr(ctx, "hierarchical_pace", None) if ctx else None
            if hierarchical_pace is None:
                hierarchical_pace = HierarchicalPaceModel(baseline=pace_tracker)
        except Exception:
            hierarchical_pace = None
    if hier_shot_rates is None and USE_HIERARCHICAL_SHOT_ZONES:
        try:
            from pipeline.shot_hierarchy import HierarchicalZoneRates
            hier_shot_rates = getattr(ctx, "hier_shot_rates", None) if ctx else None
            if hier_shot_rates is None:
                hier_shot_rates = HierarchicalZoneRates()
        except Exception:
            hier_shot_rates = None
    if minutes_model is None and USE_ROTATION_SCENARIOS:
        try:
            from pipeline.minutes_hierarchy import HierarchicalMinutesModel
            minutes_model = getattr(ctx, "minutes_model", None) if ctx else None
            if minutes_model is None:
                minutes_model = HierarchicalMinutesModel()
        except Exception:
            minutes_model = None
    if target_family_calibrator is None and USE_TARGET_FAMILY_CALIBRATION:
        try:
            from pipeline.target_calibration import TargetFamilyCalibrationRegistry
            target_family_calibrator = (
                getattr(ctx, "target_family_calibrator", None) if ctx else None
            )
            if target_family_calibrator is None:
                target_family_calibrator = TargetFamilyCalibrationRegistry()
        except Exception:
            target_family_calibrator = None

    home = TEAM_MAP.get(home_abbr, home_abbr)
    away = TEAM_MAP.get(away_abbr, away_abbr)
    gdate = pd.Timestamp(game_date)
    ctx = ctx or PredictionContext()
    if last_game_dates:
        ctx.last_date.update(last_game_dates)
    # Persist live roadmap state on context for walk-forward serve continuity.
    if hierarchical_pace is not None:
        ctx.hierarchical_pace = hierarchical_pace
    if hier_shot_rates is not None:
        ctx.hier_shot_rates = hier_shot_rates
    if minutes_model is not None:
        ctx.minutes_model = minutes_model
    if target_family_calibrator is not None:
        ctx.target_family_calibrator = target_family_calibrator

    home_starters = home_starters or ctx.lineup_cache.get(home) or ["0"]
    away_starters = away_starters or ctx.lineup_cache.get(away) or ["0"]

    odds_override = {}
    if odds_dict:
        odds_override = dict(odds_dict)
    if live_market_spread is not None:
        key = (gdate.date(), home)
        odds_override[key] = {
            "spread": live_market_spread,
            "ml": live_market_ml,
            "total": live_market_total,
            "closing_spread": live_market_spread,
        }

    feat = build_game_features(
        game_id=f"live_{home}_{away}_{gdate.date()}",
        gdate=gdate,
        home_team=home,
        away_team=away,
        home_starters=home_starters,
        away_starters=away_starters,
        elo_tracker=elo_tracker,
        hier_engine=hier_engine,
        pace_tracker=pace_tracker,
        odds_dict=odds_override or None,
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
        hierarchical_pace=hierarchical_pace,
        hier_shot_rates=hier_shot_rates,
        minutes_model=minutes_model,
        last_date=ctx.last_date,
        team_game_dates=ctx.team_game_dates,
        team_recent_net=ctx.team_recent_net,
        opponent_history=ctx.opponent_history,
        team_home_margin=ctx.team_home_margin,
        team_road_margin=ctx.team_road_margin,
        team_games_played=ctx.team_games_played,
        team_rosters_seen=ctx.team_rosters_seen,
        season_start_date=ctx.season_start_date or gdate,
        inactive_ids=inactive_ids or [],
        elo_calibrator=elo_calibrator,
    )

    feat_model = build_feature_row(feat)
    preds = meta_model.predict(feat_model)

    corrected = (
        spread_calibrator.correct(preds["pred_margin"])
        if spread_calibrator is not None
        else preds["pred_margin"]
    )

    from pipeline.config import USE_CANONICAL_SCORE_PAIR
    used_canonical_pair = False
    if score_pair_model is not None and getattr(score_pair_model, "fitted", False) and USE_CANONICAL_SCORE_PAIR:
        from pipeline.canonical_scores import apply_canonical_score_pair, prefer_canonical_margin
        preds = apply_canonical_score_pair(
            preds, feat_model, score_pair_model, legacy_margin=corrected,
            cover_calibrator=cover_prob_calibrator,
        )
        corrected = prefer_canonical_margin(preds, corrected)
        used_canonical_pair = True
    elif score_pair_model is not None and getattr(score_pair_model, "fitted", False):
        from pipeline.config import USE_SCORE_PAIR_TOTAL
        pair_out = score_pair_model.predict_scores(feat_model)
        preds["pred_home"] = pair_out["pred_home"]
        preds["pred_away"] = pair_out["pred_away"]
        preds["pred_home_pts"] = pair_out["pred_home"]
        preds["pred_away_pts"] = pair_out["pred_away"]
        preds["sigma_total"] = pair_out.get("sigma_total")
        preds["p_over"] = pair_out.get("p_over")
        if USE_SCORE_PAIR_TOTAL:
            preds["pred_total"] = pair_out["pred_total"]
            pair_margin = pair_out["pred_margin"]
            preds["score_pair_margin"] = pair_margin
            preds["margin_disagreement"] = abs(float(pair_margin) - float(corrected))
    elif total_model is not None and getattr(total_model, "fitted", False):
        total_out = total_model._predict_raw_total(feat_model)
        preds["pred_total"] = total_out["pred_total"]
        if "total_quantile_width" in total_out:
            preds["total_quantile_width"] = total_out["total_quantile_width"]
            preds["sigma_total"] = float(total_out["total_quantile_width"]) / 2.0
        preds["pred_home"] = (preds["pred_total"] + corrected) / 2.0
        preds["pred_away"] = (preds["pred_total"] - corrected) / 2.0
        preds["pred_home_pts"] = preds["pred_home"]
        preds["pred_away_pts"] = preds["pred_away"]
    else:
        preds.setdefault("pred_home_pts", preds.get("pred_home"))
        preds.setdefault("pred_away_pts", preds.get("pred_away"))
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
                float(corrected),
                structured,
                total_beta=float(HYBRID_TOTAL_BETA),
                margin_beta=float(HYBRID_MARGIN_BETA),
            )
            preds["pred_total"] = hyb["hybrid_total"]
            corrected = hyb["hybrid_margin"]
            preds["pred_margin"] = corrected
            preds["sigma_total"] = hyb["hybrid_sigma_total"]
            preds["hybrid_margin"] = hyb["hybrid_margin"]
            preds["hybrid_total"] = hyb["hybrid_total"]
        except Exception:
            pass

    if used_canonical_pair:
        preds["pred_margin"] = float(preds["pred_home"]) - float(preds["pred_away"])
        preds["pred_total"] = float(preds["pred_home"]) + float(preds["pred_away"])
        corrected = float(preds["pred_margin"])

    if minutes_model is not None and USE_ROTATION_SCENARIOS:
        try:
            from pipeline.rotation_scenarios import RotationScenarioGenerator
            from pipeline.scenario_mixer import predict_and_mix_scenarios
            roster_h = list(home_starters)
            gen = RotationScenarioGenerator(
                minutes_model,
                factor_probs=(
                    minutes_model.factor_probs()
                    if hasattr(minutes_model, "factor_probs") else None
                ),
            )
            scenarios = gen.generate(roster_h) if roster_h else []
            if scenarios:
                base_margin = float(corrected)
                base_var = max(float(feat.get("pace_var", 9.0)) * 0.25, 4.0)

                def _pred(sc):
                    repl = float(sc.minutes.get("_replacement", 0.0))
                    return base_margin - 0.02 * repl, base_var

                mixed = predict_and_mix_scenarios(scenarios, _pred)
                feat["scenario_mix_sigma"] = mixed.std
                corrected = 0.85 * corrected + 0.15 * mixed.mean
                preds["pred_margin"] = corrected
        except Exception:
            pass

    if preds.get("p_over") is None and feat_model.get("market_total"):
        from pipeline.market import total_over_prob_gaussian
        from pipeline.config import OU_DEFAULT_SIGMA
        sig = preds.get("sigma_total", OU_DEFAULT_SIGMA)
        preds["p_over"] = total_over_prob_gaussian(
            preds.get("pred_total", 0), feat_model.get("market_total"), sig,
        )

    from pipeline.config import WIN_PROB_SOURCE, REQUIRE_META_WIN_WHEN_FITTED

    use_meta_win = (
        win_model is not None and getattr(win_model, "fitted", False)
        and (
            WIN_PROB_SOURCE == "meta_win"
            or (WIN_PROB_SOURCE == "auto" and REQUIRE_META_WIN_WHEN_FITTED)
        )
    )
    if use_meta_win:
        raw_wp = win_model.predict_proba(feat_model, pred_margin=corrected)
        preds["win_prob_raw"] = raw_wp
        ml_cal = load_ml_calibrator()
        preds["win_prob"] = (
            ml_cal.transform(raw_wp, market_ml=feat.get("market_ml"), market_spread=feat.get("market_spread"))
            if ml_cal is not None and getattr(ml_cal, "_fitted", False)
            else raw_wp
        )
    elif WIN_PROB_SOURCE == "rolling_platt":
        preds["win_prob"] = meta_model.calibrate_prob(corrected)

    if target_family_calibrator is not None and USE_TARGET_FAMILY_CALIBRATION:
        try:
            if "win_prob" in preds:
                preds["win_prob"] = target_family_calibrator.get("moneyline").predict(
                    float(preds["win_prob"]), team=home,
                )
            if preds.get("p_over") is not None:
                preds["p_over"] = target_family_calibrator.get("total_over").predict(
                    float(preds["p_over"]), team=home,
                )
        except Exception:
            pass

    if spread_calibrator is not None and hasattr(spread_calibrator, "predict_interval"):
        conf_lower, conf_upper, conf_width = spread_calibrator.predict_interval(corrected, alpha=0.10)
    else:
        conf_lower, conf_upper, conf_width = corrected - 12.0, corrected + 12.0, 24.0

    q_width = preds.get("spread_quantile_width")
    if q_width is not None and np.isfinite(q_width):
        conf_width = float(q_width)
        if preds.get("spread_q10") is not None and preds.get("spread_q90") is not None:
            conf_lower = float(preds["spread_q10"])
            conf_upper = float(preds["spread_q90"])

    market_spread = feat.get("market_spread", live_market_spread)
    market_ml = feat.get("market_ml", live_market_ml)
    rating_uncertainty = float(feat.get("h_rating_uncertainty", 350) + feat.get("a_rating_uncertainty", 350))

    effective_edge_thr = variance_aware_edge_threshold(edge_thr, conf_width)
    disagreement_info = {}
    disagreement_model = load_disagreement_model()
    if disagreement_model is not None and disagreement_model.fitted:
        disagreement_info = disagreement_model.predict(
            elo_margin_calibrated=feat.get("elo_margin_calibrated"),
            model_spread=corrected,
            closing_spread=feat.get("closing_spread", market_spread),
            market_spread=market_spread,
            uncertainty=rating_uncertainty,
            spread_move=feat.get("spread_move", 0),
            h_star_out=float(feat.get("h_star_out", 0) or 0),
            a_star_out=float(feat.get("a_star_out", 0) or 0),
        )
        effective_edge_thr += float(disagreement_info.get("edge_bump", 0.0))

    from pipeline.bet_selection import (
        analysis_min_edge_pts,
        apply_bet_selection_gates,
        edge_bucket_min_for_stakes,
        edge_stake_multiplier,
        passes_confidence_actionable_gates,
        spread_lean_from_edge,
        uses_edge_gates,
    )
    from pipeline.calibration_policy import blend_ats_classifier_cover, should_use_bet_calibrator
    from pipeline.bet_confidence import build_confidence_features
    from pipeline.market import elo_meta_agreement

    from pipeline.config import (
        CONFIDENCE_SELECTION_MODE, STAKE_SIZING_MODE, MIN_ML_WIN_PCT,
        CONFIDENCE_MIN_EDGE,
        OU_BET_ENABLED, OU_MIN_EDGE,
        ATS_CLASSIFIER_BLEND, ATS_CLASSIFIER_MIN_PROB, BET_SELECTION_MODE,
    )

    min_conf = load_min_confidence_threshold()
    max_conf = load_max_confidence_threshold()

    matchup_vol_sigma = (
        float(vol_tracker.matchup_sigma(home, away)) if vol_tracker is not None else np.nan
    )

    if upset_classifier is None:
        upset_classifier = load_upset_classifier()
    upset_prob = None
    blowout_probs = preds.get("blowout_probs") if isinstance(preds.get("blowout_probs"), dict) else None
    if blowout_probs is None and hasattr(meta_model, "predict_blowout_probs"):
        try:
            blowout_probs = meta_model.predict_blowout_probs(feat_model)
        except Exception:
            blowout_probs = None
    if (
        upset_classifier is not None and getattr(upset_classifier, "fitted", False)
        and market_spread is not None and pd.notna(market_spread)
    ):
        upset_prob = upset_classifier.predict_upset_prob({
            **feat,
            "PRED_SPREAD": corrected,
            "pred_margin": corrected,
            "MARKET_SPREAD": market_spread,
            "MATCHUP_VOL_SIGMA": matchup_vol_sigma,
            "WIN_PROB": preds.get("win_prob", 0.5),
            "SPREAD_QUANTILE_WIDTH": conf_width,
            "RATING_UNCERTAINTY": rating_uncertainty,
            "DISAGREEMENT_TRUST": disagreement_info.get("disagreement_trust", 1.0),
            "H_STAR_OUT": feat.get("h_star_out", 0),
            "A_STAR_OUT": feat.get("a_star_out", 0),
        })

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

    calibrator = confidence_calibrator or load_live_calibrator()
    market_ml_away = feat.get("market_ml_away", np.nan)
    ba = bet_analysis(
        model_spread=corrected,
        market_spread=market_spread,
        model_win_prob=preds.get("win_prob"),
        market_ml=market_ml,
        min_edge_pts=analysis_min_edge_pts(effective_edge_thr),
        spread_move=feat.get("spread_move", 0),
        public_home_pct=feat.get("public_home_pct", 0.5),
        conf_width=conf_width,
        # Single calibrator call below — avoid double WalkForwardBetCalibrator.predict.
        confidence_calibrator=None,
        defer_confidence_calib=True,
        rating_uncertainty=rating_uncertainty,
        max_favorite_decimal=max_fav,
        elo_margin_calibrated=feat.get("elo_margin_calibrated"),
        disagreement_trust=disagreement_info.get("disagreement_trust", 1.0),
        phantom_injury_flag=disagreement_info.get("phantom_injury_flag", False),
        pred_home=(preds.get("pred_total", 220) + corrected) / 2.0,
        pred_away=(preds.get("pred_total", 220) - corrected) / 2.0,
        matchup_vol_sigma=matchup_vol_sigma,
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
    direction = apply_bet_selection_gates(
        direction,
        edge_pts,
        conf_width,
        elo_meta_agreement=elo_meta_agreement(
            corrected, market_spread, feat.get("elo_margin_calibrated"),
        ),
    )

    lean_for_conf = model_lean if model_lean != "Pass" else spread_lean_from_edge(edge_pts)
    raw_cover = ba.get("spread_cover_prob_raw", ba.get("cover_prob_calibrated", 0.5))
    ats_prob = None
    home_cover_ats = None
    if ats_classifier is None:
        ats_classifier = load_ats_classifier()
    if (
        ats_classifier is not None and getattr(ats_classifier, "fitted", False)
        and lean_for_conf != "Pass" and market_spread is not None and pd.notna(market_spread)
    ):
        from pipeline.ats_ev import cover_prob_for_side
        ats_feat = {**feat, "pred_margin": corrected, "market_spread": market_spread}
        home_cover_ats = ats_classifier.predict_home_cover_prob(ats_feat, market_spread)
        ats_prob = cover_prob_for_side(home_cover_ats, lean_for_conf)
        if BET_SELECTION_MODE == "edge_bucket_ats" and ats_prob < ATS_CLASSIFIER_MIN_PROB:
            direction = "Pass"
    blended_cover = raw_cover
    if ats_prob is not None:
        blended_cover = blend_ats_classifier_cover(raw_cover, ats_prob, ATS_CLASSIFIER_BLEND)

    margin_wp = float(np.clip(1.0 / (1.0 + np.exp(-float(corrected) / 12.0)), 0.01, 0.99))
    if lean_for_conf != "Pass" and calibrator is not None and should_use_bet_calibrator():
        conf_feats = build_confidence_features(
            spread_edge_pts=edge_pts,
            conf_width=conf_width,
            rating_uncertainty=rating_uncertainty,
            elo_meta_agreement=elo_meta_agreement(
                corrected, market_spread, feat.get("elo_margin_calibrated"),
            ),
            spread_cover_prob=blended_cover,
            matchup_vol_sigma=matchup_vol_sigma,
            disagreement_trust=disagreement_info.get("disagreement_trust", 1.0),
            phantom_injury_flag=disagreement_info.get("phantom_injury_flag", False),
            ats_classifier_prob=ats_prob,
            win_prob=margin_wp,
            elo_margin=float(feat.get("elo_margin_calibrated", 0) or 0),
            lean=lean_for_conf,
        )
        cal = calibrator.predict("ats", conf_feats, direction=lean_for_conf)
        conf_score = int(round(cal["confidence_score"]))
        conf_raw = cal["confidence_score_raw"]
        cover_prob = float(np.clip(cal["calibrated_prob"], 0.01, 0.99))
        conf_tier = cal["confidence_tier"]
    else:
        cover_prob = ba.get("cover_prob_calibrated", 0.5)
        conf_tier = ba.get("confidence_tier", 2)

    elo_agree = elo_meta_agreement(
        corrected, market_spread, feat.get("elo_margin_calibrated"),
    )
    actionable = passes_confidence_actionable_gates(
        lean=lean_for_conf,
        edge_pts=edge_pts,
        conf_score=conf_score,
        conf_width=conf_width,
        min_confidence=min_conf,
        max_confidence=max_conf,
        disagreement_trust=disagreement_info.get("disagreement_trust", 1.0),
        phantom_injury_flag=disagreement_info.get("phantom_injury_flag", False),
        market_spread=market_spread,
        elo_meta_agreement=elo_agree,
    )
    if lean_for_conf == "Pass" or (uses_edge_gates() and direction == "Pass"):
        actionable = False

    closing_spread = feat.get("closing_spread", market_spread)
    from pipeline.config import REQUIRE_NONNEGATIVE_CLV, MIN_CLV_POINTS
    from pipeline.metrics import compute_clv

    clv_pre = np.nan
    if pd.notna(market_spread) and pd.notna(closing_spread):
        clv_pre = compute_clv(corrected, market_spread, closing_spread)
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
    elif lean_for_conf != "Pass":
        direction = "Pass"

    edge_mult = edge_stake_multiplier(abs(edge_pts)) if uses_edge_gates() else 1.0
    edge_bucket_min = edge_bucket_min_for_stakes()
    vol_mult = 1.0
    from pipeline.config import STAKE_SIZING_MODE
    if STAKE_SIZING_MODE == "volatility_adjusted" and vol_tracker is not None:
        vol_mult = vol_tracker.stake_multiplier(home, away)

    from pipeline.config import USE_VENN_ABERS_FILTER, VENN_ABERS_MAX_WIDTH
    if (
        USE_VENN_ABERS_FILTER
        and direction != "Pass"
        and calibrator is not None
        and hasattr(calibrator, "venn_abers_width")
    ):
        va_w = calibrator.venn_abers_width(float(conf_score))
        if va_w is not None and va_w > VENN_ABERS_MAX_WIDTH:
            direction = "Pass"
    actionable = bool(direction != "Pass")

    stakes = {}
    for profile in PROFILES:
        stakes[f"stake_{profile}"] = round(
            compute_stake(
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
            ),
            4,
        )

    from pipeline.market import explain_ml_bet, select_ou_bet
    from pipeline.config import OU_DEFAULT_SIGMA

    ml_explain = explain_ml_bet(
        preds.get("win_prob", 0.5), market_ml,
        market_ml_away=feat.get("market_ml_away"),
        market_spread=market_spread,
    )
    sigma_t = preds.get("sigma_total", OU_DEFAULT_SIGMA)
    ou_dir, p_over, _ = select_ou_bet(
        preds.get("pred_total", 0),
        feat.get("market_total", live_market_total),
        sigma_t,
        min_edge=float(cfg.get("ou_min_edge", OU_MIN_EDGE)),
    )
    if preds.get("p_over") is not None:
        p_over = preds["p_over"]

    return {
        "date": str(gdate.date()),
        "home": home,
        "away": away,
        "pred_spread": round(corrected, 2),
        "pred_margin": round(corrected, 2),
        "pred_total": round(preds.get("pred_total", 0), 1),
        "pred_home": round(float(preds.get("pred_home", 0) or 0), 1),
        "pred_away": round(float(preds.get("pred_away", 0) or 0), 1),
        "pred_home_pts": round(float(preds.get("pred_home_pts", preds.get("pred_home", 0)) or 0), 1),
        "pred_away_pts": round(float(preds.get("pred_away_pts", preds.get("pred_away", 0)) or 0), 1),
        "sigma_total": round(float(sigma_t), 2) if sigma_t is not None else None,
        "p_over": round(float(p_over), 3) if p_over is not None else None,
        "p_under": round(1.0 - float(p_over), 3) if p_over is not None else None,
        "win_prob": round(preds.get("win_prob", 0.5), 3),
        "p_home": round(float(ml_explain.get("p_home", preds.get("win_prob", 0.5))), 3),
        "p_away": round(float(ml_explain.get("p_away", 1.0 - preds.get("win_prob", 0.5))), 3),
        "predicted_winner": ml_explain.get("predicted_winner", "Pass"),
        "ev_home": round(float(ml_explain["ev_home"]), 4) if pd.notna(ml_explain.get("ev_home")) else None,
        "ev_away": round(float(ml_explain["ev_away"]), 4) if pd.notna(ml_explain.get("ev_away")) else None,
        "ml_bet": ml_explain.get("ml_bet", "Pass"),
        "ml_reason": ml_explain.get("ml_reason", ""),
        "edge": round(edge_pts if np.isfinite(edge_pts) else spread_edge(corrected, market_spread) or 0, 2),
        "edge_lean": model_lean,
        "win_pct": conf_score,
        "actionable": actionable,
        "direction": direction,
        "stars": ba.get("spread_stars", ""),
        "confidence_score": conf_score,
        "win_probability_pct": conf_score,
        "confidence_score_raw": round(float(conf_raw), 3) if pd.notna(conf_raw) else None,
        "matchup_vol_sigma": round(float(matchup_vol_sigma), 3) if pd.notna(matchup_vol_sigma) else None,
        "confidence_tier": conf_tier,
        "cover_prob_calibrated": round(cover_prob, 3),
        "ou_direction": ou_dir,
        "ml_direction": ml_explain.get("ml_bet", ba.get("ml_direction", "Pass")),
        "edge_threshold": effective_edge_thr,
        "base_edge_threshold": edge_thr,
        "spread_quantile_width": round(float(q_width), 2) if q_width is not None and np.isfinite(q_width) else None,
        "phantom_injury_flag": bool(disagreement_info.get("phantom_injury_flag", False)),
        "disagreement_trust": round(float(disagreement_info.get("disagreement_trust", 1.0)), 3),
        "margin_disagreement": round(float(preds.get("margin_disagreement", 0) or 0), 2),
        "recommended_profile": recommended,
        "stake_recommended": stakes.get(f"stake_{recommended}", 0),
        **stakes,
        "features": feat,
    }
