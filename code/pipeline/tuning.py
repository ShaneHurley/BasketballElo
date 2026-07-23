"""Hyperparameter tuning."""
from __future__ import annotations

import optuna
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

from pipeline.config import (
    DEFAULT_LEAGUE_RTG,
    TUNING_BOUND_EDGE_FRACTION,
    TUNING_INVALID_SCORE,
    TUNING_POINTS_WEIGHT,
    TUNING_XPPP_WEIGHT,
)
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.model import AdaptiveParameterBounds
from pipeline.ratings import PlayerRatingTracker
from pipeline.stint_context import build_stint_context
from pipeline.utils import _parse_player_string, map_elo_params

# Original Optuna search bounds (AdaptiveParameterBounds shrinks around running best).
ELO_PARAM_BOUNDS = {
    "k_off": (0.05, 1.5),
    "k_def": (0.05, 1.5),
    "elo_scaling": (200, 1200),
    "home_boost": (0.0001, 0.007),
    "offseason_reversion": (0.1, 0.30),
    "usage_floor": (0.20, 0.32),
    "assist_split": (0.60, 0.95),
    "k_mult_half_life": (10.0, 35.0),
    "rd_floor": (25.0, 50.0),
    "garbage_time_weight": (0.0, 1.0),
    "clutch_boost": (1.0, 1.6),
    "tov_penalty": (0.7, 1.0),
    "foul_draw_boost": (1.0, 1.3),
    "variance_dampen": (0.5, 1.0),
    "xppp_actual_blend": (0.0, 0.2),
    "k_def_events": (0.05, 0.35),
    "tov_rate_threshold": (0.10, 0.25),
    "three_pa_rate_threshold": (0.35, 0.55),
}

HIER_PARAM_BOUNDS = {
    "w1": (0.1, 1.0),
    "w2": (0.1, 0.5),
    "w3": (0.1, 0.5),
    "w5": (0.1, 1.0),
    "k_off": (0.05, 0.9),
    "k_def": (0.01, 0.90),
    "league_rtg": (109.5, 114.0),
}


def _blend_mae(points_mae: float, xppp_mae: float) -> float:
    return TUNING_XPPP_WEIGHT * xppp_mae + TUNING_POINTS_WEIGHT * points_mae


def _params_at_bounds(
    trial_params: dict,
    bounds_spec: dict,
    edge_frac: float = TUNING_BOUND_EDGE_FRACTION,
) -> list[str]:
    """Return param names whose values sit within edge_frac of original bounds."""
    at_edge = []
    for name, spec in bounds_spec.items():
        if name not in trial_params:
            continue
        val = trial_params[name]
        if isinstance(spec[0], int):
            lo, hi = float(spec[0]), float(spec[1])
        else:
            lo, hi = spec
        span = hi - lo
        if span <= 0:
            continue
        if val <= lo + edge_frac * span or val >= hi - edge_frac * span:
            at_edge.append(name)
    return at_edge


def _suggest_float_param(trial, bounds, name, lo, hi, log=False):
    if bounds is not None:
        lo, hi = bounds.suggest_bounds_float(name, lo, hi, log=log)
    return trial.suggest_float(name, lo, hi, log=log)


def _suggest_int_param(trial, bounds, name, lo, hi):
    if bounds is not None:
        lo, hi = bounds.suggest_bounds_int(name, lo, hi)
    return trial.suggest_int(name, lo, hi)


def _ensure_season_column(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "game_date" in df.columns:
        df = df.sort_values(["game_date", "GAME_ID", "stint_id"]).reset_index(drop=True)
    if "season" not in df.columns and "game_date" in df.columns:
        dt = pd.to_datetime(df["game_date"])
        df["season"] = dt.dt.year + (dt.dt.month >= 9).astype(int)
    return df


def _prepare_stint_records(df: pd.DataFrame) -> list[dict]:
    """Pre-parse player strings and numeric fields once per tuning objective."""
    records = []
    cols = df.columns
    has_season = "season" in cols
    for row in df.itertuples(index=False):
        poss = float(getattr(row, "possessions", 1) or 1)
        if not (np.isfinite(poss) and poss >= 1):
            continue
        hp = _parse_player_string(row.HOME_players)
        ap = _parse_player_string(row.AWAY_players)
        rec = {
            "hp": hp,
            "ap": ap,
            "poss": poss,
            "home_xpts": float(getattr(row, "home_xpts", 0) or 0),
            "away_xpts": float(getattr(row, "away_xpts", 0) or 0),
            "home_pts": float(getattr(row, "home_pts", 0) or 0),
            "away_pts": float(getattr(row, "away_pts", 0) or 0),
            "home_usage": getattr(row, "home_usage", {}),
            "away_usage": getattr(row, "away_usage", {}),
            "period": int(getattr(row, "PERIOD", 1) or 1),
            "start_A": float(getattr(row, "HOME_SCORE_START", 0) or 0),
            "start_B": float(getattr(row, "AWAY_SCORE_START", 0) or 0),
            "end_A": float(getattr(row, "HOME_SCORE_END", 0) or 0),
            "end_B": float(getattr(row, "AWAY_SCORE_END", 0) or 0),
            "stint_ctx": build_stint_context(row),
        }
        if has_season:
            rec["season"] = getattr(row, "season")
        if hasattr(row, "GAME_ID"):
            rec["game_id"] = getattr(row, "GAME_ID")
        records.append(rec)
    return records


def _group_records_by_season(records: list[dict]) -> dict:
    grouped = {}
    for rec in records:
        grouped.setdefault(rec["season"], []).append(rec)
    return grouped


def _elo_warmup(tracker, records: list[dict]) -> None:
    for rec in records:
        tracker.process_stint(
            ids_A=rec["hp"], ids_B=rec["ap"], poss=rec["poss"],
            xpts_A=rec["home_xpts"], xpts_B=rec["away_xpts"],
            usage_A=rec["home_usage"], usage_B=rec["away_usage"],
            period=rec["period"],
            start_A=rec["start_A"], start_B=rec["start_B"],
            end_A=rec["end_A"], end_B=rec["end_B"],
            season_progress=0.5,
            stint_ctx=rec.get("stint_ctx"),
        )


def _game_margin_from_tracker(tracker, hp, ap, poss, league_xppp, home_boost, elo_scaling):
    ho_off, ho_def, _ = tracker.lineup_stats(hp)
    ao_off, ao_def, _ = tracker.lineup_stats(ap)
    pred_ppp_h = league_xppp + home_boost + (ho_off - ao_def) / elo_scaling
    pred_ppp_a = league_xppp - home_boost + (ao_off - ho_def) / elo_scaling
    return (pred_ppp_h - pred_ppp_a) * poss


def _elo_eval_fold(
    tracker, records: list[dict], league_xppp: float,
    home_boost: float, elo_scaling: int,
) -> tuple[float, float]:
    """Return (blend_mae, ats_miss_rate) for validation records."""
    yt, yp, xppp_yt, xppp_yp = [], [], [], []
    by_game: dict = {}
    for rec in records:
        gid = rec.get("game_id", 0)
        by_game.setdefault(gid, []).append(rec)

    ats_miss = 0.5
    game_margins_pred, game_margins_act = [], []
    for recs in by_game.values():
        hp, ap = recs[0]["hp"], recs[0]["ap"]
        poss = sum(r["poss"] for r in recs)
        act_margin = sum(r["home_pts"] for r in recs) - sum(r["away_pts"] for r in recs)
        pred_margin = _game_margin_from_tracker(
            tracker, hp, ap, poss, league_xppp, home_boost, elo_scaling,
        )
        game_margins_pred.append(pred_margin)
        game_margins_act.append(act_margin)

        for rec in recs:
            ho_off, ho_def, _ = tracker.lineup_stats(rec["hp"])
            ao_off, ao_def, _ = tracker.lineup_stats(rec["ap"])
            pred_ppp_h = league_xppp + home_boost + (ho_off - ao_def) / elo_scaling
            pred_ppp_a = league_xppp - home_boost + (ao_off - ho_def) / elo_scaling
            tracker.process_stint(
                ids_A=rec["hp"], ids_B=rec["ap"], poss=rec["poss"],
                xpts_A=rec["home_xpts"], xpts_B=rec["away_xpts"],
                usage_A=rec["home_usage"], usage_B=rec["away_usage"],
                period=rec["period"],
                start_A=rec["start_A"], start_B=rec["start_B"],
                end_A=rec["end_A"], end_B=rec["end_B"],
                season_progress=0.5,
                stint_ctx=rec.get("stint_ctx"),
            )
            poss_i = rec["poss"]
            yt.extend([rec["home_pts"], rec["away_pts"]])
            yp.extend([pred_ppp_h * poss_i, pred_ppp_a * poss_i])
            xppp_yt.extend([rec["home_xpts"], rec["away_xpts"]])
            xppp_yp.extend([pred_ppp_h * poss_i, pred_ppp_a * poss_i])

    if game_margins_pred:
        hits = sum(
            1 for p, a in zip(game_margins_pred, game_margins_act)
            if (p > 0) == (a > 0) or (p == 0 and a == 0)
        )
        ats_miss = 1.0 - hits / len(game_margins_pred)

    mae = _fold_mae_from_predictions(yt, yp, xppp_yt, xppp_yp)
    return mae, ats_miss


def _fold_mae_from_predictions(
    yt: list[float], yp: list[float],
    xppp_yt: list[float], xppp_yp: list[float],
) -> float:
    arr_t, arr_p = np.array(yt), np.array(yp)
    arr_x_t, arr_x_p = np.array(xppp_yt), np.array(xppp_yp)
    mask = np.isfinite(arr_t) & np.isfinite(arr_p)
    mask_x = np.isfinite(arr_x_t) & np.isfinite(arr_x_p)
    if mask.sum() > 10 and mask_x.sum() > 10:
        return _blend_mae(
            mean_absolute_error(arr_t[mask], arr_p[mask]),
            mean_absolute_error(arr_x_t[mask_x], arr_x_p[mask_x]),
        )
    return TUNING_INVALID_SCORE


MIN_SINGLE_SEASON_TUNING_RECORDS = 50

_ELO_CONFIG_KEYS = (
    "K_OFF", "K_DEF", "ELO_SCALING_FACTOR", "HOME_PPP_BOOST", "OFFSEASON_REVERSION",
    "USAGE_FLOOR", "assist_split", "k_mult_half_life", "rd_floor", "garbage_time_weight",
    "clutch_boost", "tov_penalty", "foul_draw_boost", "variance_dampen",
    "xppp_actual_blend", "k_def_events", "tov_rate_threshold", "three_pa_rate_threshold",
)


def default_elo_config() -> dict:
    """Pipeline default Elo config (used when CV tuning has no valid folds)."""
    d = PlayerRatingTracker.DEFAULTS
    return {k: d[k] for k in _ELO_CONFIG_KEYS}


def default_hier_config() -> dict:
    """Pipeline default hierarchical weights."""
    return {
        "w1": 0.50,
        "w2": 0.25,
        "w3": 0.15,
        "w5": 0.10,
        "k_off": 2.0,
        "k_def": 1.5,
        "league_avg_rtg": float(DEFAULT_LEAGUE_RTG),
    }


def _season_walkforward_mae(
    seasons: list,
    by_season: dict,
    fold_fn,
) -> float:
    fold_maes = []
    if len(seasons) >= 2:
        for i in range(1, len(seasons)):
            train_recs = []
            for s in seasons[:i]:
                train_recs.extend(by_season.get(s, []))
            val_recs = by_season.get(seasons[i], [])
            if not train_recs or not val_recs:
                fold_maes.append(TUNING_INVALID_SCORE)
                continue
            fold_maes.append(fold_fn(train_recs, val_recs))
    elif len(seasons) == 1:
        # One training season: chronological 70/30 holdout within that season.
        recs = by_season.get(seasons[0], [])
        if len(recs) >= MIN_SINGLE_SEASON_TUNING_RECORDS:
            split = int(len(recs) * 0.7)
            train_recs = recs[:split]
            val_recs = recs[split:]
            if train_recs and val_recs:
                fold_maes.append(fold_fn(train_recs, val_recs))
    if not fold_maes:
        if len(seasons) == 1:
            n = len(by_season.get(seasons[0], []))
            print(
                f"  ⚠ Tuning CV: single season with {n} stints "
                f"(need ≥{MIN_SINGLE_SEASON_TUNING_RECORDS} for holdout).",
            )
        elif len(seasons) < 2:
            print("  ⚠ Tuning CV: no seasons in training data.")
        return TUNING_INVALID_SCORE
    return float(np.mean(fold_maes))


def _hier_warmup(engine, records: list[dict]) -> None:
    for rec in records:
        engine.update(rec["hp"], rec["ap"], rec["home_pts"], rec["away_pts"], rec["poss"])


def _hier_eval_fold(engine, records: list[dict]) -> float:
    yt, yp, xppp_yt, xppp_yp = [], [], [], []
    for rec in records:
        poss = rec["poss"]
        xh, xa, _, _ = engine.predict_pts(rec["hp"], rec["ap"], poss)
        xppp_h = rec["home_xpts"] / poss if poss else 0.0
        xppp_a = rec["away_xpts"] / poss if poss else 0.0
        if np.isfinite(xh) and np.isfinite(xa):
            yt.extend([rec["home_pts"], rec["away_pts"]])
            yp.extend([xh, xa])
        xppp_yt.extend([xppp_h, xppp_a])
        xppp_yp.extend([xh / poss if poss else 0.0, xa / poss if poss else 0.0])
        engine.update(rec["hp"], rec["ap"], rec["home_pts"], rec["away_pts"], poss)
    if len(yt) <= 10:
        return TUNING_INVALID_SCORE
    points_mae = mean_absolute_error(yt, yp)
    arr_x_t, arr_x_p = np.array(xppp_yt), np.array(xppp_yp)
    mask_x = np.isfinite(arr_x_t) & np.isfinite(arr_x_p)
    xppp_mae = (
        mean_absolute_error(arr_x_t[mask_x], arr_x_p[mask_x])
        if mask_x.sum() > 10 else points_mae
    )
    return _blend_mae(points_mae, xppp_mae)


def _run_optuna_study(objective, n_trials, label, bounds_spec=None):
    def callback(study, trial):
        print(f"\nTrial {trial.number}:")
        for key, value in trial.params.items():
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
        if trial.value is not None:
            print(f"  CV MAE: {trial.value:.4f}")
        if bounds_spec and trial.params:
            at_edge = _params_at_bounds(trial.params, bounds_spec)
            if at_edge:
                print(f"  ⚠ at bound edge: {', '.join(at_edge)}")

    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=42),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True, callbacks=[callback])

    print("\n" + "=" * 50)
    print(f"✅ {label} COMPLETE")
    print(f"Best CV MAE: {study.best_value:.4f}")
    if study.best_value >= TUNING_INVALID_SCORE:
        print(
            f"  ⚠ WARNING: all trials scored {TUNING_INVALID_SCORE:.0f} — "
            "tuning had no valid CV folds (check training data / season count)."
        )
    print("Best parameters:")
    for key, value in study.best_params.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")
    if bounds_spec:
        at_edge = _params_at_bounds(study.best_params, bounds_spec)
        if at_edge:
            print(f"  ⚠ best params at bound edge: {', '.join(at_edge)}")
    print("=" * 50)
    return study


def _suggest_elo_params(trial, bounds: AdaptiveParameterBounds | None) -> dict:
    b = bounds
    eb = ELO_PARAM_BOUNDS
    return {
        "k_off": _suggest_float_param(trial, b, "k_off", *eb["k_off"]),
        "k_def": _suggest_float_param(trial, b, "k_def", *eb["k_def"]),
        "elo_scaling": _suggest_int_param(trial, b, "elo_scaling", *eb["elo_scaling"]),
        "home_boost": _suggest_float_param(trial, b, "home_boost", *eb["home_boost"]),
        "offseason_reversion": _suggest_float_param(
            trial, b, "offseason_reversion", *eb["offseason_reversion"],
        ),
        "usage_floor": _suggest_float_param(trial, b, "usage_floor", *eb["usage_floor"]),
        "assist_split": _suggest_float_param(trial, b, "assist_split", *eb["assist_split"]),
        "k_mult_half_life": _suggest_float_param(
            trial, b, "k_mult_half_life", *eb["k_mult_half_life"],
        ),
        "rd_floor": _suggest_float_param(trial, b, "rd_floor", *eb["rd_floor"]),
        "garbage_time_weight": _suggest_float_param(
            trial, b, "garbage_time_weight", *eb["garbage_time_weight"],
        ),
        "clutch_boost": _suggest_float_param(trial, b, "clutch_boost", *eb["clutch_boost"]),
        "tov_penalty": _suggest_float_param(trial, b, "tov_penalty", *eb["tov_penalty"]),
        "foul_draw_boost": _suggest_float_param(trial, b, "foul_draw_boost", *eb["foul_draw_boost"]),
        "variance_dampen": _suggest_float_param(trial, b, "variance_dampen", *eb["variance_dampen"]),
        "xppp_actual_blend": _suggest_float_param(
            trial, b, "xppp_actual_blend", *eb["xppp_actual_blend"],
        ),
        "k_def_events": _suggest_float_param(trial, b, "k_def_events", *eb["k_def_events"]),
        "tov_rate_threshold": _suggest_float_param(
            trial, b, "tov_rate_threshold", *eb["tov_rate_threshold"],
        ),
        "three_pa_rate_threshold": _suggest_float_param(
            trial, b, "three_pa_rate_threshold", *eb["three_pa_rate_threshold"],
        ),
    }


def _suggest_hier_params(trial, bounds: AdaptiveParameterBounds | None) -> dict:
    b = bounds
    hb = HIER_PARAM_BOUNDS
    return {
        "w1": _suggest_float_param(trial, b, "w1", *hb["w1"]),
        "w2": _suggest_float_param(trial, b, "w2", *hb["w2"]),
        "w3": _suggest_float_param(trial, b, "w3", *hb["w3"]),
        "w5": _suggest_float_param(trial, b, "w5", *hb["w5"]),
        "k_off": _suggest_float_param(trial, b, "k_off", *hb["k_off"]),
        "k_def": _suggest_float_param(trial, b, "k_def", *hb["k_def"]),
        "league_rtg": _suggest_float_param(trial, b, "league_rtg", *hb["league_rtg"]),
    }


def tune_elo_tracker(
    stints_df, league_xppp, n_trials=20, bounds: AdaptiveParameterBounds = None,
):
    """
    Optimized Elo Tuner using Chronological Time-Series Cross-Validation.
    Fits ratings on past historical seasons and evaluates strictly on the following season.
    """
    df = _ensure_season_column(stints_df)
    records = _prepare_stint_records(df)
    by_season = _group_records_by_season(records)
    seasons = sorted(by_season.keys())

    def objective(trial):
        params = _suggest_elo_params(trial, bounds)
        config = {
            "K_OFF": params["k_off"], "K_DEF": params["k_def"],
            "ELO_SCALING_FACTOR": params["elo_scaling"],
            "HOME_PPP_BOOST": params["home_boost"],
            "OFFSEASON_REVERSION": params["offseason_reversion"],
            "USAGE_FLOOR": params["usage_floor"],
            "assist_split": params["assist_split"],
            "k_mult_half_life": params["k_mult_half_life"],
            "rd_floor": params["rd_floor"],
            "garbage_time_weight": params["garbage_time_weight"],
            "clutch_boost": params["clutch_boost"],
            "tov_penalty": params["tov_penalty"],
            "foul_draw_boost": params["foul_draw_boost"],
            "variance_dampen": params["variance_dampen"],
            "xppp_actual_blend": params["xppp_actual_blend"],
            "k_def_events": params["k_def_events"],
            "tov_rate_threshold": params["tov_rate_threshold"],
            "three_pa_rate_threshold": params["three_pa_rate_threshold"],
        }
        home_boost = params["home_boost"]
        elo_scaling = params["elo_scaling"]

        def fold_fn(train_recs, val_recs):
            tracker = PlayerRatingTracker(config=config, league_xppp=league_xppp)
            _elo_warmup(tracker, train_recs)
            mae, ats_miss = _elo_eval_fold(
                tracker, val_recs, league_xppp, home_boost, elo_scaling,
            )
            if mae >= TUNING_INVALID_SCORE:
                return TUNING_INVALID_SCORE
            # ATS-aware blend: lower is better
            return 0.5 * mae + 0.3 * (ats_miss * 30.0) + 0.2 * (ats_miss * 20.0)

        return _season_walkforward_mae(seasons, by_season, fold_fn)

    study = _run_optuna_study(objective, n_trials, "ELO TUNING", ELO_PARAM_BOUNDS)
    if study.best_value >= TUNING_INVALID_SCORE:
        print("  ⚠ Elo tuning invalid — using pipeline defaults (not Optuna trial params).")
        return default_elo_config(), study.best_value
    return map_elo_params(study.best_params), study.best_value


def tune_hierarchical(stints_df, n_trials=30, bounds: AdaptiveParameterBounds = None):
    """
    Optimized Hierarchical Tuner using Chronological Time-Series Cross-Validation.
    Fits ratings on past historical seasons and evaluates strictly on the following season.
    """
    df = _ensure_season_column(stints_df)
    records = _prepare_stint_records(df)
    by_season = _group_records_by_season(records)
    seasons = sorted(by_season.keys())

    def objective(trial):
        params = _suggest_hier_params(trial, bounds)
        total = params["w1"] + params["w2"] + params["w3"] + params["w5"]
        if total <= 0:
            return TUNING_INVALID_SCORE

        def fold_fn(train_recs, val_recs):
            engine = HierarchicalPossessionEngine(
                params["w1"] / total, params["w2"] / total,
                params["w3"] / total, params["w5"] / total,
                k_off=params["k_off"], k_def=params["k_def"],
                league_avg_rtg=params["league_rtg"],
            )
            _hier_warmup(engine, train_recs)
            return _hier_eval_fold(engine, val_recs)

        return _season_walkforward_mae(seasons, by_season, fold_fn)

    study = _run_optuna_study(objective, n_trials, "HIERARCHICAL TUNING", HIER_PARAM_BOUNDS)
    if study.best_value >= TUNING_INVALID_SCORE:
        print("  ⚠ Hierarchical tuning invalid — using pipeline defaults.")
        return default_hier_config(), study.best_value
    best = study.best_params.copy()
    best["league_avg_rtg"] = best.pop("league_rtg")
    return best, study.best_value


print("✅ Tuning functions ready.")
