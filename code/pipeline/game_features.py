"""Shared pre-game feature builder for training, simulation, and live prediction."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

from pipeline.config import ALTITUDE_TEAMS, DEFAULT_LEAGUE_XPPP, LEAGUE_AVG_TOTAL, SOS_WINDOW
from pipeline.fatigue import fatigue_features
from pipeline.dates import coerce_game_date, is_valid_timestamp
from pipeline.feature_utils import (
    engine_implied_margins,
    schedule_density_extended,
    rest_bucket_flags,
    _top_lineup,
)
from pipeline.market import get_game_odds, market_microstructure_features, fair_home_win_prob
from pipeline.teamstats import form_feature_dict, _DEFAULT_FORM
from pipeline.availability import missing_rotation_share, star_out_flag, filter_available_weights
from pipeline.elo_calibration import augment_elo_features
from pipeline.lineup_composite import composite_lineup_rating


def _compute_sos(history, ref_date=None) -> float:
    """Mean opponent net rating; optional exponential decay when USE_DECAYED_SOS."""
    if not history:
        return 0.0
    from pipeline.config import USE_DECAYED_SOS, SOS_DECAY_LAMBDA

    weighted_sum = 0.0
    weight_total = 0.0
    for entry in history:
        if len(entry) < 2:
            continue
        first, net = entry[0], float(entry[1])
        w = 1.0
        if USE_DECAYED_SOS and ref_date is not None and hasattr(first, "year"):
            try:
                days = max(0, (pd.Timestamp(ref_date) - pd.Timestamp(first)).days)
                w = float(np.exp(-SOS_DECAY_LAMBDA * days))
            except Exception:
                w = 1.0
        weighted_sum += net * w
        weight_total += w
    if weight_total <= 0:
        return float(np.mean([float(e[1]) for e in history if len(e) >= 2]))
    return float(weighted_sum / weight_total)


def build_game_features(
    *,
    game_id,
    gdate,
    home_team,
    away_team,
    home_starters,
    away_starters,
    elo_tracker,
    hier_engine,
    pace_tracker,
    odds_dict=None,
    team_xppp_tracker=None,
    team_form_tracker=None,
    rotation_tracker=None,
    lineup_elo_tracker=None,
    chemistry_tracker=None,
    team_elo_tracker=None,
    travel_tracker=None,
    epm_tracker=None,
    ref_tracker=None,
    shot_quality_tracker=None,
    hapm_tracker=None,
    last_date=None,
    team_game_dates=None,
    team_recent_net=None,
    opponent_history=None,
    team_home_margin=None,
    team_road_margin=None,
    team_games_played=None,
    team_rosters_seen=None,
    season_start_date=None,
    inactive_ids=None,
    crew_id=None,
    gs=None,
    elo_calibrator=None,
):
    """Build a single-game feature dict matching generate_features / run_simulation."""
    last_date = last_date or {}
    team_game_dates = team_game_dates or defaultdict(lambda: deque(maxlen=20))
    team_recent_net = team_recent_net or defaultdict(lambda: deque(maxlen=10))
    opponent_history = opponent_history or defaultdict(lambda: deque(maxlen=SOS_WINDOW))
    team_home_margin = team_home_margin or defaultdict(lambda: deque(maxlen=20))
    team_road_margin = team_road_margin or defaultdict(lambda: deque(maxlen=20))
    team_games_played = team_games_played or defaultdict(int)
    team_rosters_seen = team_rosters_seen or defaultdict(set)
    inactive_ids = set(str(x) for x in (inactive_ids or []))

    prev_hint = last_date.get(home_team) or last_date.get(away_team)
    season_yr = None
    if is_valid_timestamp(season_start_date):
        season_yr = season_start_date.year
    coerced = coerce_game_date(
        gdate, game_id=game_id, prev_date=prev_hint, season_start_year=season_yr,
    )
    if coerced is not None:
        gdate = coerced
    elif not is_valid_timestamp(gdate):
        gdate = pd.Timestamp.now().normalize()
    else:
        gdate = pd.Timestamp(gdate).normalize()

    current_season = gdate.year + (1 if gdate.month >= 9 else 0)

    h_weights = rotation_tracker.expected_weights(home_team, home_starters) if rotation_tracker else []
    a_weights = rotation_tracker.expected_weights(away_team, away_starters) if rotation_tracker else []

    if inactive_ids:
        h_weights = filter_available_weights(h_weights, inactive_ids)
        a_weights = filter_available_weights(a_weights, inactive_ids)

    home_lineup = _top_lineup(h_weights) or home_starters
    away_lineup = _top_lineup(a_weights) or away_starters

    if hasattr(elo_tracker, "apply_inactivity_decay"):
        elo_tracker.apply_inactivity_decay(home_lineup + away_lineup, gdate.date())

    def _prev_ts(team):
        v = last_date.get(team)
        if is_valid_timestamp(v):
            return pd.Timestamp(v)
        return gdate - pd.Timedelta(days=7)

    h_rest = min((gdate - _prev_ts(home_team)).days, 14)
    a_rest = min((gdate - _prev_ts(away_team)).days, 14)

    sq_h = sq_a = None
    if shot_quality_tracker is not None:
        sq_h = shot_quality_tracker.get(home_team, current_season, gdate)
        sq_a = shot_quality_tracker.get(away_team, current_season, gdate)
    opp_rim_for_home_d = float(sq_a["rim_rate"]) if sq_a else 0.30
    opp_peri_for_home_d = float(sq_a["three_rate"]) if sq_a else 0.38
    opp_rim_for_away_d = float(sq_h["rim_rate"]) if sq_h else 0.30
    opp_peri_for_away_d = float(sq_h["three_rate"]) if sq_h else 0.38

    if h_weights:
        ho_off, ho_def, ho_exp = elo_tracker.weighted_lineup_stats(
            h_weights, opp_rim_rate=opp_rim_for_home_d, opp_three_rate=opp_peri_for_home_d,
        )
        h_o_rd, h_d_rd = elo_tracker.weighted_lineup_uncertainty(h_weights)
    else:
        ho_off, ho_def, ho_exp = elo_tracker.lineup_stats(
            home_starters, opp_rim_rate=opp_rim_for_home_d, opp_three_rate=opp_peri_for_home_d,
        )
        h_o_rd, h_d_rd = elo_tracker.lineup_uncertainty(home_starters)
    if a_weights:
        ao_off, ao_def, ao_exp = elo_tracker.weighted_lineup_stats(
            a_weights, opp_rim_rate=opp_rim_for_away_d, opp_three_rate=opp_peri_for_away_d,
        )
        a_o_rd, a_d_rd = elo_tracker.weighted_lineup_uncertainty(a_weights)
    else:
        ao_off, ao_def, ao_exp = elo_tracker.lineup_stats(
            away_starters, opp_rim_rate=opp_rim_for_away_d, opp_three_rate=opp_peri_for_away_d,
        )
        a_o_rd, a_d_rd = elo_tracker.lineup_uncertainty(away_starters)

    h_rim_def, h_peri_def, _ = elo_tracker._lineup_rim_peri(home_lineup, h_weights or None)
    a_rim_def, a_peri_def, _ = elo_tracker._lineup_rim_peri(away_lineup, a_weights or None)

    if epm_tracker is not None:
        # Task 032: `as_of=gdate` -- a future EPM snapshot must never be
        # joinable to this (earlier) game's T-60 features.
        ho_off = epm_tracker.blend_lineup_off(home_lineup, ho_off, as_of=gdate)
        ao_off = epm_tracker.blend_lineup_off(away_lineup, ao_off, as_of=gdate)

    h_hoff, h_hdef = hier_engine.lineup_rating(home_lineup)
    a_hoff, a_hdef = hier_engine.lineup_rating(away_lineup)

    if team_xppp_tracker is not None:
        h_roll_off, h_roll_def = team_xppp_tracker.get_rolling_xppp(home_team, current_season, gdate)
        a_roll_off, a_roll_def = team_xppp_tracker.get_rolling_xppp(away_team, current_season, gdate)
    else:
        h_roll_off = h_roll_def = a_roll_off = a_roll_def = DEFAULT_LEAGUE_XPPP

    pred_poss = pace_tracker.get_expected_pace(home_team, away_team)
    h_pace = pace_tracker.get_team_pace(home_team) if hasattr(pace_tracker, "get_team_pace") else pred_poss
    a_pace = pace_tracker.get_team_pace(away_team) if hasattr(pace_tracker, "get_team_pace") else pred_poss
    pace_diff = h_pace - a_pace

    h_games_last7, h_3in4, h_4in6 = schedule_density_extended(team_game_dates[home_team], gdate)
    a_games_last7, a_3in4, a_4in6 = schedule_density_extended(team_game_dates[away_team], gdate)

    h_new_starters = len(set(home_starters) - team_rosters_seen[home_team]) / max(len(home_starters), 1)
    a_new_starters = len(set(away_starters) - team_rosters_seen[away_team]) / max(len(away_starters), 1)

    h_missing_rotation = missing_rotation_share(h_weights, inactive_ids)
    a_missing_rotation = missing_rotation_share(a_weights, inactive_ids)
    h_star_out = star_out_flag(h_weights, inactive_ids)
    a_star_out = star_out_flag(a_weights, inactive_ids)

    days_since_start = (gdate - season_start_date).days if season_start_date is not None else 50
    h_season_phase = team_games_played[home_team] / 82.0
    a_season_phase = team_games_played[away_team] / 82.0

    go = get_game_odds(gdate, home_team, odds_dict) if odds_dict else {}
    market_spread = go.get("spread", np.nan)
    market_ml = go.get("ml", np.nan)
    market_fair_win_prob = fair_home_win_prob(market_ml) if pd.notna(market_ml) else 0.5
    market_total = go.get("total", np.nan)
    closing_spread = go.get("closing_spread", market_spread)
    market_total_minus_league = float(market_total) - LEAGUE_AVG_TOTAL if pd.notna(market_total) else 0.0
    spread_move = go.get("spread_move", 0.0) if pd.notna(go.get("spread_move", np.nan)) else 0.0
    public_home_pct = go.get("public_home_pct", 0.0) if pd.notna(go.get("public_home_pct", np.nan)) else 0.0
    micro = market_microstructure_features(spread_move, public_home_pct, market_spread)

    h_recent = np.mean(team_recent_net[home_team]) if team_recent_net[home_team] else 0.0
    a_recent = np.mean(team_recent_net[away_team]) if team_recent_net[away_team] else 0.0

    h_sos = _compute_sos(opponent_history[home_team], gdate)
    a_sos = _compute_sos(opponent_history[away_team], gdate)

    h_home_edge = (np.mean(team_home_margin[home_team]) - np.mean(team_road_margin[home_team])) \
        if team_home_margin[home_team] and team_road_margin[home_team] else 0.0
    a_road_edge = (np.mean(team_road_margin[away_team]) - np.mean(team_home_margin[away_team])) \
        if team_home_margin[away_team] and team_road_margin[away_team] else 0.0

    if team_form_tracker is not None:
        h_form = team_form_tracker.get(home_team, current_season, gdate)
        a_form = team_form_tracker.get(away_team, current_season, gdate)
        form_feats = form_feature_dict(h_form, a_form)
        if hasattr(team_form_tracker, "multi_window_feature_dict"):
            form_feats.update(
                team_form_tracker.multi_window_feature_dict(
                    home_team, away_team, current_season, gdate,
                )
            )
    else:
        h_form, a_form = dict(_DEFAULT_FORM), dict(_DEFAULT_FORM)
        form_feats = form_feature_dict(h_form, a_form)

    elo_margin, hier_margin = engine_implied_margins(
        elo_tracker, hier_engine, ho_off, ho_def, ao_off, ao_def,
        home_lineup, away_lineup, pred_poss)

    h_luck, h_defev, h_tov = elo_tracker.lineup_rolling_rates(home_lineup, h_weights or None)
    a_luck, a_defev, a_tov = elo_tracker.lineup_rolling_rates(away_lineup, a_weights or None)
    elo_luck_adj_net = (h_luck - a_luck) * 100.0
    elo_def_event_rate = h_defev - a_defev
    elo_tov_rate = h_tov - a_tov
    elo_matchup_asym = (ho_off - ao_def) - (ao_off - ho_def)
    mean_unc = (h_o_rd + h_d_rd + a_o_rd + a_d_rd) / 4.0
    elo_consistency = 1.0 / (mean_unc + 1e-6)
    elo_margin_z = elo_margin / max(pred_poss, 1.0)

    feat = {
        "GAME_ID": game_id,
        "game_date": gdate,
        "home_team": home_team,
        "away_team": away_team,
        "elo_margin": elo_margin,
        "hier_margin": hier_margin,
        "h_elo_off": ho_off, "h_elo_def": ho_def,
        "a_elo_off": ao_off, "a_elo_def": ao_def,
        "h_elo_def_rim": h_rim_def, "h_elo_def_peri": h_peri_def,
        "a_elo_def_rim": a_rim_def, "a_elo_def_peri": a_peri_def,
        "rim_def_diff": h_rim_def - a_rim_def,
        "peri_def_diff": h_peri_def - a_peri_def,
        "elo_diff_off": ho_off - ao_off, "elo_diff_def": ho_def - ao_def,
        "elo_net": (ho_off - ao_def) - (ao_off - ho_def),
        "elo_matchup_asym": elo_matchup_asym,
        "elo_luck_adj_net": elo_luck_adj_net,
        "elo_def_event_rate": elo_def_event_rate,
        "elo_tov_rate": elo_tov_rate,
        "elo_consistency": elo_consistency,
        "elo_margin_z": elo_margin_z,
        "h_hier_off": h_hoff, "h_hier_def": h_hdef,
        "a_hier_off": a_hoff, "a_hier_def": a_hdef,
        "hier_net": (h_hoff - a_hdef) - (a_hoff - h_hdef),
        "exp_poss": pred_poss,
        "h_rest": h_rest, "a_rest": a_rest,
        "h_b2b": int(h_rest <= 1), "a_b2b": int(a_rest <= 1),
        "h_rest_0": rest_bucket_flags(h_rest)["rest_0"],
        "h_rest_1": rest_bucket_flags(h_rest)["rest_1"],
        "h_rest_2": rest_bucket_flags(h_rest)["rest_2"],
        "h_rest_3plus": rest_bucket_flags(h_rest)["rest_3plus"],
        "a_rest_0": rest_bucket_flags(a_rest)["rest_0"],
        "a_rest_1": rest_bucket_flags(a_rest)["rest_1"],
        "a_rest_2": rest_bucket_flags(a_rest)["rest_2"],
        "a_rest_3plus": rest_bucket_flags(a_rest)["rest_3plus"],
        "h_games_last7": h_games_last7, "a_games_last7": a_games_last7,
        "games_last7_diff": h_games_last7 - a_games_last7,
        "h_3in4": h_3in4, "a_3in4": a_3in4,
        "h_4in6": h_4in6, "a_4in6": a_4in6,
        "is_altitude": int(home_team in ALTITUDE_TEAMS),
        "h_experience": ho_exp, "a_experience": ao_exp,
        "days_since_season_start": days_since_start,
        "h_season_phase": h_season_phase, "a_season_phase": a_season_phase,
        "h_new_starters": h_new_starters, "a_new_starters": a_new_starters,
        "h_missing_rotation": h_missing_rotation, "a_missing_rotation": a_missing_rotation,
        "h_star_out": h_star_out, "a_star_out": a_star_out,
        "h_rating_uncertainty": h_o_rd + h_d_rd,
        "a_rating_uncertainty": a_o_rd + a_d_rd,
        "uncertainty_diff": (h_o_rd + h_d_rd) - (a_o_rd + a_d_rd),
        "h_roll_off_xppp": h_roll_off, "h_roll_def_xppp": h_roll_def,
        "a_roll_off_xppp": a_roll_off, "a_roll_def_xppp": a_roll_def,
        "roll_net_xppp": (h_roll_off - a_roll_def) - (a_roll_off - h_roll_def),
        "market_spread": market_spread, "market_ml": market_ml,
        "market_fair_win_prob": market_fair_win_prob,
        "closing_spread": closing_spread,
        "market_total": market_total if pd.notna(market_total) else 0.0,
        "market_total_minus_league": market_total_minus_league,
        "spread_move": spread_move, "public_home_pct": public_home_pct,
        "h_recent_net": h_recent, "a_recent_net": a_recent, "recent_diff": h_recent - a_recent,
        "pace_diff": pace_diff, "pace_abs_diff": abs(pace_diff),
        "pace_interaction": pace_diff * ((ho_off - ao_def) - (ao_off - ho_def)),
        "h_pace": float(h_pace), "a_pace": float(a_pace),
        "h_sos": h_sos, "a_sos": a_sos, "sos_diff": h_sos - a_sos,
        "h_home_edge": h_home_edge, "a_road_edge": a_road_edge,
        "hca_net": h_home_edge - a_road_edge,
        **form_feats,
        **micro,
    }

    if lineup_elo_tracker is not None:
        feat.update(lineup_elo_tracker.feature_dict(home_lineup, away_lineup, elo_tracker))
    if chemistry_tracker is not None:
        chem_feats = chemistry_tracker.feature_dict(home_lineup, away_lineup, elo_tracker)
        feat.update(chem_feats)
        feat["usage_conflict"] = chemistry_tracker.usage_conflict(h_weights) + chemistry_tracker.usage_conflict(a_weights)
        lineup5_net = feat.get("lineup5_net", 0.0)
        feat["h_lineup_composite"] = composite_lineup_rating(
            player_off_delta=ho_off - ao_def,
            player_def_delta=ho_def - ao_off,
            lineup5_net=lineup5_net,
            chem_duo_net=chem_feats.get("h_chem_duo_net", 0.0),
            chem_trio_net=chem_feats.get("h_chem_trio_net", 0.0),
        )
        feat["a_lineup_composite"] = composite_lineup_rating(
            player_off_delta=ao_off - ho_def,
            player_def_delta=ao_def - ho_off,
            lineup5_net=-lineup5_net,
            chem_duo_net=chem_feats.get("a_chem_duo_net", 0.0),
            chem_trio_net=chem_feats.get("a_chem_trio_net", 0.0),
        )
        feat["lineup_composite_diff"] = feat["h_lineup_composite"] - feat["a_lineup_composite"]
    if shot_quality_tracker is not None:
        feat.update(shot_quality_tracker.feature_dict(home_team, away_team, current_season, gdate))
    if hapm_tracker is not None and getattr(hapm_tracker, "fitted", False):
        feat.update(hapm_tracker.feature_dict(home_lineup, away_lineup, game_id=game_id))
    else:
        feat.setdefault("hapm_net_diff", 0.0)
    if team_elo_tracker is not None:
        feat.update(team_elo_tracker.feature_dict(home_team, away_team))
    if travel_tracker is not None:
        feat.update(travel_tracker.matchup_features(home_team, away_team, gdate))
        tm = feat
        feat.update(fatigue_features(
            feat.get("h_games_last7", 0), feat.get("a_games_last7", 0),
            tm.get("h_travel_miles_7d", 0), tm.get("a_travel_miles_7d", 0),
            tm.get("h_tz_shift", 0), tm.get("a_tz_shift", 0),
        ))
    else:
        feat.update(fatigue_features(
            feat.get("h_games_last7", 0), feat.get("a_games_last7", 0), 0, 0, 0, 0,
        ))
    # Travel × rest interactions (nonlinear fatigue)
    h_travel = float(feat.get("h_travel_miles_7d", 0) or 0)
    a_travel = float(feat.get("a_travel_miles_7d", 0) or 0)
    feat["h_travel_rest_interaction"] = h_travel * float(feat.get("h_rest_0", 0) + 0.5 * feat.get("h_rest_1", 0))
    feat["a_travel_rest_interaction"] = a_travel * float(feat.get("a_rest_0", 0) + 0.5 * feat.get("a_rest_1", 0))
    feat["travel_rest_interaction_diff"] = (
        feat["h_travel_rest_interaction"] - feat["a_travel_rest_interaction"]
    )
    if epm_tracker is not None:
        # Task 032: `as_of=gdate` enforces the same future-snapshot guard
        # for the emitted `epm_*` feature columns.
        feat.update(epm_tracker.feature_dict(home_lineup, away_lineup, ho_off, ao_off, as_of=gdate))
        # Quantitative availability impact from EPM priors + star-out
        from pipeline.availability import expected_availability_impact
        feat.update(expected_availability_impact(
            home_lineup, away_lineup, epm_tracker,
            h_star_out=feat.get("h_star_out", 0),
            a_star_out=feat.get("a_star_out", 0),
            h_missing=feat.get("h_missing_rotation", 0),
            a_missing=feat.get("a_missing_rotation", 0),
            as_of=gdate,
        ))
    else:
        feat.update({
            "epm_prior_diff": 0.0, "epm_blend_off_diff": 0.0,
            "epm_missing_frac_home": 1.0, "epm_missing_frac_away": 1.0,
            "h_expected_off_drop": 0.0, "a_expected_off_drop": 0.0,
            "h_expected_def_drop": 0.0, "a_expected_def_drop": 0.0,
            "h_expected_pace_delta": 0.0, "a_expected_pace_delta": 0.0,
            "expected_off_drop_diff": 0.0, "expected_def_drop_diff": 0.0,
            "expected_pace_delta_diff": 0.0,
        })
    if ref_tracker is not None:
        feat.update(ref_tracker.features(crew_id))
    else:
        feat.update({"ref_pace_bias": 0.0, "ref_foul_bias": 0.0})

    return augment_elo_features(feat, elo_calibrator)
