"""Feature generation for training and inference."""
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pipeline.config import DEFAULT_LEAGUE_XPPP, SOS_WINDOW, TEAM_MAP
from pipeline.dates import coerce_game_date, is_valid_timestamp
from pipeline.game_features import build_game_features
from pipeline.game_updates import update_trackers_after_game
from pipeline.teamstats import precompute_game_team_stats
from pipeline.utils import _parse_player_string


def generate_features(
    stints_df: pd.DataFrame,
    hier_engine,
    elo_tracker,
    pace_tracker,
    odds_dict: dict = None,
    update_engines: bool = True,
    team_xppp_tracker=None,
    team_form_tracker=None,
    rotation_tracker=None,
    sos_window: int = None,
    lineup_elo_tracker=None,
    chemistry_tracker=None,
    team_elo_tracker=None,
    travel_tracker=None,
    epm_tracker=None,
    ref_tracker=None,
    shot_quality_tracker=None,
    hapm_tracker=None,
    elo_calibrator=None,
    hierarchical_pace=None,
    hier_shot_rates=None,
    minutes_model=None,
):
    """
    Generate a feature DataFrame for all games in stints_df, ensuring NO data leakage.

    Uses build_game_features() for train/serve parity, then applies post-game updates.
    """
    # Task 015 + Task 008: coerce mixed ISO/US dates, canonicalize per
    # GAME_ID, sort, then assert monotonic decision timestamps so a shuffled
    # `stints_df` produces identical ordered predictions/tracker state.
    from pipeline.dates import prepare_chronological_stints
    df = prepare_chronological_stints(stints_df, context="generate_features")

    if hierarchical_pace is None:
        try:
            from pipeline.config import USE_HIERARCHICAL_PACE
            if USE_HIERARCHICAL_PACE:
                from pipeline.pace_hierarchy import HierarchicalPaceModel
                hierarchical_pace = HierarchicalPaceModel(baseline=pace_tracker)
        except Exception:
            hierarchical_pace = None
    if hier_shot_rates is None:
        try:
            from pipeline.config import USE_HIERARCHICAL_SHOT_ZONES
            if USE_HIERARCHICAL_SHOT_ZONES:
                from pipeline.shot_hierarchy import HierarchicalZoneRates
                hier_shot_rates = HierarchicalZoneRates()
        except Exception:
            hier_shot_rates = None
    if minutes_model is None:
        try:
            from pipeline.config import USE_ROTATION_SCENARIOS
            if USE_ROTATION_SCENARIOS:
                from pipeline.minutes_hierarchy import HierarchicalMinutesModel
                minutes_model = HierarchicalMinutesModel()
        except Exception:
            minutes_model = None
    rows = []
    last_game_date = {}
    team_game_dates = defaultdict(lambda: deque(maxlen=20))
    season_year = None
    season_start_date = None
    team_games_played = defaultdict(int)
    team_rosters_seen = defaultdict(set)

    RECENT_WINDOW = 10
    team_recent_net = defaultdict(lambda: deque(maxlen=RECENT_WINDOW))
    team_home_margin = defaultdict(lambda: deque(maxlen=20))
    team_road_margin = defaultdict(lambda: deque(maxlen=20))

    if sos_window is None:
        sos_window = SOS_WINDOW
    opponent_history = defaultdict(lambda: deque(maxlen=sos_window))

    lineup_cache = {}
    season_player_ids = set()
    total_games = len(df["GAME_ID"].unique()) if update_engines else 1
    game_stats_map = precompute_game_team_stats(df)

    last_valid_date = None
    skipped_games = 0

    for game_id, group in tqdm(df.groupby("GAME_ID", sort=False), desc="Building features"):
        raw_gdate = group["game_date"].iloc[0]
        season_yr = season_start_date.year if is_valid_timestamp(season_start_date) else None
        gdate = coerce_game_date(
            raw_gdate, game_id=game_id, prev_date=last_valid_date, season_start_year=season_yr,
        )
        if gdate is None:
            skipped_games += 1
            continue
        last_valid_date = gdate
        home_raw = group["home_team"].iloc[0]
        away_raw = group["away_team"].iloc[0]
        home = TEAM_MAP.get(home_raw, home_raw)
        away = TEAM_MAP.get(away_raw, away_raw)

        if isinstance(gdate, pd.Timestamp):
            current_season = gdate.year + (1 if gdate.month >= 9 else 0)
        else:
            current_season = 2025

        if update_engines:
            if season_year is None:
                season_year = gdate.year
            elif gdate.month > 8 and gdate.year > season_year:
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
                season_year = gdate.year
                season_start_date = gdate
                team_games_played.clear()
                team_rosters_seen.clear()
                lineup_cache.clear()
                team_recent_net.clear()
                opponent_history.clear()
                if shot_quality_tracker is not None:
                    for t in list(prior_rosters.keys()):
                        shot_quality_tracker.on_season_boundary(t, current_season - 1)

            if season_start_date is None:
                season_start_date = gdate

        # Task 031: `home_starters`/`away_starters` used for T-60 feature
        # generation below must never be derived from *this* game's own
        # actual/current PBP lineup (Rule 4). `lineup_cache` holds only the
        # actual roster observed in a *previously completed* game (updated
        # post-game in `update_trackers_after_game`); if a team has no prior
        # game in this cache yet (first game of a season), the pre-game
        # roster is genuinely unknown and must be left empty (Rule 5) rather
        # than fabricated from this game's own PBP. Trackers downstream
        # (elo_tracker.lineup_stats, rotation_tracker.expected_weights, ...)
        # already fall back to league-average/empty-weight behavior for an
        # empty roster.
        home_starters = lineup_cache.get(home, [])
        away_starters = lineup_cache.get(away, [])
        # `actual_home_starters`/`actual_away_starters` are this game's real
        # PBP-observed lineup. They are POSTGAME-only information: used to
        # update rosters-seen bookkeeping and `lineup_cache` for *future*
        # games, never fed into this game's own T-60 feature row.
        actual_home_starters = _parse_player_string(group.iloc[0].get("HOME_players", ""))
        actual_away_starters = _parse_player_string(group.iloc[0].get("AWAY_players", ""))

        # Task 014: `act_h`/`act_a` come from `precompute_game_team_stats`,
        # which prefers the canonical, independently-verified final score
        # (Task 006/007) over stint-summed points whenever it is available.
        # This is the single source of truth for `actual_home`/`actual_away`
        # labels below — never re-sum home_pts/away_pts from filtered stints
        # here.
        gs = game_stats_map.get(game_id, {})
        act_h = float(gs.get("act_h", 0.0))
        act_a = float(gs.get("act_a", 0.0))
        tot_poss = float(gs.get("tot_poss", 0.0)) or 100.0
        home_tot_poss = float(gs.get("home_poss", 0.0)) or tot_poss / 2
        away_tot_poss = float(gs.get("away_poss", 0.0)) or tot_poss / 2
        home_tot_xpts = float(gs.get("home_xpts", 0.0))
        away_tot_xpts = float(gs.get("away_xpts", 0.0))
        home_xppp_game = home_tot_xpts / home_tot_poss if home_tot_poss > 0 else DEFAULT_LEAGUE_XPPP
        away_xppp_game = away_tot_xpts / away_tot_poss if away_tot_poss > 0 else DEFAULT_LEAGUE_XPPP

        row_feat = build_game_features(
            game_id=game_id,
            gdate=gdate,
            home_team=home,
            away_team=away,
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
            hierarchical_pace=hierarchical_pace,
            hier_shot_rates=hier_shot_rates,
            minutes_model=minutes_model,
            last_date=last_game_date,
            team_game_dates=team_game_dates,
            team_recent_net=team_recent_net,
            opponent_history=opponent_history,
            team_home_margin=team_home_margin,
            team_road_margin=team_road_margin,
            team_games_played=team_games_played,
            team_rosters_seen=team_rosters_seen,
            season_start_date=season_start_date,
            gs=gs,
            elo_calibrator=elo_calibrator,
        )

        raw_margin = act_h - act_a
        row_feat.update({
            "actual_home": act_h,
            "actual_away": act_a,
            "actual_margin": raw_margin,
            "actual_margin_capped": np.clip(raw_margin, -20.0, 20.0),
            "actual_total": act_h + act_a,
            "home_win": int(act_h > act_a),
        })
        rows.append(row_feat)

        if update_engines:
            seas_prog = len(rows) / max(total_games, 1)
            market_spread = row_feat.get("market_spread", np.nan)
            update_trackers_after_game(
                group=group,
                gs=gs,
                game_id=game_id,
                gdate=gdate,
                home=home,
                away=away,
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
                hapm_tracker=hapm_tracker,
                hierarchical_pace=hierarchical_pace,
                hier_shot_rates=hier_shot_rates,
                minutes_model=minutes_model,
                last_game_date=last_game_date,
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
                ho_off=row_feat.get("h_elo_off"),
                ho_def=row_feat.get("h_elo_def"),
                ao_off=row_feat.get("a_elo_off"),
                ao_def=row_feat.get("a_elo_def"),
            )

    if skipped_games:
        print(f"  ℹ️ features: skipped {skipped_games} games with unrecoverable dates")

    return pd.DataFrame(rows)
