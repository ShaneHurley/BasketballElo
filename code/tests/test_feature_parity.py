"""Train/serve feature parity between generate_features and build_game_features."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.config import ASSIST_SPLIT, DEFAULT_LEAGUE_XPPP, MODERN_ODDS_PATH, PBP_2026_PATH, name_to_id
from pipeline.features import generate_features
from pipeline.game_features import build_game_features
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.ingest import convert_new_pbp
from pipeline.market import load_modern_odds
from pipeline.preprocess import preprocess_pbp
from pipeline.ratings import PlayerRatingTracker
from pipeline.stints import build_stints
from pipeline.trackers import PaceTracker, TeamXpppTracker
from pipeline.teamstats import TeamFormTracker


def _load_sample_stints(max_rows=80000):
    raw = pd.read_csv(PBP_2026_PATH, nrows=max_rows, low_memory=False)
    df = convert_new_pbp(raw, name_to_id=name_to_id)
    df = preprocess_pbp(df, compute_xpoints=True)
    st = build_stints(df, assist_split=ASSIST_SPLIT)
    st["season"] = 2026
    st["game_date"] = pd.to_datetime(st["game_date"], errors="coerce")
    return st.sort_values(["game_date", "GAME_ID", "stint_id"]).reset_index(drop=True)


def test_feature_parity_first_game():
    """After identical pre-game state, both paths produce matching feature columns."""
    st = _load_sample_stints()
    odds = load_modern_odds(str(MODERN_ODDS_PATH))

    game_ids = st.drop_duplicates("GAME_ID").sort_values("game_date")["GAME_ID"].values[:5]
    subset = st[st["GAME_ID"].isin(game_ids)].copy()

    hier = HierarchicalPossessionEngine()
    elo = PlayerRatingTracker(league_xppp=DEFAULT_LEAGUE_XPPP)
    pace = PaceTracker(team_window=10, league_window=100)
    xppp = TeamXpppTracker(window_size=40, prev_season_weight=0.5)
    form = TeamFormTracker(window=15, prev_season_weight=0.4)

    feats_df = generate_features(
        subset, hier, elo, pace, odds_dict=odds, update_engines=True,
        team_xppp_tracker=xppp, team_form_tracker=form,
    )

    # Rebuild first game with build_game_features on fresh trackers replayed to that point
    hier2 = HierarchicalPossessionEngine()
    elo2 = PlayerRatingTracker(league_xppp=DEFAULT_LEAGUE_XPPP)
    pace2 = PaceTracker(team_window=10, league_window=100)
    xppp2 = TeamXpppTracker(window_size=40, prev_season_weight=0.5)
    form2 = TeamFormTracker(window=15, prev_season_weight=0.4)

    from collections import defaultdict, deque
    from pipeline.config import SOS_WINDOW, TEAM_MAP
    from pipeline.utils import _parse_player_string
    from pipeline.teamstats import precompute_game_team_stats

    last_date = {}
    team_game_dates = defaultdict(lambda: deque(maxlen=20))
    team_recent_net = defaultdict(lambda: deque(maxlen=10))
    opponent_history = defaultdict(lambda: deque(maxlen=SOS_WINDOW))
    team_home_margin = defaultdict(lambda: deque(maxlen=20))
    team_road_margin = defaultdict(lambda: deque(maxlen=20))
    team_games_played = defaultdict(int)
    team_rosters_seen = defaultdict(set)
    lineup_cache = {}
    season_start_date = None

    game_stats_map = precompute_game_team_stats(subset)
    target_gid = game_ids[2]
    built = None

    for game_id, group in subset.groupby("GAME_ID", sort=False):
        gdate = group["game_date"].iloc[0]
        home = TEAM_MAP.get(group["home_team"].iloc[0], group["home_team"].iloc[0])
        away = TEAM_MAP.get(group["away_team"].iloc[0], group["away_team"].iloc[0])
        # Task 031: mirror the fixed `generate_features`/`run_simulation`
        # lineup resolution exactly -- never fall back to this game's own
        # actual/current-game PBP lineup for the T-60 feature build. Only
        # `lineup_cache` (populated post-game, from strictly earlier games)
        # may supply a projected roster; otherwise it is genuinely unknown.
        home_starters = lineup_cache.get(home, [])
        away_starters = lineup_cache.get(away, [])
        actual_home_starters = _parse_player_string(group.iloc[0].get("HOME_players", ""))
        actual_away_starters = _parse_player_string(group.iloc[0].get("AWAY_players", ""))

        if season_start_date is None:
            season_start_date = gdate

        built = build_game_features(
            game_id=game_id,
            gdate=gdate,
            home_team=home,
            away_team=away,
            home_starters=home_starters,
            away_starters=away_starters,
            elo_tracker=elo2,
            hier_engine=hier2,
            pace_tracker=pace2,
            odds_dict=odds,
            team_xppp_tracker=xppp2,
            team_form_tracker=form2,
            last_date=last_date,
            team_game_dates=team_game_dates,
            team_recent_net=team_recent_net,
            opponent_history=opponent_history,
            team_home_margin=team_home_margin,
            team_road_margin=team_road_margin,
            team_games_played=team_games_played,
            team_rosters_seen=team_rosters_seen,
            season_start_date=season_start_date,
        )

        if game_id == target_gid:
            break

        from pipeline.game_updates import update_trackers_after_game
        gs = game_stats_map.get(game_id, {})
        act_h = float(gs.get("act_h", 0.0))
        act_a = float(gs.get("act_a", 0.0))
        home_tot_poss = float(gs.get("home_poss", 0.0)) or 50.0
        away_tot_poss = float(gs.get("away_poss", 0.0)) or 50.0
        home_xppp = float(gs.get("home_xpts", 0.0)) / home_tot_poss if home_tot_poss else 1.0
        away_xppp = float(gs.get("away_xpts", 0.0)) / away_tot_poss if away_tot_poss else 1.0
        current_season = gdate.year + (1 if gdate.month >= 9 else 0)
        update_trackers_after_game(
            group=group, gs=gs, game_id=game_id, gdate=gdate,
            home=home, away=away,
            home_starters=actual_home_starters, away_starters=actual_away_starters,
            act_h=act_h, act_a=act_a,
            home_tot_poss=home_tot_poss, away_tot_poss=away_tot_poss,
            home_xppp_game=home_xppp, away_xppp_game=away_xppp,
            current_season=current_season,
            market_spread=built.get("market_spread", np.nan),
            elo_tracker=elo2, hier_engine=hier2, pace_tracker=pace2,
            team_xppp_tracker=xppp2, team_form_tracker=form2,
            last_game_date=last_date, team_game_dates=team_game_dates,
            team_recent_net=team_recent_net, opponent_history=opponent_history,
            team_home_margin=team_home_margin, team_road_margin=team_road_margin,
            team_games_played=team_games_played, team_rosters_seen=team_rosters_seen,
            lineup_cache=lineup_cache,
            ho_off=built.get("h_elo_off"), ho_def=built.get("h_elo_def"),
            ao_off=built.get("a_elo_off"), ao_def=built.get("a_elo_def"),
        )

    row = feats_df[feats_df["GAME_ID"] == target_gid].iloc[0]
    skip = {"actual_home", "actual_away", "actual_margin", "actual_margin_capped",
            "actual_total", "home_win", "GAME_ID", "game_date", "home_team", "away_team"}
    compare_cols = [c for c in built.keys() if c in row.index and c not in skip]
    deltas = []
    for c in compare_cols:
        a, b = float(row[c]) if pd.notna(row[c]) else np.nan, float(built[c]) if pd.notna(built[c]) else np.nan
        if pd.isna(a) and pd.isna(b):
            continue
        if pd.isna(a) or pd.isna(b):
            deltas.append((c, a, b, np.nan))
        else:
            deltas.append((c, a, b, abs(a - b)))

    bad = [(c, d) for c, _, _, d in deltas if pd.notna(d) and d > 1e-4]
    assert not bad, f"Feature parity failures: {bad[:10]}"

    for col in ("h_fatigue_index", "a_fatigue_index", "fatigue_diff"):
        assert col in built, f"missing fatigue feature {col}"
        assert col in row.index, f"missing fatigue column in generate_features: {col}"

    print(f"Feature parity OK for game {target_gid} ({len(compare_cols)} columns checked)")


if __name__ == "__main__":
    test_feature_parity_first_game()
