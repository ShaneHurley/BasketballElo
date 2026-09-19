"""Shared post-game tracker updates for training and simulation."""
from __future__ import annotations

from collections import defaultdict

from pipeline.config import DEFAULT_LEAGUE_XPPP
from pipeline.utils import _parse_player_string
from pipeline.teamstats import team_game_form
from pipeline.stint_context import build_stint_context, extract_crew_id


def update_trackers_after_game(
    *,
    group,
    gs,
    game_id,
    gdate,
    home,
    away,
    home_starters,
    away_starters,
    act_h,  # Task 014: canonical final score when available (see teamstats.precompute_game_team_stats); stint-summed fallback otherwise. Never re-derive from filtered stints here.
    act_a,
    home_tot_poss,
    away_tot_poss,
    home_xppp_game,
    away_xppp_game,
    current_season,
    market_spread,
    elo_tracker,
    hier_engine,
    pace_tracker,
    team_xppp_tracker=None,
    team_form_tracker=None,
    rotation_tracker=None,
    lineup_elo_tracker=None,
    chemistry_tracker=None,
    team_elo_tracker=None,
    travel_tracker=None,
    ref_tracker=None,
    shot_quality_tracker=None,
    hapm_tracker=None,
    hierarchical_pace=None,
    hier_shot_rates=None,
    minutes_model=None,
    last_game_date=None,
    team_game_dates=None,
    team_recent_net=None,
    opponent_history=None,
    team_home_margin=None,
    team_road_margin=None,
    team_games_played=None,
    team_rosters_seen=None,
    lineup_cache=None,
    season_player_ids=None,
    seas_prog=0.5,
    ho_off=None,
    ho_def=None,
    ao_off=None,
    ao_def=None,
    h_form=None,
    a_form=None,
):
    """Apply post-game updates shared by generate_features and run_simulation."""
    if team_games_played is not None:
        team_games_played[home] += 1
        team_games_played[away] += 1
    if team_rosters_seen is not None:
        team_rosters_seen[home].update(home_starters)
        team_rosters_seen[away].update(away_starters)

    raw_margin = act_h - act_a
    if team_recent_net is not None:
        team_recent_net[home].append(raw_margin)
        team_recent_net[away].append(-raw_margin)
    if team_home_margin is not None:
        team_home_margin[home].append(raw_margin)
    if team_road_margin is not None:
        team_road_margin[away].append(-raw_margin)

    if opponent_history is not None and ho_off is not None and ao_off is not None:
        from pipeline.config import USE_DECAYED_SOS
        if USE_DECAYED_SOS:
            opponent_history[home].append((gdate, ao_off - ao_def))
            opponent_history[away].append((gdate, ho_off - ho_def))
        else:
            opponent_history[home].append((away, ao_off - ao_def))
            opponent_history[away].append((home, ho_off - ho_def))

    for _, row in group.iterrows():
        hp = _parse_player_string(row.get("HOME_players", ""))
        ap = _parse_player_string(row.get("AWAY_players", ""))
        p = float(row.get("possessions", 1))
        xh = float(row.get("home_xpts", 0))
        xa = float(row.get("away_xpts", 0))
        uh = row.get("home_usage", {}) or {}
        ua = row.get("away_usage", {}) or {}
        per = int(row.get("PERIOD", 1))
        ss = float(row.get("HOME_SCORE_START", 0))
        as_ = float(row.get("AWAY_SCORE_START", 0))
        se = float(row.get("HOME_SCORE_END", 0))
        ae = float(row.get("AWAY_SCORE_END", 0))
        rh = float(row.get("home_pts", 0))
        ra = float(row.get("away_pts", 0))

        if hasattr(elo_tracker, "process_stint"):
            stint_ctx = build_stint_context(row)
            elo_tracker.process_stint(
                ids_A=hp, ids_B=ap, poss=p,
                xpts_A=xh, xpts_B=xa,
                usage_A=uh, usage_B=ua,
                period=per,
                start_A=ss, start_B=as_,
                end_A=se, end_B=ae,
                season_progress=seas_prog,
                stint_ctx=stint_ctx,
            )
        if hasattr(hier_engine, "update"):
            hier_engine.update(hp, ap, rh, ra, p, xpts_off=xh, xpts_def=xa)
        if chemistry_tracker is not None:
            chemistry_tracker.update_stint(hp, ap, xh, xa, p, elo_tracker)
        if lineup_elo_tracker is not None:
            lineup_elo_tracker.update_stint(hp, ap, xh, xa, p, elo_tracker, True)

    if team_xppp_tracker is not None:
        team_xppp_tracker.update(home, gdate, current_season, home_xppp_game, away_xppp_game)
        team_xppp_tracker.update(away, gdate, current_season, away_xppp_game, home_xppp_game)

    if team_form_tracker is not None:
        from pipeline.config import USE_GARBAGE_WEIGHTED_FORM

        form_weight = 1.0
        if USE_GARBAGE_WEIGHTED_FORM and group is not None and "garbage" in group.columns:
            if "possessions" in group.columns:
                tot_p = float(group["possessions"].sum())
                garb_p = float(group.loc[group["garbage"], "possessions"].sum()) if tot_p > 0 else 0.0
                form_weight = max(0.3, 1.0 - garb_p / tot_p) if tot_p > 0 else 1.0
            else:
                form_weight = max(0.3, 1.0 - float(group["garbage"].mean()))
        if h_form is None:
            h_form = team_form_tracker.get(home, current_season, gdate)
        if a_form is None:
            a_form = team_form_tracker.get(away, current_season, gdate)
        team_form_tracker.update(
            home, gdate, current_season, team_game_form(gs, "home"),
            opp_off=a_form.get("off_rtg"), opp_def=a_form.get("def_rtg"),
            weight=form_weight,
        )
        team_form_tracker.update(
            away, gdate, current_season, team_game_form(gs, "away"),
            opp_off=h_form.get("off_rtg"), opp_def=h_form.get("def_rtg"),
            weight=form_weight,
        )

    if team_elo_tracker is not None:
        team_elo_tracker.update(home, away, act_h - act_a, act_h + act_a, market_spread)
    if travel_tracker is not None:
        travel_tracker.update_game(home, away, gdate)

    if last_game_date is not None:
        last_game_date[home] = gdate
        last_game_date[away] = gdate
    if team_game_dates is not None:
        team_game_dates[home].append(gdate)
        team_game_dates[away].append(gdate)

    if hasattr(pace_tracker, "update_pace"):
        pace_tracker.update_pace(home, away, home_tot_poss, away_tot_poss)
    if hierarchical_pace is not None:
        try:
            hierarchical_pace.update(home, away, home_tot_poss, away_tot_poss)
        except Exception:
            pass

    if hier_shot_rates is not None and gs is not None:
        try:
            # Approximate zone updates from team rim/three aggregates when present.
            for side, team in (("home", home), ("away", away)):
                rim = float(gs.get(f"{side}_rim_fga", 0) or 0)
                three = float(gs.get(f"{side}_three_fga", 0) or 0)
                fga = float(gs.get(f"{side}_fga", 0) or 0)
                fgm = float(gs.get(f"{side}_fgm", 0) or 0)
                fg3m = float(gs.get(f"{side}_3pm", 0) or 0)
                # Distribute makes proportionally when zone-level makes absent.
                if rim > 0:
                    made_rim = fgm * (rim / max(fga, 1.0))
                    for _ in range(int(max(rim, 0))):
                        hier_shot_rates.update_shot(
                            zone="restricted",
                            made=(_ < made_rim),
                            team=team,
                        )
                if three > 0:
                    for _ in range(int(max(three, 0))):
                        hier_shot_rates.update_shot(
                            zone="abovebreak3",
                            made=(_ < fg3m),
                            team=team,
                        )
        except Exception:
            pass

    if minutes_model is not None and rotation_tracker is not None:
        try:
            # Record active minutes proxies from rotation possession shares.
            for team, key in ((home, "HOME_players"), (away, "AWAY_players")):
                poss = defaultdict(float)
                for _, row in group.iterrows():
                    p = float(row.get("possessions", 0) or 0)
                    for pid in _parse_player_string(row.get(key, "")):
                        poss[str(pid)] += p
                if not poss:
                    continue
                max_p = max(poss.values()) or 1.0
                outs = 0
                for pid, p in poss.items():
                    # Map possessions → rough minutes (48 * share of max rotation).
                    mins = 36.0 * (p / max_p)
                    active = mins > 0.5
                    minutes_model.record_game(pid, mins, active, team=team)
                    if not active:
                        outs += 1
                if hasattr(minutes_model, "record_team_rest_night"):
                    minutes_model.record_team_rest_night(outs, roster_size=max(len(poss), 1))
        except Exception:
            pass

    if ref_tracker is not None:
        crew_id = extract_crew_id(group)
        tot_fta = float(gs.get("home_fta", 0) or 0) + float(gs.get("away_fta", 0) or 0)
        tot_poss = float(gs.get("tot_poss", 0) or 0) or (home_tot_poss + away_tot_poss)
        if crew_id:
            ref_tracker.update(crew_id, act_h + act_a, tot_fta, tot_poss)

    if shot_quality_tracker is not None and gs:
        from pipeline.shot_quality import shot_quality_stats_from_gs
        shot_quality_tracker.update(home, gdate, current_season, shot_quality_stats_from_gs(gs, "home"))
        shot_quality_tracker.update(away, gdate, current_season, shot_quality_stats_from_gs(gs, "away"))

    if rotation_tracker is not None:
        h_poss = defaultdict(float)
        a_poss = defaultdict(float)
        for _, row in group.iterrows():
            p = float(row.get("possessions", 0))
            for pid in _parse_player_string(row.get("HOME_players", "")):
                h_poss[str(pid)] += p
            for pid in _parse_player_string(row.get("AWAY_players", "")):
                a_poss[str(pid)] += p
        if h_poss:
            rotation_tracker.update_game(home, list(h_poss.keys()), list(h_poss.values()))
        if a_poss:
            rotation_tracker.update_game(away, list(a_poss.keys()), list(a_poss.values()))
        if season_player_ids is not None:
            for pid in h_poss:
                season_player_ids.add(str(pid))
            for pid in a_poss:
                season_player_ids.add(str(pid))

    if lineup_cache is not None:
        actual_first = group.iloc[0]
        lineup_cache[home] = _parse_player_string(actual_first.get("HOME_players", ""))
        lineup_cache[away] = _parse_player_string(actual_first.get("AWAY_players", ""))
