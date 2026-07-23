"""Rolling team trackers."""
from pipeline.config import DEFAULT_LEAGUE_XPPP

# Cell 8b – Team xPPP Tracker (rolling 40 games, previous season weight 0.5)
from collections import defaultdict, deque

class TeamXpppTracker:
    """
    Maintains rolling offensive and defensive xPPP (expected points per possession)
    for each team, using a weighted window:
      - last 40 games: weight 1.0
      - previous season games: weight 0.5
    """
    def __init__(self, window_size=40, prev_season_weight=0.5):
        self.window_size = window_size
        self.prev_season_weight = prev_season_weight
        # For each team, store deque of (game_date, season, off_xppp, def_xppp)
        self.history = defaultdict(lambda: deque(maxlen=window_size))

    def update(self, team, game_date, season, off_xppp, def_xppp):
        """
        Add a game's team-level offensive and defensive xPPP.
        off_xppp = total_offensive_xPoints / total_offensive_possessions
        def_xppp = total_defensive_xPoints_allowed / total_defensive_possessions
        """
        self.history[team].append((game_date, season, off_xppp, def_xppp))

    def get_rolling_xppp(self, team, current_season, current_date):
        """
        Returns (rolling_off_xppp, rolling_def_xppp) using weighted average.
        Games in current season: weight 1.0.
        Games in previous season: weight prev_season_weight.
        Only games before current_date are considered.
        """
        if team not in self.history:
            return DEFAULT_LEAGUE_XPPP, DEFAULT_LEAGUE_XPPP

        total_weight = 0.0
        sum_off = 0.0
        sum_def = 0.0

        for (game_date, season, off_xppp, def_xppp) in self.history[team]:
            if game_date >= current_date:
                continue   # never use future data
            if season == current_season:
                w = 1.0
            else:
                w = self.prev_season_weight   # previous season
            total_weight += w
            sum_off += off_xppp * w
            sum_def += def_xppp * w

        if total_weight == 0:
            return DEFAULT_LEAGUE_XPPP, DEFAULT_LEAGUE_XPPP

        return sum_off / total_weight, sum_def / total_weight

# Cell 11 – PaceTracker (updated)

import numpy as np
from collections import deque

class PaceTracker:
    """Rolling possession (pace) tracker.

    Tracks each team's OWN per-team possessions (~100) rather than the shared
    whole-game total, so ``get_team_pace`` and the ``pace_diff`` feature reflect
    a team's true tempo. ``get_expected_pace`` returns the expected *total* game
    possessions (~200) so the downstream scale (total head, engine-implied
    margins) is unchanged.
    """
    LEAGUE_TEAM_PACE = 100.0  # per-team possessions baseline

    def __init__(self, team_window=10, league_window=150, per_team=True):
        self.team_window = team_window
        self.league_window = league_window
        # per_team=True: track each team's own possessions (improved default).
        # per_team=False: legacy behaviour (store whole-game totals for both).
        self.per_team = per_team
        # team -> deque of that team's own per-team possessions
        self.team_history = {}
        # league pool of per-team possessions
        self.league_history = deque(maxlen=league_window)

    def _league_pace(self):
        return float(np.mean(self.league_history)) if self.league_history else self.LEAGUE_TEAM_PACE

    def get_team_pace(self, team):
        """Average per-team possessions for a team over its rolling window."""
        hist = self.team_history.get(team)
        if hist and len(hist) > 0:
            return float(np.mean(hist))
        return self.LEAGUE_TEAM_PACE

    def get_expected_pace(self, home_team, away_team):
        """Expected TOTAL game possessions (~200).

        Interaction estimate: per-team game pace ≈ home_pace + away_pace -
        league_pace (so two fast teams play faster, two slow teams slower);
        total possessions ≈ 2 × that.
        """
        h_pace = self.get_team_pace(home_team)
        a_pace = self.get_team_pace(away_team)
        lg_pace = self._league_pace()
        combined = (h_pace + a_pace) - lg_pace
        # per_team stores ~100-scale paces -> double for total game possessions;
        # legacy stores ~200-scale totals -> already total.
        return max(85.0, 2.0 * combined if self.per_team else combined)

    def update_pace(self, home_team, away_team, home_possessions, away_possessions=None):
        """Record possessions for the game.

        per_team=True: store each team's own possessions.
        per_team=False (legacy): store the whole-game total for both teams.
        Back-compatible: if ``away_possessions`` is omitted, treat the single
        value as a per-team estimate (legacy total / 2) for both teams.
        """
        if away_possessions is None:
            home_possessions = away_possessions = float(home_possessions) / 2.0
        home_possessions = float(home_possessions)
        away_possessions = float(away_possessions)
        if not self.per_team:
            total = home_possessions + away_possessions
            home_possessions = away_possessions = total
        if home_team not in self.team_history:
            self.team_history[home_team] = deque(maxlen=self.team_window)
        if away_team not in self.team_history:
            self.team_history[away_team] = deque(maxlen=self.team_window)
        self.team_history[home_team].append(home_possessions)
        self.team_history[away_team].append(away_possessions)
        self.league_history.append(home_possessions)
        self.league_history.append(away_possessions)

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "team_window": self.team_window,
                "league_window": self.league_window,
                "team_history": {k: list(v) for k, v in self.team_history.items()},
                "league_history": list(self.league_history),
            }, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        from collections import deque
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(team_window=data["team_window"], league_window=data["league_window"])
        obj.team_history = {k: deque(v, maxlen=obj.team_window) for k, v in data["team_history"].items()}
        obj.league_history = deque(data["league_history"], maxlen=obj.league_window)
        return obj


class RotationLineupTracker:
    """Rolling player possession share per team for rotation-weighted lineups."""

    def __init__(self, window_games=10, top_n=8):
        self.window_games = window_games
        self.top_n = top_n
        self.history = defaultdict(lambda: deque(maxlen=window_games))

    def update_game(self, team, player_ids, possessions_per_player):
        """Record per-player possessions from one game."""
        self.history[team].append(dict(zip(player_ids, possessions_per_player)))

    def expected_weights(self, team, fallback_ids=None):
        """Return list of (player_id, weight) for top rotation players."""
        if team not in self.history or not self.history[team]:
            if fallback_ids:
                n = max(len(fallback_ids), 1)
                return [(str(p), 1.0 / n) for p in fallback_ids if p]
            return []
        acc = defaultdict(float)
        for game in self.history[team]:
            for pid, poss in game.items():
                acc[str(pid)] += float(poss)
        total = sum(acc.values())
        if total <= 0:
            if fallback_ids:
                n = max(len(fallback_ids), 1)
                return [(str(p), 1.0 / n) for p in fallback_ids if p]
            return []
        ranked = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)[: self.top_n]
        return [(pid, w / total) for pid, w in ranked]

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "window_games": self.window_games,
                "top_n": self.top_n,
                "history": {k: list(v) for k, v in self.history.items()},
            }, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(window_games=data["window_games"], top_n=data["top_n"])
        for k, v in data["history"].items():
            obj.history[k] = deque(v, maxlen=obj.window_games)
        return obj