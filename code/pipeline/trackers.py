"""Rolling team trackers."""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path

import numpy as np

from pipeline.config import (
    BASE_ELO,
    DEFAULT_LEAGUE_XPPP,
    PACE_OFFSEASON_SHRINK,
    XPPP_TURNOVER_THRESHOLD,
    XPPP_WARM_START_GAMES,
)


class TeamXpppTracker:
    """
    Maintains rolling offensive and defensive xPPP (expected points per possession)
    for each team, using a weighted window:
      - last 40 games: weight 1.0
      - previous season games: weight 0.5

    Elo-derived priors warm-start only when history is empty or roster turnover
    is high; observed game xPPP remains the source of truth as games accumulate.
    """
    def __init__(
        self,
        window_size=40,
        prev_season_weight=0.5,
        warm_start_games: int | None = None,
        turnover_threshold: float | None = None,
    ):
        self.window_size = window_size
        self.prev_season_weight = prev_season_weight
        self.warm_start_games = int(
            XPPP_WARM_START_GAMES if warm_start_games is None else warm_start_games
        )
        self.turnover_threshold = float(
            XPPP_TURNOVER_THRESHOLD if turnover_threshold is None else turnover_threshold
        )
        # For each team, store deque of (game_date, season, off_xppp, def_xppp)
        self.history = defaultdict(lambda: deque(maxlen=window_size))
        self.elo_priors: dict[str, tuple[float, float]] = {}
        self.turnover_frac: dict[str, float] = {}

    def update(self, team, game_date, season, off_xppp, def_xppp):
        """
        Add a game's team-level offensive and defensive xPPP.
        off_xppp = total_offensive_xPoints / total_offensive_possessions
        def_xppp = total_defensive_xPoints_allowed / total_defensive_possessions
        """
        self.history[team].append((game_date, season, off_xppp, def_xppp))

    def set_elo_prior(self, team, off_xppp: float, def_xppp: float) -> None:
        self.elo_priors[str(team)] = (float(off_xppp), float(def_xppp))

    def set_turnover(self, team, frac: float) -> None:
        self.turnover_frac[str(team)] = float(max(0.0, min(1.0, frac)))

    def clear_priors(self) -> None:
        self.elo_priors.clear()
        self.turnover_frac.clear()

    def _observed_rolling(self, team, current_season, current_date):
        """Return (off, def, n_obs) from history strictly before current_date."""
        if team not in self.history or not self.history[team]:
            return None, None, 0

        total_weight = 0.0
        sum_off = 0.0
        sum_def = 0.0
        n_obs = 0

        for (game_date, season, off_xppp, def_xppp) in self.history[team]:
            if current_date is not None and game_date is not None and game_date >= current_date:
                continue
            if season == current_season:
                w = 1.0
            else:
                w = self.prev_season_weight
            total_weight += w
            sum_off += off_xppp * w
            sum_def += def_xppp * w
            n_obs += 1

        if total_weight == 0 or n_obs == 0:
            return None, None, 0
        return sum_off / total_weight, sum_def / total_weight, n_obs

    def get_rolling_xppp(self, team, current_season, current_date):
        """
        Returns (rolling_off_xppp, rolling_def_xppp) using weighted average.
        Games in current season: weight 1.0.
        Games in previous season: weight prev_season_weight.
        Only games before current_date are considered.

        When history is empty or roster turnover is high, blend in an Elo-derived
        prior that decays to zero after ``warm_start_games`` observed games.
        """
        team = str(team) if team is not None else team
        obs_off, obs_def, n_obs = self._observed_rolling(team, current_season, current_date)
        turnover = float(self.turnover_frac.get(team, 0.0))
        prior = self.elo_priors.get(team)
        need_warm = (n_obs == 0) or (turnover >= self.turnover_threshold)

        if not need_warm or prior is None:
            if n_obs == 0 or obs_off is None:
                return DEFAULT_LEAGUE_XPPP, DEFAULT_LEAGUE_XPPP
            return float(obs_off), float(obs_def)

        prior_off, prior_def = prior
        if n_obs == 0 or obs_off is None:
            return float(prior_off), float(prior_def)

        warm = max(1, int(self.warm_start_games))
        w_prior = max(0.0, 1.0 - (n_obs / float(warm)))
        off = w_prior * float(prior_off) + (1.0 - w_prior) * float(obs_off)
        deff = w_prior * float(prior_def) + (1.0 - w_prior) * float(obs_def)
        return off, deff

    def save_state(self, path):
        import pickle
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({
                "window_size": self.window_size,
                "prev_season_weight": self.prev_season_weight,
                "warm_start_games": self.warm_start_games,
                "turnover_threshold": self.turnover_threshold,
                "history": {k: list(v) for k, v in self.history.items()},
                "elo_priors": dict(self.elo_priors),
                "turnover_frac": dict(self.turnover_frac),
            }, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            window_size=data.get("window_size", 40),
            prev_season_weight=data.get("prev_season_weight", 0.5),
            warm_start_games=data.get("warm_start_games"),
            turnover_threshold=data.get("turnover_threshold"),
        )
        obj.history = defaultdict(
            lambda: deque(maxlen=obj.window_size),
            {k: deque(v, maxlen=obj.window_size) for k, v in data.get("history", {}).items()},
        )
        obj.elo_priors = {
            str(k): (float(v[0]), float(v[1])) for k, v in data.get("elo_priors", {}).items()
        }
        obj.turnover_frac = {
            str(k): float(v) for k, v in data.get("turnover_frac", {}).items()
        }
        return obj


def elo_team_xppp_prior(elo_tracker, player_ids, weights=None):
    """Map rotation-weighted player Elo vs BASE_ELO into team off/def xPPP priors.

    Uses the same scaling as Elo PPP:
      off = league_xppp + (lineup_off - 1500) / ELO_SCALING
      def = league_xppp - (lineup_def - 1500) / ELO_SCALING
    so better defense implies lower points allowed.
    """
    if elo_tracker is None or not player_ids:
        return DEFAULT_LEAGUE_XPPP, DEFAULT_LEAGUE_XPPP
    ids = [str(x) for x in player_ids if x is not None and str(x) != "nan"]
    if not ids:
        return DEFAULT_LEAGUE_XPPP, DEFAULT_LEAGUE_XPPP

    weight_pairs = None
    if weights is not None:
        if isinstance(weights, dict):
            weight_pairs = [(str(p), float(w)) for p, w in weights.items() if w and float(w) > 0]
        else:
            weight_pairs = [(str(p), float(w)) for p, w in weights if w and float(w) > 0]

    off_mu, def_mu, _ = elo_tracker.lineup_stats(ids, weights=weight_pairs)
    elo_cfg = getattr(elo_tracker, "cfg", {}) or {}
    scaling = float(elo_cfg.get("ELO_SCALING_FACTOR", 1000) or 1000)
    lx = float(getattr(elo_tracker, "league_xppp", DEFAULT_LEAGUE_XPPP) or DEFAULT_LEAGUE_XPPP)
    off_xppp = lx + (float(off_mu) - float(BASE_ELO)) / scaling
    def_xppp = lx - (float(def_mu) - float(BASE_ELO)) / scaling
    return float(off_xppp), float(def_xppp)


def _roster_turnover(current_ids, prior_ids) -> float:
    cur = {str(x) for x in (current_ids or []) if x is not None and str(x) != "nan"}
    prior = {str(x) for x in (prior_ids or []) if x is not None and str(x) != "nan"}
    if not cur:
        return 1.0 if prior else 0.0
    if not prior:
        return 1.0
    new = cur - prior
    return len(new) / max(len(cur), 1)


def _shrink_pace_toward_league(pace_tracker, team, factor: float) -> None:
    if pace_tracker is None or not hasattr(pace_tracker, "team_history"):
        return
    hist = pace_tracker.team_history.get(team)
    if not hist:
        return
    league = float(getattr(pace_tracker, "LEAGUE_TEAM_PACE", 100.0))
    factor = float(max(0.0, min(1.0, factor)))
    new_vals = [league + factor * (float(v) - league) for v in hist]
    maxlen = getattr(pace_tracker, "team_window", len(new_vals))
    pace_tracker.team_history[team] = deque(new_vals, maxlen=maxlen)


def refresh_team_priors_for_season(
    team_xppp_tracker: TeamXpppTracker | None,
    elo_tracker=None,
    pace_tracker=None,
    rotation_tracker=None,
    prior_rosters: dict | None = None,
    teams=None,
    *,
    apply_elo_offseason: bool = False,
    returning_player_ids=None,
    pace_shrink: float | None = None,
    lineup_cache: dict | None = None,
) -> dict:
    """Season-boundary refresh: Elo priors, turnover flags, optional pace shrink.

    ``prior_rosters`` should be the previous season's ``team_rosters_seen``
    (or equivalent) *before* it is cleared. Current rotation comes from
    ``rotation_tracker.expected_weights`` with ``lineup_cache`` as fallback.
    """
    summary = {"teams": 0, "high_turnover": 0, "priors_set": 0}
    if team_xppp_tracker is None and elo_tracker is None and pace_tracker is None:
        return summary

    if apply_elo_offseason and elo_tracker is not None and hasattr(elo_tracker, "offseason_revert"):
        elo_tracker.offseason_revert(returning_player_ids=returning_player_ids)

    shrink = float(PACE_OFFSEASON_SHRINK if pace_shrink is None else pace_shrink)
    prior_rosters = prior_rosters or {}
    lineup_cache = lineup_cache or {}

    team_set = set()
    if teams is not None:
        team_set.update(str(t) for t in teams)
    team_set.update(str(t) for t in prior_rosters.keys())
    if rotation_tracker is not None and hasattr(rotation_tracker, "history"):
        team_set.update(str(t) for t in rotation_tracker.history.keys())
    if pace_tracker is not None and hasattr(pace_tracker, "team_history"):
        team_set.update(str(t) for t in pace_tracker.team_history.keys())
    if team_xppp_tracker is not None:
        team_set.update(str(t) for t in team_xppp_tracker.history.keys())
        team_set.update(str(t) for t in team_xppp_tracker.elo_priors.keys())
    team_set.update(str(t) for t in lineup_cache.keys())

    threshold = float(
        getattr(team_xppp_tracker, "turnover_threshold", XPPP_TURNOVER_THRESHOLD)
        if team_xppp_tracker is not None
        else XPPP_TURNOVER_THRESHOLD
    )

    for team in sorted(team_set):
        summary["teams"] += 1
        current_ids = []
        weights = None
        if rotation_tracker is not None and hasattr(rotation_tracker, "expected_weights"):
            pairs = rotation_tracker.expected_weights(team, fallback_ids=lineup_cache.get(team))
            if pairs:
                current_ids = [pid for pid, _ in pairs]
                weights = pairs
        if not current_ids:
            current_ids = list(lineup_cache.get(team) or [])

        prior_ids = prior_rosters.get(team) or prior_rosters.get(str(team)) or set()
        if not isinstance(prior_ids, (set, list, tuple)):
            prior_ids = set()
        turnover = _roster_turnover(current_ids, prior_ids)
        high = turnover >= threshold
        if high:
            summary["high_turnover"] += 1

        if team_xppp_tracker is not None:
            team_xppp_tracker.set_turnover(team, turnover)
            if current_ids and elo_tracker is not None:
                off_p, def_p = elo_team_xppp_prior(elo_tracker, current_ids, weights=weights)
                team_xppp_tracker.set_elo_prior(team, off_p, def_p)
                summary["priors_set"] += 1

        # Always shrink pace a bit at offseason; high-turnover teams get a second pass.
        _shrink_pace_toward_league(pace_tracker, team, shrink)
        if high:
            _shrink_pace_toward_league(pace_tracker, team, shrink)

    return summary


class PaceTracker:
    """Rolling possession (pace) tracker.

    Tracks each team's OWN per-team possessions (~100) rather than the shared
    whole-game total, so ``get_team_pace`` and the ``pace_diff`` feature reflect
    a team's true tempo.

    ``get_expected_pace`` returns expected *shared* game possessions on the
    ~100 scale used by PPP→points conversion in ``engine_implied_margins``,
    ``matchup_rating``, ``elo_implied_total_from_row``, and Elo
    ``predict_game_margin`` / ``implied_total``. Doubling to ~200 here was a
    scale bug (``exp_poss_scale_mismatch``).
    """
    LEAGUE_TEAM_PACE = 100.0  # per-team / shared possessions baseline
    MIN_SHARED_PACE = 85.0
    MAX_SHARED_PACE = 120.0

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
        """Expected shared possessions (~100) for PPP→points conversion.

        Interaction estimate: shared pace ≈ home_pace + away_pace - league_pace
        (two fast teams play faster, two slow teams slower). Legacy
        ``per_team=False`` histories store ~200 totals and are halved.
        """
        h_pace = self.get_team_pace(home_team)
        a_pace = self.get_team_pace(away_team)
        lg_pace = self._league_pace()
        combined = (h_pace + a_pace) - lg_pace
        # per_team stores ~100-scale paces → shared possessions already.
        # legacy stores ~200-scale totals → convert to shared (~100).
        shared = float(combined if self.per_team else combined / 2.0)
        return float(np.clip(shared, self.MIN_SHARED_PACE, self.MAX_SHARED_PACE))

    def get_expected_pace_distribution(self, home_team, away_team) -> dict:
        """Mean/var/quantiles for shared possessions (baseline rolling model)."""
        mean = self.get_expected_pace(home_team, away_team)
        h_hist = list(self.team_history.get(home_team, []))
        a_hist = list(self.team_history.get(away_team, []))
        samples = []
        if h_hist and a_hist:
            lg = self._league_pace()
            for hp in h_hist[-self.team_window:]:
                for ap in a_hist[-self.team_window:]:
                    raw = (float(hp) + float(ap)) - lg
                    if not self.per_team:
                        raw /= 2.0
                    samples.append(raw)
        if len(samples) >= 4:
            arr = np.asarray(samples, dtype=float)
            var = float(np.var(arr))
            q10, q50, q90 = np.percentile(arr, [10, 50, 90])
        else:
            var = 9.0  # ~3-possession SD cold-start
            q10, q50, q90 = mean - 4.0, mean, mean + 4.0
        return {
            "pace_mean": mean,
            "pace_var": max(var, 1.0),
            "pace_std": float(np.sqrt(max(var, 1.0))),
            "pace_q10": float(np.clip(q10, self.MIN_SHARED_PACE, self.MAX_SHARED_PACE)),
            "pace_q50": float(np.clip(q50, self.MIN_SHARED_PACE, self.MAX_SHARED_PACE)),
            "pace_q90": float(np.clip(q90, self.MIN_SHARED_PACE, self.MAX_SHARED_PACE)),
        }

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
                "per_team": self.per_team,
                "team_history": {k: list(v) for k, v in self.team_history.items()},
                "league_history": list(self.league_history),
            }, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(
            team_window=data["team_window"],
            league_window=data["league_window"],
            per_team=data.get("per_team", True),
        )
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
            state = pickle.load(f)
        obj = cls(window_games=state["window_games"], top_n=state["top_n"])
        for k, v in state.get("history", {}).items():
            obj.history[k] = deque(v, maxlen=obj.window_games)
        return obj


class LeagueRollingStats:
    """Rolling league mean/std for additive ``*_z`` features (leak-free).

    Only past observations are in the buffers at feature time; call
    ``update`` *after* the game is graded / trackers advance.
    """

    DEFAULT_KEYS = ("pace_diff", "elo_net", "roll_net_xppp", "elo_margin")

    def __init__(self, window_games: int | None = None):
        from pipeline.config import LEAGUE_Z_WINDOW_GAMES
        self.window = int(LEAGUE_Z_WINDOW_GAMES if window_games is None else window_games)
        self._bufs: dict[str, deque] = defaultdict(lambda: deque(maxlen=self.window))

    def update(self, **metrics) -> None:
        for k, v in metrics.items():
            try:
                fv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(fv):
                self._bufs[k].append(fv)

    def z(self, name: str, value, *, min_n: int = 30) -> float:
        try:
            fv = float(value)
        except (TypeError, ValueError):
            return 0.0
        if not np.isfinite(fv):
            return 0.0
        buf = self._bufs.get(name)
        if not buf or len(buf) < min_n:
            return 0.0
        arr = np.asarray(buf, dtype=float)
        sd = float(arr.std())
        if sd < 1e-9:
            return 0.0
        return float((fv - float(arr.mean())) / sd)

    def feature_dict(self, raw: dict, *, keys: tuple[str, ...] | None = None) -> dict:
        """Emit ``{name}_lz`` (league-z) to avoid colliding with existing ``elo_margin_z``."""
        keys = keys or self.DEFAULT_KEYS
        out = {}
        for k in keys:
            if k in raw:
                out[f"{k}_lz"] = self.z(k, raw[k])
        return out
