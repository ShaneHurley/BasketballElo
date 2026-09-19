"""Hierarchical / partially pooled pace forecasting.

Replaces the deterministic 10-game mean with league → team offensive/defensive
tempo effects + context, emitting posterior mean/variance/quantiles while
preserving ``PaceTracker`` as the compatibility baseline.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass

import numpy as np

from pipeline.trackers import PaceTracker


@dataclass(frozen=True)
class PaceForecast:
    mean: float
    variance: float
    q10: float
    q50: float
    q90: float
    baseline_mean: float
    effective_n: float

    def as_feature_dict(self, prefix: str = "") -> dict:
        p = prefix
        return {
            f"{p}pace_mean": self.mean,
            f"{p}pace_var": self.variance,
            f"{p}pace_std": float(np.sqrt(max(self.variance, 1e-6))),
            f"{p}pace_q10": self.q10,
            f"{p}pace_q50": self.q50,
            f"{p}pace_q90": self.q90,
            f"{p}pace_baseline": self.baseline_mean,
            f"{p}pace_eff_n": self.effective_n,
        }


def _shrink(obs: float, n: float, prior: float, k: float) -> float:
    if n <= 0:
        return float(prior)
    return float((n * obs + k * prior) / (n + k))


class HierarchicalPaceModel:
    """Time-decayed empirical-Bayes pace with team O/D tempo effects.

    Observation: shared possessions ≈ league + home_off_tempo + away_def_tempo
    + away_off_tempo + home_def_tempo)/2 + context.

    For simplicity we store per-team offensive tempo (possessions forced) and
    defensive/opponent tempo (possessions allowed), shrunk toward league.
    """

    LEAGUE = 100.0
    SHRINK_K = 12.0
    HALF_LIFE_GAMES = 20.0
    MIN_PACE = 85.0
    MAX_PACE = 120.0
    RESIDUAL_VAR = 16.0  # process noise floor (~4 poss SD)

    def __init__(self, window: int = 30, baseline: PaceTracker | None = None):
        self.window = window
        self.baseline = baseline or PaceTracker(team_window=10, league_window=150, per_team=True)
        self.off_tempo = defaultdict(float)   # team → possessions when on offense (shared)
        self.def_tempo = defaultdict(float)
        self.off_n = defaultdict(float)
        self.def_n = defaultdict(float)
        self.league_hist = deque(maxlen=max(60, window * 4))
        self._game_count = 0

    def _decay_weight(self, age_games: float) -> float:
        return float(0.5 ** (age_games / max(self.HALF_LIFE_GAMES, 1.0)))

    def league_pace(self) -> float:
        if not self.league_hist:
            return self.LEAGUE
        return float(np.mean(self.league_hist))

    def update(
        self,
        home: str,
        away: str,
        home_poss: float,
        away_poss: float | None = None,
        *,
        context: float = 0.0,
    ) -> None:
        if away_poss is None:
            home_poss = away_poss = float(home_poss) / 2.0
        home_poss = float(home_poss)
        away_poss = float(away_poss)
        shared = 0.5 * (home_poss + away_poss)
        self.league_hist.append(shared)
        self.baseline.update_pace(home, away, home_poss, away_poss)

        # Soft online update with unit weight (caller walks chronologically).
        w = 1.0
        lg = self.league_pace()
        # Home offensive tempo residual vs league; away defensive allowance.
        h_off_resid = home_poss - lg
        a_def_resid = home_poss - lg
        a_off_resid = away_poss - lg
        h_def_resid = away_poss - lg

        def _upd(store, nstore, key, resid):
            n0 = nstore[key]
            n1 = n0 + w
            store[key] = (n0 * store[key] + w * resid) / n1 if n1 > 0 else resid
            nstore[key] = n1

        _upd(self.off_tempo, self.off_n, home, h_off_resid)
        _upd(self.def_tempo, self.def_n, away, a_def_resid)
        _upd(self.off_tempo, self.off_n, away, a_off_resid)
        _upd(self.def_tempo, self.def_n, home, h_def_resid)
        self._game_count += 1

    def team_off(self, team: str) -> float:
        return _shrink(self.off_tempo[team], self.off_n[team], 0.0, self.SHRINK_K)

    def team_def(self, team: str) -> float:
        return _shrink(self.def_tempo[team], self.def_n[team], 0.0, self.SHRINK_K)

    def predict(
        self,
        home: str,
        away: str,
        *,
        context: float = 0.0,
        is_altitude: bool = False,
        rest_diff: float = 0.0,
    ) -> PaceForecast:
        lg = self.league_pace()
        # Shared pace ≈ league + avg(home_off + away_def, away_off + home_def) + ctx
        side_h = self.team_off(home) + self.team_def(away)
        side_a = self.team_off(away) + self.team_def(home)
        ctx = float(context)
        if is_altitude:
            ctx += 0.8
        ctx += 0.15 * float(rest_diff)  # more rested → slight pace up
        mean = lg + 0.5 * (side_h + side_a) + ctx
        mean = float(np.clip(mean, self.MIN_PACE, self.MAX_PACE))

        eff_n = 0.5 * (
            self.off_n[home] / (self.off_n[home] + self.SHRINK_K)
            + self.def_n[away] / (self.def_n[away] + self.SHRINK_K)
            + self.off_n[away] / (self.off_n[away] + self.SHRINK_K)
            + self.def_n[home] / (self.def_n[home] + self.SHRINK_K)
        )
        # Posterior variance: process noise + parameter uncertainty.
        param_var = self.RESIDUAL_VAR / max(eff_n * 4.0 + 1.0, 1.0)
        variance = self.RESIDUAL_VAR + param_var
        sd = float(np.sqrt(variance))
        baseline_mean = self.baseline.get_expected_pace(home, away)
        # Blend toward baseline when evidence is thin.
        blend = float(np.clip(eff_n, 0.0, 1.0))
        mean = blend * mean + (1.0 - blend) * baseline_mean
        mean = float(np.clip(mean, self.MIN_PACE, self.MAX_PACE))
        return PaceForecast(
            mean=mean,
            variance=variance,
            q10=float(np.clip(mean - 1.2816 * sd, self.MIN_PACE, self.MAX_PACE)),
            q50=mean,
            q90=float(np.clip(mean + 1.2816 * sd, self.MIN_PACE, self.MAX_PACE)),
            baseline_mean=float(baseline_mean),
            effective_n=float(eff_n),
        )

    def get_expected_pace(self, home: str, away: str, **kwargs) -> float:
        return self.predict(home, away, **kwargs).mean
