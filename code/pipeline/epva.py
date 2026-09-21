"""Epic 12.1 — zone EPVA process residuals from PBP shot zones (not tracking EPV).

``epva = actual_points − zone_xpoints`` on a past-only rolling window, split into
decision (zone mix vs league policy) vs execution (make vs zone xFG%).
Optical-tracking / MDP EPV remains roadmap 12.10 blocked.
"""
from __future__ import annotations

from collections import defaultdict, deque
from typing import Any

import numpy as np
import pandas as pd

from pipeline.shot_zones import (
    CANONICAL_ZONES,
    DEFAULT_FG_PCT,
    DEFAULT_POINT_VALUE,
    ZoneCalibration,
    legacy_zone_alias,
)


def zone_xpoints(zone: str, calib: ZoneCalibration | None = None) -> float:
    if calib is not None:
        return float(calib.xpoints(zone))
    z = legacy_zone_alias(zone)
    return float(DEFAULT_FG_PCT.get(z, 0.40) * DEFAULT_POINT_VALUE.get(z, 2.0))


def league_policy_mix(calib: ZoneCalibration | None = None) -> dict[str, float]:
    """Uniform-over-canonical fallback when empirical mix is unavailable."""
    n = len(CANONICAL_ZONES)
    return {z: 1.0 / n for z in CANONICAL_ZONES}


def decision_epva(
    zone_counts: dict[str, int],
    *,
    calib: ZoneCalibration | None = None,
    policy: dict[str, float] | None = None,
) -> float:
    """Expected points from zone mix vs league policy (execution-free)."""
    total = sum(zone_counts.values())
    if total <= 0:
        return 0.0
    pol = policy or league_policy_mix(calib)
    actual_mix = {z: zone_counts.get(z, 0) / total for z in CANONICAL_ZONES}
    ep = 0.0
    for z in CANONICAL_ZONES:
        xp = zone_xpoints(z, calib)
        ep += (actual_mix.get(z, 0.0) - pol.get(z, 0.0)) * xp
    return float(ep)


def execution_epva(
    makes_points: float,
    attempts: int,
    zone_xpts_sum: float,
) -> float:
    """Actual points scored minus sum of zone expected points on attempts."""
    if attempts <= 0:
        return 0.0
    return float(makes_points - zone_xpts_sum)


def epva_from_gs_side(gs: dict | None, side: str, calib: ZoneCalibration | None = None) -> dict[str, float]:
    """Zone-proxy EPVA from aggregated rim/three FGA in ``gs`` (stint rollups)."""
    if not gs:
        return {"decision": 0.0, "execution": 0.0, "n_attempts": 0, "epva": 0.0}
    prefix = "home" if side == "home" else "away"
    rim = int(float(gs.get(f"{prefix}_rim_fga", 0) or 0))
    three = int(float(gs.get(f"{prefix}_three_fga", 0) or 0))
    fga = int(float(gs.get(f"{prefix}_fga", 0) or 0))
    fgm = float(gs.get(f"{prefix}_fgm", 0) or 0)
    fg3m = float(gs.get(f"{prefix}_3pm", 0) or gs.get(f"{prefix}_fg3m", 0) or 0)
    mid = max(0, fga - rim - three)
    if fga <= 0 and rim + three <= 0:
        return {"decision": 0.0, "execution": 0.0, "n_attempts": 0, "epva": 0.0}
    counts: dict[str, int] = {z: 0 for z in CANONICAL_ZONES}
    counts["restricted"] = max(rim, 0)
    counts["abovebreak3"] = max(three, 0)
    counts["midrange"] = max(mid, 0)
    n = sum(counts.values())
    if n <= 0:
        return {"decision": 0.0, "execution": 0.0, "n_attempts": 0, "epva": 0.0}
    dec = decision_epva(counts, calib=calib)
    made_rim = fgm * (rim / max(fga, 1.0)) if fga > 0 else 0.0
    made_three = min(fg3m, float(three))
    made_mid = max(0.0, fgm - made_rim - made_three)
    pts = 2.0 * made_rim + 3.0 * made_three + 2.0 * made_mid
    xpts = (
        rim * zone_xpoints("restricted", calib)
        + three * zone_xpoints("abovebreak3", calib)
        + mid * zone_xpoints("midrange", calib)
    )
    exe = execution_epva(pts, n, xpts)
    return {
        "decision": float(dec),
        "execution": float(exe),
        "n_attempts": int(n),
        "epva": float(dec + exe),
    }


class EpvaTracker:
    """Rolling past-only team EPVA features for walk-forward game rows.

    Shrinks observations toward a prior-season team EPVA (else league prior),
    with Bayesian weight ``W = n / (n + k)``. ``k`` is larger for execution than
    decision and scales up with days since the prior snapshot so offseason /
    roster-turnover cold starts lean on the league prior.
    """

    def __init__(
        self,
        window_games: int = 20,
        shrink_k: float = 50.0,
        *,
        prior_season_epva: dict[str, dict[str, float]] | None = None,
        league_zone_prior: dict[str, float] | None = None,
        days_since_prior: float = 120.0,
        continuity_half_life_days: float = 90.0,
    ):
        self.window_games = int(window_games)
        self.shrink_k = float(shrink_k)
        self.prior_season_epva: dict[str, dict[str, float]] = {
            str(k): dict(v) for k, v in (prior_season_epva or {}).items()
        }
        self.league_zone_prior: dict[str, float] = dict(
            league_zone_prior
            or {"decision": 0.0, "execution": 0.0}
        )
        self.days_since_prior = float(max(0.0, days_since_prior))
        self.continuity_half_life_days = float(max(1.0, continuity_half_life_days))
        self._hist: dict[str, deque] = defaultdict(
            lambda: deque(maxlen=self.window_games)
        )
        self.fitted = True

    def set_days_since_prior(self, days: float) -> None:
        self.days_since_prior = float(max(0.0, days))

    def bump_offseason(self, days: float = 120.0) -> None:
        """Increase prior age across an offseason gap (raises effective k)."""
        self.days_since_prior = float(self.days_since_prior) + float(max(0.0, days))

    def seed_priors_from_history(self) -> None:
        """Snapshot current rolling aggregates into prior_season_epva."""
        for team in list(self._hist.keys()):
            dec, exe, n = self._raw_obs(team)
            if n <= 0:
                continue
            self.prior_season_epva[str(team)] = {
                "decision": float(dec),
                "execution": float(exe),
                "n": float(n),
            }
        self.days_since_prior = 0.0

    def _time_scale(self) -> float:
        """f(Δt) ≥ 1; grows with days since prior (continuity decay)."""
        return 1.0 + self.days_since_prior / self.continuity_half_life_days

    def _prior_for(self, team: str) -> tuple[float, float]:
        prior = self.prior_season_epva.get(str(team))
        if prior:
            return float(prior.get("decision", 0.0)), float(prior.get("execution", 0.0))
        return (
            float(self.league_zone_prior.get("decision", 0.0)),
            float(self.league_zone_prior.get("execution", 0.0)),
        )

    def update_game(
        self,
        team: str,
        *,
        decision: float,
        execution: float,
        n_attempts: int,
    ) -> None:
        self._hist[str(team)].append({
            "decision": float(decision),
            "execution": float(execution),
            "n": int(n_attempts),
        })

    def _raw_obs(self, team: str) -> tuple[float, float, float]:
        hist = self._hist.get(str(team))
        if not hist:
            return 0.0, 0.0, 0.0
        n = sum(g["n"] for g in hist)
        if n <= 0:
            return 0.0, 0.0, 0.0
        w_dec = sum(g["decision"] * max(g["n"], 1) for g in hist) / max(n, 1)
        w_exe = sum(g["execution"] * max(g["n"], 1) for g in hist) / max(n, 1)
        return float(w_dec), float(w_exe), float(n)

    def _agg(self, team: str) -> tuple[float, float, float]:
        obs_dec, obs_exe, n = self._raw_obs(team)
        prior_dec, prior_exe = self._prior_for(team)
        scale = self._time_scale()
        k_dec = self.shrink_k * 0.5 * scale
        k_exe = self.shrink_k * scale
        if n <= 0:
            return float(prior_dec), float(prior_exe), 0.0
        w_dec = n / (n + k_dec)
        w_exe = n / (n + k_exe)
        dec = w_dec * obs_dec + (1.0 - w_dec) * prior_dec
        exe = w_exe * obs_exe + (1.0 - w_exe) * prior_exe
        return float(dec), float(exe), float(n)

    def feature_dict(self, home_team: str, away_team: str) -> dict[str, float]:
        h_dec, h_exe, h_n = self._agg(home_team)
        a_dec, a_exe, a_n = self._agg(away_team)
        h_epva = h_dec + h_exe
        a_epva = a_dec + a_exe
        return {
            "h_epva_decision": h_dec,
            "a_epva_decision": a_dec,
            "h_epva_execution": h_exe,
            "a_epva_execution": a_exe,
            "h_epva": h_epva,
            "a_epva": a_epva,
            "epva_diff": float(h_epva - a_epva),
            "epva_sample_min": float(min(h_n, a_n)),
        }


def epva_from_shot_rows(
    shots: pd.DataFrame,
    *,
    zone_col: str = "shot_zone",
    made_col: str = "shot_made",
    points_col: str | None = "shot_points",
    calib: ZoneCalibration | None = None,
) -> dict[str, float]:
    """Aggregate decision/execution EPVA from a shot-level frame."""
    if shots is None or shots.empty:
        return {"decision": 0.0, "execution": 0.0, "n_attempts": 0, "epva": 0.0}
    zones = [legacy_zone_alias(z) for z in shots[zone_col].tolist()]
    counts: dict[str, int] = defaultdict(int)
    for z in zones:
        counts[z] += 1
    dec = decision_epva(counts, calib=calib)
    xpts = sum(zone_xpoints(z, calib) for z in zones)
    if points_col and points_col in shots.columns:
        pts = float(pd.to_numeric(shots[points_col], errors="coerce").fillna(0).sum())
    else:
        made = pd.to_numeric(shots[made_col], errors="coerce").fillna(0)
        pts = 0.0
        for z, m in zip(zones, made.tolist()):
            if float(m) > 0.5:
                pts += DEFAULT_POINT_VALUE.get(z, 2.0)
    exe = execution_epva(pts, len(zones), xpts)
    return {
        "decision": float(dec),
        "execution": float(exe),
        "n_attempts": int(len(zones)),
        "epva": float(dec + exe),
    }
