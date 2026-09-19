"""Hierarchical minutes / availability with joint 240-minute constraint.

Extends the rolling MinutesForecastModel with player/role/team partial pooling
and Dirichlet-style share rescaling so expected minutes sum to exactly 240.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, Optional

import numpy as np

from pipeline.availability import status_to_prob
from pipeline.minutes_forecast import (
    DEFAULT_ACTIVE_PROB,
    DEFAULT_CONDITIONAL_MINUTES,
    MinutesForecastModel,
    PlayerMinutesForecast,
    TEAM_EXPECTED_MINUTES,
)


def _shrink(obs: float, n: float, prior: float, k: float) -> float:
    if n <= 0:
        return float(prior)
    return float((n * obs + k * prior) / (n + k))


@dataclass
class RolePrior:
    active_prob: float = DEFAULT_ACTIVE_PROB
    conditional_minutes: float = DEFAULT_CONDITIONAL_MINUTES
    n: float = 0.0


class HierarchicalMinutesModel(MinutesForecastModel):
    """Two-part availability + conditional minutes with partial pooling."""

    SHRINK_K = 8.0
    ROLE_BUCKETS = (8.0, 18.0, 28.0, 36.0)  # minutes thresholds → role ids 0..4

    def __init__(self, window_games: int = 15):
        super().__init__(window_games=window_games)
        self._team_active = defaultdict(lambda: deque(maxlen=window_games))
        self._team_minutes = defaultdict(lambda: deque(maxlen=window_games))
        self._role_priors: Dict[int, RolePrior] = {i: RolePrior() for i in range(5)}
        self._player_role: Dict[str, int] = {}
        self._learned_factor_probs = {"normal": 0.85, "rest_night": 0.15}
        self._rest_night_events = 0
        self._normal_events = 0

    def _role_from_minutes(self, minutes: float) -> int:
        for i, thr in enumerate(self.ROLE_BUCKETS):
            if minutes < thr:
                return i
        return len(self.ROLE_BUCKETS)

    def record_game(self, player_id: str, minutes: float, active: bool, team: str | None = None) -> None:
        super().record_game(player_id, minutes, active)
        pid = str(player_id)
        role = self._role_from_minutes(float(minutes) if active else 0.0)
        self._player_role[pid] = role
        rp = self._role_priors[role]
        n = rp.n + 1.0
        rp.active_prob = ((n - 1.0) * rp.active_prob + float(active)) / n
        if active:
            rp.conditional_minutes = (
                ((n - 1.0) * rp.conditional_minutes + float(minutes)) / n
                if rp.n > 0 else float(minutes)
            )
        rp.n = n
        if team is not None:
            self._team_active[str(team)].append(float(active))
            if active:
                self._team_minutes[str(team)].append(float(minutes))

    def record_team_rest_night(self, n_rotation_out: int, roster_size: int = 8) -> None:
        """Update learned rest-night factor probability from joint outs."""
        if roster_size <= 0:
            return
        if n_rotation_out >= max(2, roster_size // 4):
            self._rest_night_events += 1
        else:
            self._normal_events += 1
        total = self._rest_night_events + self._normal_events
        if total >= 20:
            p_rest = (self._rest_night_events + 3.0) / (total + 20.0)  # shrink to 0.15
            self._learned_factor_probs = {
                "normal": float(1.0 - p_rest),
                "rest_night": float(p_rest),
            }

    def factor_probs(self) -> Dict[str, float]:
        return dict(self._learned_factor_probs)

    def _role_prior(self, player_id: str) -> RolePrior:
        role = self._player_role.get(str(player_id), 2)
        return self._role_priors.get(role, RolePrior())

    def active_probability(self, player_id: str, status: Optional[str] = None) -> float:
        pid = str(player_id)
        hist = self._active_history.get(pid)
        n = float(len(hist)) if hist else 0.0
        raw = (sum(hist) / len(hist)) if hist else DEFAULT_ACTIVE_PROB
        prior = self._role_prior(pid).active_prob
        pooled = _shrink(raw, n, prior, self.SHRINK_K)
        if status is None:
            return float(np.clip(pooled, 0.01, 0.99))
        status_prob = status_to_prob(status)
        s = (status or "").upper()
        if "OUT" in s or ("PROBABLE" not in s and "ACTIVE" in s and "QUESTION" not in s):
            return float(status_prob)
        return float(np.clip(0.5 * pooled + 0.5 * status_prob, 0.01, 0.99))

    def conditional_minutes(self, player_id: str) -> float:
        pid = str(player_id)
        mins = self._minutes_when_active.get(pid)
        n = float(len(mins)) if mins else 0.0
        raw = (sum(mins) / len(mins)) if mins else DEFAULT_CONDITIONAL_MINUTES
        prior = self._role_prior(pid).conditional_minutes
        return float(max(0.0, _shrink(raw, n, prior, self.SHRINK_K)))

    def forecast_team(
        self,
        roster: Iterable[str],
        statuses: Optional[Dict[str, str]] = None,
        minutes_limits: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Softmax/share rescaling to exactly TEAM_EXPECTED_MINUTES."""
        statuses = statuses or {}
        minutes_limits = minutes_limits or {}
        roster = [str(p) for p in roster]
        raw: Dict[str, float] = {}
        for pid in roster:
            fc = self.player_forecast(
                pid, status=statuses.get(pid), minutes_limit=minutes_limits.get(pid),
            )
            raw[pid] = max(fc.expected_minutes, 0.0)

        total_raw = sum(raw.values())
        out: Dict[str, float] = {}
        if total_raw <= 1e-9:
            # Cold start: equal shares among roster, rest to replacement.
            if roster:
                per = min(TEAM_EXPECTED_MINUTES / len(roster), DEFAULT_CONDITIONAL_MINUTES)
                for pid in roster:
                    out[pid] = per
                out["_replacement"] = max(0.0, TEAM_EXPECTED_MINUTES - sum(out.values()))
            else:
                out["_replacement"] = TEAM_EXPECTED_MINUTES
            return out

        if total_raw > TEAM_EXPECTED_MINUTES:
            # Softmax-ish: proportional downscale (shares on simplex).
            scale = TEAM_EXPECTED_MINUTES / total_raw
            for pid, v in raw.items():
                out[pid] = v * scale
            out["_replacement"] = 0.0
        else:
            out.update(raw)
            out["_replacement"] = TEAM_EXPECTED_MINUTES - total_raw

        assert abs(sum(out.values()) - TEAM_EXPECTED_MINUTES) < 1e-6
        return out
