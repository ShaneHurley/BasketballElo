"""Task 033: T-60 active probability / minutes forecasting.

Forecasts, from prior data only:
  - ``active_probability``   : P(player suits up and plays > 0 minutes).
  - ``conditional_minutes``  : E[minutes | active], from prior games.
  - ``minutes_limit``        : an optional hard cap (e.g. a return-from-injury
                                restriction) known as of the cutoff.
  - ``replacement_minutes``  : the minutes bucket that must flow to
                                unmodeled bench/call-up players so the team
                                total is exactly 240 expected minutes.

Every quantity here is built from `record_game` calls the caller makes only
for games strictly before the forecast's own cutoff (Rule 3) -- this module
never reads the game being forecast.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional

from pipeline.availability import status_to_prob

TEAM_EXPECTED_MINUTES = 240.0
DEFAULT_CONDITIONAL_MINUTES = 20.0
DEFAULT_ACTIVE_PROB = 0.80


@dataclass
class PlayerMinutesForecast:
    player_id: str
    active_prob: float
    conditional_minutes: float
    minutes_limit: Optional[float]
    expected_minutes: float


class MinutesForecastModel:
    """Rolling, prior-games-only active-rate and minutes model."""

    def __init__(self, window_games: int = 15):
        self.window_games = window_games
        self._active_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_games))
        self._minutes_when_active: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_games))

    def record_game(self, player_id: str, minutes: float, active: bool) -> None:
        """Record one *completed* game's outcome for `player_id`. Caller is
        responsible for only calling this for games strictly before any
        subsequent forecast's cutoff."""
        pid = str(player_id)
        self._active_history[pid].append(bool(active))
        if active:
            self._minutes_when_active[pid].append(float(minutes))

    def active_probability(self, player_id: str, status: Optional[str] = None) -> float:
        """Prior active rate, optionally overridden/blended by a timestamped
        status report (Task 030) that was itself published by the cutoff."""
        pid = str(player_id)
        hist = self._active_history.get(pid)
        prior = (sum(hist) / len(hist)) if hist else DEFAULT_ACTIVE_PROB
        if status is None:
            return float(prior)
        # A definitive OUT/ACTIVE report dominates; GTD/QUESTIONABLE blend
        # with prior history since neither source alone is fully reliable.
        status_prob = status_to_prob(status)
        s = (status or "").upper()
        if "OUT" in s or ("PROBABLE" not in s and "ACTIVE" in s):
            return float(status_prob)
        return float(0.5 * prior + 0.5 * status_prob)

    def conditional_minutes(self, player_id: str) -> float:
        pid = str(player_id)
        mins = self._minutes_when_active.get(pid)
        if not mins:
            return DEFAULT_CONDITIONAL_MINUTES
        return float(sum(mins) / len(mins))

    def player_forecast(
        self,
        player_id: str,
        status: Optional[str] = None,
        minutes_limit: Optional[float] = None,
    ) -> PlayerMinutesForecast:
        pid = str(player_id)
        active_prob = self.active_probability(pid, status=status)
        cond_minutes = self.conditional_minutes(pid)
        capped = cond_minutes if minutes_limit is None else min(cond_minutes, float(minutes_limit))
        capped = max(capped, 0.0)
        return PlayerMinutesForecast(
            player_id=pid,
            active_prob=active_prob,
            conditional_minutes=cond_minutes,
            minutes_limit=minutes_limit,
            expected_minutes=active_prob * capped,
        )

    def forecast_team(
        self,
        roster: Iterable[str],
        statuses: Optional[Dict[str, str]] = None,
        minutes_limits: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        """Return ``{player_id: expected_minutes, ...,
        "_replacement": replacement_minutes}`` with the values summing to
        exactly `TEAM_EXPECTED_MINUTES` (Task 033 pass condition).

        If the modeled roster's raw expected minutes fall short of 240, the
        shortfall becomes `_replacement` minutes (bench/call-up capacity not
        individually modeled). If the modeled roster's raw expected minutes
        exceed 240 (e.g. a deep, mostly-healthy rotation), every player's
        expected minutes are scaled down proportionally and `_replacement`
        is zero -- a team can play at most 240 minutes.
        """
        statuses = statuses or {}
        minutes_limits = minutes_limits or {}
        roster = [str(p) for p in roster]

        raw: Dict[str, float] = {}
        for pid in roster:
            fc = self.player_forecast(
                pid, status=statuses.get(pid), minutes_limit=minutes_limits.get(pid),
            )
            raw[pid] = fc.expected_minutes

        total_raw = sum(raw.values())
        out: Dict[str, float] = {}
        if total_raw > TEAM_EXPECTED_MINUTES:
            scale = TEAM_EXPECTED_MINUTES / total_raw if total_raw > 0 else 0.0
            for pid, v in raw.items():
                out[pid] = v * scale
            out["_replacement"] = 0.0
        else:
            out.update(raw)
            out["_replacement"] = TEAM_EXPECTED_MINUTES - total_raw

        assert abs(sum(out.values()) - TEAM_EXPECTED_MINUTES) < 1e-6, (
            "forecast_team: expected team minutes did not sum to exactly "
            f"{TEAM_EXPECTED_MINUTES}"
        )
        return out
