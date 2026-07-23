"""Task 034: correlated multi-player rotation scenarios.

Player active/limited/out outcomes are correlated within a game (e.g. a
team-wide "load management" or "get healthy for the playoffs" night tends to
rest several rotation players together, not independently). This module
generates a small, exactly-enumerable joint distribution over rotation
scenarios using a discrete common-factor mixture:

  1. A latent team-level factor (e.g. "normal" vs "rest_night") is drawn
     first, with a fixed marginal probability per factor state.
  2. Conditional on the factor, each modeled player's ACTIVE/LIMITED/OUT
     status is drawn independently, but every player's active probability
     is shifted by the *same* factor -- this induces positive correlation
     among players' statuses across the mixture (their statuses are only
     conditionally independent, not marginally independent), exactly the
     "jointly, not independently" requirement.

Because the factor and per-player status spaces are both finite, the whole
scenario space can be enumerated exactly: probabilities are exact products
(no Monte Carlo noise) and are guaranteed to sum to one by construction.
Each scenario's minutes are built with `MinutesForecastModel.forecast_team`,
which enforces the 240-expected-team-minutes constraint (Task 033) *within
every individual scenario*, respecting the roster/minutes constraint.

Only players with genuinely uncertain availability should be branched over
(`max_players_for_scenarios`); the rest of the roster is treated as
marginal/deterministic in every scenario to keep the scenario count small
and each scenario's roster/minutes accounting exact.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from pipeline.minutes_forecast import MinutesForecastModel

STATUS_ACTIVE = "ACTIVE"
STATUS_LIMITED = "LIMITED"
STATUS_OUT = "OUT"
_STATUSES = (STATUS_ACTIVE, STATUS_LIMITED, STATUS_OUT)

DEFAULT_FACTOR_PROBS = {"normal": 0.85, "rest_night": 0.15}
DEFAULT_FACTOR_MULTIPLIER = {"normal": 1.0, "rest_night": 0.70}
DEFAULT_LIMITED_PROB = 0.15
DEFAULT_LIMITED_MINUTES_FRACTION = 0.5


@dataclass
class RotationScenario:
    scenario_id: str
    statuses: Dict[str, str]
    minutes: Dict[str, float]
    probability: float
    factor: str = ""


def _player_status_probs(active_prob: float, factor_mult: float, limited_prob: float) -> Dict[str, float]:
    a = min(max(active_prob * factor_mult, 0.0), 1.0)
    p_limited = a * limited_prob
    p_active = a * (1.0 - limited_prob)
    p_out = 1.0 - a
    total = p_active + p_limited + p_out
    return {STATUS_ACTIVE: p_active / total, STATUS_LIMITED: p_limited / total, STATUS_OUT: p_out / total}


def scenario_entropy(scenarios: Iterable[RotationScenario]) -> float:
    """Shannon entropy (nats) of the scenario probability distribution."""
    h = 0.0
    for sc in scenarios:
        p = sc.probability
        if p > 0:
            h -= p * math.log(p)
    return h


class RotationScenarioGenerator:
    def __init__(
        self,
        minutes_model: MinutesForecastModel,
        factor_probs: Optional[Dict[str, float]] = None,
        factor_multiplier: Optional[Dict[str, float]] = None,
        limited_prob: float = DEFAULT_LIMITED_PROB,
        max_players_for_scenarios: int = 4,
    ):
        self.minutes_model = minutes_model
        self.factor_probs = dict(factor_probs or DEFAULT_FACTOR_PROBS)
        self.factor_multiplier = dict(factor_multiplier or DEFAULT_FACTOR_MULTIPLIER)
        self.limited_prob = limited_prob
        self.max_players_for_scenarios = max_players_for_scenarios

        fp_total = sum(self.factor_probs.values())
        if abs(fp_total - 1.0) > 1e-9:
            raise ValueError(f"factor_probs must sum to 1.0, got {fp_total}")

    def _select_branch_players(self, roster: List[str], statuses: Optional[Dict[str, str]]) -> List[str]:
        """Rank by prior active-rate uncertainty (closest to 0.5 = most
        uncertain -> most worth branching over); ties broken by roster order
        for determinism."""
        statuses = statuses or {}

        def _uncertainty(pid):
            base = self.minutes_model.active_probability(pid, status=statuses.get(pid))
            return -abs(base - 0.5)  # larger (closer to 0) = more uncertain

        ranked = sorted(roster, key=_uncertainty, reverse=True)
        return ranked[: self.max_players_for_scenarios]

    def generate(
        self,
        roster: Iterable[str],
        statuses: Optional[Dict[str, str]] = None,
        minutes_limits: Optional[Dict[str, float]] = None,
    ) -> List[RotationScenario]:
        roster = [str(p) for p in roster]
        statuses = statuses or {}
        minutes_limits = dict(minutes_limits or {})

        branch_players = self._select_branch_players(roster, statuses)
        fixed_players = [p for p in roster if p not in branch_players]

        marginal_active_prob = {
            pid: self.minutes_model.active_probability(pid, status=statuses.get(pid))
            for pid in branch_players
        }

        scenarios: List[RotationScenario] = []
        for factor, factor_prob in self.factor_probs.items():
            mult = self.factor_multiplier.get(factor, 1.0)
            per_player_probs = {
                pid: _player_status_probs(marginal_active_prob[pid], mult, self.limited_prob)
                for pid in branch_players
            }
            for combo in itertools.product(_STATUSES, repeat=len(branch_players)):
                combo_prob = factor_prob
                combo_statuses: Dict[str, str] = dict(statuses)
                combo_limits = dict(minutes_limits)
                for pid, status in zip(branch_players, combo):
                    combo_prob *= per_player_probs[pid][status]
                    if status == STATUS_OUT:
                        combo_statuses[pid] = "OUT"
                    elif status == STATUS_LIMITED:
                        combo_statuses[pid] = "PROBABLE"
                        base_minutes = self.minutes_model.conditional_minutes(pid)
                        combo_limits[pid] = min(
                            combo_limits.get(pid, base_minutes),
                            base_minutes * DEFAULT_LIMITED_MINUTES_FRACTION,
                        )
                    else:
                        combo_statuses[pid] = statuses.get(pid, "ACTIVE")
                if combo_prob <= 0:
                    continue
                minutes = self.minutes_model.forecast_team(
                    roster, statuses=combo_statuses, minutes_limits=combo_limits,
                )
                status_label = {
                    pid: (
                        STATUS_OUT if s == STATUS_OUT else
                        STATUS_LIMITED if s == STATUS_LIMITED else
                        STATUS_ACTIVE
                    )
                    for pid, s in zip(branch_players, combo)
                }
                for pid in fixed_players:
                    status_label[pid] = STATUS_ACTIVE
                scenario_id = f"{factor}:" + ",".join(f"{p}={s}" for p, s in zip(branch_players, combo))
                scenarios.append(RotationScenario(
                    scenario_id=scenario_id,
                    statuses=status_label,
                    minutes=minutes,
                    probability=combo_prob,
                    factor=factor,
                ))

        total_prob = sum(sc.probability for sc in scenarios)
        # Exact-enumeration guard: any drift from 1.0 here signals a bug in
        # the per-player probability construction, not legitimate rounding.
        assert abs(total_prob - 1.0) < 1e-9, (
            f"rotation scenario probabilities summed to {total_prob}, not 1.0"
        )
        return scenarios
