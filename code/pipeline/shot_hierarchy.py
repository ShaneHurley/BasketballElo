"""Hierarchical shot-attempt mix and make-rate forecasts (shadow features)."""
from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

from pipeline.shot_zones import (
    CANONICAL_ZONES,
    DEFAULT_FG_PCT,
    DEFAULT_POINT_VALUE,
    SHRINK_K,
    ZoneCalibration,
    legacy_zone_alias,
)


def _shrink(obs: float, n: float, prior: float, k: float) -> float:
    if n <= 0:
        return float(prior)
    return float((n * obs + k * prior) / (n + k))


@dataclass
class HierarchicalZoneRates:
    """Partially pooled make rates: league → team → player within zone."""

    league: ZoneCalibration = field(default_factory=ZoneCalibration)
    team_fg: dict = field(default_factory=lambda: defaultdict(dict))
    team_n: dict = field(default_factory=lambda: defaultdict(dict))
    player_fg: dict = field(default_factory=lambda: defaultdict(dict))
    player_n: dict = field(default_factory=lambda: defaultdict(dict))
    team_mix: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    team_mix_n: dict = field(default_factory=lambda: defaultdict(float))
    player_mix: dict = field(default_factory=lambda: defaultdict(lambda: defaultdict(float)))
    player_mix_n: dict = field(default_factory=lambda: defaultdict(float))

    def update_shot(
        self,
        *,
        zone: str,
        made: bool,
        team: str | None = None,
        player: str | None = None,
    ) -> None:
        z = legacy_zone_alias(zone)
        made_f = float(bool(made))
        # League
        n = self.league.counts.get(z, 0) + 1
        prev = self.league.fg_pct.get(z, DEFAULT_FG_PCT.get(z, 0.40))
        self.league.fg_pct[z] = ((n - 1) * prev + made_f) / n
        self.league.counts[z] = n
        self.league.point_value[z] = DEFAULT_POINT_VALUE.get(z, 2.0)
        if team:
            tn = self.team_n[team].get(z, 0.0) + 1.0
            tp = self.team_fg[team].get(z, prev)
            self.team_fg[team][z] = ((tn - 1.0) * tp + made_f) / tn
            self.team_n[team][z] = tn
            self.team_mix[team][z] += 1.0
            self.team_mix_n[team] += 1.0
        if player:
            pn = self.player_n[player].get(z, 0.0) + 1.0
            pp = self.player_fg[player].get(z, prev)
            self.player_fg[player][z] = ((pn - 1.0) * pp + made_f) / pn
            self.player_n[player][z] = pn
            self.player_mix[player][z] += 1.0
            self.player_mix_n[player] += 1.0

    def make_prob(
        self,
        zone: str,
        *,
        team: str | None = None,
        player: str | None = None,
        opp_team: str | None = None,
    ) -> float:
        z = legacy_zone_alias(zone)
        league = self.league.fg_pct_for_zone(z)
        rate = league
        if team and z in self.team_fg.get(team, {}):
            rate = _shrink(self.team_fg[team][z], self.team_n[team].get(z, 0.0), league, SHRINK_K)
        if player and z in self.player_fg.get(player, {}):
            parent = rate
            rate = _shrink(
                self.player_fg[player][z],
                self.player_n[player].get(z, 0.0),
                parent,
                SHRINK_K,
            )
        # Opponent defensive zone allowance: shrink toward league (symmetric).
        if opp_team and z in self.team_fg.get(opp_team, {}):
            # Teams that allow high FG% in a zone → raise make prob slightly.
            opp_allow = _shrink(
                self.team_fg[opp_team][z],
                self.team_n[opp_team].get(z, 0.0),
                league,
                SHRINK_K,
            )
            rate = 0.85 * rate + 0.15 * opp_allow
        return float(np.clip(rate, 0.05, 0.95))

    def attempt_mix(self, *, team: str | None = None, player: str | None = None) -> dict:
        """Dirichlet-multinomial shrunk zone share vector."""
        league_counts = {z: float(self.league.counts.get(z, 1)) for z in CANONICAL_ZONES}
        league_n = sum(league_counts.values()) or 1.0
        league_share = {z: league_counts[z] / league_n for z in CANONICAL_ZONES}

        if player and self.player_mix_n.get(player, 0) > 0:
            n = self.player_mix_n[player]
            raw = {z: self.player_mix[player].get(z, 0.0) / n for z in CANONICAL_ZONES}
            return {
                z: _shrink(raw[z], n, league_share[z], SHRINK_K)
                for z in CANONICAL_ZONES
            }
        if team and self.team_mix_n.get(team, 0) > 0:
            n = self.team_mix_n[team]
            raw = {z: self.team_mix[team].get(z, 0.0) / n for z in CANONICAL_ZONES}
            return {
                z: _shrink(raw[z], n, league_share[z], SHRINK_K)
                for z in CANONICAL_ZONES
            }
        return league_share

    def expected_pps(
        self,
        *,
        team: str | None = None,
        player: str | None = None,
        opp_team: str | None = None,
    ) -> float:
        mix = self.attempt_mix(team=team, player=player)
        pps = 0.0
        for z, share in mix.items():
            fg = self.make_prob(z, team=team, player=player, opp_team=opp_team)
            pv = DEFAULT_POINT_VALUE.get(z, 2.0)
            pps += share * fg * pv
        return float(pps)

    def feature_dict(self, home_team: str, away_team: str) -> dict:
        h_mix = self.attempt_mix(team=home_team)
        a_mix = self.attempt_mix(team=away_team)
        h_pps = self.expected_pps(team=home_team, opp_team=away_team)
        a_pps = self.expected_pps(team=away_team, opp_team=home_team)
        out = {
            "h_hier_shot_pps": h_pps,
            "a_hier_shot_pps": a_pps,
            "hier_shot_pps_diff": h_pps - a_pps,
            "h_rim_share": h_mix.get("restricted", 0.0) + h_mix.get("paint", 0.0),
            "a_rim_share": a_mix.get("restricted", 0.0) + a_mix.get("paint", 0.0),
            "h_three_share": h_mix.get("corner3", 0.0) + h_mix.get("abovebreak3", 0.0),
            "a_three_share": a_mix.get("corner3", 0.0) + a_mix.get("abovebreak3", 0.0),
            "rim_share_diff": (
                (h_mix.get("restricted", 0.0) + h_mix.get("paint", 0.0))
                - (a_mix.get("restricted", 0.0) + a_mix.get("paint", 0.0))
            ),
            "three_share_diff": (
                (h_mix.get("corner3", 0.0) + h_mix.get("abovebreak3", 0.0))
                - (a_mix.get("corner3", 0.0) + a_mix.get("abovebreak3", 0.0))
            ),
            "h_corner3_fg": self.make_prob("corner3", team=home_team, opp_team=away_team),
            "a_corner3_fg": self.make_prob("corner3", team=away_team, opp_team=home_team),
            "h_ab3_fg": self.make_prob("abovebreak3", team=home_team, opp_team=away_team),
            "a_ab3_fg": self.make_prob("abovebreak3", team=away_team, opp_team=home_team),
        }
        return out


# Volume / attempt-mix shadow features (ablate separately from make rates).
HIER_SHOT_MIX_COLS = [
    "h_rim_share", "a_rim_share", "rim_share_diff",
    "h_three_share", "a_three_share", "three_share_diff",
]

# Accuracy / make-rate shadow features.
HIER_SHOT_MAKE_COLS = [
    "h_corner3_fg", "a_corner3_fg", "h_ab3_fg", "a_ab3_fg",
    "h_hier_shot_pps", "a_hier_shot_pps", "hier_shot_pps_diff",
]

HIER_SHOT_FEATURE_COLS = [
    *HIER_SHOT_MIX_COLS,
    *HIER_SHOT_MAKE_COLS,
]
