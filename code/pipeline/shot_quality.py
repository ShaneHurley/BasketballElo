"""Rolling pre-game shot-quality features (xEFG, rim rate, three rate)."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np

_DEFAULT_SHOT_QUALITY = {
    "xefg": 0.54,
    "rim_rate": 0.30,
    "three_rate": 0.38,
    "avg_shot_distance": 14.0,
}


class ShotQualityTracker:
    """Strictly pre-game rolling shot diet / expected FG% by team."""

    def __init__(self, window: int = 15, prev_season_weight: float = 0.4):
        self.window = window
        self.prev_season_weight = prev_season_weight
        self._history = defaultdict(lambda: deque(maxlen=window))
        self._prior_season = defaultdict(dict)

    def _rates(self, rec: dict) -> dict:
        fga = float(rec.get("fga", 0) or 0)
        if fga < 1:
            return dict(_DEFAULT_SHOT_QUALITY)
        xefg_sum = float(rec.get("xefg_sum", 0) or 0)
        rim_fga = float(rec.get("rim_fga", 0) or 0)
        three_fga = float(rec.get("three_fga", 0) or 0)
        dist_sum = float(rec.get("shot_dist_sum", 0) or 0)
        return {
            "xefg": xefg_sum / fga,
            "rim_rate": rim_fga / fga,
            "three_rate": three_fga / fga,
            "avg_shot_distance": dist_sum / fga if dist_sum > 0 else _DEFAULT_SHOT_QUALITY["avg_shot_distance"],
        }

    def update(self, team: str, gdate, season: int, game_stats: dict) -> None:
        rec = {
            "fga": float(game_stats.get("fga", 0) or 0),
            "xefg_sum": float(game_stats.get("xefg_sum", 0) or 0),
            "rim_fga": float(game_stats.get("rim_fga", 0) or 0),
            "three_fga": float(game_stats.get("three_fga", 0) or 0),
            "shot_dist_sum": float(game_stats.get("shot_dist_sum", 0) or 0),
            "gdate": gdate,
            "season": season,
        }
        key = (team, season)
        self._history[key].append(rec)

    def _blend_prior(self, team: str, season: int, current: dict) -> dict:
        prior = self._prior_season.get((team, season))
        if not prior:
            return current
        w = self.prev_season_weight
        return {k: (1 - w) * current.get(k, _DEFAULT_SHOT_QUALITY[k]) + w * prior.get(k, _DEFAULT_SHOT_QUALITY[k])
                for k in _DEFAULT_SHOT_QUALITY}

    def on_season_boundary(self, team: str, old_season: int) -> None:
        key = (team, old_season)
        hist = self._history.get(key)
        if not hist:
            return
        totals = {"fga": 0.0, "xefg_sum": 0.0, "rim_fga": 0.0, "three_fga": 0.0, "shot_dist_sum": 0.0}
        for rec in hist:
            for k in totals:
                totals[k] += float(rec.get(k, 0) or 0)
        self._prior_season[(team, old_season + 1)] = self._rates(totals)
        self._history[key].clear()

    def get(self, team: str, season: int, gdate) -> dict:
        key = (team, season)
        hist = [r for r in self._history.get(key, []) if r.get("gdate") is not None and r["gdate"] < gdate]
        if not hist:
            prior = self._prior_season.get((team, season))
            return dict(prior or _DEFAULT_SHOT_QUALITY)
        totals = {"fga": 0.0, "xefg_sum": 0.0, "rim_fga": 0.0, "three_fga": 0.0, "shot_dist_sum": 0.0}
        for rec in hist[-self.window:]:
            for k in totals:
                totals[k] += float(rec.get(k, 0) or 0)
        current = self._rates(totals)
        return self._blend_prior(team, season, current)

    def feature_dict(self, home_team: str, away_team: str, season: int, gdate) -> dict:
        h = self.get(home_team, season, gdate)
        a = self.get(away_team, season, gdate)
        return {
            "h_xefg": h["xefg"],
            "a_xefg": a["xefg"],
            "h_rim_rate": h["rim_rate"],
            "a_rim_rate": a["rim_rate"],
            "h_three_rate": h["three_rate"],
            "a_three_rate": a["three_rate"],
            "shot_quality_edge": h["xefg"] - a["xefg"],
            "rim_rate_diff": h["rim_rate"] - a["rim_rate"],
            "three_rate_diff": h["three_rate"] - a["three_rate"],
            "avg_shot_distance_diff": h["avg_shot_distance"] - a["avg_shot_distance"],
        }


def shot_quality_stats_from_gs(gs: dict, side: str) -> dict:
    prefix = "home" if side == "home" else "away"
    return {
        "fga": float(gs.get(f"{prefix}_fga", 0) or 0),
        "xefg_sum": float(gs.get(f"{prefix}_xefg_sum", 0) or 0),
        "rim_fga": float(gs.get(f"{prefix}_rim_fga", 0) or 0),
        "three_fga": float(gs.get(f"{prefix}_three_fga", 0) or 0),
        "shot_dist_sum": float(gs.get(f"{prefix}_shot_dist_sum", 0) or 0),
    }
