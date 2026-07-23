"""Team-level Bradley-Terry style Elo for spread and total."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

K_SPREAD = 20.0
K_TOTAL = 15.0
HOME_ADV = 2.5


class TeamEloTracker:
    """Separate spread and scoring strength by team."""

    def __init__(self, default: float = 1500.0):
        self.spread_mu = defaultdict(lambda: default)
        self.total_mu = defaultdict(lambda: default)
        self.games = defaultdict(int)

    def predict_spread_margin(self, home: str, away: str) -> float:
        return (self.spread_mu[home] - self.spread_mu[away]) / 25.0 + HOME_ADV

    def predict_total_adj(self, home: str, away: str) -> float:
        return (self.total_mu[home] + self.total_mu[away] - 3000.0) / 50.0

    def update(self, home: str, away: str, home_margin: float, total_pts: float,
               market_spread=None):
        exp = self.predict_spread_margin(home, away)
        err = home_margin - exp
        k = K_SPREAD / (1 + self.games[home] * 0.02)
        self.spread_mu[home] += k * err
        self.spread_mu[away] -= k * err
        exp_t = 225.0 + self.predict_total_adj(home, away)
        terr = total_pts - exp_t
        kt = K_TOTAL / (1 + self.games[home] * 0.02)
        self.total_mu[home] += kt * terr * 0.5
        self.total_mu[away] += kt * terr * 0.5
        self.games[home] += 1
        self.games[away] += 1

    def feature_dict(self, home: str, away: str):
        return {
            "team_elo_spread": self.predict_spread_margin(home, away),
            "team_elo_total_adj": self.predict_total_adj(home, away),
            "team_elo_net": (self.spread_mu[home] - self.spread_mu[away]) / 25.0,
        }

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "spread_mu": dict(self.spread_mu),
                "total_mu": dict(self.total_mu),
                "games": dict(self.games),
            }, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.spread_mu = defaultdict(lambda: 1500.0, data.get("spread_mu", {}))
        obj.total_mu = defaultdict(lambda: 1500.0, data.get("total_mu", {}))
        obj.games = defaultdict(int, data.get("games", {}))
        return obj
