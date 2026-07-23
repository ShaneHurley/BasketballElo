"""Rolling team margin-surprise volatility for stake sizing."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class TeamVolatilityTracker:
    def __init__(self, window: int = 20, min_games: int = 8):
        self.window = window
        self.min_games = min_games
        self.errors = defaultdict(lambda: deque(maxlen=window))
        self._league_sigma = 10.0

    def update(self, team: str, pred_margin: float, actual_margin: float, *, home: bool = True):
        """Record signed margin error from home perspective for one team."""
        if home:
            err = float(actual_margin) - float(pred_margin)
        else:
            err = float(-actual_margin) - float(-pred_margin)
        self.errors[team].append(err)
        all_errs = [e for q in self.errors.values() for e in q]
        if len(all_errs) >= self.min_games:
            self._league_sigma = float(np.std(all_errs, ddof=1)) if len(all_errs) > 1 else 10.0

    def update_matchup(self, home: str, away: str, pred_margin: float, actual_margin: float):
        self.update(home, pred_margin, actual_margin, home=True)
        self.update(away, pred_margin, actual_margin, home=False)

    def sigma(self, team: str) -> float:
        errs = list(self.errors.get(team, []))
        if len(errs) < self.min_games:
            return self._league_sigma
        return float(np.std(errs, ddof=1)) if len(errs) > 1 else self._league_sigma

    def matchup_sigma(self, home: str, away: str) -> float:
        return float(max(self.sigma(home), self.sigma(away)))

    def stake_multiplier(self, home: str, away: str) -> float:
        sig = self.matchup_sigma(home, away)
        lg = max(self._league_sigma, 1e-6)
        raw = 1.0 - 0.3 * (sig / lg - 1.0)
        return float(np.clip(raw, 0.5, 1.5))

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "window": self.window,
                "min_games": self.min_games,
                "errors": {k: list(v) for k, v in self.errors.items()},
                "_league_sigma": self._league_sigma,
            }, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(window=data["window"], min_games=data["min_games"])
        for k, v in data["errors"].items():
            obj.errors[k] = deque(v, maxlen=obj.window)
        obj._league_sigma = data.get("_league_sigma", 10.0)
        return obj
