"""Target-family calibration with optional team partial-pooling intercepts.

One nested calibration path per family (spread cover, total O/U, moneyline).
Team effects are shrunk intercepts with a mostly global slope — never 30
independent isotonic curves or lineup-specific post-hoc calibrators.
"""
from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np
from sklearn.linear_model import LogisticRegression


TARGET_FAMILIES = (
    "spread_cover",
    "total_over",
    "moneyline",
)


def _logit(p: float, eps: float = 1e-6) -> float:
    p = float(np.clip(p, eps, 1.0 - eps))
    return float(np.log(p / (1.0 - p)))


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-float(x))))


@dataclass
class TeamPooledCalibrator:
    """Global Platt + empirical-Bayes team intercepts."""

    window: int = 400
    min_samples: int = 40
    team_shrink_k: float = 30.0
    buffer: deque = field(default_factory=lambda: deque(maxlen=400))
    team_resid: dict = field(default_factory=lambda: defaultdict(float))
    team_n: dict = field(default_factory=lambda: defaultdict(float))
    model: LogisticRegression | None = None
    fitted: bool = False

    def __post_init__(self):
        self.buffer = deque(maxlen=self.window)

    def update(self, raw_prob: float, outcome: int, team: str | None = None) -> None:
        self.buffer.append((float(raw_prob), int(outcome), team))
        if team is not None and self.model is not None:
            # Track residual in logit space after global calibration.
            cal = self.predict(raw_prob, team=None)
            resid = float(outcome) - cal
            n = self.team_n[team] + 1.0
            self.team_resid[team] = ((n - 1.0) * self.team_resid[team] + resid) / n
            self.team_n[team] = n
        if len(self.buffer) >= self.min_samples:
            self._fit()

    def _fit(self) -> None:
        X = np.array([_logit(p) for p, _, _ in self.buffer]).reshape(-1, 1)
        y = np.array([o for _, o, _ in self.buffer], dtype=int)
        if len(np.unique(y)) < 2:
            return
        clf = LogisticRegression(C=0.5, solver="lbfgs", max_iter=200)
        clf.fit(X, y)
        self.model = clf
        self.fitted = True

    def predict(self, raw_prob: float, team: str | None = None) -> float:
        if self.model is None or not self.fitted:
            return float(np.clip(raw_prob, 0.01, 0.99))
        base = float(self.model.predict_proba(np.array([[_logit(raw_prob)]]))[0, 1])
        if team is None:
            return float(np.clip(base, 0.01, 0.99))
        n = self.team_n.get(team, 0.0)
        resid = self.team_resid.get(team, 0.0)
        # Partial pool residual toward 0; convert approx probability shift.
        w = n / (n + self.team_shrink_k) if n > 0 else 0.0
        adj = base + w * resid * 0.5
        return float(np.clip(adj, 0.01, 0.99))


class TargetFamilyCalibrationRegistry:
    """Exactly one calibrator path per target family."""

    def __init__(self):
        self._cals: dict[str, TeamPooledCalibrator] = {
            t: TeamPooledCalibrator() for t in TARGET_FAMILIES
        }

    def get(self, family: str) -> TeamPooledCalibrator:
        if family not in self._cals:
            raise KeyError(f"Unknown calibration family {family!r}; expected one of {TARGET_FAMILIES}")
        return self._cals[family]

    def families(self) -> tuple:
        return TARGET_FAMILIES
