"""Referee crew tendency features (rolling from historical assignments)."""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np


class RefTracker:
    """Track foul/pace bias by ref crew id when assignment data is available."""

    def __init__(self, window: int = 50):
        self.window = window
        self.crew_pace = defaultdict(lambda: deque(maxlen=window))
        self.crew_foul = defaultdict(lambda: deque(maxlen=window))

    def update(self, crew_id: str, total_pts: float, fta: float, possessions: float):
        if not crew_id or possessions <= 0:
            return
        pace = possessions * 2  # approx game pace
        foul_rate = fta / possessions
        self.crew_pace[crew_id].append(pace)
        self.crew_foul[crew_id].append(foul_rate)

    def features(self, crew_id: str = None):
        if not crew_id or crew_id not in self.crew_pace:
            return {"ref_pace_bias": 0.0, "ref_foul_bias": 0.0}
        pace = np.mean(self.crew_pace[crew_id]) if self.crew_pace[crew_id] else 0.0
        foul = np.mean(self.crew_foul[crew_id]) if self.crew_foul[crew_id] else 0.0
        return {
            "ref_pace_bias": pace - 100.0,
            "ref_foul_bias": foul - 0.25,
        }
