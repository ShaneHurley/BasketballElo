"""Composite fatigue index from schedule density and travel."""
from __future__ import annotations

import numpy as np


def fatigue_index(
    games_last7: float,
    travel_miles_7d: float,
    tz_shift: float,
    *,
    w_density: float = 0.35,
    w_travel: float = 0.35,
    w_tz: float = 0.30,
) -> float:
    density = float(games_last7) / 7.0
    travel = float(travel_miles_7d) / 2000.0
    tz = abs(float(tz_shift)) / 3.0
    return float(w_density * density + w_travel * travel + w_tz * tz)


def fatigue_features(h_games_last7, a_games_last7, h_travel_miles, a_travel_miles,
                     h_tz_shift, a_tz_shift) -> dict:
    h_f = fatigue_index(h_games_last7, h_travel_miles, h_tz_shift)
    a_f = fatigue_index(a_games_last7, a_travel_miles, a_tz_shift)
    return {
        "h_fatigue_index": h_f,
        "a_fatigue_index": a_f,
        "fatigue_diff": h_f - a_f,
    }
