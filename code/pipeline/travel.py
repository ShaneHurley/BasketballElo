"""Travel distance and timezone fatigue features."""
from __future__ import annotations

import math
from collections import defaultdict, deque

import numpy as np

# Arena coordinates (lat, lon) — approximate city centers
ARENA_COORDS = {
    "ATL": (33.757, -84.401), "BOS": (42.366, -71.062), "BKN": (40.683, -73.975),
    "CHA": (35.225, -80.839), "CHI": (41.881, -87.674), "CLE": (41.496, -81.688),
    "DAL": (32.790, -96.810), "DEN": (39.748, -105.007), "DET": (42.341, -83.055),
    "GSW": (37.768, -122.387), "HOU": (29.750, -95.362), "IND": (39.764, -86.155),
    "LAC": (34.043, -118.267), "LAL": (34.043, -118.267), "MEM": (35.138, -90.051),
    "MIA": (25.781, -80.188), "MIL": (43.045, -87.917), "MIN": (44.979, -93.276),
    "NOP": (29.949, -90.082), "NYK": (40.751, -73.993), "OKC": (35.463, -97.515),
    "ORL": (28.539, -81.384), "PHI": (39.901, -75.172), "PHX": (33.446, -112.071),
    "POR": (45.532, -122.667), "SAC": (38.649, -121.518), "SAS": (29.427, -98.438),
    "TOR": (43.643, -79.379), "UTA": (40.768, -111.901), "WAS": (38.898, -77.021),
}

# Historical / alternate tricodes → canonical keys in ARENA_COORDS / TZ_OFFSET.
_TEAM_ALIASES = {
    "BRK": "BKN", "NJN": "BKN", "NETS": "BKN",
    "PHO": "PHX", "SUNS": "PHX",
    "CHO": "CHA", "CHH": "CHA",
    "NOH": "NOP", "NO": "NOP", "NOR": "NOP",
    "GS": "GSW", "GOL": "GSW",
    "NY": "NYK", "NYC": "NYK",
    "SA": "SAS", "SAN": "SAS",
    "WSH": "WAS", "WIZ": "WAS",
    "UTH": "UTA", "UTAH": "UTA",
    "BK": "BKN",
}

TZ_OFFSET = {
    "ATL": -5, "BOS": -5, "BKN": -5, "CHA": -5, "CHI": -6, "CLE": -5,
    "DAL": -6, "DEN": -7, "DET": -5, "GSW": -8, "HOU": -6, "IND": -5,
    "LAC": -8, "LAL": -8, "MEM": -6, "MIA": -5, "MIL": -6, "MIN": -6,
    "NOP": -6, "NYK": -5, "OKC": -6, "ORL": -5, "PHI": -5, "PHX": -7,
    "POR": -8, "SAC": -8, "SAS": -6, "TOR": -5, "UTA": -7, "WAS": -5,
}

_DEFAULT_COORDS = (40.0, -90.0)
_DEFAULT_TZ = -6


def haversine_miles(lat1, lon1, lat2, lon2):
    r = 3959.0
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp = math.radians(lat2 - lat1)
    dl = math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _canon_team(team) -> str | None:
    if team is None:
        return None
    try:
        if isinstance(team, float) and np.isnan(team):
            return None
    except (TypeError, ValueError):
        pass
    s = str(team).strip().upper()
    if not s or s in {"NAN", "NONE", "NAT"}:
        return None
    s = _TEAM_ALIASES.get(s, s)
    return s


def arena_coords(team) -> tuple[float, float]:
    key = _canon_team(team)
    if key is None:
        return _DEFAULT_COORDS
    return ARENA_COORDS.get(key, _DEFAULT_COORDS)


def tz_offset(team) -> int:
    key = _canon_team(team)
    if key is None:
        return _DEFAULT_TZ
    return TZ_OFFSET.get(key, _DEFAULT_TZ)


class TravelTracker:
    """Rolling travel miles and timezone shifts per team."""

    def __init__(self, window_days: int = 14):
        self.window_days = window_days
        self.history = defaultdict(lambda: deque(maxlen=30))

    def _record(self, team, gdate, lat, lon, tz):
        key = _canon_team(team)
        if key is None:
            return
        self.history[key].append({"date": gdate, "lat": lat, "lon": lon, "tz": tz})

    def update_game(self, home, away, gdate, prev_home=None):
        hc = arena_coords(home)
        self._record(home, gdate, hc[0], hc[1], tz_offset(home))
        # Away plays at the home arena for this game.
        self._record(away, gdate, hc[0], hc[1], tz_offset(home))

    def features(self, team, gdate, is_home=True, venue_team=None):
        import pandas as pd

        key = _canon_team(team)
        hist = self.history.get(key, deque()) if key else deque()
        try:
            gdate = pd.Timestamp(gdate)
        except Exception:
            return self._zeros()

        miles_7 = miles_14 = tz_shift = trip_game = 0.0
        if len(hist) >= 1:
            prev = list(hist)[-1]
            # Destination for *this* tipoff: home arena (venue), not the team's own city
            # when the club is on the road.
            dest = arena_coords(venue_team if venue_team is not None else team)
            if not is_home:
                miles_7 = haversine_miles(prev["lat"], prev["lon"], dest[0], dest[1])
            tz_shift = abs(
                tz_offset(venue_team if venue_team is not None else team)
                - prev.get("tz", _DEFAULT_TZ)
            )

        recent = []
        for h in hist:
            try:
                if (gdate - pd.Timestamp(h["date"])).days <= 7:
                    recent.append(h)
            except Exception:
                continue
        trip_game = float(len(recent))
        for i in range(1, len(recent)):
            miles_14 += haversine_miles(
                recent[i - 1]["lat"], recent[i - 1]["lon"],
                recent[i]["lat"], recent[i]["lon"],
            )
        return {
            "travel_miles_7d": miles_7,
            "travel_miles_14d": miles_14,
            "tz_shift": tz_shift,
            "road_trip_index": trip_game,
        }

    @staticmethod
    def _zeros():
        return {"travel_miles_7d": 0.0, "travel_miles_14d": 0.0, "tz_shift": 0.0, "road_trip_index": 0.0}

    def matchup_features(self, home, away, gdate):
        hf = self.features(home, gdate, is_home=True, venue_team=home)
        af = self.features(away, gdate, is_home=False, venue_team=home)
        return {
            "h_travel_miles_7d": hf["travel_miles_7d"],
            "a_travel_miles_7d": af["travel_miles_7d"],
            "travel_miles_diff": hf["travel_miles_7d"] - af["travel_miles_7d"],
            "h_tz_shift": hf["tz_shift"],
            "a_tz_shift": af["tz_shift"],
            "h_road_trip": hf["road_trip_index"],
            "a_road_trip": af["road_trip_index"],
        }

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({k: list(v) for k, v in self.history.items()}, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        obj = cls()
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj.history = defaultdict(
            lambda: deque(maxlen=30),
            {k: deque(v, maxlen=30) for k, v in data.items()},
        )
        return obj
