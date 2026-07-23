"""Injury/availability and missing rotation impact."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Set

import numpy as np

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False

STATUS_PROB = {
    "OUT": 0.0, "DOUBTFUL": 0.10, "QUESTIONABLE": 0.50, "PROBABLE": 0.85, "GTD": 0.50,
}


@dataclass
class PlayerAvailability:
    player_id: str
    status: str
    play_prob: float


def status_to_prob(status: str) -> float:
    s = (status or "").upper()
    for key, prob in STATUS_PROB.items():
        if key in s:
            return prob
    return 1.0


def missing_rotation_share(expected_weights: Iterable[tuple], unavailable: Set[str]) -> float:
    """Fraction of expected rotation possessions missing tonight."""
    if not expected_weights:
        return 0.0
    total = sum(w for _, w in expected_weights)
    if total <= 0:
        return 0.0
    missing = sum(w for pid, w in expected_weights if str(pid) in unavailable)
    return float(missing / total)


def star_out_flag(expected_weights: Iterable[tuple], unavailable: Set[str], top_n: int = 2) -> int:
    if not expected_weights:
        return 0
    ranked = sorted(expected_weights, key=lambda x: -x[1])[:top_n]
    for pid, _ in ranked:
        if str(pid) in unavailable:
            return 1
    return 0


def filter_available_weights(expected_weights: List[tuple], unavailable: Set[str],
                             status_map: Optional[Dict[str, float]] = None) -> List[tuple]:
    """Return weights scaled by play probability."""
    out = []
    for pid, w in expected_weights:
        pid = str(pid)
        if pid in unavailable:
            continue
        prob = 1.0
        if status_map and pid in status_map:
            prob = status_map[pid]
        if prob > 0.05:
            out.append((pid, w * prob))
    return out


def fetch_espn_injuries() -> Dict[str, str]:
    """Fetch current NBA injury list from ESPN (best-effort)."""
    if not HAS_REQUESTS:
        return {}
    try:
        url = "https://site.api.espn.com/apis/site/v2/sports/basketball/nba/injuries"
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        data = r.json()
        out = {}
        for team in data.get("injuries", []):
            for inj in team.get("injuries", []):
                name = inj.get("athlete", {}).get("displayName", "")
                status = inj.get("status", "OUT")
                if name:
                    out[name.lower()] = status
        return out
    except Exception:
        return {}


def build_inactive_map_from_pbp(stints_df, game_id_col="GAME_ID") -> dict:
    """Historical proxy: players in rotation history but not in first stint = inactive."""
    return {}

def probabilistic_margin(mean_margin: float, std_margin: float, n_draws: int = 200, seed: int = 42):
    """Simple normal draws for lineup uncertainty."""
    rng = np.random.default_rng(seed)
    draws = rng.normal(mean_margin, max(std_margin, 1.0), n_draws)
    return float(np.mean(draws)), float(np.std(draws))


def expected_availability_impact(
    home_ids,
    away_ids,
    epm_tracker,
    *,
    h_star_out: float = 0,
    a_star_out: float = 0,
    h_missing: float = 0,
    a_missing: float = 0,
    as_of=None,
) -> dict:
    """Estimate offensive/defensive/pace drops from missing stars + EPM priors.

    Binary star-out alone understates impact; scale by prior strength and
    missing-rotation share. `as_of` (Task 032) is forwarded to the EPM
    tracker so a future-dated snapshot cannot influence this game's impact
    estimate.
    """
    def _side_impact(ids, star_out, missing):
        impacts = []
        if epm_tracker is not None and getattr(epm_tracker, "scaled_impact", None):
            impacts = [epm_tracker.scaled_impact(p, as_of=as_of) for p in (ids or []) if p]
        mean_abs = float(np.mean(np.abs(impacts))) if impacts else 25.0
        star = float(star_out or 0)
        miss = float(missing or 0)
        # Points of rating drop (approx ORtg/DRtg units)
        off_drop = star * (0.35 * mean_abs / 50.0) * 4.0 + miss * 3.0
        def_drop = star * (0.25 * mean_abs / 50.0) * 3.0 + miss * 2.0
        # Missing stars often slow or speed pace slightly; bias toward slight slowdown
        pace_delta = -star * 1.5 - miss * 0.8
        return off_drop, def_drop, pace_delta

    h_off, h_def, h_pace = _side_impact(home_ids, h_star_out, h_missing)
    a_off, a_def, a_pace = _side_impact(away_ids, a_star_out, a_missing)
    return {
        "h_expected_off_drop": float(h_off),
        "a_expected_off_drop": float(a_off),
        "h_expected_def_drop": float(h_def),
        "a_expected_def_drop": float(a_def),
        "h_expected_pace_delta": float(h_pace),
        "a_expected_pace_delta": float(a_pace),
        "expected_off_drop_diff": float(h_off - a_off),
        "expected_def_drop_diff": float(h_def - a_def),
        "expected_pace_delta_diff": float(h_pace - a_pace),
    }
