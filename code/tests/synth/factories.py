"""Deterministic synthetic factories for the formula test bench (Epic 10.1).

Output schemas mirror what ``pipeline/stints.py`` / odds loaders produce so
bench failures mean formula bugs, not fake-data artifacts.
"""
from __future__ import annotations

from typing import Any, Iterable, Mapping, Optional, Sequence

import numpy as np
import pandas as pd


def make_player_ids(n: int = 5, *, start: int = 1, prefix: str = "") -> list[str]:
    """Return ``n`` player id strings suitable for HOME_players / AWAY_players."""
    return [f"{prefix}{start + i}" for i in range(n)]


def _lineup_string(player_ids: Sequence[str | int]) -> str:
    return "-".join(str(p) for p in player_ids if p is not None and str(p) != "nan")


def make_stint(
    *,
    game_id: str = "0022500001",
    home_players: Optional[Sequence[str | int]] = None,
    away_players: Optional[Sequence[str | int]] = None,
    possessions: float = 1.0,
    home_xpts: float = 1.1,
    away_xpts: float = 1.0,
    home_pts: Optional[float] = None,
    away_pts: Optional[float] = None,
    period: int = 1,
    home_score_start: float = 0.0,
    away_score_start: float = 0.0,
    home_score_end: Optional[float] = None,
    away_score_end: Optional[float] = None,
    home_team: str = "GSW",
    away_team: str = "LAL",
    game_date: Optional[pd.Timestamp] = None,
    garbage: bool = False,
    stint_id: int = 0,
    home_usage: Optional[Mapping[str, Any]] = None,
    away_usage: Optional[Mapping[str, Any]] = None,
    **extra: Any,
) -> dict[str, Any]:
    """One stint row schema-aligned with ``game_updates`` / ``stints`` consumers.

    Unrealistic values are intentional — the bench exists to stress formulas
    outside the narrow band of real NBA data.
    """
    hp = list(home_players) if home_players is not None else make_player_ids(5, start=1)
    ap = list(away_players) if away_players is not None else make_player_ids(5, start=6)
    h_pts = float(home_pts) if home_pts is not None else float(home_xpts)
    a_pts = float(away_pts) if away_pts is not None else float(away_xpts)
    h_end = float(home_score_end) if home_score_end is not None else float(home_score_start) + h_pts
    a_end = float(away_score_end) if away_score_end is not None else float(away_score_start) + a_pts
    row: dict[str, Any] = {
        "GAME_ID": str(game_id),
        "stint_id": int(stint_id),
        "PERIOD": int(period),
        "HOME_players": _lineup_string(hp),
        "AWAY_players": _lineup_string(ap),
        "possessions": float(possessions),
        "home_xpts": float(home_xpts),
        "away_xpts": float(away_xpts),
        "home_pts": h_pts,
        "away_pts": a_pts,
        "HOME_SCORE_START": float(home_score_start),
        "AWAY_SCORE_START": float(away_score_start),
        "HOME_SCORE_END": h_end,
        "AWAY_SCORE_END": a_end,
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date if game_date is not None else pd.Timestamp("2025-11-01"),
        "garbage": bool(garbage),
        "home_usage": dict(home_usage) if home_usage else {},
        "away_usage": dict(away_usage) if away_usage else {},
    }
    row.update(extra)
    return row


def make_stint_frame(
    stints: Iterable[Mapping[str, Any]] | None = None,
    *,
    n: int = 1,
    seed: int = 0,
    **kwargs: Any,
) -> pd.DataFrame:
    """Build a stint DataFrame; if ``stints`` is None, synthesize ``n`` rows."""
    if stints is not None:
        return pd.DataFrame(list(stints))
    rng = np.random.default_rng(seed)
    rows = []
    h_score = 0.0
    a_score = 0.0
    for i in range(n):
        hx = float(rng.uniform(0.5, 2.5))
        ax = float(rng.uniform(0.5, 2.5))
        poss = float(rng.uniform(0.5, 3.0))
        row = make_stint(
            stint_id=i,
            possessions=poss,
            home_xpts=hx,
            away_xpts=ax,
            home_score_start=h_score,
            away_score_start=a_score,
            **kwargs,
        )
        h_score = float(row["HOME_SCORE_END"])
        a_score = float(row["AWAY_SCORE_END"])
        rows.append(row)
    return pd.DataFrame(rows)


def make_game(
    *,
    game_id: str = "0022500001",
    home_team: str = "GSW",
    away_team: str = "LAL",
    game_date: Optional[pd.Timestamp] = None,
    home_score: float = 110.0,
    away_score: float = 105.0,
    market_spread: float = -3.5,
    closing_spread: Optional[float] = None,
    market_total: float = 220.0,
    market_ml_home: float = -150.0,
    market_ml_away: Optional[float] = None,
    juice: float = -110.0,
    season: int = 2026,
    **extra: Any,
) -> dict[str, Any]:
    """One game-level feature/odds row for end-to-end bench scenarios."""
    row: dict[str, Any] = {
        "GAME_ID": str(game_id),
        "home_team": home_team,
        "away_team": away_team,
        "game_date": game_date if game_date is not None else pd.Timestamp("2025-11-01"),
        "ACTUAL_HOME": float(home_score),
        "ACTUAL_AWAY": float(away_score),
        "ACTUAL_MARGIN": float(home_score) - float(away_score),
        "MARKET_SPREAD": float(market_spread),
        "CLOSING_SPREAD": float(closing_spread) if closing_spread is not None else float(market_spread),
        "MARKET_TOTAL": float(market_total),
        "MARKET_ML_HOME": float(market_ml_home),
        "MARKET_ML_AWAY": float(market_ml_away) if market_ml_away is not None else np.nan,
        "JUICE": float(juice),
        "SPREAD_PRICE": float(juice),
        "season": int(season),
    }
    row.update(extra)
    return row


def make_odds(
    *,
    ml_home: float = -150.0,
    ml_away: Optional[float] = 130.0,
    spread: float = -3.5,
    total: float = 220.0,
    juice: float = -110.0,
    decision_spread: Optional[float] = None,
    closing_spread: Optional[float] = None,
    one_sided: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Odds quote dict. ``one_sided=True`` omits away ML (P0.3 stress case)."""
    out: dict[str, Any] = {
        "market_ml_home": float(ml_home),
        "market_ml_away": np.nan if one_sided else (float(ml_away) if ml_away is not None else np.nan),
        "market_spread": float(spread),
        "market_total": float(total),
        "juice": float(juice),
        "decision_spread": float(decision_spread) if decision_spread is not None else float(spread),
        "closing_spread": float(closing_spread) if closing_spread is not None else float(spread),
    }
    out.update(extra)
    return out


def make_player_season(
    *,
    player_id: str = "1",
    games: int = 82,
    poss_per_game: float = 60.0,
    start_rating: float = 1500.0,
    seed: int = 0,
) -> dict[str, Any]:
    """Synthetic player season ledger for rating-update stress (e.g. 5000-game vet)."""
    rng = np.random.default_rng(seed)
    return {
        "player_id": str(player_id),
        "games": int(games),
        "poss_per_game": float(poss_per_game),
        "start_rating": float(start_rating),
        "xppp_samples": rng.normal(1.10, 0.15, size=max(games, 1)).tolist(),
    }
