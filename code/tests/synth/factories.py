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


def _default_usage(player_ids: Sequence[str | int]) -> dict[str, list[float]]:
    """Zeroed 3-slot usage counters per player, as ``build_stints`` emits."""
    return {
        str(p): [0.0, 0.0, 0.0]
        for p in player_ids
        if p is not None and str(p) and str(p) != "nan"
    }


#: Counting-stat columns summed by ``pipeline.stints.build_stints`` and read
#: downstream by ``pipeline.game_updates`` (e.g. ``{side}_rim_fga``,
#: ``{side}_3pm``, ``{side}_fta``). ``make_stint`` fills these with 0.0
#: unless overridden so factory rows stay schema-complete (Task 10.1.3).
STINT_BOX_SCORE_COLUMNS: tuple[str, ...] = (
    "home_oreb", "away_oreb", "home_dreb", "away_dreb",
    "home_tovs_forced", "away_tovs_forced",
    "home_fouls_drawn", "away_fouls_drawn",
    "home_fgm", "away_fgm", "home_fga", "away_fga",
    "home_tov", "away_tov", "home_fta", "away_fta",
    "home_stl", "away_stl", "home_blks", "away_blks",
    "home_3pm", "away_3pm", "home_3pa", "away_3pa",
    "home_poss", "away_poss",
    "home_xefg_sum", "away_xefg_sum",
    "home_rim_fga", "away_rim_fga",
    "home_three_fga", "away_three_fga",
    "home_shot_dist_sum", "away_shot_dist_sum",
)


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
        "home_usage": dict(home_usage) if home_usage is not None else _default_usage(hp),
        "away_usage": dict(away_usage) if away_usage is not None else _default_usage(ap),
    }
    row.update(extra)
    # Schema parity with ``pipeline.stints.build_stints`` output: counting
    # stats default to 0.0 unless explicitly supplied via kwargs/extra.
    for col in STINT_BOX_SCORE_COLUMNS:
        row.setdefault(col, 0.0)
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
    spread_open: Optional[float] = None,
    spread_move: Optional[float] = None,
    public_home_pct: Optional[float] = None,
    one_sided: bool = False,
    **extra: Any,
) -> dict[str, Any]:
    """Odds quote dict. ``one_sided=True`` omits away ML (P0.3 stress case).

    Keys mirror what ``pipeline.market`` feature builders consume:
    ``market_*`` prices plus line-movement fields (``spread_open`` /
    ``spread_move`` / ``public_home_pct``). ``spread_move`` follows the
    ``market.py`` convention (decision line minus open) and defaults to 0.0
    when no distinct open is supplied.
    """
    dec = float(decision_spread) if decision_spread is not None else float(spread)
    clo = float(closing_spread) if closing_spread is not None else float(spread)
    opn = float(spread_open) if spread_open is not None else float(spread)
    move = float(spread_move) if spread_move is not None else dec - opn
    out: dict[str, Any] = {
        "market_ml_home": float(ml_home),
        "market_ml_away": np.nan if one_sided else (float(ml_away) if ml_away is not None else np.nan),
        "market_spread": float(spread),
        "market_total": float(total),
        "juice": float(juice),
        "decision_spread": dec,
        "closing_spread": clo,
        "spread_open": opn,
        "spread_move": move,
        "public_home_pct": float(public_home_pct) if public_home_pct is not None else np.nan,
    }
    out.update(extra)
    return out


#: Default tricode pool for the schedule builders.
DEFAULT_TEAMS: tuple[str, ...] = ("ATL", "BOS", "CHI", "DAL", "GSW", "LAL", "NYK", "PHX")


def _validate_teams(teams: Sequence[str]) -> list[str]:
    team_list = [str(t) for t in teams]
    if len(team_list) < 2:
        raise ValueError("need at least 2 teams to schedule games")
    if len(set(team_list)) != len(team_list):
        raise ValueError("teams must be unique — duplicate identities would allow self-games")
    return team_list


def _game_id(season: int, ordinal: int) -> str:
    """NBA-style game id: ``002`` + season-start year suffix + 5-digit ordinal."""
    return f"002{str(int(season) - 1)[-2:]}{int(ordinal):05d}"


def _ml_pair_from_spread(spread: float) -> tuple[float, float]:
    """Crude spread→American-ML map for synthetic lines (spread<0 ⇒ home favored)."""
    pts = abs(float(spread))
    fav = -float(round(110 + 25 * pts))
    dog = float(round(100 + 20 * pts))
    return (fav, dog) if spread < 0 else (dog, fav)


def _random_game_row(
    rng: np.random.Generator,
    *,
    game_id: str,
    home_team: str,
    away_team: str,
    game_date: pd.Timestamp,
    season: int,
) -> dict[str, Any]:
    """One ``make_game`` row with plausible randomized score/lines from ``rng``."""
    h = int(round(float(rng.normal(112.0, 11.0))))
    a = int(round(float(rng.normal(110.0, 11.0))))
    while a == h:  # NBA games never end tied
        a = int(round(float(rng.normal(110.0, 11.0))))
    spread = float(round(float(rng.normal(-2.5, 5.5)) * 2) / 2.0)
    total = float(round(float(rng.normal(223.5, 10.0)) * 2) / 2.0)
    ml_home, ml_away = _ml_pair_from_spread(spread)
    return make_game(
        game_id=game_id,
        home_team=home_team,
        away_team=away_team,
        game_date=game_date,
        home_score=h,
        away_score=a,
        market_spread=spread,
        market_total=total,
        market_ml_home=ml_home,
        market_ml_away=ml_away,
        season=season,
    )


def make_game_sequence(
    n_games: int = 10,
    teams: Sequence[str] = DEFAULT_TEAMS,
    start_date: Any = "2025-10-01",
    seed: int = 0,
    *,
    season: int = 2026,
) -> list[dict[str, Any]]:
    """Chronologically ordered list of synthetic game dicts (``make_game`` rows).

    The first game is on ``start_date`` and each subsequent game is 1–3 days
    later. Home/away are drawn without replacement per game, so no team ever
    plays itself. Two calls with the same ``seed`` produce identical output.
    """
    team_list = _validate_teams(teams)
    rng = np.random.default_rng(seed)
    current = pd.Timestamp(start_date)
    games: list[dict[str, Any]] = []
    for i in range(int(n_games)):
        if i:
            current = current + pd.Timedelta(days=int(rng.integers(1, 4)))
        hi, ai = (int(x) for x in rng.choice(len(team_list), size=2, replace=False))
        games.append(
            _random_game_row(
                rng,
                game_id=_game_id(season, i + 1),
                home_team=team_list[hi],
                away_team=team_list[ai],
                game_date=current,
                season=season,
            )
        )
    return games


def _pair_slots(slots: list[str]) -> list[tuple[str, str]]:
    """Pair adjacent slots into matchups, repairing self-pairs by swaps.

    A self-pair-free pairing always exists when no team holds more than half
    the slots (true here: every team holds exactly ``n_games_per_team`` of
    ``len(teams) * n_games_per_team`` slots, and ``len(teams) >= 2``), and
    each swap strictly reduces the self-pair count without creating a new
    one, so the repair scan always succeeds.
    """
    slots = list(slots)

    def _partner(i: int) -> int:
        return i + 1 if i % 2 == 0 else i - 1

    for i in range(0, len(slots), 2):
        if slots[i] != slots[i + 1]:
            continue
        t = slots[i]
        for k in range(len(slots)):
            if k in (i, i + 1):
                continue
            if slots[k] != t and slots[_partner(k)] != t:
                slots[i + 1], slots[k] = slots[k], slots[i + 1]
                break
        else:
            raise ValueError("could not build a self-game-free pairing")
    return [(slots[p], slots[p + 1]) for p in range(0, len(slots), 2)]


def make_season(
    teams: Sequence[str] = DEFAULT_TEAMS,
    n_games_per_team: int = 12,
    seed: int = 0,
    *,
    start_date: Any = "2025-10-01",
    season: int = 2026,
) -> pd.DataFrame:
    """Full synthetic season as a DataFrame of ``make_game`` rows.

    Every team plays exactly ``n_games_per_team`` games, no team ever plays
    itself, and games are chronologically ordered 1–3 days apart starting
    from ``start_date``. Deterministic given ``seed``: two calls with the
    same seed produce identical frames.
    """
    team_list = _validate_teams(teams)
    n = int(n_games_per_team)
    if n < 1:
        raise ValueError("n_games_per_team must be >= 1")
    slots = [t for t in team_list for _ in range(n)]
    if len(slots) % 2:
        raise ValueError("len(teams) * n_games_per_team must be even")
    rng = np.random.default_rng(seed)
    slots = [slots[i] for i in rng.permutation(len(slots))]
    pairs = _pair_slots(slots)
    current = pd.Timestamp(start_date)
    rows: list[dict[str, Any]] = []
    for i, (ta, tb) in enumerate(pairs):
        if i:
            current = current + pd.Timedelta(days=int(rng.integers(1, 4)))
        home, away = (ta, tb) if rng.random() < 0.5 else (tb, ta)
        rows.append(
            _random_game_row(
                rng,
                game_id=_game_id(season, i + 1),
                home_team=home,
                away_team=away,
                game_date=current,
                season=season,
            )
        )
    return pd.DataFrame(rows)


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
