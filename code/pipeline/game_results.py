"""Task 006/007: canonical one-row-per-game outcome table.

Builds a canonical `GAME_ID`-level outcome table directly from raw
play-by-play, **before** any lineup filtering, garbage-time weighting, or
possession-based stint aggregation. This is the single source of truth for
final scores and game identity used by labels, grading, MAE, totals, and
calibration-outcome paths (Task 014). Weighted stint statistics (possessions,
xPoints, garbage-time-downweighted points) remain a *separate* input used
only for ratings/features, never for labels.

Final scores are independently verified two ways (Task 007):
  1. ``max(home_score)`` / ``max(away_score)`` over all raw PBP rows for the
     game (running-score columns are monotonic non-decreasing within a game).
  2. ``sum(points)`` grouped by the scoring ``team``, mapped to home/away via
     the game's ``home_team``/``away_team`` columns.
Both methods must agree exactly for every game; any mismatch is reported by
GAME_ID and treated as a hard failure (Rule 5: never silently invent a
score).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

__all__ = [
    "CanonicalScoreMismatch",
    "build_canonical_games",
    "verify_canonical_scores",
    "load_canonical_games_from_csv",
    "canonical_finals_map",
    "attach_canonical_finals",
]


class CanonicalScoreMismatch(RuntimeError):
    """Raised when the two independent final-score methods disagree."""

    def __init__(self, mismatched_game_ids: list):
        self.mismatched_game_ids = list(mismatched_game_ids)
        super().__init__(
            f"{len(self.mismatched_game_ids)} game(s) failed independent score "
            f"verification: {self.mismatched_game_ids[:25]}"
            + (" ..." if len(self.mismatched_game_ids) > 25 else "")
        )


_REQUIRED_RAW_COLUMNS = (
    "game_id", "date", "team", "home_team", "away_team",
    "points", "home_score", "away_score",
)


def build_canonical_games(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Build the canonical one-row-per-game table from raw 2025-26 PBP.

    Parameters
    ----------
    raw_df : the raw combined-stats CSV loaded as-is (lower_snake_case
        columns: game_id, date, team, home_team, away_team, points,
        home_score, away_score, data_set, ...), i.e. *before*
        ``convert_new_pbp``/``preprocess_pbp``/``build_stints``.

    Returns
    -------
    DataFrame with one row per GAME_ID: GAME_ID, home_team, away_team,
    season_type, raw_date, tip_utc (NaN — not present in this raw source),
    final_home_score, final_away_score (method 1: max running score),
    final_home_score_check, final_away_score_check (method 2: summed team
    points), scores_match.
    """
    missing = [c for c in _REQUIRED_RAW_COLUMNS if c not in raw_df.columns]
    if missing:
        raise ValueError(f"raw_df missing required columns for canonical games: {missing}")

    df = raw_df[list(_REQUIRED_RAW_COLUMNS) + (["data_set"] if "data_set" in raw_df.columns else [])].copy()

    # Task 008: explicit mixed-format parsing — never let a parseable raw
    # date silently become NaT.
    df["_raw_date_parsed"] = pd.to_datetime(df["date"], format="mixed", errors="coerce")

    df["points"] = pd.to_numeric(df["points"], errors="coerce").fillna(0)
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")

    game_meta = (
        df.groupby("game_id", sort=False)
        .agg(
            home_team=("home_team", "first"),
            away_team=("away_team", "first"),
            raw_date=("_raw_date_parsed", "first"),
            season_type=("data_set", "first") if "data_set" in df.columns else ("game_id", "first"),
            final_home_score=("home_score", "max"),
            final_away_score=("away_score", "max"),
        )
    )
    if "data_set" not in df.columns:
        game_meta["season_type"] = "unknown"

    # Method 2: sum points by scoring team, mapped to home/away.
    pts_by_team = df.groupby(["game_id", "team"], sort=False)["points"].sum()

    def _team_points(game_id, team):
        if pd.isna(team):
            return 0.0
        try:
            return float(pts_by_team.get((game_id, team), 0.0))
        except TypeError:
            return 0.0

    game_meta = game_meta.reset_index().rename(columns={"game_id": "GAME_ID"})
    game_meta["final_home_score_check"] = [
        _team_points(gid, ht) for gid, ht in zip(game_meta["GAME_ID"], game_meta["home_team"])
    ]
    game_meta["final_away_score_check"] = [
        _team_points(gid, at) for gid, at in zip(game_meta["GAME_ID"], game_meta["away_team"])
    ]
    game_meta["scores_match"] = (
        (game_meta["final_home_score"] == game_meta["final_home_score_check"])
        & (game_meta["final_away_score"] == game_meta["final_away_score_check"])
    )
    game_meta["tip_utc"] = pd.NaT  # not present in this raw source; leave explicitly missing

    cols = [
        "GAME_ID", "home_team", "away_team", "season_type", "raw_date", "tip_utc",
        "final_home_score", "final_away_score",
        "final_home_score_check", "final_away_score_check", "scores_match",
    ]
    out = game_meta[cols].reset_index(drop=True)
    return out


def verify_canonical_scores(canonical_df: pd.DataFrame) -> list:
    """Return the list of GAME_IDs where the two independent score methods
    disagree. Empty list means every game passed verification."""
    bad = canonical_df.loc[~canonical_df["scores_match"], "GAME_ID"]
    return bad.tolist()


def load_canonical_games_from_csv(path: Path | str) -> pd.DataFrame:
    """Read the raw 2025-26 combined-stats CSV and build canonical games."""
    try:
        raw = pd.read_csv(path, engine="pyarrow")
    except Exception:
        raw = pd.read_csv(path, low_memory=False)
    return build_canonical_games(raw)


def canonical_finals_map(canonical_df: pd.DataFrame) -> dict:
    """Return {GAME_ID: (final_home_score, final_away_score)} for verified games only."""
    verified = canonical_df[canonical_df["scores_match"]]
    return {
        row.GAME_ID: (float(row.final_home_score), float(row.final_away_score))
        for row in verified.itertuples(index=False)
    }


def attach_canonical_finals(stints_df: pd.DataFrame, canonical_df: pd.DataFrame) -> pd.DataFrame:
    """Broadcast canonical (verified-only) finals onto every stint row for a
    matching GAME_ID as ``canonical_home_pts``/``canonical_away_pts``.

    Games not present in `canonical_df` (or that failed verification) get
    NaN in these columns; downstream label consumers must fall back to
    stint-summed points only in that case, never silently otherwise.
    """
    verified = canonical_df[canonical_df["scores_match"]][
        ["GAME_ID", "final_home_score", "final_away_score"]
    ].rename(columns={
        "final_home_score": "canonical_home_pts",
        "final_away_score": "canonical_away_pts",
    })
    out = stints_df.merge(verified, on="GAME_ID", how="left")
    return out
