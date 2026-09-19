"""T-60 data-coverage gates before training/scoring betting heads."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from pipeline.config import MIN_T60_DECISION_COVERAGE, MIN_T60_SEASONS_FOR_BETTING
from pipeline.dates import gameid_season_start_year
from pipeline.market_targets import decision_spread_series, valid_closing_fraction


def nba_season_start_year_series(df: pd.DataFrame) -> pd.Series:
    """Map each row to NBA season *start* year (e.g. 2024 for 2024-25).

    Prefers GAME_ID encoding, then ``simulated_season_window`` / ``season``,
    then calendar date with Oct→Jun season boundary. Never uses bare
    ``game_date.dt.year`` (that under-counts seasons spanning two calendar years).
    """
    n = len(df)
    out = pd.Series(np.full(n, np.nan), index=df.index, dtype=float)

    gid_col = None
    for c in ("GAME_ID", "game_id"):
        if c in df.columns:
            gid_col = c
            break
    if gid_col is not None:
        for idx, gid in df[gid_col].items():
            y = gameid_season_start_year(gid)
            if y is not None:
                out.loc[idx] = float(y)

    if "simulated_season_window" in df.columns:
        # Values like "2025-2026" → start year 2025
        sw = df["simulated_season_window"].astype(str)
        parsed = sw.str.extract(r"^(\d{4})")[0]
        mask = out.isna() & parsed.notna()
        out.loc[mask] = parsed.loc[mask].astype(float) - 0  # already start year if "2025-2026"
        # If window is end-year convention ("2025-2026" means season ending 2026 → start 2025)
        # extract first year which is start year — correct as-is.

    if "season" in df.columns:
        seas = pd.to_numeric(df["season"], errors="coerce")
        mask = out.isna() & seas.notna()
        if mask.any():
            # Repo season labels are typically end-year (2025 = 2024-25). Convert
            # to start year. Values already < 2100 and consistent with GAME_ID
            # start years are left as-is when GAME_ID filled most rows.
            out.loc[mask] = seas.loc[mask] - 1.0

    date_col = None
    for c in ("game_date", "DATE", "date"):
        if c in df.columns:
            date_col = c
            break
    if date_col is not None:
        dt = pd.to_datetime(df[date_col], errors="coerce")
        # Oct–Dec → season start = year; Jan–Sep → season start = year - 1.
        start = np.where(dt.dt.month >= 10, dt.dt.year, dt.dt.year - 1)
        mask = out.isna() & dt.notna()
        out.loc[mask] = pd.Series(start, index=df.index).loc[mask].astype(float)

    return out


def season_decision_coverage(df: pd.DataFrame, season_col: str = "season") -> pd.DataFrame:
    """Per-NBA-season fraction of games with a usable T-60 decision spread."""
    if df is None or df.empty:
        return pd.DataFrame(columns=["season", "n", "decision_coverage"])
    work = df.copy()
    # Always re-key by NBA season start year for the gate (ignore calendar-year season).
    work["_nba_season_start"] = nba_season_start_year_series(work)
    if work["_nba_season_start"].isna().all():
        if season_col in work.columns:
            work["_nba_season_start"] = pd.to_numeric(work[season_col], errors="coerce")
        else:
            work["_nba_season_start"] = 0
    dec = decision_spread_series(work)
    rows = []
    for season, g in work.groupby("_nba_season_start"):
        if pd.isna(season):
            continue
        if dec is None:
            cov = 0.0
        else:
            sub = dec.loc[g.index]
            cov = float(np.isfinite(sub.to_numpy(dtype=float)).mean()) if len(sub) else 0.0
        rows.append({"season": int(season), "n": int(len(g)), "decision_coverage": cov})
    return pd.DataFrame(rows)


def t60_betting_data_gate(
    df: pd.DataFrame,
    *,
    min_seasons: int | None = None,
    min_coverage: float | None = None,
    season_col: str = "season",
) -> dict[str, Any]:
    """Return allow_betting flag + diagnostics.

    Seasons below coverage remain usable for market-free rating research only.
    """
    min_seasons = int(min_seasons if min_seasons is not None else MIN_T60_SEASONS_FOR_BETTING)
    min_coverage = float(min_coverage if min_coverage is not None else MIN_T60_DECISION_COVERAGE)
    cov = season_decision_coverage(df, season_col=season_col)
    adequate = cov[cov["decision_coverage"] >= min_coverage] if not cov.empty else cov
    n_adequate = int(len(adequate))
    overall = float(valid_closing_fraction(df)) if df is not None and len(df) else 0.0
    allow = n_adequate >= min_seasons and overall >= min_coverage
    return {
        "allow_betting_heads": bool(allow),
        "n_adequate_seasons": n_adequate,
        "min_seasons_required": min_seasons,
        "min_coverage_required": min_coverage,
        "overall_decision_coverage": overall,
        "per_season": cov.to_dict(orient="records"),
        "reason": None if allow else "insufficient_t60_decision_coverage",
    }
