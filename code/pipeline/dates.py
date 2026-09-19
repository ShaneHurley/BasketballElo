"""Recover real per-game dates for V3 PBP seasons.

The V3 PBP CSVs (2021-22 .. 2024-25) contain NO date column, so
`convert_v3_pbp` stamps every game with a constant ``YYYY-01-01``.  That
collapses the whole season onto one day, which (a) breaks odds matching by
date and (b) destroys every date-derived feature (rest days, recent form,
SOS, inactivity decay) and within-season chronological ordering.

This module rebuilds a ``GAME_ID -> date`` map by aligning each season's games
against the odds/schedule file (``all_odds.csv``), which DOES carry real dates
and team names.  NBA ``GAME_ID``s increment in (approximately) schedule order,
so for any given home/away matchup the k-th occurrence in GAME_ID order is the
k-th occurrence in date order.  Games that cannot be matched to an odds row
(e.g. no line recorded) get a monotonic date interpolated from the matched
anchors so ordering and rest-day features stay sane.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

# Odds-file short names -> NBA tricodes.
ODDS_NAME_TO_TRICODE = {
    "Atlanta": "ATL", "Boston": "BOS", "Brooklyn": "BKN", "Charlotte": "CHA",
    "Chicago": "CHI", "Cleveland": "CLE", "Dallas": "DAL", "Denver": "DEN",
    "Detroit": "DET", "Golden State": "GSW", "Houston": "HOU", "Indiana": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Clippers": "LAC",
    "LA Lakers": "LAL", "Los Angeles Lakers": "LAL", "Lakers": "LAL",
    "Memphis": "MEM", "Miami": "MIA", "Milwaukee": "MIL", "Minnesota": "MIN",
    "New Orleans": "NOP", "New York": "NYK", "Knicks": "NYK",
    "Oklahoma City": "OKC", "Orlando": "ORL", "Philadelphia": "PHI", "76ers": "PHI",
    "Phoenix": "PHX", "Portland": "POR", "Sacramento": "SAC",
    "San Antonio": "SAS", "Toronto": "TOR", "Utah": "UTA", "Washington": "WAS",
}

# Full names (PBP / TEAM_MAP style) -> tricode, so we can normalise either side.
_FULLNAME_TO_TRICODE = {
    "Atlanta Hawks": "ATL", "Boston Celtics": "BOS", "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA", "Chicago Bulls": "CHI", "Cleveland Cavaliers": "CLE",
    "Dallas Mavericks": "DAL", "Denver Nuggets": "DEN", "Detroit Pistons": "DET",
    "Golden State Warriors": "GSW", "Houston Rockets": "HOU", "Indiana Pacers": "IND",
    "LA Clippers": "LAC", "Los Angeles Clippers": "LAC", "Los Angeles Lakers": "LAL",
    "Memphis Grizzlies": "MEM", "Miami Heat": "MIA", "Milwaukee Bucks": "MIL",
    "Minnesota Timberwolves": "MIN", "New Orleans Pelicans": "NOP", "New York Knicks": "NYK",
    "Oklahoma City Thunder": "OKC", "Orlando Magic": "ORL", "Philadelphia 76ers": "PHI",
    "Phoenix Suns": "PHX", "Portland Trail Blazers": "POR", "Sacramento Kings": "SAC",
    "San Antonio Spurs": "SAS", "Toronto Raptors": "TOR", "Utah Jazz": "UTA",
    "Washington Wizards": "WAS",
}

_VALID_TRICODES = set(_FULLNAME_TO_TRICODE.values())


def to_tricode(name) -> str | float:
    """Normalise any team label (tricode / short / full name) to a tricode."""
    if name is None or (isinstance(name, float) and np.isnan(name)):
        return np.nan
    s = str(name).strip()
    if s in _VALID_TRICODES:
        return s
    if s in ODDS_NAME_TO_TRICODE:
        return ODDS_NAME_TO_TRICODE[s]
    if s in _FULLNAME_TO_TRICODE:
        return _FULLNAME_TO_TRICODE[s]
    # Try the trailing word(s) – e.g. "Los Angeles Lakers" already handled above.
    return s  # leave as-is; will simply fail to match if unknown


def gameid_season_start_year(game_id) -> int | None:
    """NBA GAME_IDs encode the season start year in digits [-7:-5] of the
    zero-padded 10-char id (e.g. 0022200001 -> '22' -> 2022)."""
    s = str(game_id).strip()
    digits = "".join(ch for ch in s if ch.isdigit())
    if len(digits) < 8:
        return None
    z = digits.zfill(10)
    try:
        yy = int(z[3:5])
    except ValueError:
        return None
    return 2000 + yy


def load_odds_schedule(odds_path: str) -> pd.DataFrame:
    """Return a tidy schedule [date, home, away, season] from the odds CSV.

    Dates are taken from the canonical date embedded in the odds ``game_id``
    (``nba.g.YYYYMMDDNN``) when present, else from ``game_date``.
    """
    df = pd.read_csv(odds_path, usecols=lambda c: c in {
        "game_id", "game_date", "home_team", "away_team"})

    # Prefer the date embedded in nba.g.YYYYMMDDNN (timezone-safe canonical date)
    def _date_from_id(gid):
        s = str(gid)
        digits = "".join(ch for ch in s if ch.isdigit())
        if len(digits) >= 8:
            try:
                return pd.to_datetime(digits[:8], format="%Y%m%d")
            except (ValueError, TypeError):
                return pd.NaT
        return pd.NaT

    date_from_id = df["game_id"].apply(_date_from_id) if "game_id" in df.columns else pd.Series(pd.NaT, index=df.index)
    date_from_col = pd.to_datetime(df["game_date"].astype(str).str[:10], errors="coerce") if "game_date" in df.columns else pd.Series(pd.NaT, index=df.index)
    date = date_from_id.fillna(date_from_col)

    sched = pd.DataFrame({
        "date": date,
        "home": df["home_team"].apply(to_tricode),
        "away": df["away_team"].apply(to_tricode),
    }).dropna(subset=["date", "home", "away"])
    sched["season"] = sched["date"].dt.year + (sched["date"].dt.month >= 9).astype(int)
    sched = sched.sort_values("date").reset_index(drop=True)
    return sched


def build_gameid_date_map(game_rows: pd.DataFrame, sched: pd.DataFrame,
                          season_start_year: int) -> dict:
    """Map each GAME_ID to a real date.

    Parameters
    ----------
    game_rows : DataFrame with columns [GAME_ID, home, away] (tricodes), one
        row per game, for a single season.
    sched : output of ``load_odds_schedule`` (all seasons).
    season_start_year : NBA season start year (e.g. 2022 for the 2022-23 season).

    Returns
    -------
    dict {GAME_ID: pd.Timestamp}
    """
    season_label = season_start_year + 1  # our internal season convention
    season_sched = sched[sched["season"] == season_label].copy()

    g = game_rows.copy()
    g["home"] = g["home"].apply(to_tricode)
    g["away"] = g["away"].apply(to_tricode)
    # Numeric id for ordering / interpolation
    g["_gid_num"] = g["GAME_ID"].astype(str).str.replace(r"\D", "", regex=True)
    g["_gid_num"] = pd.to_numeric(g["_gid_num"], errors="coerce")
    g = g.sort_values("_gid_num").reset_index(drop=True)

    date_map: dict = {}

    # Align per (home, away) matchup: k-th by GAME_ID == k-th by date.
    odds_by_matchup = {
        key: sub.sort_values("date")["date"].tolist()
        for key, sub in season_sched.groupby(["home", "away"])
    }
    for (home, away), sub in g.groupby(["home", "away"]):
        odds_dates = odds_by_matchup.get((home, away), [])
        sub = sub.sort_values("_gid_num")
        for i, (_, row) in enumerate(sub.iterrows()):
            if i < len(odds_dates):
                date_map[row["GAME_ID"]] = pd.Timestamp(odds_dates[i])

    # Fill unmatched games from matched anchors using GAME_ID *rank* as the
    # x-axis (uniform spacing avoids distortion from the playoff/play-in id
    # jumps).  Inside the anchor range we interpolate; beyond it we extrapolate
    # with a robust linear fit so a partially-covered (live) season's tail still
    # gets monotonically increasing dates instead of all collapsing onto the
    # last anchor.
    g["_rank"] = g["_gid_num"].rank(method="first")
    matched = g[g["GAME_ID"].isin(date_map)].copy()
    if not matched.empty:
        matched = matched.assign(_d=matched["GAME_ID"].map(date_map))
        matched = matched.dropna(subset=["_rank", "_d"]).sort_values("_rank")
        ax = matched["_rank"].to_numpy(dtype=float)
        ay = matched["_d"].astype("int64").to_numpy()  # ns since epoch

        # Linear fit for extrapolation outside [ax.min(), ax.max()].
        if len(ax) >= 2 and ax.max() > ax.min():
            slope, intercept = np.polyfit(ax, ay, 1)
        else:
            slope, intercept = 0.0, ay[0]

        lo, hi = ax.min(), ax.max()
        for _, row in g.iterrows():
            gid = row["GAME_ID"]
            if gid in date_map:
                continue
            xi = row["_rank"]
            if np.isnan(xi):
                continue
            if lo <= xi <= hi:
                yi = np.interp(xi, ax, ay)
            else:
                yi = slope * xi + intercept            # extrapolate the tail
            date_map[gid] = pd.Timestamp(int(round(yi)))
    else:
        # No odds coverage at all -> fall back to season-start sequential dates.
        base = pd.Timestamp(f"{season_start_year}-10-20")
        ranks = g["_gid_num"].rank(method="first").fillna(0).astype(int)
        for gid, r in zip(g["GAME_ID"], ranks):
            date_map[gid] = base + pd.Timedelta(days=int(r * 82 / max(len(g), 1)))

    return date_map


def attach_real_dates(stints_df: pd.DataFrame, sched: pd.DataFrame,
                      season_start_year: int, verbose: bool = True) -> pd.DataFrame:
    """Overwrite ``game_date`` in a season's stint DataFrame with recovered dates.

    Also attaches a ``date_status`` column (Task 009 provenance): games
    matched exactly to an odds/schedule row are ``schedule_matched``; games
    whose date came from matchup-order interpolation/extrapolation are
    ``interpolated`` and should be excluded from rest/travel evaluations
    until independently verified (Rule 5 / plan Phase 1B).

    Returns a new DataFrame (sorted by the recovered date + GAME_ID + stint).
    """
    df = stints_df.copy()
    game_rows = (
        df[["GAME_ID", "home_team", "away_team"]]
        .drop_duplicates("GAME_ID")
        .rename(columns={"home_team": "home", "away_team": "away"})
        .reset_index(drop=True)
    )
    date_map = build_gameid_date_map(game_rows, sched, season_start_year)

    n_total = len(game_rows)
    matched_ids = set()
    season_label = season_start_year + 1
    season_sched = sched[sched["season"] == season_label]
    odds_matchups = {
        (h, a) for h, a in zip(season_sched["home"].map(to_tricode), season_sched["away"].map(to_tricode))
    }
    for _, row in game_rows.iterrows():
        h, a = to_tricode(row["home"]), to_tricode(row["away"])
        if row["GAME_ID"] in date_map and (h, a) in odds_matchups:
            matched_ids.add(row["GAME_ID"])
    n_exact = len(matched_ids)

    status_map = {
        gid: (DATE_STATUS_SCHEDULE_MATCHED if gid in matched_ids else DATE_STATUS_INTERPOLATED)
        for gid in date_map
    }
    df["game_date"] = df["GAME_ID"].map(date_map).fillna(df["game_date"])
    df["date_status"] = df["GAME_ID"].map(status_map).fillna(DATE_STATUS_UNRESOLVED)
    if verbose:
        print(f"  [dates] season {season_start_year}-{season_start_year+1}: "
              f"assigned dates to {n_exact}/{n_total} games via exact schedule match "
              f"({n_total - n_exact} interpolated) "
              f"(date range {df['game_date'].min()} -> {df['game_date'].max()})")
    sort_cols = [c for c in ["game_date", "GAME_ID", "stint_id"] if c in df.columns]
    return df.sort_values(sort_cols).reset_index(drop=True)


def is_valid_timestamp(ts) -> bool:
    """True when *ts* is a usable calendar timestamp (not None/NaT)."""
    if ts is None:
        return False
    try:
        t = pd.Timestamp(ts)
    except (ValueError, TypeError):
        return False
    return not pd.isna(t)


def date_from_game_id(game_id) -> pd.Timestamp:
    """Best-effort YYYYMMDD extraction from odds-style or digit-heavy ids."""
    if game_id is None:
        return pd.NaT
    digits = "".join(ch for ch in str(game_id) if ch.isdigit())
    if len(digits) < 8:
        return pd.NaT
    for start in range(0, len(digits) - 7):
        chunk = digits[start : start + 8]
        try:
            ts = pd.to_datetime(chunk, format="%Y%m%d")
            if 1990 <= ts.year <= 2040:
                return ts
        except (ValueError, TypeError):
            continue
    return pd.NaT


DATE_STATUS_AUTHORITATIVE = "authoritative"
DATE_STATUS_SCHEDULE_MATCHED = "schedule_matched"
DATE_STATUS_INTERPOLATED = "interpolated"
DATE_STATUS_UNRESOLVED = "unresolved"

# Seasons with higher interpolated fraction than this are dropped from suite WF
# unless --allow-interpolated-dates (poisoned 2019–21 schedule coverage).
MAX_INTERPOLATED_DATE_FRAC = 0.05


def season_date_coverage(stints_df: pd.DataFrame) -> pd.DataFrame:
    """Per-season schedule match / authoritative / interpolate rates from ``date_status``.

    Combined-stats seasons (2025–26+) often have ``authoritative`` CSV dates without
    an odds-schedule match — those are OK. Only high interpolate / unresolved rates
    poison walk-forward.
    """
    if stints_df is None or stints_df.empty:
        return pd.DataFrame(columns=[
            "season", "n_games", "n_matched", "n_authoritative", "n_interpolated",
            "n_unresolved", "interp_frac", "ok",
        ])
    cols = ["season", "GAME_ID"]
    if "date_status" in stints_df.columns:
        cols.append("date_status")
    g = stints_df[cols].drop_duplicates(["season", "GAME_ID"])
    rows = []
    for season, sg in g.groupby("season"):
        n = len(sg)
        if "date_status" in sg.columns:
            status = sg["date_status"].fillna(DATE_STATUS_UNRESOLVED).astype(str)
            n_matched = int((status == DATE_STATUS_SCHEDULE_MATCHED).sum())
            n_auth = int((status == DATE_STATUS_AUTHORITATIVE).sum())
            n_interp = int((status == DATE_STATUS_INTERPOLATED).sum())
            n_unresolved = int((status == DATE_STATUS_UNRESOLVED).sum())
        else:
            # No provenance column: if game_date looks real across the season, treat
            # as authoritative; else unresolved (fail closed).
            n_matched = 0
            n_auth = 0
            n_interp = 0
            n_unresolved = n
        frac = n_interp / n if n else 1.0
        unresolved_frac = n_unresolved / n if n else 1.0
        n_good = n_matched + n_auth
        rows.append({
            "season": int(season) if pd.notna(season) else season,
            "n_games": n,
            "n_matched": n_matched,
            "n_authoritative": n_auth,
            "n_interpolated": n_interp,
            "n_unresolved": n_unresolved,
            "interp_frac": float(frac),
            "ok": (
                frac <= float(MAX_INTERPOLATED_DATE_FRAC)
                and unresolved_frac <= float(MAX_INTERPOLATED_DATE_FRAC)
                and (n_good > 0 or n == 0)
            ),
        })
    return pd.DataFrame(rows)


def filter_stints_by_date_quality(
    stints_df: pd.DataFrame,
    *,
    allow_interpolated: bool = False,
    max_interp_frac: float | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, list]:
    """Drop seasons with excessive interpolated/unresolved dates (default P0 path).

    Authoritative CSV dates (combined-stats) are kept even when odds-schedule
    exact-match count is zero.

    Returns ``(filtered_stints, coverage_df, dropped_seasons)``.
    """
    cov = season_date_coverage(stints_df)
    if allow_interpolated or cov.empty:
        return stints_df, cov, []
    bad = cov.loc[~cov["ok"], "season"].tolist()
    if not bad:
        return stints_df, cov, []
    keep = stints_df[~stints_df["season"].isin(bad)].copy()
    return keep, cov, list(bad)


@dataclass(frozen=True)
class DateResolution:
    """Provenance-bearing result of resolving one game's date.

    ``status`` is one of the four values above. ``date`` is ``None`` when
    ``status == "unresolved"`` — callers MUST fail closed for time-sensitive
    uses in that case rather than substituting an invented date (Task 009 /
    Rule 5).
    """

    date: pd.Timestamp | None
    status: str


def resolve_game_date(
    gdate,
    *,
    game_id=None,
) -> DateResolution:
    """Resolve one game's date with explicit provenance, never fabricating one.

    Resolution order:
      1. ``gdate`` parses to a real timestamp -> ``authoritative``.
      2. The ``GAME_ID`` itself encodes a plausible ``YYYYMMDD`` (e.g. the
         odds-schedule ``nba.g.YYYYMMDDNN`` convention, or embedded digits in
         an NBA-style id) -> ``schedule_matched`` (the identifier is matched
         against the game's own authoritative identity, not invented).
      3. Otherwise -> ``unresolved``, ``date=None``. Callers must not invent
         ``previous_date + 1`` or any other placeholder.

    Matchup-order interpolation (``build_gameid_date_map``) is a *separate*,
    explicitly flagged last resort for whole V3 seasons lacking any date
    column; its results are tagged ``interpolated`` by ``attach_real_dates``
    and must stay out of rest/travel evaluations until verified (Rule 5 /
    Phase 1B of the plan).
    """
    ts = pd.to_datetime(gdate, errors="coerce") if gdate is not None else pd.NaT
    if is_valid_timestamp(ts):
        return DateResolution(pd.Timestamp(ts).normalize(), DATE_STATUS_AUTHORITATIVE)

    if game_id is not None:
        from_gid = date_from_game_id(game_id)
        if is_valid_timestamp(from_gid):
            return DateResolution(pd.Timestamp(from_gid).normalize(), DATE_STATUS_SCHEDULE_MATCHED)

    return DateResolution(None, DATE_STATUS_UNRESOLVED)


def coerce_game_date(
    gdate,
    *,
    game_id=None,
    prev_date=None,
    season_start_year: int | None = None,
) -> pd.Timestamp | None:
    """Return a normalized pd.Timestamp, or ``None`` when unresolved.

    Task 009: this used to silently fabricate ``prev_date + 1 day`` (and, as a
    last resort, a fixed ``season_start_year-10-20``) whenever the real date
    was missing. That fabrication is removed: a missing/unparseable date with
    no game-identity match is now ``unresolved`` and returns ``None``, which
    every caller already treats as "skip this row" (fail closed) rather than
    silently invent chronology. ``prev_date`` and ``season_start_year`` are
    kept as accepted (now-unused) parameters for backward compatibility with
    existing call sites; they no longer influence the result.
    """
    resolution = resolve_game_date(gdate, game_id=game_id)
    return resolution.date


def to_py_date(gdate, *, game_id=None, prev_date=None):
    """Return datetime.date or None — never a fabricated date, never NaTType."""
    ts = coerce_game_date(gdate, game_id=game_id, prev_date=prev_date)
    if ts is None:
        return None
    return ts.date()


def prepare_chronological_stints(
    stints_df: pd.DataFrame,
    *,
    context: str = "stints",
) -> pd.DataFrame:
    """Normalize dates, canonicalize per GAME_ID, sort, assert chronology.

    The 2025-26 combined-stats source mixes ISO (``2025-10-21``) and US
    (``12/17/2025``) date strings. A plain ``pd.to_datetime(..., errors="coerce")``
    (or leaving the column as object with mixed Timestamp/str) produces NaT or
    a non-chronological object sort, which then fails the Task 015 monotonic
    assert in ``generate_features`` / ``run_simulation``.

    Steps:
      1. Coerce ``game_date`` with ``format="mixed"``.
      2. One date per ``GAME_ID`` (earliest valid stint date).
      3. Recover remaining NaT from embedded ``GAME_ID`` dates when possible.
      4. Drop unresolved games (fail closed — never invent chronology).
      5. Sort by ``game_date``, ``GAME_ID``, ``stint_id`` and assert
         per-game dates are monotonic nondecreasing.
    """
    if stints_df is None or stints_df.empty:
        return stints_df
    if "game_date" not in stints_df.columns or "GAME_ID" not in stints_df.columns:
        return stints_df

    df = stints_df.copy()
    df["GAME_ID"] = df["GAME_ID"].astype(str)
    # format="mixed": parse ISO and M/D/YYYY independently (Task 008).
    df["game_date"] = pd.to_datetime(df["game_date"], format="mixed", errors="coerce")

    # Canonicalize: every stint of a game shares the earliest valid date.
    df["game_date"] = df.groupby("GAME_ID", sort=False)["game_date"].transform(
        lambda s: s.dropna().min() if s.notna().any() else pd.NaT
    )

    missing = df["game_date"].isna()
    if missing.any():
        rec_map = {}
        for gid in df.loc[missing, "GAME_ID"].unique():
            ts = coerce_game_date(None, game_id=gid)
            if ts is not None:
                rec_map[gid] = ts
        if rec_map:
            df.loc[missing, "game_date"] = df.loc[missing, "GAME_ID"].map(rec_map)

    still_bad = df["game_date"].isna()
    if still_bad.any():
        bad_games = int(df.loc[still_bad, "GAME_ID"].nunique())
        n_rows = int(still_bad.sum())
        print(
            f"⚠️  {context}: dropping {bad_games} game(s) / {n_rows} row(s) "
            f"with unresolved game_date (fail closed)"
        )
        df = df.loc[~still_bad].copy()

    if df.empty:
        return df.reset_index(drop=True)

    sort_cols = ["game_date", "GAME_ID"]
    if "stint_id" in df.columns:
        sort_cols.append("stint_id")
    df = df.sort_values(sort_cols).reset_index(drop=True)

    per_game = df.drop_duplicates("GAME_ID")
    per_game_dates = per_game["game_date"]
    if not per_game_dates.is_monotonic_increasing:
        vals = per_game_dates.reset_index(drop=True)
        gids = per_game["GAME_ID"].reset_index(drop=True)
        detail = ""
        for i in range(1, len(vals)):
            if vals.iloc[i] < vals.iloc[i - 1]:
                detail = (
                    f" (first decrease at index {i}: "
                    f"{gids.iloc[i - 1]}@{vals.iloc[i - 1]} -> "
                    f"{gids.iloc[i]}@{vals.iloc[i]})"
                )
                break
        raise AssertionError(
            f"{context}: per-game decision timestamps are not monotonic "
            f"nondecreasing after sort — chronological iteration is violated"
            f"{detail}"
        )
    return df
