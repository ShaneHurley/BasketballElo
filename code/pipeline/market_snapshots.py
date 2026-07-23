"""Tasks 018-023, 028, 029: a leak-proof quote-level market snapshot layer.

Replaces the game-level "one spread/one ML" dictionary in
:mod:`pipeline.market` with immutable, quote-level rows keyed by
``(GAME_ID, book, market, side, point, quote_timestamp)``. Multiple quotes for
one game remain separate rows (Task 018); every usable quote can compute
``minutes_before_tip`` against the *authoritative* tip time (Task 019, never
inferred from the quotes themselves); in-play/post-tip/ambiguous quotes are
rejected before any selection happens (Task 020); ``open_*``/``decision_*``
(T-60)/``close_*`` are three separate, deterministic selections over the
surviving rows (Tasks 021-023), with ``close_*`` fenced off from T-60 feature
building by ``assert_no_close_leak``. Task 028 keeps point CLV and
price/probability CLV in separate columns. Task 029 audits open/T-60/close
coverage by season/book and flags sources (``all_odds.csv``) whose timing
cannot be proven to be a T-60 snapshot.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

__all__ = [
    "QUOTE_COLUMNS",
    "build_quotes_frame",
    "derive_tip_utc_from_pbp",
    "attach_tip_utc",
    "reject_invalid_quotes",
    "select_open_quotes",
    "select_decision_quotes",
    "select_close_quotes",
    "assert_no_close_leak",
    "build_t60_feature_frame",
    "build_evaluation_frame",
    "point_clv",
    "price_clv",
    "match_pinnacle_quotes_to_games",
    "detect_source_horizon",
    "audit_market_coverage",
    "require_valid_t60_season",
]

# ---------------------------------------------------------------------------
# Task 018: quote-level schema
# ---------------------------------------------------------------------------

GROUP_KEY_COLS = ("GAME_ID", "book", "market", "side", "point")

QUOTE_COLUMNS = (
    "GAME_ID",          # canonical game identity (pipeline.game_results)
    "book",             # e.g. "pinnacle"
    "market",           # "moneyline" | "spread" | "total"
    "side",             # "home" | "away" | "over" | "under"
    "point",            # spread/total line; NaN for moneyline
    "price_american",
    "price_decimal",
    "quote_timestamp",  # when this price was live in the market (UTC)
    "source_timestamp", # book/feed-reported timestamp (UTC)
    "ingestion_timestamp",  # when *we* recorded it (UTC); >= source_timestamp
    "in_play",          # True if quote is known to be posted after tip
)

_REQUIRED = set(QUOTE_COLUMNS)


def build_quotes_frame(rows) -> pd.DataFrame:
    """Validate/construct the quote-level schema from an iterable of dict rows
    or an existing DataFrame. Does not deduplicate — multiple quotes for the
    same game/book/market/side/point at different timestamps are legitimate
    separate rows (Task 018 pass condition)."""
    df = pd.DataFrame(rows) if not isinstance(rows, pd.DataFrame) else rows.copy()
    missing = _REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"quotes frame missing required columns: {sorted(missing)}")
    df["quote_timestamp"] = pd.to_datetime(df["quote_timestamp"], utc=True, errors="coerce")
    df["source_timestamp"] = pd.to_datetime(df["source_timestamp"], utc=True, errors="coerce")
    df["ingestion_timestamp"] = pd.to_datetime(df["ingestion_timestamp"], utc=True, errors="coerce")
    df["in_play"] = df["in_play"].fillna(False).astype(bool)
    df["price_american"] = pd.to_numeric(df["price_american"], errors="coerce")
    df["price_decimal"] = pd.to_numeric(df["price_decimal"], errors="coerce")
    df["point"] = pd.to_numeric(df["point"], errors="coerce")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Task 019: authoritative tip UTC (never inferred from quotes)
# ---------------------------------------------------------------------------

def derive_tip_utc_from_pbp(raw_pbp_df: pd.DataFrame, game_id_col: str = "game_id",
                             time_col: str = "time_actual") -> pd.DataFrame:
    """Authoritative tip time per GAME_ID: the timestamp of the first raw PBP
    event actually recorded for that game (real wall-clock UTC "time_actual"
    column in the combined-stats source — 0% missing, independent of any
    odds/quote feed). This is *not* inferred from a market quote; it comes
    straight from the game itself, satisfying "never infer a tip time from
    the final quote" (Task 019).

    Returns a DataFrame with columns [GAME_ID, tip_utc].
    """
    if game_id_col not in raw_pbp_df.columns or time_col not in raw_pbp_df.columns:
        raise ValueError(f"raw_pbp_df must contain {game_id_col!r} and {time_col!r}")
    ts = pd.to_datetime(raw_pbp_df[time_col], utc=True, errors="coerce")
    out = (
        pd.DataFrame({"GAME_ID": raw_pbp_df[game_id_col], "_ts": ts})
        .dropna(subset=["_ts"])
        .groupby("GAME_ID", sort=False)["_ts"]
        .min()
        .reset_index()
        .rename(columns={"_ts": "tip_utc"})
    )
    return out


def attach_tip_utc(quotes_df: pd.DataFrame, tip_utc_map) -> pd.DataFrame:
    """Left-join authoritative ``tip_utc`` onto quotes by GAME_ID and compute
    ``minutes_before_tip`` for every row whose tip is known. Unmatched games
    get NaT/NaN (missing, never fabricated) rather than a guessed tip time."""
    if isinstance(tip_utc_map, dict):
        tip_df = pd.DataFrame(
            {"GAME_ID": list(tip_utc_map.keys()), "tip_utc": list(tip_utc_map.values())}
        )
    else:
        tip_df = tip_utc_map[["GAME_ID", "tip_utc"]].drop_duplicates("GAME_ID")
    tip_df = tip_df.copy()
    tip_df["tip_utc"] = pd.to_datetime(tip_df["tip_utc"], utc=True, errors="coerce")

    out = quotes_df.merge(tip_df, on="GAME_ID", how="left")
    delta = out["tip_utc"] - out["quote_timestamp"]
    out["minutes_before_tip"] = delta.dt.total_seconds() / 60.0
    return out


# ---------------------------------------------------------------------------
# Task 020: reject in-play / post-tip / ambiguous quotes
# ---------------------------------------------------------------------------

def reject_invalid_quotes(quotes_df: pd.DataFrame) -> pd.DataFrame:
    """Drop quotes that cannot legally enter any pregame snapshot:

    - ``in_play`` True (posted during/after the game);
    - unknown tip time (``tip_utc`` missing — cannot prove it was pregame);
    - ``minutes_before_tip <= 0`` (at-or-after tip);
    - flagged ``match_ambiguous`` (tied to more than one candidate game).
    """
    if "minutes_before_tip" not in quotes_df.columns:
        raise ValueError("call attach_tip_utc() before reject_invalid_quotes()")
    df = quotes_df.copy()
    keep = (~df["in_play"].astype(bool)) & df["minutes_before_tip"].notna() & (df["minutes_before_tip"] > 0)
    if "match_ambiguous" in df.columns:
        keep &= ~df["match_ambiguous"].fillna(False).astype(bool)
    return df.loc[keep].reset_index(drop=True)


# ---------------------------------------------------------------------------
# Tasks 021-023: open / decision (T-60) / close selection
# ---------------------------------------------------------------------------

def _value_cols(df: pd.DataFrame) -> list:
    return [c for c in df.columns if c not in GROUP_KEY_COLS]


def select_open_quotes(df: pd.DataFrame, group_cols=GROUP_KEY_COLS) -> pd.DataFrame:
    """First valid pregame quote per group. Sorting is entirely by content
    (timestamp + full key + price), never by input row order, so the result
    is deterministic under a shuffled input (Task 021 pass condition)."""
    group_cols = list(group_cols)
    sort_cols = group_cols + ["quote_timestamp", "price_decimal", "price_american"]
    ordered = df.sort_values(sort_cols, kind="mergesort", na_position="last")
    out = ordered.groupby(group_cols, as_index=False, sort=False).first()
    value_cols = [c for c in out.columns if c not in group_cols]
    return out.rename(columns={c: f"open_{c}" for c in value_cols})


def select_decision_quotes(
    df: pd.DataFrame, group_cols=GROUP_KEY_COLS, cutoff_minutes: float = 60.0,
) -> pd.DataFrame:
    """Latest valid quote with ``quote_timestamp <= tip_utc - cutoff_minutes``
    per group, i.e. ``minutes_before_tip >= cutoff_minutes`` (Task 022).
    Stores quote age (minutes past the T-60 boundary) and a missingness flag
    for groups with no such quote, rather than silently omitting them."""
    group_cols = list(group_cols)
    all_keys = df[group_cols].drop_duplicates().reset_index(drop=True)

    eligible = df[df["minutes_before_tip"] >= cutoff_minutes]
    sort_cols = group_cols + ["quote_timestamp", "price_decimal", "price_american"]
    eligible = eligible.sort_values(sort_cols, kind="mergesort", na_position="last")
    picked = eligible.groupby(group_cols, as_index=False, sort=False).last()

    value_cols = [c for c in picked.columns if c not in group_cols]
    picked = picked.rename(columns={c: f"decision_{c}" for c in value_cols})
    if "decision_minutes_before_tip" in picked.columns:
        picked["decision_quote_age_minutes"] = picked["decision_minutes_before_tip"] - cutoff_minutes

    out = all_keys.merge(picked, on=group_cols, how="left")
    price_col = "decision_price_decimal" if "decision_price_decimal" in out.columns else None
    out["decision_missing"] = out[price_col].isna() if price_col else True
    return out


def select_close_quotes(df: pd.DataFrame, group_cols=GROUP_KEY_COLS) -> pd.DataFrame:
    """Last valid pre-tip quote per group — **evaluation-only** (Task 023).
    Every returned value column is prefixed ``close_`` so downstream feature
    builders can be mechanically checked for the forbidden prefix."""
    group_cols = list(group_cols)
    sort_cols = group_cols + ["quote_timestamp", "price_decimal", "price_american"]
    ordered = df.sort_values(sort_cols, kind="mergesort", na_position="last")
    out = ordered.groupby(group_cols, as_index=False, sort=False).last()
    value_cols = [c for c in out.columns if c not in group_cols]
    return out.rename(columns={c: f"close_{c}" for c in value_cols})


def assert_no_close_leak(columns) -> None:
    """Raise if any column name begins with ``close_`` (Task 023 guard: "any
    code building T-60 features must reject keys starting with close_")."""
    leaked = [c for c in columns if str(c).startswith("close_")]
    if leaked:
        raise ValueError(f"T-60 feature builder touched evaluation-only close_* columns: {leaked}")


def build_t60_feature_frame(
    quotes_df: pd.DataFrame, group_cols=GROUP_KEY_COLS, cutoff_minutes: float = 60.0,
) -> pd.DataFrame:
    """Open + decision(T-60) columns only — never close_*. Self-checks its own
    output with ``assert_no_close_leak`` before returning."""
    group_cols = list(group_cols)
    opened = select_open_quotes(quotes_df, group_cols)
    decided = select_decision_quotes(quotes_df, group_cols, cutoff_minutes)
    out = opened.merge(decided, on=group_cols, how="outer")
    assert_no_close_leak(out.columns)
    return out


def build_evaluation_frame(
    quotes_df: pd.DataFrame, group_cols=GROUP_KEY_COLS, cutoff_minutes: float = 60.0,
) -> pd.DataFrame:
    """Open + decision(T-60) + close columns, for evaluation/CLV reporting
    only. Never feed this frame's close_* columns into a T-60 feature/model
    input — use build_t60_feature_frame for that."""
    group_cols = list(group_cols)
    t60 = build_t60_feature_frame(quotes_df, group_cols, cutoff_minutes)
    closed = select_close_quotes(quotes_df, group_cols)
    return t60.merge(closed, on=group_cols, how="outer")


# ---------------------------------------------------------------------------
# Task 028: point CLV vs. price/probability CLV — separate columns
# ---------------------------------------------------------------------------

def point_clv(decision_home_point: float, close_home_point: float, side: str) -> float:
    """Bet-side point CLV in raw spread/total points (Phase 3 formulas):

    - Home/Over bet: ``decision_home_point - close_home_point``
    - Away/Under bet: ``close_home_point - decision_home_point``

    This is a *point* difference, not a return — never multiply/compound it
    with ``price_clv`` (Task 028 rule).
    """
    if pd.isna(decision_home_point) or pd.isna(close_home_point):
        return float("nan")
    side = str(side).lower()
    if side in ("home", "over"):
        return float(decision_home_point) - float(close_home_point)
    if side in ("away", "under"):
        return float(close_home_point) - float(decision_home_point)
    raise ValueError(f"unknown side for point_clv: {side!r}")


def price_clv(decision_fair_prob: float, close_fair_prob: float) -> float:
    """Fair-probability CLV for the bet side actually taken: the change in the
    market's own fair (de-vigged) probability for *your side* between decision
    and close. Positive = the market moved toward your side after you bet
    (favorable line/price movement); this is a probability-domain quantity,
    kept in its own column and never combined via a multiplicative return
    identity with ``point_clv``.
    """
    if pd.isna(decision_fair_prob) or pd.isna(close_fair_prob):
        return float("nan")
    return float(close_fair_prob) - float(decision_fair_prob)


# ---------------------------------------------------------------------------
# Task 025 support: matching a two-sided quote feed to canonical games
# (unique home+away identity; ambiguous/unmatched quotes are rejected, never
# backfilled from a different game for the same team).
# ---------------------------------------------------------------------------

def match_pinnacle_quotes_to_games(
    pinnacle_df: pd.DataFrame,
    canonical_games_df: pd.DataFrame,
    *,
    to_tricode,
    match_tolerance_days: int = 1,
    book: str = "pinnacle",
) -> pd.DataFrame:
    """Expand raw ``nba_main_lines.csv``-style rows (team1/team2, decimal
    odds, timestamp) into long-format quote rows tied to exactly one
    canonical ``GAME_ID`` (both teams must match; ambiguous matchup+date
    matches are dropped rather than guessed).

    ``canonical_games_df`` must have GAME_ID, home_team, away_team, raw_date
    (tricode home/away, per pipeline.game_results). ``to_tricode`` is injected
    (pipeline.dates.to_tricode) to avoid a hard import cycle.
    """
    games = canonical_games_df.copy()
    games["home_team"] = games["home_team"].apply(to_tricode)
    games["away_team"] = games["away_team"].apply(to_tricode)
    games["raw_date"] = pd.to_datetime(games["raw_date"], errors="coerce")

    by_matchup: dict = {}
    for _, g in games.dropna(subset=["raw_date"]).iterrows():
        key = frozenset((g["home_team"], g["away_team"]))
        by_matchup.setdefault(key, []).append((g["raw_date"], g["GAME_ID"], g["home_team"], g["away_team"]))

    df = pinnacle_df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["_ts"])
    df["_t1"] = df["team1"].apply(to_tricode)
    df["_t2"] = df["team2"].apply(to_tricode)

    rows = []
    for _, r in df.iterrows():
        t1, t2 = r["_t1"], r["_t2"]
        if not isinstance(t1, str) or not isinstance(t2, str):
            continue
        candidates = by_matchup.get(frozenset((t1, t2)), [])
        ts_date = r["_ts"].normalize()
        close_matches = [
            c for c in candidates if abs((c[0] - ts_date).days) <= match_tolerance_days
        ]
        if len(close_matches) != 1:
            continue  # zero or ambiguous -> reject (Task 020/025), never guess
        _, game_id, home_tri, _away_tri = close_matches[0]
        home_is_t1 = home_tri == t1

        def side_val(home_col, away_col):
            return (r.get(home_col), r.get(away_col)) if home_is_t1 else (r.get(away_col), r.get(home_col))

        ml_home_dec, ml_away_dec = side_val("team1_moneyline", "team2_moneyline")
        sp_home_pt, sp_away_pt = side_val("team1_spread", "team2_spread")
        sp_home_odds, sp_away_odds = side_val("team1_spread_odds", "team2_spread_odds")

        base = dict(
            GAME_ID=game_id, book=book,
            quote_timestamp=r["_ts"], source_timestamp=r["_ts"], ingestion_timestamp=r["_ts"],
            in_play=False,
        )

        def _dec_to_am(dec):
            if pd.isna(dec):
                return np.nan
            try:
                dec = float(dec)
            except (TypeError, ValueError):
                return np.nan
            if not np.isfinite(dec) or dec <= 1.0:
                return np.nan
            return (dec - 1.0) * 100.0 if dec >= 2.0 else -100.0 / (dec - 1.0)

        rows.append({**base, "market": "moneyline", "side": "home", "point": np.nan,
                     "price_decimal": ml_home_dec, "price_american": _dec_to_am(ml_home_dec)})
        rows.append({**base, "market": "moneyline", "side": "away", "point": np.nan,
                     "price_decimal": ml_away_dec, "price_american": _dec_to_am(ml_away_dec)})
        rows.append({**base, "market": "spread", "side": "home", "point": sp_home_pt,
                     "price_decimal": sp_home_odds, "price_american": _dec_to_am(sp_home_odds)})
        rows.append({**base, "market": "spread", "side": "away", "point": sp_away_pt,
                     "price_decimal": sp_away_odds, "price_american": _dec_to_am(sp_away_odds)})
        rows.append({**base, "market": "total", "side": "over", "point": r.get("over_total"),
                     "price_decimal": r.get("over_total_odds"), "price_american": _dec_to_am(r.get("over_total_odds"))})
        rows.append({**base, "market": "total", "side": "under", "point": r.get("under_total"),
                     "price_decimal": r.get("under_total_odds"), "price_american": _dec_to_am(r.get("under_total_odds"))})

    if not rows:
        return build_quotes_frame(pd.DataFrame(columns=list(QUOTE_COLUMNS)))
    return build_quotes_frame(pd.DataFrame(rows))


# ---------------------------------------------------------------------------
# Task 029: coverage audit + source-horizon labeling
# ---------------------------------------------------------------------------

def detect_source_horizon(timestamp_series: pd.Series, *, time_of_day_tolerance_seconds: float = 1.0) -> str:
    """Return ``"unknown_horizon"`` when a quote source's timestamps carry no
    real intra-day precision (e.g. ``all_odds.csv``'s ``game_date`` column is
    always stamped a fixed ``HH:MM`` regardless of actual quote time), else
    ``"timestamped"``.

    A source whose time-of-day component has essentially zero variance cannot
    be proven to represent any particular horizon (open/T-60/close) and must
    not be treated as a T-60 snapshot (Phase 2B / Task 029).
    """
    ts = pd.to_datetime(timestamp_series, errors="coerce").dropna()
    if ts.empty:
        return "unknown_horizon"
    seconds_of_day = (ts.dt.hour * 3600 + ts.dt.minute * 60 + ts.dt.second).astype(float)
    if seconds_of_day.nunique() <= 1 or seconds_of_day.std() <= time_of_day_tolerance_seconds:
        return "unknown_horizon"
    return "timestamped"


def audit_market_coverage(
    canonical_games_df: pd.DataFrame,
    quotes_df: pd.DataFrame,
    *,
    cutoff_minutes: float = 60.0,
    season_col: str = "season_type",
) -> pd.DataFrame:
    """Per (season, book) report of what fraction of canonical games have a
    usable open / T-60(decision) / close quote and an exact price for each,
    across every quote market present in ``quotes_df``.
    """
    games = canonical_games_df[["GAME_ID", season_col]].drop_duplicates("GAME_ID")
    n_games = games.groupby(season_col)["GAME_ID"].nunique()

    if quotes_df.empty:
        return pd.DataFrame(columns=[
            "season", "book", "n_games", "n_with_open", "n_with_decision_t60",
            "n_with_close", "open_coverage", "decision_t60_coverage", "close_coverage",
        ])

    rows = []
    for book, book_df in quotes_df.groupby("book"):
        opened = select_open_quotes(book_df)
        decided = select_decision_quotes(book_df, cutoff_minutes=cutoff_minutes)
        closed = select_close_quotes(book_df)

        opened_g = opened.merge(games, on="GAME_ID", how="inner")
        decided_g = decided.merge(games, on="GAME_ID", how="inner")
        closed_g = closed.merge(games, on="GAME_ID", how="inner")

        for season, n_total in n_games.items():
            n_open = opened_g.loc[opened_g[season_col] == season, "GAME_ID"].nunique()
            n_decision = decided_g.loc[
                (decided_g[season_col] == season) & (~decided_g["decision_missing"]), "GAME_ID"
            ].nunique()
            n_close = closed_g.loc[closed_g[season_col] == season, "GAME_ID"].nunique()
            rows.append({
                "season": season, "book": book, "n_games": int(n_total),
                "n_with_open": int(n_open), "n_with_decision_t60": int(n_decision),
                "n_with_close": int(n_close),
                "open_coverage": n_open / n_total if n_total else 0.0,
                "decision_t60_coverage": n_decision / n_total if n_total else 0.0,
                "close_coverage": n_close / n_total if n_total else 0.0,
            })
    return pd.DataFrame(rows)


def require_valid_t60_season(coverage_df: pd.DataFrame, season, *, min_coverage: float = 0.5) -> None:
    """Raise if ``season`` lacks valid T-60 (decision) coverage across every
    book in the report. Callers must not run market-residual evaluation for a
    season that fails this check (Task 029 / plan rule)."""
    rows = coverage_df[coverage_df["season"] == season]
    if rows.empty or rows["decision_t60_coverage"].max() < min_coverage:
        raise ValueError(
            f"season {season!r} has no book with >= {min_coverage:.0%} T-60 decision-quote "
            "coverage; market-residual/CLV evaluation is blocked for this season."
        )
