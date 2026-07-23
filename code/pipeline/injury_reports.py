"""Task 030: normalized, point-in-time player status/injury reports.

Archived injury/status reports are stored as immutable rows with:
  - ``player_id``    : player identifier (string).
  - ``status``       : raw status text (``OUT``, ``DOUBTFUL``, ``QUESTIONABLE``,
                        ``PROBABLE``, ``GTD``, ``ACTIVE``, ...).
  - ``published_at``  : the report's own publication timestamp (source
                        timestamp) -- when the *status itself* became true
                        knowledge, not when we happened to scrape it.
  - ``source``        : provenance string (e.g. ``"nba_injury_report"``,
                        ``"espn"``, ``"beat_writer"``).
  - ``ingested_at``   : when this pipeline recorded the report. Always
                        ``>= published_at``; used only for audit, never for
                        the as-of filter itself (Rule 3 keys off the report's
                        own timestamp, not our ingestion latency).

The single load-bearing invariant (Task 030 pass condition): an as-of query
for cutoff ``T`` must never return a report whose ``published_at > T``, no
matter how many later reports exist in the table.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd

REQUIRED_COLUMNS = ("player_id", "status", "published_at", "source", "ingested_at")

# Reuse the same status vocabulary as `pipeline/availability.py::STATUS_PROB`
# so a normalized report's `status` maps onto the same play-probability
# table used downstream.
_KNOWN_STATUSES = ("OUT", "DOUBTFUL", "QUESTIONABLE", "PROBABLE", "GTD", "ACTIVE", "AVAILABLE")


class InjuryReportSchemaError(ValueError):
    """Raised when raw injury-report rows cannot be normalized safely."""


def normalize_injury_reports(raw_df: pd.DataFrame, column_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """Normalize arbitrary archived injury/status rows into the canonical
    ``(player_id, status, published_at, source, ingested_at)`` schema.

    Parameters
    ----------
    raw_df : DataFrame with at least player/status/publication-timestamp
        columns, optionally under different names supplied via `column_map`
        (raw_name -> canonical_name).
    column_map : optional dict renaming raw columns to the canonical ones
        before validation.

    Rows with an unparseable/missing `published_at` are dropped rather than
    fabricated (Rule 5) -- a status report we cannot date can never be safely
    used in an as-of query and must not silently pass one.
    """
    df = raw_df.copy()
    if column_map:
        df = df.rename(columns=column_map)

    missing = [c for c in ("player_id", "status", "published_at") if c not in df.columns]
    if missing:
        raise InjuryReportSchemaError(
            f"raw injury report frame missing required column(s): {missing}"
        )

    df["player_id"] = df["player_id"].astype(str)
    df["status"] = df["status"].astype(str).str.upper().str.strip()
    df["published_at"] = pd.to_datetime(df["published_at"], errors="coerce", utc=False)
    if "source" not in df.columns:
        df["source"] = "unknown"
    df["source"] = df["source"].fillna("unknown").astype(str)
    if "ingested_at" not in df.columns:
        # No explicit ingestion timestamp recorded: the report's own
        # publication timestamp is a lower bound, but we must not invent a
        # tighter ingestion time than we actually observed.
        df["ingested_at"] = df["published_at"]
    else:
        df["ingested_at"] = pd.to_datetime(df["ingested_at"], errors="coerce", utc=False)
        df["ingested_at"] = df["ingested_at"].fillna(df["published_at"])

    n_before = len(df)
    df = df.dropna(subset=["published_at"])
    dropped = n_before - len(df)
    if dropped:
        import warnings
        warnings.warn(
            f"normalize_injury_reports: dropped {dropped} row(s) with an "
            "unparseable/missing publication timestamp rather than fabricate one",
            stacklevel=2,
        )

    # Rule 3/Rule 5 guard: ingestion can never precede publication -- if raw
    # data claims otherwise, that is a provenance error, not a value to
    # silently coerce.
    bad_order = df[df["ingested_at"] < df["published_at"]]
    if len(bad_order):
        raise InjuryReportSchemaError(
            f"{len(bad_order)} report row(s) have ingested_at < published_at "
            "(a report cannot be ingested before it was published) -- fix "
            "the source data, do not silently reorder timestamps"
        )

    if "game_id" not in df.columns:
        df["game_id"] = pd.NA

    cols = ["player_id", "status", "published_at", "source", "ingested_at", "game_id"]
    out = df[cols].sort_values(["player_id", "published_at"]).reset_index(drop=True)
    return out


def status_as_of(
    reports_df: pd.DataFrame,
    player_id,
    asof_ts,
    *,
    game_id=None,
) -> Optional[dict]:
    """Return the most-recently-published report for `player_id` known as
    of `asof_ts`, or ``None`` if no report qualifies.

    Never returns a report with ``published_at > asof_ts`` (Task 030 pass
    condition) -- the filter is applied unconditionally before any
    ranking/selection logic runs.
    """
    if reports_df is None or len(reports_df) == 0:
        return None
    asof_ts = pd.Timestamp(asof_ts)
    pid = str(player_id)
    candidates = reports_df[
        (reports_df["player_id"] == pid) & (reports_df["published_at"] <= asof_ts)
    ]
    if game_id is not None and "game_id" in reports_df.columns:
        game_specific = candidates[candidates["game_id"] == game_id]
        if len(game_specific):
            candidates = game_specific
    if len(candidates) == 0:
        return None
    # Defensive re-assertion of the as-of invariant even though the filter
    # above already enforces it -- a future refactor that reorders this
    # function must not be able to silently reintroduce a leak.
    assert (candidates["published_at"] <= asof_ts).all(), (
        "status_as_of: a candidate report published after the cutoff leaked "
        "through the as-of filter"
    )
    best = candidates.sort_values("published_at").iloc[-1]
    return {
        "player_id": best["player_id"],
        "status": best["status"],
        "published_at": best["published_at"],
        "source": best["source"],
    }


def bulk_status_as_of(
    reports_df: pd.DataFrame,
    player_ids: Iterable,
    asof_ts,
    *,
    game_id=None,
) -> Dict[str, dict]:
    """Vectorized convenience wrapper over `status_as_of` for a roster."""
    out = {}
    for pid in player_ids:
        rep = status_as_of(reports_df, pid, asof_ts, game_id=game_id)
        if rep is not None:
            out[str(pid)] = rep
    return out


def load_injury_reports_csv(path) -> pd.DataFrame:
    """Load and normalize an archived injury-report CSV.

    Returns an empty, correctly-schema'd DataFrame if `path` does not exist
    -- callers must treat "no archived reports" as missing data (Rule 5),
    never as "everyone is presumed active."
    """
    path = Path(path)
    if not path.exists():
        return pd.DataFrame(columns=list(REQUIRED_COLUMNS) + ["game_id"])
    raw = pd.read_csv(path)
    return normalize_injury_reports(raw)


def status_to_play_prob(status: str) -> float:
    """Delegate to `pipeline.availability.status_to_prob` for the shared
    status -> play-probability vocabulary."""
    from pipeline.availability import status_to_prob

    return status_to_prob(status)
