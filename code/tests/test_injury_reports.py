"""Task 030: normalized archived injury/status reports must expose an
as-of query that can never return a report published after its cutoff."""
import pandas as pd
import pytest

from pipeline.injury_reports import (
    InjuryReportSchemaError,
    bulk_status_as_of,
    load_injury_reports_csv,
    normalize_injury_reports,
    status_as_of,
)


def _raw_reports():
    return pd.DataFrame([
        {"player_id": "101", "status": "questionable", "published_at": "2025-12-01 08:00:00",
         "source": "nba_injury_report"},
        {"player_id": "101", "status": "out", "published_at": "2025-12-01 17:30:00",
         "source": "nba_injury_report"},
        {"player_id": "101", "status": "active", "published_at": "2025-12-02 08:00:00",
         "source": "nba_injury_report"},
        {"player_id": "202", "status": "probable", "published_at": "2025-12-01 12:00:00",
         "source": "beat_writer"},
    ])


def test_normalize_produces_canonical_schema():
    norm = normalize_injury_reports(_raw_reports())
    assert list(norm.columns) == ["player_id", "status", "published_at", "source", "ingested_at", "game_id"]
    assert set(norm["status"]) == {"QUESTIONABLE", "OUT", "ACTIVE", "PROBABLE"}
    assert pd.api.types.is_datetime64_any_dtype(norm["published_at"])
    # ingested_at defaults to published_at when not supplied, never earlier.
    assert (norm["ingested_at"] >= norm["published_at"]).all()


def test_normalize_rejects_ingested_before_published():
    raw = pd.DataFrame([{
        "player_id": "1", "status": "OUT",
        "published_at": "2025-12-01 12:00:00",
        "ingested_at": "2025-12-01 08:00:00",  # before publication: invalid
        "source": "x",
    }])
    with pytest.raises(InjuryReportSchemaError):
        normalize_injury_reports(raw)


def test_normalize_drops_unparseable_dates_instead_of_fabricating():
    raw = pd.DataFrame([
        {"player_id": "1", "status": "OUT", "published_at": "not-a-date", "source": "x"},
        {"player_id": "2", "status": "OUT", "published_at": "2025-12-01", "source": "x"},
    ])
    with pytest.warns(UserWarning):
        norm = normalize_injury_reports(raw)
    assert len(norm) == 1
    assert norm.iloc[0]["player_id"] == "2"


def test_asof_query_never_returns_report_published_after_cutoff():
    norm = normalize_injury_reports(_raw_reports())
    # Cutoff strictly between the 08:00 QUESTIONABLE and 17:30 OUT report.
    cutoff = pd.Timestamp("2025-12-01 12:00:00")
    rep = status_as_of(norm, "101", cutoff)
    assert rep is not None
    assert rep["status"] == "QUESTIONABLE"
    assert rep["published_at"] <= cutoff


def test_asof_query_boundary_exact_timestamp_is_inclusive():
    norm = normalize_injury_reports(_raw_reports())
    cutoff = pd.Timestamp("2025-12-01 17:30:00")
    rep = status_as_of(norm, "101", cutoff)
    assert rep["status"] == "OUT"  # exactly-at-cutoff report is visible
    cutoff_before = cutoff - pd.Timedelta(seconds=1)
    rep2 = status_as_of(norm, "101", cutoff_before)
    assert rep2["status"] == "QUESTIONABLE"


def test_asof_query_never_sees_future_report_even_far_before_it():
    norm = normalize_injury_reports(_raw_reports())
    # Cutoff long before every report for this player -> no report at all.
    cutoff = pd.Timestamp("2025-11-01")
    rep = status_as_of(norm, "101", cutoff)
    assert rep is None
    # Sanity: same player DOES have reports, just none valid as-of cutoff.
    assert (norm["player_id"] == "101").any()


def test_asof_query_exhaustive_never_leaks_future_row(monkeypatch=None):
    """Property-style check: for every report's own timestamp used as a
    cutoff, the returned report (if any) must have published_at <= cutoff,
    across every row in the table."""
    norm = normalize_injury_reports(_raw_reports())
    for _, row in norm.iterrows():
        cutoff = row["published_at"]
        rep = status_as_of(norm, row["player_id"], cutoff)
        assert rep is not None
        assert rep["published_at"] <= cutoff


def test_bulk_status_as_of():
    norm = normalize_injury_reports(_raw_reports())
    cutoff = pd.Timestamp("2025-12-01 12:00:00")
    out = bulk_status_as_of(norm, ["101", "202", "999"], cutoff)
    assert out["101"]["status"] == "QUESTIONABLE"
    assert out["202"]["status"] == "PROBABLE"
    assert "999" not in out  # unknown player: missing, not fabricated


def test_load_injury_reports_csv_missing_file_returns_empty_schema(tmp_path):
    missing_path = tmp_path / "does_not_exist.csv"
    df = load_injury_reports_csv(missing_path)
    assert len(df) == 0
    assert "player_id" in df.columns
    assert "published_at" in df.columns
