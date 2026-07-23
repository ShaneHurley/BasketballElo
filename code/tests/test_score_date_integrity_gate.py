"""Task 016: comprehensive score/date integrity gate.

Covers all 1,307 2025-26 games: canonical final scores must match raw
finals (two independent methods), all seven known mixed-date games must
match their raw dates exactly, every game has exactly one home/away pair,
no duplicate game rows, and scores/periods are within plausible NBA ranges.

Per the plan's execution-order rule: "Do not proceed past step 2 if
score/date invariants fail." If this module's tests fail, later phases
(market snapshots, availability, model work) must not proceed.
"""
import pandas as pd
import pytest

from pipeline.config import PBP_2026_PATH
from pipeline.game_results import build_canonical_games, load_canonical_games_from_csv

KNOWN_CORRUPTED_DATE_GAME_IDS = {
    22500368: "2025-12-17",
    22500375: "2025-12-18",
    22500458: "2025-12-30",
    22500508: "2026-01-05",
    22500521: "2026-01-07",
    22500550: "2026-01-11",
    22500628: "2026-01-22",
}

pytestmark = pytest.mark.skipif(
    not PBP_2026_PATH.exists(), reason="2025-26 combined-stats CSV missing"
)


@pytest.fixture(scope="module")
def canonical():
    return load_canonical_games_from_csv(PBP_2026_PATH)


@pytest.fixture(scope="module")
def raw():
    try:
        return pd.read_csv(PBP_2026_PATH, engine="pyarrow")
    except Exception:
        return pd.read_csv(PBP_2026_PATH, low_memory=False)


def test_exactly_1307_unique_games(canonical):
    assert canonical["GAME_ID"].nunique() == 1307
    assert len(canonical) == 1307


def test_no_duplicate_game_rows(canonical):
    assert canonical["GAME_ID"].duplicated().sum() == 0


def test_all_games_final_scores_match_raw(canonical):
    mismatched = canonical.loc[~canonical["scores_match"], "GAME_ID"].tolist()
    assert mismatched == [], f"score invariant failed for game IDs: {mismatched}"


def test_all_seven_known_dates_match_exactly(canonical):
    canonical_by_id = canonical.set_index("GAME_ID")
    for gid, expected_date in KNOWN_CORRUPTED_DATE_GAME_IDS.items():
        assert gid in canonical_by_id.index, f"known game {gid} missing from canonical table"
        actual = canonical_by_id.loc[gid, "raw_date"]
        assert not pd.isna(actual), f"game {gid} raw_date is missing/NaT"
        assert pd.Timestamp(actual) == pd.Timestamp(expected_date), (
            f"game {gid}: expected {expected_date}, got {actual}"
        )


def test_no_row_has_unresolvable_date(raw):
    parsed = pd.to_datetime(raw["date"], format="mixed", errors="coerce")
    assert parsed.isna().sum() == 0, "every raw PBP row must have a parseable date"


def test_exactly_one_home_away_pair_per_game(raw):
    pairs_per_game = raw.groupby("game_id")[["home_team", "away_team"]].nunique()
    assert (pairs_per_game["home_team"] == 1).all(), "a game must have exactly one home team"
    assert (pairs_per_game["away_team"] == 1).all(), "a game must have exactly one away team"
    mismatched_teams = raw.groupby("game_id").apply(
        lambda g: g["home_team"].iloc[0] == g["away_team"].iloc[0]
    )
    assert not mismatched_teams.any(), "home and away team must never be identical"


def test_plausible_final_scores(canonical):
    assert (canonical["final_home_score"] >= 50).all(), "implausibly low final home score"
    assert (canonical["final_home_score"] <= 200).all(), "implausibly high final home score"
    assert (canonical["final_away_score"] >= 50).all(), "implausibly low final away score"
    assert (canonical["final_away_score"] <= 200).all(), "implausibly high final away score"


def test_plausible_periods(raw):
    assert raw["period"].min() >= 1
    assert raw["period"].max() <= 8, "more than 4 OT periods is implausible for NBA games"


def test_score_date_gate_stops_on_synthetic_corruption():
    """Sanity check that the gate actually fails closed on bad data (proves
    this isn't a vacuously-passing test suite)."""
    bad_raw = pd.DataFrame([
        {"game_id": 1, "date": "2025-10-21", "team": "GSW", "home_team": "GSW",
         "away_team": "LAL", "points": 8, "home_score": 10, "away_score": 0, "data_set": "Regular"},
    ])
    canonical = build_canonical_games(bad_raw)
    assert not canonical["scores_match"].all()
