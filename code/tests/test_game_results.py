"""Task 006/007: canonical one-row-per-game outcome table, verified two ways."""
from pathlib import Path

import pandas as pd
import pytest

from pipeline.config import PBP_2026_PATH
from pipeline.game_results import (
    CanonicalScoreMismatch,
    build_canonical_games,
    canonical_finals_map,
    load_canonical_games_from_csv,
    verify_canonical_scores,
)


@pytest.mark.skipif(not PBP_2026_PATH.exists(), reason="2025-26 combined-stats CSV missing")
def test_canonical_games_exactly_1307_unique_2025_26_games():
    canonical = load_canonical_games_from_csv(PBP_2026_PATH)
    assert canonical["GAME_ID"].nunique() == 1307
    assert len(canonical) == 1307


@pytest.mark.skipif(not PBP_2026_PATH.exists(), reason="2025-26 combined-stats CSV missing")
def test_canonical_scores_verified_two_independent_ways_for_every_game():
    canonical = load_canonical_games_from_csv(PBP_2026_PATH)
    mismatched = verify_canonical_scores(canonical)
    assert mismatched == [], f"score mismatch for game IDs: {mismatched}"
    assert canonical["scores_match"].all()


@pytest.mark.skipif(not PBP_2026_PATH.exists(), reason="2025-26 combined-stats CSV missing")
def test_canonical_finals_map_covers_all_games():
    canonical = load_canonical_games_from_csv(PBP_2026_PATH)
    finals = canonical_finals_map(canonical)
    assert len(finals) == 1307
    sample_gid = canonical["GAME_ID"].iloc[0]
    home, away = finals[sample_gid]
    assert home >= 0 and away >= 0


def test_build_canonical_games_detects_mismatch_synthetic():
    raw = pd.DataFrame([
        # Game 1: consistent — max score == summed points.
        {"game_id": 1, "date": "2025-10-21", "team": "GSW", "home_team": "GSW",
         "away_team": "LAL", "points": 3, "home_score": 3, "away_score": 0, "data_set": "Regular"},
        {"game_id": 1, "date": "2025-10-21", "team": "LAL", "home_team": "GSW",
         "away_team": "LAL", "points": 2, "home_score": 3, "away_score": 2, "data_set": "Regular"},
        # Game 2: deliberately inconsistent — home_score running total says 10,
        # but summed team points only add up to 8 (simulated corruption).
        {"game_id": 2, "date": "2025-10-22", "team": "BOS", "home_team": "BOS",
         "away_team": "MIA", "points": 8, "home_score": 10, "away_score": 0, "data_set": "Regular"},
    ])
    canonical = build_canonical_games(raw)
    mismatched = verify_canonical_scores(canonical)
    assert mismatched == [2]
    game1 = canonical[canonical["GAME_ID"] == 1].iloc[0]
    assert game1["scores_match"]
    assert game1["final_home_score"] == 3
    assert game1["final_away_score"] == 2


def test_build_canonical_games_requires_expected_columns():
    with pytest.raises(ValueError):
        build_canonical_games(pd.DataFrame({"a": [1]}))


def test_build_canonical_games_parses_mixed_date_formats():
    raw = pd.DataFrame([
        {"game_id": 1, "date": "2025-10-21", "team": "GSW", "home_team": "GSW",
         "away_team": "LAL", "points": 2, "home_score": 2, "away_score": 0, "data_set": "Regular"},
        {"game_id": 2, "date": "12/17/2025", "team": "BOS", "home_team": "BOS",
         "away_team": "MIA", "points": 2, "home_score": 2, "away_score": 0, "data_set": "Regular"},
    ])
    canonical = build_canonical_games(raw)
    canonical = canonical.set_index("GAME_ID")
    assert not pd.isna(canonical.loc[1, "raw_date"])
    assert not pd.isna(canonical.loc[2, "raw_date"])
    assert canonical.loc[2, "raw_date"] == pd.Timestamp("2025-12-17")
