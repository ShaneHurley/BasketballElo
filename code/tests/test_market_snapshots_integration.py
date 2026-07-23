"""Task 029 / Phase 2B end-to-end integration: reconstruct real open/T-60/
close snapshots for the 2025-26 season from ``nba_main_lines.csv`` (Pinnacle)
using authoritative tip times derived from the raw PBP source (never from
the quotes themselves), then audit coverage. Skipped when the large source
files are not present (e.g. a CI checkout without the data drop)."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.config import PBP_2026_PATH, PINNACLE_LINES_PATH
from pipeline.dates import to_tricode
from pipeline.game_results import load_canonical_games_from_csv
from pipeline.market_snapshots import (
    attach_tip_utc,
    audit_market_coverage,
    derive_tip_utc_from_pbp,
    match_pinnacle_quotes_to_games,
    reject_invalid_quotes,
    require_valid_t60_season,
    select_decision_quotes,
)

pytestmark = pytest.mark.skipif(
    not (PBP_2026_PATH.exists() and PINNACLE_LINES_PATH.exists()),
    reason="2025-26 PBP and/or nba_main_lines.csv not present",
)


@pytest.fixture(scope="module")
def real_2025_26_snapshot_pipeline():
    raw = pd.read_csv(
        PBP_2026_PATH,
        usecols=["game_id", "date", "team", "home_team", "away_team", "points",
                 "home_score", "away_score", "data_set", "time_actual"],
        low_memory=False,
    )
    canonical = load_canonical_games_from_csv(PBP_2026_PATH)
    tip = derive_tip_utc_from_pbp(raw)
    canonical = canonical.merge(tip, on="GAME_ID", how="left", suffixes=("_orig", ""))

    pinnacle = pd.read_csv(PINNACLE_LINES_PATH)
    quotes = match_pinnacle_quotes_to_games(pinnacle, canonical, to_tricode=to_tricode, match_tolerance_days=1)
    quotes = attach_tip_utc(quotes, canonical[["GAME_ID", "tip_utc"]])
    valid = reject_invalid_quotes(quotes)
    return canonical, valid


def test_authoritative_tip_utc_is_fully_populated_from_real_pbp(real_2025_26_snapshot_pipeline):
    canonical, _ = real_2025_26_snapshot_pipeline
    # tip_utc comes from the game's own first real PBP event, independent of
    # any odds/quote feed, and the source column has 0% missingness.
    assert canonical["tip_utc"].isna().mean() == 0.0


def test_no_post_tip_quotes_survive_rejection(real_2025_26_snapshot_pipeline):
    _, valid = real_2025_26_snapshot_pipeline
    assert (valid["minutes_before_tip"] > 0).all()
    assert not valid["in_play"].any()


def test_real_pinnacle_quotes_reconstruct_meaningful_t60_coverage(real_2025_26_snapshot_pipeline):
    """This is the concrete Phase 2B deliverable: real T-60 snapshots for
    2025-26, not a placeholder. Coverage should be well above trivial (not
    0%) but need not be 100% (~89% of the 1,164-game Regular Season slate
    has a matched Pinnacle quote at all)."""
    canonical, valid = real_2025_26_snapshot_pipeline
    dec = select_decision_quotes(valid)
    dec = dec.merge(canonical[["GAME_ID", "season_type"]].drop_duplicates("GAME_ID"), on="GAME_ID", how="left")
    reg = dec[dec["season_type"] == "NBA 2025-2026 Regular Season"]
    coverage = (~reg["decision_missing"]).mean()
    assert coverage > 0.5, f"expected meaningful T-60 coverage, got {coverage:.1%}"


def test_coverage_report_and_season_gate_are_consistent(real_2025_26_snapshot_pipeline):
    canonical, valid = real_2025_26_snapshot_pipeline
    report = audit_market_coverage(canonical, valid)
    assert (report["decision_t60_coverage"] <= 1.0).all()
    assert (report["decision_t60_coverage"] >= 0.0).all()
    for season in report["season"]:
        row = report[report["season"] == season].iloc[0]
        if row["decision_t60_coverage"] >= 0.5:
            require_valid_t60_season(report, season, min_coverage=0.5)  # must not raise
        else:
            with pytest.raises(ValueError):
                require_valid_t60_season(report, season, min_coverage=0.5)
