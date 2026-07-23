"""Tasks 018-023, 028, 029: quote-level market snapshot schema, tip-UTC join,
in-play/ambiguous rejection, open/decision(T-60)/close selection, close-leak
guard, point-vs-price CLV separation, and coverage auditing."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.market_snapshots import (
    QUOTE_COLUMNS,
    assert_no_close_leak,
    attach_tip_utc,
    audit_market_coverage,
    build_evaluation_frame,
    build_quotes_frame,
    build_t60_feature_frame,
    derive_tip_utc_from_pbp,
    detect_source_horizon,
    match_pinnacle_quotes_to_games,
    point_clv,
    price_clv,
    reject_invalid_quotes,
    require_valid_t60_season,
    select_close_quotes,
    select_decision_quotes,
    select_open_quotes,
)


def _row(game_id="G1", book="pinnacle", market="spread", side="home", point=-3.0,
         price_american=-110.0, price_decimal=1.909, quote_timestamp="2026-01-01T22:00:00Z",
         source_timestamp=None, ingestion_timestamp=None, in_play=False):
    ts = quote_timestamp
    return dict(
        GAME_ID=game_id, book=book, market=market, side=side, point=point,
        price_american=price_american, price_decimal=price_decimal,
        quote_timestamp=ts, source_timestamp=source_timestamp or ts,
        ingestion_timestamp=ingestion_timestamp or ts, in_play=in_play,
    )


# ---------------------------------------------------------------------------
# Task 018
# ---------------------------------------------------------------------------

def test_schema_has_required_columns():
    assert set(QUOTE_COLUMNS) >= {
        "GAME_ID", "book", "market", "side", "point", "price_american",
        "price_decimal", "quote_timestamp", "source_timestamp",
        "ingestion_timestamp", "in_play",
    }


def test_multiple_quotes_per_game_remain_separate_rows():
    rows = [
        _row(quote_timestamp="2026-01-01T20:00:00Z", price_american=-105.0),
        _row(quote_timestamp="2026-01-01T21:00:00Z", price_american=-108.0),
        _row(quote_timestamp="2026-01-01T22:00:00Z", price_american=-110.0),
    ]
    df = build_quotes_frame(rows)
    assert len(df) == 3
    assert df["quote_timestamp"].nunique() == 3


def test_build_quotes_frame_requires_all_columns():
    with pytest.raises(ValueError):
        build_quotes_frame(pd.DataFrame([{"GAME_ID": "G1"}]))


# ---------------------------------------------------------------------------
# Task 019
# ---------------------------------------------------------------------------

def test_derive_tip_utc_from_pbp_uses_first_real_event_not_quotes():
    raw = pd.DataFrame({
        "game_id": ["G1", "G1", "G1", "G2"],
        "time_actual": [
            "2026-01-01T23:10:05.1Z", "2026-01-01T23:10:07.0Z",
            "2026-01-02T01:30:00.0Z", "2026-01-01T20:00:00.0Z",
        ],
    })
    tip = derive_tip_utc_from_pbp(raw)
    tip = tip.set_index("GAME_ID")
    assert tip.loc["G1", "tip_utc"] == pd.Timestamp("2026-01-01T23:10:05.1Z")
    assert tip.loc["G2", "tip_utc"] == pd.Timestamp("2026-01-01T20:00:00.0Z")


def test_every_usable_quote_can_compute_minutes_before_tip():
    rows = [_row(game_id="G1", quote_timestamp="2026-01-01T21:00:00Z")]
    df = build_quotes_frame(rows)
    tip_map = {"G1": pd.Timestamp("2026-01-01T23:00:00Z")}
    out = attach_tip_utc(df, tip_map)
    assert out.loc[0, "minutes_before_tip"] == pytest.approx(120.0)


def test_missing_tip_utc_is_nan_not_fabricated():
    rows = [_row(game_id="G_UNKNOWN")]
    df = build_quotes_frame(rows)
    out = attach_tip_utc(df, {})
    assert pd.isna(out.loc[0, "tip_utc"])
    assert pd.isna(out.loc[0, "minutes_before_tip"])


# ---------------------------------------------------------------------------
# Task 020
# ---------------------------------------------------------------------------

def test_post_tip_quote_rejected():
    tip = pd.Timestamp("2026-01-01T23:00:00Z")
    rows = [
        _row(game_id="G1", quote_timestamp="2026-01-01T22:00:00Z"),  # pregame, valid
        _row(game_id="G1", quote_timestamp="2026-01-01T23:30:00Z"),  # post-tip
        _row(game_id="G1", quote_timestamp="2026-01-01T23:00:00Z"),  # exactly at tip
    ]
    df = attach_tip_utc(build_quotes_frame(rows), {"G1": tip})
    out = reject_invalid_quotes(df)
    assert len(out) == 1
    assert out.iloc[0]["quote_timestamp"] == pd.Timestamp("2026-01-01T22:00:00Z")


def test_in_play_quote_rejected():
    tip = pd.Timestamp("2026-01-01T23:00:00Z")
    rows = [_row(game_id="G1", quote_timestamp="2026-01-01T20:00:00Z", in_play=True)]
    df = attach_tip_utc(build_quotes_frame(rows), {"G1": tip})
    out = reject_invalid_quotes(df)
    assert out.empty


def test_ambiguous_match_rejected():
    tip = pd.Timestamp("2026-01-01T23:00:00Z")
    df = attach_tip_utc(build_quotes_frame([_row(game_id="G1", quote_timestamp="2026-01-01T20:00:00Z")]), {"G1": tip})
    df["match_ambiguous"] = True
    out = reject_invalid_quotes(df)
    assert out.empty


def test_reject_requires_minutes_before_tip_precomputed():
    df = build_quotes_frame([_row()])
    with pytest.raises(ValueError):
        reject_invalid_quotes(df)


# ---------------------------------------------------------------------------
# Task 021
# ---------------------------------------------------------------------------

def test_open_quote_is_first_valid_pregame_quote():
    rows = [
        _row(quote_timestamp="2026-01-01T18:00:00Z", price_american=-102.0),
        _row(quote_timestamp="2026-01-01T20:00:00Z", price_american=-110.0),
        _row(quote_timestamp="2026-01-01T21:00:00Z", price_american=-115.0),
    ]
    df = build_quotes_frame(rows)
    out = select_open_quotes(df)
    assert len(out) == 1
    assert out.iloc[0]["open_price_american"] == -102.0
    assert out.iloc[0]["open_quote_timestamp"] == pd.Timestamp("2026-01-01T18:00:00Z")


def test_open_quote_selection_deterministic_under_shuffle():
    rows = [
        _row(quote_timestamp="2026-01-01T18:00:00Z", price_american=-102.0),
        _row(quote_timestamp="2026-01-01T20:00:00Z", price_american=-110.0),
        _row(quote_timestamp="2026-01-01T19:00:00Z", price_american=-105.0),
    ]
    df = build_quotes_frame(rows)
    out1 = select_open_quotes(df)
    shuffled = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    out2 = select_open_quotes(shuffled)
    pd.testing.assert_frame_equal(
        out1.sort_values(list(out1.columns)).reset_index(drop=True),
        out2.sort_values(list(out2.columns)).reset_index(drop=True),
    )


# ---------------------------------------------------------------------------
# Task 022 — boundary tests at T-61 / T-60 / T-59
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("minutes_before_tip,expected_included", [(61, True), (60, True), (59, False)])
def test_decision_quote_boundary(minutes_before_tip, expected_included):
    tip = pd.Timestamp("2026-01-01T23:00:00Z")
    qts = tip - pd.Timedelta(minutes=minutes_before_tip)
    rows = [_row(game_id="G1", quote_timestamp=qts.isoformat())]
    df = attach_tip_utc(build_quotes_frame(rows), {"G1": tip})
    out = select_decision_quotes(df, cutoff_minutes=60.0)
    row = out.iloc[0]
    if expected_included:
        assert row["decision_missing"] == False  # noqa: E712
        assert row["decision_price_american"] == -110.0
    else:
        assert row["decision_missing"] == True  # noqa: E712
        assert pd.isna(row["decision_price_american"])


def test_decision_quote_picks_latest_eligible_quote():
    tip = pd.Timestamp("2026-01-01T23:00:00Z")
    rows = [
        _row(game_id="G1", quote_timestamp="2026-01-01T20:00:00Z", price_american=-105.0),  # T-180
        _row(game_id="G1", quote_timestamp="2026-01-01T21:30:00Z", price_american=-108.0),  # T-90
        _row(game_id="G1", quote_timestamp="2026-01-01T21:59:00Z", price_american=-111.0),  # T-61
        _row(game_id="G1", quote_timestamp="2026-01-01T22:30:00Z", price_american=-120.0),  # T-30, too late
    ]
    df = attach_tip_utc(build_quotes_frame(rows), {"G1": tip})
    out = select_decision_quotes(df, cutoff_minutes=60.0)
    assert out.iloc[0]["decision_price_american"] == -111.0
    assert out.iloc[0]["decision_quote_age_minutes"] == pytest.approx(1.0)


def test_decision_missing_when_no_quote_before_cutoff():
    tip = pd.Timestamp("2026-01-01T23:00:00Z")
    rows = [_row(game_id="G1", quote_timestamp="2026-01-01T22:30:00Z")]  # only T-30
    df = attach_tip_utc(build_quotes_frame(rows), {"G1": tip})
    out = select_decision_quotes(df, cutoff_minutes=60.0)
    assert out.iloc[0]["decision_missing"] == True  # noqa: E712


# ---------------------------------------------------------------------------
# Task 023
# ---------------------------------------------------------------------------

def test_close_is_last_valid_pregame_quote():
    tip = pd.Timestamp("2026-01-01T23:00:00Z")
    rows = [
        _row(game_id="G1", quote_timestamp="2026-01-01T20:00:00Z", price_american=-105.0),
        _row(game_id="G1", quote_timestamp="2026-01-01T22:58:00Z", price_american=-118.0),
        _row(game_id="G1", quote_timestamp="2026-01-01T23:05:00Z", price_american=-999.0),  # post-tip, excluded upstream
    ]
    df = attach_tip_utc(build_quotes_frame(rows), {"G1": tip})
    valid = reject_invalid_quotes(df)
    out = select_close_quotes(valid)
    assert out.iloc[0]["close_price_american"] == -118.0


def test_t60_feature_frame_never_contains_close_columns():
    tip = pd.Timestamp("2026-01-01T23:00:00Z")
    rows = [
        _row(game_id="G1", quote_timestamp="2026-01-01T20:00:00Z"),
        _row(game_id="G1", quote_timestamp="2026-01-01T22:58:00Z"),
    ]
    df = reject_invalid_quotes(attach_tip_utc(build_quotes_frame(rows), {"G1": tip}))
    out = build_t60_feature_frame(df)
    assert not any(c.startswith("close_") for c in out.columns)


def test_assert_no_close_leak_raises_on_close_prefixed_key():
    with pytest.raises(ValueError):
        assert_no_close_leak(["decision_price", "close_price", "open_price"])
    assert_no_close_leak(["decision_price", "open_price"])  # does not raise


def test_evaluation_frame_includes_close_for_grading_only():
    tip = pd.Timestamp("2026-01-01T23:00:00Z")
    rows = [
        _row(game_id="G1", quote_timestamp="2026-01-01T20:00:00Z"),
        _row(game_id="G1", quote_timestamp="2026-01-01T22:58:00Z", price_american=-118.0),
    ]
    df = reject_invalid_quotes(attach_tip_utc(build_quotes_frame(rows), {"G1": tip}))
    out = build_evaluation_frame(df)
    assert "close_price_american" in out.columns
    assert out.iloc[0]["close_price_american"] == -118.0


# ---------------------------------------------------------------------------
# Task 028: point CLV vs price/probability CLV
# ---------------------------------------------------------------------------

def test_point_clv_home_bet_line_moves_toward_home():
    # Bettor took Home at -3.0 (decision); line closed at -5.0 (moved further
    # toward home) -> good CLV for a home bettor.
    clv = point_clv(decision_home_point=-3.0, close_home_point=-5.0, side="home")
    assert clv == pytest.approx(2.0)


def test_point_clv_away_bet_line_moves_toward_home_is_bad_for_away():
    clv = point_clv(decision_home_point=-3.0, close_home_point=-5.0, side="away")
    assert clv == pytest.approx(-2.0)


def test_point_clv_unknown_side_raises():
    with pytest.raises(ValueError):
        point_clv(-3.0, -5.0, side="sideways")


def test_price_clv_is_probability_domain_not_point_domain():
    clv = price_clv(decision_fair_prob=0.55, close_fair_prob=0.60)
    assert clv == pytest.approx(0.05)
    # Point CLV and price CLV must be independent columns/values, not
    # combined by a multiplicative-return identity.
    pt = point_clv(-3.0, -5.0, side="home")
    assert pt != clv
    assert pt == pytest.approx(2.0)  # points, not a return/probability


def test_price_clv_missing_inputs_are_nan_not_zero():
    assert np.isnan(price_clv(np.nan, 0.5))
    assert np.isnan(point_clv(np.nan, -5.0, "home"))


# ---------------------------------------------------------------------------
# Task 029: coverage audit + source-horizon detection
# ---------------------------------------------------------------------------

def test_detect_source_horizon_flags_constant_time_of_day_as_unknown():
    ts = pd.Series(["2026-01-01 10:00:00", "2026-01-05 10:00:00", "2026-02-01 10:00:00"])
    assert detect_source_horizon(ts) == "unknown_horizon"


def test_detect_source_horizon_accepts_real_intraday_variation():
    ts = pd.Series(["2026-01-01 14:03:11", "2026-01-01 18:47:02", "2026-01-01 21:59:59"])
    assert detect_source_horizon(ts) == "timestamped"


def test_all_odds_csv_is_unknown_horizon():
    """Task 029 / Phase 2B: audit the actual timing semantics of
    all_odds.csv rather than assuming it is a T-60 snapshot. Its game_date
    time-of-day component is a fixed dummy value for every row, so it must
    be labeled unknown_horizon, not used for T-60 ATS claims."""
    from pipeline.config import MODERN_ODDS_PATH

    if not MODERN_ODDS_PATH.exists():
        pytest.skip("all_odds.csv not present")
    df = pd.read_csv(MODERN_ODDS_PATH, usecols=["game_date"])
    assert detect_source_horizon(df["game_date"]) == "unknown_horizon"


def test_nba_main_lines_csv_is_genuinely_timestamped():
    """The Pinnacle export DOES carry real per-quote timestamps (used for
    open/T-60/close reconstruction), unlike all_odds.csv."""
    from pipeline.config import PINNACLE_LINES_PATH

    if not PINNACLE_LINES_PATH.exists():
        pytest.skip("nba_main_lines.csv not present")
    df = pd.read_csv(PINNACLE_LINES_PATH, usecols=["timestamp"], nrows=5000)
    assert detect_source_horizon(df["timestamp"]) == "timestamped"


def test_audit_market_coverage_reports_by_season_and_book():
    canonical = pd.DataFrame({
        "GAME_ID": ["G1", "G2", "G3"],
        "season_type": ["Regular", "Regular", "Regular"],
    })
    tip = pd.Timestamp("2026-01-01T23:00:00Z")
    rows = [
        _row(game_id="G1", quote_timestamp="2026-01-01T20:00:00Z"),   # open + T-60 ok
        _row(game_id="G1", quote_timestamp="2026-01-01T21:00:00Z"),   # T-60 ok too (T-120)
        _row(game_id="G2", quote_timestamp="2026-01-01T22:45:00Z"),   # only T-15, no T-60
    ]
    quotes = attach_tip_utc(build_quotes_frame(rows), {"G1": tip, "G2": tip})
    valid = reject_invalid_quotes(quotes)
    report = audit_market_coverage(canonical, valid)
    row = report[report["book"] == "pinnacle"].iloc[0]
    assert row["n_games"] == 3
    assert row["n_with_open"] == 2
    assert row["n_with_decision_t60"] == 1  # G1 only; G2's only quote is T-15
    assert row["n_with_close"] == 2
    assert row["decision_t60_coverage"] == pytest.approx(1.0 / 3.0)


def test_require_valid_t60_season_blocks_low_coverage_season():
    coverage = pd.DataFrame([
        {"season": 2024, "book": "pinnacle", "decision_t60_coverage": 0.0},
        {"season": 2025, "book": "pinnacle", "decision_t60_coverage": 0.9},
    ])
    require_valid_t60_season(coverage, 2025)  # should not raise
    with pytest.raises(ValueError):
        require_valid_t60_season(coverage, 2024)
    with pytest.raises(ValueError):
        require_valid_t60_season(coverage, 1999)  # season absent entirely


# ---------------------------------------------------------------------------
# Task 025 support: unique game matching (never a different game's price)
# ---------------------------------------------------------------------------

def test_match_pinnacle_quotes_never_cross_matches_same_team_different_game():
    from pipeline.dates import to_tricode

    canonical = pd.DataFrame({
        "GAME_ID": ["G1", "G2"],
        "home_team": ["BOS", "BOS"],
        "away_team": ["MIA", "NYK"],
        "raw_date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05")],
    })
    pinnacle = pd.DataFrame([
        {"team1": "Boston Celtics", "team2": "Miami Heat", "game_link": "l1",
         "team1_moneyline": 1.5, "team2_moneyline": 2.7,
         "team1_spread": -4.5, "team1_spread_odds": 1.91,
         "team2_spread": 4.5, "team2_spread_odds": 1.91,
         "over_total": 220.5, "over_total_odds": 1.91,
         "under_total": 220.5, "under_total_odds": 1.91,
         "timestamp": "2026-01-01 20:00:00"},
        {"team1": "Boston Celtics", "team2": "New York Knicks", "game_link": "l2",
         "team1_moneyline": 1.4, "team2_moneyline": 3.0,
         "team1_spread": -6.5, "team1_spread_odds": 1.91,
         "team2_spread": 6.5, "team2_spread_odds": 1.91,
         "over_total": 215.5, "over_total_odds": 1.91,
         "under_total": 215.5, "under_total_odds": 1.91,
         "timestamp": "2026-01-05 20:00:00"},
    ])
    quotes = match_pinnacle_quotes_to_games(pinnacle, canonical, to_tricode=to_tricode)
    spread_home_g1 = quotes[(quotes["GAME_ID"] == "G1") & (quotes["market"] == "spread") & (quotes["side"] == "home")]
    spread_home_g2 = quotes[(quotes["GAME_ID"] == "G2") & (quotes["market"] == "spread") & (quotes["side"] == "home")]
    assert spread_home_g1.iloc[0]["point"] == -4.5
    assert spread_home_g2.iloc[0]["point"] == -6.5
    # BOS home spread for G1 must never leak into G2 or vice versa.
    assert spread_home_g1.iloc[0]["point"] != spread_home_g2.iloc[0]["point"]


def test_match_pinnacle_quotes_rejects_ambiguous_matchup():
    from pipeline.dates import to_tricode

    canonical = pd.DataFrame({
        "GAME_ID": ["G1", "G2"],
        "home_team": ["BOS", "BOS"],
        "away_team": ["MIA", "MIA"],
        "raw_date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-02")],
    })
    pinnacle = pd.DataFrame([{
        "team1": "Boston Celtics", "team2": "Miami Heat", "game_link": "l1",
        "team1_moneyline": 1.5, "team2_moneyline": 2.7,
        "team1_spread": -4.5, "team1_spread_odds": 1.91,
        "team2_spread": 4.5, "team2_spread_odds": 1.91,
        "over_total": 220.5, "over_total_odds": 1.91,
        "under_total": 220.5, "under_total_odds": 1.91,
        "timestamp": "2026-01-01 20:00:00",
    }])
    quotes = match_pinnacle_quotes_to_games(pinnacle, canonical, to_tricode=to_tricode, match_tolerance_days=2)
    assert quotes.empty  # both G1 and G2 are within tolerance -> ambiguous -> reject
