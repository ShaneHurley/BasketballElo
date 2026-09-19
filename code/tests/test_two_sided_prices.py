"""Tasks 024-025: retain actual home AND away ML/spread/total prices in
pipeline/market.py (never infer the away American price by negating the
home price), and fix game matching so two games sharing a team can never
exchange prices."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.market import (
    american_to_decimal,
    decimal_to_american,
    get_closing_odds,
    get_game_odds,
    implied_probability,
    load_pinnacle_lines,
)


# ---------------------------------------------------------------------------
# Task 024: round-trip price conversion + two-sided fixture tests.
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("american", [-350, -110, -105, 100, 120, 450, 900])
def test_american_decimal_round_trip(american):
    dec = american_to_decimal(american)
    back = decimal_to_american(dec)
    assert back == pytest.approx(american, abs=1.0)


def test_get_game_odds_exposes_actual_away_price_fields():
    odds_dict = {
        (pd.Timestamp("2026-01-01").date(), "BOS"): {
            "spread": -4.5, "ml": -180, "total": 220.5,
            "spread_away": 4.5, "ml_away": 155,
            "spread_home_price": -110.0, "spread_away_price": -108.0,
            "total_over_price": -105.0, "total_under_price": -112.0,
        }
    }
    out = get_game_odds(pd.Timestamp("2026-01-01"), "BOS", odds_dict)
    assert out["ml"] == -180
    assert out["ml_away"] == 155
    assert out["spread_away_points"] == 4.5
    assert out["spread_home_price"] == -110.0
    assert out["spread_away_price"] == -108.0
    # Away ML/price must be the *real* stored values, never -home.
    assert out["ml_away"] != -out["ml"]


def test_get_game_odds_missing_away_fields_are_nan_not_negated():
    odds_dict = {(pd.Timestamp("2026-01-01").date(), "BOS"): {"spread": -4.5, "ml": -180}}
    out = get_game_odds(pd.Timestamp("2026-01-01"), "BOS", odds_dict)
    assert pd.isna(out["ml_away"])  # missing, not -(-180) == 180


def test_load_pinnacle_lines_retains_both_sides_two_sided_fixture(tmp_path):
    csv_path = tmp_path / "nba_main_lines.csv"
    csv_path.write_text(
        "team1,team2,game_link,team1_moneyline,team2_moneyline,team1_spread,"
        "team1_spread_odds,team2_spread,team2_spread_odds,over_total,"
        "over_total_odds,under_total,under_total_odds,timestamp\n"
        "Boston Celtics,Miami Heat,link1,1.50,2.70,-4.5,1.91,4.5,1.91,220.5,1.91,220.5,1.91,"
        "2026-01-01 20:00:00\n"
    )
    schedule = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-01")], "home": ["BOS"], "away": ["MIA"],
    })
    odds = load_pinnacle_lines(str(csv_path), schedule_df=schedule)
    d = pd.Timestamp("2026-01-01").date()
    home_entry = odds[(d, "BOS")]
    away_entry = odds[(d, "MIA")]

    home_ml_dec, away_ml_dec = 1.50, 2.70
    expected_home_ml = decimal_to_american(home_ml_dec)
    expected_away_ml = decimal_to_american(away_ml_dec)

    assert home_entry["ml"] == pytest.approx(expected_home_ml)
    assert home_entry["ml_away"] == pytest.approx(expected_away_ml)
    assert away_entry["ml"] == pytest.approx(expected_away_ml)
    assert away_entry["ml_away"] == pytest.approx(expected_home_ml)
    # The away ML must not be a negation of the home ML (this fixture uses
    # genuinely asymmetric decimal odds).
    assert home_entry["ml_away"] != -home_entry["ml"]
    assert home_entry["spread"] == pytest.approx(-4.5)
    assert home_entry["spread_away"] == pytest.approx(4.5)


# ---------------------------------------------------------------------------
# Task 025: canonical-identity game matching; no cross-game backfill.
# ---------------------------------------------------------------------------

def test_get_closing_odds_requires_exact_date_no_latest_prior_game_fallback():
    # Same team ("BOS") has odds on two different dates; a naive "latest
    # prior game" fallback would return the Jan-5 price when asked about
    # Jan-10 (no entry that day). The fixed function must return NaN instead
    # of silently substituting a different game's price.
    odds_dict = {
        (pd.Timestamp("2026-01-05").date(), "BOS"): {"spread": -4.5, "ml": -180},
    }
    spread, ml = get_closing_odds(pd.Timestamp("2026-01-10"), "BOS", odds_dict)
    assert pd.isna(spread)
    assert pd.isna(ml)


def test_get_closing_odds_matches_exact_game():
    odds_dict = {
        (pd.Timestamp("2026-01-05").date(), "BOS"): {
            "spread": -4.5, "ml": -180, "away_team": "MIA",
        },
    }
    spread, ml = get_closing_odds(pd.Timestamp("2026-01-05"), "BOS", odds_dict, away_team="MIA")
    assert spread == -4.5
    assert ml == -180


def test_get_closing_odds_rejects_wrong_opponent_on_same_date():
    # Two different home games for BOS on the same date key would be a data
    # error, but even a single stored row for the wrong opponent must not be
    # returned when the caller specifies the real opponent.
    odds_dict = {
        (pd.Timestamp("2026-01-05").date(), "BOS"): {
            "spread": -4.5, "ml": -180, "away_team": "MIA",
        },
    }
    spread, ml = get_closing_odds(pd.Timestamp("2026-01-05"), "BOS", odds_dict, away_team="NYK")
    assert pd.isna(spread)
    assert pd.isna(ml)


def test_pinnacle_loader_separates_decision_t60_from_close(tmp_path):
    """T-60 decision line (last quote ≥60m before close) is the bet line;
    closing_spread remains the final snapshot — CLV can be nonzero."""
    csv_path = tmp_path / "nba_main_lines.csv"
    csv_path.write_text(
        "team1,team2,game_link,team1_moneyline,team2_moneyline,team1_spread,"
        "team1_spread_odds,team2_spread,team2_spread_odds,over_total,"
        "over_total_odds,under_total,under_total_odds,timestamp\n"
        "Boston Celtics,Miami Heat,link1,1.50,2.70,-4.5,1.91,4.5,1.91,220.5,1.91,220.5,1.91,"
        "2026-01-01 18:00:00\n"
        "Boston Celtics,Miami Heat,link1,1.55,2.60,-5.0,1.91,5.0,1.91,220.5,1.91,220.5,1.91,"
        "2026-01-01 22:30:00\n"
    )
    schedule = pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "home": ["BOS"], "away": ["MIA"]})
    odds = load_pinnacle_lines(str(csv_path), schedule_df=schedule)
    entry = odds[(pd.Timestamp("2026-01-01").date(), "BOS")]
    assert entry["closing_spread"] == pytest.approx(-5.0)
    assert entry["decision_spread"] == pytest.approx(-4.5)
    assert entry["spread"] == pytest.approx(-4.5)
    assert entry["spread"] != entry["closing_spread"]
    go = get_game_odds(pd.Timestamp("2026-01-01"), "BOS", odds)
    assert go["spread"] == pytest.approx(-4.5)
    assert go["closing_spread"] == pytest.approx(-5.0)
    assert go["decision_spread"] == pytest.approx(-4.5)


def test_two_games_same_team_cannot_exchange_prices_via_pinnacle_loader(tmp_path):
    csv_path = tmp_path / "nba_main_lines.csv"
    csv_path.write_text(
        "team1,team2,game_link,team1_moneyline,team2_moneyline,team1_spread,"
        "team1_spread_odds,team2_spread,team2_spread_odds,over_total,"
        "over_total_odds,under_total,under_total_odds,timestamp\n"
        "Boston Celtics,Miami Heat,link1,1.50,2.70,-4.5,1.91,4.5,1.91,220.5,1.91,220.5,1.91,"
        "2026-01-01 20:00:00\n"
        "Boston Celtics,New York Knicks,link2,1.40,3.00,-6.5,1.91,6.5,1.91,215.5,1.91,215.5,1.91,"
        "2026-01-05 20:00:00\n"
    )
    schedule = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-01"), pd.Timestamp("2026-01-05")],
        "home": ["BOS", "BOS"], "away": ["MIA", "NYK"],
    })
    odds = load_pinnacle_lines(str(csv_path), schedule_df=schedule)
    g1 = odds[(pd.Timestamp("2026-01-01").date(), "BOS")]
    g2 = odds[(pd.Timestamp("2026-01-05").date(), "BOS")]
    assert g1["spread"] != g2["spread"]
    assert g1["spread"] == pytest.approx(-4.5)
    assert g2["spread"] == pytest.approx(-6.5)
