"""Epic 10.1.3 — schema parity between synthetic factories and the pipeline.

The bench is only meaningful if factory rows look exactly like real pipeline
rows: a missing column or wrong dtype in ``make_stint`` / ``make_odds`` would
make bench failures mean fake-data artifacts instead of formula bugs.

Strategy:
- Derive the stint schema *dynamically* by running ``pipeline.stints.build_stints``
  on a minimal synthetic play-by-play frame, then require ``make_stint()`` /
  ``make_stint_frame()`` output to cover every produced column with a
  compatible dtype. If ``stints.py`` ever grows a column, this test fails
  until the factory catches up.
- Require ``make_odds()`` to expose every key consumed by ``pipeline.market``
  feature builders, and prove consumability by actually calling the builders
  with factory output (a missing key raises KeyError here, not in a bench).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.market import (
    bet_analysis,
    fair_probs_from_ml_pair,
    is_single_sided_ml_quote,
    market_microstructure_features,
    select_ml_bet,
    select_ou_bet,
    spread_kelly_fraction,
)
from pipeline.stints import build_stints
from tests.synth.factories import make_odds, make_stint, make_stint_frame
from tests.synth.presets import NAN_STORM

# Raw per-event counting columns that ``build_stints`` sums into stint rows.
_RAW_BOX_COLS = [
    "home_oreb", "away_oreb", "home_dreb", "away_dreb",
    "home_tovs_forced", "away_tovs_forced",
    "home_fouls_drawn", "away_fouls_drawn",
    "home_fgm", "away_fgm", "home_fga", "away_fga",
    "home_tov", "away_tov", "home_fta", "away_fta",
    "home_stl", "away_stl", "home_blks", "away_blks",
    "home_3pm", "away_3pm", "home_3pa", "away_3pa",
    "home_poss", "away_poss",
    "home_xefg_added", "away_xefg_added",
    "home_rim_fga", "away_rim_fga",
    "home_three_fga", "away_three_fga",
    "home_shot_dist_sum", "away_shot_dist_sum",
]

# Columns ``pipeline.game_updates`` reads off stint rows (anchored against
# source; the dynamic build_stints check above is the primary guard).
_GAME_UPDATES_CONSUMED = {
    "GAME_ID", "HOME_players", "AWAY_players", "possessions",
    "home_xpts", "away_xpts", "home_usage", "away_usage", "PERIOD",
    "HOME_SCORE_START", "AWAY_SCORE_START", "HOME_SCORE_END", "AWAY_SCORE_END",
    "home_pts", "away_pts", "game_date", "home_team", "away_team",
    "home_rim_fga", "away_rim_fga", "home_three_fga", "away_three_fga",
    "home_fga", "away_fga", "home_fgm", "away_fgm",
    "home_3pm", "away_3pm", "home_fta", "away_fta",
}

# Keys ``pipeline.market`` feature builders consume from an odds/feature row:
# fair_probs_from_ml_pair / select_ml_bet / bet_analysis (market_ml_home,
# market_ml_away, market_spread), select_ou_bet (market_total),
# spread_kelly_fraction (juice), market_microstructure_features / bet_analysis
# (spread_move, public_home_pct), and the T-60/CLV line fields
# (decision_spread, closing_spread, spread_open).
_MARKET_CONSUMED_KEYS = {
    "market_ml_home", "market_ml_away", "market_spread", "market_total",
    "juice", "decision_spread", "closing_spread", "spread_open",
    "spread_move", "public_home_pct",
}

_KIND_GROUPS = {
    "i": "numeric", "u": "numeric", "f": "numeric",
    "b": "bool", "O": "object", "M": "datetime", "m": "timedelta",
}


def _dtype_group(dtype) -> str:
    return _KIND_GROUPS[np.dtype(dtype).kind]


def _minimal_pbp_frame() -> pd.DataFrame:
    """Smallest play-by-play frame ``build_stints`` will aggregate (one stint)."""
    base = {
        "GAME_ID": "0022500001",
        "game_date": pd.Timestamp("2025-11-01"),
        "home_team": "GSW",
        "away_team": "LAL",
        "PERIOD": 1,
        "HOME_players": "1-2-3-4-5",
        "AWAY_players": "6-7-8-9-10",
        "HOME_SCORE": 0.0,
        "AWAY_SCORE": 0.0,
        "home_pts_added": 2.0,
        "away_pts_added": 1.0,
        "home_xpts_added": 1.9,
        "away_xpts_added": 0.8,
        "total_possessions": 1.0,
        "PLAYER1_ID": 1.0,
        "PLAYER2_ID": np.nan,
        "is_fg_make": 1,
        "is_fg_miss": 0,
        "is_tov": 0,
        "is_true_ft_trip": 0,
        "EVENTMSGTYPE": 1,
    }
    rows = []
    for _ in range(2):
        row = dict(base)
        for c in _RAW_BOX_COLS:
            row[c] = 0.0
        rows.append(row)
    return pd.DataFrame(rows)


@pytest.mark.bench
def test_stint_factory_covers_build_stints_output_schema():
    real = build_stints(_minimal_pbp_frame())
    assert len(real) == 1  # fixture sanity: two events, one stint
    factory_keys = set(make_stint().keys())
    missing = [c for c in real.columns if c not in factory_keys]
    assert not missing, f"make_stint() missing build_stints columns: {missing}"
    frame_keys = set(make_stint_frame(n=2, seed=0).columns)
    missing_frame = [c for c in real.columns if c not in frame_keys]
    assert not missing_frame, f"make_stint_frame() missing columns: {missing_frame}"


@pytest.mark.bench
def test_stint_factory_dtypes_compatible_with_build_stints():
    real = build_stints(_minimal_pbp_frame())
    factory_df = pd.DataFrame([make_stint()])
    for col in real.columns:
        got = _dtype_group(factory_df[col].dtype)
        want = _dtype_group(real[col].dtype)
        assert got == want, f"{col}: factory dtype group {got} != pipeline {want}"


@pytest.mark.bench
def test_stint_factory_covers_game_updates_consumed_columns():
    missing = _GAME_UPDATES_CONSUMED - set(make_stint().keys())
    assert not missing, f"make_stint() missing game_updates columns: {missing}"


@pytest.mark.bench
def test_stint_factory_value_shapes_match_pipeline():
    row = make_stint()
    assert isinstance(row["GAME_ID"], str)
    assert isinstance(row["game_date"], pd.Timestamp)
    # Lineup strings parse to five player ids, as game_updates expects.
    for key in ("HOME_players", "AWAY_players"):
        assert isinstance(row[key], str)
        ids = row[key].split("-")
        assert len(ids) == 5 and all(ids)
    # Usage dicts are keyed by lineup players with 3-slot counters,
    # exactly as build_stints emits them.
    assert set(row["home_usage"].keys()) == set(row["HOME_players"].split("-"))
    assert set(row["away_usage"].keys()) == set(row["AWAY_players"].split("-"))
    for v in row["home_usage"].values():
        assert isinstance(v, list) and len(v) == 3


@pytest.mark.bench
def test_make_odds_covers_market_feature_builder_keys():
    missing = _MARKET_CONSUMED_KEYS - set(make_odds().keys())
    assert not missing, f"make_odds() missing keys consumed by market.py: {missing}"


@pytest.mark.bench
def test_make_odds_output_consumable_by_market_builders():
    o = make_odds(ml_home=-140, ml_away=120, spread=-3.0, total=221.5)

    fh, fa = fair_probs_from_ml_pair(o["market_ml_home"], o["market_ml_away"])
    assert 0.0 < fh < 1.0
    assert fh + fa == pytest.approx(1.0, abs=1e-9)

    side, _ev, _dec = select_ml_bet(
        0.62, o["market_ml_home"],
        market_spread=o["market_spread"], market_ml_away=o["market_ml_away"],
    )
    assert side in {"Home", "Away", "Pass"}

    micro = market_microstructure_features(
        o["spread_move"], o["public_home_pct"], o["market_spread"],
    )
    assert {"reverse_line_movement", "steam_flag", "market_spread_raw", "public_away_pct"} <= set(micro)

    assert spread_kelly_fraction(0.55, o["juice"]) >= 0.0

    direction, _p_over, _edge = select_ou_bet(227.0, o["market_total"])
    assert direction in {"Over", "Under", "Pass"}

    res = bet_analysis(
        model_spread=-6.0, market_spread=o["market_spread"],
        model_win_prob=0.62, market_ml=o["market_ml_home"],
        spread_move=o["spread_move"], public_home_pct=o["public_home_pct"],
        market_ml_away=o["market_ml_away"],
    )
    assert res["spread_direction"] in {"Home", "Away", "Pass"}
    assert res["ml_direction"] in {"Home", "Away", "Pass"}


@pytest.mark.bench
def test_make_odds_one_sided_still_consumable():
    o = make_odds(ml_home=-150, one_sided=True)
    assert is_single_sided_ml_quote(o["market_ml_away"])
    fh, fa = fair_probs_from_ml_pair(o["market_ml_home"], o["market_ml_away"])
    assert fh + fa == pytest.approx(1.0, abs=1e-9)
    side, _ev, _dec = select_ml_bet(0.6, o["market_ml_home"], market_ml_away=o["market_ml_away"])
    assert side in {"Home", "Away", "Pass"}


@pytest.mark.bench
def test_nan_storm_odds_row_consumable():
    """Every-NaN odds row must flow through builders as Pass/0.5, never crash."""
    fh, fa = fair_probs_from_ml_pair(NAN_STORM["market_ml_home"], NAN_STORM["market_ml_away"])
    assert (fh, fa) == (0.5, 0.5)
    side, _ev, _dec = select_ml_bet(
        0.6, NAN_STORM["market_ml_home"], market_spread=NAN_STORM["market_spread"],
    )
    assert side == "Pass"
    micro = market_microstructure_features(
        NAN_STORM["spread_move"], NAN_STORM["public_home_pct"], NAN_STORM["market_spread"],
    )
    assert micro["steam_flag"] == 0
    assert micro["reverse_line_movement"] == 0.0
    direction, p_over, _edge = select_ou_bet(225.0, NAN_STORM["market_total"])
    assert direction == "Pass" and p_over == 0.5
