"""Tests for moneyline diagnostics helpers."""
import pandas as pd

from pipeline.ml_diagnostics import enrich_ml_columns, ml_value_plays


def _sample_results():
    return pd.DataFrame({
        "DATE": pd.to_datetime(["2024-01-01", "2024-01-02"]),
        "GAME_ID": ["1", "2"],
        "HOME": ["BOS", "LAL"],
        "AWAY": ["NYK", "GSW"],
        "ACTUAL_HOME": [110, 100],
        "ACTUAL_AWAY": [105, 108],
        "ACTUAL_MARGIN": [5, -8],
        "MARKET_ML": [-150, 200],
        "WIN_PROB": [0.62, 0.42],
        "simulated_season_window": ["2023-2024", "2023-2024"],
        "DIRECTION": ["Pass", "Pass"],
        "EDGE": [0.0, 0.0],
    })


def test_enrich_ml_columns_adds_value_type():
    df = enrich_ml_columns(_sample_results(), min_ev=0.03)
    assert "ml_value_type" in df.columns
    assert "ml_best_side" in df.columns


def test_ml_value_plays_filters_types():
    df = enrich_ml_columns(_sample_results(), min_ev=0.01, max_fav_dec=1.0)
    plays = ml_value_plays(df, min_ev=0.01, max_fav_dec=1.0)
    assert isinstance(plays, pd.DataFrame)
