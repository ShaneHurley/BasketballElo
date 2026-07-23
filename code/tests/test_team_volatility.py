"""Unit tests for TeamVolatilityTracker."""
import numpy as np

from pipeline.team_volatility import TeamVolatilityTracker


def test_volatility_stake_multiplier_shrinkage():
    tv = TeamVolatilityTracker(window=20, min_games=3)
    for i in range(10):
        tv.update("A", 5.0, 20.0 if i % 2 == 0 else -10.0, home=True)
        tv.update("B", 5.0, 0.0, home=False)
    mult = tv.stake_multiplier("A", "B")
    assert 0.5 <= mult <= 1.5


def test_save_load_roundtrip(tmp_path):
    tv = TeamVolatilityTracker(window=10, min_games=2)
    tv.update_matchup("LAL", "BOS", 3.0, 8.0)
    tv.update_matchup("LAL", "BOS", 3.0, -2.0)
    path = tmp_path / "vol.pkl"
    tv.save_state(path)
    loaded = TeamVolatilityTracker.load_state(path)
    assert loaded.stake_multiplier("LAL", "BOS") == tv.stake_multiplier("LAL", "BOS")
