"""Unit tests for edge-bucket bet selection helpers."""
import numpy as np

from pipeline.bet_selection import (
    apply_bet_selection_gates,
    edge_stake_multiplier,
    passes_edge_bucket,
    passes_quantile_width,
    use_tier_stake_gates,
)
import pipeline.config as cfg


def test_passes_edge_bucket():
    assert passes_edge_bucket(6.0, min_edge=5.5)
    assert not passes_edge_bucket(4.0, min_edge=5.5)
    assert not passes_edge_bucket(np.nan)


def test_passes_quantile_width():
    assert passes_quantile_width(20.0, max_width=25.0)
    assert not passes_quantile_width(30.0, max_width=25.0)


def test_edge_stake_multiplier():
    assert edge_stake_multiplier(6.0) == 1.0
    assert edge_stake_multiplier(9.0) == 1.5


def test_apply_bet_selection_gates_legacy(monkeypatch):
    monkeypatch.setattr(cfg, "BET_SELECTION_MODE", "legacy_tiers")
    assert apply_bet_selection_gates("Home", 3.0, 20.0) == "Home"


def test_apply_bet_selection_gates_edge_bucket(monkeypatch):
    monkeypatch.setattr(cfg, "BET_SELECTION_MODE", "edge_bucket")
    monkeypatch.setattr(cfg, "MIN_EDGE_BUCKET", 5.5)
    assert apply_bet_selection_gates("Home", 6.0, 20.0) == "Home"
    assert apply_bet_selection_gates("Home", 4.0, 20.0) == "Pass"
    assert apply_bet_selection_gates("Home", 6.0, 30.0) == "Pass"


def test_use_tier_stake_gates(monkeypatch):
    monkeypatch.setattr(cfg, "BET_SELECTION_MODE", "legacy_tiers")
    monkeypatch.setattr(cfg, "USE_TIER_STAKE_GATES", True)
    assert use_tier_stake_gates() is True
    monkeypatch.setattr(cfg, "BET_SELECTION_MODE", "edge_bucket")
    assert use_tier_stake_gates() is False


def test_confidence_only_actionable_frame(monkeypatch):
    import pandas as pd
    from pipeline.bet_selection import actionable_spread_frame, lean_spread_frame

    monkeypatch.setattr(cfg, "BET_SELECTION_MODE", "confidence_only")
    df = pd.DataFrame({
        "MARKET_SPREAD": [-3.5, 2.0, -1.0],
        "EDGE": [3.0, -2.0, 0.5],
        "EDGE_LEAN": ["Home", "Away", "Home"],
        "ACTIONABLE": [1, 0, 1],
        "DIRECTION": ["Home", "Pass", "Home"],
    })
    assert len(lean_spread_frame(df)) == 3
    assert len(actionable_spread_frame(df)) == 2


def test_apply_ou_threshold():
    import pandas as pd
    from pipeline.metrics import apply_ou_threshold

    df = pd.DataFrame({
        "PRED_TOTAL": [230.0, 210.0, 220.0],
        "MARKET_TOTAL": [220.0, 220.0, 220.0],
        "ACTUAL_HOME": [110, 100, 105],
        "ACTUAL_AWAY": [115, 105, 110],
    })
    out = apply_ou_threshold(df, 5.0)
    assert out.loc[0, "OU_DIRECTION"] == "Over"
    assert out.loc[1, "OU_DIRECTION"] == "Under"
    assert out.loc[2, "OU_DIRECTION"] == "Pass"
