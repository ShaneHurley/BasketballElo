"""Optional CLV gate for actionable spread bets."""
import pandas as pd

from pipeline.bet_selection import actionable_spread_frame
from pipeline.metrics import apply_edge_threshold


def test_actionable_frame_uses_actionable_column(monkeypatch):
    import pipeline.config as cfg

    monkeypatch.setattr(cfg, "BET_SELECTION_MODE", "confidence_only")
    df = pd.DataFrame({
        "MARKET_SPREAD": [-3.0, -3.0],
        "DIRECTION": ["Home", "Pass"],
        "ACTIONABLE": [1, 0],
    })
    out = actionable_spread_frame(df)
    assert len(out) == 1
