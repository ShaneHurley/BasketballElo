"""Epic 11.3 / P0.10 — Rename fair_spread_vigfree to honest label.

GitHub: P0.10 · Registry ID: ``bench_p0_vigfree_honesty``

``market_microstructure_features`` returned a raw single-book spread under
the key ``fair_spread_vigfree`` — but no vig removal was ever applied to it.
The key is renamed to ``market_spread_raw`` to accurately describe the
value's provenance.
"""
from __future__ import annotations

import pytest

from pipeline.market import market_microstructure_features


@pytest.mark.bench
def test_p0_10_market_spread_raw_key_present_vigfree_absent():
    """Acceptance: output uses market_spread_raw, never the old vigfree name."""
    result = market_microstructure_features(
        spread_move=0, public_home_pct=0.5, market_spread=-3.5
    )
    assert "market_spread_raw" in result
    assert "fair_spread_vigfree" not in result
    assert result["market_spread_raw"] == pytest.approx(-3.5)


@pytest.mark.bench
def test_p0_10_nan_market_spread_defaults_to_zero_under_new_key():
    """Witness: NaN market_spread still maps through the renamed key."""
    result = market_microstructure_features(
        spread_move=0, public_home_pct=0.5, market_spread=float("nan")
    )
    assert "market_spread_raw" in result
    assert "fair_spread_vigfree" not in result
    assert result["market_spread_raw"] == 0.0
