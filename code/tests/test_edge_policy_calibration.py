"""Tests for post-hoc edge policy calibration."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.edge_policy_calibration import (
    edge_policy_grid,
    policy_lean_frame,
    recommend_edge_floor,
    season_edge_summary,
)


def _toy_results(n: int = 120) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    edges = rng.uniform(1.0, 10.0, size=n)
    # Positive edge → Home lean covers more often when |edge| high.
    cover = rng.random(n) < (0.48 + 0.02 * (edges / 10.0))
    margin = np.where(cover, 5.0, -5.0)
    decision = np.full(n, -3.0)
    close = decision + rng.uniform(-1.5, 1.5, size=n)
    return pd.DataFrame({
        "MARKET_SPREAD": np.full(n, -3.0),
        "DECISION_SPREAD": decision,
        "CLOSING_SPREAD": close,
        "ACTUAL_MARGIN": margin,
        "EDGE": edges,
        "EDGE_LEAN": np.where(edges > 0, "Home", "Away"),
        "DIRECTION": "Pass",  # live gate left many Pass — lean still recoverable
        "CONFIDENCE": np.full(n, 70.0),
        "CONF_WIDTH": np.full(n, 10.0),
        "DISAGREEMENT_TRUST": np.full(n, 1.0),
        "PHANTOM_INJURY_FLAG": np.zeros(n, dtype=bool),
        "ELO_META_AGREEMENT": np.full(n, 0.8),
        "simulated_season_window": np.where(np.arange(n) < n // 2, 2025, 2026),
    })


def test_policy_lean_includes_subthreshold():
    df = _toy_results()
    lean = policy_lean_frame(df)
    assert len(lean) == len(df)
    assert (lean["abs_edge"] > 0).all()


def test_grid_more_bets_at_lower_floor():
    df = _toy_results()
    grid = edge_policy_grid(
        df, thresholds=(3.5, 5.5), use_avoid_band=False, full_gates=True, min_bets=5,
    )
    assert not grid.empty
    n_lo = int(grid.loc[grid["min_edge"] == 3.5, "n_bets"].iloc[0])
    n_hi = int(grid.loc[grid["min_edge"] == 5.5, "n_bets"].iloc[0])
    assert n_lo >= n_hi


def test_full_gates_stricter_than_edge_only():
    df = _toy_results()
    df.loc[df.index[:40], "CONFIDENCE"] = 10.0  # fail min confidence
    full = edge_policy_grid(df, thresholds=(3.5,), use_avoid_band=False, full_gates=True, min_bets=1)
    edge = edge_policy_grid(df, thresholds=(3.5,), use_avoid_band=False, full_gates=False, min_bets=1)
    assert int(full.iloc[0]["n_bets"]) <= int(edge.iloc[0]["n_bets"])


def test_recommend_respects_nonneg_clv_when_possible():
    df = _toy_results(200)
    # Force positive CLV for Home leans: decision more favorable than close.
    df["DECISION_SPREAD"] = -4.0
    df["CLOSING_SPREAD"] = -2.0
    grid = edge_policy_grid(
        df, thresholds=(3.5, 4.5, 5.5, 6.5), use_avoid_band=False, min_bets=10,
    )
    rec = recommend_edge_floor(grid, require_nonneg_clv=True, min_bets=10, min_clv_n=5)
    assert rec["reason"] in ("ok", "no_row_with_nonneg_clv", "no_row_meets_min_bets")
    if rec["reason"] == "ok":
        assert rec["recommended_min_edge"] is not None
        assert rec["candidate"]["mean_clv"] >= 0


def test_season_summary_counts():
    df = _toy_results()
    tbl = season_edge_summary(df, min_edge=5.5)
    assert set(tbl["season"]) == {2025, 2026}
    assert (tbl["n_lean"] > 0).all()
