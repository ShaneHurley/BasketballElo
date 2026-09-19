"""Tests for edge compression diagnosis helpers."""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.edge_compression import (
    confidence_floor_grid,
    gate_attrition_counts,
    lean_frame,
    rank_causes,
    season_edge_table,
)


def _toy(n: int = 80, edge_scale: float = 6.0, conf: float = 60.0) -> pd.DataFrame:
    rng = np.random.default_rng(1)
    edges = rng.uniform(1.0, edge_scale, size=n)
    return pd.DataFrame({
        "MARKET_SPREAD": np.full(n, -3.0),
        "DECISION_SPREAD": np.full(n, -3.5),
        "CLOSING_SPREAD": np.full(n, -3.0),
        "ACTUAL_MARGIN": rng.normal(0, 12, size=n),
        "PRED_SPREAD": -3.0 + edges,  # pred_residual ≈ edge when market=-3
        "EDGE": edges,
        "EDGE_LEAN": np.where(edges > 0, "Home", "Away"),
        "DIRECTION": "Pass",
        "CONFIDENCE": np.full(n, conf),
        "CONF_WIDTH": np.full(n, 12.0),
        "DISAGREEMENT_TRUST": np.full(n, 1.0),
        "PHANTOM_INJURY_FLAG": np.zeros(n, dtype=bool),
        "ELO_META_AGREEMENT": np.full(n, 0.8),
        "simulated_season_window": "2025-2026",
    })


def test_lean_and_season_table():
    df = _toy()
    lean = lean_frame(df)
    assert len(lean) == len(df)
    tbl = season_edge_table(df)
    assert len(tbl) == 1
    assert tbl.iloc[0]["n_lean"] == len(df)


def test_attrition_and_conf_grid():
    df = _toy(edge_scale=10.0, conf=40.0)
    attr = gate_attrition_counts(df, min_edge=5.5, min_conf=55.0)
    assert attr["pass_edge"] > 0
    assert attr["pass_conf"] == 0  # conf 40 < 55
    grid = confidence_floor_grid(df, floors=(40.0, 55.0), min_edge=5.5)
    n40 = int(grid.loc[grid["min_conf"] == 40.0, "n_bets"].iloc[0])
    n55 = int(grid.loc[grid["min_conf"] == 55.0, "n_bets"].iloc[0])
    assert n40 >= n55


def test_rank_causes_elo_blend():
    causes = rank_causes(
        overlap={
            "smoke_edge_p50": 6.0,
            "full_edge_p50": 2.5,
            "smoke_pred_resid_p50": 6.0,
            "full_pred_resid_p50": 2.5,
        },
        smoke_knobs={"elo_blend_alpha": 0.04},
        full_knobs={"elo_blend_alpha": 0.39},
        smoke_attr={"pass_edge": 200, "pass_conf": 150, "min_conf": 55},
        full_attr={"pass_edge": 200, "pass_conf": 20, "min_conf": 55},
    )
    codes = [c["code"] for c in causes]
    assert codes[0] in ("A", "B")
    assert "B" in codes
