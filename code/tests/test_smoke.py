"""Smoke tests for leak fixes and train/serve parity."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.model import build_feature_row, engineer_interaction_features, SAFE_FEATURE_COLS
from pipeline.ratings import PlayerRatingTracker
from pipeline.utils import map_elo_params
from pipeline.metrics import grid_search_bet_edge


def test_map_elo_params():
    cfg = map_elo_params({
        "k_off": 0.5, "k_def": 0.6, "elo_scaling": 800,
        "home_boost": 0.005, "offseason_reversion": 0.2,
        "usage_floor": 0.22, "assist_split": 0.8,
    })
    assert cfg["HOME_PPP_BOOST"] == 0.005
    assert cfg["K_OFF"] == 0.5


def test_build_feature_row_parity():
    raw = {
        "elo_net": 10.0, "exp_poss": 100.0, "h_rest": 2, "a_rest": 1,
        "roll_net_xppp": 0.05, "h_rating_uncertainty": 200, "a_rating_uncertainty": 180,
    }
    for col in SAFE_FEATURE_COLS:
        raw.setdefault(col, 0.0)
    row = build_feature_row(raw)
    assert row["elo_pace_interaction"] == 10.0 * 100.0
    assert row["elo_rest_interaction"] == 10.0 * 1.0


def test_margin_and_k_update():
    t = PlayerRatingTracker(config={"K_OFF": 0.5, "K_DEF": 0.4, "ELO_SCALING_FACTOR": 1000})
    p1, p2 = ["101", "102"]
    t.process_stint(
        ids_A=[p1], ids_B=[p2], poss=10,
        xpts_A=12, xpts_B=10, usage_A={"101": [1.0, 0, 0]}, usage_B={"102": [1.0, 0, 0]},
        period=4, start_A=80, start_B=70, end_A=85, end_B=72,
    )
    assert t.players["101"]["O_mu"] != 1500.0


def test_grid_search_bet_edge():
    rows = []
    for i in range(30):
        edge = 3.0 if i % 2 == 0 else -3.5
        direction = "Home" if edge > 0 else "Away"
        margin = 5 if direction == "Home" else -8
        spread = -2.5 if direction == "Home" else 6.0
        rows.append({"EDGE": edge, "DIRECTION": direction, "ACTUAL_MARGIN": margin, "MARKET_SPREAD": spread})
    df = pd.DataFrame(rows)
    best, grid = grid_search_bet_edge(df, edges=[1.5, 2.5, 4.0])
    assert best in [1.5, 2.5, 4.0]
    assert not grid.empty


if __name__ == "__main__":
    test_map_elo_params()
    test_build_feature_row_parity()
    test_margin_and_k_update()
    test_grid_search_bet_edge()
    print("All smoke tests passed.")
