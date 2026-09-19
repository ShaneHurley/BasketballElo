"""Extreme / adversarial presets for the formula test bench (Epic 10.1.2).

Each preset is a call-ready kwargs dict (or list thereof) for the factories.
Unrealistic values are intentional.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from tests.synth.factories import make_odds, make_player_ids, make_stint

# 150–20 Q4 blowout: stresses garbage-time weighting (P0.8 double-apply).
BLOWOUT_60 = make_stint(
    period=4,
    possessions=2.0,
    home_xpts=8.0,
    away_xpts=0.5,
    home_score_start=140.0,
    away_score_start=18.0,
    home_score_end=150.0,
    away_score_end=20.0,
    garbage=True,
)

# Zero-possession FT stint: points without possessions (stints Task 012 class).
ZERO_POSSESSION_FT_STINT = make_stint(
    possessions=0.0,
    home_xpts=0.0,
    away_xpts=0.0,
    home_pts=1.0,
    away_pts=0.0,
    home_score_start=50.0,
    away_score_start=50.0,
    home_score_end=51.0,
    away_score_end=50.0,
)

# Single-sided historical feed: away ML missing (P0.3).
ONE_SIDED_ODDS_FEED = make_odds(ml_home=-150.0, one_sided=True)

# Juice extremes for Kelly payout identity (P0.2).
JUICE_EXTREMES = (-10000, -130, -120, -115, -110, -105, -100, 100, 105, 120, 5000, 10000)

# Rookie (0 games) vs 5000-game veteran — RD / K-decay stress (P0.5).
ROOKIE_VS_5000_GAME_VET = {
    "rookie": {"player_id": "rook", "games": 0, "poss_per_game": 20.0},
    "veteran": {"player_id": "vet", "games": 5000, "poss_per_game": 70.0},
}

# Every optional odds/feature field NaN.
NAN_STORM = {
    "market_ml_home": np.nan,
    "market_ml_away": np.nan,
    "market_spread": np.nan,
    "market_total": np.nan,
    "juice": np.nan,
    "decision_spread": np.nan,
    "closing_spread": np.nan,
    "spread_move": np.nan,
    "public_home_pct": np.nan,
}

# Same five-man unit all season (max sample for lineup Elo shrinkage).
SINGLE_LINEUP_ALL_SEASON = {
    "home_players": make_player_ids(5, start=1),
    "away_players": make_player_ids(5, start=6),
}

# Max chemistry churn: new disjoint lineup every stint.
DISJOINT_LINEUPS_EVERY_STINT = [
    make_stint(
        stint_id=i,
        home_players=make_player_ids(5, start=1 + 10 * i),
        away_players=make_player_ids(5, start=6 + 10 * i),
        game_date=pd.Timestamp("2025-11-01") + pd.Timedelta(days=i),
    )
    for i in range(8)
]
