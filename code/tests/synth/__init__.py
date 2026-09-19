"""Synthetic formula test bench — factories, presets, canaries.

This package lives under ``code/tests/synth/`` on purpose: it must never be
imported from production ``pipeline/`` paths. It feeds deliberately unrealistic
and adversarial fake data through pipeline stages so mathematical invariants
(symmetry, conservation, monotonicity, boundedness) become unmissable.

See ``README.md`` in this directory and roadmap Epic 10.
"""
from __future__ import annotations

from tests.synth.factories import (
    make_game,
    make_odds,
    make_player_ids,
    make_player_season,
    make_stint,
    make_stint_frame,
)
from tests.synth.presets import (
    BLOWOUT_60,
    DISJOINT_LINEUPS_EVERY_STINT,
    JUICE_EXTREMES,
    NAN_STORM,
    ONE_SIDED_ODDS_FEED,
    ROOKIE_VS_5000_GAME_VET,
    SINGLE_LINEUP_ALL_SEASON,
    ZERO_POSSESSION_FT_STINT,
)

__all__ = [
    "make_stint",
    "make_stint_frame",
    "make_game",
    "make_odds",
    "make_player_ids",
    "make_player_season",
    "BLOWOUT_60",
    "ZERO_POSSESSION_FT_STINT",
    "ONE_SIDED_ODDS_FEED",
    "JUICE_EXTREMES",
    "ROOKIE_VS_5000_GAME_VET",
    "NAN_STORM",
    "SINGLE_LINEUP_ALL_SEASON",
    "DISJOINT_LINEUPS_EVERY_STINT",
]
