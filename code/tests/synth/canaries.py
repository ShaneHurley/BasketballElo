"""Planted leak canaries for Epic 10.4 (stub — full harness lands with 10.4).

Deliberately leaking components used only by the bench. Never import from
production prediction paths.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class PsychicFeature:
    """Column perfectly correlated with a *future* outcome label."""

    name: str = "psychic_cover"

    def attach(self, df: pd.DataFrame, outcome_col: str = "ACTUAL_MARGIN") -> pd.DataFrame:
        out = df.copy()
        y = out[outcome_col].to_numpy(dtype=float)
        out[self.name] = (y > 0).astype(float)
        return out


@dataclass
class TimeTravelerTracker:
    """Updates state using the game being predicted (classic T-60 violation)."""

    last_seen_game_id: str | None = None

    def peek_then_update(self, game_id: str, future_signal: float) -> float:
        """Return ``future_signal`` then record the game — simulates look-ahead."""
        self.last_seen_game_id = game_id
        return float(future_signal)


@dataclass
class FutureOddsQuote:
    """Close quote timestamped *before* the decision quote (provenance canary)."""

    decision_ts: pd.Timestamp
    close_ts: pd.Timestamp
    decision_spread: float
    closing_spread: float

    @classmethod
    def inverted(cls, tip: pd.Timestamp) -> "FutureOddsQuote":
        return cls(
            decision_ts=tip - pd.Timedelta(minutes=10),
            close_ts=tip - pd.Timedelta(minutes=60),  # close earlier than decision
            decision_spread=-3.5,
            closing_spread=-4.0,
        )

    def is_chronology_broken(self) -> bool:
        return self.close_ts < self.decision_ts
