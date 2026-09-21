"""Planted leak canaries for Epic 10.4.

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


@dataclass
class SliceReuseAttack:
    """Registers the identical calibration row-id slice with two different
    calibrators (Task 053 ``calibration_slice_reuse`` leak, replayed).

    The attack succeeds only if the registry accepts both registrations; a
    guarded registry must raise on the second ``register`` call.
    """

    row_ids: tuple = (101, 102, 103, 104, 105)
    target_a: str = "elo_isotonic"
    target_b: str = "metascore_isotonic"

    def attempt(self, registry) -> None:
        """Register ``row_ids`` for two distinct targets (duck-typed: any
        ``CalibrationSliceRegistry``-shaped object with ``register``)."""
        if self.target_a == self.target_b:
            raise ValueError("SliceReuseAttack requires two distinct calibrator targets")
        registry.register(self.target_a, self.row_ids)
        registry.register(self.target_b, self.row_ids)
