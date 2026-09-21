"""Epic 5.4 helpers — real quotes grade CLV; never train actuals on the line.

Production ``load_odds_dict`` already prefers ``quotes_df`` + ``tip_utc_map``
when both are supplied (``promotion_eligible=True``). These helpers build the
tip map from PBP/stints and document the wiring path without making tip_proxy
a gate on MAE work.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.market_snapshots import (
    attach_tip_utc,
    build_quotes_frame,
    derive_tip_utc_from_pbp,
    select_decision_quotes,
)


def tip_utc_map_from_pbp(
    raw_pbp_df: pd.DataFrame,
    *,
    game_id_col: str = "game_id",
    time_col: str = "time_actual",
) -> dict[str, pd.Timestamp]:
    """Authoritative tip UTC keyed by GAME_ID (never inferred from quotes)."""
    tips = derive_tip_utc_from_pbp(
        raw_pbp_df, game_id_col=game_id_col, time_col=time_col
    )
    out: dict[str, pd.Timestamp] = {}
    for _, row in tips.iterrows():
        gid = str(row["GAME_ID"])
        ts = pd.Timestamp(row["tip_utc"])
        if pd.notna(ts):
            out[gid] = ts.tz_convert("UTC") if ts.tzinfo else ts.tz_localize("UTC")
    return out


def load_quotes_csv(path: str | Path) -> pd.DataFrame:
    """Load a quote-level CSV into the market_snapshots schema."""
    df = pd.read_csv(path)
    return build_quotes_frame(df)


def decision_coverage_report(
    quotes_df: pd.DataFrame,
    tip_utc_map: dict | pd.DataFrame,
) -> dict[str, Any]:
    """Audit how many games have distinct decision≠close for CLV grading."""
    q = attach_tip_utc(quotes_df, tip_utc_map)
    decision = select_decision_quotes(q)
    n_games = int(decision["GAME_ID"].nunique()) if not decision.empty else 0
    return {
        "n_decision_games": n_games,
        "n_quote_rows": int(len(q)),
        "purpose": "CLV grading only — not an actuals training target",
    }
