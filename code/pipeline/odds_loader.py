"""Odds loading with explicit quote-source provenance (T-60 adversarial fix).

Prefers ``market_snapshots`` + authoritative tip UTC when quote-level rows are
available. Falls back to legacy ``load_pinnacle_lines`` (tip-proxy) only with an
explicit ``quote_source=\"tip_proxy\"`` flag for run manifests.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.market import canonicalize_odds_dict, load_modern_odds, load_pinnacle_lines


def load_odds_dict(
    *,
    modern_odds_path: str | Path | None = None,
    pinnacle_path: str | Path | None = None,
    schedule_df: pd.DataFrame | None = None,
    quotes_df: pd.DataFrame | None = None,
    tip_utc_map: dict | pd.DataFrame | None = None,
    cutoff_minutes: float = 60.0,
) -> tuple[dict, dict[str, Any]]:
    """Return ``(odds_dict, provenance)``.

    When ``quotes_df`` and ``tip_utc_map`` are both provided, decision lines are
    taken from ``select_decision_quotes`` (canonical T-60). Otherwise the legacy
    tip-proxy pinnacle loader is used and ``quote_source`` is ``tip_proxy``.
    """
    odds_dict: dict = {}
    provenance: dict[str, Any] = {
        "quote_source": None,
        "used_market_snapshots": False,
        "used_tip_proxy_fallback": False,
        "promotion_eligible": False,
        "warning": None,
    }

    if modern_odds_path is not None and Path(modern_odds_path).exists():
        odds_dict = load_modern_odds(str(modern_odds_path))

    if quotes_df is not None and tip_utc_map is not None and not quotes_df.empty:
        from pipeline.market_snapshots import (
            attach_tip_utc,
            reject_invalid_quotes,
            select_decision_quotes,
        )
        q = attach_tip_utc(quotes_df, tip_utc_map)
        q = reject_invalid_quotes(q)
        decided = select_decision_quotes(q, cutoff_minutes=cutoff_minutes)
        # Convert decision spread quotes into odds_dict entries when possible.
        # Minimal bridge: home-side spread rows keyed by (date, home) require
        # schedule join — callers that supply quotes should also supply schedule.
        snap_odds = _decision_quotes_to_odds_dict(decided, schedule_df)
        if snap_odds:
            odds_dict.update(snap_odds)
            provenance["quote_source"] = "market_snapshots"
            provenance["used_market_snapshots"] = True
            provenance["promotion_eligible"] = True
            odds_dict = canonicalize_odds_dict(odds_dict)
            return odds_dict, provenance

    if pinnacle_path is not None and Path(pinnacle_path).exists():
        pin = load_pinnacle_lines(str(pinnacle_path), schedule_df=schedule_df)
        odds_dict.update(pin)
        provenance["quote_source"] = "tip_proxy"
        provenance["used_tip_proxy_fallback"] = True
        provenance["promotion_eligible"] = False
        provenance["warning"] = (
            "Using legacy load_pinnacle_lines tip-proxy for T-60 cutoff. "
            "Do not use for promotion ROI claims; prefer market_snapshots + PBP tip. "
            "Runs with quote_source=tip_proxy are research-only."
        )
        print(f"  ⚠ {provenance['warning']}")

    odds_dict = canonicalize_odds_dict(odds_dict)
    if provenance["quote_source"] is None:
        provenance["quote_source"] = "modern_odds_only"
        provenance["promotion_eligible"] = False
        if provenance["warning"] is None:
            provenance["warning"] = (
                "modern_odds_only lacks decision≠close provenance; "
                "research-only for CLV/ROI promotion."
            )
    return odds_dict, provenance


def _decision_quotes_to_odds_dict(
    decided: pd.DataFrame,
    schedule_df: pd.DataFrame | None,
) -> dict:
    """Best-effort bridge from decision_* quote rows to (date, team) odds keys."""
    if decided is None or decided.empty or schedule_df is None or schedule_df.empty:
        return {}
    # ``point`` stays in GROUP_KEY_COLS; prices are decision_* prefixed.
    point_col = "decision_point" if "decision_point" in decided.columns else (
        "point" if "point" in decided.columns else None
    )
    if point_col is None:
        return {}
    sched = schedule_df.copy()
    date_col = "date" if "date" in sched.columns else "game_date"
    if date_col not in sched.columns or "GAME_ID" not in sched.columns:
        return {}
    if "home" not in sched.columns:
        return {}
    home_col = "home"

    spreads = decided
    if "market" in decided.columns:
        spreads = decided[decided["market"].astype(str).str.lower() == "spread"]
    if "decision_missing" in spreads.columns:
        spreads = spreads[~spreads["decision_missing"].fillna(False)]
    out = {}
    keep_cols = ["GAME_ID", date_col, home_col]
    merged = spreads.merge(sched[keep_cols], on="GAME_ID", how="inner")
    for _, row in merged.iterrows():
        d = pd.Timestamp(row[date_col]).date()
        home = row[home_col]
        side = str(row.get("side", "home")).lower()
        point = row.get(point_col)
        price = row.get("decision_price_american", row.get("price_american"))
        if pd.isna(point):
            continue
        entry = out.get((d, home), {})
        if side == "home":
            entry["decision_spread"] = float(point)
            entry["spread"] = float(point)
            if pd.notna(price):
                entry["spread_home_price"] = float(price)
        elif side == "away":
            entry["spread_away"] = float(point)
            if pd.notna(price):
                entry["spread_away_price"] = float(price)
        out[(d, home)] = entry
    return out
