"""Odds loader provenance and snapshot preference tests."""
from __future__ import annotations

import pandas as pd

from pipeline.market_snapshots import build_quotes_frame
from pipeline.odds_loader import load_odds_dict


def test_load_odds_prefers_snapshots_when_quotes_and_tip_present():
    tip = pd.Timestamp("2025-01-15T19:00:00Z")
    rows = [
        {
            "GAME_ID": "g1",
            "book": "pinnacle",
            "market": "spread",
            "side": "home",
            "point": -3.5,
            "price_american": -110,
            "price_decimal": 1.909,
            "quote_timestamp": tip - pd.Timedelta(minutes=90),
            "source_timestamp": tip - pd.Timedelta(minutes=90),
            "ingestion_timestamp": tip - pd.Timedelta(minutes=89),
            "in_play": False,
        }
    ]
    quotes = build_quotes_frame(rows)
    sched = pd.DataFrame({
        "GAME_ID": ["g1"],
        "date": [pd.Timestamp("2025-01-15")],
        "home": ["BOS"],
        "away": ["NYK"],
    })
    odds, prov = load_odds_dict(
        quotes_df=quotes,
        tip_utc_map={"g1": tip},
        schedule_df=sched,
        modern_odds_path=None,
        pinnacle_path=None,
    )
    assert prov["quote_source"] == "market_snapshots"
    assert prov["used_market_snapshots"] is True
    assert prov.get("promotion_eligible") is True
    assert (pd.Timestamp("2025-01-15").date(), "BOS") in odds


def test_load_odds_tip_proxy_flag_without_quotes(tmp_path):
    # No quotes/tip → tip_proxy if pinnacle missing → modern_odds_only
    odds, prov = load_odds_dict(
        modern_odds_path=None,
        pinnacle_path=None,
        quotes_df=None,
        tip_utc_map=None,
    )
    assert prov["quote_source"] == "modern_odds_only"
    assert prov["used_tip_proxy_fallback"] is False
    assert prov.get("promotion_eligible") is False


def test_load_odds_tip_proxy_allowed_without_quotes(tmp_path, monkeypatch):
    pin = tmp_path / "pinnacle.csv"
    pin.write_text("dummy\n")
    monkeypatch.setattr(
        "pipeline.odds_loader.load_pinnacle_lines",
        lambda *a, **k: {"dummy": {"spread": -3.5}},
    )
    monkeypatch.setattr("pipeline.odds_loader.canonicalize_odds_dict", lambda d: d)
    odds, prov = load_odds_dict(pinnacle_path=pin, allow_tip_proxy=True)
    assert prov["quote_source"] == "tip_proxy"
    assert prov["used_tip_proxy_fallback"] is True
    assert prov.get("promotion_eligible") is False
    assert "dummy" in odds
