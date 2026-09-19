"""Honesty banners for tip-proxy / ats_status."""
from __future__ import annotations

import json

import pandas as pd

from dashboard.services.runs import summarize_results


def test_tip_proxy_banner(tmp_path, monkeypatch):
    csv = tmp_path / "backtest_results.csv"
    pd.DataFrame({
        "ATS_WIN": [1, 0],
        "quote_source": ["tip_proxy", "tip_proxy"],
        "SPREAD_ERR": [1.0, 2.0],
    }).to_csv(csv, index=False)
    monkeypatch.setattr("dashboard.services.runs.results_csv_path", lambda run_id=None: csv)
    monkeypatch.setattr("dashboard.services.runs.load_ats_status", lambda run_id=None: None)
    monkeypatch.setattr("dashboard.services.runs.load_odds_provenance", lambda run_id=None: None)
    s = summarize_results(None)
    assert s["research_only_banner"] is True
    assert "tip_proxy" in (s["banner_message"] or "")


def test_ats_status_banner(tmp_path, monkeypatch):
    csv = tmp_path / "backtest_results.csv"
    pd.DataFrame({"ATS_WIN": [1]}).to_csv(csv, index=False)
    monkeypatch.setattr("dashboard.services.runs.results_csv_path", lambda run_id=None: csv)
    monkeypatch.setattr(
        "dashboard.services.runs.load_ats_status",
        lambda run_id=None: {"status": "research_only", "message": "ATS blocked by T-60 gate"},
    )
    monkeypatch.setattr("dashboard.services.runs.load_odds_provenance", lambda run_id=None: None)
    s = summarize_results(None)
    assert s["research_only_banner"] is True
    assert "ATS blocked" in (s["banner_message"] or "")
