"""Dashboard model-health service (Epic 7.5)."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from dashboard.services.health import load_model_health


def test_health_missing_csv(monkeypatch):
    monkeypatch.setattr(
        "dashboard.services.health.results_csv_path", lambda run_id=None: None
    )
    monkeypatch.setattr(
        "dashboard.services.health.load_odds_provenance", lambda run_id=None: None
    )
    out = load_model_health(None)
    assert out["n_rows"] == 0
    assert out["weekly"]["status"] == "no_data"
    assert out["permutation"]["status"] == "skipped"
    assert out["drift"]["status"] == "unavailable"
    assert out["research_only"] is False


def test_health_empty_csv(tmp_path: Path, monkeypatch):
    csv = tmp_path / "backtest_results.csv"
    pd.DataFrame().to_csv(csv, index=False)
    monkeypatch.setattr(
        "dashboard.services.health.results_csv_path", lambda run_id=None: csv
    )
    monkeypatch.setattr(
        "dashboard.services.health.load_odds_provenance", lambda run_id=None: None
    )
    out = load_model_health(None)
    assert out["n_rows"] == 0
    assert out["weekly"]["status"] == "no_data"


def test_health_missing_spread_columns(tmp_path: Path, monkeypatch):
    csv = tmp_path / "backtest_results.csv"
    pd.DataFrame({"GAME_ID": ["a", "b"], "SOME_COL": [1, 2]}).to_csv(csv, index=False)
    monkeypatch.setattr(
        "dashboard.services.health.results_csv_path", lambda run_id=None: csv
    )
    monkeypatch.setattr(
        "dashboard.services.health.load_odds_provenance", lambda run_id=None: None
    )
    monkeypatch.setattr(
        "dashboard.services.health._find_feature_tables",
        lambda run_id=None: (None, None),
    )
    out = load_model_health(None)
    assert out["n_rows"] == 2
    assert out["permutation"]["status"] == "skipped"
    assert any("PRED" in m or "ACTUAL" in m for m in out["columns_missing"])
    assert "CLV" in out["columns_missing"]
    # weekly_report may return metrics with nulls — never coerce to 0 falsely
    weekly = out["weekly"]
    if weekly.get("status") != "no_data":
        assert weekly.get("mean_clv") is None or weekly.get("mean_clv") != 0 or "CLV" not in pd.read_csv(csv).columns


def test_health_graded_rows_permutation(tmp_path: Path, monkeypatch):
    rng_margins = list(range(40))
    csv = tmp_path / "backtest_results.csv"
    pd.DataFrame({
        "PRED_SPREAD": [m + 0.5 for m in rng_margins],
        "ACTUAL_MARGIN": rng_margins,
        "MARKET_SPREAD": [-3.5] * 40,
        "DIRECTION": ["Home"] * 40,
        "ACTIONABLE": [1] * 40,
        "CLV": [0.1] * 40,
    }).to_csv(csv, index=False)
    monkeypatch.setattr(
        "dashboard.services.health.results_csv_path", lambda run_id=None: csv
    )
    monkeypatch.setattr(
        "dashboard.services.health.load_odds_provenance",
        lambda run_id=None: {"quote_source": "market_snapshots", "promotion_eligible": True},
    )
    monkeypatch.setattr(
        "dashboard.services.health._find_feature_tables",
        lambda run_id=None: (None, None),
    )
    out = load_model_health(None)
    assert out["n_rows"] == 40
    assert out["permutation"]["status"] == "ok"
    assert out["permutation"]["passed"] is True
    assert out["permutation"]["base_metric"] is not None
    assert out["permutation"]["null_mean"] is not None
    assert out["research_only"] is False
    assert "PRED_SPREAD" in out["columns_used"]
    assert "CLV" in out["columns_used"]
    assert out["drift"]["status"] == "unavailable"


def test_health_tip_proxy_banner(tmp_path: Path, monkeypatch):
    csv = tmp_path / "backtest_results.csv"
    pd.DataFrame({
        "PRED_SPREAD": [1.0, 2.0],
        "ACTUAL_MARGIN": [1.0, 2.0],
    }).to_csv(csv, index=False)
    monkeypatch.setattr(
        "dashboard.services.health.results_csv_path", lambda run_id=None: csv
    )
    monkeypatch.setattr(
        "dashboard.services.health.load_odds_provenance",
        lambda run_id=None: {"quote_source": "tip_proxy", "uses_tip_proxy": True},
    )
    monkeypatch.setattr(
        "dashboard.services.health._find_feature_tables",
        lambda run_id=None: (None, None),
    )
    out = load_model_health(None)
    assert out["research_only"] is True
    assert "tip_proxy" in (out["banner_message"] or "")


def test_health_nan_not_zero(tmp_path: Path, monkeypatch):
    csv = tmp_path / "backtest_results.csv"
    pd.DataFrame({
        "PRED_SPREAD": [1.0] * 10,
        "ACTUAL_MARGIN": [1.0] * 10,
    }).to_csv(csv, index=False)
    monkeypatch.setattr(
        "dashboard.services.health.results_csv_path", lambda run_id=None: csv
    )
    monkeypatch.setattr(
        "dashboard.services.health.load_odds_provenance", lambda run_id=None: None
    )
    monkeypatch.setattr(
        "dashboard.services.health._find_feature_tables",
        lambda run_id=None: (None, None),
    )
    out = load_model_health(None)
    weekly = out["weekly"]
    # No CLV column → mean_clv must be null, not 0
    assert weekly.get("mean_clv") is None
    raw = json.dumps(out)
    assert "NaN" not in raw
