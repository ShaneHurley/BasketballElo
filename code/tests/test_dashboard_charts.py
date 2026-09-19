"""ChartSpec registry tests."""
from __future__ import annotations

import json

import pandas as pd

from dashboard.services.charts import (
    CHARTS,
    build_chart,
    chart_cover_calibration,
    chart_edge_hist,
    chart_roi_by_conf,
    list_charts,
)


def test_chart_registry_nonempty():
    assert "edge_hist" in CHARTS
    assert "cover_calibration" in CHARTS
    assert "clv_by_season" in CHARTS
    ids = {c["id"] for c in list_charts()}
    assert "season_ats" in ids


def test_edge_hist_builds():
    df = pd.DataFrame({"EDGE": [0.5, -1.0, 2.0, 1.5, 0.0]})
    fig = chart_edge_hist(df)
    assert fig["data"]
    assert fig["layout"]["title"]


def test_build_chart_missing_csv(monkeypatch):
    monkeypatch.setattr("dashboard.services.charts.results_csv_path", lambda run_id=None: None)
    fig = build_chart("edge_hist", None)
    assert fig["data"] == []


def test_build_chart_from_csv(tmp_path, monkeypatch):
    csv = tmp_path / "r.csv"
    csv.write_text("EDGE,ATS_WIN\n1,1\n2,0\n-1,1\n")
    monkeypatch.setattr("dashboard.services.charts.results_csv_path", lambda run_id=None: csv)
    fig = build_chart("edge_hist", "any")
    assert fig["data"][0]["type"] == "histogram"


def test_rolling_mae_json_safe(tmp_path, monkeypatch):
    """Rolling windows emit leading NaNs; payload must still serialize (FastAPI strict JSON)."""
    csv = tmp_path / "r.csv"
    rows = ["SPREAD_ERR"] + [str(float(i % 7) + 0.5) for i in range(40)]
    csv.write_text("\n".join(rows) + "\n")
    monkeypatch.setattr("dashboard.services.charts.results_csv_path", lambda run_id=None: csv)
    fig = build_chart("rolling_mae", "any")
    assert fig["data"], "expected a line series"
    json.dumps(fig, allow_nan=False)


def test_cover_calibration_drops_empty_buckets():
    df = pd.DataFrame({
        "CALIBRATED_COVER_PROB": [0.1, 0.15, 0.55, 0.6, 0.9, 0.92],
        "ATS_WIN": [0, 0, 1, 1, 1, 0],
    })
    fig = chart_cover_calibration(df)
    assert fig["data"]
    xs = fig["data"][0]["x"]
    ys = fig["data"][0]["y"]
    assert all(v is not None and v == v for v in xs)  # no NaN
    assert all(v is not None and v == v for v in ys)
    json.dumps(fig, allow_nan=False)


def test_roi_by_conf_fallback_ats_win():
    df = pd.DataFrame({
        "CONFIDENCE_TIER": ["low", "low", "high", "high"],
        "ATS_WIN": [1, 0, 1, 1],
    })
    fig = chart_roi_by_conf(df)
    assert fig["data"]
    assert fig["data"][0]["type"] == "bar"
    json.dumps(fig, allow_nan=False)
