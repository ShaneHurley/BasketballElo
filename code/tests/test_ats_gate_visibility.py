"""T-60 gate visibility tests."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from pipeline.t60_coverage import t60_betting_data_gate


def test_gate_blocked_status_payload_shape():
    rows = []
    for year in (2023, 2024):
        for _ in range(20):
            rows.append({"season": year, "decision_spread": -3.5, "market_spread": -3.5})
    gate = t60_betting_data_gate(pd.DataFrame(rows), min_seasons=3, min_coverage=0.5)
    assert gate["allow_betting_heads"] is False
    status = {"season": 2025, "gate": gate, "status": "blocked_research_only"}
    assert status["status"] == "blocked_research_only"
    # Manifest-serializable
    blob = json.dumps({"records": [status]}, default=str)
    assert "blocked_research_only" in blob


def test_ats_status_json_roundtrip(tmp_path: Path):
    path = tmp_path / "ats_status.json"
    payload = {
        "records": [
            {"season": 2025, "status": "blocked_research_only", "gate": {"allow_betting_heads": False}}
        ]
    }
    path.write_text(json.dumps(payload))
    loaded = json.loads(path.read_text())
    assert loaded["records"][0]["status"] == "blocked_research_only"
