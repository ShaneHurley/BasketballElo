"""Task 002: the invalid pre-integrity-fix baseline manifest must explicitly
mark score labels, dates, CLV, and ROI in `newest data/` as untrusted."""
from pathlib import Path

import pytest

from pipeline.invalid_baseline import (
    KNOWN_DEFECTS,
    build_invalid_baseline_manifest,
    write_invalid_baseline_manifest,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_CSV = REPO_ROOT / "newest data" / "backtest_results.csv"
TUNING_JSON = REPO_ROOT / "newest data" / "tuning_results.json"


@pytest.mark.skipif(not BACKTEST_CSV.exists() or not TUNING_JSON.exists(),
                     reason="newest data/ source files missing")
def test_manifest_marks_accuracy_and_roi_untrusted():
    manifest = build_invalid_baseline_manifest(BACKTEST_CSV, TUNING_JSON)
    assert manifest["label"] == "invalid_pre_integrity_fix"
    stmt = manifest["trust_statement"].lower()
    for term in ("score", "date", "clv", "roi", "untrusted"):
        assert term in stmt, f"trust statement missing '{term}': {stmt}"
    for field in ("ACTUAL_HOME", "ACTUAL_MARGIN", "DATE", "CLOSING_SPREAD", "ROI"):
        assert field in manifest["untrusted_fields"]
    assert manifest["backtest_descriptive_counts"]["n_rows"] > 0
    assert manifest["backtest_descriptive_counts"]["n_unique_game_ids"] > 0


@pytest.mark.skipif(not BACKTEST_CSV.exists() or not TUNING_JSON.exists(),
                     reason="newest data/ source files missing")
def test_manifest_flags_close_equals_market_defect():
    manifest = build_invalid_baseline_manifest(BACKTEST_CSV, TUNING_JSON)
    counts = manifest["backtest_descriptive_counts"]
    assert counts["close_line_equals_market_line_for_all_lined_rows"] is True


def test_known_defects_cover_required_categories():
    ids = {d["id"] for d in KNOWN_DEFECTS}
    required = {
        "score_corruption", "date_corruption", "close_line_conflation",
        "stack_cv_future_leakage", "clv_definition_bug", "push_as_loss_bug",
        "cap_after_profit_bug",
    }
    assert required.issubset(ids)


@pytest.mark.skipif(not BACKTEST_CSV.exists() or not TUNING_JSON.exists(),
                     reason="newest data/ source files missing")
def test_write_manifest_roundtrip(tmp_path):
    out = write_invalid_baseline_manifest(BACKTEST_CSV, TUNING_JSON, tmp_path / "manifest.json")
    assert out.exists()
    import json
    saved = json.loads(out.read_text())
    assert saved["label"] == "invalid_pre_integrity_fix"
