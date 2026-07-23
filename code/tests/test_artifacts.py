"""Artifact schema validation."""
import pandas as pd
import pytest

from pipeline.artifacts import (
    ArtifactMismatchError,
    assert_manifest_compatible,
    backtest_config_snapshot,
    build_run_manifest,
    full_config_snapshot,
    hash_source_file,
    is_backtest_stale,
    validate_backtest_df,
    validate_run_manifest,
)


def _sample_df():
    return pd.DataFrame({
        "EDGE": [3.0, -2.0],
        "EDGE_LEAN": ["Home", "Away"],
        "WIN_PCT": [70, 55],
        "ACTIONABLE": [1, 0],
        "DIRECTION": ["Home", "Pass"],
        "CONFIDENCE": [70, 55],
        "MIN_CONFIDENCE_SCORE": [64, 64],
        "MARKET_SPREAD": [-3.5, 2.0],
        "ACTUAL_MARGIN": [5, -1],
    })


def test_validate_backtest_df_ok():
    assert validate_backtest_df(_sample_df()) == []


def test_is_backtest_stale_missing_columns():
    df = _sample_df().drop(columns=["ACTIONABLE"])
    assert is_backtest_stale(df)


def test_backtest_config_snapshot_keys():
    snap = backtest_config_snapshot()
    assert "BET_SELECTION_MODE" in snap
    assert "artifact_schema_version" in snap
    assert "preprocessing_schema_version" in snap
    assert "decision_cutoff_minutes_before_tip" in snap


# ── Task 003: expanded provenance ──────────────────────────────────────────

def test_full_config_snapshot_includes_schema_versions():
    snap = full_config_snapshot()
    assert snap["PREPROCESSING_SCHEMA_VERSION"] >= 1
    assert snap["FEATURE_SCHEMA_VERSION"] >= 1
    assert snap["MARKET_SNAPSHOT_SCHEMA_VERSION"] >= 1
    assert snap["VALIDATION_SCHEMA_VERSION"] >= 1


def test_build_run_manifest_has_required_provenance_fields(tmp_path):
    src = tmp_path / "source.csv"
    src.write_text("a,b\n1,2\n")
    manifest = build_run_manifest(
        source_paths={"pbp": src},
        game_count=1307,
        train_range=("2025-10-21", "2026-03-01"),
        calibration_range=("2026-03-02", "2026-04-01"),
        test_range=("2026-04-02", "2026-05-18"),
        feature_list=["h_off_rtg", "a_off_rtg"],
        model_version="v1",
        seeds={"numpy": 42},
    )
    assert manifest["source_hashes"]["pbp"] == hash_source_file(src)
    for key in (
        "artifact_schema_version", "preprocessing_schema_version",
        "feature_schema_version", "market_snapshot_schema_version",
        "validation_schema_version",
    ):
        assert key in manifest["schema_versions"]
    assert manifest["cutoff_definition"]["minutes_before_scheduled_tip"] == 60
    assert manifest["date_ranges"]["train"] == ["2025-10-21", "2026-03-01"]
    assert manifest["feature_list"] == ["a_off_rtg", "h_off_rtg"]
    assert manifest["model_version"] == "v1"
    assert manifest["seeds"] == {"numpy": 42}


def test_hash_source_file_changes_with_content(tmp_path):
    src = tmp_path / "source.csv"
    src.write_text("a,b\n1,2\n")
    h1 = hash_source_file(src, fast=False)
    src.write_text("a,b\n1,3\n")
    h2 = hash_source_file(src, fast=False)
    assert h1 != h2


def test_validate_run_manifest_detects_schema_mismatch():
    expected = build_run_manifest(model_version="v2")
    saved = dict(expected)
    saved["schema_versions"] = dict(expected["schema_versions"])
    saved["schema_versions"]["preprocessing_schema_version"] = -1
    reasons = validate_run_manifest(saved, expected)
    assert any("preprocessing_schema_version" in r for r in reasons)


def test_assert_manifest_compatible_rejects_mismatched_artifact():
    expected = build_run_manifest(model_version="v2", feature_list=["a", "b"])
    saved = dict(expected)
    saved["feature_list"] = ["a", "b", "c_new_feature"]
    with pytest.raises(ArtifactMismatchError):
        assert_manifest_compatible(saved, expected)


def test_assert_manifest_compatible_accepts_matching_artifact():
    manifest = build_run_manifest(model_version="v2", feature_list=["a", "b"])
    assert_manifest_compatible(manifest, manifest)  # no raise
