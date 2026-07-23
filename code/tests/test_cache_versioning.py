"""Task 004: cache keys must change whenever a schema version changes.

Covers the stints cache key builder in run_backtest.py and the tuning cache
key builder in pipeline/tuning_cache.py.
"""
import importlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import run_backtest  # noqa: E402
from pipeline import tuning_cache  # noqa: E402
from pipeline import config as pconfig  # noqa: E402


def test_stints_cache_key_changes_with_preprocessing_schema_version(monkeypatch, tmp_path):
    src = tmp_path / "pbp.csv"
    src.write_text("a,b\n1,2\n")
    paths = [(2025, src)]

    key_before = run_backtest._stints_cache_key(paths, False, True, 0.76)
    monkeypatch.setattr(pconfig, "PREPROCESSING_SCHEMA_VERSION", pconfig.PREPROCESSING_SCHEMA_VERSION + 1)
    key_after = run_backtest._stints_cache_key(paths, False, True, 0.76)
    assert key_before != key_after


def test_stints_cache_key_changes_with_validation_schema_version(monkeypatch, tmp_path):
    src = tmp_path / "pbp.csv"
    src.write_text("a,b\n1,2\n")
    paths = [(2025, src)]

    key_before = run_backtest._stints_cache_key(paths, False, True, 0.76)
    monkeypatch.setattr(pconfig, "VALIDATION_SCHEMA_VERSION", pconfig.VALIDATION_SCHEMA_VERSION + 1)
    key_after = run_backtest._stints_cache_key(paths, False, True, 0.76)
    assert key_before != key_after


def test_tuning_cache_key_changes_with_feature_schema_version(monkeypatch):
    key_before = tuning_cache._seasons_hash([2023, 2024])
    monkeypatch.setattr(pconfig, "FEATURE_SCHEMA_VERSION", pconfig.FEATURE_SCHEMA_VERSION + 1)
    monkeypatch.setattr(tuning_cache, "FEATURE_SCHEMA_VERSION", pconfig.FEATURE_SCHEMA_VERSION)
    key_after = tuning_cache._seasons_hash([2023, 2024])
    assert key_before != key_after


def test_tuning_cache_key_changes_with_market_snapshot_schema_version(monkeypatch):
    key_before = tuning_cache._seasons_hash([2023, 2024])
    monkeypatch.setattr(pconfig, "MARKET_SNAPSHOT_SCHEMA_VERSION", pconfig.MARKET_SNAPSHOT_SCHEMA_VERSION + 1)
    monkeypatch.setattr(tuning_cache, "MARKET_SNAPSHOT_SCHEMA_VERSION", pconfig.MARKET_SNAPSHOT_SCHEMA_VERSION)
    key_after = tuning_cache._seasons_hash([2023, 2024])
    assert key_before != key_after


def test_tuning_cache_key_stable_for_same_versions():
    k1 = tuning_cache._seasons_hash([2023, 2024])
    k2 = tuning_cache._seasons_hash([2023, 2024])
    assert k1 == k2
