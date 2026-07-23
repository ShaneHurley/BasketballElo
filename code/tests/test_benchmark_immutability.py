"""Task 057 — lock the corrected latest benchmark/config once integrity is
restored; live outcomes must never retroactively alter already-issued
predictions.
"""
import pytest

from pipeline.benchmark_lock import (
    BenchmarkImmutabilityError,
    assert_benchmark_unchanged,
    fingerprint_predictions,
    lock_benchmark,
)


class TestLockBenchmark:
    def test_first_lock_succeeds(self, tmp_path):
        manifest = {"model_version": "v1", "cutoff": "T-60"}
        path = lock_benchmark("season_2025_26", manifest, tmp_path, results_fingerprint="abc123")
        assert path.exists()

    def test_relocking_with_identical_content_is_idempotent(self, tmp_path):
        manifest = {"model_version": "v1", "cutoff": "T-60"}
        p1 = lock_benchmark("season_2025_26", manifest, tmp_path, results_fingerprint="abc123")
        p2 = lock_benchmark("season_2025_26", manifest, tmp_path, results_fingerprint="abc123")
        assert p1 == p2

    def test_relocking_with_different_manifest_raises(self, tmp_path):
        lock_benchmark("season_2025_26", {"model_version": "v1"}, tmp_path, results_fingerprint="abc123")
        with pytest.raises(BenchmarkImmutabilityError):
            lock_benchmark("season_2025_26", {"model_version": "v2"}, tmp_path, results_fingerprint="abc123")

    def test_relocking_with_different_results_fingerprint_raises(self):
        """Regression test for Rule 9: live outcomes arriving later must
        never retroactively change an already-issued prediction/benchmark
        record — a different results fingerprint under the same benchmark
        name is exactly that attempt."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            lock_benchmark("season_2025_26", {"model_version": "v1"}, d, results_fingerprint="abc123")
            with pytest.raises(BenchmarkImmutabilityError):
                lock_benchmark("season_2025_26", {"model_version": "v1"}, d, results_fingerprint="xyz999")


class TestAssertBenchmarkUnchanged:
    def test_unchanged_lock_passes(self, tmp_path):
        lock_benchmark("season_2025_26", {"model_version": "v1"}, tmp_path, results_fingerprint="abc123")
        record = assert_benchmark_unchanged("season_2025_26", tmp_path)
        assert record["manifest"]["model_version"] == "v1"

    def test_missing_lock_raises(self, tmp_path):
        with pytest.raises(BenchmarkImmutabilityError):
            assert_benchmark_unchanged("does_not_exist", tmp_path)

    def test_tampered_lock_file_detected(self, tmp_path):
        lock_benchmark("season_2025_26", {"model_version": "v1"}, tmp_path, results_fingerprint="abc123")
        path = tmp_path / "season_2025_26.lock.json"
        import json
        record = json.loads(path.read_text())
        record["manifest"]["model_version"] = "v2-tampered"
        path.write_text(json.dumps(record))
        with pytest.raises(BenchmarkImmutabilityError):
            assert_benchmark_unchanged("season_2025_26", tmp_path)


class TestFingerprintPredictions:
    def test_deterministic(self):
        fp1 = fingerprint_predictions(["g1", "g2"], [1.234567, -2.5])
        fp2 = fingerprint_predictions(["g1", "g2"], [1.234567, -2.5])
        assert fp1 == fp2

    def test_changes_when_a_prediction_value_changes(self):
        fp1 = fingerprint_predictions(["g1", "g2"], [1.0, -2.5])
        fp2 = fingerprint_predictions(["g1", "g2"], [1.1, -2.5])
        assert fp1 != fp2
