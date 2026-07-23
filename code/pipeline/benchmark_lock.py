"""Locked benchmark / artifact immutability (Task 057).

Once integrity is restored, the corrected latest benchmark and its config
must be locked: later live outcomes (or a re-run with different code/data)
must never retroactively alter an already-issued prediction or a
already-published benchmark record. This module provides a minimal,
dependency-free content-hash lock so that any attempt to silently
overwrite a locked benchmark raises instead of quietly rewriting history.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


class BenchmarkImmutabilityError(RuntimeError):
    """Raised when code attempts to silently overwrite a locked benchmark."""


def _stable_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()


def lock_benchmark(name: str, manifest: dict[str, Any], directory: Path | str, *, results_fingerprint: str | None = None) -> Path:
    """Write (or verify) an immutable lock file for a named benchmark.

    If no lock exists yet, one is created. If a lock already exists, the
    new manifest/fingerprint must hash-match the stored one exactly —
    otherwise this raises ``BenchmarkImmutabilityError`` rather than
    overwriting the file (Rule 9/Task 057: a locked prediction/benchmark
    must never be silently rewritten, even by a "corrected" re-run — a
    genuine correction must explicitly version-bump ``name``).
    """
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}.lock.json"
    payload = {"manifest": manifest, "results_fingerprint": results_fingerprint}
    new_hash = _stable_hash(payload)

    if path.exists():
        existing = json.loads(path.read_text())
        if existing.get("content_hash") != new_hash:
            raise BenchmarkImmutabilityError(
                f"Benchmark {name!r} is already locked at {path} with a "
                "different manifest/results fingerprint. Live outcomes or "
                "a re-run must never retroactively alter an issued "
                "benchmark — bump the benchmark name/version instead of "
                "overwriting this lock."
            )
        return path

    path.write_text(json.dumps({"content_hash": new_hash, **payload}, indent=2, default=str, sort_keys=True))
    return path


def assert_benchmark_unchanged(name: str, directory: Path | str) -> dict:
    """Load a locked benchmark and verify its stored hash matches its own
    recorded content (detects hand-edited/corrupted lock files)."""
    directory = Path(directory)
    path = directory / f"{name}.lock.json"
    if not path.exists():
        raise BenchmarkImmutabilityError(f"No locked benchmark named {name!r} at {path}")
    record = json.loads(path.read_text())
    stored_hash = record.get("content_hash")
    recomputed = _stable_hash({
        "manifest": record.get("manifest"),
        "results_fingerprint": record.get("results_fingerprint"),
    })
    if stored_hash != recomputed:
        raise BenchmarkImmutabilityError(
            f"Locked benchmark {name!r} at {path} has been tampered with: "
            f"stored hash {stored_hash!r} != recomputed hash {recomputed!r}."
        )
    return record


def fingerprint_predictions(prediction_ids, predicted_values) -> str:
    """Deterministic fingerprint of an issued prediction set, for pinning
    against future silent mutation (Rule 9: T-60 predictions immutable)."""
    payload = {
        "ids": [str(x) for x in prediction_ids],
        "values": [round(float(v), 6) if v is not None else None for v in predicted_values],
    }
    return _stable_hash(payload)
