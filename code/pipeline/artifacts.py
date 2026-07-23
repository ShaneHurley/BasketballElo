"""Backtest artifact schema validation, provenance, and config snapshots.

Task 003 expands the run manifest beyond the two flags previously checked so
that loading an artifact built under a different configuration, schema
version, source dataset, feature set, model version, or random seed is
rejected instead of silently reused.
"""
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd

from pipeline.config import (
    ARTIFACT_SCHEMA_VERSION,
    BET_SELECTION_MODE,
    CONFIDENCE_SELECTION_MODE,
    DECISION_CUTOFF_MINUTES_BEFORE_TIP,
    FEATURE_SCHEMA_VERSION,
    MARKET_SNAPSHOT_SCHEMA_VERSION,
    MAX_CONFIDENCE_SCORE,
    MIN_CONFIDENCE_SCORE,
    MIN_ML_WIN_PCT,
    PREPROCESSING_SCHEMA_VERSION,
    REQUIRED_BACKTEST_COLUMNS,
    STATE_DIR,
    VALIDATION_SCHEMA_VERSION,
)

__all__ = [
    "ARTIFACT_SCHEMA_VERSION",
    "REQUIRED_BACKTEST_COLUMNS",
    "SCHEMA_VERSION_KEYS",
    "backtest_config_snapshot",
    "full_config_snapshot",
    "hash_source_file",
    "git_revision",
    "build_run_manifest",
    "validate_backtest_df",
    "validate_run_manifest",
    "is_backtest_stale",
    "ArtifactMismatchError",
    "assert_manifest_compatible",
    "save_backtest_metadata",
    "load_backtest_metadata",
]


# Fields whose mismatch is always fatal (never a soft warning), because a
# mismatch means the artifact was built under different data/label/feature
# semantics and comparing it to current results would be meaningless.
SCHEMA_VERSION_KEYS = (
    "artifact_schema_version",
    "preprocessing_schema_version",
    "feature_schema_version",
    "market_snapshot_schema_version",
    "validation_schema_version",
)


def backtest_config_snapshot() -> dict[str, Any]:
    """Serializable config flags stored alongside backtest outputs."""
    return {
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "market_snapshot_schema_version": MARKET_SNAPSHOT_SCHEMA_VERSION,
        "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        "BET_SELECTION_MODE": BET_SELECTION_MODE,
        "CONFIDENCE_SELECTION_MODE": CONFIDENCE_SELECTION_MODE,
        "MIN_CONFIDENCE_SCORE": MIN_CONFIDENCE_SCORE,
        "MAX_CONFIDENCE_SCORE": MAX_CONFIDENCE_SCORE,
        "MIN_ML_WIN_PCT": MIN_ML_WIN_PCT,
        "decision_cutoff_minutes_before_tip": DECISION_CUTOFF_MINUTES_BEFORE_TIP,
    }


def full_config_snapshot() -> dict[str, Any]:
    """Every JSON-serializable UPPER_CASE constant in pipeline.config.

    Used for full-provenance artifact manifests (Task 003); intentionally
    broader than `backtest_config_snapshot`, which only carries the flags
    needed for the lighter-weight `validate_backtest_df` compatibility check.
    """
    from pipeline import config as _cfg

    out: dict[str, Any] = {}
    for name in dir(_cfg):
        if not name.isupper():
            continue
        value = getattr(_cfg, name)
        try:
            json.dumps(value)
        except TypeError:
            continue
        out[name] = value
    return out


def hash_source_file(path: Path | str, *, fast: bool = True, chunk_size: int = 1 << 20) -> str | None:
    """Return a content-derived fingerprint for a source data file.

    ``fast=True`` (default) hashes ``(name, size, mtime_ns)`` rather than the
    full file body — sufficient to detect any change to large multi-hundred-MB
    raw PBP/odds CSVs without re-reading them on every run. Pass
    ``fast=False`` for a true SHA-256 content hash when the file is small
    enough (e.g. config snapshots, small fixtures) that a mismatch must be
    proven byte-for-byte.
    """
    p = Path(path)
    if not p.exists():
        return None
    stat = p.stat()
    if fast:
        h = hashlib.sha256()
        h.update(f"{p.name}:{stat.st_size}:{stat.st_mtime_ns}".encode())
        return h.hexdigest()[:24]
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def git_revision() -> str | None:
    """Best-effort short git SHA; None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, timeout=5,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        if out.returncode == 0:
            return out.stdout.strip()
    except Exception:
        pass
    return None


def build_run_manifest(
    *,
    source_paths: dict[str, Path | str] | None = None,
    game_count: int | None = None,
    quote_count: int | None = None,
    train_range: tuple[str, str] | None = None,
    calibration_range: tuple[str, str] | None = None,
    test_range: tuple[str, str] | None = None,
    feature_list: list[str] | None = None,
    model_version: str | None = None,
    seeds: dict[str, int] | None = None,
    config_overrides: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a full-provenance manifest for one backtest/training run.

    Covers: config snapshot, source file hashes, schema versions, cutoff
    definition, train/calibration/test date ranges, feature list, model
    version, random seeds, and git revision when available.
    """
    source_paths = source_paths or {}
    manifest: dict[str, Any] = {
        "config_snapshot": full_config_snapshot(),
        "config_overrides": config_overrides or {},
        "source_hashes": {
            name: hash_source_file(path) for name, path in source_paths.items()
        },
        "schema_versions": {
            "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
            "preprocessing_schema_version": PREPROCESSING_SCHEMA_VERSION,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "market_snapshot_schema_version": MARKET_SNAPSHOT_SCHEMA_VERSION,
            "validation_schema_version": VALIDATION_SCHEMA_VERSION,
        },
        "cutoff_definition": {
            "minutes_before_scheduled_tip": DECISION_CUTOFF_MINUTES_BEFORE_TIP,
        },
        "game_count": game_count,
        "quote_count": quote_count,
        "date_ranges": {
            "train": list(train_range) if train_range else None,
            "calibration": list(calibration_range) if calibration_range else None,
            "test": list(test_range) if test_range else None,
        },
        "feature_list": sorted(feature_list) if feature_list else None,
        "model_version": model_version,
        "seeds": seeds or {},
        "git_revision": git_revision(),
        "created_at": pd.Timestamp.utcnow().isoformat(),
    }
    if extra:
        manifest.update(extra)
    return manifest


class ArtifactMismatchError(RuntimeError):
    """Raised when a saved artifact's provenance manifest does not match the
    manifest expected by the currently running code."""


def validate_run_manifest(saved: dict[str, Any], expected: dict[str, Any]) -> list[str]:
    """Return mismatch reasons between a saved manifest and the expected one.

    Every schema-version field is compared strictly. Source hashes, feature
    list, model version, and seeds are compared when both sides provide them.
    """
    reasons: list[str] = []

    saved_schema = saved.get("schema_versions", {})
    expected_schema = expected.get("schema_versions", {})
    for key in SCHEMA_VERSION_KEYS:
        sv = saved_schema.get(key)
        ev = expected_schema.get(key)
        if sv is not None and ev is not None and sv != ev:
            reasons.append(f"{key} mismatch: saved={sv!r} current={ev!r}")

    saved_hashes = saved.get("source_hashes", {}) or {}
    expected_hashes = expected.get("source_hashes", {}) or {}
    for name, ehash in expected_hashes.items():
        shash = saved_hashes.get(name)
        if shash is not None and ehash is not None and shash != ehash:
            reasons.append(f"source hash mismatch for {name!r}: saved={shash!r} current={ehash!r}")

    if expected.get("feature_list") and saved.get("feature_list"):
        if sorted(expected["feature_list"]) != sorted(saved["feature_list"]):
            reasons.append("feature_list mismatch between saved artifact and current run")

    if expected.get("model_version") and saved.get("model_version"):
        if expected["model_version"] != saved["model_version"]:
            reasons.append(
                f"model_version mismatch: saved={saved['model_version']!r} "
                f"current={expected['model_version']!r}"
            )

    saved_cutoff = saved.get("cutoff_definition", {}).get("minutes_before_scheduled_tip")
    expected_cutoff = expected.get("cutoff_definition", {}).get("minutes_before_scheduled_tip")
    if saved_cutoff is not None and expected_cutoff is not None and saved_cutoff != expected_cutoff:
        reasons.append(f"cutoff mismatch: saved={saved_cutoff!r} current={expected_cutoff!r}")

    return reasons


def assert_manifest_compatible(saved: dict[str, Any], expected: dict[str, Any]) -> None:
    """Raise ArtifactMismatchError if `saved` is incompatible with `expected`."""
    reasons = validate_run_manifest(saved, expected)
    if reasons:
        raise ArtifactMismatchError(
            "Artifact manifest incompatible with current run:\n  - " + "\n  - ".join(reasons)
        )


def validate_backtest_df(
    df: pd.DataFrame | None,
    *,
    expected: dict[str, Any] | None = None,
) -> list[str]:
    """Return human-readable reasons the backtest CSV is incompatible with current code."""
    reasons: list[str] = []
    if df is None or df.empty:
        reasons.append("backtest results empty")
        return reasons

    missing = [c for c in REQUIRED_BACKTEST_COLUMNS if c not in df.columns]
    if missing:
        reasons.append(f"missing columns: {missing}")

    expected = expected or backtest_config_snapshot()
    meta_path = default_metadata_path()
    if meta_path.exists():
        try:
            saved = json.loads(meta_path.read_text())
            for key in ("BET_SELECTION_MODE",) + SCHEMA_VERSION_KEYS:
                if key in saved and key in expected and saved[key] != expected[key]:
                    reasons.append(f"{key} mismatch: saved={saved[key]!r} current={expected[key]!r}")
        except (json.JSONDecodeError, OSError):
            reasons.append("backtest_metadata.json unreadable")

    if BET_SELECTION_MODE == "confidence_only" and "ACTIONABLE" not in df.columns:
        reasons.append("confidence_only requires ACTIONABLE column — re-run Phase 2")

    return reasons


def is_backtest_stale(df: pd.DataFrame | None, *, expected: dict[str, Any] | None = None) -> bool:
    return bool(validate_backtest_df(df, expected=expected))


def default_metadata_path() -> Path:
    return STATE_DIR / "backtest_metadata.json"


def save_backtest_metadata(path: Path | None = None, extra: dict[str, Any] | None = None) -> Path:
    path = path or default_metadata_path()
    payload = backtest_config_snapshot()
    if extra:
        payload.update(extra)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def load_backtest_metadata(path: Path | None = None) -> dict[str, Any]:
    path = path or default_metadata_path()
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError):
        return {}
