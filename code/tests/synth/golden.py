"""Golden-master helpers for Epic 10.5 (stub — snapshots land with 10.5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

GOLDEN_DIR = Path(__file__).resolve().parent / "golden_snapshots"
DEFAULT_SEED = 20260919


def golden_path(name: str) -> Path:
    return GOLDEN_DIR / f"{name}.json"


def write_golden(name: str, payload: dict[str, Any], *, schema_version: int) -> Path:
    GOLDEN_DIR.mkdir(parents=True, exist_ok=True)
    path = golden_path(name)
    body = {"FEATURE_SCHEMA_VERSION": int(schema_version), "payload": payload}
    path.write_text(json.dumps(body, indent=2, sort_keys=True, default=str))
    return path


def load_golden(name: str) -> dict[str, Any]:
    return json.loads(golden_path(name).read_text())
