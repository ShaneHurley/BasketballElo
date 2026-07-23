"""Cache tuned hyperparameters per training-season window."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from pipeline.config import (
    FEATURE_SCHEMA_VERSION,
    MARKET_SNAPSHOT_SCHEMA_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
    STATE_DIR,
    VALIDATION_SCHEMA_VERSION,
)

CACHE_DIR = STATE_DIR / "tuning_cache"
# Bump when tuning/feature logic changes so stale caches are not reused.
CACHE_VERSION = "2026-06-21-v2"


def _seasons_hash(train_seasons, stints_df=None) -> str:
    # Task 004: fold in the preprocessing/feature/market-snapshot/validation
    # schema versions so any change to those (score/date repair, stint
    # semantics, market snapshot logic, CV logic) produces a different key
    # instead of reusing a tuning result computed under different semantics.
    h = hashlib.md5()
    h.update(CACHE_VERSION.encode())
    h.update(
        f"prep={PREPROCESSING_SCHEMA_VERSION};feat={FEATURE_SCHEMA_VERSION};"
        f"mkt={MARKET_SNAPSHOT_SCHEMA_VERSION};val={VALIDATION_SCHEMA_VERSION}".encode()
    )
    h.update(",".join(str(s) for s in sorted(train_seasons)).encode())
    if stints_df is not None and not stints_df.empty:
        h.update(str(len(stints_df)).encode())
        if "GAME_ID" in stints_df.columns:
            h.update(str(stints_df["GAME_ID"].nunique()).encode())
    return h.hexdigest()[:16]


def cache_path(kind: str, train_seasons, stints_df=None) -> Path:
    key = _seasons_hash(train_seasons, stints_df)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"{kind}_{key}.json"


def load_cached(kind: str, train_seasons, stints_df=None):
    path = cache_path(kind, train_seasons, stints_df)
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def save_cached(kind: str, train_seasons, params: dict, stints_df=None):
    path = cache_path(kind, train_seasons, stints_df)
    path.write_text(json.dumps(params, indent=2))
