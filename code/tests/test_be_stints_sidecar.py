"""Stints sidecar fingerprint: reuse when unchanged; refuse mismatched --stints-cache."""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from pipeline.stint_loader import (
    StintsCacheMismatchError,
    build_stints_meta,
    data_folder_changed,
    describe_stints_cache,
    read_stints_sidecar,
    sidecar_path,
    stints_cache_key,
    write_stints_sidecar,
)


def _touch_csv(path: Path, n: int = 3) -> None:
    path.write_text("a,b\n" + "\n".join(f"{i},x" for i in range(n)))


def test_stints_cache_key_changes_when_mtime_or_size_changes(tmp_path):
    f = tmp_path / "events_2021_22_pbp_V3.csv"
    _touch_csv(f, 3)
    paths = [(2021, f)]
    k1 = stints_cache_key(paths, quick=False, walkforward_zone_pps=True, assist_split=0.76)
    _touch_csv(f, 10)  # size change
    k2 = stints_cache_key(paths, quick=False, walkforward_zone_pps=True, assist_split=0.76)
    assert k1 != k2


def test_sidecar_roundtrip_and_data_folder_changed(tmp_path):
    f = tmp_path / "events_2021_22_pbp_V3.csv"
    _touch_csv(f)
    paths = [(2021, f)]
    key = stints_cache_key(paths, quick=False, walkforward_zone_pps=True, assist_split=0.76)
    cache = tmp_path / f"stints_cache_{key}.pkl"
    # minimal pickle
    pd.DataFrame({"x": [1]}).to_pickle(cache)
    meta = build_stints_meta(
        cache_key=key,
        paths=paths,
        quick=False,
        walkforward_zone_pps=True,
        assist_split=0.76,
        data_dir=str(tmp_path),
        n_rows=1,
        seasons=[2022],
        cache_path=str(cache),
    )
    side = write_stints_sidecar(cache, meta)
    assert side.exists()
    loaded = read_stints_sidecar(cache)
    assert loaded["cache_key"] == key
    assert loaded["built_at"]
    assert not data_folder_changed(loaded, key)
    assert data_folder_changed(loaded, "deadbeefdeadbeef")
    text = describe_stints_cache(cache, current_key=key)
    assert "built_at" in text
    assert "data change: no" in text


def test_load_stints_refuses_mismatched_explicit_cache(tmp_path, monkeypatch):
    from pipeline import stint_loader as sl

    f = tmp_path / "events_2021_22_pbp_V3.csv"
    _touch_csv(f)
    paths = [(2021, f)]
    key = stints_cache_key(paths, quick=False, walkforward_zone_pps=True, assist_split=0.76)
    # Cache with WRONG key in sidecar
    bad = tmp_path / "stints_cache_oldkeyoldkey01.pkl"
    pd.DataFrame({"x": [1]}).to_pickle(bad)
    write_stints_sidecar(bad, {
        "built_at": "2020-01-01T00:00:00+00:00",
        "cache_key": "oldkeyoldkey01",
        "paths_used": [],
        "n_rows": 1,
    })

    with pytest.raises(StintsCacheMismatchError):
        sl.load_stints(
            paths=paths,
            stints_cache=bad,
            stints_mode="use",
            use_cache=True,
            force_stints_cache=False,
        )

    # force override allowed
    df, meta = sl.load_stints(
        paths=paths,
        stints_cache=bad,
        stints_mode="use",
        use_cache=True,
        force_stints_cache=True,
    )
    assert meta["cache_hit"] is True
    assert len(df) == 1
