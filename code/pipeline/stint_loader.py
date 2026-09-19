"""Shared PBP → stints loader with cache, sidecar provenance, and fingerprint checks."""
from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import pandas as pd

from pipeline.config import (
    ASSIST_SPLIT,
    FEATURE_SCHEMA_VERSION,
    MARKET_SNAPSHOT_SCHEMA_VERSION,
    PREPROCESSING_SCHEMA_VERSION,
    STATE_DIR,
    VALIDATION_SCHEMA_VERSION,
)
from pipeline.dates import attach_real_dates, load_odds_schedule
from pipeline.game_results import attach_canonical_finals, build_canonical_games
from pipeline.ingest import convert_new_pbp, convert_v3_pbp
from pipeline.preprocess import compute_zone_calibration, preprocess_pbp
from pipeline.stints import build_stints


class StintsCacheMismatchError(ValueError):
    """Raised when an explicit stints cache no longer matches current data sources."""


def read_csv_fast(path):
    """Read a (potentially huge) CSV as fast as available."""
    try:
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, low_memory=False)


def _path_fingerprints(paths: Sequence[tuple[int, Path | str]]) -> list[dict]:
    rows = []
    for season, path in sorted(paths, key=lambda kv: kv[0]):
        p = Path(path)
        row = {"season": int(season), "name": p.name, "path": str(p)}
        if p.exists():
            stt = p.stat()
            row["mtime"] = int(stt.st_mtime)
            row["size"] = int(stt.st_size)
        else:
            row["mtime"] = None
            row["size"] = None
        rows.append(row)
    return rows


def stints_cache_key(
    paths: Sequence[tuple[int, Path | str]],
    *,
    quick: bool,
    walkforward_zone_pps: bool,
    assist_split: float,
) -> str:
    """Signature over source files (mtime+size) + params + schema versions."""
    h = hashlib.md5()
    for season, path in sorted(paths, key=lambda kv: kv[0]):
        p = Path(path)
        if p.exists():
            stt = p.stat()
            h.update(f"{season}:{p.name}:{int(stt.st_mtime)}:{stt.st_size}".encode())
    h.update(f"q={quick};wzpps={walkforward_zone_pps};as={assist_split};zones=v2".encode())
    h.update(
        f"prep={PREPROCESSING_SCHEMA_VERSION};feat={FEATURE_SCHEMA_VERSION};"
        f"mkt={MARKET_SNAPSHOT_SCHEMA_VERSION};val={VALIDATION_SCHEMA_VERSION}".encode()
    )
    return h.hexdigest()[:16]


def sidecar_path(cache_path: Path | str) -> Path:
    p = Path(cache_path)
    return p.with_suffix(p.suffix + ".meta.json") if p.suffix else Path(str(p) + ".meta.json")


def build_stints_meta(
    *,
    cache_key: str,
    paths: Sequence[tuple[int, Path | str]],
    quick: bool,
    walkforward_zone_pps: bool,
    assist_split: float,
    data_dir: str | None = None,
    n_rows: int | None = None,
    seasons: list | None = None,
    cache_path: str | None = None,
) -> dict:
    """Provenance payload written beside ``stints_cache_*.pkl``."""
    return {
        "built_at": datetime.now(timezone.utc).isoformat(),
        "cache_key": cache_key,
        "data_dir": data_dir,
        "cache_path": cache_path,
        "paths_used": _path_fingerprints(paths),
        "schema_versions": {
            "preprocessing": PREPROCESSING_SCHEMA_VERSION,
            "feature": FEATURE_SCHEMA_VERSION,
            "market_snapshot": MARKET_SNAPSHOT_SCHEMA_VERSION,
            "validation": VALIDATION_SCHEMA_VERSION,
        },
        "params": {
            "quick": bool(quick),
            "walkforward_zone_pps": bool(walkforward_zone_pps),
            "assist_split": float(assist_split),
            "zones": "v2",
        },
        "n_rows": n_rows,
        "seasons": seasons,
    }


def write_stints_sidecar(cache_path: Path | str, meta: dict) -> Path:
    side = sidecar_path(cache_path)
    side.parent.mkdir(parents=True, exist_ok=True)
    side.write_text(json.dumps(meta, indent=2, default=str))
    return side


def read_stints_sidecar(cache_path: Path | str) -> dict | None:
    side = sidecar_path(cache_path)
    if not side.exists():
        return None
    try:
        return json.loads(side.read_text())
    except Exception:
        return None


def data_folder_changed(sidecar: dict | None, current_key: str) -> bool:
    if not sidecar:
        return True
    return str(sidecar.get("cache_key", "")) != str(current_key)


def resolve_paths_for_stints(
    paths: Sequence[tuple[int, Path | str]] | None = None,
) -> tuple[list[tuple[int, Path]], str | None]:
    """Resolve (season, path) list and optional data_dir string."""
    from pipeline.config import PBP_2026_PATH, V3_DATA_PATHS

    data_dir: str | None = None
    if paths is None:
        paths_all: list[tuple[int, Path]] = [
            (k, Path(v)) for k, v in V3_DATA_PATHS.items() if Path(v).exists()
        ]
        if PBP_2026_PATH.exists():
            paths_all.append((2025, Path(PBP_2026_PATH)))
        if not paths_all or len(paths_all) <= 1:
            try:
                from pipeline.data_paths import discover_pbp_paths, resolve_data_dir

                dd = resolve_data_dir(None)
                data_dir = str(dd)
                discovered = discover_pbp_paths(dd)
                if discovered:
                    paths_all = [(int(y), Path(p)) for y, p in discovered.items()]
            except Exception:
                pass
        elif paths_all:
            data_dir = str(Path(paths_all[0][1]).parent)
    else:
        paths_all = [(int(s), Path(p)) for s, p in paths]
        if paths_all:
            data_dir = str(Path(paths_all[0][1]).parent)
    return paths_all, data_dir


def describe_stints_cache(
    cache_path: Path | str,
    *,
    current_key: str | None = None,
) -> str:
    """Human-readable status for CLI / suite stage 1."""
    cp = Path(cache_path)
    side = read_stints_sidecar(cp)
    lines = [f"stints cache: {cp}"]
    if not cp.exists():
        lines.append("  status: MISSING — will rebuild")
        return "\n".join(lines)
    lines.append(f"  pickle: exists ({cp.stat().st_size:,} bytes)")
    if side:
        lines.append(f"  built_at: {side.get('built_at', '?')}")
        lines.append(f"  cache_key: {side.get('cache_key', '?')}")
        lines.append(f"  data_dir: {side.get('data_dir', '?')}")
        lines.append(f"  n_rows: {side.get('n_rows', '?')}  seasons: {side.get('seasons', '?')}")
        if current_key is not None:
            if data_folder_changed(side, current_key):
                lines.append(
                    f"  data change: YES (sidecar key {side.get('cache_key')} "
                    f"!= current {current_key}) — recommend rebuild"
                )
            else:
                lines.append("  data change: no — safe to reuse")
    else:
        lines.append("  sidecar: MISSING — cannot verify data fingerprint; prefer rebuild")
        if current_key is not None:
            # Infer key from filename stints_cache_{key}.pkl
            name = cp.name
            if name.startswith("stints_cache_") and name.endswith(".pkl"):
                file_key = name[len("stints_cache_"):-len(".pkl")]
                if file_key != current_key:
                    lines.append(
                        f"  filename key {file_key} != current {current_key} — rebuild"
                    )
    return "\n".join(lines)


def load_stints(
    *,
    paths: Sequence[tuple[int, Path | str]] | None = None,
    quick: bool = False,
    walkforward_zone_pps: bool = True,
    use_cache: bool = True,
    cache_dir: Path | str | None = None,
    stints_cache: Path | str | None = None,
    odds_path: Path | str | None = None,
    name_to_id: dict | None = None,
    assist_split: float | None = None,
    force_stints_cache: bool = False,
    stints_mode: str = "auto",
) -> tuple[pd.DataFrame, dict]:
    """Load and stitch historical stint data from PBP files.

    Parameters
    ----------
    paths
        Explicit ``(season_start_year, path)`` pairs. When None, falls back to
        ``pipeline.config.V3_DATA_PATHS`` (+ 2026 combined if present).
    cache_dir
        Directory for writing ``stints_cache_*.pkl`` (default STATE_DIR).
    stints_cache
        Explicit pickle path. Validated against current data fingerprint unless
        ``force_stints_cache=True``.
    stints_mode
        ``auto`` (reuse if key matches), ``rebuild`` (always rebuild),
        ``use`` (require ``stints_cache`` and matching fingerprint).
    force_stints_cache
        If True, load ``stints_cache`` even when fingerprint mismatches.
    odds_path
        Schedule source for V3 date recovery.
    name_to_id
        Player name→id map for combined-stats conversion.
    assist_split
        Override ASSIST_SPLIT for cache key / build_stints.

    Returns
    -------
    (stints_df, meta) where meta includes cache_hit, cache_path, built_at, paths_used.
    """
    from pipeline.config import (
        MODERN_ODDS_PATH,
        name_to_id as cfg_name_to_id,
    )

    assist = float(ASSIST_SPLIT if assist_split is None else assist_split)
    n2i = name_to_id if name_to_id is not None else cfg_name_to_id
    mode = (stints_mode or "auto").lower().strip()
    if mode not in ("auto", "rebuild", "use"):
        raise ValueError(f"stints_mode must be auto|rebuild|use, got {stints_mode!r}")

    paths_all, data_dir = resolve_paths_for_stints(paths)
    meta: dict = {
        "cache_hit": False,
        "cache_path": None,
        "paths_used": [(s, str(p)) for s, p in paths_all],
        "data_dir": data_dir,
        "built_at": None,
        "cache_key": None,
        "stints_mode": mode,
    }

    current_key = stints_cache_key(
        paths_all, quick=quick, walkforward_zone_pps=walkforward_zone_pps, assist_split=assist,
    )
    meta["cache_key"] = current_key
    out_dir = Path(cache_dir) if cache_dir is not None else Path(STATE_DIR)
    default_cache = out_dir / f"stints_cache_{current_key}.pkl"

    def _load_pickle(cp: Path, *, reason: str) -> pd.DataFrame:
        print(f"  ⚡ loading stints from {cp} ({reason}) ...")
        cached = pd.read_pickle(cp)
        side = read_stints_sidecar(cp)
        meta["cache_hit"] = True
        meta["cache_path"] = str(cp)
        meta["built_at"] = (side or {}).get("built_at")
        meta["sidecar"] = side
        print(describe_stints_cache(cp, current_key=current_key))
        print(f"  ✅ cache hit: {len(cached):,} stint rows")
        return cached

    # Explicit path / use mode
    if stints_cache is not None or mode == "use":
        if stints_cache is None:
            raise ValueError("stints_mode=use requires stints_cache=PATH")
        cp = Path(stints_cache)
        if not cp.exists():
            if mode == "use":
                raise FileNotFoundError(f"stints_mode=use but cache missing: {cp}")
            print(f"  ⚠️ --stints-cache missing ({cp}); rebuilding from source")
        else:
            side = read_stints_sidecar(cp)
            mismatched = data_folder_changed(side, current_key)
            # Also mismatch if filename key differs from current
            if cp.name.startswith("stints_cache_") and cp.name.endswith(".pkl"):
                file_key = cp.name[len("stints_cache_"):-len(".pkl")]
                if file_key and file_key != current_key:
                    mismatched = True
            if mismatched and not force_stints_cache:
                raise StintsCacheMismatchError(
                    f"Stints cache fingerprint mismatch for {cp}.\n"
                    f"  sidecar/file key != current data key {current_key}.\n"
                    f"  Data folder may have changed. Pass force_stints_cache=True "
                    f"or --force-stints-cache to override, or use --stints rebuild.\n"
                    + describe_stints_cache(cp, current_key=current_key)
                )
            if mismatched and force_stints_cache:
                print("  ⚠️ force_stints_cache: loading mismatched cache anyway")
            return _load_pickle(cp, reason="explicit path"), meta

    if mode == "rebuild":
        use_cache_read = False
    else:
        use_cache_read = use_cache

    cache_path = default_cache if use_cache else None
    meta["cache_path"] = str(cache_path) if cache_path else None

    if use_cache_read and cache_path is not None and cache_path.exists():
        try:
            return _load_pickle(cache_path, reason=f"auto key={current_key}"), meta
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ cache read failed ({e}); rebuilding from source")

    if cache_path is not None:
        print(describe_stints_cache(cache_path, current_key=current_key))

    # Schedule for V3 date recovery
    if odds_path is None:
        odds_path = MODERN_ODDS_PATH if Path(MODERN_ODDS_PATH).exists() else None
        if odds_path is None:
            root = Path(__file__).resolve().parent.parent
            alt = root / "all_odds.csv"
            odds_path = alt if alt.exists() else None

    sched = None
    try:
        if odds_path is not None and Path(odds_path).exists():
            sched = load_odds_schedule(str(odds_path))
            print(
                f"  loaded schedule for date recovery: {len(sched)} games "
                f"({sched['date'].min().date()} -> {sched['date'].max().date()})"
            )
    except Exception as e:  # noqa: BLE001
        print(f"  ⚠️ could not load schedule for date recovery: {e}")
        sched = None

    frames = []
    prev_zone_calib = None
    for season, path in sorted(paths_all, key=lambda kv: kv[0]):
        if not Path(path).exists():
            print(f"  skip missing {path}")
            continue
        print(f"  loading {path} ...")
        raw = read_csv_fast(path)
        canonical_games = None
        if season >= 2025:
            try:
                canonical_games = build_canonical_games(raw)
            except ValueError as e:
                print(f"  ⚠️ could not build canonical games for {path}: {e}")
                canonical_games = None
            df = convert_new_pbp(raw, name_to_id=n2i)
        else:
            df = convert_v3_pbp(raw)
        df = preprocess_pbp(
            df,
            compute_xpoints=True,
            zone_calib=prev_zone_calib if walkforward_zone_pps else None,
        )
        if walkforward_zone_pps:
            prev_zone_calib = compute_zone_calibration(df)
        st = build_stints(df, assist_split=assist)
        st["season"] = season + 1  # end-year label (2021 file → season 2022)
        if canonical_games is not None:
            st = attach_canonical_finals(st, canonical_games)

        if season < 2025 and sched is not None and not st.empty:
            n_dates = (
                st["game_date"].dropna().dt.normalize().nunique()
                if "game_date" in st.columns
                else 0
            )
            if n_dates <= 1:
                st = attach_real_dates(st, sched, season_start_year=season)
        elif season >= 2025 and not st.empty:
            # Combined-stats carries real tip dates; tag as authoritative when
            # present so date-quality filter does not drop 2025–26 for lacking
            # odds-schedule exact matches (all_odds ends mid-season).
            from pipeline.dates import DATE_STATUS_AUTHORITATIVE, is_valid_timestamp
            if "date_status" not in st.columns:
                st["date_status"] = DATE_STATUS_AUTHORITATIVE
            else:
                missing = st["date_status"].isna() | (st["date_status"].astype(str) == "")
                st.loc[missing, "date_status"] = DATE_STATUS_AUTHORITATIVE
            # Unresolved only when game_date itself is unusable.
            bad_date = ~st["game_date"].map(is_valid_timestamp)
            if bad_date.any():
                from pipeline.dates import DATE_STATUS_UNRESOLVED
                st.loc[bad_date, "date_status"] = DATE_STATUS_UNRESOLVED
            n_games = st["GAME_ID"].nunique() if "GAME_ID" in st.columns else len(st)
            dmin, dmax = st["game_date"].min(), st["game_date"].max()
            print(
                f"  [dates] season {season}-{season+1}: authoritative combined-stats "
                f"dates for {n_games} games (range {dmin} -> {dmax})"
            )

        frames.append(st)
        if quick and len(frames) >= 2:
            break

    if not frames:
        raise FileNotFoundError("No PBP data found for the requested seasons.")

    all_stints = pd.concat(frames, ignore_index=True)
    all_stints["game_date"] = pd.to_datetime(all_stints["game_date"], errors="coerce")
    all_stints = all_stints.sort_values(["game_date", "GAME_ID", "stint_id"]).reset_index(drop=True)

    seasons = sorted(all_stints["season"].dropna().unique().tolist()) if "season" in all_stints.columns else []
    if use_cache and cache_path is not None:
        try:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            all_stints.to_pickle(cache_path)
            side_meta = build_stints_meta(
                cache_key=current_key,
                paths=paths_all,
                quick=quick,
                walkforward_zone_pps=walkforward_zone_pps,
                assist_split=assist,
                data_dir=data_dir,
                n_rows=len(all_stints),
                seasons=seasons,
                cache_path=str(cache_path),
            )
            write_stints_sidecar(cache_path, side_meta)
            meta["built_at"] = side_meta["built_at"]
            meta["sidecar"] = side_meta
            print(f"  💾 cached stints -> {cache_path.name} ({len(all_stints):,} rows)")
            print(f"  💾 sidecar -> {sidecar_path(cache_path).name}  built_at={side_meta['built_at']}")
        except Exception as e:  # noqa: BLE001
            print(f"  ⚠️ cache write failed: {e}")

    return all_stints, meta
