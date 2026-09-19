"""Resolve local data directories and discover PBP season files."""
from __future__ import annotations

import platform
import re
import sys
from pathlib import Path
from typing import Iterable

_REPO_ROOT = Path(__file__).resolve().parent.parent
MAC_DEFAULT_DATA_DIR = Path("/Users/shurley/Documents/basketball/basketballData/data")
DEFAULT_OUTPUT_ROOT = Path("/Users/shurley/Documents/basketball/BasketballElo/output")

_V3_RE = re.compile(r"events_(\d{4})_(\d{2})_pbp_V3\.csv$", re.IGNORECASE)
_COMBINED_RE = re.compile(r"combined-stats\.csv$", re.IGNORECASE)


def is_macos() -> bool:
    return sys.platform == "darwin" or platform.system() == "Darwin"


def resolve_data_dir(explicit: str | Path | None = None) -> Path:
    """Return the PBP/odds data directory.

    Mac defaults to MAC_DEFAULT_DATA_DIR when present. Non-Mac requires
    ``explicit`` (``--data-dir``); otherwise raises SystemExit-friendly ValueError.
    """
    if explicit is not None:
        path = Path(explicit).expanduser().resolve()
        if not path.is_dir():
            raise ValueError(f"Data directory does not exist: {path}")
        return path

    if is_macos():
        if MAC_DEFAULT_DATA_DIR.is_dir():
            return MAC_DEFAULT_DATA_DIR.resolve()
        raise ValueError(
            f"macOS default data dir missing: {MAC_DEFAULT_DATA_DIR}\n"
            "Pass --data-dir PATH to designate a location."
        )

    raise ValueError(
        "Non-macOS environment: you must designate a data location with --data-dir PATH."
    )


def resolve_output_root(explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if is_macos() and DEFAULT_OUTPUT_ROOT.parent.is_dir():
        return DEFAULT_OUTPUT_ROOT
    return _REPO_ROOT / "BasketballElo" / "output"


def discover_pbp_paths(data_dir: Path) -> dict[int, Path]:
    """Map season-start year → PBP file.

    V3 files use start year (events_2021_22 → 2021). Combined-stats (2025-26)
    maps to 2025.
    """
    found: dict[int, Path] = {}
    for path in sorted(data_dir.iterdir()):
        if not path.is_file():
            continue
        m = _V3_RE.match(path.name)
        if m:
            found[int(m.group(1))] = path
            continue
        if _COMBINED_RE.search(path.name):
            found[2025] = path
    return dict(sorted(found.items()))


def resolve_odds_path(data_dir: Path, repo_root: Path | None = None) -> Path | None:
    repo_root = repo_root or _REPO_ROOT
    for candidate in (
        data_dir / "all_odds.csv",
        repo_root / "all_odds.csv",
        repo_root / "basketballData" / "all_odds.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def resolve_pinnacle_path(data_dir: Path, repo_root: Path | None = None) -> Path | None:
    repo_root = repo_root or _REPO_ROOT
    for candidate in (
        data_dir / "nba_main_lines.csv",
        repo_root / "nba_main_lines.csv",
        repo_root / "basketballData" / "nba_main_lines.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def resolve_player_list_path(data_dir: Path, repo_root: Path | None = None) -> Path | None:
    repo_root = repo_root or _REPO_ROOT
    for candidate in (
        data_dir / "nba_players_all.csv",
        repo_root / "basketballData" / "nba_players_all.csv",
        repo_root / "nba_players_all.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def load_player_names(data_dir: Path) -> tuple[dict, dict]:
    """Return (names_dict pid→name, name_to_id name→pid)."""
    import pandas as pd

    path = resolve_player_list_path(data_dir)
    if path is None:
        return {}, {}
    df = pd.read_csv(path, low_memory=False)
    pid_col = "person_id" if "person_id" in df.columns else "PERSON_ID"
    name_col = "display_first_last" if "display_first_last" in df.columns else "DISPLAY_FIRST_LAST"
    if pid_col not in df.columns or name_col not in df.columns:
        return {}, {}
    names = df.set_index(pid_col)[name_col].to_dict()
    name_to_id = {v: k for k, v in names.items()}
    return names, name_to_id


def parse_years_arg(years: str | None) -> list[int] | None:
    """Parse ``--years 2021,2022,2023`` into start-year ints."""
    if not years:
        return None
    out: list[int] = []
    for part in years.split(","):
        part = part.strip()
        if not part:
            continue
        out.append(int(part))
    return out or None


def select_season_keys(
    available: dict[int, Path],
    *,
    years: Iterable[int] | None = None,
    start_season: int | None = None,
    end_season: int | None = None,
    mini: bool = False,
    mini_n: int = 2,
) -> list[int]:
    """Choose season-start years to load.

    Mini (default when years/start/end omitted): newest ``mini_n`` seasons.
    """
    keys = sorted(available.keys())
    if not keys:
        return []

    if years is not None:
        wanted = [y for y in years if y in available]
        missing = [y for y in years if y not in available]
        if missing:
            raise ValueError(
                f"Requested years not found in data dir: {missing}. "
                f"Available: {keys}"
            )
        return sorted(wanted)

    if start_season is not None or end_season is not None:
        lo = start_season if start_season is not None else keys[0]
        hi = end_season if end_season is not None else keys[-1]
        selected = [y for y in keys if lo <= y <= hi]
        if not selected:
            raise ValueError(
                f"No seasons in [{lo}, {hi}]. Available: {keys}"
            )
        return selected

    if mini:
        return keys[-mini_n:] if len(keys) >= mini_n else keys

    raise ValueError(
        "Seasons are CLI-only for full runs. Pass --years and/or "
        "--start-season/--end-season, or omit them for the mini default "
        f"(newest {mini_n} seasons)."
    )


def auto_rolling_window(elo_trials: int, hier_trials: int, meta_trials: int) -> int:
    """Higher window when trial budget is low (plan policy). Allowed range 2–5."""
    avg = (elo_trials + hier_trials + meta_trials) / 3.0
    if avg <= 8:
        return 5
    if avg <= 15:
        return 4
    if avg <= 30:
        return 3
    return 2


def clamp_rolling_window(size: int, *, lo: int = 2, hi: int = 5) -> int:
    """Clamp model-train rolling window to the multi-year policy range."""
    return int(max(lo, min(hi, int(size))))


def split_rating_and_model_seasons(
    all_seasons: list,
    test_index: int,
    rolling_window_size: int,
    *,
    rating_history_use_all: bool = True,
) -> tuple[list, list]:
    """Split prior seasons into rating warm-start vs model-train windows.

    Returns ``(rating_history_seasons, model_train_seasons)``. Both are
    chronological lists of seasons strictly before ``all_seasons[test_index]``.
    """
    prior = list(all_seasons[:test_index])
    if not prior:
        return [], []
    window = clamp_rolling_window(rolling_window_size)
    model_train = prior[-window:] if len(prior) >= window else list(prior)
    if rating_history_use_all:
        rating_history = list(prior)
    else:
        rating_history = list(model_train)
    return rating_history, model_train
