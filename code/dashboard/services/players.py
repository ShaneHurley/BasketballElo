"""Player rating snapshot loaders for the Player Data tab."""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import pandas as pd

from dashboard.config import EXPORTS_DIR, STATE_DIR
from dashboard.paths import run_dir_for_id


def _candidate_elo_paths(run_id: str | None = None) -> list[Path]:
    paths = [
        STATE_DIR / "latest_elo.pkl",
        STATE_DIR / "elo_state.pkl",
        STATE_DIR / "player_ratings.pkl",
    ]
    rd = run_dir_for_id(run_id) if run_id else None
    if rd:
        paths.extend([
            rd / "artifacts" / "latest_elo.pkl",
            rd / "state" / "latest_elo.pkl",
            rd / "latest_elo.pkl",
        ])
    return [p for p in paths if p.exists()]


def _ratings_from_obj(obj: Any) -> list[dict]:
    rows: list[dict] = []
    # PlayerRatingTracker-like
    ratings = getattr(obj, "ratings", None) or getattr(obj, "players", None) or obj
    if isinstance(ratings, dict):
        for pid, val in ratings.items():
            if isinstance(val, dict):
                rows.append({"player_id": str(pid), **{k: _num(v) for k, v in val.items()}})
            else:
                # Glicko-ish object
                row = {"player_id": str(pid)}
                for attr in ("mu", "offense", "defense", "rd", "phi", "sigma", "games", "team", "name"):
                    if hasattr(val, attr):
                        row[attr] = _num(getattr(val, attr))
                if "mu" not in row and hasattr(val, "rating"):
                    row["mu"] = _num(val.rating)
                rows.append(row)
    elif isinstance(ratings, list):
        for item in ratings:
            if isinstance(item, dict):
                rows.append(item)
    return rows


def _num(v: Any) -> Any:
    if isinstance(v, (int, float, str, bool)) or v is None:
        return v
    try:
        return float(v)
    except Exception:
        return str(v)


def load_player_snapshot(run_id: str | None = None, search: str | None = None) -> dict[str, Any]:
    paths = _candidate_elo_paths(run_id)
    if not paths:
        return {"players": [], "source": None, "n": 0, "message": "No rating snapshot found under state/"}
    src = paths[0]
    try:
        with open(src, "rb") as f:
            obj = pickle.load(f)
        rows = _ratings_from_obj(obj)
    except Exception as exc:
        return {"players": [], "source": str(src), "n": 0, "message": f"Failed to load: {exc}"}
    if search:
        s = search.lower()
        rows = [r for r in rows if s in str(r.get("player_id", "")).lower()
                or s in str(r.get("name", "")).lower()
                or s in str(r.get("team", "")).lower()]
    # Sort by |offense| or mu
    def sort_key(r):
        for k in ("offense", "mu", "rating"):
            if k in r and r[k] is not None:
                try:
                    return -abs(float(r[k]))
                except Exception:
                    pass
        return 0
    rows = sorted(rows, key=sort_key)
    return {"players": rows[:2000], "source": str(src), "n": len(rows), "message": None}


def export_players_csv(run_id: str | None = None) -> str:
    snap = load_player_snapshot(run_id)
    out = EXPORTS_DIR / f"players_snapshot_{pd.Timestamp.utcnow().strftime('%Y%m%d_%H%M%S')}.csv"
    pd.DataFrame(snap["players"]).to_csv(out, index=False)
    return str(out)
