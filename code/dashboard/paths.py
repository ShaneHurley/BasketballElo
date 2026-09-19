"""Path sandbox: only OUTPUT_ROOT / STATE_DIR / DASHBOARD_DATA_DIR are readable."""
from __future__ import annotations

import json
from pathlib import Path

from dashboard.config import CODE_ROOT, DASHBOARD_DATA_DIR, OUTPUT_ROOT, STATE_DIR


ALLOWED_ROOTS = (OUTPUT_ROOT.resolve(), STATE_DIR.resolve(), DASHBOARD_DATA_DIR.resolve())


class PathSandboxError(ValueError):
    """Raised when a requested path escapes allowed roots."""


def safe_resolve(path: str | Path, under: Path | None = None) -> Path:
    """Resolve ``path`` and ensure it stays under an allowed root (or ``under``)."""
    p = Path(path).expanduser().resolve()
    roots = (under.resolve(),) if under is not None else ALLOWED_ROOTS
    for root in roots:
        try:
            p.relative_to(root)
            return p
        except ValueError:
            continue
    raise PathSandboxError(f"path outside sandbox: {p}")


def list_run_dirs() -> list[dict]:
    """Scan OUTPUT_ROOT for suite/backtest run folders (newest first)."""
    if not OUTPUT_ROOT.exists():
        return []
    rows = []
    for d in sorted(OUTPUT_ROOT.iterdir(), key=lambda x: x.stat().st_mtime, reverse=True):
        if not d.is_dir():
            continue
        results = None
        for rel in ("betting/backtest_results.csv", "backtest_results.csv"):
            p = d / rel
            if p.exists():
                results = p
                break
        rows.append({
            "id": d.name,
            "path": str(d),
            "has_results": results is not None,
            "results_path": str(results) if results else None,
            "mtime": d.stat().st_mtime,
        })
    return rows


def list_runs() -> list[dict]:
    """Alias used by services."""
    return list_run_dirs()


def run_dir_for_id(run_id: str | None) -> Path | None:
    """Map a run folder name to a sandboxed path; None if no id."""
    if not run_id:
        return None
    if "/" in run_id or "\\" in run_id or ".." in run_id:
        raise PathSandboxError(f"invalid run_id: {run_id!r}")
    return safe_resolve(OUTPUT_ROOT / run_id)


def latest_results_csv() -> Path | None:
    """Newest run that has a backtest_results.csv, else code/ fallback."""
    for row in list_run_dirs():
        if row.get("results_path"):
            return Path(row["results_path"])
    fallback = CODE_ROOT / "backtest_results.csv"
    return fallback if fallback.exists() else None


def results_csv_path(run_id: str | None) -> Path | None:
    """Prefer run-dir betting CSV; fall back to newest run, then code/backtest_results.csv."""
    if run_id:
        rd = run_dir_for_id(run_id)
        if rd is not None:
            for rel in ("betting/backtest_results.csv", "backtest_results.csv"):
                p = rd / rel
                if p.exists():
                    return p
        # Explicit run_id but missing CSV — do not silently switch runs.
        return None
    return latest_results_csv()


def list_artifacts(run_id: str) -> list[dict]:
    """List checkpoints / artifacts / plots under a run dir."""
    root = run_dir_for_id(run_id)
    if root is None:
        return []
    out = []
    for sub in ("checkpoints", "artifacts", "plots", "analysis_plots", "betting"):
        d = root / sub
        if not d.exists():
            continue
        for f in sorted(d.rglob("*")):
            if f.is_file():
                out.append({
                    "rel": str(f.relative_to(root)),
                    "path": str(f),
                    "size": f.stat().st_size,
                    "suffix": f.suffix.lower(),
                })
    return out


def read_json_if_exists(path: Path) -> dict | list | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def load_ui_state() -> dict:
    from dashboard.config import UI_STATE_PATH
    data = read_json_if_exists(UI_STATE_PATH)
    return data if isinstance(data, dict) else {}


def save_ui_state(state: dict) -> None:
    from dashboard.config import UI_STATE_PATH
    UI_STATE_PATH.write_text(json.dumps(state, indent=2, default=str))
