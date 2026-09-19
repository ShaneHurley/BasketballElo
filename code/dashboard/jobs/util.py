"""Shared helpers for dashboard job modules."""
from __future__ import annotations

import glob
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard.config import CODE_ROOT

# Pip name → import name for packages listed in requirements-pipeline.txt
_IMPORT_NAME: dict[str, str] = {
    "scikit-learn": "sklearn",
    "nba_api": "nba_api",
    "nba-api": "nba_api",
}

# Required for run_full_suite / backtest import path (catboost is top-level in model.py).
_REQUIRED_IMPORTS: tuple[str, ...] = (
    "numpy",
    "pandas",
    "sklearn",
    "tqdm",
    "optuna",
    "matplotlib",
    "scipy",
    "catboost",
)

# Bonus score only — not fatal if missing.
_OPTIONAL_IMPORTS: tuple[str, ...] = (
    "requests",
    "xgboost",
    "lightgbm",
    "nba_api",
)

PIPELINE_REQUIREMENTS_FILE = CODE_ROOT / "requirements-pipeline.txt"


def _pip_name_to_import(name: str) -> str:
    base = name.strip().lower()
    base = re.split(r"[<>=!~;\[]", base, maxsplit=1)[0].strip()
    return _IMPORT_NAME.get(base, base.replace("-", "_"))


def pipeline_requirement_imports() -> tuple[list[str], list[str]]:
    """Return (required_imports, optional_imports) from requirements-pipeline.txt when present."""
    required = list(_REQUIRED_IMPORTS)
    optional = list(_OPTIONAL_IMPORTS)
    path = PIPELINE_REQUIREMENTS_FILE
    if not path.exists():
        return required, optional

    in_optional = False
    file_required: list[str] = []
    file_optional: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("#"):
            low = line.lower()
            if "optional" in low:
                in_optional = True
            elif "required" in low:
                in_optional = False
            continue
        if line.startswith("-r "):
            continue
        mod = _pip_name_to_import(line)
        if not mod or mod.startswith("."):
            continue
        if in_optional:
            if mod not in file_optional:
                file_optional.append(mod)
        else:
            if mod not in file_required:
                file_required.append(mod)

    if file_required:
        # Keep hard-coded order for stable scoring, then any extras from the file.
        ordered = [m for m in _REQUIRED_IMPORTS if m in file_required]
        ordered.extend(m for m in file_required if m not in ordered)
        required = ordered
    if file_optional:
        ordered_opt = [m for m in _OPTIONAL_IMPORTS if m in file_optional]
        ordered_opt.extend(m for m in file_optional if m not in ordered_opt)
        optional = ordered_opt
    return required, optional


def probe_python(py: str) -> dict[str, Any]:
    """Probe ``py`` for required + optional pipeline imports."""
    required, optional = pipeline_requirement_imports()
    present: list[str] = []
    missing_required: list[str] = []
    missing_optional: list[str] = []
    for mod in required:
        try:
            r = subprocess.run(
                [py, "-c", f"import {mod}"],
                capture_output=True,
                timeout=30,
                env={**os.environ, "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/mpl")},
            )
            if r.returncode == 0:
                present.append(mod)
            else:
                missing_required.append(mod)
        except Exception:
            missing_required.append(mod)
    for mod in optional:
        try:
            r = subprocess.run(
                [py, "-c", f"import {mod}"],
                capture_output=True,
                timeout=30,
                env={**os.environ, "MPLCONFIGDIR": os.environ.get("MPLCONFIGDIR", "/tmp/mpl")},
            )
            if r.returncode == 0:
                present.append(mod)
            else:
                missing_optional.append(mod)
        except Exception:
            missing_optional.append(mod)

    # Score: required count * 10 + optional (so a complete required set always beats partial).
    score = (len(required) - len(missing_required)) * 10 + (
        len(optional) - len(missing_optional)
    )
    return {
        "python": py,
        "present": present,
        "missing": missing_required + missing_optional,
        "missing_required": missing_required,
        "missing_optional": missing_optional,
        "required": required,
        "optional": optional,
        "score": score,
        "ready": not missing_required,
    }


def _pipeline_python_candidates() -> list[str]:
    """Interpreters to try; PATH alone is not enough when the venv shadows python3."""
    candidates: list[str] = []
    candidates.append(sys.executable)
    for name in ("python3", "python", "python3.9", "python3.10", "python3.11", "python3.12"):
        found = shutil.which(name)
        if found:
            candidates.append(found)
    for rel in (".venv/bin/python", "venv/bin/python", ".venv_dashboard/bin/python"):
        p = CODE_ROOT / rel
        if p.exists():
            candidates.append(str(p))
    for pattern in (
        "/Library/Frameworks/Python.framework/Versions/*/bin/python3",
        str(Path.home() / ".pyenv/versions/*/bin/python"),
    ):
        for path in sorted(glob.glob(pattern), reverse=True):
            if Path(path).is_file():
                candidates.append(path)
    return candidates


def install_hint(py: str | None = None) -> str:
    """Shell one-liner to install requirements-pipeline.txt into ``py``."""
    target = py or sys.executable
    req = PIPELINE_REQUIREMENTS_FILE
    return f'"{target}" -m pip install -r "{req}"'


def ensure_pipeline_ready(py: str, *, log: list[str] | None = None) -> dict[str, Any]:
    """Re-probe ``py`` and raise if required imports are missing."""
    info = probe_python(py)
    if log is not None:
        log.append(
            f"pipeline_deps ready={info['ready']} present={info['present']} "
            f"missing_required={info['missing_required']} missing_optional={info['missing_optional']}"
        )
        if not info["ready"]:
            log.append(f"pipeline_deps install: {install_hint(py)}")
    if not info["ready"]:
        miss = ", ".join(info["missing_required"])
        raise RuntimeError(
            f"Pipeline Python is missing required packages: {miss}. "
            f"Install with: {install_hint(py)}"
        )
    return info


def pipeline_python(*, log: list[str] | None = None) -> str:
    """Pick a Python with the most complete pipeline stack.

    Override with DASHBOARD_PIPELINE_PYTHON. See requirements-pipeline.txt.
    """
    override = os.environ.get("DASHBOARD_PIPELINE_PYTHON")
    if override and Path(override).exists():
        info = probe_python(override)
        if log is not None:
            log.append(
                f"pipeline_python override={override} ready={info['ready']} "
                f"present={info['present']} missing_required={info['missing_required']} "
                f"missing_optional={info['missing_optional']}"
            )
            if not info["ready"]:
                log.append(f"pipeline_deps install: {install_hint(override)}")
        return override

    seen: set[str] = set()
    best: dict[str, Any] | None = None
    required, _optional = pipeline_requirement_imports()
    perfect = len(required) * 10 + len(_optional)
    for py in _pipeline_python_candidates():
        if not py:
            continue
        # Do NOT resolve() — a venv's python symlink points at the base
        # interpreter binary but uses different site-packages.
        key = str(Path(py))
        if key in seen:
            continue
        seen.add(key)
        info = probe_python(py)
        if best is None or info["score"] > best["score"]:
            best = info
        if info["score"] >= perfect and info["ready"]:
            break

    if best is None:
        if log is not None:
            log.append(f"pipeline_python fallback={sys.executable} (no candidates probed)")
        return sys.executable

    if log is not None:
        log.append(
            f"pipeline_python chosen={best['python']} ready={best['ready']} "
            f"present={best['present']} missing_required={best['missing_required']} "
            f"missing_optional={best['missing_optional']}"
        )
        if not best["ready"]:
            log.append(f"pipeline_deps install: {install_hint(best['python'])}")
    return str(best["python"])


def ensure_ats_win(df: pd.DataFrame) -> pd.DataFrame:
    """Add ATS_WIN when missing (common on older/ablation CSVs).

    Grades lean side vs decision/T-60 spread when present, else market spread.
    Pushes → NaN (not counted as losses).
    """
    out = df.copy()
    if "ATS_WIN" in out.columns:
        return out
    spread_col = None
    for c in ("DECISION_SPREAD", "MARKET_SPREAD", "CLOSING_SPREAD"):
        if c in out.columns:
            spread_col = c
            break
    if spread_col is None or "ACTUAL_MARGIN" not in out.columns:
        return out

    margin = pd.to_numeric(out["ACTUAL_MARGIN"], errors="coerce")
    spread = pd.to_numeric(out[spread_col], errors="coerce")
    cover = margin + spread  # >0 home covers, <0 away covers, ~0 push

    if "DIRECTION" in out.columns:
        side = out["DIRECTION"].astype(str)
        pass_rate = float(side.str.lower().isin(("pass", "nan", "none", "")).mean())
        # DIRECTION is often only filled for actionable bets; lean grades the full edge grid.
        if pass_rate > 0.5:
            if "EDGE_LEAN" in out.columns:
                side = out["EDGE_LEAN"].astype(str)
            elif "EDGE" in out.columns:
                e = pd.to_numeric(out["EDGE"], errors="coerce")
                side = pd.Series(
                    np.where(e > 0, "Home", np.where(e < 0, "Away", "Pass")),
                    index=out.index,
                )
    elif "EDGE_LEAN" in out.columns:
        side = out["EDGE_LEAN"].astype(str)
    elif "EDGE" in out.columns:
        e = pd.to_numeric(out["EDGE"], errors="coerce")
        side = pd.Series(
            np.where(e > 0, "Home", np.where(e < 0, "Away", "Pass")),
            index=out.index,
        )
    else:
        return out

    home_win = cover > 1e-9
    away_win = cover < -1e-9
    push = cover.notna() & ~home_win & ~away_win
    ats = pd.Series(np.nan, index=out.index, dtype=float)
    home_mask = side.str.lower().isin(("home", "h"))
    away_mask = side.str.lower().isin(("away", "a"))
    ats.loc[home_mask & home_win] = 1.0
    ats.loc[home_mask & away_win] = 0.0
    ats.loc[away_mask & away_win] = 1.0
    ats.loc[away_mask & home_win] = 0.0
    ats.loc[push & (home_mask | away_mask)] = np.nan
    out["ATS_WIN"] = ats
    return out
