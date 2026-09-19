"""Param schemas and CLI builders for Run Lab modules."""
from __future__ import annotations

from typing import Any

# Field dict: {name, type, label, options?, default?, help?}
# type: str | int | float | bool | enum | run_id


SUITE_CUSTOM_SCHEMA: list[dict[str, Any]] = [
    {"name": "preset", "type": "enum", "label": "Preset", "options": ["custom", "smoke", "standard", "full"],
     "default": "custom", "help": "Prefills years/trials; still editable"},
    {"name": "label", "type": "str", "label": "Run label", "default": "dash_suite"},
    {"name": "years", "type": "str", "label": "Years (comma)", "default": "2021,2022,2023,2024",
     "help": "Empty = mini (newest 2 seasons)"},
    {"name": "start_season", "type": "int", "label": "Start season", "default": None},
    {"name": "end_season", "type": "int", "label": "End season", "default": None},
    {"name": "elo_trials", "type": "int", "label": "Elo trials", "default": 12},
    {"name": "hier_trials", "type": "int", "label": "Hier trials", "default": 12},
    {"name": "meta_trials", "type": "int", "label": "Meta trials", "default": 18},
    {"name": "window", "type": "enum", "label": "Train window", "options": ["2", "3", "4", "5"], "default": "4"},
    {"name": "stints", "type": "enum", "label": "Stints", "options": ["auto", "rebuild", "reuse"], "default": "auto"},
    {"name": "fast_tuning", "type": "bool", "label": "Fast tuning", "default": True},
    {"name": "from_stage", "type": "int", "label": "From stage (0–8)", "default": 0},
    {"name": "to_stage", "type": "int", "label": "To stage (0–8)", "default": 8},
    {"name": "run_dir", "type": "run_id", "label": "Resume run dir", "default": "",
     "help": "Pick an existing output/ run to resume"},
    {"name": "skip_integrity_tests", "type": "bool", "label": "Skip integrity tests", "default": False},
    {"name": "allow_interpolated_dates", "type": "bool", "label": "Allow interpolated dates", "default": False},
    {"name": "venn_abers", "type": "bool", "label": "Venn-Abers", "default": False},
    {"name": "rolling_z", "type": "bool", "label": "Rolling Z features", "default": False},
    {"name": "skip_odds_gate", "type": "bool", "label": "Skip odds gate (debug)", "default": False},
    {"name": "no_cache", "type": "bool", "label": "No cache", "default": False},
    {"name": "force_stints_cache", "type": "bool", "label": "Force stints cache", "default": False},
]

SUITE_SMOKE_SCHEMA: list[dict[str, Any]] = [
    {"name": "skip_integrity", "type": "bool", "label": "Skip integrity tests", "default": True},
    {"name": "label", "type": "str", "label": "Label", "default": ""},
]

BACKTEST_SCHEMA: list[dict[str, Any]] = [
    {"name": "quick", "type": "bool", "label": "Quick (2 seasons)", "default": True},
    {"name": "elo_trials", "type": "int", "label": "Elo trials", "default": 12},
    {"name": "hier_trials", "type": "int", "label": "Hier trials", "default": 12},
    {"name": "meta_trials", "type": "int", "label": "Meta trials", "default": 18},
    {"name": "window", "type": "enum", "label": "Window", "options": ["2", "3", "4", "5"], "default": "4"},
    {"name": "fast_tuning", "type": "bool", "label": "Fast tuning", "default": True},
]

SUITE_PRESETS: dict[str, dict[str, Any]] = {
    "smoke": {
        "years": "",
        "elo_trials": 6,
        "hier_trials": 6,
        "meta_trials": 8,
        "window": "4",
        "fast_tuning": True,
        "skip_integrity_tests": True,
        "stints": "auto",
        "label": "dash_smoke",
    },
    "standard": {
        "years": "2021,2022,2023,2024",
        "elo_trials": 12,
        "hier_trials": 12,
        "meta_trials": 18,
        "window": "4",
        "fast_tuning": True,
        "skip_integrity_tests": False,
        "stints": "auto",
        "label": "dash_standard",
    },
    "full": {
        "years": "2020,2021,2022,2023,2024,2025",
        "elo_trials": 24,
        "hier_trials": 24,
        "meta_trials": 36,
        "window": "4",
        "fast_tuning": False,
        "skip_integrity_tests": False,
        "stints": "rebuild",
        "label": "dash_full",
    },
}


def _truthy(v: Any) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).lower() in ("1", "true", "yes", "on")


def build_suite_cli(py: str, script: str, params: dict[str, Any], *, job_id: str = "") -> list[str]:
    """Build argv for run_full_suite.py from Run Lab params."""
    p = dict(params or {})
    preset = str(p.get("preset") or "custom")
    if preset in SUITE_PRESETS:
        merged = {**SUITE_PRESETS[preset], **{k: v for k, v in p.items() if v not in (None, "", "custom")}}
        p = merged

    label = p.get("label") or (f"dash_{job_id}" if job_id else "dash_suite")
    cmd = [py, script, "--yes", "--label", str(label)]

    if _truthy(p.get("fast_tuning", True)):
        cmd.append("--fast-tuning")

    years = p.get("years")
    if years not in (None, ""):
        cmd.extend(["--years", str(years)])
    if p.get("start_season") not in (None, ""):
        cmd.extend(["--start-season", str(int(p["start_season"]))])
    if p.get("end_season") not in (None, ""):
        cmd.extend(["--end-season", str(int(p["end_season"]))])
    if p.get("elo_trials") not in (None, ""):
        cmd.extend(["--elo-trials", str(int(p["elo_trials"]))])
    if p.get("hier_trials") not in (None, ""):
        cmd.extend(["--hier-trials", str(int(p["hier_trials"]))])
    if p.get("meta_trials") not in (None, ""):
        cmd.extend(["--meta-trials", str(int(p["meta_trials"]))])
    if p.get("window") not in (None, ""):
        cmd.extend(["--window", str(int(p["window"]))])

    stints = p.get("stints") or "auto"
    if stints:
        cmd.extend(["--stints", str(stints)])

    if p.get("from_stage") not in (None, ""):
        cmd.extend(["--from-stage", str(int(p["from_stage"]))])
    if p.get("to_stage") not in (None, ""):
        cmd.extend(["--to-stage", str(int(p["to_stage"]))])

    run_dir = p.get("run_dir") or p.get("resume_run_id")
    if run_dir:
        # Accept bare run id → resolve under output/
        from dashboard.config import OUTPUT_ROOT
        rd = str(run_dir)
        path = rd if rd.startswith("/") or "\\" in rd or rd.startswith("..") else str(OUTPUT_ROOT / rd)
        cmd.extend(["--run-dir", path])

    if _truthy(p.get("skip_integrity_tests") or p.get("skip_integrity")):
        cmd.append("--skip-integrity-tests")
    if _truthy(p.get("allow_interpolated_dates")):
        cmd.append("--allow-interpolated-dates")
    if _truthy(p.get("venn_abers")):
        cmd.append("--venn-abers")
    if _truthy(p.get("rolling_z")):
        cmd.append("--rolling-z")
    if _truthy(p.get("skip_odds_gate")):
        cmd.append("--skip-odds-gate")
    if _truthy(p.get("no_cache")):
        cmd.append("--no-cache")
    if _truthy(p.get("force_stints_cache")):
        cmd.append("--force-stints-cache")
    return cmd


def build_backtest_cli(py: str, script: str, params: dict[str, Any]) -> list[str]:
    p = dict(params or {})
    cmd = [py, script]
    if _truthy(p.get("quick", True)):
        cmd.append("--quick")
    if _truthy(p.get("fast_tuning", True)):
        cmd.append("--fast-tuning")
    if p.get("elo_trials") not in (None, ""):
        cmd.extend(["--elo-trials", str(int(p["elo_trials"]))])
    if p.get("hier_trials") not in (None, ""):
        cmd.extend(["--hier-trials", str(int(p["hier_trials"]))])
    if p.get("meta_trials") not in (None, ""):
        cmd.extend(["--meta-trials", str(int(p["meta_trials"]))])
    if p.get("window") not in (None, ""):
        cmd.extend(["--window", str(int(p["window"]))])
    return cmd
