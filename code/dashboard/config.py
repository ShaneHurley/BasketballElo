"""Dashboard paths, bind defaults, and job concurrency knobs."""
from __future__ import annotations

from pathlib import Path

# code/ is the package root used by pipeline.config.STATE_DIR
CODE_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = CODE_ROOT.parent  # BasketballElo/

OUTPUT_ROOT = REPO_ROOT / "output"
STATE_DIR = CODE_ROOT / "state"
DASHBOARD_DATA_DIR = REPO_ROOT / "dashboard_data"
JOBS_DIR = DASHBOARD_DATA_DIR / "jobs"
EXPORTS_DIR = DASHBOARD_DATA_DIR / "exports"
TIMINGS_PATH = DASHBOARD_DATA_DIR / "job_timings.json"
UI_STATE_PATH = DASHBOARD_DATA_DIR / "ui_state.json"

# Localhost-only by default (LAN requires explicit --host).
DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# One heavy suite/backtest at a time; light reports share a small pool.
MAX_HEAVY_JOBS = 1
MAX_LIGHT_WORKERS = 4
HEAVY_MODULES = frozenset({
    "suite_smoke",
    "suite_custom",
    "backtest_quick",
    "player_rating_smoke",
})

# Suite stage weights for ETA (stages 0–8 in run_full_suite).
SUITE_STAGE_WEIGHTS = {
    "0_toggles": 0.02,
    "1_stints": 0.15,
    "2_integrity": 0.05,
    "3_5_walkforward": 0.55,
    "6_review": 0.08,
    "7_policy": 0.10,
    "8_persist": 0.05,
}

for _d in (DASHBOARD_DATA_DIR, JOBS_DIR, EXPORTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)
