"""Job ETA helpers: stage weights + EMA of historical module runtimes."""
from __future__ import annotations

import json
import time
from typing import Any

from dashboard.config import SUITE_STAGE_WEIGHTS, TIMINGS_PATH


def load_timings() -> dict[str, Any]:
    if not TIMINGS_PATH.exists():
        return {"ema_seconds": {}}
    try:
        return json.loads(TIMINGS_PATH.read_text())
    except Exception:
        return {"ema_seconds": {}}


def save_timings(data: dict[str, Any]) -> None:
    TIMINGS_PATH.write_text(json.dumps(data, indent=2))


def update_ema(module: str, elapsed_seconds: float, alpha: float = 0.3) -> float:
    data = load_timings()
    ema = data.setdefault("ema_seconds", {})
    prev = float(ema.get(module, elapsed_seconds))
    new = alpha * float(elapsed_seconds) + (1.0 - alpha) * prev
    ema[module] = new
    save_timings(data)
    return new


def estimate_eta_seconds(module: str, progress_pct: float, stage: str = "") -> float | None:
    """Blend remaining stage weight with historical EMA for ``module``."""
    data = load_timings()
    ema = float(data.get("ema_seconds", {}).get(module, 0.0) or 0.0)
    pct = max(0.0, min(100.0, float(progress_pct))) / 100.0
    if ema <= 0:
        # Cold start: crude defaults by module family.
        defaults = {
            "suite_smoke": 1800.0,
            "suite_custom": 7200.0,
            "backtest_quick": 900.0,
            "edge_policy_calib": 60.0,
            "ats_reliability": 15.0,
            "ml_calibration_report": 15.0,
            "totals_calibration_report": 15.0,
            "promotion_gates": 10.0,
            "export_slice": 5.0,
            "player_rating_smoke": 600.0,
        }
        ema = defaults.get(module, 120.0)
    remaining = ema * (1.0 - pct)
    # If we know suite stage, shrink remaining by unfinished weight fraction.
    if module.startswith("suite") and stage:
        total_w = sum(SUITE_STAGE_WEIGHTS.values()) or 1.0
        done_w = 0.0
        for name, w in SUITE_STAGE_WEIGHTS.items():
            done_w += w
            if name in stage or stage in name:
                break
        frac_left = max(0.0, 1.0 - done_w / total_w)
        remaining = max(remaining, ema * frac_left * 0.5)
    return float(max(0.0, remaining))


def now_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
