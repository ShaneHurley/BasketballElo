"""Parse suite/backtest job logs into structured Run Viewer telemetry."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from dashboard.config import SUITE_STAGE_WEIGHTS

# Friendly pipeline labels (stages 0–8).
PIPELINE_STEPS: list[dict[str, Any]] = [
    {"id": 0, "name": "Setup", "key": "toggles"},
    {"id": 1, "name": "Stints", "key": "stints"},
    {"id": 2, "name": "Checks", "key": "integrity"},
    {"id": 3, "name": "Engines", "key": "engines"},
    {"id": 4, "name": "Calibrators", "key": "calibrators"},
    {"id": 5, "name": "Simulate", "key": "walkforward"},
    {"id": 6, "name": "Review", "key": "review"},
    {"id": 7, "name": "Policy", "key": "policy"},
    {"id": 8, "name": "Save", "key": "persist"},
]

_TUNER_GLOSS = {
    "elo": "These knobs control how player ratings update after each game.",
    "hier": "These knobs blend team and lineup signals before predictions.",
    "meta": "This is the main score/margin model trained on past seasons.",
}

_RE_SUITE_DIR = re.compile(r"Suite run dir:\s*(.+)$")
_RE_STAGE = re.compile(r"^Stage\s+([0-8])\s+[—\-]")
_RE_SEASON = re.compile(r"SIMULATING SEASON\s+(\d{4})-(\d{4})")
_RE_WF_OVER = re.compile(r"Walk-forward over\s+(\d+)\s+seasons:\s*(.+)$")
_RE_SEASONS_START = re.compile(r"Seasons \(start years\):\s*(.+)$")
_RE_META_TRIAL = re.compile(
    r"✓\s*trial\s+(\d+)/(\d+)\s+score=([0-9.]+)",
    re.IGNORECASE,
)
_RE_META_FOLD = re.compile(
    r"trial\s+(\d+):\s*outer fold\s+(\d+)/(\d+)",
    re.IGNORECASE,
)
_RE_ELO_TUNING = re.compile(r"Tuning Elo tracker", re.I)
_RE_HIER_TUNING = re.compile(r"Tuning Hierarchical", re.I)
_RE_META_TUNING = re.compile(r"Tuning MetaScoreModel|Tuning margin model", re.I)
_RE_TRIAL_HEADER = re.compile(r"^Trial\s+(\d+)\s*:")
_RE_CV_MAE = re.compile(r"CV MAE:\s*([0-9.]+)")
_RE_PROGRESS_EH = re.compile(r"progress:\s*(\d+)/(\d+)")
_RE_BEST_CV = re.compile(r"Best CV MAE:\s*([0-9.]+)")
_RE_BEST_SCORE = re.compile(r"Best score:\s*([0-9.]+)", re.I)
_RE_ELO_DONE = re.compile(r"ELO TUNING COMPLETE", re.I)
_RE_HIER_DONE = re.compile(r"HIERARCHICAL TUNING COMPLETE", re.I)
_RE_META_DONE = re.compile(r"Margin tuning complete", re.I)
_RE_FEATURES = re.compile(r"Building features:.*?(\d+)/(\d+)")
_RE_SUITE_DONE = re.compile(r"Suite complete", re.I)
_RE_WALL = re.compile(r"Wall time:\s*([\d.]+)s")
_RE_TS = re.compile(r"^\[(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})")


def _empty_tuner(name: str) -> dict[str, Any]:
    return {
        "name": name,
        "trial": 0,
        "total": 0,
        "best_mae": None,
        "scores": [],
        "state": "pending",
        "gloss": _TUNER_GLOSS.get(name, ""),
    }


def _mean(xs: list[float]) -> float | None:
    if not xs:
        return None
    return float(sum(xs) / len(xs))


def _ema(prev: float | None, value: float, alpha: float = 0.2) -> float:
    if prev is None:
        return value
    return alpha * value + (1.0 - alpha) * prev


def parse_log_text(
    text: str,
    *,
    params: dict[str, Any] | None = None,
    started_at: str | None = None,
    elapsed_s: float | None = None,
) -> dict[str, Any]:
    """Parse a job.log body into a telemetry snapshot."""
    params = params or {}
    lines = text.splitlines()

    run_dir: str | None = None
    current_stage = 0
    stages_seen: set[int] = set()
    season_label: str | None = None
    season_index = 0
    season_total = 0
    seasons_list: list[str] = []
    season_starts: list[float] = []  # line indices as proxies if no wall clock
    trial_durations: list[float] = []
    season_durations: list[float] = []

    tuners = {
        "elo": _empty_tuner("elo"),
        "hier": _empty_tuner("hier"),
        "meta": _empty_tuner("meta"),
    }
    active_tuner: str | None = None
    last_trial_line_idx: int | None = None
    last_season_line_idx: int | None = None
    features_pct: float | None = None
    suite_done = False
    wall_s: float | None = None

    # Default totals from launch params.
    elo_trials = int(params.get("elo_trials") or 0) or 0
    hier_trials = int(params.get("hier_trials") or 0) or 0
    meta_trials = int(params.get("meta_trials") or 0) or 0
    if elo_trials:
        tuners["elo"]["total"] = elo_trials
    if hier_trials:
        tuners["hier"]["total"] = hier_trials
    if meta_trials:
        tuners["meta"]["total"] = meta_trials

    years_param = params.get("years")
    if isinstance(years_param, str) and years_param.strip():
        seasons_list = [y.strip() for y in years_param.split(",") if y.strip()]
        season_total = len(seasons_list)

    for i, raw in enumerate(lines):
        line = raw.rstrip()
        # Strip dashboard progress spam prefixes for matching.
        clean = line
        if "stage=running" in clean and "]" in clean:
            # Prefer the payload after the last nested stamp when present.
            parts = clean.split("] ")
            if len(parts) > 1:
                clean = parts[-1]

        m = _RE_SUITE_DIR.search(clean)
        if m:
            run_dir = m.group(1).strip()

        m = _RE_STAGE.search(clean.strip())
        if m:
            current_stage = int(m.group(1))
            stages_seen.add(current_stage)
            if current_stage >= 3:
                # Walk-forward block starts.
                for s in (3, 4, 5):
                    stages_seen.add(s)

        if "Running walk-forward" in clean:
            current_stage = max(current_stage, 3)
            for s in (3, 4, 5):
                stages_seen.add(s)

        m = _RE_SEASONS_START.search(clean)
        if m:
            try:
                # e.g. [2022, 2023, 2024, 2025]
                inner = m.group(1).strip()
                nums = re.findall(r"\d{4}", inner)
                if nums:
                    seasons_list = nums
                    season_total = len(nums)
            except Exception:
                pass

        m = _RE_WF_OVER.search(clean)
        if m:
            season_total = int(m.group(1))
            nums = re.findall(r"\d{4}", m.group(2))
            if nums:
                seasons_list = nums

        m = _RE_SEASON.search(clean)
        if m:
            y0, y1 = m.group(1), m.group(2)
            season_label = f"{y0}–{y1}"
            # Index by end year when possible.
            end_y = y1
            if seasons_list:
                try:
                    # seasons_list often start years; end year = start+1 label
                    idx = None
                    for j, s in enumerate(seasons_list):
                        if str(int(s) + 1) == end_y or s == end_y or s == y0:
                            idx = j
                            break
                    if idx is not None:
                        season_index = idx + 1
                    else:
                        season_index = min(season_total, season_index + 1) if season_total else season_index + 1
                except Exception:
                    season_index = season_index + 1
            else:
                season_index = season_index + 1
            if last_season_line_idx is not None:
                # Approximate duration by line gap (refined if wall clocks later).
                season_durations.append(max(1.0, float(i - last_season_line_idx) * 2.0))
            last_season_line_idx = i
            season_starts.append(float(i))
            # New season → reset tuners to pending for next round (keep historical scores).
            for t in tuners.values():
                if t["state"] == "running":
                    t["state"] = "done" if t["trial"] and t["total"] and t["trial"] >= t["total"] else t["state"]
            active_tuner = None

        if _RE_ELO_TUNING.search(clean):
            active_tuner = "elo"
            tuners["elo"]["state"] = "running"
            current_stage = max(current_stage, 3)
        if _RE_HIER_TUNING.search(clean):
            active_tuner = "hier"
            tuners["hier"]["state"] = "running"
            if tuners["elo"]["state"] == "running":
                tuners["elo"]["state"] = "done"
            current_stage = max(current_stage, 3)
        if _RE_META_TUNING.search(clean):
            active_tuner = "meta"
            tuners["meta"]["state"] = "running"
            for k in ("elo", "hier"):
                if tuners[k]["state"] == "running":
                    tuners[k]["state"] = "done"
            current_stage = max(current_stage, 3)

        m = _RE_TRIAL_HEADER.match(clean.strip())
        if m and active_tuner in ("elo", "hier"):
            tnum = int(m.group(1))
            tuners[active_tuner]["trial"] = tnum + 1  # Optuna 0-based
            if last_trial_line_idx is not None:
                trial_durations.append(max(0.5, float(i - last_trial_line_idx) * 1.5))
            last_trial_line_idx = i

        m = _RE_CV_MAE.search(clean)
        if m and active_tuner in ("elo", "hier"):
            score = float(m.group(1))
            tuners[active_tuner]["scores"].append(score)
            best = tuners[active_tuner]["best_mae"]
            if best is None or score < best:
                tuners[active_tuner]["best_mae"] = score

        m = _RE_PROGRESS_EH.search(clean)
        if m and active_tuner in ("elo", "hier"):
            tuners[active_tuner]["trial"] = int(m.group(1))
            tuners[active_tuner]["total"] = int(m.group(2))

        m = _RE_META_TRIAL.search(clean)
        if m:
            active_tuner = "meta"
            tuners["meta"]["state"] = "running"
            k, n, score = int(m.group(1)), int(m.group(2)), float(m.group(3))
            tuners["meta"]["trial"] = k
            tuners["meta"]["total"] = n
            tuners["meta"]["scores"].append(score)
            best = tuners["meta"]["best_mae"]
            if best is None or score < best:
                tuners["meta"]["best_mae"] = score
            if last_trial_line_idx is not None:
                # Prefer wall-clock if timestamps available on lines.
                trial_durations.append(max(1.0, float(i - last_trial_line_idx) * 2.5))
            last_trial_line_idx = i

        m = _RE_META_FOLD.search(clean)
        if m:
            active_tuner = "meta"
            tuners["meta"]["state"] = "running"
            # 1-based trial currently in progress (not yet completed).
            k = int(m.group(1))
            if tuners["meta"]["trial"] < k:
                # In progress trial number shown as k-1 completed often; show current.
                pass
            if not tuners["meta"]["total"] and meta_trials:
                tuners["meta"]["total"] = meta_trials
            # Represent in-progress as trial k of n (completed is k-1 until ✓).
            if tuners["meta"]["trial"] < k:
                tuners["meta"]["trial"] = max(tuners["meta"]["trial"], k - 1)

        if _RE_ELO_DONE.search(clean):
            tuners["elo"]["state"] = "done"
            if tuners["elo"]["total"] and not tuners["elo"]["trial"]:
                tuners["elo"]["trial"] = tuners["elo"]["total"]
            elif tuners["elo"]["total"]:
                tuners["elo"]["trial"] = tuners["elo"]["total"]
            m = _RE_BEST_CV.search(clean)
            # Best CV may be on following lines — handled below via scan
            if active_tuner == "elo":
                active_tuner = None

        if _RE_HIER_DONE.search(clean):
            tuners["hier"]["state"] = "done"
            if tuners["hier"]["total"]:
                tuners["hier"]["trial"] = tuners["hier"]["total"]
            if active_tuner == "hier":
                active_tuner = None

        if _RE_META_DONE.search(clean):
            tuners["meta"]["state"] = "done"
            if tuners["meta"]["total"]:
                tuners["meta"]["trial"] = tuners["meta"]["total"]
            m = _RE_BEST_SCORE.search(clean)
            if m:
                tuners["meta"]["best_mae"] = float(m.group(1))
            if active_tuner == "meta":
                active_tuner = None

        m = _RE_BEST_CV.search(clean)
        if m:
            # Attribute to last completed elo/hier.
            for key in ("hier", "elo"):
                if tuners[key]["state"] == "done" and tuners[key]["best_mae"] is None:
                    tuners[key]["best_mae"] = float(m.group(1))
                    break
            else:
                if active_tuner in ("elo", "hier"):
                    tuners[active_tuner]["best_mae"] = float(m.group(1))

        m = _RE_FEATURES.search(clean)
        if m:
            done, total = int(m.group(1)), max(1, int(m.group(2)))
            features_pct = 100.0 * done / total

        if _RE_SUITE_DONE.search(clean):
            suite_done = True
            current_stage = 8
            for s in range(9):
                stages_seen.add(s)

        m = _RE_WALL.search(clean)
        if m:
            wall_s = float(m.group(1))

    # Refine trial durations from ISO timestamps when present.
    trial_durations_ts = _trial_durations_from_timestamps(lines)
    if trial_durations_ts:
        trial_durations = trial_durations_ts

    rolling = None
    for d in trial_durations:
        rolling = _ema(rolling, d)
    last20 = _mean(trial_durations[-20:]) if trial_durations else None
    season_avg = _mean(season_durations) if season_durations else None
    slowest = max(trial_durations[-20:]) if trial_durations else None

    # Pipeline states.
    pipeline = []
    for step in PIPELINE_STEPS:
        sid = step["id"]
        if suite_done or sid < current_stage or sid in stages_seen and sid < current_stage:
            state = "done"
        elif sid == current_stage or (current_stage in (3, 4, 5) and sid in (3, 4, 5) and sid <= max(current_stage, 3)):
            # During walkforward mark 3..current as current/done.
            if current_stage in (3, 4, 5):
                if sid < 3:
                    state = "done"
                elif sid == 3 and active_tuner in ("elo", "hier", None) and tuners["meta"]["state"] != "running":
                    state = "current" if tuners["elo"]["state"] != "done" or tuners["hier"]["state"] == "running" or (
                        tuners["elo"]["state"] == "done" and tuners["hier"]["state"] != "done" and tuners["meta"]["state"] != "running"
                    ) else "done"
                    # simplify below
                else:
                    state = "upcoming"
            else:
                state = "current" if sid == current_stage else ("done" if sid < current_stage else "upcoming")
        else:
            state = "done" if sid < current_stage else "upcoming"
        pipeline.append({**step, "state": state})

    # Recompute pipeline more simply / reliably.
    pipeline = _pipeline_states(current_stage, stages_seen, suite_done, active_tuner, tuners)

    progress_pct = _compute_progress(
        current_stage=current_stage,
        suite_done=suite_done,
        season_index=season_index,
        season_total=season_total or max(1, len(seasons_list)),
        tuners=tuners,
        active_tuner=active_tuner,
        features_pct=features_pct,
    )

    headline = _headline(
        suite_done=suite_done,
        current_stage=current_stage,
        season_label=season_label,
        season_index=season_index,
        season_total=season_total,
        active_tuner=active_tuner,
        tuners=tuners,
        features_pct=features_pct,
    )
    short = _short_status(active_tuner, tuners, season_label, suite_done, current_stage)

    elapsed = elapsed_s if elapsed_s is not None else wall_s
    eta_s = _estimate_eta(
        progress_pct=progress_pct,
        elapsed_s=elapsed,
        trial_sec=last20 or rolling,
        active_tuner=active_tuner,
        tuners=tuners,
        season_index=season_index,
        season_total=season_total,
        season_avg=season_avg,
    )

    return {
        "headline": headline,
        "short_status": short,
        "progress_pct": round(progress_pct, 1),
        "pipeline": pipeline,
        "season": {
            "label": season_label,
            "index": season_index,
            "total": season_total or len(seasons_list) or 0,
            "years": seasons_list,
        },
        "tuners": tuners,
        "timing": {
            "elapsed_s": elapsed,
            "eta_s": eta_s,
            "trial_sec_rolling": round(rolling, 1) if rolling is not None else None,
            "trial_sec_last20": round(last20, 1) if last20 is not None else None,
            "season_sec_avg": round(season_avg, 1) if season_avg is not None else None,
            "slowest_trial_s": round(slowest, 1) if slowest is not None else None,
            "samples": len(trial_durations),
        },
        "features_pct": features_pct,
        "active_tuner": active_tuner,
        "current_stage": current_stage,
        "run_dir": run_dir,
        "suite_done": suite_done,
    }


def _trial_durations_from_timestamps(lines: list[str]) -> list[float]:
    """Extract durations between consecutive meta trial completions using log stamps."""
    from datetime import datetime

    stamps: list[datetime] = []
    prev_ts: datetime | None = None
    for line in lines:
        m = _RE_TS.search(line)
        if m:
            try:
                prev_ts = datetime.strptime(m.group(1), "%Y-%m-%dT%H:%M:%S")
            except Exception:
                prev_ts = None
        is_trial_done = ("✓" in line and "trial" in line.lower() and "score=" in line.lower())
        if is_trial_done:
            ts = None
            m2 = _RE_TS.search(line)
            if m2:
                try:
                    ts = datetime.strptime(m2.group(1), "%Y-%m-%dT%H:%M:%S")
                except Exception:
                    ts = None
            if ts is None:
                ts = prev_ts
            if ts is not None:
                stamps.append(ts)
    uniq: list[datetime] = []
    for s in stamps:
        if not uniq or (s - uniq[-1]).total_seconds() > 2:
            uniq.append(s)
    out: list[float] = []
    for a, b in zip(uniq, uniq[1:]):
        out.append(max(1.0, (b - a).total_seconds()))
    return out


def _pipeline_states(
    current_stage: int,
    stages_seen: set[int],
    suite_done: bool,
    active_tuner: str | None,
    tuners: dict[str, Any],
) -> list[dict[str, Any]]:
    out = []
    in_wf = current_stage in (3, 4, 5) or any(s in stages_seen for s in (3, 4, 5))
    for step in PIPELINE_STEPS:
        sid = step["id"]
        if suite_done:
            state = "done"
        elif sid < current_stage and not (in_wf and sid in (3, 4, 5) and current_stage in (3, 4, 5)):
            state = "done"
        elif current_stage >= 6:
            state = "done" if sid < current_stage else ("current" if sid == current_stage else "upcoming")
        elif in_wf and current_stage <= 5:
            # Walk-forward: Engines while elo/hier; Simulate while meta or after tuners.
            if sid < 3:
                state = "done"
            elif sid > 5:
                state = "upcoming"
            else:
                elo_h = active_tuner in ("elo", "hier") or tuners["elo"]["state"] == "running" or tuners["hier"]["state"] == "running"
                meta_r = active_tuner == "meta" or tuners["meta"]["state"] == "running"
                if elo_h:
                    state = "current" if sid == 3 else ("done" if sid < 3 else "upcoming")
                    if sid in (4, 5):
                        state = "upcoming"
                elif meta_r:
                    state = "done" if sid in (3, 4) else ("current" if sid == 5 else "upcoming")
                else:
                    # Between tuners / simulating games after meta done this season.
                    if tuners["meta"]["state"] == "done" or (
                        tuners["elo"]["state"] == "done" and tuners["hier"]["state"] == "done" and tuners["meta"]["state"] != "running"
                    ):
                        state = "done" if sid in (3, 4) else ("current" if sid == 5 else "upcoming")
                    else:
                        state = "current" if sid == 3 else ("done" if sid < 3 else "upcoming")
        elif sid == current_stage:
            state = "current"
        elif sid < current_stage:
            state = "done"
        else:
            state = "upcoming"
        if sid in stages_seen and sid < current_stage and sid not in (3, 4, 5):
            state = "done"
        out.append({**step, "state": state})
    currents = [p for p in out if p["state"] == "current"]
    if not currents and not suite_done:
        for p in out:
            if p["id"] == min(current_stage, 8):
                p["state"] = "current"
                break
    return out


def _compute_progress(
    *,
    current_stage: int,
    suite_done: bool,
    season_index: int,
    season_total: int,
    tuners: dict[str, Any],
    active_tuner: str | None,
    features_pct: float | None,
) -> float:
    if suite_done:
        return 100.0

    weights = SUITE_STAGE_WEIGHTS
    w0 = weights.get("0_toggles", 0.02)
    w1 = weights.get("1_stints", 0.15)
    w2 = weights.get("2_integrity", 0.05)
    w35 = weights.get("3_5_walkforward", 0.55)
    w6 = weights.get("6_review", 0.08)
    w7 = weights.get("7_policy", 0.10)
    w8 = weights.get("8_persist", 0.05)

    base = 0.0
    if current_stage > 0:
        base += w0
    if current_stage > 1:
        base += w1
    if current_stage > 2:
        base += w2

    if current_stage < 3:
        # Partial within early stages.
        if current_stage == 0:
            return max(1.0, 100.0 * w0 * 0.5)
        if current_stage == 1:
            return 100.0 * (w0 + w1 * 0.5)
        return 100.0 * (w0 + w1 + w2 * 0.5)

    if current_stage >= 6:
        base += w35
        if current_stage > 6:
            base += w6
        if current_stage > 7:
            base += w7
        if current_stage >= 8:
            base += w8 * 0.5
        if current_stage == 6:
            base += w6 * 0.4
        if current_stage == 7:
            base += w7 * 0.4
        return min(99.0, 100.0 * base)

    # Walk-forward fraction.
    st = max(1, season_total)
    si = max(0, min(season_index, st))
    # If we have started a season, use (si-1)/st + intra; if none yet, small bump.
    if si <= 0:
        season_frac = 0.02
    else:
        season_frac = (si - 1) / st
        intra = _tuner_fraction(tuners, active_tuner, features_pct)
        season_frac += intra / st

    return min(99.0, 100.0 * (base + w35 * min(1.0, season_frac)))


def _tuner_fraction(
    tuners: dict[str, Any],
    active_tuner: str | None,
    features_pct: float | None,
) -> float:
    """0..1 progress within a single season's train cycle."""
    # Weights within season: elo 0.25, hier 0.25, features 0.1, meta 0.4
    done = 0.0
    elo = tuners["elo"]
    hier = tuners["hier"]
    meta = tuners["meta"]

    def frac(t: dict[str, Any]) -> float:
        if t["state"] == "done":
            return 1.0
        tot = t.get("total") or 0
        tr = t.get("trial") or 0
        if tot > 0:
            return min(1.0, tr / tot)
        if t["state"] == "running":
            return 0.15
        return 0.0

    done += 0.25 * frac(elo)
    done += 0.25 * frac(hier)
    if features_pct is not None:
        done += 0.10 * min(1.0, features_pct / 100.0)
    elif meta["state"] in ("running", "done") or (elo["state"] == "done" and hier["state"] == "done"):
        done += 0.10
    done += 0.40 * frac(meta)
    return min(1.0, done)


def _headline(
    *,
    suite_done: bool,
    current_stage: int,
    season_label: str | None,
    season_index: int,
    season_total: int,
    active_tuner: str | None,
    tuners: dict[str, Any],
    features_pct: float | None,
) -> str:
    if suite_done:
        return "Suite finished — results are ready to explore."
    names = {
        0: "Setting up this run",
        1: "Loading play-by-play stints",
        2: "Running integrity checks",
        6: "Reviewing results (scoreboard & gates)",
        7: "Building betting policy grids",
        8: "Saving engines and reports",
    }
    if current_stage in names and current_stage not in (3, 4, 5):
        return names[current_stage] + "."

    season_bit = f" for season {season_label}" if season_label else ""
    if season_total and season_index:
        season_bit += f" ({season_index} of {season_total})"

    if features_pct is not None and active_tuner != "meta" and (
        tuners["hier"]["state"] == "done" or features_pct < 100
    ) and tuners["meta"]["state"] != "running":
        if features_pct < 99.5:
            return f"Building game features{season_bit} — {features_pct:.0f}%."

    if active_tuner == "elo" or tuners["elo"]["state"] == "running":
        t = tuners["elo"]
        if t["total"]:
            return f"Tuning player ratings (Elo){season_bit} — trial {t['trial']} of {t['total']}."
        return f"Tuning player ratings (Elo){season_bit}."
    if active_tuner == "hier" or tuners["hier"]["state"] == "running":
        t = tuners["hier"]
        if t["total"]:
            return f"Tuning the hierarchical engine{season_bit} — trial {t['trial']} of {t['total']}."
        return f"Tuning the hierarchical engine{season_bit}."
    if active_tuner == "meta" or tuners["meta"]["state"] == "running":
        t = tuners["meta"]
        # Show in-progress trial as next after completed.
        shown = t["trial"]
        if t["total"] and shown < t["total"] and t["state"] == "running":
            # Prefer "trial k of n" where k is completed+1 if we have fold lines.
            k = min(t["total"], max(1, shown + (0 if shown == 0 else 0)))
            if shown == 0:
                k = 1
            else:
                k = min(t["total"], shown)  # completed count from ✓ lines
            # If mid-trial, bump display.
            if shown > 0 and shown < t["total"]:
                return f"Tuning the margin model{season_bit} — trial {shown} of {t['total']} complete; continuing."
            return f"Tuning the margin model{season_bit} — trial {k} of {t['total']}."
        if t["total"]:
            return f"Tuning the margin model{season_bit} — trial {shown} of {t['total']}."
        return f"Tuning the margin model{season_bit}."

    if season_label:
        return f"Simulating games{season_bit}."
    if current_stage >= 3:
        return "Training models and simulating seasons…"
    return "Starting suite…"


def _short_status(
    active_tuner: str | None,
    tuners: dict[str, Any],
    season_label: str | None,
    suite_done: bool,
    current_stage: int,
) -> str:
    if suite_done:
        return "Done"
    season_tag = ""
    if season_label:
        # Prefer short S2024 from end year.
        parts = season_label.replace("–", "-").split("-")
        season_tag = f" · S{parts[-1]}" if parts else f" · {season_label}"
    if active_tuner == "meta" or tuners["meta"]["state"] == "running":
        t = tuners["meta"]
        if t["total"]:
            return f"Meta{season_tag} · {t['trial']}/{t['total']}"
        return f"Meta{season_tag}"
    if active_tuner == "hier" or tuners["hier"]["state"] == "running":
        t = tuners["hier"]
        if t["total"]:
            return f"Hier{season_tag} · {t['trial']}/{t['total']}"
        return f"Hier{season_tag}"
    if active_tuner == "elo" or tuners["elo"]["state"] == "running":
        t = tuners["elo"]
        if t["total"]:
            return f"Elo{season_tag} · {t['trial']}/{t['total']}"
        return f"Elo{season_tag}"
    labels = {
        0: "Setup",
        1: "Stints",
        2: "Checks",
        6: "Review",
        7: "Policy",
        8: "Save",
    }
    if current_stage in labels:
        return labels[current_stage]
    return f"Run{season_tag}" if season_tag else "Running"


def _estimate_eta(
    *,
    progress_pct: float,
    elapsed_s: float | None,
    trial_sec: float | None,
    active_tuner: str | None,
    tuners: dict[str, Any],
    season_index: int,
    season_total: int,
    season_avg: float | None,
) -> float | None:
    if elapsed_s and progress_pct > 3:
        remain_frac = max(0.0, (100.0 - progress_pct) / progress_pct)
        return max(0.0, elapsed_s * remain_frac)
    if trial_sec and active_tuner and tuners.get(active_tuner):
        t = tuners[active_tuner]
        left = max(0, (t.get("total") or 0) - (t.get("trial") or 0))
        return left * trial_sec
    if season_avg and season_total and season_index:
        left_seasons = max(0, season_total - season_index)
        return left_seasons * season_avg
    return None


def enrich_from_checkpoints(telemetry: dict[str, Any]) -> dict[str, Any]:
    """Merge suite checkpoint JSON when run_dir is known."""
    run_dir = telemetry.get("run_dir")
    if not run_dir:
        return telemetry
    root = Path(run_dir)
    if not root.is_dir():
        return telemetry

    manifest_p = root / "checkpoints" / "manifest.json"
    toggles_p = root / "checkpoints" / "runtime_toggles.json"
    if toggles_p.exists():
        try:
            toggles = json.loads(toggles_p.read_text())
            years = toggles.get("years") or toggles.get("seasons")
            if years and not telemetry["season"]["years"]:
                if isinstance(years, str):
                    years = [y.strip() for y in years.split(",") if y.strip()]
                telemetry["season"]["years"] = [str(y) for y in years]
                telemetry["season"]["total"] = len(telemetry["season"]["years"])
            for key, tkey in (("elo_trials", "elo"), ("hier_trials", "hier"), ("meta_trials", "meta")):
                if toggles.get(key) and not telemetry["tuners"][tkey]["total"]:
                    telemetry["tuners"][tkey]["total"] = int(toggles[key])
        except Exception:
            pass

    if manifest_p.exists():
        try:
            man = json.loads(manifest_p.read_text())
            stages = man.get("stages") or man
            # stages may be dict name -> {status}
            status_map: dict[int, str] = {}
            for step in PIPELINE_STEPS:
                # Try several key forms.
                for cand in (
                    f"{step['id']}_{step['key']}",
                    step["key"],
                    f"stage_{step['id']}",
                    str(step["id"]),
                ):
                    node = stages.get(cand) if isinstance(stages, dict) else None
                    if isinstance(node, dict) and node.get("status"):
                        status_map[step["id"]] = str(node["status"]).lower()
                        break
                    if isinstance(node, str):
                        status_map[step["id"]] = node.lower()
                        break
            if status_map:
                for p in telemetry["pipeline"]:
                    st = status_map.get(p["id"])
                    if st in ("ok", "done", "succeeded", "skipped"):
                        p["state"] = "done"
                    elif st in ("started", "running"):
                        if p["state"] != "done":
                            p["state"] = "current"
        except Exception:
            pass
    return telemetry


def build_job_telemetry(
    job_id: str,
    *,
    meta: dict[str, Any] | None = None,
    log_text: str | None = None,
    log_path: str | Path | None = None,
    elapsed_s: float | None = None,
) -> dict[str, Any]:
    """Build telemetry for a dashboard job id."""
    from dashboard.jobs import queue as job_queue

    if meta is None:
        meta = job_queue.load_job(job_id) or {}
    if log_text is None:
        path = Path(log_path or meta.get("log_path") or "")
        if path.is_file():
            log_text = path.read_text(errors="replace")
        else:
            log_text = ""

    # Elapsed from meta timestamps when possible.
    if elapsed_s is None and meta.get("started_at"):
        try:
            from datetime import datetime, timezone

            started = datetime.fromisoformat(str(meta["started_at"]).replace("Z", "+00:00"))
            now = datetime.now(timezone.utc)
            if started.tzinfo is None:
                started = started.replace(tzinfo=timezone.utc)
            elapsed_s = max(0.0, (now - started).total_seconds())
        except Exception:
            elapsed_s = None

    tel = parse_log_text(
        log_text or "",
        params=meta.get("params") or {},
        started_at=meta.get("started_at"),
        elapsed_s=elapsed_s,
    )
    tel = enrich_from_checkpoints(tel)
    tel["job_id"] = job_id
    tel["module"] = meta.get("module")
    tel["status"] = meta.get("status")
    return tel
