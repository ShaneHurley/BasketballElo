"""Host-side live pipeline runs (ablation / calib / watchdog) for the dashboard.

Reads ``output/logs/live_status.json`` and optionally refreshes aliveness +
log progress so the website stays current even when an external poller is slow.
"""
from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from dashboard.config import OUTPUT_ROOT, REPO_ROOT

LOGS_DIR = OUTPUT_ROOT / "logs"
STATUS_PATH = LOGS_DIR / "live_status.json"
STATUS_MD = LOGS_DIR / "LIVE_RUN_STATUS.md"

_PIDFILES = {
    "ablation": LOGS_DIR / "ablation_8_4.pid",
    "watchdog": LOGS_DIR / "ablation_then_calib_watch.pid",
    "calib": LOGS_DIR / "calib_hold_mae_v9.pid",
}

_PGREP_PATTERNS = (
    "run_rapm_hapm_ablation",
    "run_full_suite",
    "safe_ablation_then_calib_watchdog",
    "calib_hold_mae",
)

_HINT_RE = re.compile(
    r"(✓\s*trial\s+\d+|total trial\s+\d+|Best trial:|Tuning total|Tuning MetaScore|"
    r"Tuning Elo|Tuning Hierarchical|SIMULATING SEASON|ABLATION CONFIG|"
    r"Prune decision|Wrote |Stage\s+[0-8]|Suite complete)",
    re.I,
)


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        pid_i = int(pid)
    except (TypeError, ValueError):
        return False
    try:
        os.kill(pid_i, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # Process exists but we cannot signal it — still count as alive.
        return True
    except OSError:
        # Fall back to ps(1) when kill(2) is restricted (some sandboxes).
        try:
            proc = subprocess.run(
                ["ps", "-p", str(pid_i), "-o", "pid="],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            return bool((proc.stdout or "").strip())
        except (OSError, subprocess.TimeoutExpired):
            return False


def _read_pidfile(path: Path) -> int | None:
    if not path.is_file():
        return None
    try:
        return int(path.read_text().strip())
    except (OSError, ValueError):
        return None


def _rel(path: str | Path | None) -> str | None:
    if not path:
        return None
    p = Path(path)
    try:
        return str(p.relative_to(REPO_ROOT))
    except ValueError:
        return str(p)


def _mtime_iso(path: Path) -> str | None:
    if not path.is_file():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, timezone.utc).isoformat()


def _age_seconds(path: Path) -> float | None:
    if not path.is_file():
        return None
    return max(0.0, datetime.now(timezone.utc).timestamp() - path.stat().st_mtime)


def _tail_hint(log_path: Path, *, max_lines: int = 8000) -> str:
    if not log_path.is_file():
        return ""
    try:
        text = log_path.read_text(errors="replace").replace("\r", "\n")
    except OSError:
        return ""
    lines = text.splitlines()
    window = lines[-max_lines:] if len(lines) > max_lines else lines
    for line in reversed(window):
        s = line.strip()
        if not s or len(s) < 8:
            continue
        # Skip pandas PerformanceWarning spam that floods the log tail.
        if "PerformanceWarning" in s or "DataFrame is highly fragmented" in s:
            continue
        if s.startswith("X[c]") or "fillna(" in s:
            continue
        if _HINT_RE.search(s):
            return s[:200]
    for line in reversed(window[-80:]):
        s = line.strip()
        if (
            s
            and len(s) < 180
            and "PerformanceWarning" not in s
            and not s.startswith("X[c]")
        ):
            return s[:200]
    return ""


def _pgrep_snapshot() -> list[dict[str, Any]]:
    """Best-effort process list; empty on sandbox / permission failures."""
    out: list[dict[str, Any]] = []
    try:
        proc = subprocess.run(
            ["pgrep", "-fl", "|".join(_PGREP_PATTERNS)],
            capture_output=True,
            text=True,
            timeout=3,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return out
    for line in (proc.stdout or "").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if not parts:
            continue
        try:
            pid = int(parts[0])
        except ValueError:
            continue
        cmd = parts[1] if len(parts) > 1 else ""
        low = cmd.lower()
        # Ignore shells / pollers / editors that merely mention the script names.
        if any(tok in low for tok in ("/bin/zsh", "/bin/bash", "/bin/sh", "pgrep", "rg ", "cursor")):
            continue
        if "python" not in low and "python3" not in low:
            continue
        role = "other"
        if "run_rapm_hapm_ablation" in low:
            role = "ablation"
        elif "safe_ablation_then_calib" in low:
            role = "watchdog"
        elif "run_full_suite" in low or "calib_hold_mae" in low:
            role = "calib"
        else:
            continue
        out.append({"pid": pid, "role": role, "command": cmd[:240], "alive": _alive(pid)})
    return out


def _latest_log(glob_pat: str) -> Path | None:
    matches = sorted(LOGS_DIR.glob(glob_pat), key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0] if matches else None


def _load_status_file() -> dict[str, Any]:
    if not STATUS_PATH.is_file():
        return {}
    try:
        data = json.loads(STATUS_PATH.read_text())
        return data if isinstance(data, dict) else {}
    except (OSError, json.JSONDecodeError):
        return {}


def refresh_live_status(*, write: bool = True) -> dict[str, Any]:
    """Merge pidfiles + pgrep + log tails into a fresh status payload."""
    base = _load_status_file()
    processes = _pgrep_snapshot()

    ablation_pid = _read_pidfile(_PIDFILES["ablation"]) or base.get("ablation_pid")
    watchdog_pid = _read_pidfile(_PIDFILES["watchdog"]) or base.get("watchdog_pid")
    calib_pid = _read_pidfile(_PIDFILES["calib"]) or base.get("calib_pid")

    # Prefer live pgrep when pidfile is stale — but keep a healthy pidfile PID.
    for row in processes:
        if not row.get("alive"):
            continue
        if row["role"] == "ablation":
            if not ablation_pid or not _alive(ablation_pid):
                ablation_pid = row["pid"]
        elif row["role"] == "watchdog":
            if not watchdog_pid or not _alive(watchdog_pid):
                watchdog_pid = row["pid"]
        elif row["role"] == "calib":
            if not calib_pid or not _alive(calib_pid):
                calib_pid = row["pid"]

    ablation_log = Path(base["log"]) if base.get("log") else _latest_log("ablation_8_4_*.log")
    if ablation_log is None:
        ablation_log = LOGS_DIR / "ablation_8_4_20260921_171345.log"
    watchdog_log = (
        Path(base["watchdog_log"])
        if base.get("watchdog_log")
        else _latest_log("ablation_then_calib_watch_*.log")
    )
    calib_log = _latest_log("calib_hold_mae_v9_*.log")

    run_dir = Path(base["run_dir"]) if base.get("run_dir") else None
    if run_dir is None:
        cands = sorted(
            OUTPUT_ROOT.glob("*ablation_8_4*"),
            key=lambda p: p.stat().st_mtime if p.exists() else 0,
            reverse=True,
        )
        run_dir = next((p for p in cands if p.is_dir()), None)

    report_exists = False
    prune_exists = (OUTPUT_ROOT / "baselines" / "ablation_8_4_prune_decision.json").exists()
    if run_dir and run_dir.is_dir():
        report_exists = (run_dir / "ablation_8_4_report.json").exists()

    last_trial = ""
    if ablation_log and ablation_log.is_file():
        last_trial = _tail_hint(ablation_log)
    if calib_pid and _alive(calib_pid) and calib_log:
        last_trial = _tail_hint(calib_log) or last_trial

    abl_alive = _alive(ablation_pid)
    watch_alive = _alive(watchdog_pid)
    cal_alive = _alive(calib_pid)

    if cal_alive:
        progress = f"calib_hold_mae_v9 running pid={calib_pid}"
    elif abl_alive:
        progress = "ablation running; watchdog waiting" if watch_alive else "ablation running; no watchdog"
    elif report_exists and prune_exists and not cal_alive:
        progress = "ablation done; waiting for calib" if watch_alive else "ablation done; watchdog missing"
    elif not abl_alive and not cal_alive:
        progress = base.get("progress") or "idle"
    else:
        progress = base.get("progress") or "unknown"

    payload: dict[str, Any] = {
        "at": _utc_now(),
        "ablation_pid": ablation_pid,
        "ablation_alive": abl_alive,
        "watchdog_pid": watchdog_pid,
        "watchdog_alive": watch_alive,
        "calib_pid": calib_pid,
        "calib_alive": cal_alive,
        "log": str(ablation_log) if ablation_log else None,
        "watchdog_log": str(watchdog_log) if watchdog_log else None,
        "calib_log": str(calib_log) if calib_log else None,
        "run_dir": str(run_dir) if run_dir else None,
        "report_exists": report_exists,
        "prune_exists": prune_exists,
        "last_trial": last_trial or base.get("last_trial") or "",
        "progress": progress,
        "log_mtime": _mtime_iso(ablation_log) if ablation_log else None,
        "log_age_s": round(_age_seconds(ablation_log), 1) if ablation_log else None,
        "arms_seen": base.get("arms_seen"),
    }
    if write:
        try:
            LOGS_DIR.mkdir(parents=True, exist_ok=True)
            STATUS_PATH.write_text(json.dumps(payload, indent=2))
        except OSError:
            pass
    return payload


def get_live_runs(*, refresh: bool = True) -> dict[str, Any]:
    """API payload for ``/api/live_runs``."""
    status = refresh_live_status(write=refresh) if refresh else (_load_status_file() or refresh_live_status(write=False))
    processes = _pgrep_snapshot()

    def role_row(
        role: str,
        pid: Any,
        alive: bool,
        command: str,
        log_path: str | None,
    ) -> dict[str, Any]:
        return {
            "role": role,
            "pid": pid,
            "alive": bool(alive),
            "command": command,
            "log": _rel(log_path),
            "log_abs": log_path,
            "log_mtime": _mtime_iso(Path(log_path)) if log_path else None,
            "log_age_s": round(_age_seconds(Path(log_path)), 1) if log_path else None,
        }

    # Match commands from pgrep when possible.
    cmd_by_role = {r["role"]: r["command"] for r in processes if r.get("alive")}
    runs = [
        role_row(
            "ablation",
            status.get("ablation_pid"),
            bool(status.get("ablation_alive")),
            cmd_by_role.get("ablation", "run_rapm_hapm_ablation.py"),
            status.get("log"),
        ),
        role_row(
            "watchdog",
            status.get("watchdog_pid"),
            bool(status.get("watchdog_alive")),
            cmd_by_role.get("watchdog", "safe_ablation_then_calib_watchdog.py"),
            status.get("watchdog_log"),
        ),
        role_row(
            "calib",
            status.get("calib_pid"),
            bool(status.get("calib_alive")),
            cmd_by_role.get("calib", "run_full_suite.py (calib_hold_mae_v9)"),
            status.get("calib_log"),
        ),
    ]

    heartbeat = status.get("at")
    heartbeat_age_s = None
    if heartbeat:
        try:
            hb = datetime.fromisoformat(str(heartbeat).replace("Z", "+00:00"))
            if hb.tzinfo is None:
                hb = hb.replace(tzinfo=timezone.utc)
            heartbeat_age_s = round((datetime.now(timezone.utc) - hb).total_seconds(), 1)
        except ValueError:
            heartbeat_age_s = None

    return {
        "ok": True,
        "heartbeat_at": heartbeat,
        "heartbeat_age_s": heartbeat_age_s,
        "progress": status.get("progress"),
        "last_trial": status.get("last_trial"),
        "report_exists": status.get("report_exists"),
        "prune_exists": status.get("prune_exists"),
        "run_dir": _rel(status.get("run_dir")),
        "status_json": _rel(STATUS_PATH),
        "status_md": _rel(STATUS_MD) if STATUS_MD.exists() else None,
        "runs": runs,
        "processes": processes,
        "raw": {
            k: status.get(k)
            for k in (
                "ablation_pid",
                "ablation_alive",
                "watchdog_pid",
                "watchdog_alive",
                "calib_pid",
                "calib_alive",
                "log_mtime",
                "log_age_s",
                "arms_seen",
            )
        },
    }
