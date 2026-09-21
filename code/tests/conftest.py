"""Pytest config for the BasketballElo test suite."""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_REPORT_PATH = REPO_ROOT / "output" / "bench" / "latest.json"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--regenerate-golden",
        action="store_true",
        default=False,
        help=(
            "Rewrite the committed Epic 10.5 golden-master snapshot "
            "(tests/synth/golden_snapshots/golden_season.json) instead of "
            "comparing against it. Pair with a FEATURE_SCHEMA_VERSION bump."
        ),
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "bench: synthetic formula test bench (Epic 10) — extreme/adversarial inputs",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    config._bench_nodeids = {  # type: ignore[attr-defined]
        item.nodeid for item in items if item.get_closest_marker("bench")
    }


def pytest_sessionfinish(session: pytest.Session, exitstatus: int) -> None:
    """Write output/bench/latest.json only when this session collected bench tests."""
    bench_nodeids = getattr(session.config, "_bench_nodeids", set())
    if not bench_nodeids:
        return
    reporter = session.config.pluginmanager.get_plugin("terminalreporter")
    if reporter is None:
        return

    per_file: dict[str, dict[str, int]] = defaultdict(
        lambda: {"passed": 0, "failed": 0, "xfailed": 0, "xpassed": 0, "skipped": 0, "error": 0}
    )
    outcome_keys = ("passed", "failed", "xfailed", "xpassed", "skipped", "error")
    for outcome in outcome_keys:
        for rep in reporter.stats.get(outcome, []):
            nodeid = getattr(rep, "nodeid", "")
            if nodeid not in bench_nodeids:
                continue
            fpath = nodeid.split("::", 1)[0]
            per_file[fpath][outcome] += 1

    totals = {k: 0 for k in outcome_keys}
    for counts in per_file.values():
        for k in outcome_keys:
            totals[k] += counts[k]

    payload = {
        "exitstatus": int(exitstatus),
        "totals": totals,
        "files": dict(per_file),
    }
    BENCH_REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    BENCH_REPORT_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True))
