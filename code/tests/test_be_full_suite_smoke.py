"""Suite stage smoke: stage 0 writes toggles; stage 1 dry path with mock stints."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch

import pandas as pd

CODE_ROOT = Path(__file__).resolve().parent.parent
SUITE = CODE_ROOT / "run_full_suite.py"


def test_suite_script_exists():
    assert SUITE.exists()


def test_suite_stage0_writes_toggles(tmp_path):
    out = tmp_path / "output"
    out.mkdir()
    env = {
        **dict(**{k: v for k, v in __import__("os").environ.items()}),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(CODE_ROOT),
    }
    proc = subprocess.run(
        [
            sys.executable, str(SUITE),
            "--stage", "0",
            "--yes",
            "--output-root", str(out),
            "--label", "smoke",
            "--window", "4",
        ],
        cwd=str(CODE_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stdout + "\n" + proc.stderr
    runs = list(out.glob("*_smoke"))
    assert runs, "expected a suite run directory"
    toggles = runs[0] / "checkpoints" / "runtime_toggles.json"
    assert toggles.exists()
    data = json.loads(toggles.read_text())
    assert data["window"] == 4
    assert data["never_use_newest_data_baseline"] is True
    manifest = json.loads((runs[0] / "checkpoints" / "manifest.json").read_text())
    assert "0" in manifest.get("stages", {})


def test_suite_stage1_dry_path_prints_built_at(tmp_path, monkeypatch):
    """Stage 0 then from-stage 1 with mocked load_stints (no real PBP rebuild)."""
    if str(CODE_ROOT) not in sys.path:
        sys.path.insert(0, str(CODE_ROOT))
    import run_full_suite as suite

    out = tmp_path / "output"
    out.mkdir()
    # Stage 0
    env = {
        **dict(**{k: v for k, v in __import__("os").environ.items()}),
        "OPENBLAS_NUM_THREADS": "1",
        "OMP_NUM_THREADS": "1",
        "PYTHONPATH": str(CODE_ROOT),
    }
    proc = subprocess.run(
        [
            sys.executable, str(SUITE),
            "--stage", "0", "--yes",
            "--output-root", str(out),
            "--label", "stage1dry",
            "--window", "4",
        ],
        cwd=str(CODE_ROOT),
        env=env,
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    run_dir = next(out.glob("*_stage1dry"))

    fake = tmp_path / "events_2023_24_pbp_V3.csv"
    fake.write_text("x\n1\n")
    fake_stints = pd.DataFrame({
        "GAME_ID": ["0022300001"],
        "game_date": [pd.Timestamp("2023-10-24")],
        "home_team": ["BOS"],
        "away_team": ["NYK"],
        "season": [2024],
    })
    built_at = "2026-07-30T12:00:00+00:00"
    meta = {
        "built_at": built_at,
        "cache_key": "abc123abc123abcd",
        "cache_hit": True,
        "cache_path": str(tmp_path / "stints_cache_abc.pkl"),
        "n_rows": 1,
        "paths_used": [],
    }

    args = suite.build_parser().parse_args([
        "--from-stage", "1", "--to-stage", "1", "--yes",
        "--stints", "auto",
        "--run-dir", str(run_dir),
        "--output-root", str(out),
        "--years", "2023",
    ])
    # Ensure toggles exist (stage0 wrote them)
    toggles = suite._read_json(run_dir / "checkpoints" / "runtime_toggles.json")
    toggles["years"] = [2023]
    toggles["mini"] = True
    suite._write_json(run_dir / "checkpoints" / "runtime_toggles.json", toggles)

    with patch("pipeline.data_paths.discover_pbp_paths", return_value={2023: fake}), \
         patch("pipeline.data_paths.select_season_keys", return_value=[2023]), \
         patch("pipeline.data_paths.resolve_data_dir", return_value=tmp_path), \
         patch("pipeline.data_paths.load_player_names", return_value=({}, {})), \
         patch("pipeline.data_paths.resolve_odds_path", return_value=None), \
         patch("pipeline.data_paths.resolve_pinnacle_path", return_value=None), \
         patch("pipeline.stint_loader.resolve_paths_for_stints", return_value=([(2023, fake)], [])), \
         patch("pipeline.stint_loader.load_stints", return_value=(fake_stints, meta)):
        stints, info = suite.stage1_stints(args, run_dir, toggles)

    assert len(stints) == 1
    assert info["built_at"] == built_at
    info_path = run_dir / "checkpoints" / "stints_info.json"
    assert info_path.exists()
    saved = json.loads(info_path.read_text())
    assert saved["built_at"] == built_at
    assert (run_dir / "artifacts" / "suite_stints.pkl").exists()
