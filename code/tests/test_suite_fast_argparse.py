"""FAST research preset: argparse knobs only (no suite run)."""
from __future__ import annotations

import sys
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent.parent
if str(CODE_ROOT) not in sys.path:
    sys.path.insert(0, str(CODE_ROOT))

import run_full_suite as suite


def test_preset_fast_forces_trials_and_stops_before_persist():
    args = suite.parse_suite_args(["--preset", "fast"])
    assert args.elo_trials == 3
    assert args.hier_trials == 3
    assert args.meta_trials == 3
    assert args.to_stage <= 6
    assert args.allow_tip_proxy is True
    assert args.fast_preset is True


def test_preset_fast_help_documents_research_only():
    help_text = suite.build_parser().format_help()
    assert "--preset" in help_text
    assert "research-only" in help_text.lower()
    assert "config.py" in help_text
