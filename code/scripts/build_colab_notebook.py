#!/usr/bin/env python3
"""Rebuild nba_unified_pipeline_colab.ipynb to import from the repo `code/` package.

Writes the notebook to the BasketballElo repo root (next to `code/`, `old code/`,
and `example data/`). The notebook never inlines pipeline modules — it adds
`code/` to ``sys.path`` and imports from the production package synced there.
"""
from __future__ import annotations

import json
from pathlib import Path

# Prefer writing into the GitHub clone when present; fall back to this repo root.
HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
GITHUB = ROOT / "BasketballElo"
OUT = (GITHUB if GITHUB.is_dir() else ROOT) / "nba_unified_pipeline_colab.ipynb"
BUILDER_OUT = (GITHUB if GITHUB.is_dir() else ROOT) / "code" / "scripts" / "build_colab_notebook.py"


def _md(text: str) -> dict:
    lines = text.split("\n")
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else []),
    }


def _code(text: str) -> dict:
    lines = text.split("\n")
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": [ln + "\n" for ln in lines[:-1]] + ([lines[-1] + "\n"] if lines else []),
    }


def build_notebook() -> dict:
    cells = []

    cells.append(_md("""# NBA Spread Prediction — Unified Pipeline (2026 integrity rebuild)

Import-based Colab/local notebook. **All production code lives in `code/`**
(`code/pipeline`, `code/tests`, `code/scripts`, `code/run_backtest.py`).

This rebuild reflects the integrity/evaluation repairs:
- Canonical final-score labels (no garbage-time zeroing of outcomes)
- Mixed-date parsing + fail-closed unresolved dates
- T−60 open/decision/close market snapshots (`code/pipeline/market_snapshots.py`)
- Past-only OOF stacking / HAPM / calibration-slice isolation
- Correct bet-side CLV, push handling, caps-before-profit, fractional Kelly

## Workflow
0. **Setup** — deps, Drive mount, put `code/` on `sys.path`
1. **Load** — stints + odds (`run_backtest.load_stints`)
2. **Integrity checks** — canonical games, leak registry summary
3. **Backtest** — walk-forward (`run_multi_year_backtest_walkforward`)
4. **Evaluation** — exact-price ROI profiles + diagnostics plots
5. **Predict** — live game via `predict_game` (when `state/` artifacts exist)

Feature flags: `pipeline/config.py` (`BET_SELECTION_MODE`, etc.).
Leak status: `code/LEAK_REGISTRY.md`."""))

    cells.append(_code("""# Install dependencies (Colab / fresh env)
!pip install -q catboost optuna scikit-learn pandas numpy tqdm matplotlib pyarrow lightgbm scipy"""))

    cells.append(_code("""import sys
from pathlib import Path

IN_COLAB = False
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    # Point this at the cloned BasketballElo folder on Drive
    REPO = Path("/content/drive/MyDrive/BasketballElo")
else:
    # Notebook lives at repo root next to code/
    REPO = Path(".").resolve()
    if not (REPO / "code" / "pipeline").is_dir():
        # Allow running from inside code/ or from a parent workspace
        for cand in (REPO / "BasketballElo", REPO.parent / "BasketballElo", REPO):
            if (cand / "code" / "pipeline").is_dir():
                REPO = cand.resolve()
                break

CODE = REPO / "code"
assert (CODE / "pipeline").is_dir(), f"Missing code/pipeline under {REPO}"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

print("Repo:", REPO)
print("Code:", CODE)
print("IN_COLAB:", IN_COLAB)
import pipeline
print("pipeline from:", Path(pipeline.__file__).resolve())"""))

    cells.append(_code("""# Runtime toggles
QUICK = False          # True → fewer seasons / faster smoke run
FAST_TUNING = False    # lighter Optuna CV
FAST_BACKTEST = FAST_TUNING
ELO_TRIALS = 5 if QUICK else 30
HIER_TRIALS = 5 if QUICK else 30
META_TRIALS = 8 if QUICK else 40
ROLLING_WINDOW = 3"""))

    cells.append(_code("""# Inspect / optionally override pipeline flags before backtest
import pipeline.config as cfg
print("BET_SELECTION_MODE:", cfg.BET_SELECTION_MODE)
print("CALIBRATION_MODE:", cfg.CALIBRATION_MODE)
print("STAKE_SIZING_MODE:", cfg.STAKE_SIZING_MODE)
print("FEATURE_SCHEMA_VERSION:", getattr(cfg, "FEATURE_SCHEMA_VERSION", "?"))
print("MARKET_SNAPSHOT_SCHEMA_VERSION:", getattr(cfg, "MARKET_SNAPSHOT_SCHEMA_VERSION", "?"))
# cfg.BET_SELECTION_MODE = "edge_bucket"  # only after ablation gate passes"""))

    cells.append(_code("""# Integrity snapshot (canonical games + leak registry)
from pipeline.game_results import build_canonical_games
from pipeline.invalid_baseline import write_invalid_baseline_manifest
import pandas as pd

print("--- Leak registry (first 40 lines) ---")
reg = CODE / "LEAK_REGISTRY.md"
if reg.exists():
    print("\\n".join(reg.read_text().splitlines()[:40]))
else:
    print("LEAK_REGISTRY.md not found")

# Optional: rebuild invalid-baseline manifest from quarantined newest-data exports
# if those CSVs are present next to the notebook.
newest = REPO / "newest data"
if not newest.is_dir():
    newest = REPO.parent / "newest data"
print("Newest-data dir:", newest if newest.is_dir() else "(absent)")"""))

    cells.append(_code("""from run_backtest import load_stints
from pipeline.market import load_modern_odds, load_pinnacle_lines
from pipeline.config import MODERN_ODDS_PATH, PINNACLE_LINES_PATH
from pipeline.diagnostics import diagnose_loaded_data, run_backtest_diagnostics
from pipeline.metrics import (
    benchmark_results,
    benchmark_betting_roi,
    add_all_profile_columns,
    add_clv_columns,
)
from pipeline.backtest import run_multi_year_backtest_walkforward
from pipeline.diagnostics import generate_betting_plots

stints = load_stints(quick=QUICK)

# Prefer example data shipped with the GitHub repo, then config paths, then parent workspace
example = REPO / "example data"
odds_candidates = [
    example / "all_odds.csv",
    Path(MODERN_ODDS_PATH) if MODERN_ODDS_PATH else None,
    REPO / "all_odds.csv",
    REPO.parent / "all_odds.csv",
]
odds_path = next((p for p in odds_candidates if p is not None and Path(p).exists()), None)
assert odds_path is not None, "all_odds.csv not found"
odds_dict = load_modern_odds(str(odds_path))
print("Odds source:", odds_path, "games keys:", len(odds_dict))

pin_candidates = [
    example / "nba_main_lines.csv",
    Path(PINNACLE_LINES_PATH) if PINNACLE_LINES_PATH else None,
    REPO / "nba_main_lines.csv",
    REPO.parent / "nba_main_lines.csv",
]
pin_path = next((p for p in pin_candidates if p is not None and Path(p).exists()), None)
if pin_path is not None:
    sched = (
        stints[stints["season"] == 2026][["GAME_ID", "game_date", "home_team", "away_team"]]
        .drop_duplicates("GAME_ID")
        .rename(columns={"game_date": "date", "home_team": "home", "away_team": "away"})
    )
    pin = load_pinnacle_lines(str(pin_path), schedule_df=sched if not sched.empty else None)
    odds_dict.update(pin)
    print(f"Merged {len(pin)} Pinnacle lines from {pin_path}")

diagnose_loaded_data(stints, odds_dict)"""))

    cells.append(_code("""results = run_multi_year_backtest_walkforward(
    stints,
    odds_dict=odds_dict,
    n_tuning_trials_elo=ELO_TRIALS,
    n_tuning_trials_hier=HIER_TRIALS,
    n_tuning_trials_meta=META_TRIALS,
    rolling_window_size=ROLLING_WINDOW,
    fast_tuning=FAST_BACKTEST,
)
print(f"Results: {len(results):,} games")
benchmark_results(results)

# Exact-price / capped-stake evaluation (Tasks 037-047)
results = add_clv_columns(results)
results = add_all_profile_columns(results)
for p in ("conservative", "moderate", "aggressive"):
    benchmark_betting_roi(results, profile=p)

out_csv = REPO / "backtest_results.csv"
results.to_csv(out_csv, index=False)
print("Saved", out_csv)"""))

    cells.append(_code("""diag_dir = REPO / "analysis_plots"
diag_dir.mkdir(parents=True, exist_ok=True)
run_backtest_diagnostics(results, save_dir=diag_dir, show_graphs=True)
generate_betting_plots(results, save_dir=diag_dir)
print("Plots →", diag_dir)"""))

    cells.append(_code("""# Post-hoc ablation (edge_bucket vs baseline on saved export)
from pipeline.ablation_posthoc import posthoc_ablation_summary
from pipeline.ablation import passes_ablation_gate

summary = posthoc_ablation_summary(results)
display(summary)
base = summary[summary["config"] == "baseline_current"].iloc[0].to_dict()
edge = summary[summary["config"] == "edge_bucket"].iloc[0].to_dict()
print("Ablation gate:", passes_ablation_gate(base, edge))"""))

    cells.append(_code("""# Live prediction (loads state/*.pkl when present)
from pipeline.predict import predict_game, PredictionContext
from pipeline.ratings import PlayerRatingTracker
from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.trackers import PaceTracker, TeamXpppTracker
from pipeline.teamstats import TeamFormTracker
from pipeline.model import MetaScoreModel
from pipeline.market import SpreadCalibrator

state_dir = REPO / "state"
state_dir.mkdir(parents=True, exist_ok=True)
prefix = state_dir / "latest"
meta_path = prefix.with_name(prefix.name + "_meta.pkl")

if meta_path.exists():
    meta = MetaScoreModel.load(meta_path)
    elo = PlayerRatingTracker.load_state(prefix.with_name(prefix.name + "_elo.pkl"))
    hier = HierarchicalPossessionEngine.load_state(prefix.with_name(prefix.name + "_hier.pkl"))
    pace_path = prefix.with_name(prefix.name + "_pace.pkl")
    pace = PaceTracker.load_state(pace_path) if pace_path.exists() else PaceTracker()
    ctx_path = state_dir / "predict_context.pkl"
    ctx = PredictionContext.load_state(ctx_path) if ctx_path.exists() else PredictionContext()
    print("Models loaded from", state_dir)
    print("Call predict_game(home, away, date, hier, elo, meta, pace, ...) with your matchup.")
else:
    print("No trained artifacts in", state_dir, "- run the backtest/train phases first.")"""))

    cells.append(_md("""## Notes
- Prefer **T−60 decision quotes** from `pipeline.market_snapshots` for new residual/CLV work; the legacy `load_pinnacle_lines` path may still conflate bet line with close until Stage 7 rewiring.
- Do not treat quarantined `newest data/` exports as a valid accuracy baseline — see `code/LEAK_REGISTRY.md`.
- Rebuild this notebook after code changes: `python code/scripts/build_colab_notebook.py`"""))

    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3",
            },
            "language_info": {"name": "python"},
        },
        "cells": cells,
    }


def main() -> None:
    nb = build_notebook()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(nb, indent=1))
    print(f"Wrote {OUT} ({len(nb['cells'])} cells)")


if __name__ == "__main__":
    main()
