#!/usr/bin/env python3
"""Rebuild nba_unified_pipeline_colab.ipynb with ALL pipeline modules inlined for Colab."""
from __future__ import annotations

import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # BasketballElo/code
PIPE = ROOT / "pipeline"
# Write package-local + parent BasketballElo/ + repo-root twins
BE_ROOT = ROOT.parent  # BasketballElo/
REPO_ROOT = BE_ROOT.parent
OUT = BE_ROOT / "nba_unified_pipeline_colab.ipynb"
OUT_CODE = ROOT / "nba_unified_pipeline_colab.ipynb"
OUT_REPO = REPO_ROOT / "nba_unified_pipeline_colab.ipynb"
WORKFLOW_SRC = REPO_ROOT / "a new day" / "working code" / "nba_unified_pipeline_colab.ipynb"

MODULE_ORDER = [
    "utils", "shot_zones", "ingest", "preprocess", "stints", "dates", "game_results",
    "data_paths", "stint_loader", "dataset_roles",
    "ratings", "hierarchical", "trackers", "teamstats", "shot_quality",
    "skellam", "market", "feature_utils", "availability", "epm_priors",
    "travel", "fatigue", "team_elo", "lineup_elo", "chemistry",
    "lineup_composite", "hapm", "refs", "calibrators", "stint_context",
    "game_updates", "elo_calibration", "cv", "venn_abers", "bet_confidence",
    "bet_selection", "artifacts", "market_targets", "market_disagreement",
    "market_snapshots", "devig", "bet_grading", "stake_profiles",
    "oof", "model", "metrics", "calibration_metrics", "ml_calibration",
    "game_features", "features",
    "tuning", "tuning_cache", "calibration_policy", "calibration_registry",
    "ats_classifier", "upset_classifier",
    "team_volatility", "scenario_mixer", "rotation_scenarios", "minutes_forecast",
    "injury_reports", "simulate", "predict", "backtest",
    "ablation", "ablation_posthoc", "edge_analysis", "diagnostics",
    "ml_diagnostics", "confidence_diagnostics", "validate_plan",
    "invalid_baseline", "benchmark_lock", "experiment_exports",
    "monitoring", "shap_prune",
]

MARKDOWN_INTRO = """# NBA Spread Prediction — Unified Colab Pipeline

Self-contained, **leak-free** spread model (player ELO + hierarchical possessions).
Runs end-to-end in Google Colab against your `basketballData` Drive folder.

**Suite alignment:** Local CLI stages live in `BasketballElo/code/run_full_suite.py`
(stages 0–8). Notebook phases map as: Phase 0↔stage0, Phase 1↔stage1,
Phase 2↔stages 3–5, Phase 2a–2e↔stage6 review/diagnostics, Phase 3–4↔stage8 persist,
Phase 5↔daily. Prefer **MAE / ECE / CLV** over ATS when reviewing. Never train from
quarantined `newest data/` baselines.

Each phase **saves artifacts to disk** and can **reload them** on the next session
(set `USE_SAVED_ARTIFACTS = False` to always recompute, or `FORCE_RECOMPUTE["<phase>"] = True` for one phase).

---

## Table of contents

| Step | Section | Saved artifacts |
|------|---------|-----------------|
| — | [Configuration & Drive](#configuration--google-drive-mount) | — |
| — | [Checkpoint helpers](#checkpoint-helpers-save--load) | `checkpoints/` manifest |
| — | [Pipeline modules](#pipeline-source-inlined-modules--complete) | — |
| 0 | [Runtime toggles](#phase-0--runtime-toggles) | `checkpoints/runtime_toggles.json` |
| 1 | [Load data](#phase-1--load-data) | stints cache, `odds_dict.pkl`, `load_summary.csv` |
| 2 | [Walk-forward backtest](#phase-2--walk-forward-backtest) | `backtest_results.csv`, `state/tuning_results.json` |
| 2a | [Unified confidence diagnostics](#phase-2a--unified-confidence-diagnostics) | Inline tuning in backtest; plots in `analysis_plots/14–16_*.png` |
| 2b | [Spread diagnostics + ablation](#phase-2b--spread-diagnostics--ablation) | `diag_summary.csv`, `diag_edges.pkl`, `ablation_posthoc.json` |
| 2c | [Bankroll & confidence plots](#phase-2c--bankroll--confidence-plots) | `analysis_plots/*.png` |
| 2d | [Moneyline diagnostics](#phase-2d--moneyline-diagnostics-metawin-calibration) | `ml_diagnostics_summary.csv`, `13_ml_winprob_reliability.png` |
| 2e | [ML value tracker](#phase-2e--ml-underdog--favorite-value-tracker) | `ml_value_plays.csv` |
| 3 | [ML + Total + Score-pair](#phase-3--dedicated-ml--total--score-pair-models) | `state/win.pkl`, `state/total.pkl`, `state/score_pair.pkl`, `checkpoints/final_feats.parquet` |
| 4 | [Spread / final engines](#phase-4--spread--final-engines) | `state/meta.pkl`, `state/*.pkl` |
| 5 | [Daily prediction](#phase-5--daily-prediction) | `checkpoints/last_prediction.json` |

---

## Key improvements (2026 pipeline)
- **Tuning speed (no accuracy cut)** — Hier O(1) 5-man weights, combo cache, fused predict+update, incremental season CV, process-parallel Optuna (`OPTUNA_N_JOBS`)
- **Specialized markets** — separate WIN vs TOTAL feature matrices; Elo de-emphasized on totals
- **Score-pair model** — independent home/away score heads → total + margin + σ
- **Gaussian O/U** — `P(Over|line)` from `N(pred_total, σ²)` with CRPS in diagnostics
- **Both-sides ML EV** — predict winner, price home *and* away, bet only if +EV else Pass
- **Multi-window form** — 3/5/10/20/season rolling + nonlinear rest buckets
- **edge_bucket primary selection** — `|edge| ≥ 5.5` is the lean filter; confidence is secondary
- **Hard confidence floor** — `MIN_CONFIDENCE_SCORE=55`; adaptive gate may raise up to +8, never lower
- **Avoid bands** — skip `|edge|` in `(3–4)` and `(5.5–6)` without Elo agreement
- **Phase 2a weight clamps** — `cover_scale` capped so confidence cannot explode/clump
- **Continuous confidence ranking** — logistic/heuristic score; isotonic only for cover probs/stakes
- **Tier-2 stake kill** — confidence tier 2 stake multiplier = 0
- **Favorite ML shrink** — extra shrink when `|spread|≥8` or fav win% ≥ 0.75; ECE gate keeps raw if cal worse
- **Upset / blowout heads** — `UpsetClassifier` + MetaScore `P(|m|≥10/15/20)` wired into EV/stakes
- **ATS classifier blend** — always mixed into cover prob before confidence calibration
- **Season-1 calibrator** — fit on training calib split before first simulated season
- **Selection comparison** — `compare_selection_strategies()` after backtest (hybrid, edge-scaled, edge-only)
- **Artifact schema validation** — stale `backtest_results.csv` detected on reload
- All modules inlined below — **no external repo required** on Colab"""


def _lines(source: str) -> list[str]:
    return [ln + "\n" for ln in source.splitlines()]


def _md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": _lines(text)}


def _code(text: str) -> dict:
    return {
        "cell_type": "code",
        "metadata": {},
        "outputs": [],
        "execution_count": None,
        "source": _lines(text.rstrip("\n")),
    }


def _strip_pipeline_imports(src: str) -> str:
    """Remove cross-module pipeline imports; notebook runs in flat namespace."""
    out: list[str] = []
    skip = False
    for line in src.splitlines():
        stripped = line.strip()
        if skip:
            if ")" in line:
                skip = False
            continue
        if stripped.startswith("from pipeline.") and "(" in stripped and ")" not in stripped:
            skip = True
            continue
        if stripped.startswith("from pipeline.config import "):
            imports = stripped[len("from pipeline.config import "):]
            if "(" not in imports:
                for part in imports.split(","):
                    part = part.strip()
                    if not part:
                        continue
                    if " as " in part:
                        name, alias = [x.strip() for x in part.split(" as ", 1)]
                        out.append(f"{alias} = {name}")
                    # bare names already defined in inlined config cell
            continue
        # Multi-line: from pipeline.config import (\n  FOO,\n)
        if stripped.startswith("from pipeline.config import"):
            continue
        if stripped.startswith("from pipeline import "):
            # e.g. "from pipeline import config as cfg" — drop; cfg.X rewritten below
            continue
        if stripped.startswith("from pipeline.") or stripped.startswith("import pipeline"):
            continue
        # Bare config-module attrs only (cfg.FOO). Do NOT rewrite cfg.get(...) —
        # local/param dicts are often named cfg (feature_utils, predict).
        line = re.sub(r"(?<![.\w])cfg\.([A-Z_][A-Z0-9_]*)", r"\1", line)
        # getattr(cfg, "KEY", default) → globals().get("KEY", default)
        line = re.sub(
            r'getattr\(\s*cfg\s*,\s*"([A-Z_][A-Z0-9_]*)"\s*,\s*([^)]+)\)',
            r'globals().get("\1", \2)',
            line,
        )
        # hasattr(cfg, key) / getattr(cfg, key) / setattr(cfg, key, val)
        # when key is a variable — notebook config lives in globals().
        line = re.sub(r"hasattr\(\s*cfg\s*,\s*(\w+)\s*\)", r"(\1 in globals())", line)
        line = re.sub(r"getattr\(\s*cfg\s*,\s*(\w+)\s*\)", r"globals()[\1]", line)
        line = re.sub(
            r"setattr\(\s*cfg\s*,\s*(\w+)\s*,\s*([^)]+)\)",
            r"globals().__setitem__(\1, \2)",
            line,
        )
        out.append(line)
    text = "\n".join(out)
    # Alias used by artifacts.full_config_snapshot
    text = re.sub(r"(?<![.\w])_cfg\b", "globals()", text)
    # dir(_cfg) already becomes dir(globals()); fix value fetch pattern left as
    # getattr(globals(), name) which is wrong — normalize common leftover.
    text = text.replace("getattr(globals(), name)", "globals()[name]")
    text = text.replace("for name in dir(globals()):", "for name in list(globals()):")
    return _sanitize_inlined_source(text)


def _strip_main_blocks(text: str) -> str:
    """Drop ``if __name__ == "__main__":`` blocks — notebooks set __name__ to __main__."""
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    skipping = False
    for line in lines:
        if not skipping and re.match(r"""^if\s+__name__\s*==\s*['\"]__main__['\"]\s*:""", line):
            skipping = True
            continue
        if skipping:
            if line.strip() == "":
                continue
            if line[:1] in (" ", "\t"):
                continue
            skipping = False
        out.append(line)
    return "".join(out)


def _remove_empty_try_except(text: str) -> str:
    """Remove try/except whose try body was emptied by stripping pipeline imports.

    Pattern left behind::
        try:
        except Exception:
            return None
    """
    lines = text.splitlines(keepends=True)
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        m = re.match(r"^([ \t]*)try:\s*\n?$", lines[i])
        if not m:
            out.append(lines[i])
            i += 1
            continue
        indent = m.group(1)
        j = i + 1
        # Skip blank lines inside try
        while j < n and lines[j].strip() == "":
            j += 1
        # Empty try → next non-blank is except at same indent
        if j < n and re.match(rf"^{re.escape(indent)}except\b.*:\s*\n?$", lines[j]):
            j += 1
            # Skip except body (more-indented lines) and blanks
            while j < n:
                if lines[j].strip() == "":
                    j += 1
                    continue
                if lines[j].startswith(indent + " ") or lines[j].startswith(indent + "\t"):
                    j += 1
                    continue
                break
            i = j
            continue
        out.append(lines[i])
        i += 1
    return "".join(out)


def _sanitize_inlined_source(text: str) -> str:
    """Remove empty try/except blocks left when config imports are stripped."""
    text = _strip_main_blocks(text)
    text = _remove_empty_try_except(text)
    text = re.sub(
        r"try:\s*\nexcept ImportError:\s*\n(?:\s+pass\s*\n)?",
        "",
        text,
    )
    text = re.sub(
        r"try:\s*\nexcept NameError:\s*\n\s+try:\s*\nexcept ImportError:\s*\n(?:\s+[^\n]+\n)?",
        "",
        text,
    )
    # Second pass after regex cleanup
    text = _remove_empty_try_except(text)
    return text


def _module_cell(name: str) -> dict:
    path = PIPE / f"{name}.py"
    if not path.exists():
        raise FileNotFoundError(path)
    raw = path.read_text()
    body = _strip_pipeline_imports(raw)
    header = f"# ── module: {name} " + "─" * max(1, 65 - len(name))
    return _code(f"{header}\n{body}")


def _config_cell() -> dict:
    cfg = (PIPE / "config.py").read_text()
    m = re.search(r"BASE_ELO\s*=", cfg)
    tail = cfg[m.start():] if m else ""
    tail = re.sub(r"_REPO_ROOT\s*=\s*Path\(__file__\).*?\n", "", tail, flags=re.S)
    tail = tail.replace("STATE_DIR = _REPO_ROOT / \"state\"", "STATE_DIR = ROOT / \"state\"")
    tail = re.sub(
        r"PBP_2026_PATH = .*?\n(?:if not PBP_2026_PATH\.exists\(\):\n    PBP_2026_PATH = .*?\n)?",
        'PBP_2026_PATH = ROOT / "[10-21-2025]-[05-18-2026]-combined-stats.csv"\n',
        tail,
        count=1,
    )
    tail = re.sub(
        r"if not MODERN_ODDS_PATH\.exists\(\):\n    MODERN_ODDS_PATH = .*?\n",
        "",
        tail,
    )
    tail = re.sub(
        r"if not PINNACLE_LINES_PATH\.exists\(\):\n    PINNACLE_LINES_PATH = .*?\n",
        "",
        tail,
    )
    head = '''# ============================================================================
#  CONFIG  ·  Google Drive mount + paths + constants  (no data leaks)
# ============================================================================
import math, re, copy, gc, pickle, json, warnings
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

IN_COLAB = False
try:
    import google.colab  # noqa: F401
    IN_COLAB = True
except ImportError:
    pass

if IN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    ROOT = Path("/content/drive/MyDrive/basketballData")
else:
    ROOT = Path("./basketballData")
ROOT.mkdir(parents=True, exist_ok=True)

V3_DATA_PATHS = {
    2021: ROOT / "events_2021_22_pbp_V3.csv",
    2022: ROOT / "events_2022_23_pbp_V3.csv",
    2023: ROOT / "events_2023_24_pbp_V3.csv",
    2024: ROOT / "events_2024_25_pbp_V3.csv",
}
PBP_2026_PATH    = ROOT / "[10-21-2025]-[05-18-2026]-combined-stats.csv"
MODERN_ODDS_PATH = ROOT / "all_odds.csv"
PINNACLE_LINES_PATH = ROOT / "nba_main_lines.csv"
PLAYER_LIST_PATH = ROOT / "nba_players_all.csv"
SPREAD_CSV_PATH = ROOT / "nba_betting_spread.csv"
ML_CSV_PATH = ROOT / "nba_betting_money_line.csv"
'''
    return _code(head + tail)


CHECKPOINT_CELL = '''# ============================================================================
#  CHECKPOINT HELPERS · save / load phase artifacts under ROOT/checkpoints/
# ============================================================================
CHECKPOINT_DIR = ROOT / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
(ROOT / "analysis_plots").mkdir(parents=True, exist_ok=True)

# Master switch: True = load saved artifacts when on disk; False = always run full compute.
# If True but a file is missing, that phase still runs end-to-end automatically.
USE_SAVED_ARTIFACTS = False

# Per-phase override: True forces recompute for that phase even when USE_SAVED_ARTIFACTS=True.
FORCE_RECOMPUTE = {
    "0": False,
    "1": False,
    "2": False,
    "2a": False,
    "2b": False,
    "2c": False,
    "2d": False,
    "2e": False,
    "3": False,
    "4": False,
    "5": False,
}

ARTIFACTS = {
    "toggles": CHECKPOINT_DIR / "runtime_toggles.json",
    "odds": CHECKPOINT_DIR / "odds_dict.pkl",
    "load_summary": CHECKPOINT_DIR / "load_summary.csv",
    "phase1_manifest": CHECKPOINT_DIR / "phase1_manifest.json",
    "backtest": ROOT / "backtest_results.csv",
    "tuning": STATE_DIR / "tuning_results.json",
    "diag_summary": CHECKPOINT_DIR / "diag_summary.csv",
    "diag_edges": CHECKPOINT_DIR / "diag_edges.pkl",
    "ablation": STATE_DIR / "ablation_posthoc.json",
    "ml_summary": ROOT / "analysis_plots" / "ml_diagnostics_summary.csv",
    "ml_plays": ROOT / "ml_value_plays.csv",
    "confidence_weights": STATE_DIR / "confidence_weights_by_season.json",
    "final_feats": CHECKPOINT_DIR / "final_feats.parquet",
    "meta_win_tuning": CHECKPOINT_DIR / "meta_win_tuning.json",
    "meta_total_tuning": CHECKPOINT_DIR / "meta_total_tuning.json",
    "prediction": CHECKPOINT_DIR / "last_prediction.json",
}


def _artifact_ready(*keys: str) -> bool:
  return all(ARTIFACTS[k].exists() for k in keys)


def use_saved_artifacts(phase: str) -> bool:
    """Return True when this phase may load from disk instead of recomputing."""
    if not USE_SAVED_ARTIFACTS:
        return False
    return not FORCE_RECOMPUTE.get(phase, False)


def save_json_artifact(key: str, payload: dict) -> Path:
    path = ARTIFACTS[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
    print(f"  💾 saved → {path}")
    return path


def load_json_artifact(key: str) -> dict:
    return json.loads(ARTIFACTS[key].read_text())


def save_pickle_artifact(key: str, obj) -> Path:
    path = ARTIFACTS[key]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)
    print(f"  💾 saved → {path}")
    return path


def load_pickle_artifact(key: str):
    with open(ARTIFACTS[key], "rb") as f:
        return pickle.load(f)


def load_results_csv() -> pd.DataFrame:
    path = ARTIFACTS["backtest"]
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path, low_memory=False)
    if "DATE" in df.columns:
        df["DATE"] = pd.to_datetime(df["DATE"], errors="coerce")
    return df


def ensure_results() -> pd.DataFrame:
    """Load backtest CSV if `results` is missing or empty (safe between phase cells)."""
    global results
    try:
        empty = results is None or results.empty
    except NameError:
        empty = True
        results = pd.DataFrame()
    if empty and use_saved_artifacts("2") and ARTIFACTS["backtest"].exists():
        results = load_results_csv()
        if not results.empty and "STAKE_MODERATE" not in results.columns:
            results = add_all_profile_columns(results)
        reasons = validate_backtest_df(results)
        if reasons:
            print("⚠️  Saved backtest may be stale:")
            for r in reasons:
                print(f"   · {r}")
            if is_backtest_stale(results):
                print("   → Set USE_SAVED_ARTIFACTS=False and re-run Phase 2")
    try:
        return results
    except NameError:
        return pd.DataFrame()


print("Checkpoint dir:", CHECKPOINT_DIR)
print(f"USE_SAVED_ARTIFACTS={USE_SAVED_ARTIFACTS}  (False = always recompute)")
print("Force recompute:", {k: v for k, v in FORCE_RECOMPUTE.items() if v} or "(none)")'''


def _workflow_phase_cells() -> list[dict]:
    """Ordered Phase 0–4 workflow with per-phase save/load."""
    return [
        _md("---\n\n## Phase 0 — Runtime toggles\n\nSaves `checkpoints/runtime_toggles.json`. Set `USE_SAVED_ARTIFACTS = False` to force a full retrain.\n\n**Speed (accuracy-preserving):** Hier/Elo tuning uses O(1) weights, combo caching, single predict+update, incremental season CV, and process-parallel Optuna. Keep full trial counts (`FAST_BACKTEST=False`); raise `OPTUNA_N_JOBS` on Colab high-RAM / local multi-core."),
        _code('''# ============================================================================
#  PHASE 0 · Runtime toggles (speed vs accuracy)
# ============================================================================
USE_SAVED_ARTIFACTS = False  # False: always recompute (recommended after model changes)
QUICK = False
FAST_BACKTEST = False  # True = fewer trials (faster, less search). Keep False for full accuracy.
USE_STINTS_CACHE = True
USE_TUNING_CACHE = True
WALKFORWARD_ZONE_PPS = True

# Process-parallel Optuna workers for Elo/Hier tuning (same n_trials + full CV).
# Colab free ~2; Colab Pro / local Mac often 4–8. Set 1 for fully serial TPE order.
import os
OPTUNA_N_JOBS = int(os.environ.get("OPTUNA_N_JOBS", "4"))
os.environ["OPTUNA_N_JOBS"] = str(max(1, OPTUNA_N_JOBS))

if FAST_BACKTEST:
    ELO_TRIALS, HIER_TRIALS, META_TRIALS, TOTAL_TRIALS, META_WIN_TRIALS, WINDOW = 12, 15, 15, 8, 12, 2
else:
    ELO_TRIALS, HIER_TRIALS, META_TRIALS, TOTAL_TRIALS, META_WIN_TRIALS, WINDOW = 150, 140, 50, 20, 25, 4

toggle_payload = {
    "USE_SAVED_ARTIFACTS": USE_SAVED_ARTIFACTS,
    "QUICK": QUICK,
    "FAST_BACKTEST": FAST_BACKTEST,
    "USE_STINTS_CACHE": USE_STINTS_CACHE,
    "USE_TUNING_CACHE": USE_TUNING_CACHE,
    "WALKFORWARD_ZONE_PPS": WALKFORWARD_ZONE_PPS,
    "OPTUNA_N_JOBS": OPTUNA_N_JOBS,
    "ELO_TRIALS": ELO_TRIALS,
    "HIER_TRIALS": HIER_TRIALS,
    "META_TRIALS": META_TRIALS,
    "TOTAL_TRIALS": TOTAL_TRIALS,
    "META_WIN_TRIALS": META_WIN_TRIALS,
    "WINDOW": WINDOW,
    "BET_SELECTION_MODE": BET_SELECTION_MODE,
    "MIN_EDGE_BUCKET": MIN_EDGE_BUCKET,
    "MIN_CONFIDENCE_SCORE": MIN_CONFIDENCE_SCORE,
    "CONFIDENCE_MIN_EDGE": CONFIDENCE_MIN_EDGE,
    "CONFIDENCE_EDGE_SCALED": CONFIDENCE_EDGE_SCALED,
    "MIN_DISAGREEMENT_TRUST": MIN_DISAGREEMENT_TRUST,
    "SKIP_PHANTOM_INJURY": SKIP_PHANTOM_INJURY,
    "SKIP_TIGHT_SPREAD": SKIP_TIGHT_SPREAD,
    "EDGE_AVOID_BAND": EDGE_AVOID_BAND,
    "TOTAL_HEAD_CV_SANITY_CAP": TOTAL_HEAD_CV_SANITY_CAP,
    "MAX_QUANTILE_WIDTH": MAX_QUANTILE_WIDTH,
    "STAKE_SIZING_MODE": STAKE_SIZING_MODE,
    "CONFIDENCE_MODE": CONFIDENCE_MODE,
}
save_json_artifact("toggles", toggle_payload)
print(
    f"Toggles: USE_SAVED={USE_SAVED_ARTIFACTS}  QUICK={QUICK}  FAST_BACKTEST={FAST_BACKTEST}  "
    f"stints_cache={USE_STINTS_CACHE}  tuning_cache={USE_TUNING_CACHE}  "
    f"OPTUNA_N_JOBS={os.environ['OPTUNA_N_JOBS']}"
)
print(f"Trials: elo={ELO_TRIALS} hier={HIER_TRIALS} meta={META_TRIALS} total={TOTAL_TRIALS} meta_win={META_WIN_TRIALS} window={WINDOW}")'''),

        _md("---\n\n## Phase 1 — Load data\n\nLoads from checkpoint when `USE_SAVED_ARTIFACTS=True` and files exist; otherwise builds stints + odds."),
        _code('''# ============================================================================
#  PHASE 1 · Load odds + build leak-free stint timelines
# ============================================================================
import hashlib
from IPython.display import display

def read_csv_fast(path):
    try:
        return pd.read_csv(path, engine="pyarrow")
    except Exception:
        return pd.read_csv(path, low_memory=False)

def _stints_cache_key(paths, quick, walkforward_zone_pps):
    """Match pipeline.stint_loader.stints_cache_key (schema versions included)."""
    h = hashlib.md5()
    for season, path in sorted(paths, key=lambda kv: kv[0]):
        p = Path(path)
        if p.exists():
            stt = p.stat()
            h.update(f"{season}:{p.name}:{int(stt.st_mtime)}:{stt.st_size}".encode())
    h.update(f"q={quick};wzpps={walkforward_zone_pps};as={ASSIST_SPLIT};zones=v2".encode())
    h.update(
        f"prep={PREPROCESSING_SCHEMA_VERSION};feat={FEATURE_SCHEMA_VERSION};"
        f"mkt={MARKET_SNAPSHOT_SCHEMA_VERSION};val={VALIDATION_SCHEMA_VERSION}".encode()
    )
    return h.hexdigest()[:16]

def load_all_stints(quick=False, walkforward_zone_pps=True, use_cache=True) -> pd.DataFrame:
    # Prefer shared loader when available (sidecar built_at + fingerprint).
    try:
        from pipeline.stint_loader import load_stints as _load_stints
        paths = list(V3_DATA_PATHS.items())
        if PBP_2026_PATH.exists():
            paths.append((2025, PBP_2026_PATH))
        st, meta = _load_stints(
            paths=paths, quick=quick, walkforward_zone_pps=walkforward_zone_pps,
            use_cache=use_cache, cache_dir=STATE_DIR, stints_mode="rebuild" if not use_cache else "auto",
        )
        print(f"  stints built_at={meta.get('built_at')}  key={meta.get('cache_key')}  hit={meta.get('cache_hit')}")
        return st
    except Exception as _e:
        print(f"  ⚠️ stint_loader fallback ({_e})")
    paths = list(V3_DATA_PATHS.items())
    if PBP_2026_PATH.exists():
        paths.append((2025, PBP_2026_PATH))
    cache_path = None
    if use_cache:
        cache_path = STATE_DIR / f"stints_cache_{_stints_cache_key(paths, quick, walkforward_zone_pps)}.pkl"
        if cache_path.exists():
            try:
                print(f"  ⚡ loading cached stints from {cache_path.name} ...")
                cached = pd.read_pickle(cache_path)
                print(f"  ✅ cache hit: {len(cached):,} stint rows")
                return cached
            except Exception as e:
                print(f"  ⚠️ cache read failed ({e}); rebuilding from source")
    sched = None
    try:
        if MODERN_ODDS_PATH.exists():
            sched = load_odds_schedule(str(MODERN_ODDS_PATH))
            print(f"  schedule for date recovery: {len(sched)} games")
    except Exception as e:
        print(f"  ⚠️ schedule load failed ({e})")
        sched = None
    frames = []
    prev_zone_pps = None
    for season, path in sorted(paths, key=lambda kv: kv[0]):
        if not Path(path).exists():
            print(f"  skip missing {path}")
            continue
        print(f"  loading {Path(path).name} ...")
        raw = read_csv_fast(path)
        df = convert_new_pbp(raw, name_to_id=name_to_id) if season >= 2025 else convert_v3_pbp(raw)
        df = preprocess_pbp(df, compute_xpoints=True,
                            zone_pps=prev_zone_pps if walkforward_zone_pps else None)
        if walkforward_zone_pps:
            _zp = compute_zone_pps(df)
            if _zp:
                prev_zone_pps = _zp
        st = build_stints(df, assist_split=ASSIST_SPLIT)
        st["season"] = season + 1
        if season < 2025 and sched is not None and not st.empty:
            _gd = pd.to_datetime(st["game_date"], errors="coerce")
            if _gd.dt.normalize().nunique() <= 1:
                st = attach_real_dates(st, sched, season_start_year=season)
        frames.append(st)
        del raw, df
        gc.collect()
        if quick and len(frames) >= 2:
            break
    if not frames:
        raise FileNotFoundError("No PBP data found under basketballData/")
    allst = pd.concat(frames, ignore_index=True)
    allst["game_date"] = pd.to_datetime(allst["game_date"], format="mixed", errors="coerce")
    allst = allst.sort_values(["game_date", "GAME_ID", "stint_id"]).reset_index(drop=True)
    if use_cache and cache_path is not None:
        try:
            allst.to_pickle(cache_path)
            print(f"  💾 cached stints -> {cache_path.name} ({len(allst):,} rows)")
        except Exception as e:
            print(f"  ⚠️ cache write failed: {e}")
    return allst

phase1_loaded = False
if use_saved_artifacts("1") and _artifact_ready("odds", "phase1_manifest"):
    print("⚡ Loading Phase 1 artifacts from disk ...")
    manifest = load_json_artifact("phase1_manifest")
    modern_odds_dict = load_pickle_artifact("odds")
    all_stints = load_all_stints(
        quick=manifest.get("QUICK", QUICK),
        walkforward_zone_pps=manifest.get("WALKFORWARD_ZONE_PPS", WALKFORWARD_ZONE_PPS),
        use_cache=True,
    )
    if ARTIFACTS["load_summary"].exists():
        load_summary = pd.read_csv(ARTIFACTS["load_summary"])
        display(load_summary.round(3))
    phase1_loaded = True
    print(f"✅ loaded stints={len(all_stints):,}  odds={len(modern_odds_dict):,}")

if not phase1_loaded:
    print("⏳ Building stint timelines...")
    all_stints = load_all_stints(
        quick=QUICK,
        walkforward_zone_pps=WALKFORWARD_ZONE_PPS,
        use_cache=USE_STINTS_CACHE,
    )
    print(f"✅ stints={len(all_stints):,}  games={all_stints['GAME_ID'].nunique():,}  seasons={sorted(all_stints['season'].unique())}")

    print("⏳ Loading modern Vegas lines...")
    modern_odds_dict = load_modern_odds(str(MODERN_ODDS_PATH)) if MODERN_ODDS_PATH.exists() else {}
    print(f"   all_odds entries: {len(modern_odds_dict):,}")

    if PINNACLE_LINES_PATH.exists():
        _sched_2026 = (
            all_stints[all_stints["season"] == 2026][["GAME_ID", "game_date", "home_team", "away_team"]]
            .drop_duplicates("GAME_ID")
            .rename(columns={"game_date": "date", "home_team": "home", "away_team": "away"})
        )
        _pin = load_pinnacle_lines(
            str(PINNACLE_LINES_PATH),
            schedule_df=_sched_2026 if not _sched_2026.empty else None,
        )
        modern_odds_dict.update(_pin)
        print(f"   merged {len(_pin):,} Pinnacle 2026 entries -> total {len(modern_odds_dict):,}")
    else:
        print("   (no nba_main_lines.csv found — using all_odds.csv only)")

    load_summary = diagnose_loaded_data(all_stints, modern_odds_dict)
    display(load_summary.round(3))

    save_pickle_artifact("odds", modern_odds_dict)
    load_summary.to_csv(ARTIFACTS["load_summary"], index=False)
    save_json_artifact("phase1_manifest", {
        "QUICK": QUICK,
        "WALKFORWARD_ZONE_PPS": WALKFORWARD_ZONE_PPS,
        "n_stint_rows": int(len(all_stints)),
        "n_games": int(all_stints["GAME_ID"].nunique()),
        "seasons": [int(s) for s in sorted(all_stints["season"].unique())],
        "n_odds": int(len(modern_odds_dict)),
    })'''),

        _md("---\n\n## Phase 2 — Walk-forward backtest\n\nLoads `backtest_results.csv` when `USE_SAVED_ARTIFACTS=True`; otherwise runs full walk-forward."),
        _code('''# ============================================================================
#  PHASE 2 · Walk-forward backtest (no leakage) + ROI-first stake profiles
# ============================================================================
import numpy as np

results = pd.DataFrame()
if use_saved_artifacts("2") and ARTIFACTS["backtest"].exists():
    print(f"⚡ Loading backtest results from {ARTIFACTS['backtest']} ...")
    results = load_results_csv()
    if not results.empty:
        if "STAKE_MODERATE" not in results.columns:
            results = add_all_profile_columns(results)
        print(f"✅ loaded {len(results):,} games from CSV")
        if ARTIFACTS["tuning"].exists():
            print(f"   tuning config: {ARTIFACTS['tuning']}")

if results.empty:
    print("⏳ Running full walk-forward backtest (no saved results or USE_SAVED_ARTIFACTS=False) ...")
    results = run_multi_year_backtest_walkforward(
        all_stints,
        odds_dict=modern_odds_dict,
        n_tuning_trials_elo=ELO_TRIALS,
        n_tuning_trials_hier=HIER_TRIALS,
        n_tuning_trials_meta=META_TRIALS,
        n_tuning_trials_total=TOTAL_TRIALS,
        rolling_window_size=WINDOW,
        tune_total_head=True,
        train_win_model=True,
        use_tuning_cache=USE_TUNING_CACHE,
        fast_tuning=FAST_BACKTEST,
    )

if not results.empty:
    if "STAKE_MODERATE" not in results.columns:
        results = add_all_profile_columns(results)

    benchmark_results(results)
    print_accuracy_layers(results)

    best_edge, grid = grid_search_bet_edge(results)
    print("\\nGOOD_BET_EDGE grid search:")
    print(grid.to_string(index=False) if not grid.empty else "  insufficient bets")
    print(f"Recommended OPTIMAL_BET_EDGE = {best_edge}")

    min_conf_last = (
        float(results["MIN_CONFIDENCE_SCORE"].iloc[-1])
        if "MIN_CONFIDENCE_SCORE" in results.columns
        else float(MIN_CONFIDENCE_SCORE)
    )
    strat = compare_selection_strategies(
        results,
        min_conf=min_conf_last,
        confidence_min_edge=CONFIDENCE_MIN_EDGE,
    )
    print("\\nSelection strategy comparison (same walk-forward export):")
    print(strat.to_string(index=False) if not strat.empty else "  insufficient data")

    profile_stats = {}
    recommended = "moderate"
    best_roi = -1e9
    for profile in ("conservative", "moderate", "aggressive"):
        stats = benchmark_betting_roi(results, profile=profile)
        profile_stats[profile] = stats
        roi = stats.get("roi", stats.get("roi_pct", float("nan")))
        ci_lo = stats.get("ci_lo", float("nan"))
        print(f"  {profile}: ROI={roi:.2f}%  bets={stats.get('n_bets', 0)}  CI=[{ci_lo:.2f}, {stats.get('ci_hi', float('nan')):.2f}]")
        if np.isfinite(roi) and np.isfinite(ci_lo) and ci_lo > 0 and roi > best_roi:
            best_roi = roi
            recommended = profile
    print(f"\\nRecommended stake profile (blind walk-forward): {recommended}")

    edge_thr = float(results["EDGE_THRESHOLD"].iloc[-1]) if "EDGE_THRESHOLD" in results.columns else best_edge
    fav_dec = float(results["MAX_FAVORITE_DECIMAL"].iloc[-1]) if "MAX_FAVORITE_DECIMAL" in results.columns else 1.45
    ou_thr = float(results["OU_MIN_EDGE"].iloc[-1]) if "OU_MIN_EDGE" in results.columns else 3.0
    elo_cal = {}
    knobs_path = STATE_DIR / "elo_calibration_knobs.json"
    if knobs_path.exists():
        elo_cal = json.loads(knobs_path.read_text())

    tuning_payload = {
        "optimal_bet_edge": best_edge,
        "walkforward_edge_threshold": edge_thr,
        "max_favorite_decimal": fav_dec,
        "ou_min_edge": ou_thr,
        "recommended_profile": recommended,
        "min_confidence_score": float(results["MIN_CONFIDENCE_SCORE"].iloc[-1]) if "MIN_CONFIDENCE_SCORE" in results.columns else MIN_CONFIDENCE_SCORE,
        "suggested_min_confidence": float(results["MIN_CONFIDENCE_SCORE"].iloc[-1]) if "MIN_CONFIDENCE_SCORE" in results.columns else MIN_CONFIDENCE_SCORE,
        "confidence_mode": CONFIDENCE_MODE,
        "confidence_min_edge": CONFIDENCE_MIN_EDGE,
        "confidence_edge_scaled": CONFIDENCE_EDGE_SCALED,
        "spread_calib_window": SPREAD_CALIB_WINDOW,
        "last_walkforward_season": results["simulated_season_window"].iloc[-1] if "simulated_season_window" in results.columns else None,
        "blind_ats_roi": {p: profile_stats.get(p, {}).get("roi") for p in profile_stats},
        "blind_roi_ci": {
            p: {"lo": profile_stats.get(p, {}).get("ci_lo"), "hi": profile_stats.get(p, {}).get("ci_hi")}
            for p in profile_stats
        },
        "elo_calibration": elo_cal,
        "grid": grid.to_dict(orient="records") if not grid.empty else [],
    }
    ARTIFACTS["tuning"].parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS["tuning"].write_text(json.dumps(tuning_payload, indent=2))
    results.to_csv(ARTIFACTS["backtest"], index=False)
    print(f"💾 Saved → {ARTIFACTS['backtest']}")
    print(f"💾 Tuning config → {ARTIFACTS['tuning']}")
else:
    print("No backtest results produced.")'''),

        _md("---\n\n## Phase 2a — Unified confidence diagnostics\n\n**Tuning runs inline** during the walk-forward backtest (`CONFIDENCE_MODE='unified'`, `isotonic` on composite score).\nThis cell plots reliability / threshold grids from saved results — it does **not** re-tune unless you re-run backtest without inline Phase 2a."),
        _code('''# ============================================================================
#  PHASE 2a · Unified confidence diagnostics (tuning is inline in walk-forward)
# ============================================================================
%matplotlib inline

results = ensure_results()
conf_diag = {}
conf_plots = (
    "14_confidence_reliability.png",
    "14b_raw_confidence_reliability.png",
    "15_roi_vs_confidence_threshold.png",
    "16_confidence_calibration_ablation.png",
    "16b_unified_weight_tuning_roi.png",
    "16c_confidence_calibration_brier.png",
)
plot_dir = ROOT / "analysis_plots"

inline_p2a = (
    not results.empty
    and "PHASE2A_INLINE" in results.columns
    and results["PHASE2A_INLINE"].fillna(0).astype(int).max() > 0
)

if use_saved_artifacts("2a") and ARTIFACTS["confidence_weights"].exists() and all((plot_dir / n).exists() for n in conf_plots):
    print("⚡ Phase 2a confidence artifacts already on disk (skipping recompute)")
    import json
    with open(ARTIFACTS["confidence_weights"]) as f:
        conf_diag = json.load(f)
elif not results.empty:
    if inline_p2a:
        print("✅ Phase 2a weights were tuned inline during walk-forward — diagnostics/plots only")
    conf_diag = run_phase_2a_unified_confidence(
        results,
        save_dir=plot_dir,
        show_plots=True,
    )
else:
    print("Skip Phase 2a — no backtest results.")

from IPython.display import Image, display
for name in conf_plots:
    p = plot_dir / name
    if p.exists():
        display(Image(filename=str(p)))'''),

        _md("---\n\n## Phase 2b — Spread diagnostics + ablation\n\nSaves `checkpoints/diag_summary.csv`, `diag_edges.pkl`, `state/ablation_posthoc.json`."),
        _code('''# ============================================================================
#  PHASE 2b · Backtest diagnostics (per season + graphs) + spread ablation
# ============================================================================
%matplotlib inline
from IPython.display import display

results = ensure_results()
diag_summary = pd.DataFrame()
diag_edges = {}

if use_saved_artifacts("2b") and _artifact_ready("diag_summary", "diag_edges"):
    print("⚡ Loading Phase 2b artifacts from disk ...")
    diag_summary = pd.read_csv(ARTIFACTS["diag_summary"])
    diag_edges = load_pickle_artifact("diag_edges")
    display(diag_summary.round(3))
    for season, tbl in diag_edges.items():
        print(f"\\nEdge buckets · {season}")
        display(tbl.round(3))
    if ARTIFACTS["ablation"].exists():
        ab_summary = pd.read_json(ARTIFACTS["ablation"])
        print("\\n--- Post-hoc spread ablation (loaded) ---")
        display(ab_summary.round(4))

elif not results.empty:
    diag_summary, diag_edges = run_backtest_diagnostics(
        results,
        save_dir=ROOT / "analysis_plots",
        show_graphs=True,
    )
    display(diag_summary.round(3))
    for season, tbl in diag_edges.items():
        print(f"\\nEdge buckets · {season}")
        display(tbl.round(3))

    diag_summary.to_csv(ARTIFACTS["diag_summary"], index=False)
    save_pickle_artifact("diag_edges", diag_edges)
    print(f"💾 Saved → {ARTIFACTS['diag_summary']}")

    print("\\n--- Post-hoc spread ablation ---")
    ab_summary = posthoc_ablation_summary(results)
    display(ab_summary.round(4))
    if not ab_summary.empty:
        base = ab_summary[ab_summary["config"] == "baseline_current"]
        full = ab_summary[ab_summary["config"] == "edge_bucket_full"]
        if not base.empty and not full.empty:
            print("Ablation gate (baseline vs edge_bucket_full):",
                  passes_ablation_gate(base.iloc[0].to_dict(), full.iloc[0].to_dict()))
    print("\\n--- Calibration mode ablation configs (full re-run required) ---")
    for name in calibration_ablation_configs():
        print(f"  · {name}")
    ARTIFACTS["ablation"].parent.mkdir(parents=True, exist_ok=True)
    ARTIFACTS["ablation"].write_text(ab_summary.to_json(orient="records", indent=2))
    print(f"💾 Saved → {ARTIFACTS['ablation']}")
else:
    print("Skip diagnostics — no backtest results.")'''),

        _md("---\n\n## Phase 2c — Bankroll & confidence plots\n\nSaves PNGs under `analysis_plots/`. Skips if plots exist and `FORCE_RECOMPUTE[\"2c\"]` is False."),
        _code('''# ============================================================================
#  PHASE 2c · ROI / bankroll / confidence plots
# ============================================================================
%matplotlib inline

results = ensure_results()
plot_dir = ROOT / "analysis_plots"
key_plots = (
    "10_bankroll_by_profile.png",
    "11_roi_by_confidence_tier.png",
    "12_confidence_reliability.png",
    "12b_cover_prob_reliability_by_season.png",
)

if results.empty:
    print("Skip betting plots — no backtest results.")
elif use_saved_artifacts("2c") and all((plot_dir / n).exists() for n in key_plots):
    print(f"⚡ Loading existing plots from {plot_dir} ...")
else:
    generate_betting_plots(results, save_dir=plot_dir)
    print(f"💾 Plots saved → {plot_dir}/")

from IPython.display import Image, display
for name in key_plots + ("13_ml_winprob_reliability.png",):
    p = plot_dir / name
    if p.exists():
        display(Image(filename=str(p)))'''),

        _md("---\n\n## Phase 2d — Moneyline diagnostics (MetaWin calibration)\n\nSaves `analysis_plots/ml_diagnostics_summary.csv` and reliability plot."),
        _code('''# ============================================================================
#  PHASE 2d · Moneyline training diagnostics (WIN_PROB / MetaWin)
# ============================================================================
results = ensure_results()
MIN_ML_EV = 0.03
MAX_FAV_DEC = float(results["MAX_FAVORITE_DECIMAL"].iloc[-1]) if (
    not results.empty and "MAX_FAVORITE_DECIMAL" in results.columns
) else 1.45

ml_summary = pd.DataFrame()
if use_saved_artifacts("2d") and ARTIFACTS["ml_summary"].exists():
    print(f"⚡ Loading ML diagnostics from {ARTIFACTS['ml_summary']} ...")
    ml_summary = pd.read_csv(ARTIFACTS["ml_summary"])
    display(ml_summary.round(3))
elif not results.empty:
    ml_summary = run_ml_diagnostics(
        results,
        min_ev=MIN_ML_EV,
        max_fav_dec=MAX_FAV_DEC,
        save_dir=ROOT / "analysis_plots",
        show_plots=True,
    )
    display(ml_summary.round(3))
    print(f"Using walk-forward config: edge≥{WALKFORWARD_EDGE_MIN_FLOOR}, "
          f"ML min EV={MIN_ML_EV:.0%}, max favorite decimal={MAX_FAV_DEC:.2f}")
else:
    print("Skip ML diagnostics — no backtest results.")'''),

        _md("---\n\n## Phase 2e — ML underdog / favorite value tracker\n\nSaves `ml_value_plays.csv` for tracking underdog value and undervalued favorites."),
        _code('''# ============================================================================
#  PHASE 2e · ML value plays (underdog / undervalued favorite tracker)
# ============================================================================
results = ensure_results()
MIN_ML_EV = globals().get("MIN_ML_EV", 0.03)
MAX_FAV_DEC = globals().get(
    "MAX_FAV_DEC",
    float(results["MAX_FAVORITE_DECIMAL"].iloc[-1]) if (
        not results.empty and "MAX_FAVORITE_DECIMAL" in results.columns
    ) else 1.45,
)
ml_plays = pd.DataFrame()
if use_saved_artifacts("2e") and ARTIFACTS["ml_plays"].exists():
    print(f"⚡ Loading ML value plays from {ARTIFACTS['ml_plays']} ...")
    ml_plays = pd.read_csv(ARTIFACTS["ml_plays"], low_memory=False)
elif not results.empty:
    ml_plays = export_ml_tracking(
        results,
        ARTIFACTS["ml_plays"],
        min_ev=MIN_ML_EV,
        max_fav_dec=MAX_FAV_DEC,
    )
else:
    print("Skip ML tracker — no backtest results.")

if not ml_plays.empty:
    display(ml_plays.head(20))
    by_type = ml_plays.groupby("ml_value_type").agg(
        n=("ml_value_type", "count"),
        win_pct=("ml_won", "mean"),
        avg_ev=("ml_best_ev", "mean"),
        roi=("ml_flat_profit", "mean"),
    )
    print("\\nML value plays by type:")
    display(by_type.round(3))
elif use_saved_artifacts("2e"):
    print("No ML value plays at current thresholds.")'''),

        _md("---\n\n## Phase 3 — Dedicated ML + Total + Score-pair models\n\nTrains tuned `MetaWinModel`, `MetaTotalModel`, and `MetaScorePairModel`; saves `state/win.pkl`, `state/total.pkl`, `state/score_pair.pkl`, and `checkpoints/final_feats.parquet` for Phase 4."),
        _code('''# ============================================================================
#  PHASE 3 · Dedicated Moneyline + Total + Score-pair training
# ============================================================================
ml_total_loaded = False
if (
    use_saved_artifacts("3")
    and (STATE_DIR / "win.pkl").exists()
    and (STATE_DIR / "total.pkl").exists()
):
    print(f"⚡ Loading ML/Total models from {STATE_DIR} ...")
    final_win = MetaWinModel.load(STATE_DIR / "win.pkl")
    final_total = MetaTotalModel.load(STATE_DIR / "total.pkl")
    final_score_pair = (
        MetaScorePairModel.load(STATE_DIR / "score_pair.pkl")
        if (STATE_DIR / "score_pair.pkl").exists()
        else None
    )
    if final_score_pair is not None:
        print("   loaded score_pair.pkl")
    if ARTIFACTS["final_feats"].exists():
        final_feats = pd.read_parquet(ARTIFACTS["final_feats"])
        print(f"   loaded feature matrix: {len(final_feats):,} rows")
    else:
        final_feats = None
    ml_total_loaded = True

if not ml_total_loaded:
    train_df = all_stints.copy()
    best_elo_raw = tune_elo_tracker(train_df, DEFAULT_LEAGUE_XPPP, n_trials=ELO_TRIALS)
    best_elo = map_elo_params(best_elo_raw)
    best_hier = tune_hierarchical(train_df, n_trials=HIER_TRIALS)

    feat_hier = HierarchicalPossessionEngine(**best_hier)
    feat_elo = PlayerRatingTracker(config=best_elo, league_xppp=DEFAULT_LEAGUE_XPPP)
    feat_pace = PaceTracker(team_window=10, league_window=100)
    feat_xppp = TeamXpppTracker(window_size=40, prev_season_weight=0.5)
    feat_form = TeamFormTracker(window=15, prev_season_weight=0.4)
    feat_rotation = RotationLineupTracker()
    feat_lineup_elo = LineupEloTracker()
    feat_chemistry = ChemistryTracker()
    feat_team_elo = TeamEloTracker()
    feat_travel = TravelTracker()

    final_feats = generate_features(
        train_df, feat_hier, feat_elo, feat_pace,
        odds_dict=modern_odds_dict, update_engines=True,
        team_xppp_tracker=feat_xppp, team_form_tracker=feat_form,
        rotation_tracker=feat_rotation, lineup_elo_tracker=feat_lineup_elo,
        chemistry_tracker=feat_chemistry, team_elo_tracker=feat_team_elo,
        travel_tracker=feat_travel,
    )
    final_feats = engineer_interaction_features(final_feats)
    final_feats.to_parquet(ARTIFACTS["final_feats"], index=False)
    print(f"💾 Saved feature matrix → {ARTIFACTS['final_feats']}")

    calib_cut = max(len(final_feats) // 5, 80)
    meta_train = final_feats.iloc[:-calib_cut]
    meta_calib = final_feats.iloc[-calib_cut:]

    margin_params = tune_margin_model(
        final_feats, n_trials=max(8, META_TRIALS // 2),
        fast_mode=FAST_BACKTEST or META_TRIALS <= 5,
    )
    margin_model = MetaScoreModel(
        ridge_alpha=margin_params["ridge_alpha"],
        cb_params=margin_params["cb_params"],
        huber_epsilon=margin_params["huber_epsilon"],
        use_isotonic_calibration=True,
        use_elo_stack=True,
        total_mode="external",
    )
    margin_model.fit(
        meta_train, meta_train["actual_home"], meta_train["actual_away"],
        calib_df=meta_calib,
    )
    meta_train = meta_train.copy()
    meta_calib = meta_calib.copy()
    meta_train["pred_margin"] = margin_model._predict_raw(meta_train)["pred_margin"]
    meta_calib["pred_margin"] = margin_model._predict_raw(meta_calib)["pred_margin"]

    best_win = tune_meta_win_model(
        meta_train, n_trials=META_WIN_TRIALS,
        pred_margin_col="pred_margin",
        fast_mode=FAST_BACKTEST or META_WIN_TRIALS <= 5,
    )
    save_json_artifact("meta_win_tuning", best_win)
    final_win = MetaWinModel(
        C=best_win.get("C", 0.1),
        feature_cols=best_win.get("feature_cols") or win_feature_cols(meta_train),
        elo_win_blend=best_win.get("elo_win_blend"),
        calib_method=best_win.get("calib_method", "isotonic"),
    )
    final_win.fit(
        meta_train,
        (meta_train["actual_home"] > meta_train["actual_away"]).astype(int),
        calib_df=meta_calib,
    )
    final_win.save(STATE_DIR / "win.pkl")
    print(f"💾 Saved → {STATE_DIR / 'win.pkl'}")

    best_total = tune_total_model(
        final_feats, n_trials=TOTAL_TRIALS,
        feature_cols=total_feature_cols(final_feats),
        fast_mode=FAST_BACKTEST or TOTAL_TRIALS <= 5,
    )
    save_json_artifact("meta_total_tuning", best_total)
    best_total = dict(best_total)
    cv_score = float(best_total.pop("_cv_score", best_total.pop("_cv_mae", np.nan)))
    if np.isfinite(cv_score) and cv_score > float(TOTAL_HEAD_CV_SANITY_CAP):
        print(f"  ⚠ Total-model CV score {cv_score:.1f} exceeds cap — using defaults")
        best_total.setdefault("ridge_alpha", 10.0)
        best_total.setdefault("huber_epsilon", 1.35)
        best_total.setdefault("total_elo_beta", TOTAL_ELO_BETA)
        best_total.setdefault("train_target", TOTAL_TRAIN_TARGET)
        best_total.setdefault("feature_cols", total_feature_cols(final_feats))
    final_total = MetaTotalModel(
        ridge_alpha=best_total.get("ridge_alpha", 10.0),
        cb_params=best_total.get("cb_params"),
        huber_epsilon=best_total.get("huber_epsilon", 1.35),
        feature_cols=best_total.get("feature_cols") or total_feature_cols(final_feats),
        total_elo_beta=best_total.get("total_elo_beta", TOTAL_ELO_BETA),
        train_target=best_total.get("train_target", TOTAL_TRAIN_TARGET),
    )
    final_total.fit(final_feats)
    final_total.save(STATE_DIR / "total.pkl")
    print(f"💾 Saved → {STATE_DIR / 'total.pkl'}")

    # Dual home/away score heads (specialized totals path)
    final_score_pair = MetaScorePairModel(
        feature_cols=total_feature_cols(final_feats),
        train_target="absolute",
    )
    final_score_pair.fit(final_feats)
    final_score_pair.save(STATE_DIR / "score_pair.pkl")
    print(f"💾 Saved → {STATE_DIR / 'score_pair.pkl'}")
    print("✅ Phase 3 ML + Total + Score-pair models trained")'''),

        _md("---\n\n## Phase 4 — Spread / final engines\n\nTrains margin `MetaScoreModel` and persists engines. Reloads `final_feats` from Phase 3 when available."),
        _code('''# ============================================================================
#  PHASE 4 · Train spread engines + MetaScoreModel (margin-only)
# ============================================================================
engines_loaded = False
if use_saved_artifacts("4") and (STATE_DIR / "meta.pkl").exists():
    print(f"⚡ Loading spread engines from {STATE_DIR} ...")
    final_hier = HierarchicalPossessionEngine.load_state(STATE_DIR / "hier.pkl")
    final_elo = PlayerRatingTracker.load_state(STATE_DIR / "elo.pkl")
    final_pace = PaceTracker.load_state(STATE_DIR / "pace.pkl") if (STATE_DIR / "pace.pkl").exists() else PaceTracker()
    if (STATE_DIR / "xppp.pkl").exists():
        final_xppp = TeamXpppTracker.load_state(STATE_DIR / "xppp.pkl")
    else:
        final_xppp = TeamXpppTracker(window_size=40, prev_season_weight=0.5)
    final_form = TeamFormTracker.load_state(STATE_DIR / "form.pkl") if (STATE_DIR / "form.pkl").exists() else TeamFormTracker()
    final_rotation = RotationLineupTracker.load_state(STATE_DIR / "rotation.pkl") if (STATE_DIR / "rotation.pkl").exists() else RotationLineupTracker()
    final_lineup_elo = LineupEloTracker.load_state(STATE_DIR / "lineup_elo.pkl") if (STATE_DIR / "lineup_elo.pkl").exists() else LineupEloTracker()
    final_chemistry = ChemistryTracker.load_state(STATE_DIR / "chemistry.pkl") if (STATE_DIR / "chemistry.pkl").exists() else ChemistryTracker()
    final_team_elo = TeamEloTracker.load_state(STATE_DIR / "team_elo.pkl") if (STATE_DIR / "team_elo.pkl").exists() else TeamEloTracker()
    final_travel = TravelTracker.load_state(STATE_DIR / "travel.pkl") if (STATE_DIR / "travel.pkl").exists() else TravelTracker()
    final_meta = MetaScoreModel.load(STATE_DIR / "meta.pkl")
    if "final_win" not in dir() and (STATE_DIR / "win.pkl").exists():
        final_win = MetaWinModel.load(STATE_DIR / "win.pkl")
    if "final_total" not in dir() and (STATE_DIR / "total.pkl").exists():
        final_total = MetaTotalModel.load(STATE_DIR / "total.pkl")
    if "final_score_pair" not in dir() and (STATE_DIR / "score_pair.pkl").exists():
        final_score_pair = MetaScorePairModel.load(STATE_DIR / "score_pair.pkl")
    if (STATE_DIR / "last_dates.pkl").exists():
        with open(STATE_DIR / "last_dates.pkl", "rb") as f:
            last_game_dates = pickle.load(f)
    else:
        last_game_dates = {}
    engines_loaded = True
    print("✅ Spread engines loaded from disk")

if not engines_loaded:
    train_df = all_stints.copy()
    if "final_feats" not in dir():
        final_feats = None
    # Prefer Phase 3 walked engines (feat_*) so Phase 5 doesn't get virgin trackers
    # while Meta trains on Phase 3 features.
    _phase3_engines = all(
        name in dir() and globals().get(name) is not None
        for name in (
            "feat_elo", "feat_hier", "feat_pace", "feat_xppp", "feat_form",
            "feat_rotation", "feat_lineup_elo", "feat_chemistry", "feat_team_elo",
            "feat_travel",
        )
    )
    if final_feats is None and ARTIFACTS["final_feats"].exists():
        final_feats = pd.read_parquet(ARTIFACTS["final_feats"])
        print(f"⚡ Loaded feature matrix from {ARTIFACTS['final_feats']}")

    if _phase3_engines and final_feats is not None:
        print("♻️  Reusing Phase 3 walked engines (feat_*) for Phase 4/5")
        final_elo, final_hier, final_pace = feat_elo, feat_hier, feat_pace
        final_xppp, final_form = feat_xppp, feat_form
        final_rotation, final_lineup_elo = feat_rotation, feat_lineup_elo
        final_chemistry, final_team_elo = feat_chemistry, feat_team_elo
        final_travel = feat_travel
    else:
        # Always walk engines — never train Meta on features from one walk
        # while saving un-updated virgin trackers for Phase 5.
        best_elo_raw = tune_elo_tracker(train_df, DEFAULT_LEAGUE_XPPP, n_trials=ELO_TRIALS)
        best_elo = map_elo_params(best_elo_raw)
        best_hier = tune_hierarchical(train_df, n_trials=HIER_TRIALS)
        final_hier = HierarchicalPossessionEngine(**best_hier)
        final_elo = PlayerRatingTracker(config=best_elo, league_xppp=DEFAULT_LEAGUE_XPPP)
        final_pace = PaceTracker(team_window=10, league_window=100)
        final_xppp = TeamXpppTracker(window_size=40, prev_season_weight=0.5)
        final_form = TeamFormTracker(window=15, prev_season_weight=0.4)
        final_rotation = RotationLineupTracker()
        final_lineup_elo = LineupEloTracker()
        final_chemistry = ChemistryTracker()
        final_team_elo = TeamEloTracker()
        final_travel = TravelTracker()
        walked = generate_features(
            train_df, final_hier, final_elo, final_pace,
            odds_dict=modern_odds_dict, update_engines=True,
            team_xppp_tracker=final_xppp, team_form_tracker=final_form,
            rotation_tracker=final_rotation, lineup_elo_tracker=final_lineup_elo,
            chemistry_tracker=final_chemistry, team_elo_tracker=final_team_elo,
            travel_tracker=final_travel,
        )
        walked = engineer_interaction_features(walked)
        if final_feats is None:
            final_feats = walked
            final_feats.to_parquet(ARTIFACTS["final_feats"], index=False)
            print(f"💾 Saved feature matrix → {ARTIFACTS['final_feats']}")
        else:
            print("ℹ️  Kept existing final_feats; engines walked to match Phase 5 state")

    calib_cut = max(len(final_feats) // 5, 80)
    meta_train = final_feats.iloc[:-calib_cut]
    meta_calib = final_feats.iloc[-calib_cut:]
    elo_knobs = tune_elo_calibrator(meta_train, calib_df=meta_calib)
    elo_calibrator = WalkForwardEloCalibrator(knobs=elo_knobs)
    elo_calibrator.fit(meta_train, calib_df=meta_calib)
    final_feats = apply_elo_calibration_df(final_feats, elo_calibrator)
    meta_train = final_feats.iloc[:-calib_cut]
    meta_calib = final_feats.iloc[-calib_cut:]

    best_meta = tune_margin_model(
        final_feats, n_trials=META_TRIALS, fast_mode=FAST_BACKTEST or META_TRIALS <= 5,
    )
    elo_knobs.elo_blend_alpha = best_meta.get("elo_blend_alpha", elo_knobs.elo_blend_alpha)
    elo_knobs.elo_ridge_alpha = best_meta.get("elo_ridge_alpha", elo_knobs.elo_ridge_alpha)
    final_meta = MetaScoreModel(
        ridge_alpha=best_meta["ridge_alpha"], cb_params=best_meta["cb_params"],
        huber_epsilon=best_meta["huber_epsilon"], use_isotonic_calibration=True,
        use_elo_stack=True, total_mode="external",
        elo_blend_alpha=elo_knobs.elo_blend_alpha,
        elo_ridge_alpha=elo_knobs.elo_ridge_alpha,
    )
    final_meta.fit(
        meta_train, meta_train["actual_home"], meta_train["actual_away"],
        calib_df=meta_calib,
    )

    final_elo.save_state(STATE_DIR / "elo.pkl")
    final_hier.save_state(STATE_DIR / "hier.pkl")
    final_pace.save_state(STATE_DIR / "pace.pkl")
    final_xppp.save_state(STATE_DIR / "xppp.pkl")
    final_form.save_state(STATE_DIR / "form.pkl")
    final_meta.save(STATE_DIR / "meta.pkl")
    final_rotation.save_state(STATE_DIR / "rotation.pkl")
    final_lineup_elo.save_state(STATE_DIR / "lineup_elo.pkl")
    final_chemistry.save_state(STATE_DIR / "chemistry.pkl")
    final_team_elo.save_state(STATE_DIR / "team_elo.pkl")
    final_travel.save_state(STATE_DIR / "travel.pkl")
    if elo_calibrator.fitted:
        elo_calibrator.save(STATE_DIR / "elo_calibrator.pkl")
    save_elo_knobs(elo_knobs, STATE_DIR / "elo_calibration_knobs.json")
    last_game_dates = train_df.groupby("home_team")["game_date"].max().to_dict()
    for t, d in train_df.groupby("away_team")["game_date"].max().to_dict().items():
        last_game_dates[t] = max(last_game_dates.get(t, d), d)
    with open(STATE_DIR / "last_dates.pkl", "wb") as f:
        pickle.dump(last_game_dates, f)
    print(f"✅ Spread engines saved to {STATE_DIR}")

final_ats = None
final_vol = TeamVolatilityTracker()
try:
    ats_path = STATE_DIR / "latest_ats.pkl"
    if ats_path.exists():
        final_ats = ATSClassifier.load(ats_path)
        print("Loaded ATS classifier from", ats_path)
except Exception as e:
    print("ATS classifier load skipped:", e)'''),

        _md("---\n\n## Phase 5 — Daily prediction\n\nLoads spread (`meta`), moneyline (`win`), total, and score-pair models for live picks.\n\nOutputs for each game: `pred_margin`, `p_home`/`p_away`, `ml_bet` (Home/Away/Pass), `pred_home_pts`/`pred_away_pts`, `pred_total`, `sigma_total`, `p_over`."),
        _code('''# ============================================================================
#  PHASE 5 · Daily prediction for an upcoming game
# ============================================================================
_needed = ("final_hier", "final_elo", "final_meta", "final_pace")
_missing = [n for n in _needed if n not in dir() or globals().get(n) is None]
if _missing:
    raise RuntimeError(
        f"Phase 5 missing engines {_missing}. Run Phase 4 first "
        f"(or set USE_SAVED_ARTIFACTS=True with state/*.pkl present)."
    )
if "final_xppp" not in dir() or final_xppp is None:
    if (STATE_DIR / "xppp.pkl").exists():
        final_xppp = TeamXpppTracker.load_state(STATE_DIR / "xppp.pkl")
    else:
        final_xppp = TeamXpppTracker(window_size=40, prev_season_weight=0.5)
if "final_form" not in dir() or final_form is None:
    final_form = TeamFormTracker()
if "final_rotation" not in dir() or final_rotation is None:
    final_rotation = RotationLineupTracker()
if "final_lineup_elo" not in dir() or final_lineup_elo is None:
    final_lineup_elo = LineupEloTracker()
if "final_chemistry" not in dir() or final_chemistry is None:
    final_chemistry = ChemistryTracker()
if "final_team_elo" not in dir() or final_team_elo is None:
    final_team_elo = TeamEloTracker()
if "final_travel" not in dir() or final_travel is None:
    final_travel = TravelTracker()
if "last_game_dates" not in dir() or last_game_dates is None:
    last_game_dates = {}

# Seed Elo xPPP priors from current rotation (empty-history warm-start; low
# turnover leaves observed rolling xPPP as source of truth).
_prior = {}
if "final_rotation" in dir() and final_rotation is not None:
    for _t in list(getattr(final_rotation, "history", {}).keys()):
        _pairs = final_rotation.expected_weights(_t)
        _prior[_t] = {pid for pid, _ in _pairs} if _pairs else set()
refresh_team_priors_for_season(
    final_xppp,
    elo_tracker=final_elo,
    pace_tracker=final_pace if "final_pace" in dir() else None,
    rotation_tracker=final_rotation,
    prior_rosters=_prior,
)

if "final_score_pair" not in dir() and (STATE_DIR / "score_pair.pkl").exists():
    final_score_pair = MetaScorePairModel.load(STATE_DIR / "score_pair.pkl")
if "final_win" not in dir() and (STATE_DIR / "win.pkl").exists():
    final_win = MetaWinModel.load(STATE_DIR / "win.pkl")
if "final_total" not in dir() and (STATE_DIR / "total.pkl").exists():
    final_total = MetaTotalModel.load(STATE_DIR / "total.pkl")

elo_cal = WalkForwardEloCalibrator.load(STATE_DIR / "elo_calibrator.pkl") \\
    if (STATE_DIR / "elo_calibrator.pkl").exists() else None
bet_cal = WalkForwardBetCalibrator.load(STATE_DIR / "bet_calibrator.pkl") \\
    if (STATE_DIR / "bet_calibrator.pkl").exists() else None
tuning_cfg = load_tuning_config()
print(f"Using profile: {tuning_cfg.get('recommended_profile', 'moderate')}")
print(f"Confidence: mode={CONFIDENCE_MODE}  min_score={load_min_confidence_threshold()}  "
      f"selection={CONFIDENCE_SELECTION_MODE}")
print(f"OU Gaussian={OU_USE_GAUSSIAN_PROB}  OU_MIN_PROB={OU_MIN_PROB}  "
      f"USE_SCORE_PAIR_TOTAL={USE_SCORE_PAIR_TOTAL}  ML_BET_PREDICTED_WINNER_ONLY={ML_BET_PREDICTED_WINNER_ONLY}")
if tuning_cfg.get("elo_calibration"):
    ec = tuning_cfg["elo_calibration"]
    print(f"ELO knobs: blend_alpha={ec.get('elo_blend_alpha')} ridge_alpha={ec.get('elo_ridge_alpha')} huber={ec.get('huber_epsilon')}")

pred = predict_game(
    home_abbr="BOS",
    away_abbr="LAL",
    game_date="2026-10-28",
    hier_engine=final_hier,
    elo_tracker=final_elo,
    meta_model=final_meta,
    pace_tracker=final_pace,
    team_xppp_tracker=final_xppp,
    team_form_tracker=final_form,
    rotation_tracker=final_rotation,
    lineup_elo_tracker=final_lineup_elo,
    chemistry_tracker=final_chemistry,
    team_elo_tracker=final_team_elo,
    travel_tracker=final_travel,
    win_model=final_win if "final_win" in dir() else None,
    total_model=final_total if "final_total" in dir() else None,
    score_pair_model=final_score_pair if "final_score_pair" in dir() else None,
    ats_classifier=final_ats if "final_ats" in dir() else None,
    vol_tracker=final_vol if "final_vol" in dir() else None,
    elo_calibrator=elo_cal,
    confidence_calibrator=bet_cal,
    tuning_config=tuning_cfg,
    last_game_dates=last_game_dates,
    home_starters=None,
    away_starters=None,
    live_market_spread=None,
    live_market_ml=None,
    auto_load_models=False,
)

# Highlight the four specialized outputs + ML decision
keys_first = [
    "pred_spread", "pred_margin", "cover_prob_calibrated",
    "p_home", "p_away", "predicted_winner", "ev_home", "ev_away", "ml_bet", "ml_reason",
    "pred_home_pts", "pred_away_pts",
    "pred_total", "sigma_total", "p_over", "p_under", "ou_direction",
    "direction", "actionable", "confidence_score",
]
print("\\n── Specialized market outputs ──")
for k in keys_first:
    if k in pred:
        print(f"{k:>22}: {pred[k]}")
print("\\n── Full prediction dict ──")
for k, v in pred.items():
    if k == "features":
        print(f"{k:>22}: <{len(v) if isinstance(v, dict) else '?'} features>")
        continue
    print(f"{k:>22}: {v}")

payload = {k: v for k, v in pred.items() if k != "features"}
save_json_artifact("prediction", {
    "home": "BOS",
    "away": "LAL",
    "game_date": "2026-10-28",
    "prediction": {k: (float(v) if isinstance(v, (np.floating, float)) else v) for k, v in payload.items()},
})'''),
    ]


def _default_phase_cells() -> list[dict]:
    return _workflow_phase_cells()


def build_notebook() -> dict:
    cells: list[dict] = [
        _md(MARKDOWN_INTRO),
        _code(
            "# Install dependencies (Colab). Safe to re-run.\n"
            "!pip install -q catboost optuna scikit-learn scipy pandas numpy tqdm matplotlib pyarrow lightgbm"
        ),
        _md("## Configuration & Google Drive mount"),
        _config_cell(),
        _md("## Checkpoint helpers (save / load)\n\nSet `USE_SAVED_ARTIFACTS = False` in Phase 0 to force full retrain. Per-phase `FORCE_RECOMPUTE` still overrides individual steps."),
        _code(CHECKPOINT_CELL),
        _md("## Pipeline source (inlined modules — complete)"),
    ]

    for name in MODULE_ORDER:
        cells.append(_module_cell(name))

    cells.extend(_workflow_phase_cells())
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "pygments_lexer": "ipython3"},
        },
        "cells": cells,
    }


def main():
    nb = build_notebook()
    text = json.dumps(nb, indent=1)
    for dest in (OUT, OUT_CODE, OUT_REPO):
        try:
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(text)
            print(f"Wrote {dest} ({dest.stat().st_size:,} bytes)")
        except Exception as e:
            print(f"⚠️ could not write {dest}: {e}")
    n_code = sum(1 for c in nb["cells"] if c["cell_type"] == "code")
    print(f"  cells: {len(nb['cells'])} ({n_code} code)")
    print(f"  modules inlined: {len(MODULE_ORDER)}")

    # Optional structural validation (can be slow / hang on large notebooks).
    import os
    import subprocess
    if os.environ.get("VALIDATE_NOTEBOOK", "").strip() not in ("1", "true", "True"):
        return
    r = subprocess.run(
        [sys.executable, str(ROOT / "tests" / "test_notebook_exec.py")],
        cwd=str(ROOT),
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        print("⚠️  Notebook validation failed:")
        print(r.stdout[-2000:] if r.stdout else "")
        print(r.stderr[-2000:] if r.stderr else "")
        raise SystemExit(r.returncode)
    print(r.stdout.strip())


if __name__ == "__main__":
    import sys
    main()
