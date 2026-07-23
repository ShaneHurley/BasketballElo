"""Task 002: record the invalid pre-integrity-fix baseline as a quarantined,
explicitly-untrusted diagnostic manifest.

The files under ``newest data/`` (``backtest_results.csv``,
``tuning_results.json``) were produced before the Stage 0/1 integrity fixes
in this phase. They must never be read again as a source of truth for
accuracy or ROI. This module only extracts descriptive counts (row/column
counts, date range, config knobs actually used) and enumerates the already
known defects; it never asserts the accuracy/ROI numbers are valid.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

MANIFEST_LABEL = "invalid_pre_integrity_fix"

KNOWN_DEFECTS = [
    {
        "id": "score_corruption",
        "description": (
            "ACTUAL_HOME/ACTUAL_AWAY are summed from garbage-time-zeroed, "
            "possession-filtered stints instead of canonical raw final scores. "
            "~1,090/1,307 2025-26 games understated by ~17.8 combined points on average."
        ),
    },
    {
        "id": "date_corruption",
        "description": (
            "Seven 2025-26 games with M/D/YYYY raw dates become NaT under naive "
            "pd.to_datetime and then receive a fabricated prev_date + 1 day date."
        ),
    },
    {
        "id": "close_line_conflation",
        "description": (
            "MARKET_SPREAD == CLOSING_SPREAD for every lined row in this export "
            "(confirmed via manifest check below), so T-60/close cannot be "
            "distinguished and CLV is identically zero."
        ),
    },
    {
        "id": "stack_cv_future_leakage",
        "description": "ChronologicalPartitionCV stack folds can see future rows (pipeline/cv.py).",
    },
    {
        "id": "clv_definition_bug",
        "description": "CLV defined from change in model-edge magnitude rather than bet-side line improvement (pipeline/metrics.py).",
    },
    {
        "id": "push_as_loss_bug",
        "description": "Pushes are treated as losses in several grading paths instead of returning stake with zero profit.",
    },
    {
        "id": "cap_after_profit_bug",
        "description": "Profit is computed before slate/correlation stake caps are applied in some paths (fast-path bypass in pipeline/metrics.py).",
    },
]


def _safe_read_backtest(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception:
        return None


def _safe_read_tuning(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def build_invalid_baseline_manifest(
    backtest_csv_path: Path | str,
    tuning_json_path: Path | str,
) -> dict[str, Any]:
    """Return a manifest of descriptive counts + known defects.

    This function NEVER reports accuracy/ROI numbers as trustworthy; any such
    values are captured only as "raw_untrusted_value" fields alongside an
    explicit warning.
    """
    backtest_path = Path(backtest_csv_path)
    tuning_path = Path(tuning_json_path)

    df = _safe_read_backtest(backtest_path)
    tuning = _safe_read_tuning(tuning_path)

    manifest: dict[str, Any] = {
        "label": MANIFEST_LABEL,
        "trust_statement": (
            "UNTRUSTED: score labels, dates, CLV, and ROI in this manifest and its "
            "source files are NOT valid measurements of model accuracy or "
            "profitability. They are quarantined diagnostic artifacts only, "
            "captured to document known defects before repair."
        ),
        "source_files": {
            "backtest_csv": str(backtest_path),
            "tuning_json": str(tuning_path),
            "backtest_csv_exists": backtest_path.exists(),
            "tuning_json_exists": tuning_path.exists(),
        },
        "known_defects": KNOWN_DEFECTS,
        "untrusted_fields": ["ACTUAL_HOME", "ACTUAL_AWAY", "ACTUAL_MARGIN", "DATE",
                              "MARKET_SPREAD", "CLOSING_SPREAD", "CLV", "ROI",
                              "blind_ats_roi", "blind_roi_ci"],
    }

    if df is not None and not df.empty:
        close_equals_market = None
        if "MARKET_SPREAD" in df.columns and "CLOSING_SPREAD" in df.columns:
            lined = df.dropna(subset=["MARKET_SPREAD", "CLOSING_SPREAD"])
            close_equals_market = bool(
                len(lined) > 0 and (lined["MARKET_SPREAD"] == lined["CLOSING_SPREAD"]).all()
            )
        manifest["backtest_descriptive_counts"] = {
            "n_rows": int(len(df)),
            "n_columns": int(len(df.columns)),
            "n_unique_game_ids": int(df["GAME_ID"].nunique()) if "GAME_ID" in df.columns else None,
            "date_min": str(df["DATE"].min()) if "DATE" in df.columns else None,
            "date_max": str(df["DATE"].max()) if "DATE" in df.columns else None,
            "n_actionable_rows_raw_untrusted": (
                int(df["ACTIONABLE"].sum()) if "ACTIONABLE" in df.columns else None
            ),
            "close_line_equals_market_line_for_all_lined_rows": close_equals_market,
        }
    else:
        manifest["backtest_descriptive_counts"] = None

    if tuning:
        manifest["tuning_descriptive_fields_raw_untrusted"] = {
            "optimal_bet_edge": tuning.get("optimal_bet_edge"),
            "recommended_profile": tuning.get("recommended_profile"),
            "blind_ats_roi_raw_untrusted_value": tuning.get("blind_ats_roi"),
            "blind_roi_ci_raw_untrusted_value": tuning.get("blind_roi_ci"),
        }
    else:
        manifest["tuning_descriptive_fields_raw_untrusted"] = None

    return manifest


def write_invalid_baseline_manifest(
    backtest_csv_path: Path | str,
    tuning_json_path: Path | str,
    out_path: Path | str,
) -> Path:
    manifest = build_invalid_baseline_manifest(backtest_csv_path, tuning_json_path)
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2, default=str))
    return out


if __name__ == "__main__":
    from pipeline.config import ROOT as DATA_ROOT

    repo_root = Path(__file__).resolve().parent.parent
    backtest_csv = repo_root / "newest data" / "backtest_results.csv"
    tuning_json = repo_root / "newest data" / "tuning_results.json"
    out = repo_root / "state" / "invalid_baseline_manifest.json"
    write_invalid_baseline_manifest(backtest_csv, tuning_json, out)
    print(f"Wrote invalid-baseline manifest to {out}")
