"""Model-health summaries for the Documentation tab (Epic 7.5).

Wraps ``pipeline.monitoring`` and ``pipeline.negative_controls`` without
editing those modules. Never invents zeros for missing MAE / ATS / CLV.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from dashboard.paths import results_csv_path, run_dir_for_id
from dashboard.services.runs import load_odds_provenance

# Cap so /api/model-health stays snappy on large CSVs.
_N_PERM = 20
_WEEKLY_WINDOW = 30

# Candidate column pairs for residual MAE (first match wins).
_PRED_COLS = ("PRED_SPREAD", "PRED_MARGIN", "pred_spread", "pred_margin")
_ACTUAL_COLS = ("ACTUAL_MARGIN", "actual_margin")


def _json_num(x: Any) -> float | int | None:
    """Serialize floats; map NaN/inf to null (never coerce to 0)."""
    if x is None:
        return None
    try:
        if isinstance(x, (bool, np.bool_)):
            return bool(x)
        if isinstance(x, (int, np.integer)):
            return int(x)
        v = float(x)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(v):
        return None
    return v


def _sanitize(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return _json_num(obj)
    if isinstance(obj, (np.integer, int)) and not isinstance(obj, bool):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if obj is None:
        return None
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


def _resolve_margin_cols(df: pd.DataFrame) -> tuple[str | None, str | None]:
    pred = next((c for c in _PRED_COLS if c in df.columns), None)
    actual = next((c for c in _ACTUAL_COLS if c in df.columns), None)
    if pred and actual:
        return pred, actual
    # Derive actual margin from home/away scores when present.
    if pred and "ACTUAL_HOME" in df.columns and "ACTUAL_AWAY" in df.columns:
        return pred, "__derived_margin__"
    return pred, actual


def _actual_margin_series(df: pd.DataFrame, actual_col: str) -> pd.Series:
    if actual_col == "__derived_margin__":
        return (
            pd.to_numeric(df["ACTUAL_HOME"], errors="coerce")
            - pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")
        )
    return pd.to_numeric(df[actual_col], errors="coerce")


def _find_feature_tables(run_id: str | None) -> tuple[Path | None, Path | None]:
    """Locate optional train/live feature CSVs under the run dir."""
    rd = run_dir_for_id(run_id) if run_id else None
    if rd is None:
        return None, None
    train_candidates = (
        rd / "artifacts" / "train_features.csv",
        rd / "artifacts" / "features_train.csv",
        rd / "train_features.csv",
    )
    live_candidates = (
        rd / "artifacts" / "live_features.csv",
        rd / "artifacts" / "features_live.csv",
        rd / "live_features.csv",
    )
    train = next((p for p in train_candidates if p.is_file()), None)
    live = next((p for p in live_candidates if p.is_file()), None)
    return train, live


def load_model_health(
    run_id: str | None = None,
    *,
    window: int = _WEEKLY_WINDOW,
    n_perm: int = _N_PERM,
) -> dict[str, Any]:
    """Build the Documentation-tab health payload for a run (or latest)."""
    rid = None if run_id in (None, "", "latest") else run_id
    path = results_csv_path(rid)
    provenance = load_odds_provenance(rid) or {}
    quote_source = provenance.get("quote_source")
    tip_proxy = (
        quote_source == "tip_proxy"
        or bool(provenance.get("uses_tip_proxy"))
        or bool(provenance.get("used_tip_proxy_fallback"))
        or bool(provenance.get("tip_proxy"))
    )
    research_only = tip_proxy or provenance.get("promotion_eligible") is False

    out: dict[str, Any] = {
        "run_id": rid,
        "results_path": str(path) if path else None,
        "n_rows": 0,
        "columns_used": [],
        "columns_missing": [],
        "weekly": {"status": "no_data"},
        "permutation": {
            "status": "skipped",
            "reason": "no results CSV",
        },
        "drift": {
            "status": "unavailable",
            "reason": "Feature tables were not saved for this run — drift cannot be computed.",
            "rows": [],
        },
        "quote_source": quote_source,
        "research_only": bool(research_only),
        "banner_message": (
            "Research-only: quote_source=tip_proxy. Do not use for promotion ROI."
            if tip_proxy
            else (
                "Research-only: odds provenance is not promotion-eligible."
                if research_only
                else None
            )
        ),
        "window": int(window),
    }

    if path is None or not path.exists():
        out["columns_missing"] = list(_PRED_COLS[:1]) + list(_ACTUAL_COLS[:1])
        return _sanitize(out)

    try:
        df = pd.read_csv(path, nrows=50_000)
    except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError):
        out["permutation"] = {
            "status": "skipped",
            "reason": "results CSV could not be read",
        }
        return _sanitize(out)

    out["n_rows"] = int(len(df))
    if df.empty:
        return _sanitize(out)

    # ---- weekly report ----
    from pipeline.monitoring import weekly_report

    try:
        weekly = weekly_report(df, window=window)
    except (KeyError, ValueError, TypeError) as exc:
        # monitoring.weekly_report assumes betting columns (e.g. MARKET_SPREAD).
        # Incomplete CSVs must not crash the dashboard — surface an honest skip.
        weekly = {
            "status": "no_data",
            "detail": f"weekly_report unavailable: {exc}",
        }
        pred_col_w, actual_col_w = _resolve_margin_cols(df)
        if pred_col_w and actual_col_w:
            tail = df.tail(window)
            y_pred = pd.to_numeric(tail[pred_col_w], errors="coerce")
            y_true = _actual_margin_series(tail, actual_col_w)
            mask = y_pred.notna() & y_true.notna()
            if mask.any():
                weekly = {
                    "n_games": int(len(tail)),
                    "n_bets": 0,
                    "ats_pct": None,
                    "ats_ci_lo": None,
                    "ats_ci_hi": None,
                    "spread_mae": _json_num((y_pred[mask] - y_true[mask]).abs().mean()),
                    "mean_clv": None,
                    "alert_underperformance": False,
                    "detail": f"partial weekly (no ATS columns): {exc}",
                }
    out["weekly"] = _sanitize(weekly)

    # ---- permutation null on spread residual MAE ----
    pred_col, actual_col = _resolve_margin_cols(df)
    used: list[str] = []
    missing: list[str] = []
    if pred_col:
        used.append(pred_col)
    else:
        missing.append("PRED_SPREAD (or PRED_MARGIN)")
    if actual_col == "__derived_margin__":
        used.extend(["ACTUAL_HOME", "ACTUAL_AWAY"])
    elif actual_col:
        used.append(actual_col)
    else:
        missing.append("ACTUAL_MARGIN (or ACTUAL_HOME/ACTUAL_AWAY)")

    if "CLV" in df.columns:
        used.append("CLV")
    else:
        missing.append("CLV")

    out["columns_used"] = used
    out["columns_missing"] = missing

    if pred_col and actual_col:
        from pipeline.negative_controls import shuffled_outcomes_destroy_edge

        y_pred = pd.to_numeric(df[pred_col], errors="coerce")
        y_true = _actual_margin_series(df, actual_col)
        mask = y_pred.notna() & y_true.notna()
        yp = y_pred[mask].to_numpy(dtype=float)
        yt = y_true[mask].to_numpy(dtype=float)
        if len(yp) < 5:
            out["permutation"] = {
                "status": "skipped",
                "reason": f"need at least 5 finite residual pairs (have {len(yp)})",
            }
        else:
            def mae(a: np.ndarray, b: np.ndarray) -> float:
                return float(np.mean(np.abs(a - b)))

            ctrl = shuffled_outcomes_destroy_edge(
                yt, yp, metric_fn=mae, n_perm=n_perm, seed=0, improve_when_lower=True
            )
            out["permutation"] = {
                "status": "ok",
                "passed": bool(ctrl["passed"]),
                "base_metric": _json_num(ctrl.get("base_metric")),
                "null_mean": _json_num(ctrl.get("null_mean")),
                "detail": ctrl.get("detail"),
                "n_perm": int(n_perm),
                "n_pairs": int(len(yp)),
                "metric": "spread_residual_mae",
            }
    else:
        out["permutation"] = {
            "status": "skipped",
            "reason": "missing spread prediction / actual columns: "
            + ", ".join(missing),
        }

    # ---- feature drift (optional artifacts) ----
    train_path, live_path = _find_feature_tables(rid)
    if train_path is not None and live_path is not None:
        from pipeline.monitoring import feature_drift

        try:
            train_df = pd.read_csv(train_path)
            live_df = pd.read_csv(live_path)
            drifted = feature_drift(train_df, live_df)
            rows = []
            if drifted is not None and not drifted.empty:
                rows = [
                    {
                        "feature": str(r["feature"]),
                        "train_mean": _json_num(r["train_mean"]),
                        "live_mean": _json_num(r["live_mean"]),
                        "z_score": _json_num(r["z_score"]),
                    }
                    for _, r in drifted.iterrows()
                ]
            out["drift"] = {
                "status": "ok",
                "reason": None,
                "train_path": str(train_path),
                "live_path": str(live_path),
                "rows": rows,
            }
        except (OSError, pd.errors.EmptyDataError, pd.errors.ParserError, ValueError) as exc:
            out["drift"] = {
                "status": "unavailable",
                "reason": f"feature tables present but unreadable: {exc}",
                "rows": [],
            }
    else:
        out["drift"] = {
            "status": "unavailable",
            "reason": (
                "Feature tables were not saved for this run — drift cannot be computed."
            ),
            "rows": [],
        }

    return _sanitize(out)
