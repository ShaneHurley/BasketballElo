#!/usr/bin/env python3
"""Run live promotion gates on a backtest_results.csv (subprocess helper).

Avoids importing pipeline from the thin dashboard venv (tqdm/sklearn).
Uses the current ``pipeline.promotion_gates`` API — not the deleted
``ats_promote`` / ``ml_promote`` / ``total_promote`` wrappers.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from pipeline.promotion_gates import (  # noqa: E402
    beats_market_and_baseline,
    clv_promotion_allowed,
    multi_season_gate,
    policy_retune_allowed,
    tip_proxy_roi_blocked,
)


def _ensure_ats_win(df: pd.DataFrame) -> pd.DataFrame:
    if "ATS_WIN" in df.columns:
        return df
    spread_col = None
    for c in ("DECISION_SPREAD", "MARKET_SPREAD", "CLOSING_SPREAD"):
        if c in df.columns:
            spread_col = c
            break
    if spread_col is None or "ACTUAL_MARGIN" not in df.columns:
        return df
    margin = pd.to_numeric(df["ACTUAL_MARGIN"], errors="coerce")
    spread = pd.to_numeric(df[spread_col], errors="coerce")
    cover = margin + spread
    if "EDGE_LEAN" in df.columns:
        side = df["EDGE_LEAN"].astype(str)
    elif "DIRECTION" in df.columns:
        side = df["DIRECTION"].astype(str)
    elif "EDGE" in df.columns:
        e = pd.to_numeric(df["EDGE"], errors="coerce")
        side = pd.Series(np.where(e > 0, "Home", np.where(e < 0, "Away", "Pass")), index=df.index)
    else:
        return df
    home_win = cover > 1e-9
    away_win = cover < -1e-9
    ats = pd.Series(np.nan, index=df.index, dtype=float)
    home_mask = side.str.lower().isin(("home", "h"))
    away_mask = side.str.lower().isin(("away", "a"))
    ats.loc[home_mask & home_win] = 1.0
    ats.loc[home_mask & away_win] = 0.0
    ats.loc[away_mask & away_win] = 1.0
    ats.loc[away_mask & home_win] = 0.0
    out = df.copy()
    out["ATS_WIN"] = ats
    return out


def _finite_mean(series: pd.Series) -> float:
    v = pd.to_numeric(series, errors="coerce")
    m = float(v.mean()) if v.notna().any() else float("nan")
    return m


def main() -> int:
    p = argparse.ArgumentParser(description="Promotion gates scorecard")
    p.add_argument("--results", required=True, help="Path to backtest_results.csv")
    p.add_argument("--out", required=True, help="Output JSON path")
    args = p.parse_args()

    path = Path(args.results)
    if not path.is_file():
        print(f"CSV not found: {path}", file=sys.stderr)
        return 1

    df = _ensure_ats_win(pd.read_csv(path, low_memory=False))
    if "simulated_season_window" in df.columns:
        season_col = "simulated_season_window"
    elif "DATE" in df.columns:
        df = df.copy()
        df["_season"] = pd.to_datetime(df["DATE"], errors="coerce").dt.year
        season_col = "_season"
    else:
        season_col = None

    spread_mae = (
        _finite_mean(df["ABS_ERR"]) if "ABS_ERR" in df.columns
        else _finite_mean(df["SPREAD_ERR"]) if "SPREAD_ERR" in df.columns
        else float("nan")
    )
    market_mae = (
        _finite_mean(df["MARKET_ABS_ERR"]) if "MARKET_ABS_ERR" in df.columns
        else float("nan")
    )
    brier_model = 0.25
    if "CALIBRATED_COVER_PROB" in df.columns and "ATS_WIN" in df.columns:
        pr = pd.to_numeric(df["CALIBRATED_COVER_PROB"], errors="coerce")
        y = pd.to_numeric(df["ATS_WIN"], errors="coerce")
        m = pr.notna() & y.notna()
        if m.any():
            brier_model = float(np.mean((pr[m] - y[m]) ** 2))

    candidate = {
        "spread_mae": spread_mae,
        "market_spread_mae": market_mae,
        "brier": brier_model,
        "interval_coverage": _finite_mean(df["INTERVAL_HIT"]) if "INTERVAL_HIT" in df.columns else float("nan"),
        "market_error_corr": float("nan"),
        "margin_dispersion_ratio": float("nan"),
    }
    baseline = {
        "spread_mae": market_mae if np.isfinite(market_mae) else spread_mae,
        "market_spread_mae": market_mae,
        "brier": 0.25,
    }

    per_season_c: list[dict] = []
    per_season_b: list[dict] = []
    if season_col is not None and "ABS_ERR" in df.columns:
        for _, g in df.groupby(season_col, dropna=True):
            per_season_c.append({"spread_mae": _finite_mean(g["ABS_ERR"])})
            if "MARKET_ABS_ERR" in g.columns:
                per_season_b.append({"spread_mae": _finite_mean(g["MARKET_ABS_ERR"])})
            else:
                per_season_b.append({"spread_mae": _finite_mean(g["ABS_ERR"])})
    multi = multi_season_gate(per_season_c, per_season_b) if per_season_c else {
        "wins": 0, "n_seasons": 0, "collapses": 0, "passed": False, "mean_delta": float("nan"),
    }

    n_finite_clv = 0
    if "POINT_CLV" in df.columns:
        n_finite_clv = int(pd.to_numeric(df["POINT_CLV"], errors="coerce").notna().sum())

    provenance = None
    prov_path = path.parent / "odds_provenance.json"
    if not prov_path.is_file():
        prov_path = path.parent / "artifacts" / "odds_provenance.json"
    if prov_path.is_file():
        try:
            provenance = json.loads(prov_path.read_text())
        except json.JSONDecodeError:
            provenance = None

    out = {
        "beats_market_and_baseline": beats_market_and_baseline(candidate, baseline),
        "multi_season_gate": multi,
        "clv_promotion_allowed": clv_promotion_allowed(provenance, n_finite_clv),
        "policy_retune_allowed": policy_retune_allowed(candidate, odds_provenance=provenance),
        "tip_proxy_roi_blocked": tip_proxy_roi_blocked(provenance),
        "n_finite_clv": n_finite_clv,
        "candidate": candidate,
        "baseline": baseline,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, default=str))
    print(f"Wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
