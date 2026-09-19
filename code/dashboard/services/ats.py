"""ATS tab helpers: presets and scorecard metrics."""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from dashboard.jobs.util import ensure_ats_win
from dashboard.paths import results_csv_path


ATS_PRESETS = {
    "calibration_mode": ["legacy_stack", "simplified", "beta_ats"],
    "confidence_method": ["auto", "isotonic", "platt", "logistic_l2", "logistic_l1"],
    "bet_selection_mode": ["edge", "ev", "hybrid"],
}


def _actionable_mask(df: pd.DataFrame) -> pd.Series:
    if "ACTIONABLE" in df.columns:
        a = pd.to_numeric(df["ACTIONABLE"], errors="coerce").fillna(0)
        return a.astype(float) > 0.5
    if "DIRECTION" in df.columns:
        side = df["DIRECTION"].astype(str).str.lower()
        return ~side.isin(("pass", "nan", "none", ""))
    return pd.Series(False, index=df.index)


def ats_scorecard(run_id: str | None) -> dict[str, Any]:
    path = results_csv_path(run_id)
    out: dict[str, Any] = {
        "run_id": run_id,
        "metrics": {},
        "by_season": [],
        "push_rate": None,
        "populations": {},
        "warnings": [],
    }
    if path is None or not path.exists():
        return out
    df = ensure_ats_win(pd.read_csv(path))
    act_mask = _actionable_mask(df)
    lean_y = pd.to_numeric(df["ATS_WIN"], errors="coerce") if "ATS_WIN" in df.columns else None
    n_lean = int(lean_y.notna().sum()) if lean_y is not None else 0
    n_actionable = int(act_mask.sum())
    out["populations"] = {
        "n_games": int(len(df)),
        "n_lean_graded": n_lean,
        "n_actionable": n_actionable,
        "note": (
            "Primary quality = paired MAE / model−market MAE / market-error corr / "
            "interval coverage. Lean ATS hit and lean EV are diagnostics only."
        ),
    }

    # --- Primary accuracy diagnostics (ALL games) ---
    if "PRED_SPREAD" in df.columns and "ACTUAL_MARGIN" in df.columns:
        pred = pd.to_numeric(df["PRED_SPREAD"], errors="coerce")
        act = pd.to_numeric(df["ACTUAL_MARGIN"], errors="coerce")
        out["metrics"]["spread_mae"] = float((pred - act).abs().mean())
        line = df["DECISION_SPREAD"] if "DECISION_SPREAD" in df.columns else df.get("MARKET_SPREAD")
        if line is not None:
            mkt = -pd.to_numeric(line, errors="coerce")
            mkt_mae = float((mkt - act).abs().mean())
            out["metrics"]["market_spread_mae"] = mkt_mae
            out["metrics"]["model_minus_market_spread_mae"] = float(
                out["metrics"]["spread_mae"] - mkt_mae
            )
            mean_abs_pred = float(pred.abs().mean())
            mean_abs_mkt = float(mkt.abs().mean())
            out["metrics"]["margin_dispersion_ratio"] = (
                mean_abs_pred / mean_abs_mkt if mean_abs_mkt > 1e-9 else None
            )
            edge = pred - mkt
            realized = act - mkt
            mm = edge.notna() & realized.notna()
            if int(mm.sum()) >= 30:
                out["metrics"]["market_error_corr"] = float(edge[mm].corr(realized[mm]))
            mm2 = pred.notna() & mkt.notna()
            if int(mm2.sum()) >= 30 and float(mkt[mm2].std()) > 1e-9:
                out["metrics"]["margin_vs_market_slope"] = float(
                    np.polyfit(mkt[mm2], pred[mm2], 1)[0]
                )
                out["metrics"]["margin_within2_pct"] = float((pred[mm2].abs() <= 2.0).mean())

    if {"PRED_HOME", "ACTUAL_HOME", "PRED_AWAY", "ACTUAL_AWAY"}.issubset(df.columns):
        ph = pd.to_numeric(df["PRED_HOME"], errors="coerce")
        pa = pd.to_numeric(df["PRED_AWAY"], errors="coerce")
        ah = pd.to_numeric(df["ACTUAL_HOME"], errors="coerce")
        aa = pd.to_numeric(df["ACTUAL_AWAY"], errors="coerce")
        out["metrics"]["home_mae"] = float((ph - ah).abs().mean())
        out["metrics"]["away_mae"] = float((pa - aa).abs().mean())
        out["metrics"]["paired_score_mae"] = float(
            0.5 * ((ph - ah).abs() + (pa - aa).abs()).mean()
        )
        out["populations"]["score_accuracy"] = "all_games"
    if "FORECAST_SOURCE" in df.columns and len(df):
        mode = df["FORECAST_SOURCE"].astype(str).mode()
        out["metrics"]["forecast_source"] = str(mode.iloc[0]) if len(mode) else None

    if {"CONF_LOWER", "CONF_UPPER", "ACTUAL_MARGIN"}.issubset(df.columns):
        lo = pd.to_numeric(df["CONF_LOWER"], errors="coerce")
        hi = pd.to_numeric(df["CONF_UPPER"], errors="coerce")
        act = pd.to_numeric(df["ACTUAL_MARGIN"], errors="coerce")
        m = lo.notna() & hi.notna() & act.notna()
        if m.any():
            inside = (act[m] >= lo[m]) & (act[m] <= hi[m])
            out["metrics"]["interval_coverage"] = float(inside.mean())
            out["metrics"]["mean_conf_width"] = float((hi[m] - lo[m]).mean())
    # Prefer explicit 80% margin bands when present.
    for lo_c, hi_c, key in (
        ("MARGIN_QLO_80", "MARGIN_QHI_80", "interval_coverage_80"),
        ("SPREAD_Q10", "SPREAD_Q90", "interval_coverage_80"),
        ("MARGIN_QLO_50", "MARGIN_QHI_50", "interval_coverage_50"),
        ("MARGIN_QLO_95", "MARGIN_QHI_95", "interval_coverage_95"),
    ):
        if {lo_c, hi_c, "ACTUAL_MARGIN"}.issubset(df.columns):
            lo = pd.to_numeric(df[lo_c], errors="coerce")
            hi = pd.to_numeric(df[hi_c], errors="coerce")
            act = pd.to_numeric(df["ACTUAL_MARGIN"], errors="coerce")
            m = lo.notna() & hi.notna() & act.notna()
            if int(m.sum()) >= 20:
                out["metrics"][key] = float(((act[m] >= lo[m]) & (act[m] <= hi[m])).mean())
                if key == "interval_coverage_80":
                    out["metrics"]["interval_coverage"] = out["metrics"][key]
                    out["metrics"]["mean_conf_width"] = float((hi[m] - lo[m]).mean())
                    break

    # --- Betting populations (explicitly separated) ---
    out["metrics"]["n_lean_graded"] = n_lean
    out["metrics"]["n_actionable"] = n_actionable
    scored = df.loc[act_mask] if n_actionable > 0 else df.iloc[0:0]
    if n_actionable > 0 and "ATS_WIN" in scored.columns:
        y = pd.to_numeric(scored["ATS_WIN"], errors="coerce")
        out["metrics"]["ats_hit_actionable"] = float(y.mean()) if y.notna().any() else None
        out["metrics"]["n_decided"] = int(y.notna().sum())
        out["metrics"]["population"] = "actionable"
        out["metrics"]["ats_hit"] = out["metrics"]["ats_hit_actionable"]
    else:
        out["metrics"]["ats_hit_actionable"] = None
        out["metrics"]["n_decided"] = 0
        out["metrics"]["population"] = "no_actionable"
        out["metrics"]["ats_hit"] = None
        out["warnings"].append(
            "No actionable bets — do not treat lean hit/EV as stakeable edge."
        )

    if lean_y is not None:
        out["metrics"]["ats_hit_lean"] = float(lean_y.mean()) if lean_y.notna().any() else None

    if "ATS_PUSH" in df.columns:
        out["push_rate"] = float(pd.to_numeric(df["ATS_PUSH"], errors="coerce").mean())

    # Edge/EV: actionable-only when available; lean EV flagged as diagnostic.
    if n_actionable > 0 and "EDGE" in scored.columns:
        out["metrics"]["mean_abs_edge"] = float(
            pd.to_numeric(scored["EDGE"], errors="coerce").abs().mean()
        )
    elif "EDGE" in df.columns:
        out["metrics"]["mean_abs_edge_lean"] = float(
            pd.to_numeric(df["EDGE"], errors="coerce").abs().mean()
        )
        out["metrics"]["mean_abs_edge"] = out["metrics"]["mean_abs_edge_lean"]

    if n_actionable > 0 and ("ATS_EV" in scored.columns or "ATS_EV_HOME" in scored.columns):
        col = "ATS_EV" if "ATS_EV" in scored.columns else "ATS_EV_HOME"
        out["metrics"]["mean_ats_ev"] = float(pd.to_numeric(scored[col], errors="coerce").mean())
        out["metrics"]["ev_population"] = "actionable"
    elif "ATS_EV" in df.columns or "ATS_EV_HOME" in df.columns:
        col = "ATS_EV" if "ATS_EV" in df.columns else "ATS_EV_HOME"
        out["metrics"]["mean_ats_ev_lean"] = float(pd.to_numeric(df[col], errors="coerce").mean())
        out["metrics"]["mean_ats_ev"] = None  # do not promote lean EV as primary
        out["metrics"]["ev_population"] = "lean_diagnostic_only"
        out["warnings"].append(
            "Lean ATS EV is diagnostic only and often inflated; not a staking signal."
        )

    if "CLV" in df.columns:
        clv = pd.to_numeric(df["CLV"], errors="coerce")
        out["metrics"]["n_finite_clv"] = int(clv.notna().sum())
    else:
        out["metrics"]["n_finite_clv"] = 0

    # CLV eligibility from run provenance if present alongside results.
    out["metrics"]["clv_claim_allowed"] = False
    try:
        from dashboard.paths import run_dir_for_id, read_json_if_exists
        rd = run_dir_for_id(run_id) if run_id else None
        prov = None
        if rd is not None:
            for rel in ("odds_provenance.json", "checkpoints/odds_provenance.json", "manifest.json"):
                p = rd / rel
                data = read_json_if_exists(p)
                if isinstance(data, dict):
                    if "promotion_eligible" in data:
                        prov = data
                        break
                    if "odds_provenance" in data and isinstance(data["odds_provenance"], dict):
                        prov = data["odds_provenance"]
                        break
        if prov and prov.get("promotion_eligible") and int(out["metrics"].get("n_finite_clv") or 0) >= 200:
            out["metrics"]["clv_claim_allowed"] = True
        elif int(out["metrics"].get("n_finite_clv") or 0) == 0:
            out["warnings"].append("No finite CLV rows — ROI claims blocked.")
    except Exception:
        pass

    # Edge evidence flag for UI.
    mec = out["metrics"].get("market_error_corr")
    if mec is None or (isinstance(mec, float) and (not np.isfinite(mec) or mec <= 0.02)):
        out["warnings"].append(
            "Market-error correlation ≤ 0 — predicted edge is not informative yet."
        )
    cov = out["metrics"].get("interval_coverage")
    if cov is not None and np.isfinite(cov) and abs(float(cov) - 0.80) > 0.12:
        out["warnings"].append(
            f"Interval coverage {float(cov):.2f} far from ~0.80 — do not loosen width gates."
        )

    season = "simulated_season_window" if "simulated_season_window" in df.columns else None
    if season and "ATS_WIN" in df.columns:
        for s, g in df.groupby(season):
            g_act = g.loc[_actionable_mask(g)]
            y = pd.to_numeric(
                g_act["ATS_WIN"] if len(g_act) else g["ATS_WIN"],
                errors="coerce",
            )
            out["by_season"].append({
                "season": s,
                "n": int(y.notna().sum()),
                "n_actionable": int(len(g_act)),
                "hit": float(y.mean()) if y.notna().any() else None,
            })
    return out
