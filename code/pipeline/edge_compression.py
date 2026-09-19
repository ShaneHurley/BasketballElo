"""Diagnose lean-|edge| compression vs smoke (MAE→ECE→CLV; no floor change)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from pipeline.bet_selection import (
    normalize_edge_avoid_bands,
    passes_edge_avoid_band,
    spread_side_series,
)
from pipeline.config import (
    CONFIDENCE_EDGE_SCALED,
    CONFIDENCE_MIN_EDGE,
    EDGE_AVOID_BAND,
    EDGE_AVOID_BAND_MIN_ELO_AGREE,
    MAX_QUANTILE_WIDTH,
    MIN_CONFIDENCE_SCORE,
    MIN_DISAGREEMENT_TRUST,
    MIN_EDGE_BUCKET,
    SKIP_PHANTOM_INJURY,
)
from pipeline.metrics import add_clv_columns, ats_win_series


def _season_col(df: pd.DataFrame) -> str | None:
    for c in ("simulated_season_window", "season", "SEASON"):
        if c in df.columns:
            return c
    return None


def lean_frame(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty or "MARKET_SPREAD" not in df.columns:
        return pd.DataFrame()
    m = df[pd.to_numeric(df["MARKET_SPREAD"], errors="coerce").notna()].copy()
    if m.empty or "EDGE" not in m.columns:
        return pd.DataFrame()
    side = spread_side_series(m)
    m = m[side.isin(("Home", "Away"))].copy()
    m["_side"] = side.loc[m.index]
    m["abs_edge"] = pd.to_numeric(m["EDGE"], errors="coerce").abs()
    m = m[m["abs_edge"].notna() & (m["abs_edge"] > 0)]
    if "PRED_SPREAD" in m.columns:
        pred = pd.to_numeric(m["PRED_SPREAD"], errors="coerce")
        mkt = pd.to_numeric(m["MARKET_SPREAD"], errors="coerce")
        m["pred_residual"] = pred + mkt
    if "ACTUAL_MARGIN" in m.columns:
        act = pd.to_numeric(m["ACTUAL_MARGIN"], errors="coerce")
        mkt = pd.to_numeric(m["MARKET_SPREAD"], errors="coerce")
        m["realized_residual"] = act + mkt
    return m


def edge_percentiles(series: pd.Series) -> dict[str, float]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {k: float("nan") for k in ("p25", "p50", "p75", "p90", "mean", "frac_ge_5_5")}
    return {
        "p25": float(s.quantile(0.25)),
        "p50": float(s.quantile(0.50)),
        "p75": float(s.quantile(0.75)),
        "p90": float(s.quantile(0.90)),
        "mean": float(s.mean()),
        "frac_ge_5_5": float((s >= 5.5).mean()),
    }


def season_edge_table(df: pd.DataFrame) -> pd.DataFrame:
    lean = lean_frame(df)
    if lean.empty:
        return pd.DataFrame()
    scol = _season_col(lean) or "_all"
    if scol not in lean.columns:
        lean = lean.copy()
        lean[scol] = "all"
    rows = []
    for season, sub in lean.groupby(scol, sort=True):
        ep = edge_percentiles(sub["abs_edge"])
        conf_hi = pd.to_numeric(sub.loc[sub["abs_edge"] >= 5.5, "CONFIDENCE"], errors="coerce")
        if "CONFIDENCE" not in sub.columns:
            conf_hi = pd.Series(dtype=float)
        pred_r = pd.to_numeric(sub.get("pred_residual"), errors="coerce").abs()
        real_r = pd.to_numeric(sub.get("realized_residual"), errors="coerce").abs()
        dec = pd.to_numeric(sub.get("DECISION_SPREAD"), errors="coerce")
        clo = pd.to_numeric(sub.get("CLOSING_SPREAD"), errors="coerce")
        dnc = (dec.notna() & clo.notna() & (dec != clo))
        act = sub["DIRECTION"].isin(("Home", "Away")) if "DIRECTION" in sub.columns else pd.Series(False, index=sub.index)
        rows.append({
            "season": season,
            "n_lean": int(len(sub)),
            "n_actionable": int(act.sum()),
            **{f"edge_{k}": v for k, v in ep.items()},
            "pred_resid_p50": float(pred_r.median()) if pred_r.notna().any() else np.nan,
            "realized_resid_p50": float(real_r.median()) if real_r.notna().any() else np.nan,
            "conf_on_hi_edge_mean": float(conf_hi.mean()) if len(conf_hi) else np.nan,
            "conf_on_hi_edge_p50": float(conf_hi.median()) if len(conf_hi) else np.nan,
            "n_decision_ne_close": int(dnc.sum()),
            "n_hi_edge_decision_ne_close": int((dnc & (sub["abs_edge"] >= 5.5)).sum()),
        })
    return pd.DataFrame(rows)


def gate_attrition_counts(
    df: pd.DataFrame,
    *,
    min_edge: float | None = None,
    min_conf: float | None = None,
) -> dict[str, int]:
    """Successive filters on lean rows (diagnosis; not live policy mutate)."""
    lean = lean_frame(df)
    n0 = len(lean)
    if n0 == 0:
        return {"n_lean": 0}
    min_edge = float(MIN_EDGE_BUCKET if min_edge is None else min_edge)
    min_conf = float(MIN_CONFIDENCE_SCORE if min_conf is None else min_conf)
    edge = lean["abs_edge"]
    conf = pd.to_numeric(lean.get("CONFIDENCE", lean.get("WIN_PCT")), errors="coerce")
    width = pd.to_numeric(
        lean.get("CONF_WIDTH", lean.get("SPREAD_QUANTILE_WIDTH")), errors="coerce",
    )
    trust = pd.to_numeric(lean.get("DISAGREEMENT_TRUST"), errors="coerce").fillna(1.0)
    phantom = (
        pd.to_numeric(lean["PHANTOM_INJURY_FLAG"], errors="coerce").fillna(0).astype(bool)
        if "PHANTOM_INJURY_FLAG" in lean.columns
        else pd.Series(False, index=lean.index)
    )
    agree = pd.to_numeric(lean.get("ELO_META_AGREEMENT"), errors="coerce")

    pass_edge = edge >= min_edge
    if CONFIDENCE_EDGE_SCALED:
        need = np.where(edge < 5.0, min_conf + 2.0, np.where(edge < 8.0, min_conf + 1.0, min_conf))
    else:
        need = min_conf
    pass_conf = pass_edge & conf.notna() & (conf >= need)
    if MAX_QUANTILE_WIDTH is None:
        pass_width = pass_conf
    else:
        pass_width = pass_conf & (width.isna() | (width <= float(MAX_QUANTILE_WIDTH)))
    pass_trust = pass_width & (trust >= float(MIN_DISAGREEMENT_TRUST))
    pass_phantom = pass_trust & (~phantom) if SKIP_PHANTOM_INJURY else pass_trust
    bands = normalize_edge_avoid_bands(EDGE_AVOID_BAND)
    if bands:
        keep_band = []
        for ae, ag in zip(edge, agree):
            keep_band.append(
                passes_edge_avoid_band(
                    float(ae),
                    elo_meta_agreement=None if pd.isna(ag) else float(ag),
                )
            )
        pass_band = pass_phantom & np.asarray(keep_band)
    else:
        pass_band = pass_phantom

    return {
        "n_lean": n0,
        "pass_edge": int(pass_edge.sum()),
        "pass_conf": int(pass_conf.sum()),
        "pass_width": int(pass_width.sum()),
        "pass_trust": int(pass_trust.sum()),
        "pass_phantom": int(pass_phantom.sum()),
        "pass_avoid_band": int(pass_band.sum()),
        "min_edge": min_edge,
        "min_conf": min_conf,
        "confidence_min_edge_cfg": float(CONFIDENCE_MIN_EDGE),
        "avoid_band_agree": float(EDGE_AVOID_BAND_MIN_ELO_AGREE),
    }


def confidence_floor_grid(
    df: pd.DataFrame,
    *,
    floors: tuple[float, ...] = (40.0, 45.0, 50.0, 55.0),
    min_edge: float = 5.5,
) -> pd.DataFrame:
    """Post-hoc conf floors at fixed min_edge (CSV ablation; no config write)."""
    lean = lean_frame(df)
    if lean.empty:
        return pd.DataFrame()
    rows = []
    for fl in floors:
        counts = gate_attrition_counts(df, min_edge=min_edge, min_conf=float(fl))
        n = int(counts.get("pass_avoid_band", 0))
        # Score the selected set
        edge_ok = lean["abs_edge"] >= float(min_edge)
        conf = pd.to_numeric(lean.get("CONFIDENCE", lean.get("WIN_PCT")), errors="coerce")
        if CONFIDENCE_EDGE_SCALED:
            need = np.where(
                lean["abs_edge"] < 5.0, fl + 2.0,
                np.where(lean["abs_edge"] < 8.0, fl + 1.0, fl),
            )
        else:
            need = fl
        selected = lean[edge_ok & conf.notna() & (conf >= need)].copy()
        # apply remaining gates via attrition already counted; re-filter avoid/trust/phantom
        agree = pd.to_numeric(selected.get("ELO_META_AGREEMENT"), errors="coerce")
        trust = pd.to_numeric(selected.get("DISAGREEMENT_TRUST"), errors="coerce").fillna(1.0)
        width = pd.to_numeric(
            selected.get("CONF_WIDTH", selected.get("SPREAD_QUANTILE_WIDTH")), errors="coerce",
        )
        phantom = (
            pd.to_numeric(selected["PHANTOM_INJURY_FLAG"], errors="coerce").fillna(0).astype(bool)
            if "PHANTOM_INJURY_FLAG" in selected.columns
            else pd.Series(False, index=selected.index)
        )
        keep = (trust >= float(MIN_DISAGREEMENT_TRUST)).to_numpy(dtype=bool)
        if SKIP_PHANTOM_INJURY:
            keep = keep & (~phantom.to_numpy(dtype=bool))
        if MAX_QUANTILE_WIDTH is not None:
            keep = keep & (width.isna() | (width <= float(MAX_QUANTILE_WIDTH))).to_numpy(dtype=bool)
        keep_band = np.asarray([
            passes_edge_avoid_band(float(ae), elo_meta_agreement=None if pd.isna(ag) else float(ag))
            for ae, ag in zip(selected["abs_edge"], agree)
        ], dtype=bool)
        selected = selected.loc[keep & keep_band].copy()
        selected["DIRECTION"] = selected["_side"].astype(str)
        wins = ats_win_series(selected) if len(selected) else pd.Series(dtype=bool)
        ats = float(wins.mean()) if len(wins) else np.nan
        scored = add_clv_columns(selected) if len(selected) else selected
        clv = pd.to_numeric(scored.get("CLV"), errors="coerce") if len(selected) else pd.Series(dtype=float)
        finite = clv[np.isfinite(clv)] if len(clv) else clv
        rows.append({
            "min_edge": float(min_edge),
            "min_conf": float(fl),
            "n_bets": int(len(selected)),
            "n_graded": int(len(wins)),
            "ats_pct": ats,
            "flat_roi": float(ats * (100.0 / 110.0) - (1.0 - ats)) if np.isfinite(ats) else np.nan,
            "mean_clv": float(finite.mean()) if len(finite) else np.nan,
            "n_clv": int(len(finite)),
            "attrition_pass_conf": int(counts.get("pass_conf", 0)),
            "attrition_pass_final": int(counts.get("pass_avoid_band", 0)),
        })
    return pd.DataFrame(rows)


def load_run_knobs(run_dir: Path | None) -> dict[str, Any]:
    if run_dir is None:
        return {}
    run_dir = Path(run_dir)
    out: dict[str, Any] = {"run_dir": str(run_dir)}
    for rel in (
        "artifacts/elo_calibration_knobs.json",
        "artifacts/tuning_results.json",
        "artifacts/config_snapshot.json",
    ):
        p = run_dir / rel
        if not p.is_file():
            continue
        try:
            data = json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            continue
        key = Path(rel).stem
        if key == "elo_calibration_knobs":
            out["elo_blend_alpha"] = data.get("elo_blend_alpha")
            out["elo_ridge_alpha"] = data.get("elo_ridge_alpha")
            out["huber_epsilon"] = data.get("huber_epsilon")
        elif key == "tuning_results":
            out["walkforward_edge_threshold"] = data.get("walkforward_edge_threshold")
            out["optimal_bet_edge"] = data.get("optimal_bet_edge")
        out[key] = data
    return out


def compare_runs(
    smoke_df: pd.DataFrame,
    full_df: pd.DataFrame,
    *,
    smoke_dir: Path | None = None,
    full_dir: Path | None = None,
) -> dict[str, Any]:
    smoke_seasons = season_edge_table(smoke_df)
    full_seasons = season_edge_table(full_df)
    smoke_knobs = load_run_knobs(smoke_dir)
    full_knobs = load_run_knobs(full_dir)

    # Overlap season comparison (prefer 2025-2026)
    overlap = None
    if not smoke_seasons.empty and not full_seasons.empty:
        common = set(smoke_seasons["season"]) & set(full_seasons["season"])
        prefer = "2025-2026"
        key = prefer if prefer in common else (sorted(common)[-1] if common else None)
        if key is not None:
            s = smoke_seasons[smoke_seasons["season"] == key].iloc[0].to_dict()
            f = full_seasons[full_seasons["season"] == key].iloc[0].to_dict()
            overlap = {
                "season": key,
                "smoke_edge_p50": s.get("edge_p50"),
                "full_edge_p50": f.get("edge_p50"),
                "smoke_frac_ge_5_5": s.get("edge_frac_ge_5_5"),
                "full_frac_ge_5_5": f.get("edge_frac_ge_5_5"),
                "smoke_conf_hi_mean": s.get("conf_on_hi_edge_mean"),
                "full_conf_hi_mean": f.get("conf_on_hi_edge_mean"),
                "smoke_pred_resid_p50": s.get("pred_resid_p50"),
                "full_pred_resid_p50": f.get("pred_resid_p50"),
                "smoke_n_actionable": s.get("n_actionable"),
                "full_n_actionable": f.get("n_actionable"),
            }

    smoke_attr = gate_attrition_counts(smoke_df)
    full_attr = gate_attrition_counts(full_df)

    causes = rank_causes(overlap, smoke_knobs, full_knobs, smoke_attr, full_attr)
    return {
        "smoke_knobs": {k: v for k, v in smoke_knobs.items() if k not in ("elo_calibration_knobs", "tuning_results", "config_snapshot")},
        "full_knobs": {k: v for k, v in full_knobs.items() if k not in ("elo_calibration_knobs", "tuning_results", "config_snapshot")},
        "smoke_elo_blend_alpha": smoke_knobs.get("elo_blend_alpha"),
        "full_elo_blend_alpha": full_knobs.get("elo_blend_alpha"),
        "smoke_seasons": smoke_seasons,
        "full_seasons": full_seasons,
        "overlap": overlap,
        "smoke_attrition": smoke_attr,
        "full_attrition": full_attr,
        "ranked_causes": causes,
    }


def rank_causes(
    overlap: dict | None,
    smoke_knobs: dict,
    full_knobs: dict,
    smoke_attr: dict,
    full_attr: dict,
) -> list[dict[str, Any]]:
    """Heuristic ranking: A residual hug, B Elo/calib blend, C confidence."""
    ranked: list[dict[str, Any]] = []
    s_a = smoke_knobs.get("elo_blend_alpha")
    f_a = full_knobs.get("elo_blend_alpha")
    if s_a is not None and f_a is not None and float(f_a) > float(s_a) + 0.1:
        ranked.append({
            "code": "B",
            "name": "elo_blend_overweight",
            "priority": 1,
            "detail": (
                f"full elo_blend_alpha={float(f_a):.3f} vs smoke={float(s_a):.3f}; "
                "higher blend pulls PRED toward Elo/market-calibrated anchor"
            ),
        })
    if overlap:
        sp = overlap.get("smoke_pred_resid_p50")
        fp = overlap.get("full_pred_resid_p50")
        se = overlap.get("smoke_edge_p50")
        fe = overlap.get("full_edge_p50")
        if (
            sp is not None and fp is not None
            and np.isfinite(sp) and np.isfinite(fp)
            and float(fp) + 0.5 < float(sp)
        ) or (
            se is not None and fe is not None
            and np.isfinite(se) and np.isfinite(fe)
            and float(fe) + 1.0 < float(se)
        ):
            ranked.append({
                "code": "A",
                "name": "residual_underprediction",
                "priority": 1 if not ranked else 2,
                "detail": (
                    f"overlap pred_resid_p50 smoke={sp} full={fp}; "
                    f"edge_p50 smoke={se} full={fe}"
                ),
            })
    # Confidence: large drop from pass_edge → pass_conf
    for label, attr in (("smoke", smoke_attr), ("full", full_attr)):
        pe, pc = attr.get("pass_edge", 0), attr.get("pass_conf", 0)
        if pe and pc / max(pe, 1) < 0.35:
            ranked.append({
                "code": "C",
                "name": "confidence_below_floor",
                "priority": 2 if label == "full" else 3,
                "detail": (
                    f"{label}: pass_edge={pe} → pass_conf={pc} "
                    f"(min_conf={attr.get('min_conf')})"
                ),
            })
            break
    ranked.sort(key=lambda r: r["priority"])
    # Deduplicate by code keeping best priority
    seen = set()
    out = []
    for r in ranked:
        if r["code"] in seen:
            continue
        seen.add(r["code"])
        out.append(r)
    if not out:
        out.append({
            "code": "unknown",
            "name": "inconclusive",
            "priority": 99,
            "detail": "No clear A/B/C signal; inspect season tables manually",
        })
    return out


def print_report(report: dict[str, Any], *, conf_grid: pd.DataFrame | None = None) -> None:
    print("\n" + "=" * 64)
    print("EDGE COMPRESSION DIAGNOSIS (smoke vs full)")
    print("=" * 64)
    print(
        f"elo_blend_alpha: smoke={report.get('smoke_elo_blend_alpha')}  "
        f"full={report.get('full_elo_blend_alpha')}"
    )
    ov = report.get("overlap") or {}
    if ov:
        print(f"\nOverlap season {ov.get('season')}:")
        print(
            f"  edge_p50 smoke={ov.get('smoke_edge_p50')} full={ov.get('full_edge_p50')}  "
            f"frac≥5.5 smoke={ov.get('smoke_frac_ge_5_5')} full={ov.get('full_frac_ge_5_5')}"
        )
        print(
            f"  pred_resid_p50 smoke={ov.get('smoke_pred_resid_p50')} "
            f"full={ov.get('full_pred_resid_p50')}"
        )
        print(
            f"  conf_hi_mean smoke={ov.get('smoke_conf_hi_mean')} "
            f"full={ov.get('full_conf_hi_mean')}"
        )
        print(
            f"  n_actionable smoke={ov.get('smoke_n_actionable')} "
            f"full={ov.get('full_n_actionable')}"
        )
    print("\nGate attrition (full):", report.get("full_attrition"))
    print("Gate attrition (smoke):", report.get("smoke_attrition"))
    print("\nRanked causes:")
    for c in report.get("ranked_causes") or []:
        print(f"  [{c['code']}] {c['name']}: {c['detail']}")
    fs = report.get("full_seasons")
    if isinstance(fs, pd.DataFrame) and not fs.empty:
        print("\nFull seasons:")
        print(fs.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    ss = report.get("smoke_seasons")
    if isinstance(ss, pd.DataFrame) and not ss.empty:
        print("\nSmoke seasons:")
        print(ss.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    if conf_grid is not None and not conf_grid.empty:
        print("\nConfidence floor grid (min_edge=5.5, full CSV):")
        print(conf_grid.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(
        "\nDo not lower MIN_EDGE_BUCKET from this report. "
        "Next: cap Elo blend or fix residual expressiveness, then re-smoke."
    )
