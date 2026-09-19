"""Offline validation of upset/blowout/calibration plan against saved backtest CSV."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from pipeline.bet_selection import (
    actionable_min_edge,
    normalize_edge_avoid_bands,
    passes_confidence_actionable_gates,
    passes_edge_avoid_band,
)
from pipeline.calibration_metrics import compute_ece
from pipeline.config import (
    CONFIDENCE_MIN_EDGE,
    COVER_PROB_CAP_HI,
    COVER_PROB_CAP_LO,
    MIN_CONFIDENCE_SCORE,
    MIN_EDGE_BUCKET,
    OPTIMAL_BET_EDGE,
)
from pipeline.diagnostics import blowout_accuracy_table, favorite_loss_by_spread_bin
from pipeline.market import spread_cover_prob
from pipeline.metrics import add_clv_columns
from pipeline.upset_classifier import UpsetClassifier, favorite_lost_label


def _ats_roi(ats_rate: float) -> float:
    return float(ats_rate * (100.0 / 110.0) - (1.0 - ats_rate))


def validate_plan_on_backtest_csv(
    csv_path: str | Path,
    *,
    min_conf: float | None = None,
    min_edge: float | None = None,
) -> dict:
    """Replay hardened gates + report calibration / upset / blowout health metrics.

    This is the Phase-4 check that can run without a multi-hour walk-forward:
    it evaluates the new defaults against an existing `backtest_results.csv`.
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    if "simulated_season_window" not in df.columns and "DATE" in df.columns:
        df["simulated_season_window"] = "unknown"

    min_conf = float(MIN_CONFIDENCE_SCORE if min_conf is None else min_conf)
    min_edge = float(actionable_min_edge() if min_edge is None else min_edge)

    # Gate simulation under new defaults
    actionable = []
    for _, row in df.iterrows():
        lean = row.get("EDGE_LEAN", row.get("DIRECTION", "Pass"))
        if lean in ("Pass", None, "", "nan") or (isinstance(lean, float) and pd.isna(lean)):
            edge = float(row.get("EDGE", 0) or 0)
            lean = "Home" if edge > 0 else ("Away" if edge < 0 else "Pass")
        ok = passes_confidence_actionable_gates(
            lean=str(lean),
            edge_pts=float(row.get("EDGE", 0) or 0),
            conf_score=float(row.get("CONFIDENCE", 0) or 0),
            conf_width=float(row.get("CONF_WIDTH", 24) or 24),
            min_confidence=min_conf,
            disagreement_trust=float(row.get("DISAGREEMENT_TRUST", 1.0) or 1.0),
            phantom_injury_flag=bool(int(row.get("PHANTOM_INJURY_FLAG", 0) or 0)),
            market_spread=row.get("MARKET_SPREAD"),
            elo_meta_agreement=row.get("ELO_META_AGREEMENT"),
            min_edge=min_edge,
        )
        actionable.append(int(ok))
    df = df.copy()
    df["_plan_actionable"] = actionable

    bets = df[df["_plan_actionable"] == 1].copy()
    ats_n = 0
    ats_wins = 0
    if not bets.empty and "ACTUAL_MARGIN" in bets.columns and "MARKET_SPREAD" in bets.columns:
        for _, r in bets.iterrows():
            cover = float(r["ACTUAL_MARGIN"]) + float(r["MARKET_SPREAD"])
            if abs(cover) < 1e-9:
                continue
            side = r.get("EDGE_LEAN", r.get("DIRECTION", "Pass"))
            if side not in ("Home", "Away"):
                edge = float(r.get("EDGE", 0) or 0)
                side = "Home" if edge > 0 else "Away"
            win = (side == "Home" and cover > 0) or (side == "Away" and cover < 0)
            ats_n += 1
            ats_wins += int(win)
    ats_pct = (ats_wins / ats_n) if ats_n else float("nan")

    # Cover ECE with cap applied
    cover_ece = float("nan")
    if "COVER_PROB_CALIBRATED" in df.columns and ats_n > 0:
        y, p = [], []
        for _, r in bets.iterrows():
            cover = float(r["ACTUAL_MARGIN"]) + float(r["MARKET_SPREAD"])
            if abs(cover) < 1e-9:
                continue
            side = r.get("EDGE_LEAN", r.get("DIRECTION", "Pass"))
            if side not in ("Home", "Away"):
                edge = float(r.get("EDGE", 0) or 0)
                side = "Home" if edge > 0 else "Away"
            y.append(1.0 if ((side == "Home" and cover > 0) or (side == "Away" and cover < 0)) else 0.0)
            cp = float(r.get("COVER_PROB_CALIBRATED", 0.5) or 0.5)
            p.append(float(np.clip(cp, COVER_PROB_CAP_LO, COVER_PROB_CAP_HI)))
        if y:
            cover_ece = float(compute_ece(np.asarray(y), np.asarray(p)))

    # Heteroscedastic cover sanity: high vol → lower extreme probs
    hi_vol = spread_cover_prob(10.0, -6.0, conf_width=24.0, matchup_vol_sigma=18.0)
    lo_vol = spread_cover_prob(10.0, -6.0, conf_width=24.0, matchup_vol_sigma=8.0)

    # Upset classifier quick fit on first half → score second half
    upset_brier = float("nan")
    mid = len(df) // 2
    if mid > 100:
        clf = UpsetClassifier()
        train = df.iloc[:mid]
        test = df.iloc[mid:]
        clf.fit(train)
        if clf.fitted:
            y_u, p_u = [], []
            for _, r in test.iterrows():
                lab = favorite_lost_label(r)
                if lab is None:
                    continue
                y_u.append(lab)
                p_u.append(clf.predict_upset_prob(r.to_dict()))
            if len(y_u) >= 30:
                from pipeline.calibration_metrics import compute_brier
                upset_brier = float(compute_brier(np.asarray(y_u), np.asarray(p_u)))

    # CLV: identical decision/close → NaN (not false zero)
    clv_df = add_clv_columns(df.head(200).copy())
    identical = (
        clv_df["DECISION_SPREAD"].notna()
        & clv_df["CLOSING_SPREAD"].notna()
        & (clv_df["DECISION_SPREAD"] == clv_df["CLOSING_SPREAD"])
    )
    clv_identical_nan = bool(
        identical.any() and clv_df.loc[identical, "CLV"].isna().all()
    )

    # Season volume stability under new gates
    season_counts = {}
    if "simulated_season_window" in df.columns:
        for s, g in df.groupby("simulated_season_window"):
            season_counts[str(s)] = int(g["_plan_actionable"].sum())
    seasons_sorted = sorted(season_counts.keys())
    volumes = [season_counts[s] for s in seasons_sorted]
    volume_collapse = False
    nonzero = [v for v in volumes if v > 0]
    if len(nonzero) >= 2:
        volume_collapse = nonzero[-1] < 0.4 * nonzero[0]

    # Upset / blowout diagnostic slices
    fav_tbl = favorite_loss_by_spread_bin(df)
    blow_tbl = blowout_accuracy_table(df)

    # Scenario-mix proxy sanity
    from pipeline.market import scenario_mix_sigma_proxy, dampen_cover_for_upset
    mix_lo = scenario_mix_sigma_proxy(matchup_vol_sigma=10.0, conf_width=24.0)
    mix_hi = scenario_mix_sigma_proxy(
        matchup_vol_sigma=10.0, conf_width=24.0,
        h_missing_rotation=0.5, h_star_out=1.0,
    )
    damp_ok = dampen_cover_for_upset(
        0.70, direction="Home", market_spread=-8.0, upset_prob=0.8,
    ) < 0.70

    report = {
        "n_games": int(len(df)),
        "n_actionable_plan_gates": int(ats_n),
        "ats_pct": ats_pct,
        "flat_roi": _ats_roi(ats_pct) if np.isfinite(ats_pct) else float("nan"),
        "cover_ece_capped": cover_ece,
        "hi_vol_cover_p": float(hi_vol),
        "lo_vol_cover_p": float(lo_vol),
        "vol_flattens_cover": bool(abs(hi_vol - 0.5) <= abs(lo_vol - 0.5) + 1e-9),
        "upset_brier_holdout": upset_brier,
        "clv_identical_is_nan": clv_identical_nan,
        "optimal_bet_edge_config": float(OPTIMAL_BET_EDGE),
        "min_conf_config": float(min_conf),
        "min_edge_config": float(min_edge),
        "min_edge_bucket": float(MIN_EDGE_BUCKET),
        "confidence_min_edge": float(CONFIDENCE_MIN_EDGE),
        "avoid_bands": normalize_edge_avoid_bands(
            __import__("pipeline.config", fromlist=["EDGE_AVOID_BAND"]).EDGE_AVOID_BAND
        ),
        "favorite_loss_bins": fav_tbl.to_dict(orient="records") if not fav_tbl.empty else [],
        "blowout_table": blow_tbl.to_dict(orient="records") if not blow_tbl.empty else [],
        "passes_avoid_band_3_5": passes_edge_avoid_band(3.5, elo_meta_agreement=0.0) is False,
        "passes_avoid_band_6_5": passes_edge_avoid_band(6.5, elo_meta_agreement=0.0) is True,
        "season_actionable_counts": season_counts,
        "volume_collapse_flag": volume_collapse,
        "scenario_mix_proxy_ok": bool(mix_hi is not None and mix_lo is not None and mix_hi >= mix_lo),
        "upset_ats_dampen_ok": damp_ok,
        "meets_roi_floor": bool(np.isfinite(ats_pct) and _ats_roi(ats_pct) >= 0.08),
        "meets_cover_ece_floor": bool(np.isfinite(cover_ece) and cover_ece < 0.05),
        "meets_volume_stable": not volume_collapse,
    }
    return report


def print_validation_report(report: dict) -> None:
    print("\n=== Gate validation (offline on backtest CSV) ===")
    print(f"  games={report['n_games']}  actionable={report['n_actionable_plan_gates']}")
    if np.isfinite(report["ats_pct"]):
        print(f"  ATS={report['ats_pct']:.1%}  ROI={report['flat_roi']:+.1%}")
    print(f"  cover ECE (capped)={report['cover_ece_capped']}")
    print(
        f"  vol cover flatten ok={report['vol_flattens_cover']}  "
        f"(hi={report['hi_vol_cover_p']:.3f} lo={report['lo_vol_cover_p']:.3f})"
    )
    print(f"  upset holdout Brier={report['upset_brier_holdout']}")
    print(f"  CLV identical→NaN={report['clv_identical_is_nan']}")
    print(
        f"  gates: min_conf={report['min_conf_config']}  min_edge={report['min_edge_config']}  "
        f"bucket={report.get('min_edge_bucket')}  OPTIMAL_BET_EDGE={report['optimal_bet_edge_config']}  "
        f"avoid={report['avoid_bands']}"
    )
    print(f"  season actionable={report.get('season_actionable_counts')}")
    print(
        f"  volume_collapse={report.get('volume_collapse_flag')}  "
        f"scenario_mix_ok={report.get('scenario_mix_proxy_ok')}  "
        f"upset_dampen_ok={report.get('upset_ats_dampen_ok')}"
    )
    print(
        f"  success: ROI≥8%={report.get('meets_roi_floor')}  "
        f"cover ECE<0.05={report.get('meets_cover_ece_floor')}  "
        f"volume_stable={report.get('meets_volume_stable')}"
    )


if __name__ == "__main__":
    import sys
    try:
        _root = Path(__file__).resolve().parents[1]
    except NameError:
        _root = Path.cwd()
    default = _root / "newest data" / "backtest_results.csv"
    # Skip auto-run when inlined into a notebook (no CLI argv / no real __file__).
    if "ipykernel" in sys.modules or "google.colab" in sys.modules:
        pass
    else:
        csv = Path(sys.argv[1]) if len(sys.argv) > 1 else default
        rep = validate_plan_on_backtest_csv(csv)
        print_validation_report(rep)
