"""Phase 2a unified confidence diagnostics (prior-year calibration + weight tuning)."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from pipeline.bet_confidence import (
    WalkForwardBetCalibrator,
    confidence_weights_path,
    weights_differ_from_default,
    yearly_confidence_weight_tuning,
)
from pipeline.calibration_metrics import compute_brier, compute_ece, reliability_bins
from pipeline.config import (
    CONFIDENCE_CALIB_METHOD,
    CONFIDENCE_MODE,
    CONFIDENCE_WEIGHT_SHRINK,
    MIN_CONFIDENCE_SCORE,
    STATE_DIR,
)
from pipeline.diagnostics import _ats_frame, _non_push_ats, _norm_results, _save_or_show
from pipeline.metrics import BREAKEVEN_ATS


def _roi_from_winrate(wp: float) -> float:
    return float(wp * (100.0 / 110.0) - (1.0 - wp))


def confidence_score_table(df: pd.DataFrame, score_col: str = "CONFIDENCE", n_bins: int = 10) -> pd.DataFrame:
    """ATS win rate and ROI by confidence score deciles."""
    bets = _non_push_ats(_ats_frame(_norm_results(df)))
    if bets.empty or score_col not in bets.columns:
        return pd.DataFrame()
    bets = bets.dropna(subset=[score_col])
    if bets.empty:
        return pd.DataFrame()
    try:
        bets = bets.copy()
        bets["score_bin"] = pd.qcut(bets[score_col], q=min(n_bins, bets[score_col].nunique()), duplicates="drop")
    except ValueError:
        edges = np.linspace(bets[score_col].min(), bets[score_col].max(), n_bins + 1)
        bets["score_bin"] = pd.cut(bets[score_col], bins=edges)
    rows = []
    for label, sub in bets.groupby("score_bin", observed=True):
        wp = float(sub["ats_win"].mean())
        rows.append({
            "bin": str(label),
            "score_lo": float(sub[score_col].min()),
            "score_hi": float(sub[score_col].max()),
            "score_mean": float(sub[score_col].mean()),
            "n_bets": len(sub),
            "ats_pct": wp,
            "roi": _roi_from_winrate(wp),
        })
    return pd.DataFrame(rows)


def confidence_threshold_grid(
    df: pd.DataFrame,
    thresholds=range(50, 66),
    score_col: str = "CONFIDENCE",
) -> pd.DataFrame:
    bets = _non_push_ats(_ats_frame(_norm_results(df)))
    if bets.empty or score_col not in bets.columns:
        return pd.DataFrame()
    rows = []
    for thr in thresholds:
        sub = bets[bets[score_col] >= thr]
        if len(sub) < 20:
            continue
        wp = float(sub["ats_win"].mean())
        rows.append({
            "min_confidence": thr,
            "n_bets": len(sub),
            "ats_pct": wp,
            "flat_roi": _roi_from_winrate(wp),
        })
    return pd.DataFrame(rows)


def prior_year_calibration_ablation(
    results_df: pd.DataFrame,
    methods: tuple[str, ...] = ("logistic_isotonic", "logistic_features", "isotonic", "platt", "none"),
    season_col: str = "simulated_season_window",
    show_progress: bool = True,
) -> pd.DataFrame:
    """Calibrate each test season from the immediately prior season only."""
    df = _norm_results(results_df)
    if season_col not in df.columns:
        return pd.DataFrame()

    seasons = sorted(df[season_col].dropna().unique())
    tasks = [
        (seasons[i - 1], test_season, method)
        for i, test_season in enumerate(seasons)
        if i > 0
        for method in methods
    ]
    rows = []
    for train_season, test_season, method in tqdm(
        tasks,
        desc="2a · calibration ablation",
        disable=not show_progress,
        leave=False,
    ):
        prior = df[df[season_col] == train_season]
        test = df[df[season_col] == test_season]
        cal = WalkForwardBetCalibrator(method=method)
        cal.fit(prior, method=method, scope="all_prior")
        outcomes, probs = [], []
        for _, row in test.iterrows():
            y = WalkForwardBetCalibrator._ats_outcome(row)
            if y is None:
                continue
            row_dict = cal._enrich_ats_row(row)
            raw = cal._build_score(row_dict, "ats")
            feat = cal._row_features(row_dict, "ats")
            prob = cal._predict_prob("ats", raw, feat)
            probs.append(prob)
            outcomes.append(y)
        if len(outcomes) < 20:
            continue
        y = np.asarray(outcomes, dtype=float)
        p = np.asarray(probs, dtype=float)
        wp = float(y.mean())
        rows.append({
            "season": test_season,
            "train_season": train_season,
            "method": method,
            "n_bets": len(y),
            "ats_pct": wp,
            "roi": _roi_from_winrate(wp),
            "brier": compute_brier(y, p),
            "ece": compute_ece(y, p),
        })
    return pd.DataFrame(rows)


def run_phase_2a_unified_confidence(
    results_df: pd.DataFrame,
    *,
    save_dir: str | Path | None = None,
    show_plots: bool = True,
    show_progress: bool = True,
) -> dict:
    """
    Phase 2a diagnostics (plots + tables).
    Weight tuning runs **inline during walk-forward backtest** when CONFIDENCE_MODE='unified'.
    This cell replays diagnostics only unless backtest was run without inline Phase 2a.
    """
    if results_df is None or results_df.empty:
        print("No results for Phase 2a confidence diagnostics.")
        return {}

    df = _norm_results(results_df)
    save_path = Path(save_dir) if save_dir else None
    out: dict = {
        "mode": CONFIDENCE_MODE,
        "calib_method": CONFIDENCE_CALIB_METHOD,
        "calib_scope": "prior_year",
        "weight_shrink": CONFIDENCE_WEIGHT_SHRINK,
        "min_confidence_default": MIN_CONFIDENCE_SCORE,
        "inline_phase2a": bool(
            "PHASE2A_INLINE" in df.columns and df["PHASE2A_INLINE"].fillna(0).astype(int).max() > 0
        ),
    }

    # Phase 2 default (from backtest CSV)
    phase_steps = tqdm(
        total=7,
        desc="Phase 2a",
        disable=not show_progress,
        unit="step",
    )

    phase_steps.set_description("2a · default deciles")
    default_table = confidence_score_table(df, "CONFIDENCE")
    default_raw = confidence_score_table(df, "CONFIDENCE_RAW") if "CONFIDENCE_RAW" in df.columns else pd.DataFrame()
    phase_steps.update(1)

    phase_steps.set_description("2a · weight tuning")
    weight_tuning = yearly_confidence_weight_tuning(
        df, method=CONFIDENCE_CALIB_METHOD, show_progress=show_progress,
    )
    phase_steps.update(1)

    phase_steps.set_description("2a · threshold grid")
    thr_grid = confidence_threshold_grid(df)
    phase_steps.update(1)

    phase_steps.set_description("2a · calibration ablation")
    ablation = prior_year_calibration_ablation(df, show_progress=show_progress)
    phase_steps.update(1)

    phase_steps.set_description("2a · summarize")

    out["default_confidence_deciles"] = default_table.to_dict(orient="records")
    out["default_raw_deciles"] = default_raw.to_dict(orient="records") if not default_raw.empty else []
    out["unified_weight_tuning"] = weight_tuning.to_dict(orient="records")
    out["threshold_grid"] = thr_grid.to_dict(orient="records")
    out["prior_year_calibration_ablation"] = ablation.to_dict(orient="records")

    if not weight_tuning.empty:
        latest = weight_tuning.iloc[-1]
        suggested = {
            k.replace("w_", ""): float(latest[k])
            for k in weight_tuning.columns if k.startswith("w_")
        }
        out["suggested_weights_latest"] = suggested
        if "train_min_confidence" in latest:
            out["suggested_min_confidence"] = int(latest["train_min_confidence"])
        elif not thr_grid.empty:
            out["suggested_min_confidence"] = int(
                thr_grid.loc[thr_grid["flat_roi"].idxmax()]["min_confidence"]
            )
        else:
            out["suggested_min_confidence"] = MIN_CONFIDENCE_SCORE
        out["weights_changed_latest"] = bool(latest.get("weights_changed", False))

    phase_steps.update(1)

    phase_steps.set_description("2a · plots")
    if show_plots:
        import matplotlib.pyplot as plt

        if not default_table.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(default_table["score_mean"], default_table["ats_pct"], "o-", color="#4C72B0", lw=2, label="Default")
            ax.axhline(BREAKEVEN_ATS, ls="--", color="gray", label="Break-even")
            ax.set_xlabel("Mean confidence score")
            ax.set_ylabel("ATS win rate")
            ax.set_title("Phase 2 Default Confidence Reliability", fontweight="bold")
            ax.legend()
            _save_or_show(fig, save_path, "14_confidence_reliability.png")

        if not default_raw.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(default_raw["score_mean"], default_raw["ats_pct"], "o-", color="#C44E52", lw=2)
            ax.axhline(BREAKEVEN_ATS, ls="--", color="gray")
            ax.set_xlabel("Mean raw confidence score")
            ax.set_ylabel("ATS win rate")
            ax.set_title("Raw Confidence Score Reliability", fontweight="bold")
            _save_or_show(fig, save_path, "14b_raw_confidence_reliability.png")

        if not thr_grid.empty:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.plot(thr_grid["min_confidence"], thr_grid["flat_roi"], "o-", color="#55A868", lw=2)
            ax.axhline(0, ls="--", color="black")
            ax.set_xlabel("Minimum confidence threshold")
            ax.set_ylabel("Flat ROI")
            ax.set_title("ROI vs Minimum Confidence Gate", fontweight="bold")
            _save_or_show(fig, save_path, "15_roi_vs_confidence_threshold.png")

        if not ablation.empty:
            fig, ax = plt.subplots(figsize=(9, 5))
            for method, g in ablation.groupby("method"):
                ax.plot(g["season"].astype(str), g["ece"], "o-", label=method, lw=1.8)
            ax.set_xlabel("Test season")
            ax.set_ylabel("ECE (lower is better)")
            ax.set_title("Prior-Year Calibration Method Ablation (ECE)", fontweight="bold")
            ax.legend(fontsize=8)
            plt.xticks(rotation=20)
            _save_or_show(fig, save_path, "16_confidence_calibration_ablation.png")

            fig, ax = plt.subplots(figsize=(9, 5))
            for method, g in ablation.groupby("method"):
                ax.plot(g["season"].astype(str), g["brier"], "s--", label=method, lw=1.5)
            ax.set_xlabel("Test season")
            ax.set_ylabel("Brier score")
            ax.set_title("Prior-Year Calibration Method Ablation (Brier)", fontweight="bold")
            ax.legend(fontsize=8)
            plt.xticks(rotation=20)
            _save_or_show(fig, save_path, "16c_confidence_calibration_brier.png")

        if not weight_tuning.empty and "test_roi" in weight_tuning.columns:
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(
                weight_tuning["test_season"].astype(str),
                weight_tuning["test_roi"],
                "o-",
                color="#8172B2",
                lw=2,
                label="Unified tuned weights",
            )
            ax.plot(
                weight_tuning["test_season"].astype(str),
                weight_tuning["train_roi"],
                "s--",
                color="#CCB974",
                lw=1.5,
                label="In-sample (train season)",
            )
            ax.axhline(0, ls="--", color="gray")
            ax.set_xlabel("Test season")
            ax.set_ylabel("ATS ROI")
            ax.set_title("Unified Weight Tuning — Prior Year Fit", fontweight="bold")
            ax.legend()
            plt.xticks(rotation=20)
            _save_or_show(fig, save_path, "16b_unified_weight_tuning_roi.png")

        bets = _non_push_ats(_ats_frame(df))
        if not bets.empty and "CONFIDENCE" in bets.columns:
            fig, ax = plt.subplots(figsize=(7, 7))
            rb = reliability_bins(bets["ats_win"].values, bets["CONFIDENCE"].values / 100.0, n_bins=8)
            if not rb.empty:
                ax.plot([0, 1], [0, 1], "--", color="gray", label="Perfect")
                ax.scatter(rb["pred_mean"], rb["obs_rate"], s=rb["n"] * 3, color="#4C72B0", zorder=3)
                ax.set_xlabel("Default calibrated confidence / 100")
                ax.set_ylabel("Observed ATS win rate")
                ax.set_title("Default Confidence Reliability Diagram", fontweight="bold")
                ax.legend()
            _save_or_show(fig, save_path, "12_confidence_reliability.png")

    phase_steps.update(1)

    phase_steps.set_description("2a · save")
    if save_path:
        save_path.mkdir(parents=True, exist_ok=True)
        if not default_table.empty:
            default_table.to_csv(save_path / "confidence_deciles_default.csv", index=False)
        if not weight_tuning.empty:
            weight_tuning.to_csv(save_path / "unified_confidence_weights_by_season.csv", index=False)
        if not thr_grid.empty:
            thr_grid.to_csv(save_path / "confidence_threshold_grid.csv", index=False)
        if not ablation.empty:
            ablation.to_csv(save_path / "confidence_calibration_ablation.csv", index=False)

    summary_path = confidence_weights_path()
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    ablation_path = STATE_DIR / "confidence_calibration_ablation.json"
    with open(ablation_path, "w") as f:
        json.dump(out, f, indent=2, default=str)

    phase_steps.update(1)
    phase_steps.close()

    print("\n" + "=" * 72)
    print("PHASE 2a · UNIFIED CONFIDENCE (inline walk-forward + diagnostics)")
    print("=" * 72)
    inline = (
        "PHASE2A_INLINE" in df.columns
        and df["PHASE2A_INLINE"].fillna(0).astype(int).max() > 0
    )
    if inline:
        print("Weights/calibration were tuned each season inside the walk-forward backtest.")
        print("This section shows reliability plots and threshold grids on those scores.\n")
    else:
        print("Replaying Phase 2a tuning post-hoc (re-run backtest with CONFIDENCE_MODE='unified' for inline).\n")

    if not default_table.empty:
        print("Default CONFIDENCE deciles (from Phase 2 backtest):")
        print(default_table.round(3).to_string(index=False))

    if not weight_tuning.empty:
        print("\nSuggested per-variable weights (latest season):")
        wcols = [c for c in weight_tuning.columns if c.startswith("w_")]
        display_cols = [
            "test_season", "train_season", "train_roi", "test_roi",
            "train_min_confidence", "train_gated_n_bets", "test_gated_n_bets",
        ] + wcols[:4]
        display_cols = [c for c in display_cols if c in weight_tuning.columns]
        print(weight_tuning[display_cols].round(4).tail(3).to_string(index=False))
        if not out.get("weights_changed_latest"):
            print("\n⚠️  Weight tuner returned defaults for latest season — "
                  "try increasing CONFIDENCE_WEIGHT_SEARCH_SAMPLES or check train bet count.")
        if "suggested_weights_latest" in out:
            print("\nApply for live/unified mode (pipeline/config.py CONFIDENCE_MODE='unified'):")
            for k, v in out["suggested_weights_latest"].items():
                rng_col = f"range_{k}"
                rng = weight_tuning.iloc[-1].get(rng_col, "")
                print(f"  {k}: {v:.4f}  {rng}")

    print(f"\n  Weight suggestions → {summary_path}")
    if save_path:
        print(f"  Plots/tables → {save_path}/")

    return out


# Backward-compatible alias
def run_confidence_diagnostics(results_df, **kwargs):
    return run_phase_2a_unified_confidence(results_df, **kwargs)
