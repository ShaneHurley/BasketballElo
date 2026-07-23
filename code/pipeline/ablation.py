"""Walk-forward ablation harness.

Runs the full walk-forward backtest under several configurations and reports,
per season and overall: spread MAE, total MAE, ATS win%, ROI (-110), and Brier.

Because the market already prices most box-score signal, *more features is not
automatically better* - this harness exists so every change is kept only if it
improves out-of-sample. Intended to run where the PBP data lives (e.g. Colab).

Acceptance gate (pooled ALL season row):
  - ATS ROI improves OR stays within 0.3% while spread MAE improves
  - Never flip default if spread MAE worsens by >0.1 without ROI gain

Example
-------
>>> from pipeline.ablation import run_ablation
>>> summary, detail = run_ablation(all_stints, modern_odds_dict,
...                                n_tuning_trials_meta=10)
>>> summary   # one row per (config, season) + overall
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from pipeline.backtest import run_multi_year_backtest_walkforward
from pipeline.model import (
    SAFE_FEATURE_COLS,
    CORE_FEATURE_COLS,
    CONTEXT_FEATURE_COLS,
    ENGINE_MARGIN_COLS,
    SCHEDULE_FEATURE_COLS,
    MARKET_TOTAL_COLS,
    ELO_INTERACTION_COLS,
    SHOT_QUALITY_COLS,
    LINEUP_COMPOSITE_COLS,
)
from pipeline.model import HCA_FEATURE_COLS
from pipeline.teamstats import (
    FORM_FEATURE_COLS, FORM_FEATURE_COLS_BASE, FOUR_FACTOR_COLS, OPP_ADJ_COLS, PYTHAG_COLS,
)

BREAKEVEN = 110.0 / 210.0

# Walk-forward validation checklist (Phase 6) — compare configs on these metrics.
VALIDATION_CHECKLIST = """
Elo-first validation checklist (run via run_ablation):
  1. Ablation grid: baseline_current vs edge_bucket vs edge_bucket_ats vs simplified_calib
  2. Primary (ATS): spread MAE, ATS win%, ROI, mean CLV, CLV-weighted ROI
  3. Secondary (ML): ML winner accuracy, ML ROI, Brier on win prob
  4. Tertiary (O/U): total MAE, O/U ROI when ou_bet enabled
  5. Elo diagnostics: elo_vs_market distribution, raw_spread_mae vs spread_mae gap
  6. Clear state/tuning_cache when Elo update logic changes materially

Default flip gate (passes_ablation_gate):
  - pooled ROI delta >= -0.003 OR (MAE improves and ROI within 0.3%)
  - spread_mae must not worsen by >0.1 unless ROI improves materially
  - tertiary: ML Brier, total MAE, CRPS — never raw hit-rate
  - use promote_ablation_winners(summary_df) after a grid run; apply to config after review
"""


def _without(cols, drop):
    drop = set(drop)
    return [c for c in cols if c not in drop]


def _config_overrides(**kwargs) -> dict:
    """Build ablation spec with optional config flag overrides (runtime only)."""
    return kwargs


def calibration_ablation_configs() -> dict:
    """Cover-probability / calibration mode experiments (require full re-run)."""
    base = default_configs()
    keys = ("baseline_current", "simplified_calib", "beta_ats_calib", "skellam_cover")
    return {k: base[k] for k in keys if k in base}


def default_configs() -> dict:
    """Named configurations.

    Each value is a dict with optional keys:
      - feature_cols: feature subset (None = full SAFE_FEATURE_COLS)
      - model_kwargs: passed to MetaScoreModel (e.g. margin_cap, total_mode)
      - backtest_kwargs: passed to the backtest (e.g. walkforward_edge)
      - config_overrides: dict of pipeline.config attribute overrides applied at runtime

    Note: behavioural fixes that are now global defaults (per-team pace,
    walk-forward zone PPS, the linear spread calibrator) are present in *every*
    config; toggle-able groups below isolate feature blocks and bet selection.
    """
    legacy_feats = CORE_FEATURE_COLS + FORM_FEATURE_COLS_BASE
    elo_heavy_drop = [
        "hier_net", "hier_margin",
        "h_hier_off", "h_hier_def", "a_hier_off", "a_hier_def",
    ]
    hier_heavy_drop = [
        "elo_net", "elo_margin",
        "h_elo_off", "h_elo_def", "a_elo_off", "a_elo_def",
        "elo_diff_off", "elo_diff_def",
        *ELO_INTERACTION_COLS,
    ]
    form_only_feats = (
        FORM_FEATURE_COLS + CONTEXT_FEATURE_COLS + SCHEDULE_FEATURE_COLS
        + HCA_FEATURE_COLS + MARKET_TOTAL_COLS
    )
    return {
        # Reference snapshot of current production defaults (Phase 0 baseline).
        "baseline_current": {},
        # TRUE legacy baseline: legacy features, fixed 225 total, +/-20 cap,
        # fixed 2.5 edge, AND the global fixes (per-team pace, linear calib,
        # variance-aware win-prob) all turned OFF.
        "baseline_legacy": {"feature_cols": legacy_feats,
                            "model_kwargs": {"margin_cap": 20.0, "total_mode": "fixed"},
                            "backtest_kwargs": {"walkforward_edge": False,
                                                "pace_per_team": False,
                                                "spread_calib_mode": "additive",
                                                "variance_aware_winprob": False}},
        # All changes on.
        "improved_full": {},
        # Bet selection experiments (Phase 1+3) — applied via config_overrides at runtime.
        "edge_bucket": {"config_overrides": {
            "BET_SELECTION_MODE": "edge_bucket",
            "USE_TIER_STAKE_GATES": False,
            "WALKFORWARD_EDGE_MIN_FLOOR": 5.5,
        }},
        "edge_bucket_ats": {"config_overrides": {
            "BET_SELECTION_MODE": "edge_bucket_ats",
            "USE_TIER_STAKE_GATES": False,
            "WALKFORWARD_EDGE_MIN_FLOOR": 5.5,
        }},
        "simplified_calib": {"config_overrides": {
            "CALIBRATION_MODE": "simplified",
            "REQUIRE_META_WIN_WHEN_FITTED": True,
        }},
        "beta_ats_calib": {"config_overrides": {
            "CALIBRATION_MODE": "beta_ats",
            "REQUIRE_META_WIN_WHEN_FITTED": True,
        }},
        "volatility_staking": {"config_overrides": {
            "STAKE_SIZING_MODE": "volatility_adjusted",
        }},
        "skellam_cover": {"config_overrides": {"USE_SKELLAM_COVER_PROB": True}},
        "lgbm_stack": {"model_kwargs": {"use_lightgbm_base": True}},
        "garbage_form": {"config_overrides": {"USE_GARBAGE_WEIGHTED_FORM": True}},
        "decayed_sos": {"config_overrides": {"USE_DECAYED_SOS": True}},
        # Improved minus each new feature group, to isolate incremental value.
        "no_four_factors": {"feature_cols": _without(SAFE_FEATURE_COLS, FOUR_FACTOR_COLS)},
        "no_pythagorean": {"feature_cols": _without(SAFE_FEATURE_COLS, PYTHAG_COLS)},
        "no_opp_adj": {"feature_cols": _without(SAFE_FEATURE_COLS, OPP_ADJ_COLS)},
        "no_hca": {"feature_cols": _without(SAFE_FEATURE_COLS, HCA_FEATURE_COLS)},
        "no_context": {"feature_cols": _without(SAFE_FEATURE_COLS, CONTEXT_FEATURE_COLS)},
        "no_engine_margin": {"feature_cols": _without(SAFE_FEATURE_COLS, ENGINE_MARGIN_COLS)},
        "no_schedule": {"feature_cols": _without(SAFE_FEATURE_COLS, SCHEDULE_FEATURE_COLS)},
        # Isolate the prediction fixes (real total + wider cap) on legacy features.
        "fixes_only": {"feature_cols": legacy_feats},
        # Full features but fixed 2.5 edge (isolates walk-forward edge selection).
        "fixed_edge_2p5": {"backtest_kwargs": {"walkforward_edge": False}},
        # Isolate each global fix (everything else stays improved).
        "legacy_pace": {"backtest_kwargs": {"pace_per_team": False}},
        "additive_calib": {"backtest_kwargs": {"spread_calib_mode": "additive"}},
        "no_variance_winprob": {"backtest_kwargs": {"variance_aware_winprob": False}},
        # Engine / feature isolation configs (Phase 0 measurement).
        "elo_heavy": {"feature_cols": _without(SAFE_FEATURE_COLS, elo_heavy_drop)},
        "hier_heavy": {"feature_cols": _without(SAFE_FEATURE_COLS, hier_heavy_drop)},
        "form_only": {"feature_cols": form_only_feats},
        "no_elo_interactions": {"feature_cols": _without(SAFE_FEATURE_COLS, ELO_INTERACTION_COLS)},
        # Total-head ablations (Phase 1).
        "fixed_total_225": {"model_kwargs": {"total_mode": "fixed", "league_avg_total": 225.0}},
        "model_total_market": {"model_kwargs": {"total_mode": "model"}},
        # Elo-first two-stage stack (Phase 3).
        "elo_heavy_stack": {"model_kwargs": {"use_elo_stack": True}},
        "no_elo_stack": {"model_kwargs": {"use_elo_stack": False}},
        # Market-residual prediction (Phase 4).
        "residual_blend": {"model_kwargs": {"prediction_mode": "blend", "residual_alpha": 0.5}},
        "residual_full": {"model_kwargs": {"prediction_mode": "residual", "residual_alpha": 0.7}},
        "purged_cv": {"model_kwargs": {"use_purged_cv": True}},
        "shuffled_kfold_cv": {"model_kwargs": {"use_purged_cv": False}},
        # Elo-first betting accuracy experiments
        "dynamic_elo_blend": {"model_kwargs": {"dynamic_elo_blend": True}},
        "no_dynamic_elo_blend": {"model_kwargs": {"dynamic_elo_blend": False}},
        "elo_gated_bets": {"backtest_kwargs": {"require_elo_agreement": True}},
        "elo_total_anchor": {"model_kwargs": {"total_elo_beta": 0.35}},
        "standalone_total_model": {
            "model_kwargs": {"total_mode": "external"},
            "backtest_kwargs": {"tune_total_head": True},
        },
        "ml_isotonic_calib": {"config_overrides": {"ML_CALIB_METHOD": "isotonic"}},
        "ml_no_spread_feature": {"config_overrides": {"ML_USE_SPREAD_FEATURES": False}},
        "total_market_residual": {"config_overrides": {"TOTAL_TRAIN_TARGET": "market_residual"}},
        "no_shot_quality": {"feature_cols": _without(SAFE_FEATURE_COLS, SHOT_QUALITY_COLS)},
        "no_lineup_composite": {"feature_cols": _without(SAFE_FEATURE_COLS, LINEUP_COMPOSITE_COLS)},
        "hapm_priors": {"backtest_kwargs": {"use_hapm_priors": True}},
        # Specialized market pipeline v2
        "gaussian_ou_gates": {"config_overrides": {
            "OU_USE_GAUSSIAN_PROB": True,
            "OU_MIN_PROB": 0.55,
        }},
        "score_pair_total": {"config_overrides": {"USE_SCORE_PAIR_TOTAL": True}},
        "ml_both_sides_ev": {"config_overrides": {"ML_BET_PREDICTED_WINNER_ONLY": False}},
        "ml_winner_only": {"config_overrides": {"ML_BET_PREDICTED_WINNER_ONLY": True}},
        "stacked_lgbm_catboost": {"model_kwargs": {"use_lightgbm_base": True}},
        "no_elo_on_totals": {"config_overrides": {"TOTAL_ELO_BETA": 0.0}},
    }


def ml_ablation_configs() -> dict:
    base = default_configs()
    keys = ("baseline_current", "ml_isotonic_calib", "ml_no_spread_feature")
    return {k: base[k] for k in keys if k in base}


def total_ablation_configs() -> dict:
    base = default_configs()
    keys = ("baseline_current", "standalone_total_model", "total_market_residual", "elo_total_anchor")
    return {k: base[k] for k in keys if k in base}


def passes_ablation_gate(
    baseline: dict,
    candidate: dict,
    *,
    roi_tol: float = 0.003,
    mae_tol: float = 0.1,
    ml_brier_tol: float = 0.01,
    total_mae_tol: float = 0.5,
    crps_tol: float = 0.5,
) -> bool:
    """Return True if candidate meets pooled acceptance criteria vs baseline.

    Hit-rate / raw accuracy is never a promotion criterion. Primary gates:
    ATS ROI + spread MAE; tertiary: ML Brier, total MAE/CRPS.
    """
    b_roi = float(baseline.get("roi", np.nan))
    c_roi = float(candidate.get("roi", np.nan))
    b_mae = float(baseline.get("spread_mae", np.nan))
    c_mae = float(candidate.get("spread_mae", np.nan))
    if not np.isfinite(c_roi) or not np.isfinite(c_mae):
        return False
    roi_ok = c_roi >= b_roi - roi_tol or (c_mae < b_mae and c_roi >= b_roi - roi_tol)
    mae_regression = c_mae > b_mae + mae_tol
    roi_gain = c_roi > b_roi + roi_tol
    if mae_regression and not roi_gain:
        return False
    if not roi_ok:
        return False
    b_ml_brier = baseline.get("ml_brier", baseline.get("brier"))
    c_ml_brier = candidate.get("ml_brier", candidate.get("brier"))
    if b_ml_brier is not None and c_ml_brier is not None:
        if np.isfinite(b_ml_brier) and np.isfinite(c_ml_brier):
            if c_ml_brier > float(b_ml_brier) + ml_brier_tol and not roi_gain:
                return False
    b_total_mae = baseline.get("total_mae")
    c_total_mae = candidate.get("total_mae")
    if b_total_mae is not None and c_total_mae is not None:
        if np.isfinite(b_total_mae) and np.isfinite(c_total_mae):
            if c_total_mae > float(b_total_mae) + total_mae_tol and not roi_gain:
                return False
    b_crps = baseline.get("crps")
    c_crps = candidate.get("crps")
    if b_crps is not None and c_crps is not None:
        if np.isfinite(b_crps) and np.isfinite(c_crps):
            if c_crps > float(b_crps) + crps_tol and not roi_gain:
                return False
    return True


def recommend_default_flips(summary_df: pd.DataFrame, baseline_name: str = "baseline_current") -> list[str]:
    """List config names that pass the gate vs baseline (does not mutate config.py)."""
    if summary_df is None or summary_df.empty:
        return []
    pooled = summary_df[summary_df["season"] == "ALL"]
    base_row = pooled[pooled["config"] == baseline_name]
    if base_row.empty:
        return []
    baseline = base_row.iloc[0].to_dict()
    passed = []
    for cfg in pooled["config"].unique():
        if cfg == baseline_name:
            continue
        row = pooled[pooled["config"] == cfg].iloc[0].to_dict()
        if passes_ablation_gate(baseline, row):
            passed.append(cfg)
    return passed


# Config override keys that are safe to promote into pipeline.config when ablation passes.
_PROMOTABLE_OVERRIDES = {
    "gaussian_ou_gates": {"OU_USE_GAUSSIAN_PROB": True, "OU_MIN_PROB": 0.55},
    "ml_isotonic_calib": {"ML_CALIB_METHOD": "isotonic"},
    "ml_both_sides_ev": {"ML_BET_PREDICTED_WINNER_ONLY": False},
    "simplified_calib": {"CALIBRATION_MODE": "simplified"},
    "beta_ats_calib": {"CALIBRATION_MODE": "beta_ats"},
    "skellam_cover": {"USE_SKELLAM_COVER_PROB": True},
    "total_market_residual": {"TOTAL_TRAIN_TARGET": "market_residual"},
    "no_elo_on_totals": {"TOTAL_ELO_BETA": 0.0},
    "score_pair_total": {"USE_SCORE_PAIR_TOTAL": True},
    "lgbm_stack": {"USE_LIGHTGBM_BASE": True},
    "stacked_lgbm_catboost": {"USE_LIGHTGBM_BASE": True},
}


def promote_ablation_winners(
    summary_df: pd.DataFrame,
    baseline_name: str = "baseline_current",
    *,
    apply: bool = False,
) -> dict:
    """Recommend (and optionally apply in-process) config flips from ablation.

    Does not rewrite config.py on disk — returns the mapping and, if apply=True,
    patches pipeline.config for the current process (useful in notebooks).
    """
    winners = recommend_default_flips(summary_df, baseline_name=baseline_name)
    promotions = {}
    for name in winners:
        if name in _PROMOTABLE_OVERRIDES:
            promotions[name] = dict(_PROMOTABLE_OVERRIDES[name])
    if apply and promotions:
        import pipeline.config as cfg
        for overrides in promotions.values():
            for key, val in overrides.items():
                if hasattr(cfg, key):
                    setattr(cfg, key, val)
    return {
        "winners": winners,
        "promotions": promotions,
        "note": (
            "Apply returned promotions to pipeline.config.py after review. "
            "Hit-rate was not used as a gate."
        ),
    }


def compute_metrics(results_df: pd.DataFrame, season_col="simulated_season_window") -> pd.DataFrame:
    """Re-export from metrics (avoids circular import with backtest)."""
    from pipeline.metrics import compute_metrics as _compute_metrics

    return _compute_metrics(results_df, season_col=season_col)


def _apply_config_overrides(overrides: dict | None):
    """Temporarily patch pipeline.config flags for one ablation run."""
    if not overrides:
        return {}
    import pipeline.config as cfg
    saved = {}
    for key, val in overrides.items():
        if hasattr(cfg, key):
            saved[key] = getattr(cfg, key)
            setattr(cfg, key, val)
    return saved


def _restore_config_overrides(saved: dict):
    if not saved:
        return
    import pipeline.config as cfg
    for key, val in saved.items():
        setattr(cfg, key, val)


def run_ablation(stints_df, odds_dict=None, configs: dict = None,
                 rolling_window_size: int = 3,
                 n_tuning_trials_elo: int = 10,
                 n_tuning_trials_hier: int = 10,
                 n_tuning_trials_meta: int = 10) -> tuple:
    """Run each configuration end-to-end and return (summary_df, detail_dict).

    summary_df: config x season metrics (long form), good for direct comparison.
    detail_dict: {config_name: raw results DataFrame}.
    """
    if configs is None:
        configs = default_configs()

    detail = {}
    summaries = []
    for name, spec in configs.items():
        spec = spec or {}
        feature_cols = spec.get("feature_cols")
        model_kwargs = spec.get("model_kwargs")
        backtest_kwargs = spec.get("backtest_kwargs", {})
        saved_cfg = _apply_config_overrides(spec.get("config_overrides"))
        print(f"\n{'#'*70}\n# ABLATION CONFIG: {name}\n{'#'*70}")
        try:
            results = run_multi_year_backtest_walkforward(
                stints_df,
                odds_dict=odds_dict,
                n_tuning_trials_elo=n_tuning_trials_elo,
                n_tuning_trials_hier=n_tuning_trials_hier,
                n_tuning_trials_meta=n_tuning_trials_meta,
                rolling_window_size=rolling_window_size,
                model_kwargs=model_kwargs,
                feature_cols=feature_cols,
                **backtest_kwargs,
            )
        finally:
            _restore_config_overrides(saved_cfg)
        detail[name] = results
        m = compute_metrics(results)
        if not m.empty:
            m.insert(0, "config", name)
            summaries.append(m)
            print(f"\n=== {name} (overall) ===")
            print(m[m["season"] == "ALL"].to_string(index=False,
                  float_format=lambda v: f"{v:.3f}"))

    summary_df = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()

    if not summary_df.empty:
        print(f"\n{'='*70}\nOVERALL COMPARISON (season == ALL)\n{'='*70}")
        piv = summary_df[summary_df["season"] == "ALL"].set_index("config")
        print(piv[["spread_mae", "total_mae", "ats_pct", "roi", "brier", "n_bets"]]
              .to_string(float_format=lambda v: f"{v:.3f}"))
        rec = recommend_default_flips(summary_df)
        if rec:
            print(f"\nConfigs passing ablation gate vs baseline_current: {rec}")
            print("(Defaults in config.py are NOT auto-flipped — review and enable manually.)")
        else:
            print("\nNo configs passed ablation gate vs baseline_current.")
    return summary_df, detail
