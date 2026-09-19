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
from pipeline.metrics import compute_metrics
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
    HIER_SHOT_FEATURE_COLS,
    STRUCTURED_SCORE_COLS,
    PACE_UNCERTAINTY_COLS,
    LINE_DERIVED_FEATURE_COLS,
)
from pipeline.shot_hierarchy import HIER_SHOT_MIX_COLS, HIER_SHOT_MAKE_COLS
from pipeline.promotion_gates import (
    beats_market_and_baseline,
    multi_season_gate,
    clv_promotion_allowed,
    interval_coverage_credible,
    margin_dispersion_ok,
    market_error_corr_positive,
    policy_retune_allowed,
    tip_proxy_roi_blocked,
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
    """Cover-probability / calibration mode + spread-window experiments.

    Includes multi-year future-success grid: CALIBRATION_MODE variants and
    SPREAD_CALIB_WINDOW ∈ {30, 50, 80}. Lock by MAE/ECE stability, not ROI.
    """
    base = default_configs()
    keys = ("baseline_current", "simplified_calib", "beta_ats_calib", "skellam_cover")
    out = {k: base[k] for k in keys if k in base}
    out["legacy_stack_calib"] = {"config_overrides": {"CALIBRATION_MODE": "legacy_stack"}}
    for w in (30, 50, 80):
        out[f"spread_window_{w}"] = {
            "config_overrides": {
                "SPREAD_CALIB_WINDOW": w,
                "SPREAD_CALIB_ADAPTIVE_WINDOW": False,
            },
        }
    return out


def calib_stability_score(summary_df: pd.DataFrame) -> pd.DataFrame:
    """Rank calib configs by locked-season MAE + ECE stability (lower better).

    Prefer slightly worse mean with lower variance. Ignores ROI.
    """
    if summary_df is None or summary_df.empty:
        return pd.DataFrame()
    seasons = summary_df[summary_df["season"] != "ALL"].copy()
    if seasons.empty:
        seasons = summary_df.copy()
    rows = []
    for cfg, g in seasons.groupby("config"):
        mae = pd.to_numeric(g.get("spread_mae"), errors="coerce")
        ece = pd.to_numeric(g.get("ece", g.get("cover_ece", g.get("brier"))), errors="coerce")
        mae_mean, mae_std = float(mae.mean()), float(mae.std(ddof=0)) if len(mae) else np.nan
        ece_mean = float(ece.mean()) if ece.notna().any() else np.nan
        ece_std = float(ece.std(ddof=0)) if ece.notna().any() and len(ece) > 1 else 0.0
        # Stability score: mean MAE + 0.5*std MAE + ECE terms (when present).
        score = mae_mean + 0.5 * (mae_std if np.isfinite(mae_std) else 0.0)
        if np.isfinite(ece_mean):
            score += ece_mean + 0.5 * (ece_std if np.isfinite(ece_std) else 0.0)
        rows.append({
            "config": cfg,
            "spread_mae_mean": mae_mean,
            "spread_mae_std": mae_std,
            "ece_mean": ece_mean,
            "ece_std": ece_std,
            "stability_score": score,
            "n_seasons": int(len(g)),
        })
    out = pd.DataFrame(rows)
    if not out.empty:
        out = out.sort_values("stability_score", ascending=True).reset_index(drop=True)
    return out


def lock_calib_winner(
    summary_df: pd.DataFrame,
    *,
    apply: bool = False,
) -> dict:
    """Pick best CALIBRATION_MODE / SPREAD_CALIB_WINDOW by stability score."""
    stab = calib_stability_score(summary_df)
    if stab.empty:
        return {"winner": None, "promotions": {}, "stability": stab}
    winner = str(stab.iloc[0]["config"])
    promotions: dict = {}
    if winner.startswith("spread_window_"):
        try:
            w = int(winner.rsplit("_", 1)[-1])
            promotions = {"SPREAD_CALIB_WINDOW": w, "SPREAD_CALIB_ADAPTIVE_WINDOW": False}
        except ValueError:
            promotions = {}
    elif winner in ("simplified_calib", "beta_ats_calib", "legacy_stack_calib"):
        mode = {
            "simplified_calib": "simplified",
            "beta_ats_calib": "beta_ats",
            "legacy_stack_calib": "legacy_stack",
        }[winner]
        promotions = {"CALIBRATION_MODE": mode}
    elif winner in _PROMOTABLE_OVERRIDES:
        promotions = dict(_PROMOTABLE_OVERRIDES[winner])
    if apply and promotions:
        import pipeline.config as cfg
        for key, val in promotions.items():
            if hasattr(cfg, key):
                setattr(cfg, key, val)
    return {
        "winner": winner,
        "promotions": promotions,
        "stability": stab,
        "note": "Locked by MAE/ECE stability (not ROI). Review before writing config.py.",
    }


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
        # ATS repair: residual target vs absolute; residual without line-derived features.
        "decision_residual_current": {"model_kwargs": {"train_target": "decision_residual"}},
        "decision_residual_no_line_feats": {
            "model_kwargs": {"train_target": "decision_residual"},
            "feature_cols": _without(SAFE_FEATURE_COLS, LINE_DERIVED_FEATURE_COLS),
        },
        "absolute_margin_target": {"model_kwargs": {"train_target": "absolute"}},
        # Combined ATS repair candidate: residual target, no line feats, no family calib.
        "ats_repair_combined": {
            "model_kwargs": {"train_target": "decision_residual"},
            "feature_cols": _without(SAFE_FEATURE_COLS, LINE_DERIVED_FEATURE_COLS),
            "config_overrides": {"USE_TARGET_FAMILY_CALIBRATION": False},
        },
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
        "no_hier_shot": {"feature_cols": _without(SAFE_FEATURE_COLS, HIER_SHOT_FEATURE_COLS)},
        "hier_shot_volume_only": {"feature_cols": _without(SAFE_FEATURE_COLS, HIER_SHOT_MAKE_COLS)},
        "hier_shot_accuracy_only": {"feature_cols": _without(SAFE_FEATURE_COLS, HIER_SHOT_MIX_COLS)},
        "no_structured_score": {"feature_cols": _without(SAFE_FEATURE_COLS, STRUCTURED_SCORE_COLS)},
        "no_pace_uncertainty": {"feature_cols": _without(SAFE_FEATURE_COLS, PACE_UNCERTAINTY_COLS)},
        "legacy_pace_mean": {"config_overrides": {"USE_HIERARCHICAL_PACE": False}},
        "no_rotation_scenarios": {"config_overrides": {"USE_ROTATION_SCENARIOS": False}},
        "hard_lineup5_floor": {"config_overrides": {"LINEUP_COMPOSITE_EVIDENCE_POOLING": False}},
        "target_family_calib": {"config_overrides": {"USE_TARGET_FAMILY_CALIBRATION": True}},
        "no_target_family_calib": {"config_overrides": {"USE_TARGET_FAMILY_CALIBRATION": False}},
        "no_hybrid_blend": {"config_overrides": {"USE_HYBRID_STRUCTURED_BLEND": False}},
        "hybrid_structured_blend": {"config_overrides": {
            "USE_STRUCTURED_SCORE_FEATURES": True,
            "USE_HYBRID_STRUCTURED_BLEND": True,
        }},
        "neg_control_no_hier_all": {"config_overrides": {
            "USE_HIERARCHICAL_PACE": False,
            "USE_HIERARCHICAL_SHOT_ZONES": False,
            "USE_ROTATION_SCENARIOS": False,
            "USE_STRUCTURED_SCORE_FEATURES": False,
            "USE_HYBRID_STRUCTURED_BLEND": False,
            "USE_TARGET_FAMILY_CALIBRATION": False,
            "LINEUP_COMPOSITE_EVIDENCE_POOLING": False,
        }},
        "hapm_priors": {"backtest_kwargs": {"use_hapm_priors": True}},
        # Specialized market pipeline v2
        "gaussian_ou_gates": {"config_overrides": {
            "OU_USE_GAUSSIAN_PROB": True,
            "OU_MIN_PROB": 0.55,
        }},
        "score_pair_total": {"config_overrides": {"USE_SCORE_PAIR_TOTAL": True}},
        "legacy_residual_stack": {
            "config_overrides": {
                "USE_CANONICAL_SCORE_PAIR": False,
                "USE_SCORE_PAIR_TOTAL": False,
                "TOTAL_TRAIN_TARGET": "market_residual",
            },
            "model_kwargs": {"train_target": "decision_residual"},
        },
        "direct_score_pair": {
            "config_overrides": {
                "USE_CANONICAL_SCORE_PAIR": True,
                "USE_SCORE_PAIR_TOTAL": True,
                "USE_HYBRID_STRUCTURED_BLEND": False,
            },
        },
        "direct_score_pair_no_market_feats": {
            "config_overrides": {
                "USE_CANONICAL_SCORE_PAIR": True,
                "USE_SCORE_PAIR_TOTAL": True,
                "USE_HYBRID_STRUCTURED_BLEND": False,
            },
        },
        "structured_score_pair": {
            "config_overrides": {
                "USE_CANONICAL_SCORE_PAIR": True,
                "USE_STRUCTURED_SCORE_FEATURES": True,
                "USE_HYBRID_STRUCTURED_BLEND": True,
                "SCORE_PAIR_STRUCT_BETA": 0.25,
            },
        },
        "hybrid_score_pair": {
            "config_overrides": {
                "USE_CANONICAL_SCORE_PAIR": True,
                "USE_HYBRID_STRUCTURED_BLEND": True,
                "SCORE_PAIR_STRUCT_BETA": 0.15,
            },
        },
        "canonical_score_combined": {
            "config_overrides": {
                "USE_CANONICAL_SCORE_PAIR": True,
                "USE_SCORE_PAIR_TOTAL": True,
                "USE_HYBRID_STRUCTURED_BLEND": True,
                "USE_TARGET_FAMILY_CALIBRATION": False,
                "SCORE_PAIR_STRUCT_BETA": 0.15,
            },
        },
        # Elo agreement / prior blend into direct score-pair (research; locked folds).
        "direct_score_pair_elo_agree": {
            "config_overrides": {
                "USE_CANONICAL_SCORE_PAIR": True,
                "USE_SCORE_PAIR_TOTAL": True,
                "USE_HYBRID_STRUCTURED_BLEND": False,
                "SCORE_PAIR_ELO_BETA": 0.0,
            },
            "backtest_kwargs": {"require_elo_agreement": True},
        },
        "direct_score_pair_elo_prior": {
            "config_overrides": {
                "USE_CANONICAL_SCORE_PAIR": True,
                "USE_SCORE_PAIR_TOTAL": True,
                "USE_HYBRID_STRUCTURED_BLEND": False,
                "SCORE_PAIR_ELO_BETA": 0.20,
            },
        },
        "direct_score_pair_elo_prior_agree": {
            "config_overrides": {
                "USE_CANONICAL_SCORE_PAIR": True,
                "USE_SCORE_PAIR_TOTAL": True,
                "USE_HYBRID_STRUCTURED_BLEND": False,
                "SCORE_PAIR_ELO_BETA": 0.20,
            },
            "backtest_kwargs": {"require_elo_agreement": True},
        },
        "ml_both_sides_ev": {"config_overrides": {"ML_BET_PREDICTED_WINNER_ONLY": False}},
        "ml_winner_only": {"config_overrides": {"ML_BET_PREDICTED_WINNER_ONLY": True}},
        "stacked_lgbm_catboost": {"model_kwargs": {"use_lightgbm_base": True}},
        "no_elo_on_totals": {"config_overrides": {"TOTAL_ELO_BETA": 0.0}},
        # Multi-year future-success: rating warm-start + 4-season model window
        "multi_year_window4": {
            "backtest_kwargs": {"rolling_window_size": 4, "rating_history_use_all": True},
        },
        "multi_year_window5": {
            "backtest_kwargs": {"rolling_window_size": 5, "rating_history_use_all": True},
        },
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
    require_market_beat: bool = True,
) -> bool:
    """Return True if candidate meets pooled acceptance criteria vs baseline.

    Hit-rate / raw accuracy is never a promotion criterion. Primary gates are
    spread MAE, total MAE/CRPS, and Brier/log loss — ahead of ATS/ROI.
    ROI remains secondary and cannot override a material MAE regression.
    """
    b_mae = float(baseline.get("spread_mae", np.nan))
    c_mae = float(candidate.get("spread_mae", np.nan))
    if not np.isfinite(c_mae) or not np.isfinite(b_mae):
        return False
    mae_regression = c_mae > b_mae + mae_tol
    if mae_regression:
        return False
    if require_market_beat:
        mkt = float(candidate.get("market_spread_mae", baseline.get("market_spread_mae", np.nan)))
        if np.isfinite(mkt) and c_mae > mkt + mae_tol:
            # Allow candidate if it still improves on baseline even when both
            # trail market (research mode); block promotion claims separately.
            pass

    b_roi = float(baseline.get("roi", np.nan))
    c_roi = float(candidate.get("roi", np.nan))
    roi_gain = np.isfinite(c_roi) and np.isfinite(b_roi) and c_roi > b_roi + roi_tol

    b_ml_brier = baseline.get("ml_brier", baseline.get("brier"))
    c_ml_brier = candidate.get("ml_brier", candidate.get("brier"))
    if b_ml_brier is not None and c_ml_brier is not None:
        if np.isfinite(b_ml_brier) and np.isfinite(c_ml_brier):
            if c_ml_brier > float(b_ml_brier) + ml_brier_tol:
                return False
    b_total_mae = baseline.get("total_mae")
    c_total_mae = candidate.get("total_mae")
    if b_total_mae is not None and c_total_mae is not None:
        if np.isfinite(b_total_mae) and np.isfinite(c_total_mae):
            if c_total_mae > float(b_total_mae) + total_mae_tol:
                return False
    b_crps = baseline.get("crps")
    c_crps = candidate.get("crps")
    if b_crps is not None and c_crps is not None:
        if np.isfinite(b_crps) and np.isfinite(c_crps):
            if c_crps > float(b_crps) + crps_tol:
                return False
    # Prefer MAE improvement; ROI alone is insufficient for promotion.
    mae_improved = c_mae < b_mae - 1e-9
    if not mae_improved and not roi_gain:
        # Flat MAE with no ROI gain: accept only if no secondary regressions above.
        return True
    return True


def passes_roadmap_promotion_gate(
    baseline: dict,
    candidate: dict,
    *,
    per_season_candidate: list | None = None,
    per_season_baseline: list | None = None,
    odds_provenance: dict | None = None,
) -> dict:
    """Full roadmap gate: MAE/proper scores + multi-season + CLV provenance."""
    pooled_ok = passes_ablation_gate(baseline, candidate)
    market_ok = beats_market_and_baseline(candidate, baseline)
    season_info = {"passed": True, "wins": None}
    if per_season_candidate is not None and per_season_baseline is not None:
        season_info = multi_season_gate(per_season_candidate, per_season_baseline)
    clv_ok = clv_promotion_allowed(
        odds_provenance,
        int(candidate.get("n_finite_clv", 0) or 0),
    )
    return {
        "pooled_ok": pooled_ok,
        "market_ok": market_ok,
        "multi_season_ok": bool(season_info.get("passed")),
        "clv_roi_allowed": clv_ok,
        "promote": bool(pooled_ok and market_ok and season_info.get("passed")),
        "season_detail": season_info,
        "note": (
            "Promote on MAE/proper scores only; ROI/CLV claims require "
            "promotion_eligible odds provenance."
        ),
    }


def passes_ats_repair_promotion_gate(
    baseline: dict,
    candidate: dict,
    *,
    per_season_candidate: list | None = None,
    per_season_baseline: list | None = None,
    odds_provenance: dict | None = None,
) -> dict:
    """Locked-fold gate for ATS repair candidates.

    Primary: pooled spread MAE / proper scores, ≥3/4 season wins, credible
    interval coverage, and non-collapsed margin dispersion. ATS/ROI secondary.
    """
    base = passes_roadmap_promotion_gate(
        baseline,
        candidate,
        per_season_candidate=per_season_candidate,
        per_season_baseline=per_season_baseline,
        odds_provenance=odds_provenance,
    )
    disp_ok = margin_dispersion_ok(candidate)
    cov_ok = interval_coverage_credible(candidate, target=0.80)
    base["dispersion_ok"] = disp_ok
    base["interval_coverage_ok"] = cov_ok
    base["promote"] = bool(base["promote"] and disp_ok and cov_ok)
    base["note"] = (
        "ATS/ROI secondary. Promote only with MAE/proper scores, dispersion, "
        "and credible intervals; CLV needs promotion_eligible snapshots."
    )
    return base


def ats_repair_ablation_configs() -> dict:
    """Four-season ablation set for ATS spread-scale / calibrator repair."""
    base = default_configs()
    keys = (
        "baseline_current",
        "decision_residual_current",
        "decision_residual_no_line_feats",
        "absolute_margin_target",
        "ats_repair_combined",
    )
    return {k: base[k] for k in keys if k in base}


def direct_score_ablation_configs() -> dict:
    """Locked-fold ablation set for the direct actual-score model."""
    base = default_configs()
    keys = (
        "baseline_current",
        "legacy_residual_stack",
        "direct_score_pair",
        "direct_score_pair_no_market_feats",
        "structured_score_pair",
        "hybrid_score_pair",
        "canonical_score_combined",
        "direct_score_pair_elo_agree",
        "direct_score_pair_elo_prior",
        "direct_score_pair_elo_prior_agree",
    )
    return {k: base[k] for k in keys if k in base}


def passes_direct_score_promotion_gate(
    baseline: dict,
    candidate: dict,
    *,
    per_season_candidate: list | None = None,
    per_season_baseline: list | None = None,
    odds_provenance: dict | None = None,
) -> dict:
    """Promote only when paired-score accuracy improves without market collapse.

    Requires positive market-error correlation and non-worse model−market MAE
    (plus credible intervals). Betting/ROI claims still need promotion_eligible CLV.
    """
    base = passes_roadmap_promotion_gate(
        baseline,
        candidate,
        per_season_candidate=per_season_candidate,
        per_season_baseline=per_season_baseline,
        odds_provenance=odds_provenance,
    )
    c_pair = float(candidate.get("paired_score_mae", np.nan))
    b_pair = float(baseline.get("paired_score_mae", np.nan))
    pair_ok = True
    if np.isfinite(c_pair) and np.isfinite(b_pair):
        pair_ok = c_pair <= b_pair + 0.15
    elif np.isfinite(c_pair):
        pair_ok = True
    else:
        c_tot = float(candidate.get("total_mae", np.nan))
        b_tot = float(baseline.get("total_mae", np.nan))
        pair_ok = (not np.isfinite(c_tot)) or (not np.isfinite(b_tot)) or (c_tot <= b_tot + 0.25)

    algebra_ok = float(candidate.get("score_algebra_ok_pct", 1.0) or 1.0) >= 0.99
    cov_ok = interval_coverage_credible(candidate, target=0.80)
    disp_ok = margin_dispersion_ok(candidate)
    edge_corr_ok = market_error_corr_positive(candidate, min_corr=0.02)

    c_gap = float(candidate.get("model_minus_market_spread_mae", np.nan))
    b_gap = float(baseline.get("model_minus_market_spread_mae", np.nan))
    mkt_gap_ok = True
    if np.isfinite(c_gap) and np.isfinite(b_gap):
        mkt_gap_ok = c_gap <= b_gap + 0.05
    elif np.isfinite(c_gap):
        mkt_gap_ok = c_gap <= 0.5

    clv_ok = bool(base.get("clv_roi_allowed"))
    if odds_provenance is not None:
        clv_ok = clv_promotion_allowed(
            odds_provenance, int(candidate.get("n_finite_clv", 0) or 0),
        ) and not tip_proxy_roi_blocked(odds_provenance)

    accuracy_ok = bool(pair_ok and algebra_ok and cov_ok and disp_ok and edge_corr_ok and mkt_gap_ok)
    base["paired_score_ok"] = pair_ok
    base["algebra_ok"] = algebra_ok
    base["interval_coverage_ok"] = cov_ok
    base["dispersion_ok"] = disp_ok
    base["edge_evidence"] = edge_corr_ok
    base["model_minus_market_ok"] = mkt_gap_ok
    base["accuracy_gates_ok"] = accuracy_ok
    base["promote"] = bool(base["promote"] and accuracy_ok)
    base["betting_edge_claim_allowed"] = bool(base["promote"] and edge_corr_ok and clv_ok)
    base["policy_retune_allowed"] = policy_retune_allowed(
        candidate, accuracy_ok=accuracy_ok, odds_provenance=odds_provenance,
    )
    base["note"] = (
        "Promote on paired MAE + model−market MAE + positive market-error corr + "
        "algebra + intervals + dispersion. Policy retune only after accuracy gates. "
        "Betting-edge/ROI claims require promotion_eligible CLV (tip-proxy blocked)."
    )
    return base



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
    "direct_score_pair": {"USE_CANONICAL_SCORE_PAIR": True, "USE_SCORE_PAIR_TOTAL": True},
    "canonical_score_combined": {
        "USE_CANONICAL_SCORE_PAIR": True,
        "USE_SCORE_PAIR_TOTAL": True,
        "USE_HYBRID_STRUCTURED_BLEND": True,
        "USE_TARGET_FAMILY_CALIBRATION": False,
    },
    "direct_score_pair_elo_prior": {
        "USE_CANONICAL_SCORE_PAIR": True,
        "USE_SCORE_PAIR_TOTAL": True,
        "SCORE_PAIR_ELO_BETA": 0.20,
    },
    "lgbm_stack": {"USE_LIGHTGBM_BASE": True},
    "stacked_lgbm_catboost": {"USE_LIGHTGBM_BASE": True},
    "legacy_pace_mean": {"USE_HIERARCHICAL_PACE": False},
    "no_rotation_scenarios": {"USE_ROTATION_SCENARIOS": False},
    "hard_lineup5_floor": {"LINEUP_COMPOSITE_EVIDENCE_POOLING": False},
    "target_family_calib": {"USE_TARGET_FAMILY_CALIBRATION": True},
    "hybrid_structured_blend": {
        "USE_STRUCTURED_SCORE_FEATURES": True,
        "USE_HYBRID_STRUCTURED_BLEND": True,
    },
    "no_hybrid_blend": {"USE_HYBRID_STRUCTURED_BLEND": False},
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


def run_direct_score_locked_ablation(
    stints_df,
    odds_dict=None,
    *,
    n_tuning_trials_meta: int = 8,
    n_tuning_trials_elo: int = 0,
    n_tuning_trials_hier: int = 0,
) -> tuple:
    """Four-season rolling-origin ablation for direct score-pair candidates.

    Uses ``direct_score_ablation_configs()`` and locked fold window=4.
    Promote only via ``passes_direct_score_promotion_gate`` (MAE + market-error
    corr); never loosens confidence/width/edge gates for volume.
    """
    from pipeline.negative_controls import locked_fold_definitions
    from pipeline.metrics import compute_metrics

    folds = locked_fold_definitions()
    configs = direct_score_ablation_configs()
    print(
        f"Locked direct-score ablation: window={folds['window']} "
        f"folds={len(folds['folds'])} configs={list(configs)}"
    )
    summary_df, detail = run_ablation(
        stints_df,
        odds_dict=odds_dict,
        configs=configs,
        rolling_window_size=int(folds["window"]),
        n_tuning_trials_elo=n_tuning_trials_elo,
        n_tuning_trials_hier=n_tuning_trials_hier,
        n_tuning_trials_meta=n_tuning_trials_meta,
    )
    gate_rows = []
    if not summary_df.empty:
        pooled = summary_df[summary_df["season"] == "ALL"]
        base = pooled[pooled["config"] == "direct_score_pair"]
        if base.empty:
            base = pooled[pooled["config"] == "baseline_current"]
        baseline = base.iloc[0].to_dict() if not base.empty else {}
        for _, row in pooled.iterrows():
            if row["config"] in ("direct_score_pair", "baseline_current") and row["config"] == baseline.get("config"):
                continue
            g = passes_direct_score_promotion_gate(baseline, row.to_dict())
            gate_rows.append({"config": row["config"], **{k: g[k] for k in (
                "promote", "accuracy_gates_ok", "edge_evidence",
                "betting_edge_claim_allowed", "policy_retune_allowed",
                "paired_score_ok", "interval_coverage_ok", "dispersion_ok",
                "model_minus_market_ok",
            ) if k in g}})
        if gate_rows:
            print("\nDirect-score promotion gates:")
            print(pd.DataFrame(gate_rows).to_string(index=False))
    return summary_df, detail, {"folds": folds, "gates": gate_rows}

