"""Leakage-safe shared score forecasts for ATS / ML / totals heads (T-60 Phase 3).

Produces one cross-fitted schema per game: expected home/away points, fair
margin/total, residuals to T-60 market, win probability, uncertainty, and
data-quality flags. Downstream betting heads must train on OOF columns only.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from pipeline.oof import PastOnlyGroupCV


SHARED_FORECAST_COLS = (
    "sf_pred_home",
    "sf_pred_away",
    "sf_fair_margin",
    "sf_fair_total",
    "sf_decision_residual",
    "sf_total_residual",
    "sf_home_win_prob",
    "sf_margin_sigma",
    "sf_total_sigma",
    "sf_rotation_var",
    "sf_rating_uncertainty",
    "sf_dq_missing_decision",
    "sf_dq_missing_total",
    "sf_dq_high_uncertainty",
    "sf_oof_covered",
)


@dataclass(frozen=True)
class SharedForecastRow:
    pred_home: float
    pred_away: float
    fair_margin: float
    fair_total: float
    decision_residual: float
    total_residual: float
    home_win_prob: float
    margin_sigma: float
    total_sigma: float
    rotation_var: float
    rating_uncertainty: float
    dq_missing_decision: int
    dq_missing_total: int
    dq_high_uncertainty: int
    oof_covered: int = 1

    def as_dict(self, prefix: str = "sf_") -> dict[str, float]:
        raw = asdict(self)
        mapping = {
            "pred_home": f"{prefix}pred_home",
            "pred_away": f"{prefix}pred_away",
            "fair_margin": f"{prefix}fair_margin",
            "fair_total": f"{prefix}fair_total",
            "decision_residual": f"{prefix}decision_residual",
            "total_residual": f"{prefix}total_residual",
            "home_win_prob": f"{prefix}home_win_prob",
            "margin_sigma": f"{prefix}margin_sigma",
            "total_sigma": f"{prefix}total_sigma",
            "rotation_var": f"{prefix}rotation_var",
            "rating_uncertainty": f"{prefix}rating_uncertainty",
            "dq_missing_decision": f"{prefix}dq_missing_decision",
            "dq_missing_total": f"{prefix}dq_missing_total",
            "dq_high_uncertainty": f"{prefix}dq_high_uncertainty",
            "oof_covered": f"{prefix}oof_covered",
        }
        return {mapping[k]: float(v) for k, v in raw.items()}


def _norm_cdf(x: float) -> float:
    # Abramowitz–Stegun approximation; avoids scipy dependency.
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989423 * np.exp(-0.5 * x * x)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    return 1.0 - p if x > 0 else p


def row_from_scores(
    pred_home: float,
    pred_away: float,
    *,
    decision_spread: float | None = None,
    market_total: float | None = None,
    margin_sigma: float = 12.0,
    total_sigma: float = 14.0,
    rotation_var: float = 0.0,
    rating_uncertainty: float = 350.0,
    high_unc_threshold: float = 500.0,
    oof_covered: int = 1,
) -> SharedForecastRow:
    ph = float(pred_home)
    pa = float(pred_away)
    margin = ph - pa
    total = ph + pa
    miss_dec = int(decision_spread is None or not np.isfinite(float(decision_spread)))
    miss_tot = int(market_total is None or not np.isfinite(float(market_total)) or float(market_total) <= 0)
    dec = float(decision_spread) if not miss_dec else np.nan
    mt = float(market_total) if not miss_tot else np.nan
    # Market-implied home margin ≈ -decision_spread (home gets points when spread > 0).
    decision_resid = (margin + dec) if not miss_dec else np.nan
    total_resid = (total - mt) if not miss_tot else np.nan
    sigma_m = max(float(margin_sigma), 1e-3)
    home_win = float(_norm_cdf(margin / sigma_m))
    return SharedForecastRow(
        pred_home=ph,
        pred_away=pa,
        fair_margin=margin,
        fair_total=total,
        decision_residual=float(decision_resid) if np.isfinite(decision_resid) else np.nan,
        total_residual=float(total_resid) if np.isfinite(total_resid) else np.nan,
        home_win_prob=home_win,
        margin_sigma=sigma_m,
        total_sigma=max(float(total_sigma), 1e-3),
        rotation_var=float(rotation_var),
        rating_uncertainty=float(rating_uncertainty),
        dq_missing_decision=miss_dec,
        dq_missing_total=miss_tot,
        dq_high_uncertainty=int(float(rating_uncertainty) >= float(high_unc_threshold)),
        oof_covered=int(oof_covered),
    )


def attach_shared_forecast_columns(df: pd.DataFrame, rows: Sequence[Mapping[str, Any]]) -> pd.DataFrame:
    """Merge shared-forecast dicts into a feature frame (train/serve parity)."""
    out = df.copy()
    for col in SHARED_FORECAST_COLS:
        out[col] = np.nan
    for i, row in enumerate(rows):
        if i >= len(out):
            break
        for k, v in row.items():
            if k in out.columns:
                out.iloc[i, out.columns.get_loc(k)] = v
    return out


def emit_shared_forecast_from_feature_row(feat: Mapping[str, Any], preds: Mapping[str, Any]) -> dict[str, float]:
    """Serve-time helper: build schema from live feature + model predictions."""
    pred_home = preds.get("pred_home")
    pred_away = preds.get("pred_away")
    if pred_home is None or pred_away is None:
        margin = float(preds.get("pred_margin", feat.get("matchup_margin", feat.get("elo_margin", 0.0))) or 0.0)
        total = float(preds.get("pred_total", feat.get("matchup_total", 220.0)) or 220.0)
        pred_home = (total + margin) / 2.0
        pred_away = (total - margin) / 2.0
    unc = float(feat.get("h_rating_uncertainty", 0) or 0) + float(feat.get("a_rating_uncertainty", 0) or 0)
    rotation_var = float(feat.get("scenario_mix_sigma", feat.get("matchup_uncertainty", 0.0)) or 0.0)
    row = row_from_scores(
        float(pred_home),
        float(pred_away),
        decision_spread=feat.get("decision_spread", feat.get("market_spread")),
        market_total=feat.get("market_total"),
        margin_sigma=float(preds.get("sigma_margin", 12.0) or 12.0),
        total_sigma=float(preds.get("sigma_total", 14.0) or 14.0),
        rotation_var=rotation_var,
        rating_uncertainty=unc if unc > 0 else float(feat.get("matchup_uncertainty", 350.0) or 350.0),
        oof_covered=1,
    )
    return row.as_dict()


def cross_fit_score_pair_forecasts(
    X_feat: np.ndarray,
    y_home: np.ndarray,
    y_away: np.ndarray,
    groups,
    *,
    fit_fn,
    predict_fn,
    n_splits: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
    """Cross-fit home/away score predictions with past-only folds.

    ``fit_fn(X, y_h, y_a) -> artifact``
    ``predict_fn(artifact, X) -> (pred_home, pred_away)`` arrays.
    Returns OOF home, OOF away, covered mask, and final fit artifact.
    """
    X_feat = np.asarray(X_feat, dtype=float)
    y_home = np.asarray(y_home, dtype=float)
    y_away = np.asarray(y_away, dtype=float)
    n = len(X_feat)
    groups = np.arange(n) if groups is None else np.asarray(groups)
    y_margin = y_home - y_away
    cv = PastOnlyGroupCV(n_splits=n_splits)
    oof_h = np.full(n, np.nan)
    oof_a = np.full(n, np.nan)
    covered = np.zeros(n, dtype=bool)
    for train_idx, val_idx in cv.split(X_feat, y_margin, groups):
        art = fit_fn(X_feat[train_idx], y_home[train_idx], y_away[train_idx])
        ph, pa = predict_fn(art, X_feat[val_idx])
        oof_h[val_idx] = np.asarray(ph, dtype=float)
        oof_a[val_idx] = np.asarray(pa, dtype=float)
        covered[val_idx] = True
    final = fit_fn(X_feat, y_home, y_away)
    setattr(final, "fit_max_timestamp", groups.max())
    return oof_h, oof_a, covered, final


def build_oof_shared_frame(
    base_df: pd.DataFrame,
    oof_home: np.ndarray,
    oof_away: np.ndarray,
    covered: np.ndarray,
) -> pd.DataFrame:
    """Attach OOF shared-forecast columns for downstream head training."""
    rows = []
    for i in range(len(base_df)):
        if not covered[i] or not np.isfinite(oof_home[i]) or not np.isfinite(oof_away[i]):
            rows.append({c: np.nan for c in SHARED_FORECAST_COLS})
            rows[-1]["sf_oof_covered"] = 0.0
            continue
        feat = base_df.iloc[i]
        row = row_from_scores(
            float(oof_home[i]),
            float(oof_away[i]),
            decision_spread=feat.get("decision_spread", feat.get("market_spread")),
            market_total=feat.get("market_total"),
            margin_sigma=float(feat.get("matchup_uncertainty", 12.0) or 12.0) / 50.0 + 10.0,
            total_sigma=14.0,
            rotation_var=float(feat.get("scenario_mix_sigma", 0.0) or 0.0),
            rating_uncertainty=float(feat.get("h_rating_uncertainty", 0) or 0)
            + float(feat.get("a_rating_uncertainty", 0) or 0),
            oof_covered=1,
        )
        rows.append(row.as_dict())
    return attach_shared_forecast_columns(base_df, rows)


def attach_oof_shared_forecasts(
    train_feats: pd.DataFrame,
    meta_model=None,
    *,
    groups=None,
    n_splits: int = 5,
    feature_cols: Sequence[str] | None = None,
    score_pair_model=None,
) -> pd.DataFrame:
    """Attach past-only OOF ``sf_*`` columns — never in-sample meta predictions.

    Uses a lightweight score-pair ridge refit inside ``PastOnlyGroupCV`` folds so
    downstream heads cannot train on the same-fold meta stack outputs. ``meta_model``
    is accepted for API compatibility / future wiring but is not used for OOF values.

    When ``score_pair_model`` is provided, OOF features prefer its score-safe
    feature schema (without fitting the in-sample stacks again).
    """
    del meta_model  # OOF must not call in-sample _predict_raw on the fitted meta.
    if train_feats is None or train_feats.empty:
        return train_feats

    work = train_feats.copy()
    if "actual_home" not in work.columns or "actual_away" not in work.columns:
        for c in SHARED_FORECAST_COLS:
            work[c] = np.nan
        work["sf_oof_covered"] = 0.0
        return work

    if feature_cols is None and score_pair_model is not None:
        fitted = getattr(score_pair_model, "_fit_cols_", None) or []
        if fitted:
            feature_cols = list(fitted)
        else:
            try:
                from pipeline.score_targets import score_safe_feature_cols
                feature_cols = score_safe_feature_cols(work)
            except Exception:
                feature_cols = None

    default_cols = [
        "elo_margin", "elo_net", "hier_net", "exp_poss", "matchup_margin", "matchup_total",
        "h_rating_uncertainty", "a_rating_uncertainty", "pace_diff",
    ]
    cols = [c for c in (feature_cols or default_cols) if c in work.columns]
    # Strip market columns even if a caller passes them.
    try:
        from pipeline.score_targets import filter_score_safe_features
        cols = filter_score_safe_features(cols, available=work.columns, enforce=False)
    except Exception:
        pass
    if not cols:
        for c in SHARED_FORECAST_COLS:
            work[c] = np.nan
        work["sf_oof_covered"] = 0.0
        return work

    X = work[cols].fillna(0.0).to_numpy(dtype=float)
    y_h = work["actual_home"].to_numpy(dtype=float)
    y_a = work["actual_away"].to_numpy(dtype=float)
    if groups is None:
        if "game_date" in work.columns:
            groups = pd.to_datetime(work["game_date"], errors="coerce").astype("int64").to_numpy()
        else:
            groups = np.arange(len(work))

    from sklearn.linear_model import Ridge

    class _Pair:
        def __init__(self):
            self.h = Ridge(alpha=10.0)
            self.a = Ridge(alpha=10.0)

        def fit(self, X_tr, yh, ya):
            self.h.fit(X_tr, yh)
            self.a.fit(X_tr, ya)
            return self

        def predict(self, X_te):
            return self.h.predict(X_te), self.a.predict(X_te)

    def fit_fn(X_tr, yh, ya):
        return _Pair().fit(X_tr, yh, ya)

    def predict_fn(art, X_te):
        return art.predict(X_te)

    try:
        oof_h, oof_a, covered, _final = cross_fit_score_pair_forecasts(
            X, y_h, y_a, groups, fit_fn=fit_fn, predict_fn=predict_fn, n_splits=n_splits,
        )
    except Exception:
        for c in SHARED_FORECAST_COLS:
            work[c] = np.nan
        work["sf_oof_covered"] = 0.0
        return work

    return build_oof_shared_frame(work, oof_h, oof_a, covered)


# Re-export for feature-group documentation
FEATURE_GROUPS = {
    "rating_matchup": [
        "elo_margin", "elo_net", "hier_net", "matchup_margin", "matchup_home_pp100",
        "matchup_away_pp100", "matchup_off_vs_def_home", "matchup_off_vs_def_away",
    ],
    "rotation_availability": [
        "h_missing_rotation", "a_missing_rotation", "h_star_out", "a_star_out",
        "h_rating_uncertainty", "a_rating_uncertainty", "matchup_uncertainty",
    ],
    "pace_scoring": [
        "exp_poss", "pace_diff", "matchup_total", "matchup_home_pts", "matchup_away_pts",
        "market_total", "sf_fair_total",
    ],
    "schedule_travel": [
        "h_rest", "a_rest", "h_b2b", "a_b2b", "travel_miles_diff", "fatigue_diff",
    ],
    "t60_market": [
        "decision_spread", "market_spread", "market_ml", "market_ml_away",
        "spread_home_price", "spread_away_price", "market_fair_win_prob",
    ],
    "uncertainty": [
        "sf_margin_sigma", "sf_total_sigma", "sf_rotation_var", "sf_rating_uncertainty",
        "sf_dq_high_uncertainty",
    ],
}
