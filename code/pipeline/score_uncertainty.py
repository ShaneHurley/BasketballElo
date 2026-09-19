"""Past-only score uncertainty and cover-probability calibration.

Uses OOF home/away residuals to set σ_home, σ_away, and residual correlation
for the joint score distribution. Cover probabilities are derived from the
calibrated margin distribution vs DECISION_SPREAD, then optionally Platt/isotonic
calibrated on past OOF covers only.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def oof_score_residual_moments(
    pred_home: np.ndarray,
    pred_away: np.ndarray,
    actual_home: np.ndarray,
    actual_away: np.ndarray,
    *,
    sigma_floor: float = 8.0,
    default_corr: float = 0.35,
    min_n: int = 40,
) -> dict[str, float]:
    """Estimate residual σ and corr from OOF predictions (past-only)."""
    ph = np.asarray(pred_home, dtype=float)
    pa = np.asarray(pred_away, dtype=float)
    ah = np.asarray(actual_home, dtype=float)
    aa = np.asarray(actual_away, dtype=float)
    m = np.isfinite(ph) & np.isfinite(pa) & np.isfinite(ah) & np.isfinite(aa)
    n = int(m.sum())
    if n < int(min_n):
        return {
            "sigma_home": float(sigma_floor),
            "sigma_away": float(sigma_floor),
            "residual_corr": float(default_corr),
            "n": n,
            "source": "floor_default",
        }
    rh = ah[m] - ph[m]
    ra = aa[m] - pa[m]
    sh = float(max(sigma_floor, np.sqrt(np.mean(rh ** 2))))
    sa = float(max(sigma_floor, np.sqrt(np.mean(ra ** 2))))
    corr = float(default_corr)
    if float(np.std(rh)) > 1e-9 and float(np.std(ra)) > 1e-9:
        c = float(np.corrcoef(rh, ra)[0, 1])
        if np.isfinite(c):
            corr = float(np.clip(c, -0.95, 0.95))
    return {
        "sigma_home": sh,
        "sigma_away": sa,
        "residual_corr": corr,
        "n": n,
        "source": "oof_residuals",
    }


def apply_oof_uncertainty_to_score_pair(
    score_pair_model,
    train_feats: pd.DataFrame,
    *,
    sigma_floor: float = 8.0,
    default_corr: float = 0.35,
) -> dict[str, float]:
    """Update a fitted MetaScorePairModel with OOF residual moments from sf_* cols."""
    if score_pair_model is None or not getattr(score_pair_model, "fitted", False):
        return {"source": "skipped"}
    if "sf_pred_home" not in train_feats.columns or "actual_home" not in train_feats.columns:
        return {"source": "missing_oof_cols"}
    moments = oof_score_residual_moments(
        train_feats["sf_pred_home"].to_numpy(),
        train_feats["sf_pred_away"].to_numpy(),
        train_feats["actual_home"].to_numpy(),
        train_feats["actual_away"].to_numpy(),
        sigma_floor=sigma_floor,
        default_corr=default_corr,
    )
    score_pair_model.set_walkforward_rmse(
        rmse_home=moments["sigma_home"],
        rmse_away=moments["sigma_away"],
        residual_corr=moments["residual_corr"],
    )
    return moments


class CoverProbCalibrator:
    """Past-only calibrator for P(home covers) from raw Gaussian cover probs."""

    def __init__(self, method: str = "platt"):
        self.method = method
        self._model = None
        self.fitted = False
        self.n_fit = 0

    def fit(
        self,
        raw_home_cover: np.ndarray,
        y_home_cover: np.ndarray,
        *,
        min_n: int = 80,
    ) -> "CoverProbCalibrator":
        p = np.asarray(raw_home_cover, dtype=float)
        y = np.asarray(y_home_cover, dtype=float)
        m = np.isfinite(p) & np.isfinite(y)
        p, y = p[m], y[m]
        self.n_fit = int(len(y))
        if self.n_fit < int(min_n) or float(np.unique(y).size) < 2:
            self.fitted = False
            self._model = None
            return self
        p = np.clip(p, 1e-4, 1.0 - 1e-4)
        if self.method == "isotonic":
            from sklearn.isotonic import IsotonicRegression
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
            iso.fit(p, y)
            self._model = ("isotonic", iso)
        else:
            from sklearn.linear_model import LogisticRegression
            # Platt: logit(p) → y
            logit = np.log(p / (1.0 - p)).reshape(-1, 1)
            lr = LogisticRegression(C=1.0, solver="lbfgs", max_iter=200)
            lr.fit(logit, y.astype(int))
            self._model = ("platt", lr)
        self.fitted = True
        return self

    def transform(self, raw_home_cover: float | np.ndarray) -> float | np.ndarray:
        scalar = np.isscalar(raw_home_cover)
        p = np.atleast_1d(np.asarray(raw_home_cover, dtype=float))
        out = np.clip(p, 0.01, 0.99)
        if not self.fitted or self._model is None:
            return float(out[0]) if scalar else out
        kind, model = self._model
        pp = np.clip(p, 1e-4, 1.0 - 1e-4)
        if kind == "isotonic":
            cal = np.clip(model.predict(pp), 0.01, 0.99)
        else:
            logit = np.log(pp / (1.0 - pp)).reshape(-1, 1)
            cal = np.clip(model.predict_proba(logit)[:, 1], 0.01, 0.99)
        return float(cal[0]) if scalar else cal


def home_cover_labels_from_frame(df: pd.DataFrame) -> np.ndarray:
    """Binary home-cover labels vs DECISION_SPREAD (fallback MARKET_SPREAD)."""
    act = pd.to_numeric(df.get("actual_margin", df.get("ACTUAL_MARGIN")), errors="coerce")
    line = df.get("decision_spread", df.get("DECISION_SPREAD"))
    if line is None:
        line = df.get("market_spread", df.get("MARKET_SPREAD"))
    line = pd.to_numeric(line, errors="coerce")
    # Home covers when actual_margin + decision_spread > 0; pushes → nan
    cover_margin = act + line
    y = np.where(cover_margin > 0, 1.0, np.where(cover_margin < 0, 0.0, np.nan))
    return y.astype(float)


def fit_cover_calibrator_from_oof(
    train_feats: pd.DataFrame,
    *,
    sigma_margin: float,
    method: str = "platt",
) -> CoverProbCalibrator:
    """Fit cover calibrator on OOF pair margins vs decision line."""
    from pipeline.score_targets import margin_home_cover_prob

    cal = CoverProbCalibrator(method=method)
    if "sf_fair_margin" not in train_feats.columns:
        return cal
    pred_m = pd.to_numeric(train_feats["sf_fair_margin"], errors="coerce")
    line = train_feats.get("decision_spread", train_feats.get("market_spread"))
    line = pd.to_numeric(line, errors="coerce")
    y = home_cover_labels_from_frame(train_feats)
    raw = []
    for pm, ln in zip(pred_m.to_numpy(), line.to_numpy()):
        if np.isfinite(pm) and np.isfinite(ln):
            raw.append(margin_home_cover_prob(float(pm), float(ln), float(sigma_margin)))
        else:
            raw.append(np.nan)
    cal.fit(np.asarray(raw, dtype=float), y)
    return cal


def phantom_ev_gap(mean_claimed_ev: float, realized_roi: float, *, max_gap: float = 0.08) -> bool:
    """True when claimed EV is materially inflated vs realized ROI (diagnostic fail)."""
    if not np.isfinite(mean_claimed_ev) or not np.isfinite(realized_roi):
        return True
    return float(mean_claimed_ev) - float(realized_roi) > float(max_gap)
