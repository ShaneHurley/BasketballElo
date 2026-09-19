"""Walk-forward calibration for MetaWin home win probability (prior season only)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pipeline.config import (
    ML_CALIB_METHOD,
    ML_CALIB_REQUIRE_ECE_IMPROVEMENT,
    ML_SPLIT_CALIB_BY_FAVORITE,
    STATE_DIR,
)


def _home_win_series(df: pd.DataFrame) -> pd.Series:
    if "ACTUAL_HOME" not in df.columns or "ACTUAL_AWAY" not in df.columns:
        return pd.Series(dtype=float)
    return (df["ACTUAL_HOME"].astype(float) > df["ACTUAL_AWAY"].astype(float)).astype(int)


def prior_year_ml_frame(prior_df: pd.DataFrame, season_col: str = "simulated_season_window") -> pd.DataFrame:
    """Last simulated season only (walk-forward calib slice)."""
    if prior_df is None or prior_df.empty:
        return pd.DataFrame()
    if season_col not in prior_df.columns:
        return prior_df
    seasons = sorted(prior_df[season_col].dropna().unique())
    if not seasons:
        return prior_df
    return prior_df[prior_df[season_col] == seasons[-1]].copy()


def _is_home_favorite(row) -> bool | None:
    """True if home is market favorite; None if unknown / pick'em."""
    for key in ("MARKET_ML", "market_ml"):
        ml = row.get(key) if hasattr(row, "get") else None
        if ml is not None and pd.notna(ml):
            try:
                return float(ml) < 0
            except (TypeError, ValueError):
                pass
    for key in ("MARKET_SPREAD", "market_spread", "CLOSING_SPREAD"):
        spr = row.get(key) if hasattr(row, "get") else None
        if spr is not None and pd.notna(spr):
            try:
                s = float(spr)
                if abs(s) < 1e-9:
                    return None
                return s < 0  # home favored when home spread negative
            except (TypeError, ValueError):
                pass
    return None


def _fit_1d_calibrator(p: np.ndarray, y: np.ndarray, method: str):
    if len(p) < 40 or len(np.unique(y)) < 2:
        return None
    if method == "isotonic":
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(p, y)
        return iso
    if method in ("platt", "logistic"):
        clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=200)
        clf.fit(p.reshape(-1, 1), y)
        return clf
    return None


def _apply_model(model, p: float) -> float:
    p = float(np.clip(p, 0.01, 0.99))
    if model is None:
        return p
    if isinstance(model, IsotonicRegression):
        return float(np.clip(model.predict([p])[0], 0.01, 0.99))
    return float(np.clip(model.predict_proba(np.array([[p]], dtype=float))[0, 1], 0.01, 0.99))


def _ece(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> float:
    from pipeline.calibration_metrics import compute_ece
    return float(compute_ece(y, p, n_bins=n_bins))


class WalkForwardMLCalibrator:
    """Map raw MetaWin P(home) → calibrated P(home) using prior-season outcomes.

    Optionally fits separate underdog vs favorite maps and only enables
    calibration when prior-year ECE improves vs raw.
    """

    def __init__(self, method: str | None = None):
        self.method = (method or ML_CALIB_METHOD or "platt").lower()
        self.model = None  # global fallback
        self.model_fav = None
        self.model_dog = None
        self._fitted = False
        self._use_calibrated = False
        self._split = bool(ML_SPLIT_CALIB_BY_FAVORITE)
        self._fit_season: str | None = None
        self._raw_ece: float | None = None
        self._cal_ece: float | None = None

    def fit(
        self,
        prior_df: pd.DataFrame,
        *,
        prob_col: str = "WIN_PROB_RAW",
        scope: str = "prior_year",
        season_col: str = "simulated_season_window",
        min_samples: int = 80,
    ) -> "WalkForwardMLCalibrator":
        df = prior_year_ml_frame(prior_df, season_col=season_col) if scope == "prior_year" else prior_df
        self._fitted = False
        self._use_calibrated = False
        self.model = self.model_fav = self.model_dog = None
        if df is None or df.empty:
            return self

        col = prob_col if prob_col in df.columns else "WIN_PROB"
        if col not in df.columns:
            return self

        y = _home_win_series(df)
        p = pd.to_numeric(df[col], errors="coerce")
        mask = p.notna() & y.notna()
        p_arr = p[mask].astype(float).values
        y_arr = y[mask].astype(int).values
        rows = df.loc[mask]
        if len(p_arr) < min_samples or len(np.unique(y_arr)) < 2:
            return self

        if season_col in df.columns and df[season_col].notna().any():
            self._fit_season = str(df[season_col].dropna().iloc[-1])

        self.model = _fit_1d_calibrator(p_arr, y_arr, self.method)
        if self._split:
            fav_idx, dog_idx = [], []
            for i, (_, row) in enumerate(rows.iterrows()):
                fav = _is_home_favorite(row)
                if fav is True:
                    fav_idx.append(i)
                elif fav is False:
                    dog_idx.append(i)
            if len(fav_idx) >= 40:
                self.model_fav = _fit_1d_calibrator(p_arr[fav_idx], y_arr[fav_idx], self.method)
            if len(dog_idx) >= 40:
                self.model_dog = _fit_1d_calibrator(p_arr[dog_idx], y_arr[dog_idx], self.method)

        raw_ece = _ece(y_arr, p_arr)
        cal_p = np.asarray([
            self._transform_unchecked(float(pp), _is_home_favorite(row))
            for pp, (_, row) in zip(p_arr, rows.iterrows())
        ], dtype=float)
        cal_ece = _ece(y_arr, cal_p)
        self._raw_ece = raw_ece
        self._cal_ece = cal_ece
        self._fitted = self.model is not None or self.model_fav is not None or self.model_dog is not None
        if not self._fitted:
            return self
        if ML_CALIB_REQUIRE_ECE_IMPROVEMENT:
            # Strict: calibrated must beat raw; else EV path uses WIN_PROB_RAW.
            self._use_calibrated = bool(
                np.isfinite(cal_ece) and np.isfinite(raw_ece) and cal_ece <= raw_ece + 1e-9
            )
        else:
            self._use_calibrated = True
        return self

    def _transform_unchecked(self, p_home: float, home_is_fav: bool | None) -> float:
        if self._split and home_is_fav is True and self.model_fav is not None:
            return _apply_model(self.model_fav, p_home)
        if self._split and home_is_fav is False and self.model_dog is not None:
            return _apply_model(self.model_dog, p_home)
        return _apply_model(self.model, p_home)

    def transform(
        self,
        p_home: float,
        *,
        market_ml: float | None = None,
        market_spread: float | None = None,
    ) -> float:
        """Calibrate P(home) when fit improved ECE; else return raw (clipped)."""
        if p_home is None or not np.isfinite(p_home):
            return 0.5
        raw = float(np.clip(p_home, 0.01, 0.99))
        if not self._fitted or not self._use_calibrated:
            return raw
        home_is_fav = None
        if market_ml is not None and pd.notna(market_ml):
            home_is_fav = float(market_ml) < 0
        elif market_spread is not None and pd.notna(market_spread):
            s = float(market_spread)
            if abs(s) >= 1e-9:
                home_is_fav = s < 0
        return self._transform_unchecked(raw, home_is_fav)

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "WalkForwardMLCalibrator":
        with open(path, "rb") as f:
            return pickle.load(f)


def default_ml_calibrator_path() -> Path:
    return STATE_DIR / "ml_calibrator.pkl"
