"""Walk-forward calibration for MetaWin home win probability (prior season only)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pipeline.config import ML_CALIB_METHOD, STATE_DIR


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


class WalkForwardMLCalibrator:
    """Map raw MetaWin P(home) → calibrated P(home) using prior-season outcomes."""

    def __init__(self, method: str | None = None):
        self.method = (method or ML_CALIB_METHOD or "platt").lower()
        self.model = None
        self._fitted = False
        self._fit_season: str | None = None

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
        if df is None or df.empty:
            self._fitted = False
            return self

        col = prob_col if prob_col in df.columns else "WIN_PROB"
        if col not in df.columns:
            self._fitted = False
            return self

        y = _home_win_series(df)
        p = pd.to_numeric(df[col], errors="coerce")
        mask = p.notna() & y.notna()
        p = p[mask].astype(float).values
        y = y[mask].astype(int).values
        if len(p) < min_samples or len(np.unique(y)) < 2:
            self._fitted = False
            return self

        if season_col in df.columns and df[season_col].notna().any():
            self._fit_season = str(df[season_col].dropna().iloc[-1])

        if self.method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(p, y)
            self.model = iso
        elif self.method in ("platt", "logistic"):
            clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=200)
            clf.fit(p.reshape(-1, 1), y)
            self.model = clf
        else:
            self.model = None
            self._fitted = False
            return self

        self._fitted = True
        return self

    def transform(self, p_home: float) -> float:
        if not self._fitted or self.model is None or p_home is None or not np.isfinite(p_home):
            return float(np.clip(p_home, 0.01, 0.99))
        p = float(np.clip(p_home, 0.01, 0.99))
        if isinstance(self.model, IsotonicRegression):
            out = float(self.model.predict([p])[0])
        else:
            out = float(self.model.predict_proba(np.array([[p]], dtype=float))[0, 1])
        return float(np.clip(out, 0.01, 0.99))

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
