"""Walk-forward ATS cover classifier (direct P(cover) optimization)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline.market import elo_meta_agreement
from pipeline.market_targets import closing_spread_series

DERIVED_COLS = [
    "model_edge", "abs_model_edge", "rating_uncertainty_sum",
    "elo_meta_agreement_feat", "spread_quantile_width_feat",
]

EXTRA_FEATURE_DEFAULTS = {
    "model_edge": 0.0,
    "abs_model_edge": 0.0,
    "rating_uncertainty_sum": 700.0,
    "elo_meta_agreement_feat": 1.0,
    "spread_quantile_width_feat": 24.0,
}


class ATSClassifier:
    def __init__(self, feature_cols=None, calibrator: str = "isotonic", C: float = 0.1):
        from pipeline.model import SAFE_FEATURE_COLS
        self.feature_cols = list(feature_cols) if feature_cols else [
            c for c in SAFE_FEATURE_COLS if c in SAFE_FEATURE_COLS
        ][:40]
        self.calibrator_name = calibrator
        self.model = LogisticRegression(C=C, solver="lbfgs", max_iter=300)
        self.scaler = StandardScaler()
        self.iso = None
        self._fit_cols: list[str] = []
        self.fitted = False

    @staticmethod
    def _close_spread(row) -> float:
        for k in ("closing_spread", "CLOSING_SPREAD", "market_spread", "MARKET_SPREAD"):
            v = row.get(k) if hasattr(row, "get") else None
            if v is not None and pd.notna(v):
                return float(v)
        return np.nan

    @staticmethod
    def _pred_spread(row) -> float:
        for k in ("pred_margin", "PRED_SPREAD", "pred_spread"):
            v = row.get(k) if hasattr(row, "get") else None
            if v is not None and pd.notna(v):
                return float(v)
        return 0.0

    def _enrich_row(self, row: dict) -> dict:
        r = dict(row)
        pred = self._pred_spread(r)
        close = self._close_spread(r)
        mkt = r.get("market_spread", r.get("MARKET_SPREAD", close))
        edge = pred + float(mkt) if pd.notna(mkt) else pred + close
        r["model_edge"] = edge
        r["abs_model_edge"] = abs(edge)
        h_unc = float(r.get("h_rating_uncertainty", r.get("H_RATING_UNCERTAINTY", 350)) or 350)
        a_unc = float(r.get("a_rating_uncertainty", r.get("A_RATING_UNCERTAINTY", 350)) or 350)
        r["rating_uncertainty_sum"] = h_unc + a_unc
        r["elo_meta_agreement_feat"] = float(
            r.get("elo_meta_agreement", r.get("ELO_META_AGREEMENT", 1.0)) or 1.0
        )
        r["spread_quantile_width_feat"] = float(
            r.get("spread_quantile_width", r.get("SPREAD_QUANTILE_WIDTH", 24)) or 24
        )
        return r

    def _build_matrix(self, df: pd.DataFrame) -> np.ndarray:
        cols = list(self._fit_cols)
        X = pd.DataFrame(index=df.index)
        for c in cols:
            if c in df.columns:
                X[c] = df[c].fillna(0)
            else:
                X[c] = 0.0
        for _, row in df.iterrows():
            enriched = self._enrich_row(row.to_dict())
            for c in DERIVED_COLS:
                if c not in cols:
                    continue
                idx = row.name
                X.loc[idx, c] = enriched.get(c, EXTRA_FEATURE_DEFAULTS.get(c, 0))
        return X[cols].fillna(0).to_numpy(dtype=float)

    @staticmethod
    def labels_from_df(df: pd.DataFrame) -> np.ndarray:
        close = closing_spread_series(df)
        if close is None:
            close = pd.to_numeric(df.get("market_spread", df.get("MARKET_SPREAD")), errors="coerce")
        margin = df["actual_margin"] if "actual_margin" in df.columns else df["ACTUAL_MARGIN"]
        cover = margin + close
        pred = df.get("pred_margin", df.get("PRED_SPREAD", 0))
        edge = pred + close
        direction = np.where(edge >= 0, "Home", "Away")
        y = np.where(cover > 0, 1, 0)
        y = np.where(direction == "Away", 1 - y, y)
        push = cover == 0
        y = y.astype(float)
        y[push] = np.nan
        return y

    def fit(self, train_df: pd.DataFrame, calib_df: pd.DataFrame | None = None):
        train_df = train_df.dropna(subset=["actual_margin"] if "actual_margin" in train_df.columns else ["ACTUAL_MARGIN"])
        y = self.labels_from_df(train_df)
        mask = np.isfinite(y)
        train_df = train_df.loc[mask].copy()
        y = y[mask].astype(int)
        if len(train_df) < 50 or len(np.unique(y)) < 2:
            self.fitted = False
            return self

        base_cols = [c for c in self.feature_cols if c in train_df.columns]
        self._fit_cols = base_cols + [c for c in DERIVED_COLS if c not in base_cols]
        X = self._build_matrix(train_df)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)

        if calib_df is not None and not calib_df.empty:
            cy = self.labels_from_df(calib_df)
            cm = np.isfinite(cy)
            calib_df = calib_df.loc[cm].copy()
            cy = cy[cm].astype(int)
            if len(calib_df) >= 30 and len(np.unique(cy)) > 1:
                Xc = self.scaler.transform(self._build_matrix(calib_df))
                raw = self.model.predict_proba(Xc)[:, 1]
                if self.calibrator_name == "beta":
                    try:
                        from betacal import BetaCalibration
                        self.iso = BetaCalibration(parameters="ab")
                        self.iso.fit(raw, cy)
                    except ImportError:
                        self.iso = IsotonicRegression(out_of_bounds="clip")
                        self.iso.fit(raw, cy)
                else:
                    self.iso = IsotonicRegression(out_of_bounds="clip")
                    self.iso.fit(raw, cy)
        self.fitted = True
        return self

    def predict_cover_prob(self, feat_dict: dict, direction: str, market_spread: float) -> float:
        if not self.fitted:
            return 0.524
        row = self._enrich_row(dict(feat_dict))
        row["market_spread"] = market_spread
        X = np.array([[row.get(c, EXTRA_FEATURE_DEFAULTS.get(c, 0)) for c in self._fit_cols]], dtype=float)
        Xs = self.scaler.transform(X)
        raw = float(self.model.predict_proba(Xs)[0, 1])
        if direction == "Away":
            raw = 1.0 - raw
        if self.iso is not None:
            try:
                prob = float(self.iso.predict([raw if direction == "Home" else 1 - raw])[0])
            except Exception:
                prob = float(self.iso.predict(np.array([raw]))[0])
            if direction == "Away":
                prob = 1.0 - prob if self.calibrator_name != "beta" else 1.0 - prob
            return float(np.clip(prob if direction == "Home" else 1.0 - (1.0 - prob), 0.01, 0.99))
        return float(np.clip(raw if direction == "Home" else 1.0 - raw, 0.01, 0.99))

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as f:
            return pickle.load(f)
