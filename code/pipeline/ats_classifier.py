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


def _finite_or(value, default):
    """Return float(value) when finite; otherwise float(default)."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return float(default)
    return f if np.isfinite(f) else float(default)


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
    def _decision_spread(row) -> float:
        """T-60 decision/market line for features — never closing (CLV-only)."""
        for k in ("decision_spread", "DECISION_SPREAD", "market_spread", "MARKET_SPREAD"):
            v = row.get(k) if hasattr(row, "get") else None
            if v is not None and pd.notna(v):
                return float(v)
        return np.nan

    @staticmethod
    def _close_spread(row) -> float:
        """Backward-compatible alias; does NOT fall back to closing_spread."""
        return ATSClassifier._decision_spread(row)

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
        decision = self._decision_spread(r)
        if pd.isna(decision):
            r["decision_missing"] = 1
            # Match training-time fillna(0) — never leave NaN in the feature row.
            r["model_edge"] = EXTRA_FEATURE_DEFAULTS["model_edge"]
            r["abs_model_edge"] = EXTRA_FEATURE_DEFAULTS["abs_model_edge"]
        else:
            r["decision_missing"] = 0
            edge = pred + float(decision)
            r["model_edge"] = edge
            r["abs_model_edge"] = abs(edge)
        h_unc = _finite_or(
            r.get("h_rating_uncertainty", r.get("H_RATING_UNCERTAINTY")), 350
        )
        a_unc = _finite_or(
            r.get("a_rating_uncertainty", r.get("A_RATING_UNCERTAINTY")), 350
        )
        r["rating_uncertainty_sum"] = h_unc + a_unc
        r["elo_meta_agreement_feat"] = _finite_or(
            r.get("elo_meta_agreement", r.get("ELO_META_AGREEMENT")), 1.0
        )
        r["spread_quantile_width_feat"] = _finite_or(
            r.get("spread_quantile_width", r.get("SPREAD_QUANTILE_WIDTH")), 24.0
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
        """Canonical home-cover labels vs T-60 decision line (pushes → NaN).

        ``home_cover = actual_margin + decision_home_spread > 0``.
        Direction flip to the selected bet side happens once at predict time.
        """
        from pipeline.market_targets import decision_spread_series

        decision = decision_spread_series(df)
        if decision is None:
            decision = pd.to_numeric(
                df.get("decision_spread", df.get("market_spread", df.get("MARKET_SPREAD"))),
                errors="coerce",
            )
        margin = df["actual_margin"] if "actual_margin" in df.columns else df["ACTUAL_MARGIN"]
        cover = margin + decision
        # Canonical home-cover label; Away conversion is predict-time only.
        y = np.where(cover > 0, 1, 0).astype(float)
        push = cover == 0
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

    def predict_home_cover_prob(self, feat_dict: dict, market_spread: float) -> float:
        """Canonical P(home covers vs decision/T-60 line). Never side-flipped."""
        if not self.fitted:
            return 0.524
        row = dict(feat_dict)
        row["market_spread"] = market_spread
        row["decision_spread"] = market_spread
        row = self._enrich_row(row)
        X = np.array(
            [[
                _finite_or(row.get(c), EXTRA_FEATURE_DEFAULTS.get(c, 0.0))
                for c in self._fit_cols
            ]],
            dtype=float,
        )
        Xs = self.scaler.transform(X)
        p_home = float(self.model.predict_proba(Xs)[0, 1])
        if self.iso is not None:
            try:
                p_home = float(self.iso.predict([p_home])[0])
            except Exception:
                pass
        return float(np.clip(p_home, 0.01, 0.99))

    def predict_cover_prob(self, feat_dict: dict, direction: str, market_spread: float) -> float:
        """Backward-compatible wrapper.

        Returns P(home covers). Callers that need the chosen-side probability
        must convert once via ``ats_ev.cover_prob_for_side``.
        """
        return self.predict_home_cover_prob(feat_dict, market_spread)

    def save(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path):
        with open(path, "rb") as f:
            return pickle.load(f)
