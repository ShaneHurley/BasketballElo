"""Walk-forward upset classifier: P(market favorite loses the game)."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from pipeline.config import STATE_DIR

FEATURE_KEYS = [
    "abs_market_spread",
    "model_vs_market",
    "abs_model_vs_market",
    "matchup_vol_sigma",
    "spread_quantile_width",
    "elo_meta_agreement",
    "rating_uncertainty_sum",
    "h_star_out",
    "a_star_out",
    "disagreement_trust",
    "win_prob_fav",
    # Plan 2B extras: rest / travel / form volatility / Pythag residual / rotation
    "rest_diff_fav",
    "fav_b2b",
    "travel_diff_fav",
    "form_std_sum",
    "pythag_residual_diff",
    "missing_rotation_sum",
]


def _market_spread(row) -> float:
    for k in ("MARKET_SPREAD", "market_spread", "CLOSING_SPREAD", "closing_spread"):
        v = row.get(k) if hasattr(row, "get") else None
        if v is not None and pd.notna(v):
            return float(v)
    return np.nan


def _pred_margin(row) -> float:
    for k in ("PRED_SPREAD", "pred_margin", "RAW_PRED_MARGIN"):
        v = row.get(k) if hasattr(row, "get") else None
        if v is not None and pd.notna(v):
            return float(v)
    return 0.0


def _actual_margin(row) -> float:
    for k in ("ACTUAL_MARGIN", "actual_margin"):
        v = row.get(k) if hasattr(row, "get") else None
        if v is not None and pd.notna(v):
            return float(v)
    if "ACTUAL_HOME" in row and "ACTUAL_AWAY" in row:
        return float(row["ACTUAL_HOME"]) - float(row["ACTUAL_AWAY"])
    return np.nan


def favorite_lost_label(row) -> float | None:
    """1 if market favorite lost (moneyline), None if push / unknown."""
    mkt = _market_spread(row)
    margin = _actual_margin(row)
    if pd.isna(mkt) or pd.isna(margin) or abs(float(mkt)) < 1e-9:
        return None
    # Home favorite when market_spread < 0; favorite wins if margin has same sign as -mkt.
    home_fav = float(mkt) < 0
    home_won = float(margin) > 0
    if float(margin) == 0:
        return None
    fav_won = home_won if home_fav else (not home_won)
    return float(not fav_won)


def _num(row: dict, *keys, default: float = 0.0) -> float:
    for k in keys:
        v = row.get(k)
        if v is not None and pd.notna(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                continue
    return float(default)


def _feature_row(row: dict) -> dict:
    mkt = _market_spread(row)
    pred = _pred_margin(row)
    model_vs = pred + float(mkt) if pd.notna(mkt) else 0.0
    vol = row.get("MATCHUP_VOL_SIGMA", row.get("matchup_vol_sigma", 12.0))
    width = row.get("SPREAD_QUANTILE_WIDTH", row.get("spread_quantile_width",
                   row.get("CONF_WIDTH", 24.0)))
    agree = row.get("ELO_META_AGREEMENT", row.get("elo_meta_agreement", 1.0))
    unc = row.get("RATING_UNCERTAINTY", row.get("rating_uncertainty_sum", 700.0))
    win_prob = float(row.get("WIN_PROB", row.get("win_prob", 0.5)) or 0.5)
    home_fav = (float(mkt) < 0) if pd.notna(mkt) else True
    win_prob_fav = win_prob if home_fav else (1.0 - win_prob)

    h_rest = _num(row, "h_rest", "H_REST", default=2.0)
    a_rest = _num(row, "a_rest", "A_REST", default=2.0)
    h_b2b = _num(row, "h_b2b", "H_B2B", default=float(h_rest <= 1))
    a_b2b = _num(row, "a_b2b", "A_B2B", default=float(a_rest <= 1))
    h_travel = _num(row, "h_travel_miles_7d", "H_TRAVEL_MILES_7D")
    a_travel = _num(row, "a_travel_miles_7d", "A_TRAVEL_MILES_7D")
    travel_diff = _num(row, "travel_miles_diff", default=h_travel - a_travel)
    h_std = _num(row, "h_pts_std_roll_10", "h_pts_std_roll_5", "H_PTS_STD_ROLL_10")
    a_std = _num(row, "a_pts_std_roll_10", "a_pts_std_roll_5", "A_PTS_STD_ROLL_10")
    h_pyth = _num(row, "h_pythag_residual", "H_PYTHAG_RESIDUAL")
    a_pyth = _num(row, "a_pythag_residual", "A_PYTHAG_RESIDUAL")
    h_miss = _num(row, "h_missing_rotation", "H_MISSING_ROTATION")
    a_miss = _num(row, "a_missing_rotation", "A_MISSING_ROTATION")

    # Signed so positive = favorite is more rested / traveled less / etc.
    if home_fav:
        rest_diff_fav = h_rest - a_rest
        fav_b2b = h_b2b
        travel_diff_fav = h_travel - a_travel if (h_travel or a_travel) else travel_diff
        pythag_residual_diff = h_pyth - a_pyth
    else:
        rest_diff_fav = a_rest - h_rest
        fav_b2b = a_b2b
        travel_diff_fav = a_travel - h_travel if (h_travel or a_travel) else -travel_diff
        pythag_residual_diff = a_pyth - h_pyth

    return {
        "abs_market_spread": abs(float(mkt)) if pd.notna(mkt) else 0.0,
        "model_vs_market": float(model_vs),
        "abs_model_vs_market": abs(float(model_vs)),
        "matchup_vol_sigma": float(vol) if vol is not None and pd.notna(vol) else 12.0,
        "spread_quantile_width": float(width) if width is not None and pd.notna(width) else 24.0,
        "elo_meta_agreement": float(agree) if agree is not None and pd.notna(agree) else 1.0,
        "rating_uncertainty_sum": float(unc) if unc is not None and pd.notna(unc) else 700.0,
        "h_star_out": float(row.get("H_STAR_OUT", row.get("h_star_out", 0)) or 0),
        "a_star_out": float(row.get("A_STAR_OUT", row.get("a_star_out", 0)) or 0),
        "disagreement_trust": float(
            row.get("DISAGREEMENT_TRUST", row.get("disagreement_trust", 1.0)) or 1.0
        ),
        "win_prob_fav": float(win_prob_fav),
        "rest_diff_fav": float(rest_diff_fav),
        "fav_b2b": float(fav_b2b),
        "travel_diff_fav": float(travel_diff_fav),
        "form_std_sum": float(h_std + a_std),
        "pythag_residual_diff": float(pythag_residual_diff),
        "missing_rotation_sum": float(h_miss + a_miss),
    }


class UpsetClassifier:
    """Logistic + isotonic P(market favorite loses)."""

    def __init__(self, C: float = 0.1):
        self.C = float(C)
        self.model = LogisticRegression(C=self.C, solver="lbfgs", max_iter=300)
        self.scaler = StandardScaler()
        self.iso = None
        self.fitted = False

    def _matrix(self, df: pd.DataFrame) -> np.ndarray:
        rows = []
        for _, row in df.iterrows():
            feat = _feature_row(row.to_dict())
            rows.append([feat[k] for k in FEATURE_KEYS])
        return np.asarray(rows, dtype=float)

    def fit(self, train_df: pd.DataFrame, calib_df: pd.DataFrame | None = None):
        if train_df is None or train_df.empty:
            self.fitted = False
            return self
        y_list, keep = [], []
        for idx, row in train_df.iterrows():
            lab = favorite_lost_label(row)
            if lab is None:
                continue
            y_list.append(int(lab))
            keep.append(idx)
        if len(keep) < 50 or len(set(y_list)) < 2:
            self.fitted = False
            return self
        X = self._matrix(train_df.loc[keep])
        y = np.asarray(y_list, dtype=int)
        Xs = self.scaler.fit_transform(X)
        self.model.fit(Xs, y)
        self.iso = None
        if calib_df is not None and not calib_df.empty:
            cy, ckeep = [], []
            for idx, row in calib_df.iterrows():
                lab = favorite_lost_label(row)
                if lab is None:
                    continue
                cy.append(int(lab))
                ckeep.append(idx)
            if len(ckeep) >= 30 and len(set(cy)) > 1:
                Xc = self.scaler.transform(self._matrix(calib_df.loc[ckeep]))
                raw = self.model.predict_proba(Xc)[:, 1]
                self.iso = IsotonicRegression(out_of_bounds="clip")
                self.iso.fit(raw, np.asarray(cy, dtype=int))
        self.fitted = True
        return self

    def predict_upset_prob(self, feat_dict: dict) -> float:
        if not self.fitted:
            return 0.5
        feat = _feature_row(feat_dict)
        x = np.asarray([[feat[k] for k in FEATURE_KEYS]], dtype=float)
        xs = self.scaler.transform(x)
        raw = float(self.model.predict_proba(xs)[0, 1])
        if self.iso is not None:
            return float(np.clip(self.iso.predict([raw])[0], 0.01, 0.99))
        return float(np.clip(raw, 0.01, 0.99))

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str) -> "UpsetClassifier":
        with open(path, "rb") as f:
            return pickle.load(f)


def default_upset_classifier_path() -> Path:
    return STATE_DIR / "upset_classifier.pkl"
