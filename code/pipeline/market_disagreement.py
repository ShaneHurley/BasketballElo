"""Walk-forward market disagreement / phantom-injury gate."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from pipeline.config import STATE_DIR


def _disagreement_features(
    elo_vs_close: float,
    meta_vs_close: float,
    uncertainty: float = 350.0,
    spread_move: float = 0.0,
    star_out: float = 0.0,
) -> np.ndarray:
    return np.array([
        abs(float(elo_vs_close or 0.0)),
        abs(float(meta_vs_close or 0.0)),
        abs(float(elo_vs_close or 0.0) - float(meta_vs_close or 0.0)),
        np.sign(float(elo_vs_close or 0.0)) * np.sign(float(meta_vs_close or 0.0)),
        float(uncertainty or 350.0) / 350.0,
        abs(float(spread_move or 0.0)),
        float(star_out or 0.0),
    ], dtype=float)


def phantom_injury_flag(
    elo_vs_close: float,
    meta_vs_close: float,
    h_star_out: float = 0.0,
    a_star_out: float = 0.0,
    min_elo_edge: float = 4.0,
    max_meta_edge: float = 2.5,
) -> bool:
    """Large Elo–market gap without rotation flags — market may know injury news."""
    if h_star_out > 0 or a_star_out > 0:
        return False
    elo_e = abs(float(elo_vs_close or 0.0))
    meta_e = abs(float(meta_vs_close or 0.0))
    if elo_e < min_elo_edge:
        return False
    if meta_e > max_meta_edge:
        return False
    if np.sign(elo_vs_close or 0) != np.sign(meta_vs_close or 0) and meta_e >= 1.5:
        return True
    return elo_e >= min_elo_edge + 1.0 and meta_e <= max_meta_edge


class WalkForwardMarketDisagreementModel:
    """Secondary walk-forward model for model-vs-market disagreement nights."""

    def __init__(self, min_train: int = 80, min_elo_edge: float = 3.0):
        self.min_train = min_train
        self.min_elo_edge = min_elo_edge
        self._iso = None
        self._logit = None
        self._fitted = False
        self._phantom_cover_rate = 0.5

    @property
    def fitted(self) -> bool:
        return self._fitted

    def fit(self, prior_df: pd.DataFrame) -> "WalkForwardMarketDisagreementModel":
        if prior_df is None or prior_df.empty:
            return self
        need = {"ACTUAL_MARGIN", "MARKET_SPREAD", "PRED_SPREAD"}
        if not need.issubset(prior_df.columns):
            return self

        d = prior_df[prior_df["MARKET_SPREAD"].notna()].copy()
        close = d["CLOSING_SPREAD"] if "CLOSING_SPREAD" in d.columns else d["MARKET_SPREAD"]
        d["close"] = close
        d["cover"] = d["ACTUAL_MARGIN"] + d["MARKET_SPREAD"]
        d["meta_vs_close"] = d["PRED_SPREAD"] + d["close"]

        elo_col = "ELO_MARGIN_CALIBRATED" if "ELO_MARGIN_CALIBRATED" in d.columns else None
        if elo_col is None:
            return self
        d["elo_vs_close"] = d[elo_col] + d["close"]
        d["abs_elo_edge"] = d["elo_vs_close"].abs()

        active = d[d["abs_elo_edge"] >= self.min_elo_edge]
        if len(active) < self.min_train:
            return self

        X, y, scores = [], [], []
        for _, r in active.iterrows():
            star = max(float(r.get("H_STAR_OUT", 0) or 0), float(r.get("A_STAR_OUT", 0) or 0))
            unc = float(r.get("RATING_UNCERTAINTY", 350) or 350)
            sm = float(r.get("SPREAD_MOVE", 0) or 0)
            elo_e = float(r["elo_vs_close"])
            meta_e = float(r["meta_vs_close"])
            X.append(_disagreement_features(elo_e, meta_e, unc, sm, star))
            edge = float(r["PRED_SPREAD"]) + float(r["MARKET_SPREAD"])
            home_side = edge > 0
            covered = float(r["cover"]) > 0 if home_side else float(r["cover"]) < 0
            y.append(int(covered))
            scores.append(abs(elo_e))

        X_arr = np.vstack(X)
        y_arr = np.asarray(y, dtype=int)

        try:
            self._logit = LogisticRegression(C=0.5, max_iter=300)
            self._logit.fit(X_arr, y_arr)
        except Exception:
            self._logit = None

        try:
            self._iso = IsotonicRegression(out_of_bounds="clip")
            self._iso.fit(scores, y_arr)
        except Exception:
            self._iso = None

        phantom_mask = active.apply(
            lambda r: phantom_injury_flag(
                r["elo_vs_close"], r["meta_vs_close"],
                r.get("H_STAR_OUT", 0), r.get("A_STAR_OUT", 0),
            ),
            axis=1,
        )
        if phantom_mask.sum() >= 15:
            ph = active[phantom_mask]
            ph_edge = ph["PRED_SPREAD"] + ph["MARKET_SPREAD"]
            ph_home = ph_edge > 0
            ph_cover = ph["cover"]
            ph_win = ((ph_home & (ph_cover > 0)) | (~ph_home & (ph_cover < 0))).astype(float)
            self._phantom_cover_rate = float(ph_win.mean())
        else:
            self._phantom_cover_rate = float(y_arr.mean())

        self._fitted = True
        return self

    def predict(
        self,
        *,
        elo_margin_calibrated: float,
        model_spread: float,
        closing_spread: float,
        market_spread: float | None = None,
        uncertainty: float = 350.0,
        spread_move: float = 0.0,
        h_star_out: float = 0.0,
        a_star_out: float = 0.0,
    ) -> dict:
        close = closing_spread if pd.notna(closing_spread) else market_spread
        if close is None or pd.isna(close):
            return {
                "disagreement_trust": 1.0,
                "phantom_injury_flag": False,
                "edge_bump": 0.0,
                "model_side_cover_prob": 0.524,
            }

        elo_vs = float(elo_margin_calibrated or 0.0) + float(close)
        meta_vs = float(model_spread or 0.0) + float(close)
        phantom = phantom_injury_flag(elo_vs, meta_vs, h_star_out, a_star_out)
        feats = _disagreement_features(
            elo_vs, meta_vs, uncertainty, spread_move,
            max(h_star_out, a_star_out),
        ).reshape(1, -1)

        prob = 0.524
        if self._logit is not None:
            try:
                prob = float(self._logit.predict_proba(feats)[0, 1])
            except Exception:
                pass
        elif self._iso is not None and abs(elo_vs) >= self.min_elo_edge:
            try:
                prob = float(self._iso.predict([abs(elo_vs)])[0])
            except Exception:
                pass

        trust = float(np.clip(prob / 0.524, 0.35, 1.25))
        edge_bump = 0.0
        if phantom:
            if self._phantom_cover_rate < 0.50:
                trust *= 0.65
                edge_bump = 1.5
            else:
                edge_bump = 0.5

        if abs(elo_vs) >= 6.0 and abs(elo_vs - meta_vs) >= 4.0:
            edge_bump = max(edge_bump, 1.0)
            trust *= 0.85

        return {
            "disagreement_trust": float(np.clip(trust, 0.2, 1.2)),
            "phantom_injury_flag": bool(phantom),
            "edge_bump": float(edge_bump),
            "model_side_cover_prob": float(np.clip(prob, 0.01, 0.99)),
            "elo_vs_close": elo_vs,
            "meta_vs_close": meta_vs,
        }

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "WalkForwardMarketDisagreementModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def default_disagreement_path() -> Path:
    return Path(STATE_DIR) / "market_disagreement.pkl"
