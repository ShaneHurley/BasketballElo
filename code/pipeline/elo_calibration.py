"""Walk-forward calibration for ELO-derived margins (unseen-data generalization)."""
from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import HuberRegressor
from sklearn.preprocessing import StandardScaler

from pipeline.config import STATE_DIR

# Three leak-free calibration groups (fit walk-forward, chronological).
ELO_CALIB_GROUPS = {
    "engine": [
        "elo_margin",
        "hier_margin",
        "lineup5_net",
        "team_elo_spread",
        "exp_poss",
        "elo_net",
        "elo_luck_adj_net",
        "elo_def_event_rate",
        "elo_tov_rate",
        "elo_matchup_asym",
    ],
    "uncertainty": [
        "uncertainty_diff",
        "h_rating_uncertainty",
        "a_rating_uncertainty",
        "elo_uncertainty_adj",
    ],
    "market": [
        "market_spread",
    ],
}

ELO_CALIBRATED_COLS = [
    "elo_margin_calibrated",
    "elo_hier_blend",
    "elo_vs_hier_spread",
    "elo_vs_market",
]


@dataclass
class EloCalibrationKnobs:
    """Tuned, walk-forward-safe defaults for ELO calibration + meta anchor."""

    huber_epsilon: float = 1.35
    min_samples: int = 80
    refit_g2_isotonic: bool = True
    hier_blend_elo: float = 0.6
    hier_blend_hier: float = 0.4
    elo_blend_alpha: float = 0.35
    elo_ridge_alpha: float = 3.0
    group_train_mae: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "EloCalibrationKnobs":
        if not d:
            return cls()
        known = {k: v for k, v in d.items() if k in cls.__dataclass_fields__}
        return cls(**known)


def _feat_vector(feat: dict, cols: list) -> np.ndarray:
    row = []
    for c in cols:
        v = feat.get(c, 0.0)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            v = 0.0
        row.append(float(v))
    return np.asarray(row, dtype=float)


def _group_matrix(df: pd.DataFrame, cols: list) -> tuple[np.ndarray, list]:
    avail = [c for c in cols if c in df.columns]
    if not avail:
        return np.zeros((len(df), 1)), avail
    return df[avail].fillna(0.0).to_numpy(dtype=float), avail


def _ensure_uncertainty_col(df: pd.DataFrame) -> pd.DataFrame:
    if "elo_uncertainty_adj" in df.columns:
        return df
    out = df.copy()
    tu = out.get("h_rating_uncertainty", pd.Series(350, index=out.index)).fillna(350) + \
         out.get("a_rating_uncertainty", pd.Series(350, index=out.index)).fillna(350)
    out["elo_uncertainty_adj"] = out.get("elo_net", 0).fillna(0) / (tu + 1e-6)
    return out


class WalkForwardEloCalibrator:
    """
    Three-group walk-forward ELO margin calibrator.

    Group 1 (engine): player/hier/lineup/team ratings → base margin.
    Group 2 (uncertainty): rating uncertainty adjustments (Huber on train,
        isotonic re-fit on calib slice after groups 1 & 3 are fixed).
    Group 3 (market): closing spread anchor on remaining residual.

    All groups use only pre-game features; calib slice is strictly later games.
    """

    def __init__(self, knobs: EloCalibrationKnobs | None = None):
        self.knobs = knobs or EloCalibrationKnobs()
        self.group_cols: dict[str, list] = {k: list(v) for k, v in ELO_CALIB_GROUPS.items()}
        self.scalers: dict[str, StandardScaler] = {
            g: StandardScaler() for g in ELO_CALIB_GROUPS
        }
        self.models: dict[str, HuberRegressor | None] = {
            g: None for g in ELO_CALIB_GROUPS
        }
        self.g2_isotonic: IsotonicRegression | None = None
        self.fitted = False
        self.coef_report: dict = {}

    def _huber(self) -> HuberRegressor:
        return HuberRegressor(
            epsilon=self.knobs.huber_epsilon,
            max_iter=300,
        )

    def _fit_group(
        self,
        df: pd.DataFrame,
        group: str,
        target: np.ndarray,
    ) -> tuple[np.ndarray, list]:
        X, cols = _group_matrix(df, self.group_cols[group])
        if len(cols) == 0 or len(target) < max(20, self.knobs.min_samples // 4):
            return np.zeros(len(df)), cols
        self.group_cols[group] = cols
        Xs = self.scalers[group].fit_transform(X)
        model = self._huber()
        model.fit(Xs, target)
        self.models[group] = model
        pred = model.predict(Xs)
        self.coef_report[group] = dict(zip(cols, model.coef_.tolist()))
        self.coef_report[f"{group}_intercept"] = float(model.intercept_)
        return pred, cols

    def _predict_group(
        self,
        df: pd.DataFrame,
        group: str,
        *,
        use_isotonic: bool = False,
    ) -> np.ndarray:
        cols = self.group_cols.get(group, [])
        if not cols or self.models.get(group) is None:
            return np.zeros(len(df))
        X, _ = _group_matrix(df, cols)
        Xs = self.scalers[group].transform(X)
        lin = self.models[group].predict(Xs)
        if group == "uncertainty" and use_isotonic and self.g2_isotonic is not None:
            return self.g2_isotonic.predict(lin)
        return lin

    def fit(
        self,
        train_df: pd.DataFrame,
        calib_df: pd.DataFrame | None = None,
    ):
        """Walk-forward fit: train groups 1→2→3, then re-fit group-2 isotonic on calib."""
        if train_df is None or train_df.empty or "actual_margin" not in train_df.columns:
            self.fitted = False
            return self

        train_df = _ensure_uncertainty_col(train_df)
        if calib_df is not None and not calib_df.empty:
            calib_df = _ensure_uncertainty_col(calib_df)

        if len(train_df) < self.knobs.min_samples:
            self.fitted = False
            return self

        y = train_df["actual_margin"].to_numpy(dtype=float)

        pred1, _ = self._fit_group(train_df, "engine", y)
        resid1 = y - pred1

        pred2, _ = self._fit_group(train_df, "uncertainty", resid1)
        resid2 = resid1 - pred2

        pred3, _ = self._fit_group(train_df, "market", resid2)

        train_mae = float(np.mean(np.abs(y - (pred1 + pred2 + pred3))))
        self.knobs.group_train_mae = {
            "engine": float(np.mean(np.abs(y - pred1))),
            "uncertainty": float(np.mean(np.abs(resid1 - pred2))),
            "market": float(np.mean(np.abs(resid2 - pred3))),
            "combined": train_mae,
        }
        self.fitted = True

        # Re-fit group-2 isotonic on calib after groups 1 & 3 are fixed (walk-forward).
        self.g2_isotonic = None
        if self.knobs.refit_g2_isotonic and calib_df is not None and not calib_df.empty:
            cols2 = self.group_cols.get("uncertainty", [])
            if self.models.get("uncertainty") is not None and cols2:
                p1_c = self._predict_group(calib_df, "engine")
                p3_c = self._predict_group(calib_df, "market")
                y_c = calib_df["actual_margin"].to_numpy(dtype=float)
                target_g2 = y_c - p1_c - p3_c
                X2, _ = _group_matrix(calib_df, cols2)
                X2s = self.scalers["uncertainty"].transform(X2)
                lin2 = self.models["uncertainty"].predict(X2s)
                if len(lin2) >= 30 and len(np.unique(np.sign(target_g2))) > 1:
                    self.g2_isotonic = IsotonicRegression(out_of_bounds="clip")
                    self.g2_isotonic.fit(lin2, target_g2)
        return self

    def predict(self, feat: dict) -> float:
        if not self.fitted:
            return float(feat.get("elo_margin", 0.0) or 0.0)
        row = _ensure_uncertainty_col(pd.DataFrame([feat]))
        return float(self.predict_batch(row)[0])

    def predict_batch(self, df: pd.DataFrame) -> np.ndarray:
        if not self.fitted or df is None or df.empty:
            if df is not None and "elo_margin" in df.columns:
                return df["elo_margin"].fillna(0.0).to_numpy(dtype=float)
            return np.zeros(len(df) if df is not None else 0)
        df = _ensure_uncertainty_col(df)
        p1 = self._predict_group(df, "engine")
        p2 = self._predict_group(df, "uncertainty", use_isotonic=True)
        p3 = self._predict_group(df, "market")
        return p1 + p2 + p3

    def predict_interval(self, feat: dict, alpha: float = 0.10) -> tuple[float, float, float]:
        pred = self.predict(feat)
        stored = getattr(self, "_residuals", None)
        if stored is not None and len(stored) >= 20:
            a = float(alpha)
            if not np.isfinite(a) or a <= 0.0 or a >= 1.0:
                a = 0.10
            q = float(np.quantile(np.abs(stored), 1.0 - a))
        else:
            q = getattr(self, "_resid_q", 12.0)
        return pred - q, pred + q, 2.0 * q

    def update_residuals(self, residuals: list[float]):
        if len(residuals) >= 20:
            self._residuals = [float(x) for x in residuals]
            self._resid_q = float(np.quantile(np.abs(self._residuals), 0.90))

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str):
        with open(path, "rb") as f:
            return pickle.load(f)


def default_elo_calibrator_path() -> Path:
    return STATE_DIR / "elo_calibrator.pkl"


def default_elo_knobs_path() -> Path:
    return STATE_DIR / "elo_calibration_knobs.json"


def save_elo_knobs(knobs: EloCalibrationKnobs, path: Path | str | None = None):
    path = Path(path or default_elo_knobs_path())
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    path.write_text(json.dumps(knobs.to_dict(), indent=2))


def load_elo_knobs(path: Path | str | None = None) -> EloCalibrationKnobs:
    path = Path(path or default_elo_knobs_path())
    if not path.exists():
        return EloCalibrationKnobs()
    import json
    return EloCalibrationKnobs.from_dict(json.loads(path.read_text()))


def augment_elo_features(
    feat: dict,
    calibrator: WalkForwardEloCalibrator | None = None,
    knobs: EloCalibrationKnobs | None = None,
) -> dict:
    """Add calibrated ELO margin and blend features to a game feature dict."""
    out = dict(feat)
    if "elo_uncertainty_adj" not in out or pd.isna(out.get("elo_uncertainty_adj")):
        tu = float(out.get("h_rating_uncertainty", 350) + out.get("a_rating_uncertainty", 350))
        out["elo_uncertainty_adj"] = float(out.get("elo_net", 0) or 0) / (tu + 1e-6)

    k = knobs or (calibrator.knobs if calibrator is not None else EloCalibrationKnobs())
    elo_raw = float(out.get("elo_margin", 0.0) or 0.0)
    hier = float(out.get("hier_margin", 0.0) or 0.0)

    if calibrator is not None and calibrator.fitted:
        cal = calibrator.predict(out)
    else:
        unc = float(out.get("h_rating_uncertainty", 350) + out.get("a_rating_uncertainty", 350))
        w_elo = min(0.85, max(0.55, 1.0 - (unc - 200) / 800))
        hier_part = (1.0 - w_elo) * 0.5 * (elo_raw + hier)
        cal = w_elo * elo_raw + hier_part

    out["elo_margin_calibrated"] = cal
    out["elo_hier_blend"] = k.hier_blend_elo * elo_raw + k.hier_blend_hier * hier
    out["elo_vs_hier_spread"] = elo_raw - hier
    if out.get("market_spread") is not None and not pd.isna(out.get("market_spread")):
        mkt = -float(out["market_spread"])
        out["elo_vs_market"] = cal - mkt
    else:
        out["elo_vs_market"] = 0.0
    return out


def tune_elo_blend_weights(features_df: pd.DataFrame) -> dict:
    """
    Walk-forward grid search for ELO / hier blend weights (chronological tail MAE).
    Returns suggested overrides for augment + PlayerRatingTracker scaling hints.
    """
    if features_df is None or features_df.empty:
        return {}
    df = features_df.dropna(subset=["actual_margin", "elo_margin"]).copy()
    if len(df) < 100:
        return {}

    if "game_date" in df.columns:
        df = df.sort_values("game_date")
    inner_n = max(len(df) // 5, 50)
    inner_train = df.iloc[:-inner_n]
    inner_val = df.iloc[-inner_n:]
    if len(inner_val) < 30:
        inner_train, inner_val = df, df.iloc[-max(30, len(df) // 10):]

    best_mae, best = 999.0, {}
    y = inner_val["actual_margin"].values
    elo = inner_val["elo_margin"].values
    hier = inner_val.get("hier_margin", pd.Series(0, index=inner_val.index)).fillna(0).values

    for w_elo in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        for w_hier in (0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4):
            if w_elo + w_hier > 1.01:
                continue
            pred = w_elo * elo + w_hier * hier
            mae = np.mean(np.abs(pred - y))
            if mae < best_mae:
                best_mae = mae
                best = {
                    "hier_blend_elo": w_elo,
                    "hier_blend_hier": w_hier,
                    "blend_mae": mae,
                }

    if len(inner_train) >= 50:
        elo_tr = inner_train["elo_margin"].values
        y_tr = inner_train["actual_margin"].values
        A = np.column_stack([elo_tr, np.ones(len(elo_tr))])
        coef, _, _, _ = np.linalg.lstsq(A, y_tr, rcond=None)
        best["elo_scale"] = float(coef[0])
        best["elo_bias"] = float(coef[1])
    return best


def tune_elo_calibrator(
    train_df: pd.DataFrame,
    calib_df: pd.DataFrame | None = None,
) -> EloCalibrationKnobs:
    """
    Walk-forward grid for calibrator hyperparameters (small grid, no leakage).

    Uses chronological tail of train_df as inner validation; calib_df only simulates
    the outer walk-forward calib slice for isotonic re-fit scoring.
    """
    knobs = EloCalibrationKnobs()
    if train_df is None or train_df.empty or len(train_df) < 120:
        return knobs

    df = _ensure_uncertainty_col(train_df)
    if "game_date" in df.columns:
        df = df.sort_values("game_date").reset_index(drop=True)

    inner_cut = max(len(df) // 5, 80)
    inner_train = df.iloc[:-inner_cut]
    inner_val = df.iloc[-inner_cut:]
    calib_slice = calib_df
    if calib_slice is None or calib_slice.empty:
        calib_slice = inner_val.iloc[max(0, len(inner_val) // 2):]

    best_mae = 999.0
    best_knobs = EloCalibrationKnobs()

    for huber_eps in (1.15, 1.35, 1.55, 1.75):
        for min_s in (60, 80, 100):
            if len(inner_train) < min_s:
                continue
            for refit_iso in (True, False):
                trial = EloCalibrationKnobs(
                    huber_epsilon=huber_eps,
                    min_samples=min_s,
                    refit_g2_isotonic=refit_iso,
                )
                cal = WalkForwardEloCalibrator(knobs=trial)
                cal.fit(inner_train, calib_df=calib_slice)
                if not cal.fitted:
                    continue
                pred = cal.predict_batch(inner_val)
                mae = float(np.mean(np.abs(inner_val["actual_margin"].values - pred)))
                if mae < best_mae:
                    best_mae = mae
                    best_knobs = trial
                    best_knobs.group_train_mae = {"inner_val_mae": mae}

    blend = tune_elo_blend_weights(inner_train)
    if blend:
        best_knobs.hier_blend_elo = blend.get("hier_blend_elo", best_knobs.hier_blend_elo)
        best_knobs.hier_blend_hier = blend.get("hier_blend_hier", best_knobs.hier_blend_hier)
    return best_knobs


def apply_elo_calibration_df(
    df: pd.DataFrame,
    calibrator: WalkForwardEloCalibrator | None = None,
) -> pd.DataFrame:
    """Apply walk-forward ELO calibration columns (vectorized)."""
    if df is None or df.empty:
        return df
    out = _ensure_uncertainty_col(df.copy())
    knobs = calibrator.knobs if calibrator is not None else EloCalibrationKnobs()

    if calibrator is not None and calibrator.fitted:
        cal_vals = calibrator.predict_batch(out)
    else:
        elo = out.get("elo_margin", pd.Series(0, index=out.index)).fillna(0).values
        hier = out.get("hier_margin", pd.Series(0, index=out.index)).fillna(0).values
        unc = (
            out.get("h_rating_uncertainty", pd.Series(350, index=out.index)).fillna(350).values
            + out.get("a_rating_uncertainty", pd.Series(350, index=out.index)).fillna(350).values
        )
        w_elo = np.clip(1.0 - (unc - 200) / 800, 0.55, 0.85)
        cal_vals = w_elo * elo + (1.0 - w_elo) * 0.5 * (elo + hier)

    out["elo_margin_calibrated"] = cal_vals
    elo_raw = out.get("elo_margin", pd.Series(0, index=out.index)).fillna(0)
    hier_raw = out.get("hier_margin", pd.Series(0, index=out.index)).fillna(0)
    out["elo_hier_blend"] = knobs.hier_blend_elo * elo_raw + knobs.hier_blend_hier * hier_raw
    out["elo_vs_hier_spread"] = elo_raw - hier_raw
    if "market_spread" in out.columns:
        mkt = -out["market_spread"].fillna(0.0)
        out["elo_vs_market"] = out["elo_margin_calibrated"] - mkt
    else:
        out["elo_vs_market"] = 0.0
    return out
