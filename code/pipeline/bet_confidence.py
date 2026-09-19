"""Walk-forward empirical confidence calibrators per bet type."""
from __future__ import annotations

import pickle
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from pipeline.config import (
    CONFIDENCE_CALIB_METHOD,
    CONFIDENCE_MODE,
    CONFIDENCE_MIN_EDGE,
    CONFIDENCE_WEIGHT_SEARCH_SAMPLES,
    CONFIDENCE_WEIGHT_SHRINK,
    STATE_DIR,
)

EDGE_BUCKETS = [0, 2, 4, 6, 8, np.inf]
EDGE_LABELS = ["0-2", "2-4", "4-6", "6-8", "8+"]

DEFAULT_BUCKET_ATS_LIFT = {
    "0-2": -0.02,
    "2-4": -0.012,
    "4-6": +0.030,
    "6-8": +0.003,
    "8+": +0.058,
}

# Hand-tuned score coefficients (Phase 2 default calibration).
DEFAULT_CONFIDENCE_WEIGHTS: dict[str, float] = {
    "edge_slope": 1.5,
    "bucket_lift_scale": 40.0,
    "interval_scale": 6.0,
    "interval_cap": 28.0,
    "unc_penalty_scale": 5.0,
    "unc_center": 180.0,
    "unc_span": 400.0,
    "agree_bonus": 8.0,
    "agree_penalty": -4.0,
    "cover_scale": 35.0,
    "breakeven_cover": 0.524,
    "vol_penalty_scale": 8.0,
    "trust_scale": 12.0,
    "trust_center": 0.75,
    "phantom_penalty": 4.0,
    "ats_scale": 20.0,
    "win_prob_scale": 28.0,
    "elo_margin_scale": 1.25,
    "elo_align_bonus": 7.0,
}

# Wide multiplicative search bounds (× default) for Phase 2a first-year tuning.
DEFAULT_WEIGHT_SEARCH_MULT = (0.25, 2.5)


def clamp_confidence_weights(weights: dict[str, float]) -> dict[str, float]:
    """Clamp Phase 2a weights to absolute bounds (blocks cover_scale explosions)."""
    from pipeline.config import CONFIDENCE_WEIGHT_ABS_BOUNDS

    out = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    out.update(weights or {})
    bounds = CONFIDENCE_WEIGHT_ABS_BOUNDS or {}
    for k, v in list(out.items()):
        if k not in DEFAULT_CONFIDENCE_WEIGHTS:
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            fv = float(DEFAULT_CONFIDENCE_WEIGHTS[k])
        if k in bounds:
            lo, hi = bounds[k]
            fv = float(np.clip(fv, float(lo), float(hi)))
        out[k] = fv
    return out


# Order for logistic_features / logistic_isotonic calibration vector.
# All features are oriented so higher → historically better ATS cover rate.
CONFIDENCE_FEATURE_NAMES = (
    "composite_norm",
    "abs_edge_norm",
    "spread_cover_prob",
    "ats_classifier_prob",
    "win_prob_side",
    "disagreement_trust",
    "elo_meta_agreement",
    "elo_lean_align",
    "conf_tightness",
    "unc_quality",
    "vol_quality",
    "no_phantom",
)


def lean_side_win_prob(win_prob: float, lean: str) -> float:
    wp = float(win_prob if win_prob is not None else 0.5)
    if lean == "Away":
        return 1.0 - wp
    return wp


def elo_lean_align(lean: str, elo_margin: float) -> float:
    if lean in ("Pass", None, "", "nan") or (isinstance(lean, float) and pd.isna(lean)):
        return 0.5
    em = float(elo_margin or 0)
    return 1.0 if (lean == "Home") == (em >= 0) else 0.0


def _edge_bucket(abs_edge: float) -> str:
    for lo, hi, lbl in zip(EDGE_BUCKETS[:-1], EDGE_BUCKETS[1:], EDGE_LABELS):
        if lo <= abs_edge < hi:
            return lbl
    return EDGE_LABELS[-1]


def _tier_from_prob(p: float) -> int:
    if p >= 0.62:
        return 5
    if p >= 0.58:
        return 4
    if p >= 0.555:
        return 3
    if p >= 0.535:
        return 2
    return 1


def _stars_from_tier(tier: int, direction: str = "Home") -> str:
    if direction == "Pass" or tier < 2:
        return "Pass"
    labels = {
        2: "⭐⭐ (Lean)",
        3: "⭐⭐⭐ (Good)",
        4: "⭐⭐⭐⭐ (Strong)",
        5: "⭐⭐⭐⭐⭐ (Elite)",
    }
    return labels.get(tier, "⭐⭐ (Lean)")


def empirical_bucket_ats_rates(prior_df: pd.DataFrame) -> dict[str, float]:
    """ATS win rate by |edge| bucket from prior walk-forward results."""
    if prior_df is None or prior_df.empty or "EDGE" not in prior_df.columns:
        return {}
    d = prior_df[prior_df["MARKET_SPREAD"].notna()].copy()
    if "DIRECTION" in d.columns:
        d = d[d["DIRECTION"] != "Pass"]
    elif "EDGE_LEAN" in d.columns:
        d = d[d["EDGE_LEAN"].fillna("Pass") != "Pass"]
    if d.empty:
        return {}
    cover = d["ACTUAL_MARGIN"] + d["MARKET_SPREAD"]
    if "DIRECTION" in d.columns:
        home = d["DIRECTION"] == "Home"
    else:
        home = d["EDGE_LEAN"] == "Home"
    d = d.assign(
        ats_win=((home & (cover > 0)) | (~home & (cover < 0))).astype(float),
        abs_edge=d["EDGE"].abs(),
    )
    d = d[d["ats_win"].notna()]
    rates = {}
    for lbl in EDGE_LABELS:
        lo, hi = EDGE_BUCKETS[EDGE_LABELS.index(lbl)], EDGE_BUCKETS[EDGE_LABELS.index(lbl) + 1]
        sub = d[(d["abs_edge"] >= lo) & (d["abs_edge"] < hi)]
        if len(sub) >= 15:
            rates[lbl] = float(sub["ats_win"].mean())
    return rates


def prior_year_frame(
    df: pd.DataFrame,
    season_col: str = "simulated_season_window",
) -> pd.DataFrame:
    """Return only the most recent season in a results frame."""
    if df is None or df.empty:
        return pd.DataFrame()
    col = season_col if season_col in df.columns else ("_season" if "_season" in df.columns else None)
    if col is None:
        return df.copy()
    seasons = sorted(df[col].dropna().unique())
    if not seasons:
        return pd.DataFrame()
    return df[df[col] == seasons[-1]].copy()


def weight_search_ranges(
    center: dict[str, float] | None = None,
    *,
    shrink: float = 1.0,
    wide_mult: tuple[float, float] = DEFAULT_WEIGHT_SEARCH_MULT,
) -> dict[str, tuple[float, float]]:
    """Per-variable (lo, hi) bounds; shrink toward center each tuning year."""
    from pipeline.config import CONFIDENCE_WEIGHT_ABS_BOUNDS

    center = clamp_confidence_weights(center or DEFAULT_CONFIDENCE_WEIGHTS)
    shrink = float(np.clip(shrink, 0.05, 1.0))
    lo_mult, hi_mult = wide_mult
    abs_bounds = CONFIDENCE_WEIGHT_ABS_BOUNDS or {}
    ranges: dict[str, tuple[float, float]] = {}
    for key, default in DEFAULT_CONFIDENCE_WEIGHTS.items():
        c = float(center.get(key, default))
        if shrink >= 0.999:
            wlo, whi = c * lo_mult, c * hi_mult
        else:
            wlo, whi = c * lo_mult, c * hi_mult
            wlo = c - (c - wlo) * shrink
            whi = c + (whi - c) * shrink
        if key.endswith("_center") or key == "breakeven_cover":
            pad = max(abs(c) * 0.15 * shrink, 0.02)
            ranges[key] = (c - pad, c + pad)
        elif key == "interval_cap":
            ranges[key] = (max(18.0, c - 6.0 * shrink), min(36.0, c + 6.0 * shrink))
        else:
            ranges[key] = (min(wlo, whi), max(wlo, whi))
        if key in abs_bounds:
            alo, ahi = abs_bounds[key]
            lo, hi = ranges[key]
            ranges[key] = (max(float(alo), float(lo)), min(float(ahi), float(hi)))
    edge_floor = 0.5 * float(DEFAULT_CONFIDENCE_WEIGHTS["edge_slope"])
    lo, hi = ranges["edge_slope"]
    ranges["edge_slope"] = (max(lo, edge_floor), hi)
    return ranges


def sample_weight_candidates(
    ranges: dict[str, tuple[float, float]],
    n_samples: int,
    rng: np.random.Generator | None = None,
    *,
    include_defaults: bool = False,
) -> list[dict[str, float]]:
    rng = rng or np.random.default_rng(42)
    keys = list(DEFAULT_CONFIDENCE_WEIGHTS.keys())
    out: list[dict[str, float]] = []
    if include_defaults:
        out.append(dict(DEFAULT_CONFIDENCE_WEIGHTS))
    for _ in range(max(1, n_samples)):
        w = dict(DEFAULT_CONFIDENCE_WEIGHTS)
        for k in keys:
            lo, hi = ranges.get(k, (DEFAULT_CONFIDENCE_WEIGHTS[k], DEFAULT_CONFIDENCE_WEIGHTS[k]))
            w[k] = float(rng.uniform(lo, hi))
        edge_floor = 0.5 * float(DEFAULT_CONFIDENCE_WEIGHTS["edge_slope"])
        w["edge_slope"] = max(w["edge_slope"], edge_floor)
        out.append(clamp_confidence_weights(w))
    return out


def shrink_weight_center(
    center: dict[str, float],
    best: dict[str, float],
    alpha: float = 0.65,
) -> dict[str, float]:
    """Blend prior center toward newly tuned weights (year-over-year refinement)."""
    merged = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    merged.update(center)
    for k in DEFAULT_CONFIDENCE_WEIGHTS:
        merged[k] = float((1.0 - alpha) * merged[k] + alpha * best.get(k, merged[k]))
    return clamp_confidence_weights(merged)


def bucket_lift_map(rates: dict[str, float], breakeven: float = 110.0 / 210.0) -> dict[str, float]:
    if not rates:
        return dict(DEFAULT_BUCKET_ATS_LIFT)
    return {lbl: rates.get(lbl, breakeven) - breakeven for lbl in EDGE_LABELS}


def build_confidence_features(
    *,
    spread_edge_pts: float = 0.0,
    conf_width: float = 24.0,
    rating_uncertainty: float = 350.0,
    elo_meta_agreement: float = 1.0,
    spread_cover_prob: float = 0.5,
    matchup_vol_sigma: float | None = None,
    league_vol_sigma: float = 10.0,
    disagreement_trust: float = 1.0,
    phantom_injury_flag: bool | int = False,
    ats_classifier_prob: float | None = None,
    win_prob: float | None = None,
    elo_margin: float | None = None,
    lean: str = "Home",
    **_,
) -> dict:
    """Normalized feature dict for unified ATS confidence scoring."""
    abs_edge = abs(float(spread_edge_pts or 0))
    cw = float(conf_width or 24)
    unc = float(rating_uncertainty or 350)
    vol_sig = float(matchup_vol_sigma) if matchup_vol_sigma is not None and np.isfinite(matchup_vol_sigma) else float(league_vol_sigma)
    lg = max(float(league_vol_sigma or 10.0), 1e-6)
    vol_ratio = vol_sig / lg
    ats_p = float(ats_classifier_prob) if ats_classifier_prob is not None and np.isfinite(ats_classifier_prob) else float(spread_cover_prob or 0.5)
    em = float(elo_margin or 0.0)
    return {
        "spread_edge_pts": float(spread_edge_pts or 0),
        "edge": abs_edge,
        "abs_edge": abs_edge,
        "conf_width": cw,
        "rating_uncertainty": unc,
        "elo_meta_agreement": float(elo_meta_agreement or 0.0),
        "spread_cover_prob": float(spread_cover_prob or 0.5),
        "matchup_vol_sigma": vol_sig,
        "league_vol_sigma": lg,
        "vol_ratio": vol_ratio,
        "disagreement_trust": float(np.clip(disagreement_trust or 1.0, 0.0, 1.0)),
        "phantom_injury_flag": int(bool(phantom_injury_flag)),
        "ats_classifier_prob": ats_p,
        "win_prob_side": lean_side_win_prob(win_prob if win_prob is not None else 0.5, lean),
        "abs_elo_margin": abs(em),
        "elo_lean_align": elo_lean_align(lean, em),
    }


def build_calibrated_feature_vector(
    features: dict,
    *,
    weights: dict[str, float] | None = None,
    bucket_lifts: dict[str, float] | None = None,
) -> np.ndarray:
    """Monotone confidence-aligned vector for logistic calibration."""
    f = build_confidence_features(**features) if "abs_edge" not in features else features
    w = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    if weights:
        w.update(weights)
    lifts = bucket_lifts or DEFAULT_BUCKET_ATS_LIFT

    composite = ats_confidence_score(f, lifts, w)
    cw = float(f["conf_width"])
    unc = float(f["rating_uncertainty"])
    vol_ratio = float(f.get("vol_ratio", 1.0))
    cap = float(w["interval_cap"])

    return np.array([
        composite / 120.0,
        float(f["abs_edge"]) / 12.0,
        float(f["spread_cover_prob"]),
        float(f["ats_classifier_prob"]),
        float(f["win_prob_side"]),
        float(f["disagreement_trust"]),
        float(f["elo_meta_agreement"]),
        float(f["elo_lean_align"]),
        max(0.0, (cap - min(cw, cap)) / cap),
        max(0.0, 1.0 - max(0.0, (unc - w["unc_center"]) / w["unc_span"])),
        max(0.0, min(2.0, 2.0 - vol_ratio)),
        1.0 - float(f.get("phantom_injury_flag", 0)),
    ], dtype=float)


def confidence_feature_vector(
    features: dict,
    *,
    weights: dict[str, float] | None = None,
    bucket_lifts: dict[str, float] | None = None,
) -> np.ndarray:
    """Fixed-order numeric vector for logistic_features calibration."""
    if weights is not None or bucket_lifts is not None or "composite_norm" in features:
        return build_calibrated_feature_vector(
            features, weights=weights, bucket_lifts=bucket_lifts,
        )
    f = build_confidence_features(**features) if "abs_edge" not in features else features
    return build_calibrated_feature_vector(f, weights=weights, bucket_lifts=bucket_lifts)


def ats_confidence_score(
    features: dict,
    bucket_lifts: dict[str, float] | None = None,
    weights: dict[str, float] | None = None,
) -> float:
    """Unified ATS ranking score (higher → historically better cover rate)."""
    f = build_confidence_features(**features) if "abs_edge" not in features else features
    w = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    if weights:
        w.update(weights)

    abs_edge = float(f["abs_edge"])
    conf_width = float(f["conf_width"])
    unc = float(f["rating_uncertainty"])
    elo_agree = float(f["elo_meta_agreement"])
    cover_p = float(f["spread_cover_prob"])
    vol_ratio = float(f.get("vol_ratio", 1.0))
    trust = float(f.get("disagreement_trust", 1.0))
    phantom = int(f.get("phantom_injury_flag", 0))
    ats_p = float(f.get("ats_classifier_prob", cover_p))
    win_p = float(f.get("win_prob_side", 0.5))
    elo_abs = float(f.get("abs_elo_margin", 0.0))
    elo_align = float(f.get("elo_lean_align", 0.5))

    lifts = bucket_lifts or DEFAULT_BUCKET_ATS_LIFT
    bucket = _edge_bucket(abs_edge)
    bucket_bonus = lifts.get(bucket, 0.0) * w["bucket_lift_scale"]
    edge_term = abs_edge * w["edge_slope"] + bucket_bonus

    cap = w["interval_cap"]
    interval_bonus = max(0.0, (cap - min(conf_width, cap)) / cap) * w["interval_scale"]
    unc_penalty = max(0.0, (unc - w["unc_center"]) / w["unc_span"]) * w["unc_penalty_scale"]
    agree_bonus = w["agree_bonus"] if elo_agree >= 0.5 else w["agree_penalty"]
    cover_term = (cover_p - w["breakeven_cover"]) * w["cover_scale"]

    vol_penalty = max(0.0, (vol_ratio - 1.0)) * w["vol_penalty_scale"]
    trust_bonus = (trust - w["trust_center"]) * w["trust_scale"]
    phantom_penalty = w["phantom_penalty"] if phantom else 0.0
    ats_bonus = (ats_p - w["breakeven_cover"]) * w["ats_scale"]
    win_bonus = (win_p - w["breakeven_cover"]) * w["win_prob_scale"]
    elo_bonus = elo_abs * w["elo_margin_scale"] + (elo_align - 0.5) * 2.0 * w["elo_align_bonus"]

    return float(
        edge_term + interval_bonus - unc_penalty + agree_bonus + cover_term
        - vol_penalty + trust_bonus - phantom_penalty + ats_bonus + win_bonus + elo_bonus
    )


def _heuristic_calibrated_prob(raw_score: float, raw_cover: float) -> float:
    return float(np.clip(0.35 * raw_cover + 0.65 * (0.524 + raw_score / 120.0), 0.01, 0.99))


class WalkForwardBetCalibrator:
    """P(cover | confidence) per bet type; default weights in Phase 2, tuned in Phase 2a."""

    def __init__(self, method: str | None = None, confidence_weights: dict[str, float] | None = None,
                 logistic_c: float = 0.1):
        raw = method or CONFIDENCE_CALIB_METHOD
        # "auto" is resolved in fit_walkforward_confidence_calibrator; default fit path uses logistic_features.
        self.method = "logistic_features" if str(raw).lower() == "auto" else raw
        self.logistic_c = float(logistic_c)
        self.confidence_weights = clamp_confidence_weights(
            confidence_weights or DEFAULT_CONFIDENCE_WEIGHTS
        )
        self.models: dict[str, object | None] = {"ats": None, "ml": None, "ou": None}
        self.scalers: dict[str, StandardScaler | None] = {"ats": None, "ml": None, "ou": None}
        self.bucket_lifts: dict[str, float] = dict(DEFAULT_BUCKET_ATS_LIFT)
        self._fitted = False
        self._va_ats_scores: np.ndarray | None = None
        self._va_ats_outcomes: np.ndarray | None = None
        self._calib_method_ats: str = self.method
        self._fit_season: str | None = None

    @staticmethod
    def _ats_outcome(row) -> float | None:
        # Prefer T-60 decision spread for calibration labels (not closing/legacy market).
        spread = row.get("DECISION_SPREAD", row.get("decision_spread"))
        if spread is None or (isinstance(spread, float) and pd.isna(spread)):
            spread = row.get("MARKET_SPREAD", row.get("market_spread"))
        if spread is None or pd.isna(spread):
            return None
        direction = row.get("DIRECTION", row.get("EDGE_LEAN", "Pass"))
        edge = float(row.get("EDGE", row.get("spread_edge_pts", 0)) or 0)
        if direction in ("Pass", "nan", "", None) or (isinstance(direction, float) and pd.isna(direction)):
            if abs(edge) < 1e-9:
                return None
            direction = "Home" if edge > 0 else "Away"
        cover = float(row["ACTUAL_MARGIN"]) + float(spread)
        if cover == 0:
            return None
        home = direction == "Home"
        return float((home and cover > 0) or (not home and cover < 0))

    @staticmethod
    def _ml_outcome(row) -> float | None:
        if row.get("ML_DIRECTION") == "Pass" or pd.isna(row.get("MARKET_ML")):
            return None
        home_win = row["ACTUAL_HOME"] > row["ACTUAL_AWAY"]
        if row["ML_DIRECTION"] == "Home":
            return float(home_win)
        return float(not home_win)

    @staticmethod
    def _ou_outcome(row) -> float | None:
        if row.get("OU_DIRECTION") == "Pass" or pd.isna(row.get("MARKET_TOTAL")):
            return None
        actual = row["ACTUAL_HOME"] + row["ACTUAL_AWAY"]
        if row["OU_DIRECTION"] == "Over":
            return float(actual > row["MARKET_TOTAL"])
        return float(actual < row["MARKET_TOTAL"])

    def _row_features(self, row, bet_type: str) -> dict:
        abs_edge = abs(float(row.get("EDGE", row.get("spread_edge_pts", 0)) or 0))
        vol_sig = row.get("MATCHUP_VOL_SIGMA", row.get("matchup_vol_sigma"))
        direction = row.get("DIRECTION", row.get("EDGE_LEAN", "Pass"))
        if direction in ("Pass", "nan", "", None) or (isinstance(direction, float) and pd.isna(direction)):
            direction = "Home" if float(row.get("EDGE", 0) or 0) > 0 else "Away"
        elo_m = row.get("elo_margin_calibrated", row.get("ELO_MARGIN_CALIBRATED", 0))
        return build_confidence_features(
            spread_edge_pts=float(row.get("EDGE", row.get("spread_edge_pts", 0)) or 0),
            conf_width=float(row.get("CONF_WIDTH", 24) or 24),
            rating_uncertainty=float(row.get("RATING_UNCERTAINTY", 350) or 350),
            elo_meta_agreement=float(row.get("ELO_META_AGREEMENT", 1.0) or 0.0),
            spread_cover_prob=float(
                row.get("SPREAD_COVER_PROB", row.get("SPREAD_COVER_PROB_RAW", row.get("COVER_PROB_RAW", 0.5))) or 0.5
            ),
            matchup_vol_sigma=float(vol_sig) if vol_sig is not None and pd.notna(vol_sig) else None,
            disagreement_trust=float(row.get("DISAGREEMENT_TRUST", 1.0) or 1.0),
            phantom_injury_flag=int(row.get("PHANTOM_INJURY_FLAG", 0) or 0),
            ats_classifier_prob=float(row.get("ATS_CLASSIFIER_PROB", np.nan))
            if pd.notna(row.get("ATS_CLASSIFIER_PROB", np.nan)) else None,
            win_prob=float(row.get("WIN_PROB", row.get("win_prob", 0.5)) or 0.5),
            elo_margin=float(elo_m) if elo_m is not None and pd.notna(elo_m) else 0.0,
            lean=str(direction),
        )

    def _build_score(self, row, bet_type: str) -> float:
        feat = self._row_features(row, bet_type)
        if bet_type == "ats":
            return ats_confidence_score(feat, self.bucket_lifts, self.confidence_weights)
        if bet_type == "ml":
            unc = feat["rating_uncertainty"]
            return (
                abs(feat.get("ml_ev", 0)) * 200.0
                + abs(feat.get("win_prob", 0.5) - 0.5) * 50.0
                + (700 - min(unc, 700)) / 20.0
            )
        return float(feat.get("total_edge", 0)) * 1.5

    def _enrich_ats_row(self, row) -> dict:
        from pipeline.market import elo_meta_agreement, spread_cover_prob

        r = row.to_dict() if hasattr(row, "to_dict") else dict(row)
        pred = r.get("PRED_SPREAD", r.get("RAW_PRED_MARGIN", 0))
        mkt = r.get("MARKET_SPREAD", np.nan)
        cw = float(r.get("CONF_WIDTH", 24) or 24)
        direction = r.get("DIRECTION", r.get("EDGE_LEAN", "Pass"))
        if direction in ("Pass", "nan", "", None) or (isinstance(direction, float) and pd.isna(direction)):
            direction = "Home" if float(r.get("EDGE", 0) or 0) > 0 else "Away"
        if "ELO_META_AGREEMENT" not in r or pd.isna(r.get("ELO_META_AGREEMENT")):
            r["ELO_META_AGREEMENT"] = elo_meta_agreement(
                pred, mkt, r.get("elo_margin_calibrated", r.get("ELO_MARGIN_CALIBRATED")),
            )
        if "SPREAD_COVER_PROB" not in r or pd.isna(r.get("SPREAD_COVER_PROB")):
            r["SPREAD_COVER_PROB"] = spread_cover_prob(
                pred, mkt, cw, direction=direction,
                matchup_vol_sigma=r.get("MATCHUP_VOL_SIGMA", r.get("matchup_vol_sigma")),
            )
        return r

    def _fit_calibrator(self, bet_type: str, scores: np.ndarray, outcomes: np.ndarray,
                        feature_matrix: np.ndarray | None, method: str, logistic_c: float | None = None):
        if len(scores) < 30 or len(set(outcomes.tolist())) < 2:
            self.models[bet_type] = None
            self.scalers[bet_type] = None
            return

        y = np.asarray(outcomes, dtype=int)
        c = float(self.logistic_c if logistic_c is None else logistic_c)
        if method in ("logistic_features", "logistic_isotonic") and feature_matrix is not None:
            scaler = StandardScaler()
            X = scaler.fit_transform(feature_matrix)
            clf = LogisticRegression(C=c, solver="lbfgs", max_iter=300)
            clf.fit(X, y)
            if method == "logistic_isotonic":
                train_p = clf.predict_proba(X)[:, 1]
                iso = IsotonicRegression(out_of_bounds="clip")
                iso.fit(train_p, y)
                self.models[bet_type] = {"logistic": clf, "isotonic": iso}
            else:
                self.models[bet_type] = clf
            self.scalers[bet_type] = scaler
        elif method == "platt":
            clf = LogisticRegression(C=0.1, solver="lbfgs", max_iter=200)
            clf.fit(np.asarray(scores, dtype=float).reshape(-1, 1), y)
            self.models[bet_type] = clf
            self.scalers[bet_type] = None
        elif method == "isotonic":
            iso = IsotonicRegression(out_of_bounds="clip")
            iso.fit(np.asarray(scores, dtype=float), y)
            self.models[bet_type] = iso
            self.scalers[bet_type] = None
        else:
            self.models[bet_type] = None
            self.scalers[bet_type] = None

    def _calibration_frame(self, prior_df: pd.DataFrame, scope: str | None = None) -> pd.DataFrame:
        from pipeline.config import CONFIDENCE_CALIB_SCOPE

        use_scope = scope or CONFIDENCE_CALIB_SCOPE
        if use_scope == "prior_year":
            calib = prior_year_frame(prior_df)
            return calib if not calib.empty else prior_df
        return prior_df

    def fit(
        self,
        prior_df: pd.DataFrame,
        method: str | None = None,
        scope: str | None = None,
        confidence_weights: dict[str, float] | None = None,
    ):
        if prior_df is None or prior_df.empty:
            self._fitted = False
            return self

        if confidence_weights:
            self.confidence_weights = clamp_confidence_weights({
                **self.confidence_weights,
                **confidence_weights,
            })

        calib_df = self._calibration_frame(prior_df, scope=scope)
        if calib_df.empty:
            self._fitted = False
            return self

        season_col = "simulated_season_window" if "simulated_season_window" in calib_df.columns else "_season"
        if season_col in calib_df.columns and calib_df[season_col].notna().any():
            self._fit_season = str(calib_df[season_col].dropna().iloc[-1])

        calib_method = method or self.method
        self._calib_method_ats = calib_method
        rates = empirical_bucket_ats_rates(calib_df)
        self.bucket_lifts = bucket_lift_map(rates)

        for bet_type, outcome_fn in [
            ("ats", self._ats_outcome),
            ("ml", self._ml_outcome),
            ("ou", self._ou_outcome),
        ]:
            scores, outcomes, feat_rows = [], [], []
            for _, row in calib_df.iterrows():
                y = outcome_fn(row)
                if y is None:
                    continue
                row_dict = self._enrich_ats_row(row) if bet_type == "ats" else row
                feat = self._row_features(row_dict, bet_type)
                scores.append(self._build_score(row_dict, bet_type))
                outcomes.append(y)
                if bet_type == "ats":
                    feat_rows.append(
                        build_calibrated_feature_vector(
                            feat,
                            weights=self.confidence_weights,
                            bucket_lifts=self.bucket_lifts,
                        )
                    )

            scores_arr = np.asarray(scores, dtype=float)
            outcomes_arr = np.asarray(outcomes, dtype=int)
            feat_mat = np.vstack(feat_rows) if feat_rows else None
            use_method = calib_method if bet_type == "ats" else "isotonic"
            self._fit_calibrator(bet_type, scores_arr, outcomes_arr, feat_mat, use_method)

            if bet_type == "ats":
                self._va_ats_scores = scores_arr
                self._va_ats_outcomes = outcomes_arr

        self._fitted = True
        return self

    def _predict_prob(self, bet_type: str, raw_score: float, features: dict) -> float:
        model = self.models.get(bet_type)
        method = self._calib_method_ats if bet_type == "ats" else "isotonic"
        raw_cover = float(features.get("spread_cover_prob", 0.5) or 0.5)

        if model is None or method == "none":
            return _heuristic_calibrated_prob(raw_score, raw_cover)

        if method in ("logistic_features", "logistic_isotonic") and bet_type == "ats":
            scaler = self.scalers.get(bet_type)
            if scaler is None:
                return _heuristic_calibrated_prob(raw_score, raw_cover)
            vec = build_calibrated_feature_vector(
                features,
                weights=self.confidence_weights,
                bucket_lifts=self.bucket_lifts,
            ).reshape(1, -1)
            x = scaler.transform(vec)
            if isinstance(model, dict):
                p = float(model["logistic"].predict_proba(x)[0, 1])
                return float(model["isotonic"].predict([p])[0])
            return float(model.predict_proba(x)[0, 1])

        if method in ("platt",) or isinstance(model, LogisticRegression):
            return float(model.predict_proba(np.array([[raw_score]], dtype=float))[0, 1])

        if isinstance(model, IsotonicRegression):
            return float(model.predict([raw_score])[0])

        return _heuristic_calibrated_prob(raw_score, raw_cover)

    def _predict_rank_prob(self, bet_type: str, raw_score: float, features: dict) -> float:
        """Continuous ranking probability — logistic / heuristic, never isotonic clumps."""
        feat = features if "abs_edge" in features else features
        raw_cover = float(
            feat.get("spread_cover_prob", feat.get("SPREAD_COVER_PROB", 0.5)) or 0.5
        )
        method = getattr(self, "_calib_method_ats", self.method) or self.method
        model = self.models.get(bet_type)

        if method in ("logistic_features", "logistic_isotonic") and bet_type == "ats":
            scaler = self.scalers.get(bet_type)
            if scaler is not None and model is not None:
                vec = build_calibrated_feature_vector(
                    feat,
                    weights=self.confidence_weights,
                    bucket_lifts=self.bucket_lifts,
                ).reshape(1, -1)
                x = scaler.transform(vec)
                logit_model = model["logistic"] if isinstance(model, dict) else model
                if hasattr(logit_model, "predict_proba"):
                    return float(np.clip(logit_model.predict_proba(x)[0, 1], 0.01, 0.99))

        if method in ("platt",) or isinstance(model, LogisticRegression):
            if model is not None and hasattr(model, "predict_proba"):
                return float(np.clip(
                    model.predict_proba(np.array([[raw_score]], dtype=float))[0, 1],
                    0.01, 0.99,
                ))

        return _heuristic_calibrated_prob(raw_score, raw_cover)

    def predict(self, bet_type: str, features: dict, direction: str = "Home") -> dict:
        if bet_type == "ats":
            feat = build_confidence_features(**features) if "abs_edge" not in features else features
            score = ats_confidence_score(feat, self.bucket_lifts, self.confidence_weights)
        else:
            score = self._build_score(features, bet_type)
            feat = features

        # Stake/ECE path: season-picked calibrator (may be isotonic) + cap.
        prob = self._predict_prob(bet_type, score, features if bet_type != "ats" else feat)
        # Ranking path: continuous logistic/heuristic → CONFIDENCE score.
        rank_p = self._predict_rank_prob(bet_type, score, feat if bet_type == "ats" else features)

        if bet_type == "ats":
            from pipeline.config import (
                CONFIDENCE_PROB_TEMPERATURE,
                COVER_PROB_CAP_HI,
                COVER_PROB_CAP_LO,
            )
            temp = float(CONFIDENCE_PROB_TEMPERATURE or 1.0)
            if temp > 1.0 and 0.0 < prob < 1.0:
                logit = np.log(prob / (1.0 - prob))
                prob = float(1.0 / (1.0 + np.exp(-logit / temp)))
            lo = float(COVER_PROB_CAP_LO) if COVER_PROB_CAP_LO is not None else 0.01
            hi = float(COVER_PROB_CAP_HI) if COVER_PROB_CAP_HI is not None else 0.99
            try:
                from pipeline.market import dampen_cover_for_upset
                prob = dampen_cover_for_upset(
                    prob,
                    direction=direction,
                    market_spread=feat.get("market_spread", feat.get("MARKET_SPREAD")),
                    upset_prob=feat.get("upset_prob"),
                )
            except Exception:
                pass
            prob = float(np.clip(prob, lo, hi))
            if temp > 1.0 and 0.0 < rank_p < 1.0:
                rlogit = np.log(rank_p / (1.0 - rank_p))
                rank_p = float(1.0 / (1.0 + np.exp(-rlogit / temp)))
            rank_p = float(np.clip(rank_p, 0.01, 0.99))
        else:
            prob = float(np.clip(prob, 0.01, 0.99))
            rank_p = float(np.clip(rank_p, 0.01, 0.99))

        # Continuous 0–100 ranking score (not isotonic-quantized cover%).
        conf_score = float(rank_p * 100.0)
        try:
            from pipeline.config import CONFIDENCE_SCORE_SOFT_MAX
            soft_max = CONFIDENCE_SCORE_SOFT_MAX
            if soft_max is not None and conf_score > float(soft_max):
                over = conf_score - float(soft_max)
                conf_score = float(soft_max) + 0.35 * over
                conf_score = min(99.0, conf_score)
        except Exception:
            pass
        conf_score_int = int(round(conf_score))
        tier = _tier_from_prob(prob)
        return {
            "calibrated_prob": prob,
            "confidence_score": conf_score_int,
            "confidence_score_continuous": conf_score,
            "confidence_tier": tier,
            "stars": _stars_from_tier(tier, direction),
            "confidence_score_raw": score,
            "rank_prob": rank_p,
        }

    def venn_abers_width(self, score: float) -> float | None:
        if self._va_ats_scores is None or self._va_ats_outcomes is None:
            return None
        if len(self._va_ats_scores) < 30:
            return None
        from pipeline.venn_abers import scores_to_interval
        _, _, _, width = scores_to_interval(
            self._va_ats_scores, self._va_ats_outcomes, np.array([float(score)]),
        )
        return float(width[0])

    def save(self, path: Path | str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: Path | str):
        with open(path, "rb") as f:
            return pickle.load(f)


def _ats_roi_from_outcomes(outcomes: np.ndarray) -> float:
    if len(outcomes) == 0:
        return -1e9
    wp = float(np.mean(outcomes))
    return float(wp * (100.0 / 110.0) - (1.0 - wp))


def _confidence_training_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Gate-eligible rows for confidence calibration (|edge| + avoid bands)."""
    if df is None or df.empty:
        return pd.DataFrame()
    from pipeline.bet_selection import passes_edge_avoid_band
    from pipeline.config import MIN_EDGE_BUCKET

    m = df.copy()
    if "EDGE" not in m.columns:
        return m
    edge = pd.to_numeric(m["EDGE"], errors="coerce").fillna(0)
    abs_edge = edge.abs()
    floor = float(max(float(CONFIDENCE_MIN_EDGE), float(MIN_EDGE_BUCKET)))
    m = m[abs_edge >= floor].copy()
    if m.empty:
        return m
    agree = m.get("ELO_META_AGREEMENT")
    keep = []
    for i, row in m.iterrows():
        ae = float(abs(float(row.get("EDGE", 0) or 0)))
        ag = None
        if agree is not None:
            try:
                ag = float(row.get("ELO_META_AGREEMENT"))
            except (TypeError, ValueError):
                ag = None
        keep.append(passes_edge_avoid_band(ae, elo_meta_agreement=ag))
    return m.loc[np.asarray(keep, dtype=bool)].copy()


def build_calib_split_ats_frame(
    calib_features: pd.DataFrame,
    *,
    pred_margin_col: str = "pred_margin",
    season_label: str = "calib-split",
) -> pd.DataFrame:
    """Build ATS-shaped rows from the training calibration split (season-1 calibrator)."""
    if calib_features is None or calib_features.empty:
        return pd.DataFrame()
    df = calib_features.copy()
    mkt_col = "market_spread" if "market_spread" in df.columns else "closing_spread"
    if mkt_col not in df.columns or pred_margin_col not in df.columns:
        return pd.DataFrame()
    if "actual_margin" not in df.columns:
        if "actual_home" in df.columns and "actual_away" in df.columns:
            df["actual_margin"] = df["actual_home"] - df["actual_away"]
        else:
            return pd.DataFrame()
    mkt = pd.to_numeric(df[mkt_col], errors="coerce")
    pm = pd.to_numeric(df[pred_margin_col], errors="coerce")
    am = pd.to_numeric(df["actual_margin"], errors="coerce")
    valid = mkt.notna() & pm.notna() & am.notna()
    if not valid.any():
        return pd.DataFrame()
    out = df.loc[valid].copy()
    out["MARKET_SPREAD"] = mkt.loc[out.index].values
    out["PRED_SPREAD"] = pm.loc[out.index].values
    out["ACTUAL_MARGIN"] = am.loc[out.index].values
    out["EDGE"] = out["PRED_SPREAD"] + out["MARKET_SPREAD"]
    out["DIRECTION"] = np.where(
        out["EDGE"] > 0, "Home", np.where(out["EDGE"] < 0, "Away", "Pass"),
    )
    out["EDGE_LEAN"] = out["DIRECTION"]
    out["CONF_WIDTH"] = 24.0
    if "elo_margin_calibrated" in out.columns:
        out["ELO_MARGIN_CALIBRATED"] = out["elo_margin_calibrated"]
    out["simulated_season_window"] = season_label
    return out


def _scored_ats_bets(cal: WalkForwardBetCalibrator, df: pd.DataFrame) -> list[tuple[float, int, float]]:
    """Return (outcome, confidence_score 0-100, calibrated_prob) per scored row."""
    scored: list[tuple[float, int, float]] = []
    train_df = _confidence_training_frame(df)
    for _, row in train_df.iterrows():
        y = WalkForwardBetCalibrator._ats_outcome(row)
        if y is None:
            continue
        row_dict = cal._enrich_ats_row(row)
        raw = cal._build_score(row_dict, "ats")
        feat = cal._row_features(row_dict, "ats")
        prob = float(cal._predict_prob("ats", raw, feat))
        conf_score = int(round(prob * 100))
        scored.append((float(y), conf_score, prob))
    return scored


def _pick_confidence_band_gated_roi(
    scored: list[tuple[float, int, float]],
    *,
    min_thresholds=(55, 56, 57, 58, 59, 60),
    max_thresholds: tuple[int | None, ...] = (None, 61, 62, 63, 64, 65),
    min_bets: int = 35,
) -> tuple[float, float, float | None, int]:
    """Pick (roi, min_thr, max_thr, n_bets) maximizing flat ATS ROI in a score band."""
    best_roi, best_min, best_max, best_n = -1e9, 58, None, 0
    for lo in min_thresholds:
        for hi in max_thresholds:
            if hi is not None and hi < lo:
                continue
            sub = [
                y for y, cs, _p in scored
                if cs >= lo and (hi is None or cs <= hi)
            ]
            if len(sub) < min_bets:
                continue
            roi = _ats_roi_from_outcomes(np.asarray(sub, dtype=float))
            if roi > best_roi:
                best_roi, best_min, best_max, best_n = roi, float(lo), hi, len(sub)
    return best_roi, best_min, best_max, best_n


def _pick_threshold_gated_roi(
    scored: list[tuple[float, int, float]],
    thresholds=range(50, 71),
    min_bets: int = 35,
) -> tuple[float, float, int, int]:
    """Maximize gated ROI; tie-break by lower ECE."""
    from pipeline.calibration_metrics import compute_ece

    best_roi, best_ece, best_thr, best_n = -1e9, 1e9, 58, 0
    for thr in thresholds:
        sub = [(y, p) for y, cs, p in scored if cs >= thr]
        if len(sub) < min_bets:
            continue
        y_arr = np.asarray([x[0] for x in sub], dtype=float)
        p_arr = np.asarray([x[1] for x in sub], dtype=float)
        roi = _ats_roi_from_outcomes(y_arr)
        ece = float(compute_ece(y_arr, p_arr))
        if roi > best_roi + 1e-9 or (abs(roi - best_roi) <= 1e-9 and ece < best_ece):
            best_roi, best_ece, best_thr, best_n = roi, ece, int(thr), len(sub)
    return best_roi, best_ece, best_thr, best_n


def weights_differ_from_default(weights: dict[str, float], rtol: float = 0.02) -> bool:
    for k, v in DEFAULT_CONFIDENCE_WEIGHTS.items():
        w = float(weights.get(k, v))
        if abs(w - v) > max(abs(v) * rtol, 1e-6):
            return True
    return False


def _rank_spearman(outcomes, scores) -> float:
    """Spearman rank correlation between confidence score and ATS outcome."""
    y = np.asarray(outcomes, dtype=float)
    s = np.asarray(scores, dtype=float)
    mask = np.isfinite(y) & np.isfinite(s)
    if mask.sum() < 12:
        return 0.0
    y, s = y[mask], s[mask]
    if np.std(s) < 1e-9:
        return 0.0
    r = np.corrcoef(np.argsort(np.argsort(s)), np.argsort(np.argsort(y)))[0, 1]
    return float(r) if np.isfinite(r) else 0.0


def logistic_feature_importance(cal: WalkForwardBetCalibrator) -> dict[str, float]:
    """Signed logistic coefficients mapped to human-readable feature names."""
    model = cal.models.get("ats")
    if model is None:
        return {}
    clf = model["logistic"] if isinstance(model, dict) else model
    if not hasattr(clf, "coef_"):
        return {}
    coefs = np.asarray(clf.coef_).ravel()
    names = list(CONFIDENCE_FEATURE_NAMES)
    if len(coefs) != len(names):
        return {}
    return {n: float(c) for n, c in zip(names, coefs)}


def prepare_walkforward_calibrator(
    cal: WalkForwardBetCalibrator,
    prior_df: pd.DataFrame,
    *,
    season_index: int = 1,
    method: str | None = None,
) -> WalkForwardBetCalibrator:
    """Deprecated wrapper — use fit_walkforward_confidence_calibrator."""
    fitted, _meta = fit_walkforward_confidence_calibrator(
        prior_df,
        season_index=season_index,
        weight_center=cal.confidence_weights,
        method=method,
        existing_calibrator=cal,
    )
    return fitted


def fit_walkforward_confidence_calibrator(
    prior_df: pd.DataFrame,
    *,
    train_season_label: str = "",
    test_season_label: str = "",
    season_index: int = 1,
    weight_center: dict[str, float] | None = None,
    method: str | None = None,
    calib_scope: str | None = None,
    show_progress: bool = False,
    existing_calibrator: WalkForwardBetCalibrator | None = None,
) -> tuple[WalkForwardBetCalibrator, dict]:
    """Phase 2a inline: tune + fit confidence calibrator on prior season only (leak-free).

    When ``method`` / ``CONFIDENCE_CALIB_METHOD`` is ``auto``, pick the candidate
    method with lowest prior-year ECE (tie-break: Brier), preferring
    ``logistic_features`` when ECE is within 0.002 of the best.
    """
    from pipeline.calibration_metrics import compute_brier, compute_ece
    from pipeline.config import (
        CONFIDENCE_CALIB_METHOD,
        CONFIDENCE_CALIB_METHOD_CANDIDATES,
        CONFIDENCE_CALIB_SCOPE,
        CONFIDENCE_WEIGHT_SHRINK,
        MAX_CONFIDENCE_SCORE,
        MIN_CONFIDENCE_SCORE,
        TUNE_CONFIDENCE_WEIGHTS_IN_BACKTEST,
    )
    from pipeline.metrics import walkforward_confidence_gate

    if prior_df is None or prior_df.empty:
        cal = existing_calibrator or WalkForwardBetCalibrator(method=method)
        cal._fitted = False
        return cal, {}

    train_df = _confidence_training_frame(prior_df)
    if train_df.empty:
        train_df = prior_df.copy()

    center = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    if weight_center:
        center.update(weight_center)
    requested = (method or CONFIDENCE_CALIB_METHOD or "isotonic").lower()
    scope = calib_scope or CONFIDENCE_CALIB_SCOPE

    def _eval_method(m: str) -> tuple[float, float, WalkForwardBetCalibrator, dict]:
        """Return (ece, brier, fitted_cal, composite_weights)."""
        local_center = dict(center)
        local_cal = WalkForwardBetCalibrator(method=m, confidence_weights=local_center)
        composite = dict(local_center)
        if TUNE_CONFIDENCE_WEIGHTS_IN_BACKTEST:
            shrink = 1.0 if season_index <= 1 else CONFIDENCE_WEIGHT_SHRINK
            ranges = weight_search_ranges(local_center, shrink=shrink)
            composite, local_cal, _roi, _thr, _ece, _n = tune_confidence_weights(
                train_df,
                center=local_center,
                ranges=ranges,
                method=m,
                objective="hybrid",
                show_progress=False,
                progress_desc=None,
            )
            if m in ("logistic_features", "logistic_isotonic"):
                best_c, best_obj, best_local = local_cal.logistic_c, -1e9, local_cal
                for c in (0.02, 0.05, 0.1, 0.25, 0.5, 1.0):
                    trial = WalkForwardBetCalibrator(
                        method=m, confidence_weights=composite, logistic_c=c,
                    )
                    trial.fit(train_df, method=m, scope=scope, confidence_weights=composite)
                    scored = _scored_ats_bets(trial, train_df)
                    if len(scored) < 35:
                        continue
                    y = np.asarray([s[0] for s in scored], dtype=float)
                    cs = [s[1] for s in scored]
                    p = np.asarray([s[2] for s in scored], dtype=float)
                    ece = float(compute_ece(y, p))
                    sp = _rank_spearman(y, cs)
                    band_roi, _bmin, _bmax, band_n = _pick_confidence_band_gated_roi(
                        scored, min_bets=35,
                    )
                    obj = sp + 0.5 * band_roi - ece
                    if obj > best_obj:
                        best_obj, best_c, best_local = obj, c, trial
                local_cal = best_local
                local_cal.logistic_c = best_c
        else:
            local_cal.fit(train_df, method=m, scope=scope, confidence_weights=local_center)

        if not local_cal._fitted:
            local_cal.fit(train_df, method=m, scope=scope, confidence_weights=composite)

        # Negative edge coef → fall back to isotonic for this candidate.
        use_m = m
        if m in ("logistic_features", "logistic_isotonic"):
            feat_w = logistic_feature_importance(local_cal)
            if float(feat_w.get("abs_edge_norm", 0.0)) < 0:
                use_m = "isotonic"
                local_cal = WalkForwardBetCalibrator(method="isotonic", confidence_weights=composite)
                local_cal.fit(train_df, method="isotonic", scope=scope, confidence_weights=composite)

        scored = _scored_ats_bets(local_cal, train_df)
        if len(scored) < 20:
            return 1e9, 1e9, local_cal, composite
        y = np.asarray([s[0] for s in scored], dtype=float)
        p = np.asarray([s[2] for s in scored], dtype=float)
        return float(compute_ece(y, p)), float(compute_brier(y, p)), local_cal, composite

    method_scores: list[dict] = []
    if requested == "auto":
        candidates = list(CONFIDENCE_CALIB_METHOD_CANDIDATES) or ["logistic_features", "isotonic", "platt"]
        best_key = None
        best_cal = None
        best_composite = dict(center)
        best_method = "isotonic"
        for m in candidates:
            ece, brier, cal, composite = _eval_method(m)
            method_scores.append({"method": m, "ece": ece, "brier": brier})
            # Prefer logistic_features when within 0.002 ECE of the best.
            key = (ece, 0 if m == "logistic_features" else 1, brier)
            if best_key is None or key < best_key:
                best_key = key
                best_cal = cal
                best_composite = composite
                best_method = cal._calib_method_ats if hasattr(cal, "_calib_method_ats") else m
        calib_method = best_method
        best_cal = best_cal or WalkForwardBetCalibrator(method="isotonic", confidence_weights=center)
        composite_weights = best_composite
    else:
        calib_method = requested
        ece, brier, best_cal, composite_weights = _eval_method(calib_method)
        method_scores.append({"method": calib_method, "ece": ece, "brier": brier})

    if not best_cal._fitted:
        best_cal.fit(train_df, method=calib_method, scope=scope, confidence_weights=composite_weights)

    feat_w = logistic_feature_importance(best_cal)
    calib_method = getattr(best_cal, "_calib_method_ats", calib_method) or calib_method

    scored = _scored_ats_bets(best_cal, train_df)
    y_all = np.asarray([s[0] for s in scored], dtype=float) if scored else np.array([])
    p_all = np.asarray([s[2] for s in scored], dtype=float) if scored else np.array([])
    cs_all = [s[1] for s in scored]

    from pipeline.config import COVER_CALIB_REQUIRE_ECE_IMPROVEMENT, COVER_CALIB_REQUIRE_LOGLOSS_IMPROVEMENT
    from pipeline.calibration_metrics import compute_log_loss

    raw_ece = np.nan
    raw_ll = np.nan
    ece_gate_pass = True
    if COVER_CALIB_REQUIRE_ECE_IMPROVEMENT and len(y_all) >= 20:
        # Raw cover probs from the training frame (pre-calibrator).
        raw_p = []
        for _, row in train_df.iterrows():
            rp = row.get(
                "SPREAD_COVER_PROB_RAW",
                row.get("COVER_PROB_RAW", row.get("SPREAD_COVER_PROB", np.nan)),
            )
            if rp is None or (isinstance(rp, float) and not np.isfinite(rp)):
                continue
            raw_p.append(float(rp))
        if len(raw_p) == len(y_all):
            raw_ece = float(compute_ece(y_all, np.asarray(raw_p, dtype=float)))
            cal_ece = float(compute_ece(y_all, p_all)) if len(p_all) else np.nan
            if np.isfinite(raw_ece) and np.isfinite(cal_ece) and cal_ece > raw_ece + 1e-9:
                ece_gate_pass = False
                best_cal._fitted = False
                print(
                    f"  ⏭ Cover calib ECE gate: cal={cal_ece:.4f} > raw={raw_ece:.4f} — using raw"
                )
            if (
                COVER_CALIB_REQUIRE_LOGLOSS_IMPROVEMENT
                and best_cal._fitted
                and len(p_all)
            ):
                raw_ll = float(compute_log_loss(y_all, np.asarray(raw_p, dtype=float)))
                cal_ll = float(compute_log_loss(y_all, p_all))
                if np.isfinite(raw_ll) and np.isfinite(cal_ll) and cal_ll > raw_ll + 1e-9:
                    ece_gate_pass = False
                    best_cal._fitted = False
                    print(
                        f"  ⏭ Cover calib log-loss gate: cal={cal_ll:.4f} > raw={raw_ll:.4f} — using raw"
                    )

    conf_thr, conf_max = walkforward_confidence_gate(
        train_df,
        default_min=MIN_CONFIDENCE_SCORE,
        default_max=MAX_CONFIDENCE_SCORE,
        bet_calibrator=best_cal if best_cal._fitted else None,
    )
    gated = [
        y for y, cs, _p in scored
        if cs >= conf_thr and (conf_max is None or cs <= conf_max)
    ]
    train_roi = _ats_roi_from_outcomes(np.asarray(gated, dtype=float)) if gated else np.nan
    train_ece = float(compute_ece(y_all, p_all)) if len(y_all) else np.nan
    train_sp = _rank_spearman(y_all, cs_all) if len(y_all) else np.nan

    meta = {
        "train_season": train_season_label,
        "test_season": test_season_label,
        "calib_method": calib_method,
        "calib_method_requested": requested,
        "method_scores": method_scores,
        "logistic_c": float(best_cal.logistic_c),
        "train_roi": float(train_roi) if np.isfinite(train_roi) else np.nan,
        "train_ece": train_ece,
        "raw_cover_ece": float(raw_ece) if np.isfinite(raw_ece) else np.nan,
        "ece_gate_pass": bool(ece_gate_pass),
        "train_spearman": train_sp,
        "train_min_confidence": float(conf_thr),
        "train_max_confidence": float(conf_max) if conf_max is not None else None,
        "train_gated_n_bets": len(gated),
        "n_train_bets": len(scored),
        "feature_weights": feat_w,
        "suggested_weights": dict(composite_weights),
        "weights_changed": (
            weights_differ_from_default(composite_weights)
            or any(abs(v) > 1e-6 for v in feat_w.values())
        ),
        "inline_phase2a": True,
    }
    for k, v in composite_weights.items():
        meta[f"w_{k}"] = round(float(v), 4)
    for k, v in feat_w.items():
        meta[f"coef_{k}"] = round(float(v), 4)

    return best_cal, meta


def save_phase2a_walkforward_state(
    season_rows: list[dict],
    *,
    latest_center: dict[str, float] | None = None,
) -> Path:
    """Persist inline Phase 2a tuning from walk-forward backtest."""
    from pipeline.config import CONFIDENCE_CALIB_METHOD, MIN_CONFIDENCE_SCORE

    path = confidence_weights_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    chosen_method = CONFIDENCE_CALIB_METHOD
    if season_rows and season_rows[-1].get("calib_method"):
        chosen_method = season_rows[-1]["calib_method"]
    payload: dict = {
        "inline_phase2a": True,
        "calib_method": chosen_method,
        "unified_weight_tuning": season_rows,
    }
    if season_rows:
        latest = season_rows[-1]
        payload["suggested_weights_latest"] = latest.get("suggested_weights", dict(DEFAULT_CONFIDENCE_WEIGHTS))
        payload["suggested_min_confidence"] = int(latest.get("train_min_confidence", MIN_CONFIDENCE_SCORE))
        if latest.get("train_max_confidence") is not None:
            payload["suggested_max_confidence"] = int(latest["train_max_confidence"])
        payload["suggested_logistic_c"] = latest.get("logistic_c")
        payload["feature_weights_latest"] = latest.get("feature_weights", {})
        payload["weights_changed_latest"] = bool(latest.get("weights_changed", False))
        payload["method_scores_latest"] = latest.get("method_scores", [])
    if latest_center:
        payload["weight_center_latest"] = {k: float(v) for k, v in latest_center.items()}
    path.write_text(json.dumps(payload, indent=2, default=str))
    return path


def tune_confidence_weights(
    train_df: pd.DataFrame,
    *,
    center: dict[str, float] | None = None,
    ranges: dict[str, tuple[float, float]] | None = None,
    method: str | None = None,
    n_samples: int | None = None,
    min_bets: int = 35,
    thresholds=range(50, 71),
    show_progress: bool = True,
    progress_desc: str | None = None,
    objective: str = "roi",
) -> tuple[dict[str, float], WalkForwardBetCalibrator, float, int, float, int]:
    """Search weight floats on prior season.

    objective: roi | hybrid (rank correlation + low ECE + ROI at best threshold).
    """
    from pipeline.calibration_metrics import compute_ece

    w_center = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    if center:
        w_center.update(center)
    search_ranges = ranges or weight_search_ranges(w_center, shrink=1.0)
    n = n_samples or CONFIDENCE_WEIGHT_SEARCH_SAMPLES
    calib_method = method or CONFIDENCE_CALIB_METHOD

    best_weights = dict(w_center)
    best_roi, best_ece, best_thr, best_n = -1e9, 1e9, 58, 0
    best_score = -1e9
    best_cal = WalkForwardBetCalibrator(method=calib_method, confidence_weights=best_weights)

    candidates = list(sample_weight_candidates(search_ranges, n, include_defaults=True))
    cand_bar = tqdm(
        candidates,
        desc=progress_desc or "2a · weight candidates",
        disable=not show_progress,
        leave=False,
    )
    for weights in cand_bar:
        cal = WalkForwardBetCalibrator(method=calib_method, confidence_weights=weights)
        cal.fit(train_df, method=calib_method, scope="all_prior", confidence_weights=weights)
        scored = _scored_ats_bets(cal, train_df)
        if len(scored) < min_bets:
            continue
        roi, ece, thr, n_bets = _pick_threshold_gated_roi(scored, thresholds=thresholds, min_bets=min_bets)
        if objective == "hybrid":
            y = np.asarray([s[0] for s in scored], dtype=float)
            cs = [s[1] for s in scored]
            sp = _rank_spearman(y, cs)
            band_roi, _bmin, _bmax, band_n = _pick_confidence_band_gated_roi(
                scored, min_bets=min_bets,
            )
            combo = sp - 0.5 * ece + 0.5 * band_roi
            if combo > best_score:
                best_score = combo
                best_roi = band_roi
                best_ece = ece
                best_thr = int(_bmin) if band_n >= min_bets else thr
                best_n = band_n if band_n >= min_bets else n_bets
                best_weights = clamp_confidence_weights(weights)
                best_cal = cal
        elif roi > best_roi + 1e-9 or (abs(roi - best_roi) <= 1e-9 and ece < best_ece):
            best_roi, best_ece, best_thr, best_n = roi, ece, thr, n_bets
            best_weights = clamp_confidence_weights(weights)
            best_cal = cal

    best_weights = clamp_confidence_weights(best_weights)
    best_cal = WalkForwardBetCalibrator(method=calib_method, confidence_weights=best_weights)
    best_cal.fit(train_df, method=calib_method, scope="all_prior", confidence_weights=best_weights)
    return best_weights, best_cal, best_roi, best_thr, best_ece, best_n


def yearly_confidence_weight_tuning(
    results_df: pd.DataFrame,
    *,
    season_col: str = "simulated_season_window",
    shrink: float | None = None,
    method: str | None = None,
    show_progress: bool = True,
) -> pd.DataFrame:
    """Phase 2a post-hoc replay — uses same fit_walkforward_confidence_calibrator as inline backtest."""
    from pipeline.diagnostics import _norm_results

    df = _norm_results(results_df)
    if season_col not in df.columns:
        return pd.DataFrame()

    if "PHASE2A_INLINE" in df.columns and df["PHASE2A_INLINE"].fillna(0).astype(int).max() > 0:
        path = confidence_weights_path()
        if path.exists():
            try:
                payload = json.loads(path.read_text())
                if payload.get("inline_phase2a"):
                    print("⚡ Phase 2a ran inline during walk-forward — loading saved tuning (no re-fit).")
                rows = payload.get("unified_weight_tuning", [])
                if rows:
                    return pd.DataFrame(rows)
            except (OSError, json.JSONDecodeError):
                pass

    shrink = CONFIDENCE_WEIGHT_SHRINK if shrink is None else shrink
    seasons = sorted(df[season_col].dropna().unique())
    center = dict(DEFAULT_CONFIDENCE_WEIGHTS)
    rows = []

    season_pairs = [(seasons[i - 1], test_season) for i, test_season in enumerate(seasons) if i > 0]
    season_bar = tqdm(
        season_pairs,
        desc="2a · tune seasons",
        disable=not show_progress,
        leave=False,
    )
    for i, (train_season, test_season) in enumerate(season_bar, start=1):
        season_bar.set_postfix(train=str(train_season), test=str(test_season))
        train_df = df[df[season_col] == train_season]
        if train_df.empty:
            continue

        cal, meta = fit_walkforward_confidence_calibrator(
            train_df,
            train_season_label=str(train_season),
            test_season_label=str(test_season),
            season_index=i,
            weight_center=center,
            method=method,
            show_progress=show_progress,
        )
        composite_weights = meta.get("suggested_weights", center)

        test_df = df[df[season_col] == test_season]
        thr = int(meta.get("train_min_confidence", MIN_CONFIDENCE_SCORE))
        tmax = meta.get("train_max_confidence")
        outcomes, probs = [], []
        for _, row in test_df.iterrows():
            y = WalkForwardBetCalibrator._ats_outcome(row)
            if y is None:
                continue
            row_dict = cal._enrich_ats_row(row)
            raw = cal._build_score(row_dict, "ats")
            feat = cal._row_features(row_dict, "ats")
            prob = cal._predict_prob("ats", raw, feat)
            conf_score = int(round(prob * 100))
            if conf_score < thr or (tmax is not None and conf_score > tmax):
                continue
            probs.append(prob)
            outcomes.append(y)
        test_roi = _ats_roi_from_outcomes(np.asarray(outcomes, dtype=float)) if outcomes else np.nan
        from pipeline.calibration_metrics import compute_ece
        test_ece = float(compute_ece(np.asarray(outcomes), np.asarray(probs))) if outcomes else np.nan

        row = dict(meta)
        row.update({
            "test_roi": test_roi,
            "test_ece": test_ece,
            "test_gated_n_bets": len(outcomes),
        })
        search_ranges = weight_search_ranges(center, shrink=1.0 if i == 1 else shrink)
        for k, v in composite_weights.items():
            row[f"w_{k}"] = round(float(v), 4)
            lo, hi = search_ranges.get(k, (v, v))
            row[f"range_{k}"] = f"[{lo:.3g}, {hi:.3g}]"
        rows.append(row)
        center = shrink_weight_center(center, composite_weights)

    return pd.DataFrame(rows)


def confidence_weights_path(name: str = "confidence_weights_by_season.json") -> Path:
    return STATE_DIR / name


def load_latest_confidence_weights() -> dict[str, float]:
    """Load suggested composite weights from inline Phase 2a / tuning JSON."""
    path = confidence_weights_path()
    if not path.exists():
        return dict(DEFAULT_CONFIDENCE_WEIGHTS)
    try:
        with open(path) as f:
            payload = json.load(f)
        if payload.get("inline_phase2a") and payload.get("suggested_weights_latest"):
            out = dict(DEFAULT_CONFIDENCE_WEIGHTS)
            out.update({k: float(v) for k, v in payload["suggested_weights_latest"].items()})
            return out
        suggested = payload.get("suggested_weights_latest")
        if suggested:
            out = dict(DEFAULT_CONFIDENCE_WEIGHTS)
            out.update({k: float(v) for k, v in suggested.items()})
            return out
        tuning = payload.get("unified_weight_tuning")
        if tuning:
            latest = tuning[-1]
            out = dict(DEFAULT_CONFIDENCE_WEIGHTS)
            for k, v in latest.items():
                if str(k).startswith("w_"):
                    out[str(k).replace("w_", "")] = float(v)
            return out
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return dict(DEFAULT_CONFIDENCE_WEIGHTS)


def load_suggested_min_confidence(default: float | None = None) -> float | None:
    """Phase 2a suggested minimum confidence gate (None if not saved)."""
    from pipeline.config import MIN_CONFIDENCE_SCORE

    path = confidence_weights_path()
    if not path.exists():
        return None
    try:
        with open(path) as f:
            payload = json.load(f)
        if "suggested_min_confidence" in payload:
            return float(payload["suggested_min_confidence"])
    except (OSError, json.JSONDecodeError, TypeError, ValueError):
        pass
    return None


def default_calibrator_path(name: str = "bet_calibrator.pkl") -> Path:
    return STATE_DIR / name
