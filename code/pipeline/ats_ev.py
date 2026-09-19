"""ATS exact-price EV utilities (T-60 Phase 4).

Uses the canonical home_cover probability and the actual quoted price on the
chosen side — never a fixed -110 assumption for decisioning.
"""
from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from pipeline.market import american_to_decimal


def _to_american(price, *, odds_format: str | None = None) -> float:
    """Convert price to American odds.

    ``odds_format`` may be ``\"american\"`` or ``\"decimal\"``. When omitted,
    values with abs >= 100 are treated as American; values in (1.0, 50.0) as
    decimal. Ambiguous values outside those bands return NaN.
    """
    if price is None or (isinstance(price, float) and not np.isfinite(price)):
        return np.nan
    p = float(price)
    fmt = (odds_format or "").lower() or None
    if fmt == "american":
        return p
    if fmt == "decimal":
        if p <= 1.0:
            return np.nan
        if p >= 2.0:
            return (p - 1.0) * 100.0
        return -100.0 / (p - 1.0)
    if abs(p) >= 100:
        return p
    if 1.0 < p < 50.0:
        if p >= 2.0:
            return (p - 1.0) * 100.0
        return -100.0 / (p - 1.0)
    return np.nan


def break_even_prob(american_price, *, odds_format: str | None = None) -> float:
    am = _to_american(american_price, odds_format=odds_format)
    if not np.isfinite(am):
        return float("nan")
    dec = american_to_decimal(am)
    return 1.0 / dec if dec > 0 else np.nan


def ats_side_price(side: str, spread_home_price, spread_away_price) -> float:
    if side == "Home":
        return _to_american(spread_home_price)
    if side == "Away":
        return _to_american(spread_away_price)
    return np.nan


def cover_prob_for_side(home_cover_prob: float, side: str) -> float:
    p = float(home_cover_prob)
    if side == "Home":
        return p
    if side == "Away":
        return 1.0 - p
    return np.nan


def ats_ev(home_cover_prob: float, side: str, american_price) -> float:
    """Expected value of a 1-unit stake on ``side`` at ``american_price``."""
    p = cover_prob_for_side(home_cover_prob, side)
    am = _to_american(american_price)
    if not np.isfinite(p) or not np.isfinite(am):
        return np.nan
    dec = american_to_decimal(am)
    return p * dec - 1.0


def select_ats_bet_ev(
    home_cover_prob: float,
    *,
    decision_spread,
    spread_home_price=None,
    spread_away_price=None,
    min_ev: float = 0.0,
    min_cover_prob: float | None = None,
) -> dict[str, Any]:
    """Choose Home/Away/Pass by exact-price EV after one cover-prob prediction.

    ``home_cover_prob`` must be P(home covers vs decision line). Side conversion
    happens exactly once here.
    """
    out = {
        "side": "Pass",
        "fair_cover_prob": float(home_cover_prob) if home_cover_prob is not None else np.nan,
        "break_even_prob": np.nan,
        "expected_value": np.nan,
        "edge": np.nan,
        "price": np.nan,
        "decision_spread": decision_spread,
        "pass_reason": None,
    }
    if decision_spread is None or not np.isfinite(float(decision_spread)):
        out["pass_reason"] = "missing_decision_spread"
        return out
    if home_cover_prob is None or not np.isfinite(float(home_cover_prob)):
        out["pass_reason"] = "missing_cover_prob"
        return out

    candidates = []
    for side in ("Home", "Away"):
        price = ats_side_price(side, spread_home_price, spread_away_price)
        if not np.isfinite(price):
            # Missing price → assume -110 for research, but flag it.
            price = -110.0
            price_assumed = True
            be = break_even_prob(price)  # defined for -110
        else:
            price_assumed = False
            be = break_even_prob(price)
            if not np.isfinite(be):
                continue
        p = cover_prob_for_side(home_cover_prob, side)
        ev = ats_ev(home_cover_prob, side, price)
        if min_cover_prob is not None and p < float(min_cover_prob):
            continue
        if np.isfinite(ev) and ev > float(min_ev):
            candidates.append((ev, side, price, p, be, price_assumed))

    if not candidates:
        out["pass_reason"] = "no_side_clears_ev"
        # Still report best side for diagnostics.
        home_ev = ats_ev(home_cover_prob, "Home", spread_home_price if pd.notna(spread_home_price) else -110)
        away_ev = ats_ev(home_cover_prob, "Away", spread_away_price if pd.notna(spread_away_price) else -110)
        if np.nanmax([home_ev, away_ev]) == home_ev:
            out["fair_cover_prob"] = cover_prob_for_side(home_cover_prob, "Home")
            out["expected_value"] = home_ev
        else:
            out["fair_cover_prob"] = cover_prob_for_side(home_cover_prob, "Away")
            out["expected_value"] = away_ev
        return out

    candidates.sort(key=lambda x: x[0], reverse=True)
    ev, side, price, p, be, assumed = candidates[0]
    out.update({
        "side": side,
        "fair_cover_prob": p,
        "break_even_prob": be,
        "expected_value": ev,
        "edge": float(p - be) if np.isfinite(be) else np.nan,
        "price": price,
        "pass_reason": "price_assumed_minus_110" if assumed else None,
    })
    return out


def home_cover_prob_from_margin(
    pred_margin: float,
    decision_spread,
    *,
    sigma: float = 12.0,
) -> float:
    """P(home covers) from fair margin vs T-60 decision line (no side flip).

    Home covers when ``actual_margin + decision_spread > 0``. Under a normal
    margin model centered at ``pred_margin``, that is
    ``P(M > -decision_spread) = 1 - Φ((-decision - pred) / σ)``.
    """
    if decision_spread is None or not np.isfinite(float(decision_spread)):
        return float("nan")
    if pred_margin is None or not np.isfinite(float(pred_margin)):
        return float("nan")
    sig = max(float(sigma), 1e-3)
    # Equivalent: Φ((pred_margin + decision_spread) / σ)
    z = (float(pred_margin) + float(decision_spread)) / sig
    try:
        from scipy.stats import norm
        return float(np.clip(norm.cdf(z), 0.01, 0.99))
    except ImportError:
        # Abramowitz–Stegun fallback
        t = 1.0 / (1.0 + 0.2316419 * abs(z))
        d = 0.3989423 * np.exp(-0.5 * z * z)
        p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
        cdf = 1.0 - p if z > 0 else p
        return float(np.clip(cdf, 0.01, 0.99))


def canonical_home_cover_prob(
    *,
    ats_classifier=None,
    feat: Mapping[str, Any] | None = None,
    decision_spread=None,
    pred_margin=None,
    sigma: float = 12.0,
    direction_specific_cover: float | None = None,
    lean: str | None = None,
) -> float:
    """Always return P(home covers), never P(chosen-side covers).

    Prefer the ATS classifier called with direction=\"Home\". Fall back to a
    margin-distribution estimate. Never treat a direction-flipped cover
    probability as home_cover without un-flipping it.
    """
    feat = feat or {}
    dec = decision_spread if decision_spread is not None else feat.get(
        "decision_spread", feat.get("market_spread")
    )
    if (
        ats_classifier is not None
        and getattr(ats_classifier, "fitted", False)
        and dec is not None
        and np.isfinite(float(dec))
    ):
        ats_feat = {**feat, "pred_margin": pred_margin if pred_margin is not None else feat.get("pred_margin")}
        return float(ats_classifier.predict_cover_prob(ats_feat, "Home", float(dec)))
    if pred_margin is not None and dec is not None:
        return home_cover_prob_from_margin(float(pred_margin), dec, sigma=sigma)
    # Last resort: un-flip a direction-specific cover if lean is known.
    if direction_specific_cover is not None and lean in ("Home", "Away"):
        p = float(direction_specific_cover)
        return p if lean == "Home" else float(1.0 - p)
    return float("nan")


def ats_decision_record(
    home_cover_prob: float,
    feat: Mapping[str, Any],
    *,
    min_ev: float = 0.0,
    quote_age_minutes: float | None = None,
    data_quality_reasons: list[str] | None = None,
    lineup_scenarios: int | None = None,
    interval: tuple[float, float] | None = None,
) -> dict[str, Any]:
    """Full ATS output contract for logging / paper trading.

    ``home_cover_prob`` MUST be P(home covers), not P(chosen lean covers).
    """
    decision = select_ats_bet_ev(
        home_cover_prob,
        decision_spread=feat.get("decision_spread", feat.get("market_spread")),
        spread_home_price=feat.get("spread_home_price"),
        spread_away_price=feat.get("spread_away_price"),
        min_ev=min_ev,
    )
    decision["quote_age_minutes"] = quote_age_minutes
    decision["data_quality_reasons"] = list(data_quality_reasons or [])
    decision["lineup_scenarios"] = lineup_scenarios
    decision["interval"] = interval
    if feat.get("market_total_missing"):
        decision["data_quality_reasons"].append("market_total_missing")
    if float(feat.get("h_missing_rotation", 0) or 0) + float(feat.get("a_missing_rotation", 0) or 0) > 0.35:
        decision["data_quality_reasons"].append("high_missing_rotation")
    return decision
