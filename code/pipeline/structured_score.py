"""Structured pace×shot scoring features and hybrid joint-score helpers."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StructuredScoreForecast:
    home_pts: float
    away_pts: float
    total: float
    margin: float
    sigma_total: float
    sigma_margin: float
    corr: float

    def as_feature_dict(self, prefix: str = "struct_") -> dict:
        p = prefix
        return {
            f"{p}home_pts": self.home_pts,
            f"{p}away_pts": self.away_pts,
            f"{p}total": self.total,
            f"{p}margin": self.margin,
            f"{p}sigma_total": self.sigma_total,
            f"{p}sigma_margin": self.sigma_margin,
            f"{p}score_corr": self.corr,
        }


STRUCTURED_SCORE_COLS = [
    "struct_home_pts", "struct_away_pts", "struct_total", "struct_margin",
    "struct_sigma_total", "struct_sigma_margin", "struct_score_corr",
]


def structured_score_from_pace_pps(
    *,
    exp_poss: float,
    home_pps: float,
    away_pps: float,
    pace_var: float = 9.0,
    home_eff_sd: float = 0.08,
    away_eff_sd: float = 0.08,
    shared_env_sd: float = 0.04,
    home_court_pps: float = 0.02,
) -> StructuredScoreForecast:
    """Joint score forecast from shared possessions and points-per-shot proxies.

    Approximates PPP via PPS (points per shot attempt mix) with a 1.0 scale
    when PPS is already in points-per-possession-ish units (~1.1). Shared pace
    shock induces positive score correlation.
    """
    poss = float(max(exp_poss, 1.0))
    h_ppp = float(home_pps) + float(home_court_pps)
    a_ppp = float(away_pps)
    # If inputs look like FG%×value (~1.0–1.3), treat as PPP directly.
    home_pts = h_ppp * poss
    away_pts = a_ppp * poss
    total = home_pts + away_pts
    margin = home_pts - away_pts

    # Delta method variance with shared pace shock.
    # home = (μ_h + ε_shared + ε_h) * (P + δ)
    # Var ≈ ppp^2 * Var(P) + poss^2 * Var(ppp) + 2 cov terms
    var_p = max(float(pace_var), 1.0)
    var_h = poss ** 2 * (home_eff_sd ** 2 + shared_env_sd ** 2) + (h_ppp ** 2) * var_p
    var_a = poss ** 2 * (away_eff_sd ** 2 + shared_env_sd ** 2) + (a_ppp ** 2) * var_p
    cov = poss ** 2 * (shared_env_sd ** 2) + h_ppp * a_ppp * var_p
    var_total = max(var_h + var_a + 2.0 * cov, 1.0)
    var_margin = max(var_h + var_a - 2.0 * cov, 1.0)
    corr = float(cov / max(np.sqrt(var_h * var_a), 1e-6))
    corr = float(np.clip(corr, -0.95, 0.95))
    return StructuredScoreForecast(
        home_pts=float(home_pts),
        away_pts=float(away_pts),
        total=float(total),
        margin=float(margin),
        sigma_total=float(np.sqrt(var_total)),
        sigma_margin=float(np.sqrt(var_margin)),
        corr=corr,
    )


def hybrid_blend(
    direct_total: float,
    direct_margin: float,
    structured: StructuredScoreForecast,
    *,
    total_beta: float = 0.25,
    margin_beta: float = 0.15,
) -> dict:
    """Blend direct head predictions with structured prior."""
    tb = float(np.clip(total_beta, 0.0, 1.0))
    mb = float(np.clip(margin_beta, 0.0, 1.0))
    total = (1.0 - tb) * float(direct_total) + tb * structured.total
    margin = (1.0 - mb) * float(direct_margin) + mb * structured.margin
    return {
        "hybrid_total": total,
        "hybrid_margin": margin,
        "hybrid_home": (total + margin) / 2.0,
        "hybrid_away": (total - margin) / 2.0,
        "hybrid_sigma_total": structured.sigma_total,
        "hybrid_sigma_margin": structured.sigma_margin,
    }
