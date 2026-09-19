"""Direct actual-score training contract (home/away points as canonical labels).

Canonical score-training labels
-------------------------------
- ``actual_home`` / ``actual_away`` are the only supervised score targets.
- Margin, total, winner, and cover outcomes are *derived* after prediction:
  ``margin = home - away``, ``total = home + away``.

Market boundary
---------------
Vegas spreads/totals/moneylines may be used only *after* score prediction for
pricing, edge, grading, benchmarking, and CLV. They must not appear in the
score-pair feature matrix or as score-training labels. Closing lines are
evaluation-only.

Forecast source
---------------
When ``USE_CANONICAL_SCORE_PAIR`` is True, ``MetaScorePairModel`` is the sole
source of truth for home/away points in production paths. Margin-residual and
market-total heads remain research ablations only.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Explicit market / line columns banned from score-pair training features.
SCORE_MARKET_BAN_COLS: tuple[str, ...] = (
    # Direct market lines
    "market_spread",
    "decision_spread",
    "closing_spread",
    "market_total",
    "market_total_minus_league",
    "market_total_missing",
    "market_ml",
    "market_ml_away",
    # Line movement / public / fair-implied
    "spread_move",
    "public_home_pct",
    "public_away_pct",
    "market_fair_win_prob",
    "reverse_line_movement",
    "steam_flag",
    "market_spread_raw",
    # Market-relative residuals / edges that reconstruct the line
    "elo_vs_market",
    "elo_edge_pts",
    # Closing / CLV leakage
    "closing_total",
    "point_clv",
    "clv",
)

# Shared-forecast residuals that encode market offsets (OK for betting heads,
# banned for score training).
SCORE_SF_MARKET_BAN_COLS: tuple[str, ...] = (
    "sf_decision_residual",
    "sf_total_residual",
)

# Canonical label columns for direct score training.
SCORE_LABEL_COLS: tuple[str, ...] = ("actual_home", "actual_away")

# Derived (never train score heads on these as y).
DERIVED_SCORE_TARGETS: tuple[str, ...] = (
    "actual_margin",
    "actual_total",
    "home_win",
    "ats_cover_home",
)

# Feature families that must be tip-available and past-only for score training.
SCORE_FEATURE_PROVENANCE: dict[str, str] = {
    "ratings": "past games only (Elo/Hier trackers updated post-game)",
    "form": "rolling windows ending before tip",
    "pace": "hierarchical pace posterior from past possessions",
    "efficiency": "rolling offense/defense rates from past games",
    "rotations": "projected minutes / availability known pre-tip",
    "injuries": "inactive lists / star-out flags at decision time",
    "rest_travel": "schedule density and travel computed from past dates",
    "matchup": "offense-vs-defense projections from past ratings",
}


def assert_score_feature_provenance(cols: Iterable[str]) -> dict[str, list[str]]:
    """Classify score features into provenance buckets; raise on market leakage."""
    assert_score_safe_features(cols, context="feature_provenance")
    buckets = {k: [] for k in SCORE_FEATURE_PROVENANCE}
    buckets["other"] = []
    for c in cols:
        assigned = False
        name = c.lower()
        if any(x in name for x in ("elo", "hier", "hapm", "team_elo", "lineup5")):
            buckets["ratings"].append(c); assigned = True
        if any(x in name for x in ("roll", "recent", "form", "streak")):
            buckets["form"].append(c); assigned = True
        if "pace" in name or name in ("exp_poss",):
            buckets["pace"].append(c); assigned = True
        if any(x in name for x in ("rtg", "efg", "tov", "orb", "ftr", "xppp", "pps")):
            buckets["efficiency"].append(c); assigned = True
        if any(x in name for x in ("rotation", "minutes", "lineup_composite", "chem")):
            buckets["rotations"].append(c); assigned = True
        if any(x in name for x in ("star_out", "inactive", "injury", "availability")):
            buckets["injuries"].append(c); assigned = True
        if any(x in name for x in ("rest", "b2b", "travel", "fatigue", "tz_", "3in4", "4in6", "games_last")):
            buckets["rest_travel"].append(c); assigned = True
        if "matchup" in name:
            buckets["matchup"].append(c); assigned = True
        if not assigned:
            buckets["other"].append(c)
    return buckets


def score_market_ban_set() -> set[str]:
    return set(SCORE_MARKET_BAN_COLS) | set(SCORE_SF_MARKET_BAN_COLS)


def assert_score_safe_features(cols: Iterable[str], *, context: str = "score_pair") -> list[str]:
    """Raise if any market/line column slips into the score feature matrix."""
    ban = score_market_ban_set()
    bad = sorted({c for c in cols if c in ban})
    if bad:
        raise ValueError(
            f"{context}: market/line features forbidden in score training: {bad}"
        )
    return list(cols)


def filter_score_safe_features(
    cols: Sequence[str],
    *,
    available: Sequence[str] | None = None,
    enforce: bool = True,
) -> list[str]:
    """Drop banned market columns; optionally intersect with available columns."""
    ban = score_market_ban_set()
    out = [c for c in cols if c not in ban]
    if available is not None:
        avail = set(available)
        out = [c for c in out if c in avail]
    if enforce:
        assert_score_safe_features(out)
    return list(dict.fromkeys(out))


def score_safe_feature_cols(df=None) -> list[str]:
    """Build the default score-safe feature list from SAFE_FEATURE_COLS."""
    from pipeline.model import SAFE_FEATURE_COLS, LINE_DERIVED_FEATURE_COLS

    ban = score_market_ban_set() | set(LINE_DERIVED_FEATURE_COLS)
    cols = [c for c in SAFE_FEATURE_COLS if c not in ban]
    # Never train on label columns or derived targets as features.
    label_ban = set(SCORE_LABEL_COLS) | set(DERIVED_SCORE_TARGETS) | {
        "actual_home", "actual_away", "actual_margin", "actual_total",
    }
    cols = [c for c in cols if c not in label_ban]
    if df is not None:
        cols = [c for c in cols if c in df.columns]
    return filter_score_safe_features(cols, enforce=True)


def joint_score_moments(
    sigma_home: float,
    sigma_away: float,
    corr: float,
) -> dict[str, float]:
    """Margin/total variances from bivariate home/away residual model."""
    sh = float(max(sigma_home, 1e-6))
    sa = float(max(sigma_away, 1e-6))
    rho = float(np.clip(corr, -0.99, 0.99))
    var_h, var_a = sh * sh, sa * sa
    cov = rho * sh * sa
    var_margin = var_h + var_a - 2.0 * cov
    var_total = var_h + var_a + 2.0 * cov
    return {
        "sigma_home": sh,
        "sigma_away": sa,
        "corr": rho,
        "cov": float(cov),
        "sigma_margin": float(np.sqrt(max(var_margin, 1e-6))),
        "sigma_total": float(np.sqrt(max(var_total, 1e-6))),
    }


def gaussian_interval(mu: float, sigma: float, coverage: float) -> tuple[float, float]:
    """Symmetric normal predictive interval for nominal coverage in (0, 1)."""
    c = float(np.clip(coverage, 0.01, 0.999))
    # Approximate Φ^{-1}((1+c)/2) without scipy (Acklam / Beasley-Springer style).
    # For common coverages use fixed z-scores.
    z_table = {0.50: 0.67448975, 0.80: 1.28155157, 0.90: 1.64485363, 0.95: 1.95996398}
    key = round(c, 2)
    if key in z_table:
        z = z_table[key]
    else:
        # Inverse erf approximation via numpy iteration on known erfc.
        from math import erfc
        target = (1.0 + c) / 2.0
        lo, hi = 0.0, 6.0
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            # Φ(mid) = 0.5 * erfc(-mid/√2)
            cdf = 0.5 * erfc(-mid / np.sqrt(2.0))
            if cdf < target:
                lo = mid
            else:
                hi = mid
        z = 0.5 * (lo + hi)
    s = float(max(sigma, 1e-6))
    return float(mu - z * s), float(mu + z * s)


def score_predictive_intervals(
    pred_home: float,
    pred_away: float,
    *,
    sigma_home: float,
    sigma_away: float,
    corr: float,
    coverages: Sequence[float] = (0.50, 0.80, 0.95),
) -> dict[str, float]:
    """Home/away/margin/total intervals from joint residual distribution."""
    moments = joint_score_moments(sigma_home, sigma_away, corr)
    ph, pa = float(pred_home), float(pred_away)
    margin, total = ph - pa, ph + pa
    out: dict[str, float] = {
        **moments,
        "pred_home": ph,
        "pred_away": pa,
        "pred_margin": margin,
        "pred_total": total,
    }
    mapping = {
        0.50: "50",
        0.80: "80",
        0.95: "95",
    }
    for cov in coverages:
        tag = mapping.get(round(cov, 2), f"{int(round(cov * 100))}")
        for name, mu, sig in (
            ("home", ph, moments["sigma_home"]),
            ("away", pa, moments["sigma_away"]),
            ("margin", margin, moments["sigma_margin"]),
            ("total", total, moments["sigma_total"]),
        ):
            lo, hi = gaussian_interval(mu, sig, float(cov))
            out[f"{name}_qlo_{tag}"] = lo
            out[f"{name}_qhi_{tag}"] = hi
    # CONF_* aliases match conformal ~80% (alpha=0.10) used by MAX_QUANTILE_WIDTH.
    # Full 50/80/95 bands remain in {home,away,margin,total}_q{lo,hi}_{50,80,95}.
    out["CONF_LOWER"] = out.get("margin_qlo_80", margin - 1.28155 * moments["sigma_margin"])
    out["CONF_UPPER"] = out.get("margin_qhi_80", margin + 1.28155 * moments["sigma_margin"])
    out["CONF_WIDTH"] = float(out["CONF_UPPER"] - out["CONF_LOWER"])
    out["spread_q10"] = out["CONF_LOWER"]
    out["spread_q90"] = out["CONF_UPPER"]
    out["spread_quantile_width"] = out["CONF_WIDTH"]
    out["spread_q25"] = out.get("margin_qlo_50")
    out["spread_q75"] = out.get("margin_qhi_50")
    return out


def apply_structured_to_pair(
    pred_home: float,
    pred_away: float,
    *,
    struct_home: float | None,
    struct_away: float | None,
    home_beta: float,
    away_beta: float | None = None,
) -> tuple[float, float]:
    """Blend structured pace/efficiency prior into paired scores before deriving markets."""
    bh = float(np.clip(home_beta, 0.0, 1.0))
    ba = float(np.clip(away_beta if away_beta is not None else home_beta, 0.0, 1.0))
    ph, pa = float(pred_home), float(pred_away)
    if struct_home is not None and np.isfinite(struct_home):
        ph = (1.0 - bh) * ph + bh * float(struct_home)
    if struct_away is not None and np.isfinite(struct_away):
        pa = (1.0 - ba) * pa + ba * float(struct_away)
    return ph, pa


def assert_pair_algebra(pred_home: float, pred_away: float, pred_margin: float, pred_total: float,
                        *, tol: float = 1e-6) -> None:
    if abs((pred_home - pred_away) - pred_margin) > tol:
        raise AssertionError(
            f"margin algebra failed: {pred_home}-{pred_away} != {pred_margin}"
        )
    if abs((pred_home + pred_away) - pred_total) > tol:
        raise AssertionError(
            f"total algebra failed: {pred_home}+{pred_away} != {pred_total}"
        )


def margin_win_prob(pred_margin: float, sigma_margin: float) -> float:
    """P(home wins) from N(margin, sigma^2)."""
    s = float(max(sigma_margin, 1e-6))
    z = float(pred_margin) / s
    # Φ(z)
    from math import erfc
    return float(np.clip(0.5 * erfc(-z / np.sqrt(2.0)), 0.01, 0.99))


def margin_home_cover_prob(pred_margin: float, decision_spread: float, sigma_margin: float) -> float:
    """P(home covers) = P(margin + decision_spread > 0) under normal margin."""
    s = float(max(sigma_margin, 1e-6))
    # Home covers when actual_margin + decision_spread > 0.
    # Under model: E[margin] = pred_margin → P(margin > -decision_spread)
    z = (float(pred_margin) + float(decision_spread)) / s
    from math import erfc
    return float(np.clip(0.5 * erfc(-z / np.sqrt(2.0)), 0.01, 0.99))


def total_over_prob(pred_total: float, market_total: float, sigma_total: float) -> float:
    s = float(max(sigma_total, 1e-6))
    z = (float(pred_total) - float(market_total)) / s
    from math import erfc
    return float(np.clip(0.5 * erfc(-z / np.sqrt(2.0)), 0.01, 0.99))
