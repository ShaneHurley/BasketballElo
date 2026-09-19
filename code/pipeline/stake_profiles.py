"""Stake sizing profiles for ATS, ML, and O/U bets."""
from __future__ import annotations

import numpy as np

from pipeline.config import (
    KELLY_FRACTION_CAP,
    SLATE_CORRELATION_MIN_BETS,
    SLATE_CORRELATION_PENALTY,
    STAKE_SIZING_MODE,
    USE_TIER_STAKE_GATES,
    CONFIDENCE_STAKE_MODE,
    CONFIDENCE_SELECTION_MODE,
)
from pipeline.market import spread_kelly_fraction, american_to_decimal
from pipeline.bet_selection import confidence_tier_stake_mult, use_tier_stake_gates, uses_edge_gates

PROFILES = {
    "conservative": {
        "edge_buffer": 1.0,
        "kelly_fraction": 0.10,
        "max_daily_exposure": 0.05,
        "min_confidence_tier": 3,
    },
    "moderate": {
        "edge_buffer": 0.0,
        "kelly_fraction": 0.25,
        "max_daily_exposure": 0.12,
        "min_confidence_tier": 2,
    },
    "aggressive": {
        "edge_buffer": -0.5,
        "kelly_fraction": 0.40,
        "max_daily_exposure": 0.20,
        "min_confidence_tier": 2,
    },
    "edge_moderate": {
        "edge_buffer": 0.0,
        "kelly_fraction": 0.25,
        "max_daily_exposure": 0.12,
        "min_confidence_tier": 1,
    },
}


def profile_edge_threshold(base_threshold: float, profile: str) -> float:
    cfg = PROFILES[profile]
    return max(1.0, base_threshold + cfg["edge_buffer"])


def compute_stake(
    profile: str,
    *,
    direction: str,
    edge_pts: float,
    edge_threshold: float,
    cover_prob: float,
    confidence_tier: int,
    conf_width: float = 24.0,
    rating_uncertainty: float = 350.0,
    juice: float = -110,
    edge_stake_mult: float = 1.0,
    edge_bucket_min: float | None = None,
    volatility_mult: float = 1.0,
    use_tier_gates: bool | None = None,
    confidence_score: int | None = None,
) -> float:
    """Return bankroll fraction to stake (0 if no bet)."""
    if direction == "Pass":
        return 0.0
    cfg = PROFILES[profile]
    if uses_edge_gates():
        thr = profile_edge_threshold(edge_threshold, profile)
        if abs(edge_pts) < thr:
            return 0.0
        if edge_bucket_min is not None and abs(edge_pts) < edge_bucket_min:
            return 0.0
    tier_gates = use_tier_stake_gates() if use_tier_gates is None else use_tier_gates
    if tier_gates and CONFIDENCE_SELECTION_MODE != "min_score" and confidence_tier < cfg["min_confidence_tier"]:
        return 0.0

    kelly = spread_kelly_fraction(cover_prob, juice=juice)
    interval_penalty = min(1.0, 24.0 / max(conf_width, 6.0))
    unc_penalty = max(0.0, 1.0 - (rating_uncertainty - 150) / 400)

    kelly_frac = cfg["kelly_fraction"]
    if STAKE_SIZING_MODE == "edge_scaled":
        kelly_frac = min(kelly_frac, KELLY_FRACTION_CAP)
        norm_width = min(1.0, max(0.0, (float(conf_width) - 12.0) / 24.0))
        interval_penalty *= max(0.0, 1.0 - norm_width)

    stake = kelly * kelly_frac * interval_penalty * unc_penalty
    stake *= float(edge_stake_mult) * float(volatility_mult)
    stake *= confidence_tier_stake_mult(int(confidence_tier or 1))
    if CONFIDENCE_STAKE_MODE and confidence_score is not None:
        stake *= float(np.clip(confidence_score / 100.0, 0.35, 1.0))
    return float(max(0.0, min(stake, cfg["max_daily_exposure"])))


def compute_ml_stake(
    profile: str,
    *,
    ml_direction: str,
    win_prob: float,
    market_ml: float,
    confidence_tier: int,
    max_favorite_decimal: float = 1.45,
    use_tier_gates: bool | None = None,
) -> float:
    if ml_direction == "Pass" or pd_isna(market_ml):
        return 0.0
    cfg = PROFILES[profile]
    tier_gates = use_tier_stake_gates() if use_tier_gates is None else use_tier_gates
    if tier_gates and confidence_tier < cfg["min_confidence_tier"]:
        return 0.0
    dec_home = american_to_decimal(market_ml)
    dec_away = american_to_decimal(-market_ml)
    if ml_direction == "Home":
        dec = dec_home
        p = win_prob
    else:
        dec = dec_away
        p = 1.0 - win_prob
    if dec < max_favorite_decimal:
        return 0.0
    b = dec - 1.0
    if b <= 0:
        return 0.0
    kelly = max(0.0, (p * b - (1 - p)) / b)
    kf = cfg["kelly_fraction"]
    if STAKE_SIZING_MODE == "edge_scaled":
        kf = min(kf, KELLY_FRACTION_CAP)
    return float(min(kelly * kf, cfg["max_daily_exposure"]))


def pd_isna(x):
    try:
        import pandas as pd
        return pd.isna(x)
    except Exception:
        return x is None or (isinstance(x, float) and np.isnan(x))


def apply_daily_caps(
    df,
    stake_col: str,
    date_col: str = "DATE",
    profile: str = "moderate",
    slate_correlation_penalty: float | None = None,
):
    """Cap same-day total exposure for a stake column."""
    if df is None or df.empty or stake_col not in df.columns:
        return df
    out = df.copy()
    cap = PROFILES[profile]["max_daily_exposure"]
    rho = SLATE_CORRELATION_PENALTY if slate_correlation_penalty is None else slate_correlation_penalty
    if date_col not in out.columns:
        return out
    for _, idx in out.groupby(date_col).groups.items():
        stakes = out.loc[idx, stake_col].fillna(0.0)
        total = stakes.sum()
        n_active = int((stakes > 0).sum())
        scale = 1.0
        if n_active > SLATE_CORRELATION_MIN_BETS and rho > 0:
            scale = 1.0 / (1.0 + rho * (n_active - 1))
        if total > cap and total > 0:
            out.loc[idx, stake_col] = stakes * (cap / total) * scale
        elif scale < 1.0:
            out.loc[idx, stake_col] = stakes * scale
    return out


def ats_profit(stake: float, won: bool, juice: float = -110) -> float:
    """Legacy boolean helper. Prefer ``exact_price_profit`` + ``grade_*`` for
    push-aware, exact-price grading (Tasks 037-040)."""
    if stake <= 0:
        return 0.0
    from pipeline.bet_grading import exact_price_profit
    return exact_price_profit(stake, "win" if won else "loss", juice)


# ---------------------------------------------------------------------------
# Tasks 044-047: fractional Kelly, uncertainty shrink, slate sizing, bankroll
# ---------------------------------------------------------------------------

FRACTIONAL_KELLY_CANDIDATES = (0.10, 0.25, 0.50)


def kelly_fraction(p: float, decimal_odds: float) -> float:
    """Exact Kelly ``f* = (bp - q) / b`` (Task 044). Non-positive -> 0."""
    if p is None or not np.isfinite(p) or decimal_odds is None or not np.isfinite(decimal_odds):
        return 0.0
    b = float(decimal_odds) - 1.0
    if b <= 0.0:
        return 0.0
    q = 1.0 - float(p)
    f_star = (b * float(p) - q) / b
    return float(max(0.0, f_star))


def robust_fractional_kelly(
    p: float,
    decimal_odds: float,
    *,
    market_fair_p: float | None = None,
    uncertainty: float = 0.0,
    fraction: float = 0.25,
) -> float:
    """Shrink ``p`` toward market fair / lower-bound before fractional Kelly.

    Task 045: greater ``uncertainty`` must never increase the recommended stake.
    Full Kelly (fraction=1.0) is never the production default.
    """
    if fraction <= 0.0:
        return 0.0
    fraction = float(min(max(fraction, 0.0), 0.50))  # never full Kelly in production
    p = float(p) if p is not None and np.isfinite(p) else 0.0
    unc = float(max(0.0, uncertainty or 0.0))
    # Posterior lower bound: subtract uncertainty mass from p.
    p_robust = max(0.0, p - unc)
    if market_fair_p is not None and np.isfinite(market_fair_p):
        # Shrink toward market fair; more uncertainty → more shrink.
        shrink = min(1.0, unc)
        p_robust = (1.0 - shrink) * p_robust + shrink * float(market_fair_p)
    f_star = kelly_fraction(p_robust, decimal_odds)
    return float(max(0.0, f_star * fraction))


def _nearest_psd(cov: np.ndarray) -> np.ndarray:
    """Project a symmetric matrix onto the PSD cone (eigenvalue clip)."""
    cov = 0.5 * (cov + cov.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.clip(eigvals, 0.0, None)
    return eigvecs @ np.diag(eigvals) @ eigvecs.T


def optimize_slate_stakes(
    raw_stakes: np.ndarray,
    cov: np.ndarray | None = None,
    *,
    shrink: float = 0.85,
    per_bet_cap: float = 0.05,
    slate_cap: float = 0.12,
    game_caps: np.ndarray | None = None,
) -> np.ndarray:
    """Jointly size simultaneous bets with a heavily shrunk covariance (Task 046).

    ``shrink`` toward independence (identity scaled by diagonal variances).
    Hard caps: per-bet and slate total; optional per-game caps.
    """
    stakes = np.asarray(raw_stakes, dtype=float).copy()
    n = len(stakes)
    if n == 0:
        return stakes
    stakes = np.clip(stakes, 0.0, per_bet_cap)
    if cov is None:
        cov = np.eye(n)
    else:
        cov = np.asarray(cov, dtype=float)
        if cov.shape != (n, n):
            raise ValueError(f"cov shape {cov.shape} != ({n}, {n})")
        var = np.diag(cov).copy()
        var[var <= 0] = 1.0
        indep = np.diag(var)
        cov = (1.0 - shrink) * cov + shrink * indep
        cov = _nearest_psd(cov)
        # Soft risk penalty: scale down stakes whose correlated exposure is high.
        risk = np.sqrt(np.clip(np.diag(cov @ np.diag(stakes) @ cov), 0.0, None) + 1e-12)
        # Keep identity path (shrink=1) unchanged; only damp correlated books.
        if shrink < 1.0 - 1e-12:
            scale = 1.0 / (1.0 + risk)
            stakes = stakes * scale
            stakes = np.clip(stakes, 0.0, per_bet_cap)
    if game_caps is not None:
        # game_caps is a vector of group ids; cap sum within each group.
        game_caps = np.asarray(game_caps)
        for gid in np.unique(game_caps):
            mask = game_caps == gid
            total = stakes[mask].sum()
            cap = per_bet_cap  # one bet-equivalent per game by default
            if total > cap > 0:
                stakes[mask] *= cap / total
    total = stakes.sum()
    if total > slate_cap > 0:
        stakes *= slate_cap / total
    # Final PSD check for callers that inspect cov separately is in tests.
    return stakes


def simulate_bankroll(
    win_probs: np.ndarray,
    decimal_odds: np.ndarray,
    fractions: np.ndarray,
    *,
    n_paths: int = 2000,
    seed: int = 42,
    miscalibration: float = 0.0,
    correlation: float = 0.0,
) -> dict:
    """Posterior-predictive bankroll Monte Carlo (Task 047).

    Returns growth / loss / drawdown stats. Worse ``miscalibration`` or higher
    ``correlation`` must never produce a *less* conservative recommended stake
    fraction when used by ``recommend_fraction_under_risk``.
    """
    rng = np.random.default_rng(seed)
    p = np.asarray(win_probs, dtype=float)
    dec = np.asarray(decimal_odds, dtype=float)
    f = np.asarray(fractions, dtype=float)
    n = len(p)
    # Apply miscalibration: true win rate = p - miscalibration (harder).
    p_true = np.clip(p - float(miscalibration), 0.01, 0.99)
    # Correlated Bernoulli via Gaussian copula with equicorrelation.
    rho = float(np.clip(correlation, 0.0, 0.95))
    if n == 0:
        return {
            "expected_log_growth": 0.0,
            "prob_loss": 0.0,
            "p05_return": 0.0,
            "p10_return": 0.0,
            "max_drawdown": 0.0,
            "time_under_water": 0.0,
        }
    mean = np.zeros(n)
    cov = (1.0 - rho) * np.eye(n) + rho * np.ones((n, n))
    z = rng.multivariate_normal(mean, cov, size=n_paths)
    u = 0.5 * (1.0 + np.tanh(z / np.sqrt(2.0)))  # cheap normal-cdf approx
    wins = u < p_true[None, :]
    # Wealth path: start at 1.0, apply each bet sequentially within a path.
    wealth = np.ones(n_paths)
    peak = np.ones(n_paths)
    max_dd = np.zeros(n_paths)
    underwater = np.zeros(n_paths)
    log_growth = np.zeros(n_paths)
    for j in range(n):
        payoff = np.where(wins[:, j], 1.0 + f[j] * (dec[j] - 1.0), 1.0 - f[j])
        wealth *= np.maximum(payoff, 1e-12)
        peak = np.maximum(peak, wealth)
        dd = (wealth - peak) / peak
        max_dd = np.minimum(max_dd, dd)
        underwater += (wealth < 1.0).astype(float)
    log_growth = np.log(np.maximum(wealth, 1e-12))
    returns = wealth - 1.0
    return {
        "expected_log_growth": float(np.mean(log_growth)),
        "prob_loss": float(np.mean(returns < 0.0)),
        "p05_return": float(np.quantile(returns, 0.05)),
        "p10_return": float(np.quantile(returns, 0.10)),
        "max_drawdown": float(np.mean(max_dd)),
        "time_under_water": float(np.mean(underwater / max(n, 1))),
    }


def recommend_fraction_under_risk(
    win_probs: np.ndarray,
    decimal_odds: np.ndarray,
    *,
    candidates=FRACTIONAL_KELLY_CANDIDATES,
    max_drawdown_tol: float = -0.25,
    miscalibration: float = 0.0,
    correlation: float = 0.0,
    seed: int = 42,
) -> float:
    """Pick the largest fractional Kelly that still clears drawdown tolerance.

    Pre-registered warning/stop: if even 0.10 violates tolerance, return 0
    (abstain). Worse calibration / higher correlation can only shrink or hold
    the recommended fraction — never increase it.
    """
    best = 0.0
    for frac in sorted(candidates):
        fractions = np.full(len(win_probs), float(frac))
        stats = simulate_bankroll(
            win_probs, decimal_odds, fractions,
            miscalibration=miscalibration, correlation=correlation, seed=seed,
        )
        if stats["p10_return"] >= max_drawdown_tol and stats["expected_log_growth"] > 0:
            best = float(frac)
    return best
