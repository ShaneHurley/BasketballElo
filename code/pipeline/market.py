"""Odds loading and betting helpers."""
from __future__ import annotations

from collections import deque

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pipeline.config import GOOD_BET_EDGE, TEAM_MAP

# ──────────────────────────────────────────────────────────────────────────────
# 1. LEAKAGE‑FREE ODDS RETRIEVAL (ONLY PAST DATES)
# ──────────────────────────────────────────────────────────────────────────────

_ABBR_MAP = {
    'Atlanta': 'ATL', 'Boston': 'BOS', 'Brooklyn': 'BKN', 'Charlotte': 'CHA',
    'Chicago': 'CHI', 'Cleveland': 'CLE', 'Dallas': 'DAL', 'Denver': 'DEN',
    'Detroit': 'DET', 'Golden State': 'GSW', 'Houston': 'HOU', 'Indiana': 'IND',
    'LA Clippers': 'LAC', 'Los Angeles Clippers': 'LAC', 'Clippers': 'LAC',
    'LA Lakers': 'LAL', 'Los Angeles Lakers': 'LAL', 'Lakers': 'LAL',
    'Memphis': 'MEM', 'Miami': 'MIA', 'Milwaukee': 'MIL', 'Minnesota': 'MIN',
    'New Orleans': 'NOP', 'New York': 'NYK', 'Knicks': 'NYK',
    'Oklahoma City': 'OKC', 'Orlando': 'ORL', 'Philadelphia': 'PHI', '76ers': 'PHI',
    'Phoenix': 'PHX', 'Portland': 'POR', 'Sacramento': 'SAC',
    'San Antonio': 'SAS', 'Toronto': 'TOR', 'Utah': 'UTA', 'Washington': 'WAS'
}
_TEAM_ALIASES = {
    'ATL': ['ATL', 'Atlanta', 'Atlanta Hawks'], 'BOS': ['BOS', 'Boston', 'Boston Celtics'],
    'BKN': ['BKN', 'Brooklyn', 'Brooklyn Nets'], 'CHA': ['CHA', 'Charlotte', 'Charlotte Hornets'],
    'CHI': ['CHI', 'Chicago', 'Chicago Bulls'], 'CLE': ['CLE', 'Cleveland', 'Cleveland Cavaliers'],
    'DAL': ['DAL', 'Dallas', 'Dallas Mavericks'], 'DEN': ['DEN', 'Denver', 'Denver Nuggets'],
    'DET': ['DET', 'Detroit', 'Detroit Pistons'], 'GSW': ['GSW', 'Golden State', 'Golden State Warriors'],
    'HOU': ['HOU', 'Houston', 'Houston Rockets'], 'IND': ['IND', 'Indiana', 'Indiana Pacers'],
    'LAC': ['LAC', 'LA Clippers', 'Los Angeles Clippers', 'Clippers'],
    'LAL': ['LAL', 'LA Lakers', 'Los Angeles Lakers', 'Lakers'],
    'MEM': ['MEM', 'Memphis', 'Memphis Grizzlies'], 'MIA': ['MIA', 'Miami', 'Miami Heat'],
    'MIL': ['MIL', 'Milwaukee', 'Milwaukee Bucks'], 'MIN': ['MIN', 'Minnesota', 'Minnesota Timberwolves'],
    'NOP': ['NOP', 'New Orleans', 'New Orleans Pelicans'],
    'NYK': ['NYK', 'New York', 'New York Knicks', 'Knicks'],
    'OKC': ['OKC', 'Oklahoma City', 'Oklahoma City Thunder'], 'ORL': ['ORL', 'Orlando', 'Orlando Magic'],
    'PHI': ['PHI', 'Philadelphia', 'Philadelphia 76ers', '76ers'],
    'PHX': ['PHX', 'Phoenix', 'Phoenix Suns'], 'POR': ['POR', 'Portland', 'Portland Trail Blazers'],
    'SAC': ['SAC', 'Sacramento', 'Sacramento Kings'], 'SAS': ['SAS', 'San Antonio', 'San Antonio Spurs'],
    'TOR': ['TOR', 'Toronto', 'Toronto Raptors'], 'UTA': ['UTA', 'Utah', 'Utah Jazz'],
    'WAS': ['WAS', 'Washington', 'Washington Wizards']
}


def get_game_odds(game_date, home_team, odds_dict):
    """Return spread, ML, total, public%, opening spread, line movement, closing spread.

    Task 024: also passes through the *actual* away-side ML/spread/total
    prices when the source entry provides them (``ml_away``,
    ``spread_away_points``, ``spread_home_price``, ``spread_away_price``,
    ``total_over_price``, ``total_under_price``). These are never derived by
    negating the home price — they are only populated when the underlying
    odds source (e.g. ``load_pinnacle_lines``) recorded the real two-sided
    quote. Absent that, they stay ``NaN`` (missing, not fabricated) rather
    than silently falling back to a negated home price.
    """
    out = {
        "spread": np.nan, "ml": np.nan, "total": np.nan,
        "public_home_pct": np.nan, "spread_open": np.nan, "spread_move": np.nan,
        "closing_spread": np.nan,
        "ml_away": np.nan, "spread_away_points": np.nan,
        "spread_home_price": np.nan, "spread_away_price": np.nan,
        "total_over_price": np.nan, "total_under_price": np.nan,
    }
    if not odds_dict:
        return out
    try:
        game_dt = pd.to_datetime(game_date).date()
    except Exception:
        return out
    abbr = _ABBR_MAP.get(str(home_team).strip(), str(home_team).strip())
    aliases = _TEAM_ALIASES.get(abbr, [home_team])
    for alias in aliases:
        key = (game_dt, alias)
        if key in odds_dict:
            vals = odds_dict[key]
            if isinstance(vals, dict):
                out["spread"] = vals.get("spread", np.nan)
                out["ml"] = vals.get("ml", np.nan)
                out["total"] = vals.get("total", np.nan)
                out["public_home_pct"] = vals.get("public_home_pct", np.nan)
                out["spread_open"] = vals.get("spread_open", vals.get("spread", np.nan))
                out["closing_spread"] = vals.get("closing_spread", vals.get("spread", np.nan))
                so, sc = out["spread_open"], out["spread"]
                if pd.notna(so) and pd.notna(sc):
                    out["spread_move"] = float(sc) - float(so)
                out["ml_away"] = vals.get("ml_away", np.nan)
                out["spread_away_points"] = vals.get("spread_away", np.nan)
                out["spread_home_price"] = vals.get("spread_home_price", np.nan)
                out["spread_away_price"] = vals.get("spread_away_price", np.nan)
                out["total_over_price"] = vals.get("total_over_price", np.nan)
                out["total_under_price"] = vals.get("total_under_price", np.nan)
            else:
                out["spread"], out["ml"] = vals[0], vals[1]
            break
    return out


def get_odds(game_date, home_team, odds_dict):
    """Fetch spread and moneyline for a game (backward-compatible)."""
    g = get_game_odds(game_date, home_team, odds_dict)
    return g["spread"], g["ml"]

# We'll also need a function to get closing spread for CLV tracking.
# We'll implement a separate function to fetch closing line (latest available).
def get_closing_odds(game_date, home_team, odds_dict, away_team=None):
    """Return closing spread and moneyline for *this exact game*.

    Task 025 fix: this used to silently fall back to "the latest odds row on
    or before ``game_date`` for the home team", which could — and did —
    return a *different game's* price whenever the home team's own line for
    this date/opponent was missing (e.g. it would happily return a stale
    price from that team's previous home game). That fallback is removed:
    a match now requires the home team on the exact ``game_date`` AND (when
    supplied) the opponent, matching the canonical-game-identity requirement
    of Task 025/Phase 2A ("never backfill the latest prior game for the same
    team"). No match -> ``(NaN, NaN)``, never a substituted price.
    """
    if not odds_dict:
        return np.nan, np.nan

    try:
        game_dt = pd.to_datetime(game_date).date()
        if pd.isnull(game_dt):
            return np.nan, np.nan
    except Exception:
        return np.nan, np.nan

    home_abbr = _ABBR_MAP.get(str(home_team).strip(), str(home_team).strip())
    home_aliases = set(_TEAM_ALIASES.get(home_abbr, [home_team]))

    away_aliases = None
    if away_team is not None:
        away_abbr = _ABBR_MAP.get(str(away_team).strip(), str(away_team).strip())
        away_aliases = set(_TEAM_ALIASES.get(away_abbr, [away_team]))

    for alias in home_aliases:
        key = (game_dt, alias)
        if key not in odds_dict:
            continue
        vals = odds_dict[key]
        if not isinstance(vals, dict):
            continue
        if away_aliases is not None:
            entry_away = vals.get("away_team") or vals.get("opponent")
            if entry_away is not None and entry_away not in away_aliases:
                continue  # this row is for the right team/date but wrong opponent
        spread = vals.get("closing_spread", vals.get("spread", np.nan))
        ml = vals.get("ml", np.nan)
        return spread, ml

    return np.nan, np.nan

# ──────────────────────────────────────────────────────────────────────────────
# 2. EDGE AND EXPECTED VALUE CALCULATIONS (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def american_to_decimal(odds):
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 1 + odds / 100.0
    else:
        return 1 + 100.0 / abs(odds)

def implied_probability(odds):
    if pd.isna(odds):
        return np.nan
    if odds > 0:
        return 100.0 / (odds + 100)
    else:
        return abs(odds) / (abs(odds) + 100)


def devig_two_way(prob_a: float, prob_b: float) -> tuple[float, float]:
    """Multiplicative de-vig: normalize two implied probabilities to sum to 1."""
    if prob_a is None or prob_b is None or not np.isfinite(prob_a) or not np.isfinite(prob_b):
        return 0.5, 0.5
    total = float(prob_a) + float(prob_b)
    if total <= 1e-9:
        return 0.5, 0.5
    return float(prob_a) / total, float(prob_b) / total


def fair_probs_from_home_ml(market_ml_home) -> tuple[float, float]:
    """Fair home/away win probabilities from a single home ML quote (no vig)."""
    if pd.isna(market_ml_home):
        return 0.5, 0.5
    p_home = implied_probability(market_ml_home)
    p_away = implied_probability(-market_ml_home)
    return devig_two_way(p_home, p_away)


def fair_home_win_prob(market_ml_home) -> float:
    return fair_probs_from_home_ml(market_ml_home)[0]


def spread_edge(model_spread, market_spread):
    if pd.isna(market_spread):
        return np.nan
    return model_spread + market_spread

def moneyline_edge(model_win_prob_home, market_ml_home):
    if pd.isna(market_ml_home):
        return np.nan
    dec = american_to_decimal(market_ml_home)
    ev = (model_win_prob_home * dec) - 1.0
    return ev


UNDERDOG_DECIMAL = 2.0  # +100 American


def ml_prob_for_side(model_win_prob_home, market_ml_home, side: str) -> float:
    """Model win probability for one side, optionally shrunk toward market implied."""
    from pipeline.config import ML_MARKET_DEVIG, ML_MARKET_SHRINK, ML_UNDERDOG_EXTRA_SHRINK

    if side == "Home":
        p = float(model_win_prob_home)
        if ML_MARKET_DEVIG:
            implied = fair_home_win_prob(market_ml_home)
        else:
            implied = implied_probability(market_ml_home)
        dec = american_to_decimal(market_ml_home)
    else:
        p = 1.0 - float(model_win_prob_home)
        if ML_MARKET_DEVIG:
            implied = fair_probs_from_home_ml(market_ml_home)[1]
        else:
            implied = implied_probability(-market_ml_home)
        dec = american_to_decimal(-market_ml_home)
    shrink = float(ML_MARKET_SHRINK or 0.0)
    if dec >= UNDERDOG_DECIMAL:
        shrink = min(1.0, shrink + float(ML_UNDERDOG_EXTRA_SHRINK or 0.0))
    if shrink > 0 and pd.notna(implied):
        p = (1.0 - shrink) * p + shrink * float(implied)
    return float(np.clip(p, 0.01, 0.99))


def ml_min_ev_for_side(min_ev: float, decimal_odds: float) -> float:
    from pipeline.config import ML_UNDERDOG_MIN_EV

    if ML_UNDERDOG_MIN_EV is not None and decimal_odds >= UNDERDOG_DECIMAL:
        return max(min_ev, float(ML_UNDERDOG_MIN_EV))
    return min_ev


def ml_predicted_winner_side(model_win_prob_home: float) -> str:
    """Side the model expects to win outright (WIN_PROB > 0.5 → Home)."""
    if pd.isna(model_win_prob_home):
        return "Pass"
    if model_win_prob_home > 0.5:
        return "Home"
    if model_win_prob_home < 0.5:
        return "Away"
    return "Pass"


def ml_is_coin_flip(model_win_prob_home: float, band: float | None = None) -> bool:
    """True when the model has no strong lean on the winner."""
    from pipeline.config import ML_COIN_FLIP_BAND

    if pd.isna(model_win_prob_home):
        return False
    half_band = float(ML_COIN_FLIP_BAND if band is None else band)
    return abs(float(model_win_prob_home) - 0.5) <= half_band


def ml_underdog_bet_side(dec_home: float, dec_away: float) -> tuple[str | None, float]:
    """Return (side, decimal_odds) for the underdog, if any."""
    if dec_home >= UNDERDOG_DECIMAL and dec_home >= dec_away:
        return "Home", dec_home
    if dec_away >= UNDERDOG_DECIMAL:
        return "Away", dec_away
    return None, np.nan


def select_ml_bet(
    model_win_prob_home,
    market_ml_home,
    *,
    min_ev: float | None = None,
    max_favorite_decimal: float = 1.45,
    min_win_pct: float | None = None,
    winner_only: bool | None = None,
    coin_flip_band: float | None = None,
    coin_flip_underdog_only: bool | None = None,
    coin_flip_underdog_min_ev: float | None = None,
) -> tuple[str, float, float]:
    """Pick ML bet when calibrated EV and win% pass; Pass otherwise."""
    from pipeline.config import (
        ML_BET_PREDICTED_WINNER_ONLY,
        ML_COIN_FLIP_BAND,
        ML_COIN_FLIP_UNDERDOG_MIN_EV,
        ML_COIN_FLIP_UNDERDOG_ONLY,
        ML_MIN_EV,
        MIN_ML_WIN_PCT,
    )

    if pd.isna(market_ml_home) or model_win_prob_home is None or pd.isna(model_win_prob_home):
        return "Pass", np.nan, np.nan

    if min_ev is None:
        min_ev = ML_MIN_EV
    if min_win_pct is None:
        min_win_pct = float(MIN_ML_WIN_PCT)
    if winner_only is None:
        winner_only = ML_BET_PREDICTED_WINNER_ONLY
    if coin_flip_band is None:
        coin_flip_band = ML_COIN_FLIP_BAND
    if coin_flip_underdog_only is None:
        coin_flip_underdog_only = ML_COIN_FLIP_UNDERDOG_ONLY
    if coin_flip_underdog_min_ev is None:
        coin_flip_underdog_min_ev = ML_COIN_FLIP_UNDERDOG_MIN_EV

    dec_home = american_to_decimal(market_ml_home)
    dec_away = american_to_decimal(-market_ml_home)
    p_home = ml_prob_for_side(model_win_prob_home, market_ml_home, "Home")
    p_away = ml_prob_for_side(model_win_prob_home, market_ml_home, "Away")
    ev_home = (p_home * dec_home) - 1.0
    ev_away = (p_away * dec_away) - 1.0
    min_ev_home = ml_min_ev_for_side(min_ev, dec_home)
    min_ev_away = ml_min_ev_for_side(min_ev, dec_away)
    min_p = float(min_win_pct) / 100.0

    def _qualifies(side: str, ev: float, thr: float, dec: float, prob: float) -> bool:
        return (
            ev > 0
            and ev > thr
            and dec >= max_favorite_decimal
            and prob >= min_p
        )

    if winner_only:
        if ml_is_coin_flip(model_win_prob_home, coin_flip_band):
            if not coin_flip_underdog_only:
                return "Pass", np.nan, np.nan
            ud_side, ud_dec = ml_underdog_bet_side(dec_home, dec_away)
            if ud_side is not None:
                ud_ev = ev_home if ud_side == "Home" else ev_away
                ud_p = p_home if ud_side == "Home" else p_away
                ud_thr = max(
                    ml_min_ev_for_side(min_ev, ud_dec),
                    float(coin_flip_underdog_min_ev),
                )
                if _qualifies(ud_side, ud_ev, ud_thr, ud_dec, ud_p):
                    return ud_side, ud_ev, ud_dec
            return "Pass", np.nan, np.nan

        pick = ml_predicted_winner_side(model_win_prob_home)
        if pick == "Home" and _qualifies("Home", ev_home, min_ev_home, dec_home, p_home):
            return "Home", ev_home, dec_home
        if pick == "Away" and _qualifies("Away", ev_away, min_ev_away, dec_away, p_away):
            return "Away", ev_away, dec_away
        return "Pass", np.nan, np.nan

    candidates = []
    if _qualifies("Home", ev_home, min_ev_home, dec_home, p_home):
        candidates.append(("Home", ev_home, dec_home))
    if _qualifies("Away", ev_away, min_ev_away, dec_away, p_away):
        candidates.append(("Away", ev_away, dec_away))
    if not candidates:
        return "Pass", np.nan, np.nan
    return max(candidates, key=lambda x: x[1])

def market_microstructure_features(spread_move, public_home_pct, market_spread=np.nan):
    """RLM, steam, and vig-free helpers for model features."""
    sm = float(spread_move) if pd.notna(spread_move) else 0.0
    pub = float(public_home_pct) if pd.notna(public_home_pct) else 0.5
    # Reverse line movement: line moved toward home despite public on away (or vice versa)
    rlm = 0.0
    if abs(sm) >= 0.5:
        if sm > 0 and pub < 0.45:
            rlm = abs(sm)
        elif sm < 0 and pub > 0.55:
            rlm = abs(sm)
    steam = int(abs(sm) >= 1.0)
    fair_spread = market_spread  # placeholder when single-book; extend with multi-book de-vig
    return {
        "reverse_line_movement": rlm,
        "steam_flag": steam,
        "fair_spread_vigfree": fair_spread if pd.notna(fair_spread) else 0.0,
        "public_away_pct": 1.0 - pub,
    }


def spread_kelly_fraction(cover_prob: float, juice: float = -110) -> float:
    """Full Kelly fraction for ATS at given juice (e.g. -110)."""
    if cover_prob is None or not np.isfinite(cover_prob):
        return 0.0
    b = 100.0 / 110.0 if juice == -110 else abs(juice) / 100.0
    kelly = (cover_prob * b - (1.0 - cover_prob)) / b
    return float(max(0.0, kelly))


def spread_cover_prob(
    model_spread: float,
    market_spread: float,
    conf_width: float = 24.0,
    direction: str = "Home",
) -> float:
    """P(bet side covers) from Gaussian margin model.

    Uses corrected model spread vs market line; sigma scales with conformal width.
    """
    if pd.isna(model_spread) or pd.isna(market_spread):
        return 0.5
    sigma = max(float(conf_width) / 2.5, 4.0)
    z = (float(model_spread) + float(market_spread)) / sigma
    z = float(np.clip(z, -4.0, 4.0))
    try:
        from scipy.stats import norm
        p_home = float(norm.cdf(z))
    except ImportError:
        p_home = 0.5 * (1.0 + float(np.tanh(z * 0.7978845608)))
    if direction == "Away":
        return 1.0 - p_home
    return p_home


# Alias used in roadmap / docs
spread_cover_prob_gaussian = spread_cover_prob


def total_over_prob_gaussian(
    pred_total,
    market_total,
    sigma,
) -> float:
    """P(Over | L) under N(mu=pred_total, sigma^2).

    P(Over) = 1 - Phi((L - mu) / sigma)
    """
    if pd.isna(pred_total) or pd.isna(market_total) or pd.isna(sigma):
        return 0.5
    sig = float(sigma)
    if not np.isfinite(sig) or sig <= 0:
        return 0.5
    mu = float(pred_total)
    line = float(market_total)
    z = (line - mu) / sig
    z = float(np.clip(z, -6.0, 6.0))
    try:
        from scipy.stats import norm
        return float(1.0 - norm.cdf(z))
    except ImportError:
        # erf approximation of CDF
        return float(0.5 * (1.0 - np.tanh(z * 0.7978845608)))


def total_under_prob_gaussian(pred_total, market_total, sigma) -> float:
    return float(1.0 - total_over_prob_gaussian(pred_total, market_total, sigma))


def crps_gaussian(actual, mu, sigma) -> float:
    """CRPS for a Gaussian forecast N(mu, sigma^2) vs scalar actual."""
    if pd.isna(actual) or pd.isna(mu) or pd.isna(sigma):
        return np.nan
    sig = float(sigma)
    if not np.isfinite(sig) or sig <= 0:
        return np.nan
    y = float(actual)
    m = float(mu)
    z = (y - m) / sig
    try:
        from scipy.stats import norm
        # CRPS(N(m,s^2), y) = s * (z(2Phi(z)-1) + 2phi(z) - 1/sqrt(pi))
        return float(
            sig * (z * (2.0 * norm.cdf(z) - 1.0) + 2.0 * norm.pdf(z) - 1.0 / np.sqrt(np.pi))
        )
    except ImportError:
        # Fallback without scipy: absolute error scaled (not exact CRPS)
        return float(abs(y - m))


def select_ou_bet(
    pred_total,
    market_total,
    sigma=None,
    *,
    min_edge: float | None = None,
    min_prob: float | None = None,
    use_gaussian: bool | None = None,
) -> tuple[str, float, float]:
    """Pick Over / Under / Pass from edge and/or Gaussian P(side).

    Returns (direction, p_over, total_edge).
    """
    from pipeline.config import (
        OU_MIN_EDGE,
        OU_MIN_PROB,
        OU_USE_GAUSSIAN_PROB,
    )

    if pd.isna(pred_total) or pd.isna(market_total) or float(market_total) <= 0:
        return "Pass", 0.5, np.nan

    if min_edge is None:
        min_edge = float(OU_MIN_EDGE)
    if min_prob is None:
        min_prob = float(OU_MIN_PROB)
    if use_gaussian is None:
        use_gaussian = bool(OU_USE_GAUSSIAN_PROB)

    mu = float(pred_total)
    line = float(market_total)
    edge = mu - line
    p_over = 0.5
    if use_gaussian and sigma is not None and np.isfinite(float(sigma)) and float(sigma) > 0:
        p_over = total_over_prob_gaussian(mu, line, sigma)
    else:
        # Soft logistic proxy from edge when sigma missing
        p_over = float(1.0 / (1.0 + np.exp(-edge / 6.0)))
        p_over = float(np.clip(p_over, 0.01, 0.99))

    p_under = 1.0 - p_over
    # Require edge AND probability confidence (when gaussian on)
    if abs(edge) < float(min_edge):
        return "Pass", p_over, edge

    if p_over >= float(min_prob) and edge > 0:
        return "Over", p_over, edge
    if p_under >= float(min_prob) and edge < 0:
        return "Under", p_over, edge
    # Edge clears but prob gate fails → Pass
    if use_gaussian:
        return "Pass", p_over, edge
    # Legacy edge-only mode when gaussian disabled
    return ("Over" if edge > 0 else "Under"), p_over, edge


def explain_ml_bet(
    model_win_prob_home,
    market_ml_home,
    *,
    min_ev: float | None = None,
    **kwargs,
) -> dict:
    """Both-sides ML decision with EV breakdown for predict/simulate UX."""
    from pipeline.config import ML_MIN_EV

    if min_ev is None:
        min_ev = ML_MIN_EV

    predicted_winner = ml_predicted_winner_side(model_win_prob_home)
    p_home_raw = float(model_win_prob_home) if model_win_prob_home is not None and pd.notna(model_win_prob_home) else np.nan
    p_away_raw = (1.0 - p_home_raw) if np.isfinite(p_home_raw) else np.nan

    if pd.isna(market_ml_home) or not np.isfinite(p_home_raw):
        return {
            "predicted_winner": predicted_winner,
            "p_home": p_home_raw,
            "p_away": p_away_raw,
            "ev_home": np.nan,
            "ev_away": np.nan,
            "ml_bet": "Pass",
            "ml_ev": np.nan,
            "ml_decimal": np.nan,
            "ml_reason": "missing_odds_or_prob",
        }

    p_home = ml_prob_for_side(p_home_raw, market_ml_home, "Home")
    p_away = ml_prob_for_side(p_home_raw, market_ml_home, "Away")
    dec_home = american_to_decimal(market_ml_home)
    dec_away = american_to_decimal(-market_ml_home)
    ev_home = (p_home * dec_home) - 1.0
    ev_away = (p_away * dec_away) - 1.0

    side, ev_side, dec_side = select_ml_bet(
        p_home_raw, market_ml_home, min_ev=min_ev, **kwargs,
    )

    if side == "Pass":
        reason = "no_side_clears_ev_and_winpct"
        if ml_is_coin_flip(p_home_raw):
            reason = "coin_flip_pass"
        elif predicted_winner == "Home" and ev_home <= float(min_ev):
            reason = "predicted_winner_home_but_ev_too_low"
        elif predicted_winner == "Away" and ev_away <= float(min_ev):
            reason = "predicted_winner_away_but_ev_too_low"
    elif side != predicted_winner and predicted_winner in ("Home", "Away"):
        reason = f"value_on_{side.lower()}_not_predicted_winner"
    elif side == predicted_winner:
        reason = f"bet_predicted_winner_{side.lower()}_with_positive_ev"
    else:
        reason = "selected"

    return {
        "predicted_winner": predicted_winner,
        "p_home": float(p_home_raw),
        "p_away": float(p_away_raw),
        "p_home_shrunk": float(p_home),
        "p_away_shrunk": float(p_away),
        "ev_home": float(ev_home),
        "ev_away": float(ev_away),
        "ml_bet": side,
        "ml_ev": float(ev_side) if pd.notna(ev_side) else np.nan,
        "ml_decimal": float(dec_side) if pd.notna(dec_side) else np.nan,
        "ml_reason": reason,
    }


def elo_meta_agreement(model_spread, market_spread, elo_margin_calibrated) -> float:
    """1.0 when Elo-calibrated margin and meta spread agree on ATS side vs market."""
    if pd.isna(market_spread) or elo_margin_calibrated is None or pd.isna(elo_margin_calibrated):
        return 1.0
    if pd.isna(model_spread):
        return 1.0
    meta_side = np.sign(float(model_spread) + float(market_spread))
    elo_side = np.sign(float(elo_margin_calibrated) + float(market_spread))
    if meta_side == 0 or elo_side == 0:
        return 1.0
    return float(meta_side == elo_side)


def bet_analysis(model_spread, market_spread, model_win_prob=None, market_ml=None,
                 min_edge_pts=2.5, min_ev=None, conf_width=None,
                 spread_move=0.0, public_home_pct=0.5, require_rlm=False,
                 confidence_calibrator=None, rating_uncertainty=None,
                 max_favorite_decimal=1.45,
                 require_elo_agreement=False, elo_margin_calibrated=None,
                 elo_agreement_extra_edge=1.0, elo_agreement=None,
                 disagreement_trust=1.0, phantom_injury_flag=False,
                 pred_home=None, pred_away=None,
                 matchup_vol_sigma=None, ats_classifier_prob=None):
    from pipeline.config import ML_MIN_EV

    if min_ev is None:
        min_ev = ML_MIN_EV

    result = {
        "spread_edge_pts": np.nan,
        "spread_direction": "Pass",
        "spread_confidence": 0,
        "spread_stars": "No market",
        "ml_ev": np.nan,
        "ml_direction": "Pass",
        "ml_confidence": 0
    }

    if not pd.isna(market_spread):
        edge_pts = spread_edge(model_spread, market_spread)
        result["spread_edge_pts"] = edge_pts
        abs_edge = abs(edge_pts)
        if phantom_injury_flag and float(disagreement_trust or 1.0) < 0.75:
            result["spread_direction"] = "Pass"
            result["spread_stars"] = "Pass (phantom injury)"
        elif abs_edge >= min_edge_pts or min_edge_pts <= 0:
            direction = "Home" if edge_pts > 0 else "Away"
            if require_rlm:
                micro = market_microstructure_features(spread_move, public_home_pct, market_spread)
                if micro["reverse_line_movement"] < 0.25 and micro["steam_flag"] == 0:
                    direction = "Pass"
            if direction != "Pass":
                if require_elo_agreement and elo_margin_calibrated is not None and not pd.isna(elo_margin_calibrated):
                    elo_edge = float(elo_margin_calibrated) + float(market_spread)
                    if np.sign(edge_pts) != np.sign(elo_edge) and abs_edge < min_edge_pts + elo_agreement_extra_edge:
                        direction = "Pass"
                result["spread_direction"] = direction
                cw = conf_width or 24.0
                if elo_agreement is None:
                    elo_agreement = elo_meta_agreement(
                        model_spread, market_spread, elo_margin_calibrated,
                    )
                spread_cover_p = spread_cover_prob(
                    model_spread, market_spread, cw, direction=direction,
                )
                from pipeline.config import USE_SKELLAM_COVER_PROB
                if USE_SKELLAM_COVER_PROB and pred_home is not None and pred_away is not None:
                    from pipeline.skellam import cover_prob_skellam
                    spread_cover_p = cover_prob_skellam(
                        pred_home, pred_away, market_spread, direction=direction,
                    )
                result["spread_cover_prob_raw"] = spread_cover_p
                from pipeline.bet_confidence import build_confidence_features
                ats_feats = build_confidence_features(
                    spread_edge_pts=edge_pts,
                    conf_width=cw,
                    rating_uncertainty=rating_uncertainty or 350.0,
                    elo_meta_agreement=elo_agreement,
                    spread_cover_prob=spread_cover_p,
                    matchup_vol_sigma=matchup_vol_sigma,
                    disagreement_trust=disagreement_trust,
                    phantom_injury_flag=phantom_injury_flag,
                    ats_classifier_prob=ats_classifier_prob,
                )
                ats_feats["win_prob"] = model_win_prob or 0.5
                if confidence_calibrator is not None:
                    cal = confidence_calibrator.predict("ats", ats_feats, direction=direction)
                    result["spread_confidence"] = int(round(cal["confidence_score"]))
                    result["spread_stars"] = cal["stars"]
                    result["cover_prob_calibrated"] = float(np.clip(cal["calibrated_prob"], 0.01, 0.99))
                    result["confidence_tier"] = cal["confidence_tier"]
                    result["confidence_score_raw"] = cal["confidence_score_raw"]
                else:
                    from pipeline.bet_confidence import ats_confidence_score, _tier_from_prob, _stars_from_tier
                    raw_score = ats_confidence_score(ats_feats)
                    prob = float(np.clip(0.35 * spread_cover_p + 0.65 * (0.524 + raw_score / 120.0), 0.01, 0.99))
                    tier = _tier_from_prob(prob)
                    result["spread_confidence"] = int(round(prob * 100))
                    result["cover_prob_calibrated"] = prob
                    result["confidence_tier"] = tier
                    result["spread_stars"] = _stars_from_tier(tier, direction)
                    result["confidence_score_raw"] = raw_score
        else:
            result["spread_direction"] = "Pass"
            result["spread_stars"] = "Pass"

    if (model_win_prob is not None) and not pd.isna(market_ml):
        side, ev_side, dec_side = select_ml_bet(
            model_win_prob,
            market_ml,
            min_ev=min_ev,
            max_favorite_decimal=max_favorite_decimal,
        )
        p_home = ml_prob_for_side(model_win_prob, market_ml, "Home")
        ev_home = (p_home * american_to_decimal(market_ml)) - 1.0
        result["ml_ev"] = ev_home
        if side != "Pass":
            result["ml_direction"] = side
            if confidence_calibrator is not None:
                cal = confidence_calibrator.predict(
                    "ml",
                    {"win_prob": model_win_prob, "ml_ev": ev_side,
                     "rating_uncertainty": rating_uncertainty or 350.0},
                    direction=side,
                )
                result["ml_confidence"] = cal["confidence_score"]
                result["ml_confidence_tier"] = cal["confidence_tier"]
            else:
                result["ml_confidence"] = min(100, int(ev_side * 1000))
        else:
            result["ml_direction"] = "Pass"

    return result

# ──────────────────────────────────────────────────────────────────────────────
# 3. FINANCIAL CALCULATOR (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def calculate_financials(df, unit_size=10.0, use_fractional_kelly=False):
    df = df.copy()
    df["SPREAD_PROFIT"] = 0.0
    win_payout = unit_size * (100 / 110)

    if use_fractional_kelly and "KELLY_FRACTION" in df.columns:
        effective_unit = unit_size * df["KELLY_FRACTION"].fillna(0)
    else:
        effective_unit = unit_size

    mask_win = df["ATS_WIN"] == 1
    mask_loss = df["ATS_LOSS"] == 1
    df.loc[mask_win, "SPREAD_PROFIT"] = win_payout * (effective_unit / unit_size)
    df.loc[mask_loss, "SPREAD_PROFIT"] = -effective_unit

    df["ML_PROFIT"] = 0.0
    df["ML_BET"] = "Pass"
    if "MARKET_ML" in df.columns and "WIN_PROB" in df.columns:
        for idx, row in df.iterrows():
            ml = row["MARKET_ML"]
            if pd.isna(ml):
                continue
            model_prob_home = row["WIN_PROB"]
            side, ev_side, dec = select_ml_bet(model_prob_home, ml)
            if side == "Pass":
                continue
            df.at[idx, "ML_BET"] = side
            home_win = row["ACTUAL_HOME"] > row["ACTUAL_AWAY"]
            won = home_win if side == "Home" else not home_win
            if won:
                df.at[idx, "ML_PROFIT"] = unit_size * (dec - 1)
            else:
                df.at[idx, "ML_PROFIT"] = -unit_size
    return df

# ──────────────────────────────────────────────────────────────────────────────
# 4. REPORTING AND GRAPHING (unchanged)
# ──────────────────────────────────────────────────────────────────────────────

def print_interval_ats_report(results_df):
    if results_df.empty:
        return None
    df = results_df[results_df["MARKET_SPREAD"].notna() & (results_df["DIRECTION"] != "Pass")].copy()
    if df.empty:
        print("  [No Vegas Market lines found or zero active bets triggered]")
        return None

    home_covers = (df["ACTUAL_MARGIN"] + df["MARKET_SPREAD"]) > 0
    away_covers = (df["ACTUAL_MARGIN"] + df["MARKET_SPREAD"]) < 0
    pushes = (df["ACTUAL_MARGIN"] + df["MARKET_SPREAD"]) == 0

    df["ATS_WIN"] = (((df["DIRECTION"] == "Home") & home_covers) |
                     ((df["DIRECTION"] == "Away") & away_covers)).astype(int)
    df["ATS_LOSS"] = (((df["DIRECTION"] == "Home") & away_covers) |
                      ((df["DIRECTION"] == "Away") & home_covers)).astype(int)
    df["ATS_PUSH"] = pushes.astype(int)

    df = calculate_financials(df, unit_size=10.0)

    bins = [-1.0, 2.999, 4.999, 7.999, float('inf')]
    labels = ["0.0 - 2.9", "3.0 - 4.9", "5.0 - 7.9", "8.0+"]
    df["EDGE_TIER"] = pd.cut(df["EDGE"], bins=bins, labels=labels, right=True)

    print("══════════════════════════════════════════════════════════════")
    print("      SPREAD BETTING PERFORMANCE ($10 Base Unit)              ")
    print("══════════════════════════════════════════════════════════════")
    for tier in labels:
        tier_df = df[df["EDGE_TIER"] == tier]
        total = len(tier_df)
        if total == 0:
            continue
        w = tier_df["ATS_WIN"].sum()
        l = tier_df["ATS_LOSS"].sum()
        p = tier_df["ATS_PUSH"].sum()
        win_pct = w / (w + l) if (w + l) > 0 else 0.0
        profit = tier_df["SPREAD_PROFIT"].sum()
        print(f" Edge {tier:>9} pts | {total:3d} bets | {w:2d}-{l:2d}-{p:1d} | {win_pct:5.1%} ATS | Net: ${profit:+.2f}")

    ml_df = df[df["ML_BET"] != "Pass"]
    if not ml_df.empty:
        ml_wins = (ml_df["ML_PROFIT"] > 0).sum()
        ml_losses = (ml_df["ML_PROFIT"] < 0).sum()
        ml_profit = ml_df["ML_PROFIT"].sum()
        ml_win_pct = ml_wins / len(ml_df)
        print("──────────────────────────────────────────────────────────────")
        print("      MONEYLINE VALUE PERFORMANCE ($10 Base Unit)             ")
        print("──────────────────────────────────────────────────────────────")
        print(f" Total ML Plays: {len(ml_df):3d} | {ml_wins:2d} Wins - {ml_losses:2d} Losses | {ml_win_pct:5.1%} Hit Rate")
        print(f" Total Net Profit: ${ml_profit:+.2f}")
    print("══════════════════════════════════════════════════════════════\n")
    return df

def plot_financial_performance(financial_df, title="Cumulative_Profit"):
    if financial_df is None or financial_df.empty:
        return
    df = financial_df.sort_values("DATE").reset_index(drop=True)
    df["CUM_SPREAD"] = df["SPREAD_PROFIT"].cumsum()
    plt.figure(figsize=(14, 7))
    plt.plot(df.index, df["CUM_SPREAD"], label="ATS Spread Profit ($10/bet)", color="#2ca02c", linewidth=2.5)
    if "ML_PROFIT" in df.columns:
        df["CUM_ML"] = df["ML_PROFIT"].cumsum()
        plt.plot(df.index, df["CUM_ML"], label="Moneyline Profit ($10/bet)", color="#1f77b4", linewidth=2.5, alpha=0.8)
    plt.axhline(0, color='red', linestyle='--', linewidth=1.5, alpha=0.6)
    plt.title(f"Simulation Equity Curve: {title}", fontsize=14, fontweight="bold")
    plt.xlabel("Number of Bets Placed (Chronological)", fontsize=12)
    plt.ylabel("Cumulative Profit ($)", fontsize=12)
    plt.legend(loc="upper left", fontsize=11)
    plt.grid(alpha=0.3)
    filename = f"equity_curve_{title.replace(' ', '_').lower()}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"✅ Equity curve graph saved as '{filename}'")

class SpreadCalibrator:
    """Rolling spread calibration.

    Fits ``actual_margin ≈ a + b · pred_margin`` over a rolling window, so it
    corrects both a global additive bias **and** systematic slope error (e.g.
    shrinking big favorites). Falls back to additive-bias correction before
    ``min_samples`` points accrue, matching the previous behaviour at cold start.
    """
    def __init__(self, window=50, min_samples=30, slope_clip=(0.3, 1.7), mode="linear"):
        self.window = window
        self.min_samples = min_samples
        self.slope_clip = slope_clip
        self.mode = mode   # "linear" (a + b*pred) or "additive" (legacy bias-only)
        self.preds = deque(maxlen=window)
        self.actuals = deque(maxlen=window)
        self.errors = deque(maxlen=window)
        self.abs_residuals = deque(maxlen=window)
        self.bias = 0.0
        self.a = 0.0
        self.b = 1.0

    def update(self, pred, actual):
        self.preds.append(pred)
        self.actuals.append(actual)
        self.errors.append(pred - actual)
        self.abs_residuals.append(abs(pred - actual))
        self.bias = float(np.mean(self.errors))
        if len(self.preds) >= self.min_samples:
            x = np.asarray(self.preds, dtype=float)
            y = np.asarray(self.actuals, dtype=float)
            vx = x.var()
            if vx > 1e-6:
                b = np.cov(x, y, bias=True)[0, 1] / vx
                self.b = float(np.clip(b, *self.slope_clip))
                self.a = float(y.mean() - self.b * x.mean())
            else:
                self.a, self.b = 0.0, 1.0

    def correct(self, pred):
        if self.mode == "linear" and len(self.preds) >= self.min_samples:
            return self.a + self.b * pred
        return pred - self.bias

    def predict_interval(self, pred, alpha=0.10):
        """Rolling conformal-style symmetric interval from |pred - actual| residuals."""
        if len(self.abs_residuals) >= self.min_samples:
            q = float(np.quantile(np.asarray(self.abs_residuals, dtype=float), 1.0 - alpha / 2.0))
            return pred - q, pred + q, 2.0 * q
        half = 12.0
        return pred - half, pred + half, 2.0 * half

    @classmethod
    def fit_static(cls, preds, actuals, window=50, min_samples=30, slope_clip=(0.3, 1.7), mode="linear"):
        """Offline fit on a calibration slice (no future leakage).

        .. warning::
            The fitted calibrator's ``a``/``b`` reflect the *end state*
            after replaying every ``(pred, actual)`` pair. Applying
            ``.correct()`` with that end state to every row of the same
            slice (as ``pipeline/backtest.py``'s ``static_cal`` path does)
            corrects early rows with parameters informed by later rows in
            the same slice — this is exactly the "static-calibration-to-
            rolling-inference mismatch" named in Task 054. Use
            :func:`replay_rolling_spread_calibration` instead when you need
            row-by-row corrected predictions that match how live inference
            actually calibrates (predict, *then* observe/update).
        """
        obj = cls(window=max(window, len(preds)), min_samples=min_samples,
                  slope_clip=slope_clip, mode=mode)
        for p, a in zip(preds, actuals):
            if pd.notna(p) and pd.notna(a):
                obj.update(float(p), float(a))
        return obj


def replay_rolling_spread_calibration(preds, actuals, window=50, min_samples=30,
                                       slope_clip=(0.3, 1.7), mode="linear",
                                       calibrator=None):
    """Row-by-row rolling spread calibration replay (Task 054).

    For each row, in order, this calls ``calibrator.correct(pred)`` — using
    only state accumulated from *strictly earlier* rows in this same call —
    before ``calibrator.update(pred, actual)`` observes that row's outcome.
    This is exactly the sequence live inference uses (``pipeline/simulate.py``
    calls ``.correct()`` to produce a prediction, then ``.update()`` once the
    result is known) and is the single correct way to reproduce rolling
    spread calibration inside a training-time simulation — replacing any
    "fit once on the whole slice, then re-correct every row in it with the
    final state" (``SpreadCalibrator.fit_static`` + blanket ``.correct()``)
    pattern, which lets later rows in a slice leak into earlier corrections.

    Returns
    -------
    (corrected, calibrator) : np.ndarray, SpreadCalibrator
        ``corrected[i]`` uses only rows ``< i`` from this call plus whatever
        state ``calibrator`` already had when passed in (so training
        simulation and live inference can share literally the same
        calibrator instance across a season boundary).
    """
    cal = calibrator if calibrator is not None else SpreadCalibrator(
        window=window, min_samples=min_samples, slope_clip=slope_clip, mode=mode,
    )
    corrected = np.empty(len(preds), dtype=float)
    for i, (p, a) in enumerate(zip(preds, actuals)):
        p = float(p)
        corrected[i] = cal.correct(p)
        if pd.notna(a):
            cal.update(p, float(a))
    return corrected, cal


def load_modern_odds(csv_path, date_col="game_date", home_col="home_team",
                     spread_col="spread_home_points", ml_col="money_home_odds",
                     time_col=None, game_start_hour=19):
    """Load opening lines keyed by (game_date, home_team)."""
    odds_dict = {}
    try:
        df = pd.read_csv(csv_path)
        df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
        if time_col and time_col in df.columns:
            df["timestamp"] = pd.to_datetime(df[time_col], errors="coerce")
            df = df.sort_values(by=[date_col, "timestamp"])
        else:
            df = df.sort_values(by=[date_col])

        df["game_date_only"] = df[date_col].dt.date
        grouped = df.groupby(["game_date_only", home_col], as_index=False).first()

        for _, row in grouped.iterrows():
            dt_val = row["game_date_only"]
            if pd.isna(dt_val):
                continue
            home = str(row[home_col]).strip()
            if home == "Los Angeles Lakers":
                home = "LA Lakers"
            if home == "Los Angeles Clippers":
                home = "LA Clippers"

            s_val = row[spread_col] if pd.notna(row.get(spread_col)) else np.nan
            m_val = row[ml_col] if pd.notna(row.get(ml_col)) else np.nan

            if pd.notna(s_val) and pd.notna(m_val):
                if m_val < 0 and s_val > 0:
                    s_val = -abs(s_val)
                elif m_val > 0 and s_val < 0:
                    s_val = abs(s_val)

            entry = {"spread": float(s_val), "ml": float(m_val), "spread_open": float(s_val)}
            if pd.notna(row.get("total_over_points")):
                entry["total"] = float(row["total_over_points"])
            pub = row.get("spread_home_stake_percentage")
            if pd.notna(pub):
                entry["public_home_pct"] = float(pub) / 100.0
            odds_dict[(dt_val, home)] = entry
    except Exception as e:
        print(f"Warning loading modern odds: {e}")
    return odds_dict


def decimal_to_american(dec):
    """Convert decimal odds to American odds."""
    try:
        dec = float(dec)
    except (TypeError, ValueError):
        return np.nan
    if not np.isfinite(dec) or dec <= 1.0:
        return np.nan
    if dec >= 2.0:
        return round((dec - 1.0) * 100.0)
    return round(-100.0 / (dec - 1.0))


def load_pinnacle_lines(csv_path, schedule_df=None, match_tolerance_days=2):
    """Load Pinnacle-style lines (nba_main_lines.csv) into the odds_dict format.

    The file has columns team1/team2, decimal moneyline/spread/total odds, and a
    ``timestamp`` (no game date). We take each game's CLOSING snapshot (latest
    timestamp, ~tipoff) and convert decimal odds to American.

    If ``schedule_df`` (columns: date, home, away as tricodes) is provided, each
    odds game is matched to the real schedule game by team matchup and nearest
    date, so the authoritative game date and home/away come from the schedule.
    Otherwise we fall back to dual (closing_date, team) keys storing each team's
    own line, which still lets get_odds resolve the home team.

    Returns a dict keyed by (date, team) -> {"spread", "ml", "total"}.
    """
    from pipeline.dates import to_tricode
    out = {}
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:  # noqa: BLE001
        print(f"Warning loading pinnacle lines: {e}")
        return out
    if "timestamp" not in df.columns or df.empty:
        return out

    df["ts"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["ts"])
    key = "game_link" if "game_link" in df.columns else None

    # Closing and opening snapshots per game.
    if key:
        closing = df.sort_values("ts").groupby(key, as_index=False).last()
        opening = df.sort_values("ts").groupby(key, as_index=False).first()
        open_map = {}
        for _, orow in opening.iterrows():
            gl = orow.get("game_link")
            t1o, t2o = to_tricode(orow.get("team1")), to_tricode(orow.get("team2"))
            open_map[gl] = (t1o, t2o, orow.get("team1_spread"), orow.get("team2_spread"))
    else:
        closing = df.sort_values("ts")
        open_map = {}

    # Pre-index schedule by matchup (frozenset of tricodes) for fast matching.
    sched_idx = {}
    if schedule_df is not None and not schedule_df.empty:
        s = schedule_df.copy()
        s["home"] = s["home"].apply(to_tricode)
        s["away"] = s["away"].apply(to_tricode)
        s["date"] = pd.to_datetime(s["date"], errors="coerce")
        for _, r in s.dropna(subset=["date", "home", "away"]).iterrows():
            sched_idx.setdefault(frozenset((r["home"], r["away"])), []).append(
                (r["date"], r["home"], r["away"]))

    for _, row in closing.iterrows():
        t1 = to_tricode(row.get("team1"))
        t2 = to_tricode(row.get("team2"))
        if not isinstance(t1, str) or not isinstance(t2, str):
            continue
        ml1 = decimal_to_american(row.get("team1_moneyline"))
        ml2 = decimal_to_american(row.get("team2_moneyline"))
        spr1 = row.get("team1_spread")
        spr2 = row.get("team2_spread")
        total = row.get("over_total")
        ts_date = row["ts"].date()

        matched = None
        cand = sched_idx.get(frozenset((t1, t2)))
        if cand:
            # nearest scheduled date to the closing timestamp
            best = min(cand, key=lambda c: abs((c[0].date() - ts_date).days))
            if abs((best[0].date() - ts_date).days) <= match_tolerance_days:
                matched = best

        spr1_price, spr2_price = _f(row.get("team1_spread_odds")), _f(row.get("team2_spread_odds"))
        over_price, under_price = _f(row.get("over_total_odds")), _f(row.get("under_total_odds"))

        if matched is not None:
            gdate, home, away = matched
            d = gdate.date()
            spread_open = np.nan
            if key and row.get("game_link") in open_map:
                t1o, t2o, os1, os2 = open_map[row.get("game_link")]
                spread_open = _f(os1 if home == t1o else os2)
            # Task 024: keep BOTH teams' actual quoted prices — never derive
            # the away side by negating the home side.
            if home == t1:
                close_spread, close_ml = _f(spr1), _f(ml1)
                away_spread, away_ml = _f(spr2), _f(ml2)
                home_price, away_price = spr1_price, spr2_price
            else:
                close_spread, close_ml = _f(spr2), _f(ml2)
                away_spread, away_ml = _f(spr1), _f(ml1)
                home_price, away_price = spr2_price, spr1_price
            entry = {
                "spread": close_spread, "ml": close_ml, "total": _f(total),
                "spread_open": spread_open if pd.notna(spread_open) else close_spread,
                "closing_spread": close_spread,
                "away_team": away,
                "spread_away": away_spread, "ml_away": away_ml,
                "spread_home_price": home_price, "spread_away_price": away_price,
                "total_over_price": over_price, "total_under_price": under_price,
            }
            if pd.notna(entry["spread_open"]) and pd.notna(entry["spread"]):
                entry["spread_move"] = float(entry["spread"]) - float(entry["spread_open"])
            out[(d, home)] = entry
            # Also store the away team's own perspective for this exact game
            # so a lookup by (date, away_team) never needs to fall back to a
            # different game (Task 025).
            away_entry = {
                "spread": away_spread, "ml": away_ml, "total": _f(total),
                "spread_open": spread_open if pd.notna(spread_open) else close_spread,
                "closing_spread": away_spread,
                "away_team": home,
                "spread_away": close_spread, "ml_away": close_ml,
                "spread_home_price": away_price, "spread_away_price": home_price,
                "total_over_price": over_price, "total_under_price": under_price,
            }
            out[(d, away)] = away_entry
        else:
            spread_open = np.nan
            if key and row.get("game_link") in open_map:
                t1o, t2o, os1, os2 = open_map[row.get("game_link")]
                spread_open = _f(os1) if t1 == t1o else _f(os2)
            pairs = ((t1, spr1, ml1, t2, spr2, ml2, spr1_price, spr2_price),
                     (t2, spr2, ml2, t1, spr1, ml1, spr2_price, spr1_price))
            for team, spr, ml, opp, opp_spr, opp_ml, own_price, opp_price in pairs:
                cs, cm = _f(spr), _f(ml)
                entry = {
                    "spread": cs, "ml": cm, "total": _f(total),
                    "spread_open": spread_open if pd.notna(spread_open) else cs,
                    "closing_spread": cs,
                    "away_team": opp,
                    "spread_away": _f(opp_spr), "ml_away": _f(opp_ml),
                    "spread_home_price": own_price, "spread_away_price": opp_price,
                    "total_over_price": over_price, "total_under_price": under_price,
                }
                if pd.notna(entry["spread_open"]) and pd.notna(entry["spread"]):
                    entry["spread_move"] = float(entry["spread"]) - float(entry["spread_open"])
                out[(ts_date, team)] = entry
    return out


def _f(v):
    try:
        v = float(v)
        return v if np.isfinite(v) else np.nan
    except (TypeError, ValueError):
        return np.nan


print("✅ Leakage‑free Vegas helpers, ML/spread edge calculators, and financials loaded.")