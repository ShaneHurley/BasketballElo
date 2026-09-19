"""Canonical offense-lineup vs defense matchup rating layer (T-60 Phase 2).

Produces interpretable points-per-100 matchup forecasts from projected
rotations without leaking current-game minutes. Elo/Glicko player ratings
remain the online updater; this module is the single inspectable interface
consumed by ATS / ML / totals heads.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class MatchupForecast:
    """Home/away scoring rates and derived margin/total moments."""

    home_pp100: float
    away_pp100: float
    home_offense: float
    home_defense: float
    away_offense: float
    away_defense: float
    exp_poss: float
    home_court_pp100: float
    margin: float
    total: float
    home_pts: float
    away_pts: float
    uncertainty: float
    home_offense_pp100: float
    away_offense_pp100: float

    def as_feature_dict(self, prefix: str = "matchup_") -> dict[str, float]:
        return {
            f"{prefix}home_pp100": self.home_pp100,
            f"{prefix}away_pp100": self.away_pp100,
            f"{prefix}margin": self.margin,
            f"{prefix}total": self.total,
            f"{prefix}home_pts": self.home_pts,
            f"{prefix}away_pts": self.away_pts,
            f"{prefix}uncertainty": self.uncertainty,
            f"{prefix}home_offense_pp100": self.home_offense_pp100,
            f"{prefix}away_offense_pp100": self.away_offense_pp100,
            f"{prefix}off_vs_def_home": self.home_offense - self.away_defense,
            f"{prefix}off_vs_def_away": self.away_offense - self.home_defense,
        }


def elo_to_pp100_delta(elo_mu: float, *, base: float = 1500.0, scaling: float = 1000.0) -> float:
    """Convert Elo μ to points-per-100 contribution relative to league average."""
    return (float(elo_mu) - float(base)) / (float(scaling) / 100.0)


def pp100_delta_to_elo(pp100: float, *, base: float = 1500.0, scaling: float = 1000.0) -> float:
    return float(base) + float(pp100) * (float(scaling) / 100.0)


def matchup_pp100(
    home_offense: float,
    home_defense: float,
    away_offense: float,
    away_defense: float,
    *,
    league_pp100: float = 110.0,
    home_court_pp100: float = 2.0,
    context_home: float = 0.0,
    context_away: float = 0.0,
) -> tuple[float, float]:
    """Offense vs opponent defense on a points-per-100 scale.

    ``home_pp100 = league + (home_off - away_def) + HCA + context``
    ``away_pp100 = league + (away_off - home_def) + context``
    Ratings are already in Elo μ space; convert via ``elo_to_pp100_delta``.
    """
    h_off = elo_to_pp100_delta(home_offense)
    h_def = elo_to_pp100_delta(home_defense)
    a_off = elo_to_pp100_delta(away_offense)
    a_def = elo_to_pp100_delta(away_defense)
    home_pp100 = league_pp100 + (h_off - a_def) + home_court_pp100 + context_home
    away_pp100 = league_pp100 + (a_off - h_def) + context_away
    return float(home_pp100), float(away_pp100)


def forecast_from_lineup_elos(
    home_off: float,
    home_def: float,
    away_off: float,
    away_def: float,
    exp_poss: float,
    *,
    league_pp100: float = 110.0,
    home_court_pp100: float = 2.0,
    home_unc: float = 350.0,
    away_unc: float = 350.0,
    context_home: float = 0.0,
    context_away: float = 0.0,
    synergy_home_pp100: float = 0.0,
    synergy_away_pp100: float = 0.0,
) -> MatchupForecast:
    """Build a matchup forecast from projected lineup Elo μ values."""
    poss = float(max(exp_poss, 1.0))
    home_pp100, away_pp100 = matchup_pp100(
        home_off, home_def, away_off, away_def,
        league_pp100=league_pp100,
        home_court_pp100=home_court_pp100,
        context_home=context_home + synergy_home_pp100,
        context_away=context_away + synergy_away_pp100,
    )
    home_pts = home_pp100 * poss / 100.0
    away_pts = away_pp100 * poss / 100.0
    unc = (float(home_unc) + float(away_unc)) / 2.0
    return MatchupForecast(
        home_pp100=home_pp100,
        away_pp100=away_pp100,
        home_offense=float(home_off),
        home_defense=float(home_def),
        away_offense=float(away_off),
        away_defense=float(away_def),
        exp_poss=poss,
        home_court_pp100=float(home_court_pp100),
        margin=home_pts - away_pts,
        total=home_pts + away_pts,
        home_pts=home_pts,
        away_pts=away_pts,
        uncertainty=unc,
        home_offense_pp100=elo_to_pp100_delta(home_off),
        away_offense_pp100=elo_to_pp100_delta(away_off),
    )


def moment_match_scenarios(
    scenarios: Sequence[MatchupForecast],
    weights: Sequence[float] | None = None,
) -> MatchupForecast:
    """Combine rotation scenarios into mean margin/total with mixture variance."""
    if not scenarios:
        raise ValueError("scenarios must be non-empty")
    n = len(scenarios)
    if weights is None:
        w = np.ones(n, dtype=float) / n
    else:
        w = np.asarray(weights, dtype=float)
        if len(w) != n:
            raise ValueError("weights length must match scenarios")
        s = float(w.sum())
        if s <= 0:
            raise ValueError("weights must sum to positive")
        w = w / s

    home_pts = np.array([s.home_pts for s in scenarios], dtype=float)
    away_pts = np.array([s.away_pts for s in scenarios], dtype=float)
    margins = home_pts - away_pts
    totals = home_pts + away_pts
    mean_home = float(np.dot(w, home_pts))
    mean_away = float(np.dot(w, away_pts))
    mean_margin = float(np.dot(w, margins))
    mean_total = float(np.dot(w, totals))
    # Mixture variance of margin as uncertainty proxy (in points).
    var_margin = float(np.dot(w, (margins - mean_margin) ** 2))
    base_unc = float(np.dot(w, [s.uncertainty for s in scenarios]))
    try:
        from pipeline.config import SCENARIO_VAR_TO_RD_SCALE
        scale = float(SCENARIO_VAR_TO_RD_SCALE)
    except Exception:
        scale = 50.0
    unc = float(np.sqrt(max(var_margin, 0.0)) * scale + base_unc)

    ref = scenarios[0]
    poss = float(np.dot(w, [s.exp_poss for s in scenarios]))
    return MatchupForecast(
        home_pp100=mean_home / poss * 100.0,
        away_pp100=mean_away / poss * 100.0,
        home_offense=ref.home_offense,
        home_defense=ref.home_defense,
        away_offense=ref.away_offense,
        away_defense=ref.away_defense,
        exp_poss=poss,
        home_court_pp100=ref.home_court_pp100,
        margin=mean_margin,
        total=mean_total,
        home_pts=mean_home,
        away_pts=mean_away,
        uncertainty=unc,
        home_offense_pp100=ref.home_offense_pp100,
        away_offense_pp100=ref.away_offense_pp100,
    )


def forecast_from_trackers(
    elo_tracker,
    home_lineup: Sequence[Any],
    away_lineup: Sequence[Any],
    exp_poss: float,
    *,
    home_weights=None,
    away_weights=None,
    league_pp100: float = 110.0,
    home_court_pp100: float | None = None,
    opp_rim_home_d: float | None = None,
    opp_three_home_d: float | None = None,
    opp_rim_away_d: float | None = None,
    opp_three_away_d: float | None = None,
    synergy_home_pp100: float = 0.0,
    synergy_away_pp100: float = 0.0,
) -> MatchupForecast:
    """Snapshot matchup from a PlayerRatingTracker at T-60 (no updates)."""
    scaling = float(elo_tracker.cfg.get("ELO_SCALING_FACTOR", 1000))
    if home_court_pp100 is None:
        # HOME_PPP_BOOST is PPP; convert to pts/100.
        home_court_pp100 = float(elo_tracker.cfg.get("HOME_PPP_BOOST", 0.002)) * 100.0

    kwargs_h = {}
    kwargs_a = {}
    if opp_rim_home_d is not None and opp_three_home_d is not None:
        kwargs_h = {"opp_rim_rate": opp_rim_home_d, "opp_three_rate": opp_three_home_d}
    if opp_rim_away_d is not None and opp_three_away_d is not None:
        kwargs_a = {"opp_rim_rate": opp_rim_away_d, "opp_three_rate": opp_three_away_d}

    if home_weights:
        ho, hd, _ = elo_tracker.weighted_lineup_stats(home_weights, **kwargs_h)
        h_o_rd, h_d_rd = elo_tracker.weighted_lineup_uncertainty(home_weights)
    else:
        ho, hd, _ = elo_tracker.lineup_stats(home_lineup, **kwargs_h)
        h_o_rd, h_d_rd = elo_tracker.lineup_uncertainty(home_lineup)
    if away_weights:
        ao, ad, _ = elo_tracker.weighted_lineup_stats(away_weights, **kwargs_a)
        a_o_rd, a_d_rd = elo_tracker.weighted_lineup_uncertainty(away_weights)
    else:
        ao, ad, _ = elo_tracker.lineup_stats(away_lineup, **kwargs_a)
        a_o_rd, a_d_rd = elo_tracker.lineup_uncertainty(away_lineup)

    # Express league on Elo scale: league_pp100 ≈ league_xppp * 100
    return forecast_from_lineup_elos(
        ho, hd, ao, ad, exp_poss,
        league_pp100=league_pp100,
        home_court_pp100=home_court_pp100,
        home_unc=h_o_rd + h_d_rd,
        away_unc=a_o_rd + a_d_rd,
        synergy_home_pp100=synergy_home_pp100,
        synergy_away_pp100=synergy_away_pp100,
    )


# ---------------------------------------------------------------------------
# Synthetic invariants (used by tests)
# ---------------------------------------------------------------------------

def assert_equal_lineups_home_court_only(
    home_off: float = 1500.0,
    *,
    home_court_pp100: float = 2.0,
    exp_poss: float = 100.0,
    tol: float = 1e-6,
) -> None:
    f = forecast_from_lineup_elos(
        home_off, home_off, home_off, home_off, exp_poss,
        home_court_pp100=home_court_pp100,
    )
    expected_margin = home_court_pp100 * exp_poss / 100.0
    if abs(f.margin - expected_margin) > tol:
        raise AssertionError(f"equal lineups should yield only HCA: got {f.margin}, want {expected_margin}")


def assert_offense_raises_own_scoring(delta_elo: float = 50.0, exp_poss: float = 100.0) -> None:
    base = forecast_from_lineup_elos(1500, 1500, 1500, 1500, exp_poss)
    boosted = forecast_from_lineup_elos(1500 + delta_elo, 1500, 1500, 1500, exp_poss)
    if not (boosted.home_pts > base.home_pts and abs(boosted.away_pts - base.away_pts) < 1e-6):
        raise AssertionError("raising home offense must raise only home scoring")


def assert_defense_lowers_opp_scoring(delta_elo: float = 50.0, exp_poss: float = 100.0) -> None:
    base = forecast_from_lineup_elos(1500, 1500, 1500, 1500, exp_poss)
    better_def = forecast_from_lineup_elos(1500, 1500 + delta_elo, 1500, 1500, exp_poss)
    if not (better_def.away_pts < base.away_pts):
        raise AssertionError("raising home defense must lower away scoring")


def assert_home_away_swap_reverses_margin(exp_poss: float = 100.0, home_court_pp100: float = 2.0) -> None:
    f = forecast_from_lineup_elos(1550, 1480, 1520, 1490, exp_poss, home_court_pp100=home_court_pp100)
    # Swap sides with zero HCA: margin should negate after removing HCA from first.
    g = forecast_from_lineup_elos(1520, 1490, 1550, 1480, exp_poss, home_court_pp100=0.0)
    f_no_hca = forecast_from_lineup_elos(1550, 1480, 1520, 1490, exp_poss, home_court_pp100=0.0)
    if abs(f_no_hca.margin + g.margin) > 1e-5:
        raise AssertionError("swapping home/away without HCA must negate margin")


def residual_update_signs(
    act_ppp_home: float,
    exp_ppp_home: float,
    act_ppp_away: float,
    exp_ppp_away: float,
) -> dict[str, float]:
    """Documented residual signs for a stint update (offense vs defense).

    Home offense residual: act_home - exp_home (positive → raise home O).
    Away defense residual for home scoring: -(act_home - exp_home)
      (home over-scored → lower away D / raise away D rating depending on
      convention; here we return the raw errors used by PlayerRatingTracker).
    """
    err_h = float(act_ppp_home - exp_ppp_home)
    err_a = float(act_ppp_away - exp_ppp_away)
    return {
        "home_offense_error": err_h,
        "away_offense_error": err_a,
        "home_defense_error_vs_away": -err_a,
        "away_defense_error_vs_home": -err_h,
    }
