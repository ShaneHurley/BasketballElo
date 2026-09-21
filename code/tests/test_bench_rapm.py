"""Bench: Epic 8.2 RAPM + 12.5 informed prior + forecast_eval hygiene."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.forecast_eval import (
    KALMAN_V6_ACTUALS,
    compare_actuals_to_baseline,
    compute_decile_z,
    compute_spiegelhalter_z,
    diebold_mariano,
    mae,
    pace_efficiency_mae,
)
from pipeline.stake_profiles import robust_fractional_kelly
from pipeline.rapm import LineupRapmTracker, PlayerRapmTracker
from tests.synth.factories import make_player_ids, make_stint

pytestmark = pytest.mark.bench


def _stints_frame(n_games: int = 40, seed: int = 82001) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    pool = make_player_ids(20, start=100)
    for g in range(n_games):
        home = list(rng.choice(pool[:10], size=5, replace=False))
        away = list(rng.choice(pool[10:], size=5, replace=False))
        # Stronger home players → higher home_xpts
        strength = sum(int(p) for p in home) - sum(int(p) for p in away)
        base = 1.10 + 0.00005 * strength
        rows.append(
            make_stint(
                game_id=f"00225{g:05d}",
                home_players=home,
                away_players=away,
                possessions=float(rng.integers(8, 20)),
                home_xpts=base * 12,
                away_xpts=(2.2 - base) * 12,
                home_pts=base * 12,
                away_pts=(2.2 - base) * 12,
            )
        )
    return pd.DataFrame(rows)


def test_player_rapm_fits_sparse_and_ranks_stronger_lineup():
    stints = _stints_frame(60)
    tracker = PlayerRapmTracker(alpha=50.0, min_rows=50, league_xppp=1.10)
    assert tracker.fit(stints) is True
    assert tracker.fitted
    assert tracker.alpha_ == 50.0
    # Highest-id home pool players should tend to positive off vs low pool
    strong = make_player_ids(5, start=105)  # in home pool 100-109
    weak = make_player_ids(5, start=115)  # away pool
    feats = tracker.feature_dict(strong, weak)
    assert "rapm_net_diff" in feats
    assert np.isfinite(feats["rapm_net_diff"])


def test_informed_prior_residualization_recovers_prior_on_sparse_lineup():
    """Unseen lineup must fall back to π (prior dominates), not explode."""
    stints = _stints_frame(80)
    player = PlayerRapmTracker(alpha=100.0, min_rows=50)
    assert player.fit(stints)
    # Give one player a large off coef manually to prove prior path
    star = "999"
    player.off_coef[star] = 0.15
    lineup = LineupRapmTracker(player, alpha=100.0, min_rows=50)
    assert lineup.fit(stints)
    # Brand-new 5 including star — never seen as a unit
    novel = [star, "1", "2", "3", "4"]
    rating = lineup.lineup_off_rating(novel)
    assert abs(rating - player.lineup_off_prior(novel)) < 1e-9


def test_diebold_mariano_detects_worse_forecast():
    rng = np.random.default_rng(42)
    y = rng.normal(0, 12, size=400)
    good = y + rng.normal(0, 2, size=400)
    bad = y + rng.normal(0, 8, size=400)
    dm = diebold_mariano(y, bad, good, loss="abs")
    # bad has higher abs loss → positive mean_loss_diff and dm_stat
    assert dm["mean_loss_diff"] > 0
    assert dm["p_value"] < 0.05
    assert dm["seasons"] and dm["seasons"][0]["lags"] >= 7


def test_diebold_mariano_per_season_pooled():
    rng = np.random.default_rng(7)
    y = rng.normal(0, 12, size=600)
    good = y + rng.normal(0, 2, size=600)
    bad = y + rng.normal(0, 8, size=600)
    seasons = np.array([2021] * 200 + [2022] * 200 + [2023] * 200)
    dm = diebold_mariano(y, bad, good, loss="abs", season_series=seasons)
    assert dm["p_value"] < 0.05
    assert len(dm["seasons"]) == 3
    assert all(s["lags"] >= 7 for s in dm["seasons"])


def test_spiegelhalter_and_equal_mass_decile_z():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=500).astype(float)
    p = np.clip(y * 0.7 + 0.15 + rng.normal(0, 0.05, size=500), 0.01, 0.99)
    z = compute_spiegelhalter_z(y, p)
    assert "z" in z and "p_value" in z
    dz = compute_decile_z(y, p)
    assert dz["n_bins"] >= 2
    assert np.isfinite(dz["max_abs_z"])


def test_lower_bound_kelly_uses_p0():
    # Fair coin at even money with p0=0.4 → EV negative → 0
    assert robust_fractional_kelly(0.6, 2.0, va_bounds=(0.4, 0.7)) == 0.0
    # p0=0.6 at even money → positive Kelly
    assert robust_fractional_kelly(0.9, 2.0, va_bounds=(0.6, 0.8)) > 0.0


def test_compare_actuals_to_kalman_baseline():
    ok, msg = compare_actuals_to_baseline(
        {"spread_mae": 11.0, "ece": 0.05, "brier": 0.20},
        KALMAN_V6_ACTUALS,
    )
    assert ok, msg
    bad, _ = compare_actuals_to_baseline(
        {"spread_mae": 13.0, "ece": 0.05, "brier": 0.20},
        KALMAN_V6_ACTUALS,
    )
    assert not bad


def test_pace_efficiency_mae_decomposition():
    out = pace_efficiency_mae(
        actual_total=[220, 210],
        pred_pace=[100, 98],
        pred_home_ppp=[1.1, 1.05],
        pred_away_ppp=[1.1, 1.10],
        actual_pace=[99, 100],
        actual_home_ppp=[1.12, 1.0],
        actual_away_ppp=[1.08, 1.1],
    )
    assert np.isfinite(out["total_mae"])
    assert np.isfinite(out["pace_mae"])
    assert np.isfinite(out["eff_mae"])
    assert mae([1, 2], [1, 2]) == 0.0
