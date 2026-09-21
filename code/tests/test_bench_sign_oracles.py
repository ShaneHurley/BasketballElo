"""Epic 10 — 4-way O/D residual sign oracles.

Home/away symmetry can pass even when away offense is trained on ``-err_home``
and away defense on ``err_away``. These tests use a huge PPP gap (1.50 vs 0.70)
so a sign flip moves ratings by several Elo, not 1e-3.

Contract (``matchup_rating.residual_update_signs`` / lineup Elo / hierarchical):

- A offense += A's scoring residual
- B defense += minus A's scoring residual
- B offense += B's scoring residual
- A defense += minus B's scoring residual
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.lineup_elo import LineupEloTracker
from pipeline.matchup_rating import residual_update_signs
from pipeline.ratings import PlayerRatingTracker

HOME = ["h1", "h2", "h3", "h4", "h5"]
AWAY = ["a1", "a2", "a3", "a4", "a5"]
HOME_I = [1101, 1102, 1103, 1104, 1105]
AWAY_I = [2101, 2102, 2103, 2104, 2105]

PPP_HIGH = 1.50
PPP_LOW = 0.70
POSS = 200.0
MIN_DELTA = 5.0


def _player_tracker():
    return PlayerRatingTracker(config={"HOME_PPP_BOOST": 0.0}, league_xppp=1.10)


def _mean_od(tracker: PlayerRatingTracker, ids) -> tuple[float, float]:
    o = float(np.mean([tracker._get(p)["O_mu"] for p in ids]))
    d = float(np.mean([tracker._get(p)["D_mu"] for p in ids]))
    return o, d


@pytest.mark.bench
def test_residual_update_signs_contract():
    signs = residual_update_signs(PPP_HIGH, 1.10, PPP_LOW, 1.10)
    assert signs["home_offense_error"] > 0
    assert signs["away_offense_error"] < 0
    assert signs["away_defense_error_vs_home"] == pytest.approx(-signs["home_offense_error"])
    assert signs["home_defense_error_vs_away"] == pytest.approx(-signs["away_offense_error"])


@pytest.mark.bench
def test_player_elo_four_way_signs_home_blowout():
    """Home scores 1.50, away 0.70 — home O up, away D down, away O down, home D up."""
    tr = _player_tracker()
    tr.process_stint(
        HOME, AWAY, POSS, PPP_HIGH * POSS, PPP_LOW * POSS,
        period=1, season_progress=1.0, stint_ctx={"garbage": False, "clutch": False},
    )
    ho, hd = _mean_od(tr, HOME)
    ao, ad = _mean_od(tr, AWAY)
    assert ho - 1500.0 > MIN_DELTA, f"home O did not rise: {ho}"
    assert 1500.0 - ad > MIN_DELTA, f"away D did not fall after allowing 1.50: {ad}"
    assert 1500.0 - ao > MIN_DELTA, f"away O did not fall after scoring 0.70: {ao}"
    assert hd - 1500.0 > MIN_DELTA, f"home D did not rise after holding to 0.70: {hd}"


@pytest.mark.bench
def test_player_elo_four_way_signs_when_same_players_are_away():
    """The same blowout with sides swapped must move the same players the same way."""
    tr = _player_tracker()
    # Strong unit is AWAY, scores 1.50; weak unit is HOME, scores 0.70.
    tr.process_stint(
        HOME, AWAY, POSS, PPP_LOW * POSS, PPP_HIGH * POSS,
        period=1, season_progress=1.0, stint_ctx={"garbage": False, "clutch": False},
    )
    strong_o, strong_d = _mean_od(tr, AWAY)
    weak_o, weak_d = _mean_od(tr, HOME)
    assert strong_o - 1500.0 > MIN_DELTA, f"strong-as-away O did not rise: {strong_o}"
    assert 1500.0 - weak_d > MIN_DELTA, f"weak-as-home D did not fall: {weak_d}"
    assert 1500.0 - weak_o > MIN_DELTA, f"weak-as-home O did not fall: {weak_o}"
    assert strong_d - 1500.0 > MIN_DELTA, f"strong-as-away D did not rise: {strong_d}"


@pytest.mark.bench
def test_lineup_elo_four_way_signs():
    tracker = LineupEloTracker(league_xppp=1.10, home_boost=0.0)
    tracker.update_stint(HOME, AWAY, PPP_HIGH * POSS, PPP_LOW * POSS, POSS)
    kh, ka = tracker._key(HOME), tracker._key(AWAY)
    assert tracker.off[kh] > MIN_DELTA
    assert tracker.dff[ka] < -MIN_DELTA
    assert tracker.off[ka] < -MIN_DELTA
    assert tracker.dff[kh] > MIN_DELTA


@pytest.mark.bench
def test_hierarchical_four_way_signs():
    eng = HierarchicalPossessionEngine(home_boost_rtg=0.0, update_mode="points")
    eng.update(HOME_I, AWAY_I, PPP_HIGH * POSS, PPP_LOW * POSS, POSS)
    h5, a5 = tuple(sorted(HOME_I)), tuple(sorted(AWAY_I))
    assert eng.off[h5] > 0
    assert eng.dff[a5] < 0
    assert eng.off[a5] < 0
    assert eng.dff[h5] > 0


@pytest.mark.bench
def test_player_elo_sign_stable_across_possession_scale():
    """O/D delta signs must not flip when possessions go from 10 to 200."""
    signs = []
    for poss in (10.0, 50.0, 200.0):
        tr = _player_tracker()
        tr.process_stint(
            HOME, AWAY, poss, PPP_HIGH * poss, PPP_LOW * poss,
            period=1, season_progress=1.0, stint_ctx={"garbage": False, "clutch": False},
        )
        ho, hd = _mean_od(tr, HOME)
        ao, ad = _mean_od(tr, AWAY)
        signs.append((ho > 1500, ad < 1500, ao < 1500, hd > 1500))
    assert len(set(signs)) == 1
    assert signs[0] == (True, True, True, True)


@pytest.mark.bench
@pytest.mark.parametrize("ppp_h,ppp_a", [(0.0, 3.0), (3.0, 0.0)])
def test_player_elo_extreme_ppp_signs(ppp_h, ppp_a):
    tr = _player_tracker()
    tr.process_stint(
        HOME, AWAY, POSS, ppp_h * POSS, ppp_a * POSS,
        period=1, season_progress=1.0, stint_ctx={"garbage": False, "clutch": False},
    )
    ho, hd = _mean_od(tr, HOME)
    ao, ad = _mean_od(tr, AWAY)
    if ppp_h > ppp_a:
        assert ho > 1500 and ad < 1500
    else:
        assert ao > 1500 and hd < 1500
    for pid in HOME + AWAY:
        pl = tr._get(pid)
        assert math.isfinite(pl["O_mu"]) and math.isfinite(pl["D_mu"])


@pytest.mark.bench
@pytest.mark.parametrize("poss", [1e-6, 500.0])
def test_player_elo_extreme_possessions_keep_finite_signed_ratings(poss):
    tr = _player_tracker()
    tr.process_stint(
        HOME, AWAY, poss, PPP_HIGH * poss, PPP_LOW * poss,
        period=1, season_progress=1.0, stint_ctx={"garbage": False, "clutch": False},
    )
    ho, hd = _mean_od(tr, HOME)
    ao, ad = _mean_od(tr, AWAY)
    for pid in HOME + AWAY:
        pl = tr._get(pid)
        assert math.isfinite(pl["O_mu"]) and math.isfinite(pl["D_mu"])
    assert ho >= 1500.0 and ad <= 1500.0
    assert ao <= 1500.0 and hd >= 1500.0


@pytest.mark.bench
def test_player_elo_nonfinite_inputs_do_not_nan_ratings():
    tr = _player_tracker()
    tr.process_stint(
        HOME, AWAY, float("nan"), 110.0, 90.0,
        period=1, season_progress=1.0, stint_ctx={"garbage": False, "clutch": False},
    )
    tr.process_stint(
        HOME, AWAY, POSS, float("inf"), 90.0,
        period=1, season_progress=1.0, stint_ctx={"garbage": False, "clutch": False},
    )
    for pid in HOME + AWAY:
        pl = tr._get(pid)
        assert math.isfinite(pl["O_mu"]) and math.isfinite(pl["D_mu"])
        assert pl["O_mu"] == pytest.approx(1500.0)
        assert pl["D_mu"] == pytest.approx(1500.0)


@pytest.mark.bench
def test_neutral_stint_does_not_move_ratings():
    """League-average scoring on both sides, no HCA → stay at 1500."""
    tr = _player_tracker()
    tr.process_stint(
        HOME, AWAY, POSS, 1.10 * POSS, 1.10 * POSS,
        period=1, season_progress=1.0, stint_ctx={"garbage": False, "clutch": False},
    )
    for pid in HOME + AWAY:
        pl = tr._get(pid)
        assert pl["O_mu"] == pytest.approx(1500.0, abs=0.5)
        assert pl["D_mu"] == pytest.approx(1500.0, abs=0.5)


@pytest.mark.bench
def test_offense_and_defense_deltas_have_opposite_signs():
    """Home over-scoring must raise home O and lower away D (not both up)."""
    tr = _player_tracker()
    tr.process_stint(
        HOME, AWAY, POSS, PPP_HIGH * POSS, PPP_LOW * POSS,
        period=1, season_progress=1.0, stint_ctx={"garbage": False, "clutch": False},
    )
    ho, hd = _mean_od(tr, HOME)
    ao, ad = _mean_od(tr, AWAY)
    assert (ho - 1500.0) * (ad - 1500.0) < 0
    assert (ao - 1500.0) * (hd - 1500.0) < 0


@pytest.mark.bench
def test_three_engines_agree_on_four_way_signs():
    """Player Elo, lineup Elo, and hierarchical must share residual directions."""
    player = _player_tracker()
    lineup = LineupEloTracker(league_xppp=1.10, home_boost=0.0)
    hier = HierarchicalPossessionEngine(home_boost_rtg=0.0, update_mode="points")
    player.process_stint(
        HOME, AWAY, POSS, PPP_HIGH * POSS, PPP_LOW * POSS,
        period=1, season_progress=1.0, stint_ctx={"garbage": False, "clutch": False},
    )
    lineup.update_stint(HOME, AWAY, PPP_HIGH * POSS, PPP_LOW * POSS, POSS)
    hier.update(HOME_I, AWAY_I, PPP_HIGH * POSS, PPP_LOW * POSS, POSS)
    ho, hd = _mean_od(player, HOME)
    ao, ad = _mean_od(player, AWAY)
    kh, ka = lineup._key(HOME), lineup._key(AWAY)
    h5, a5 = tuple(sorted(HOME_I)), tuple(sorted(AWAY_I))
    player_signs = (ho > 1500, ad < 1500, ao < 1500, hd > 1500)
    lineup_signs = (lineup.off[kh] > 0, lineup.dff[ka] < 0, lineup.off[ka] < 0, lineup.dff[kh] > 0)
    hier_signs = (hier.off[h5] > 0, hier.dff[a5] < 0, hier.off[a5] < 0, hier.dff[h5] > 0)
    assert player_signs == lineup_signs == hier_signs == (True, True, True, True)


@pytest.mark.bench
def test_process_stint_source_keeps_four_way_away_signs():
    """Guard against re-crossing away O/D: away D uses -err_A, away O uses err_B."""
    import inspect

    src = inspect.getsource(PlayerRatingTracker.process_stint)
    assert "-err_A" in src
    assert "err_B * wt_b" in src
    assert "ids_B, -err_A" in src or "ids_B, -err_A * wt_b" in src
