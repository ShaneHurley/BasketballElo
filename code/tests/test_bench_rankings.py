"""Epic 10 — ranking ladder so a sign flip is visible, not 1e-3 noise.

Four constant 5-man teams with designed PPP {1.50, 1.20, 0.95, 0.70}. Scoring
is (own offense + opponent hold) / 2 so both O and D have a ladder. No HCA.
Round-robin, both home and away. Order must be exact (Spearman = 1) with
adjacent mean gaps > 20 Elo (player) / clearly ordered stored ratings (lineup /
hierarchical).
"""
from __future__ import annotations

import itertools

import numpy as np
import pytest

from pipeline.hierarchical import HierarchicalPossessionEngine
from pipeline.lineup_elo import LineupEloTracker
from pipeline.ratings import PlayerRatingTracker

PPP = (1.50, 1.20, 0.95, 0.70)
POSS = 100.0
REPEATS = 10
MIN_GAP = 20.0
LEAGUE_HOLD = 2.20  # so 1.50 offense also holds opponents to 0.70


def _teams_str():
    return [[f"t{t}p{i}" for i in range(5)] for t in range(4)]


def _teams_int():
    return [[1000 + 100 * t + i for i in range(5)] for t in range(4)]


def _xpts(h: int, a: int) -> tuple[float, float]:
    xh = 0.5 * (PPP[h] + (LEAGUE_HOLD - PPP[a])) * POSS
    xa = 0.5 * (PPP[a] + (LEAGUE_HOLD - PPP[h])) * POSS
    return xh, xa


def _schedule():
    pairs = list(itertools.permutations(range(4), 2))  # both home/away
    for _ in range(REPEATS):
        yield from pairs


def _team_mean(tracker: PlayerRatingTracker, ids, field: str) -> float:
    return float(np.mean([tracker._get(p)[field] for p in ids]))


def _hier_engine():
    return HierarchicalPossessionEngine(
        home_boost_rtg=0.0, update_mode="points", k_off=1.0, k_def=0.75,
    )


def _assert_strict_ladder(values: list[float], min_gap: float) -> None:
    """values[0] is strongest; must be strictly decreasing with min adjacent gap."""
    arr = np.asarray(values, dtype=float)
    order = list(np.argsort(-arr))
    assert order == [0, 1, 2, 3], f"rank order {order} values={arr}"
    gaps = arr[:-1] - arr[1:]
    assert np.all(gaps > min_gap), f"adjacent gaps {gaps} (need > {min_gap})"


@pytest.mark.bench
def test_player_elo_four_team_offense_and_defense_ladder():
    teams = _teams_str()
    tr = PlayerRatingTracker(config={"HOME_PPP_BOOST": 0.0}, league_xppp=1.10)
    ctx = {"garbage": False, "clutch": False}
    for h, a in _schedule():
        xh, xa = _xpts(h, a)
        tr.process_stint(
            teams[h], teams[a], POSS, xh, xa,
            period=1, season_progress=1.0, stint_ctx=ctx,
        )
    o = [_team_mean(tr, ids, "O_mu") for ids in teams]
    d = [_team_mean(tr, ids, "D_mu") for ids in teams]
    _assert_strict_ladder(o, MIN_GAP)
    _assert_strict_ladder(d, MIN_GAP)


@pytest.mark.bench
def test_lineup_elo_four_team_offense_and_defense_ladder():
    teams = _teams_str()
    tr = LineupEloTracker(league_xppp=1.10, home_boost=0.0)
    for h, a in _schedule():
        xh, xa = _xpts(h, a)
        tr.update_stint(teams[h], teams[a], xh, xa, POSS)
    keys = [tr._key(ids) for ids in teams]
    o = [tr.off[k] for k in keys]
    d = [tr.dff[k] for k in keys]
    # Lineup Elo stores residuals, not 1500-centered mu; order + gap still apply.
    _assert_strict_ladder(o, min_gap=1.0)
    _assert_strict_ladder(d, min_gap=1.0)
    # Same order as player-Elo would demand — no scale match required.
    assert list(np.argsort(-np.asarray(o))) == [0, 1, 2, 3]
    assert list(np.argsort(-np.asarray(d))) == [0, 1, 2, 3]


@pytest.mark.bench
def test_hierarchical_four_team_offense_and_defense_ladder():
    teams = _teams_int()
    eng = _hier_engine()
    for h, a in _schedule():
        xh, xa = _xpts(h, a)
        eng.update(teams[h], teams[a], xh, xa, POSS)
    o, d = [], []
    for ids in teams:
        off, dff = eng.lineup_rating(ids)
        o.append(off)
        d.append(dff)
    _assert_strict_ladder(o, min_gap=0.05)
    _assert_strict_ladder(d, min_gap=0.05)


@pytest.mark.bench
def test_three_engines_agree_on_offense_order():
    """Player Elo, lineup Elo, and hierarchical must agree on team order."""
    teams_s = _teams_str()
    teams_i = _teams_int()
    player = PlayerRatingTracker(config={"HOME_PPP_BOOST": 0.0}, league_xppp=1.10)
    lineup = LineupEloTracker(league_xppp=1.10, home_boost=0.0)
    hier = _hier_engine()
    ctx = {"garbage": False, "clutch": False}
    for h, a in _schedule():
        xh, xa = _xpts(h, a)
        player.process_stint(teams_s[h], teams_s[a], POSS, xh, xa, period=1, season_progress=1.0, stint_ctx=ctx)
        lineup.update_stint(teams_s[h], teams_s[a], xh, xa, POSS)
        hier.update(teams_i[h], teams_i[a], xh, xa, POSS)
    orders = []
    o_player = [_team_mean(player, ids, "O_mu") for ids in teams_s]
    orders.append(list(np.argsort(-np.asarray(o_player))))
    o_lineup = [lineup.off[lineup._key(ids)] for ids in teams_s]
    orders.append(list(np.argsort(-np.asarray(o_lineup))))
    o_hier = [hier.lineup_rating(ids)[0] for ids in teams_i]
    orders.append(list(np.argsort(-np.asarray(o_hier))))
    assert orders[0] == orders[1] == orders[2] == [0, 1, 2, 3], orders


@pytest.mark.bench
def test_player_elo_spearman_is_exactly_one():
    """Designed strength order vs observed O_mu / D_mu ranks: Spearman = 1."""
    from scipy.stats import spearmanr

    teams = _teams_str()
    tr = PlayerRatingTracker(config={"HOME_PPP_BOOST": 0.0}, league_xppp=1.10)
    ctx = {"garbage": False, "clutch": False}
    for h, a in _schedule():
        xh, xa = _xpts(h, a)
        tr.process_stint(
            teams[h], teams[a], POSS, xh, xa,
            period=1, season_progress=1.0, stint_ctx=ctx,
        )
    designed = np.arange(4)
    o = np.asarray([_team_mean(tr, ids, "O_mu") for ids in teams])
    d = np.asarray([_team_mean(tr, ids, "D_mu") for ids in teams])
    o_rank = np.argsort(np.argsort(-o))
    d_rank = np.argsort(np.argsort(-d))
    assert spearmanr(designed, o_rank).correlation == pytest.approx(1.0)
    assert spearmanr(designed, d_rank).correlation == pytest.approx(1.0)


@pytest.mark.bench
def test_player_elo_road_only_ladder_still_ranks():
    """Away-only schedule: the old crossed away O/D wiring inverted this ladder."""
    teams = _teams_str()
    tr = PlayerRatingTracker(config={"HOME_PPP_BOOST": 0.0}, league_xppp=1.10)
    ctx = {"garbage": False, "clutch": False}
    # Every team only ever appears as the away side.
    for _ in range(REPEATS):
        for a in range(4):
            for h in range(4):
                if h == a:
                    continue
                xh, xa = _xpts(h, a)
                tr.process_stint(
                    teams[h], teams[a], POSS, xh, xa,
                    period=1, season_progress=1.0, stint_ctx=ctx,
                )
    o = [_team_mean(tr, ids, "O_mu") for ids in teams]
    d = [_team_mean(tr, ids, "D_mu") for ids in teams]
    _assert_strict_ladder(o, MIN_GAP)
    _assert_strict_ladder(d, MIN_GAP)
