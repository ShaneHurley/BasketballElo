"""Epic 10 / P0.1 — home-only lineup training bug.

GitHub: P0.1 · Registry ID: ``asymmetric_lineup_training_bias``

``ChemistryTracker`` and ``LineupEloTracker`` are fed one ``update_stint``
call per stint from ``pipeline/game_updates.py`` with the home lineup as
``off_ids`` and the away lineup as ``def_ids``. Before the fix:

- ``ChemistryTracker.update_stint`` only accumulated ``duo_off``/``trio_off``
  and ``on_off["on"]`` for the *offense* argument (home). The away lineup's
  duo/trio residuals and on-court history were never written, and
  ``on_off["off"]`` (points allowed while on defense) was never written for
  *any* player.
- ``LineupEloTracker.update_stint`` only ever updated ``self.off[key_off]``
  and dumped ``-err*0.5`` (half of the offense's own error) into
  ``self.dff[key_off]`` — the away lineup's ``off``/``dff`` were never
  touched, and the "defensive" update wasn't a real defensive signal at all.

Both trackers are now internally symmetric (single stint call trains both
directions), mirroring ``pipeline/hierarchical.py``'s ``_apply_update``.

These tests assert the away side is populated exactly like the home side
would be, that ``on_off["off"]`` is written, and that the shrinkage helpers
behave correctly at the 0-possession and huge-n limits.
"""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
import pytest

from pipeline.chemistry import MIN_DUO_POSS, SHRINK, ChemistryTracker
from pipeline.lineup_elo import LineupEloTracker

HOME = ["h1", "h2", "h3", "h4", "h5"]
AWAY = ["a1", "a2", "a3", "a4", "a5"]


def _duo_keys(ids):
    return {tuple(sorted(p)) for p in itertools.combinations(sorted(str(x) for x in ids), 2)}


def _trio_keys(ids):
    return {tuple(sorted(p)) for p in itertools.combinations(sorted(str(x) for x in ids), 3)}


# ---------------------------------------------------------------------------
# ChemistryTracker
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_chemistry_mirrored_update_populates_away_duo_and_trio():
    """P0.1 acceptance: away-only duo/trio pairs must get real state, not
    just whatever overlaps with the home side (which would be none here,
    since HOME and AWAY share no player ids)."""
    tracker = ChemistryTracker(league_xppp=1.10)
    for _ in range(20):
        tracker.update_stint(HOME, AWAY, xpts_off=60.0, xpts_def=55.0, possessions=60.0)

    home_duos = _duo_keys(HOME)
    away_duos = _duo_keys(AWAY)
    assert home_duos.isdisjoint(away_duos)

    total_poss = 20 * 60.0
    for k in home_duos:
        assert tracker.duo_poss[k] == total_poss, f"home duo {k} undertrained"
        assert tracker.duo_off[k] != 0.0, f"home duo {k} residual never written"
    for k in away_duos:
        # Exact equality with the home tally: a partial mirror (e.g. away
        # credited half possessions) must fail, not merely "> 0".
        assert tracker.duo_poss[k] == total_poss, f"away duo {k} not trained (P0.1 regression)"
        assert tracker.duo_off[k] != 0.0, f"away duo {k} residual never written (P0.1 regression)"

    for k in _trio_keys(HOME):
        assert tracker.trio_poss[k] == total_poss
        assert tracker.trio_off[k] != 0.0
    for k in _trio_keys(AWAY):
        assert tracker.trio_poss[k] == total_poss, f"away trio {k} not trained (P0.1 regression)"
        assert tracker.trio_off[k] != 0.0, f"away trio {k} residual never written (P0.1 regression)"


@pytest.mark.bench
def test_chemistry_on_off_written_for_both_sides():
    """on_off['on'] must be nonzero for both home and away players, and
    on_off['off'] (off-court-offense leg / points allowed on D) must now be
    written for both sides too."""
    tracker = ChemistryTracker(league_xppp=1.10)
    for _ in range(20):
        tracker.update_stint(HOME, AWAY, xpts_off=65.0, xpts_def=55.0, possessions=60.0)

    total_poss = 20 * 60.0
    for pid in HOME + AWAY:
        st = tracker.on_off[pid]
        assert st["on_p"] == total_poss, f"{pid} on_p not fully mirrored"
        assert st["on"] > 0

    # on_off["off"] must have real accumulated possessions for every player
    # (each side spends time on defense too within the same stints), and the
    # points recorded there must be what the *opponent* scored: home's off
    # leg sees away's 55/60 ppp, away's off leg sees home's 65/60 ppp. A fix
    # that writes off_p but fills it with the player's own scoring fails here.
    for pid in HOME + AWAY:
        st = tracker.on_off[pid]
        assert st["off_p"] == total_poss, f"{pid} off_p never written (P0.1 regression)"
    for pid in HOME:
        st = tracker.on_off[pid]
        assert st["off"] / st["off_p"] == pytest.approx(55.0 / 60.0)
    for pid in AWAY:
        st = tracker.on_off[pid]
        assert st["off"] / st["off_p"] == pytest.approx(65.0 / 60.0)


@pytest.mark.bench
def test_chemistry_onoff_net_uses_on_minus_off_when_warm():
    """Once both legs have enough possessions, lineup_chemistry's on/off
    component should reflect on_ppp - off_ppp, not just on_ppp vs a static
    league baseline."""
    tracker = ChemistryTracker(league_xppp=1.10)
    # Home offense heavily outperforms while away offense underperforms,
    # so home players should show a strongly positive on/off net and away
    # players a strongly negative one.
    for _ in range(40):
        tracker.update_stint(HOME, AWAY, xpts_off=90.0, xpts_def=30.0, possessions=60.0)

    _, _, h_onoff, _ = tracker.lineup_chemistry(HOME)
    _, _, a_onoff, _ = tracker.lineup_chemistry(AWAY)
    assert h_onoff > 0
    assert a_onoff < 0
    assert h_onoff > a_onoff


@pytest.mark.bench
def test_chemistry_shrink_zero_possessions_and_huge_n_limits():
    tracker = ChemistryTracker()
    assert tracker._shrink(5.0, 0) == 0.0
    assert tracker._shrink(5.0, -3) == 0.0
    # 100k possessions: shrunk value within 1% of raw.
    assert tracker._shrink(5.0, 100_000.0) == pytest.approx(5.0, rel=1e-2)
    huge_n = 1_000_000.0
    shrunk = tracker._shrink(5.0, huge_n)
    assert shrunk == pytest.approx(5.0, rel=1e-3)


@pytest.mark.bench
def test_chemistry_duo_net_min_duo_poss_threshold():
    """MIN_DUO_POSS gate in _duo_net/_trio_net: a duo below 50 possessions is
    excluded entirely — 0.0 regardless of how extreme its raw residual is —
    while exactly 50 enters with shrinkage, and 100k converges to raw."""
    tracker = ChemistryTracker(league_xppp=1.10)
    duo = ("p1", "p2")

    # 49 possessions (< MIN_DUO_POSS): excluded no matter how large |raw|.
    # A shrink-only gate (no threshold) would return raw * 49/(49+SHRINK) != 0.
    tracker.duo_off[duo] = 1e6
    tracker.duo_poss[duo] = float(MIN_DUO_POSS - 1)
    assert tracker._duo_net(["p1", "p2"]) == (0.0, 0.0)
    tracker.duo_off[duo] = -1e6
    assert tracker._duo_net(["p1", "p2"]) == (0.0, 0.0)

    # Same gate applies to trios.
    trio = ("p1", "p2", "p3")
    tracker.trio_off[trio] = 1e6
    tracker.trio_poss[trio] = float(MIN_DUO_POSS - 1)
    assert tracker._trio_net(["p1", "p2", "p3"]) == (0.0, 0.0)

    # Exactly MIN_DUO_POSS: included, shrunk by n/(n+SHRINK).
    tracker.duo_off[duo] = 5.0
    tracker.duo_poss[duo] = float(MIN_DUO_POSS)
    val, w = tracker._duo_net(["p1", "p2"])
    assert val == pytest.approx(5.0 * MIN_DUO_POSS / (MIN_DUO_POSS + SHRINK))
    assert w == pytest.approx(float(MIN_DUO_POSS))

    # 100,000 possessions: shrunk value within 1% of raw.
    tracker.duo_poss[duo] = 100_000.0
    val, _ = tracker._duo_net(["p1", "p2"])
    assert val == pytest.approx(5.0, rel=1e-2)

    # 0 possessions (duo never seen): zero.
    assert tracker._duo_net(["x1", "x2"]) == (0.0, 0.0)


# ---------------------------------------------------------------------------
# LineupEloTracker
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_lineup_elo_mirrored_update_trains_away_lineup():
    """P0.1 acceptance: the away 5-man unit must accumulate its own off/dff
    state, not just whatever the home unit happened to write."""
    tracker = LineupEloTracker(league_xppp=1.10)
    key_home = tracker._key(HOME)
    key_away = tracker._key(AWAY)
    for _ in range(20):
        tracker.update_stint(HOME, AWAY, xpts_off=65.0, xpts_def=55.0, possessions=60.0)

    assert tracker.poss[key_home] > 0
    assert tracker.poss[key_away] > 0, "away lineup poss never trained (P0.1 regression)"
    assert key_home in tracker.off
    assert key_away in tracker.off, "away lineup off never trained (P0.1 regression)"
    assert key_home in tracker.dff
    assert key_away in tracker.dff, "away lineup dff never trained (P0.1 regression)"


@pytest.mark.bench
def test_lineup_elo_defense_trained_from_opponent_scoring_not_own_offense():
    """dff must move opposite to what the *opponent* scored, and must not
    simply be a fixed fraction of the same lineup's own offensive error."""
    tracker = LineupEloTracker(league_xppp=1.10)
    key_home = tracker._key(HOME)
    key_away = tracker._key(AWAY)

    # Home offense wildly overperforms (high err_a); away offense is exactly
    # at league average (err_b == 0). If dff were still being computed from
    # the *offense's own* error (old bug: dff[key_off] += k*(-err*0.5)),
    # home's dff would move due to its own huge offensive error. Instead it
    # must be away's dff that moves (away allowed the huge overperformance),
    # while home's dff should only move in response to away's (zero) error.
    for _ in range(10):
        tracker.update_stint(
            HOME, AWAY, xpts_off=120.0, xpts_def=66.0, possessions=60.0,
        )

    assert tracker.dff[key_away] < 0, "away conceded a lot but dff did not drop"
    assert tracker.dff[key_home] == pytest.approx(0.0, abs=1e-9), (
        "home dff moved from its own offensive error instead of away's "
        "(unchanged, exactly-average) scoring — P0.1 regression"
    )


@pytest.mark.bench
def test_lineup_elo_shrink_zero_possessions_and_huge_n_limits():
    """James-Stein-style shrinkage in lineup_rating: 0 possessions must fall
    back fully to the player-implied prior (w=0); huge n must converge to
    the tracker's own raw stored off/dff (w->1)."""
    tracker = LineupEloTracker(league_xppp=1.10)
    key = tracker._key(HOME)

    # No data at all -> falls back to player-implied prior scaled by 0.1.
    off0, dff0, chem0, n0 = tracker.lineup_rating(HOME)
    assert n0 == 0.0
    assert off0 == pytest.approx(0.0)
    assert dff0 == pytest.approx(0.0)

    # Huge n -> weight w = n / (n + SHRINK_K) ~= 1, so lineup_rating should
    # converge to the raw stored off/dff values, not the player prior.
    tracker.off[key] = 42.0
    tracker.dff[key] = -17.0
    tracker.chemistry[key] = 3.0
    tracker.poss[key] = 1_000_000.0
    off1, dff1, chem1, n1 = tracker.lineup_rating(HOME)
    assert off1 == pytest.approx(42.0, rel=1e-3)
    assert dff1 == pytest.approx(-17.0, rel=1e-3)


# ---------------------------------------------------------------------------
# Adversarial: catch a "fix" that mirrors on_off writes but still only
# trains home for the actual rating stores (duo_off/off/dff).
# ---------------------------------------------------------------------------

@pytest.mark.bench
def test_adversarial_away_and_home_produce_independent_nonzero_ratings():
    """A shallow fix could write on_off state for both sides while leaving
    duo_off/off/dff still keyed only off `off_ids` (home). Guard against
    that by asserting home and away end up with *different*, independently
    meaningful raw ratings driven by their own distinct scoring lines."""
    chem = ChemistryTracker(league_xppp=1.10)
    elo = LineupEloTracker(league_xppp=1.10)

    # Home scores well above expectation; away scores well below.
    for _ in range(30):
        chem.update_stint(HOME, AWAY, xpts_off=90.0, xpts_def=30.0, possessions=60.0)
        elo.update_stint(HOME, AWAY, xpts_off=90.0, xpts_def=30.0, possessions=60.0)

    home_duo, _ = chem._duo_net(HOME)
    away_duo, _ = chem._duo_net(AWAY)
    assert home_duo != 0.0
    assert away_duo != 0.0
    assert home_duo != away_duo
    # Sign check: a direction-swapped mirror (away trained on home's scoring
    # line) still yields nonzero, unequal away values and would pass the
    # assertions above — only the correct signs prove each side was trained
    # on its OWN scoring line (home +0.4/poss residual, away -0.6/poss).
    assert home_duo > 0 > away_duo

    key_home, key_away = elo._key(HOME), elo._key(AWAY)
    assert elo.off[key_home] != 0.0
    assert elo.off[key_away] != 0.0
    assert elo.off[key_home] != elo.off[key_away]
    assert elo.dff[key_home] != elo.dff[key_away]
    assert elo.off[key_home] > 0 > elo.off[key_away]
    # Home defense looked good (away scored 0.5 ppp); away defense bad.
    assert elo.dff[key_home] > 0 > elo.dff[key_away]


@pytest.mark.bench
def test_adversarial_away_state_proportional_to_away_appearances():
    """Wrong-reason guard under rotation: with lineups resampled every stint,
    each duo's stored possessions must equal exactly the possessions that duo
    actually spent on the floor — not zero (P0.1), not the other side's
    tally, not a flat per-stint credit, and no phantom duo keys. Same for
    every player's on_p/off_p."""
    rng = np.random.default_rng(20260919)
    home_pool = [f"h{i}" for i in range(8)]
    away_pool = [f"a{i}" for i in range(8)]
    tracker = ChemistryTracker(league_xppp=1.10)
    exp_duo_poss = defaultdict(float)
    exp_on_p = defaultdict(float)
    exp_off_p = defaultdict(float)
    n_stints = 0
    for _ in range(60):
        home = sorted(str(x) for x in rng.choice(home_pool, size=5, replace=False))
        away = sorted(str(x) for x in rng.choice(away_pool, size=5, replace=False))
        poss = float(rng.integers(20, 81))
        x_off = round(float(rng.normal(1.12, 0.15)) * poss, 3)
        x_def = round(float(rng.normal(1.08, 0.15)) * poss, 3)
        tracker.update_stint(home, away, xpts_off=x_off, xpts_def=x_def, possessions=poss)
        n_stints += 1
        for k in _duo_keys(home) | _duo_keys(away):
            exp_duo_poss[k] += poss
        for pid in home + away:
            # Every player is on offense for one leg and defense for the
            # other within the same stint, so both counters get `poss`.
            exp_on_p[pid] += poss
            exp_off_p[pid] += poss

    assert n_stints == 60 and exp_duo_poss, "rng produced no stints — vacuous test"
    for k, exp_poss in exp_duo_poss.items():
        assert tracker.duo_poss[k] == pytest.approx(exp_poss), (
            f"duo {k}: stored {tracker.duo_poss[k]} vs actual floor time {exp_poss}"
        )
    # No duo key outside the observed set may exist (no phantom writes).
    assert set(tracker.duo_poss) == set(exp_duo_poss)
    for pid in home_pool + away_pool:
        st = tracker.on_off[pid]
        assert st["on_p"] == pytest.approx(exp_on_p[pid]), f"{pid} on_p mismatch"
        assert st["off_p"] == pytest.approx(exp_off_p[pid]), f"{pid} off_p mismatch (P0.1 regression)"
