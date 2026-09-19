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

import pytest

from pipeline.chemistry import ChemistryTracker
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

    for k in home_duos:
        assert tracker.duo_poss[k] > 0, f"home duo {k} not trained"
    for k in away_duos:
        assert tracker.duo_poss[k] > 0, f"away duo {k} not trained (P0.1 regression)"

    for k in _trio_keys(HOME):
        assert tracker.trio_poss[k] > 0
    for k in _trio_keys(AWAY):
        assert tracker.trio_poss[k] > 0, f"away trio {k} not trained (P0.1 regression)"


@pytest.mark.bench
def test_chemistry_on_off_written_for_both_sides():
    """on_off['on'] must be nonzero for both home and away players, and
    on_off['off'] (off-court-offense leg / points allowed on D) must now be
    written for both sides too."""
    tracker = ChemistryTracker(league_xppp=1.10)
    for _ in range(20):
        tracker.update_stint(HOME, AWAY, xpts_off=65.0, xpts_def=55.0, possessions=60.0)

    for pid in HOME + AWAY:
        st = tracker.on_off[pid]
        assert st["on_p"] > 0, f"{pid} on_p not populated"
        assert st["on"] > 0

    # on_off["off"] must have real accumulated possessions for every player
    # (each side spends time on defense too within the same stints).
    for pid in HOME + AWAY:
        st = tracker.on_off[pid]
        assert st["off_p"] > 0, f"{pid} off_p never written (P0.1 regression)"


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
    huge_n = 1_000_000.0
    shrunk = tracker._shrink(5.0, huge_n)
    assert shrunk == pytest.approx(5.0, rel=1e-3)


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

    key_home, key_away = elo._key(HOME), elo._key(AWAY)
    assert elo.off[key_home] != 0.0
    assert elo.off[key_away] != 0.0
    assert elo.off[key_home] != elo.off[key_away]
    assert elo.dff[key_home] != elo.dff[key_away]
