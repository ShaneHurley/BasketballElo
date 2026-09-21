"""Epic 10.2.1 — PlayerRatingTracker formula-invariant battery.

Synthetic-bench invariants for ``pipeline/ratings.py::PlayerRatingTracker``:

1. **Home/away symmetry** — two identical stints with the teams swapped must
   produce mirrored player ratings (tracker 1's home == tracker 2's away).
2. **RD bounded** — every RD field stays inside ``[rd_floor, rd_cap]`` across
   500 stints plus inactivity-decay pulses that slam O/D RD to ``default_rd``.
3. **K-decay monotonicity** — the effective Kalman×k_mult recovered from applied
   rating deltas is non-increasing over 50 games.
4. **Garbage-weight idempotence (P0.8)** — ``_weight_stint`` applies the
   garbage-time discount exactly once; repeated identical calls agree.

Plus the P0.5 invariant "RD responds to outcome surprise" (Epic 9.1 Kalman).
"""
from __future__ import annotations

import datetime
import math

import numpy as np
import pytest

from pipeline.ratings import KALMAN_R_PER_POSS, PlayerRatingTracker
from tests.synth.factories import make_player_ids, make_stint, make_stint_frame
from tests.synth.presets import BLOWOUT_60, SINGLE_LINEUP_ALL_SEASON


def _feed(tracker, row, *, usage_A=None, usage_B=None, season_progress=0.5, stint_ctx=None):
    """Push one ``make_stint`` row through ``PlayerRatingTracker.process_stint``."""
    ids_A = [p for p in str(row["HOME_players"]).split("-") if p]
    ids_B = [p for p in str(row["AWAY_players"]).split("-") if p]
    ctx = {"garbage": bool(row["garbage"])} if stint_ctx is None else dict(stint_ctx)
    tracker.process_stint(
        ids_A,
        ids_B,
        float(row["possessions"]),
        float(row["home_xpts"]),
        float(row["away_xpts"]),
        usage_A=usage_A,
        usage_B=usage_B,
        period=int(row["PERIOD"]),
        start_A=float(row["HOME_SCORE_START"]),
        start_B=float(row["AWAY_SCORE_START"]),
        end_A=float(row["HOME_SCORE_END"]),
        end_B=float(row["AWAY_SCORE_END"]),
        season_progress=season_progress,
        stint_ctx=ctx,
    )
    return ids_A, ids_B


@pytest.mark.bench
def test_home_away_symmetry_swapped_stints_mirror():
    """Invariant 1 — home/away symmetry.

    Two identical stints with the teams swapped are fed to two fresh
    trackers; tracker 1's home-player ratings must equal tracker 2's
    away-player ratings (and vice versa) to within float tolerance.

    Two asymmetries are *intentional* and are neutralized via config so the
    mirror is exact:

    * ``HOME_PPP_BOOST`` — home-court advantage exists precisely to break
      home/away symmetry, so it is set to 0.0 here;
    * ``player_games`` increments once per *sub-update* (off, def_rim,
      def_peri, def), so the off-leg and def-legs of the same stint see
      slightly different ``k_mult`` values; ``k_mult_half_life=1e12``
      compresses that intra-stint decay below float noise.

    Non-uniform usage weights and a swapped PBP context (turnover penalty on
    one side, foul-draw boost on the other, asymmetric rim/3pa credit rates)
    make the mirror non-trivial: every context key travels with its team.
    """
    rng = np.random.default_rng(20261021)
    home_ids = make_player_ids(5, start=1, prefix="h")
    away_ids = make_player_ids(5, start=6, prefix="a")

    def _usage(ids):
        return {
            p: (
                float(rng.uniform(0.5, 2.0)),
                float(rng.uniform(0.0, 1.0)),
                float(rng.uniform(0.0, 1.0)),
            )
            for p in ids
        }

    usage_home = _usage(home_ids)  # usage travels with the team across the swap
    usage_away = _usage(away_ids)

    poss = 2.0
    # home_ppp + away_ppp == 2 * league_xppp keeps the swapped expectations mirrored.
    home_xpts, away_xpts = 2.3, 2.1  # sums to 4.4 == 2 * 1.10 * poss
    base = make_stint(
        home_players=home_ids, away_players=away_ids, possessions=poss,
        home_xpts=home_xpts, away_xpts=away_xpts, period=1,
    )
    swapped = make_stint(
        home_players=away_ids, away_players=home_ids, possessions=poss,
        home_xpts=away_xpts, away_xpts=home_xpts, period=1,
    )

    ctx_base = {
        "garbage": False, "clutch": False,
        "home_tov_rate": 0.20, "away_foul_draw_rate": 0.15,
        "away_rim_rate": 0.25, "away_3pa_rate": 0.30,
    }
    ctx_swap = {
        "garbage": False, "clutch": False,
        "away_tov_rate": 0.20, "home_foul_draw_rate": 0.15,
        "home_rim_rate": 0.25, "home_3pa_rate": 0.30,
    }

    cfg = {"HOME_PPP_BOOST": 0.0, "k_mult_half_life": 1e12}
    t1 = PlayerRatingTracker(config=dict(cfg))
    t2 = PlayerRatingTracker(config=dict(cfg))

    _feed(t1, base, usage_A=usage_home, usage_B=usage_away, stint_ctx=ctx_base)
    _feed(t2, swapped, usage_A=usage_away, usage_B=usage_home, stint_ctx=ctx_swap)

    rating_fields = (
        "O_mu", "D_mu", "D_rim_mu", "D_peri_mu",
        "O_rd", "D_rd", "D_rim_rd", "D_peri_rd",
        "O_sigma", "D_sigma", "Possessions",
    )
    for pid in home_ids + away_ids:
        assert t1.player_games[pid] == t2.player_games[pid]
        for field in rating_fields:
            mirrored = t1.players[pid][field]
            assert mirrored == pytest.approx(t2.players[pid][field], rel=1e-9, abs=1e-12), (
                f"home/away mirror broke for player {pid} field {field}: "
                f"{mirrored} != {t2.players[pid][field]}"
            )

    # Non-vacuousness: the stint actually moved ratings — a no-op
    # process_stint would mirror trivially at the 1500.0 defaults.
    assert t1.players[home_ids[0]]["O_mu"] != pytest.approx(1500.0)
    assert t1.players[home_ids[0]]["O_rd"] < 350.0


@pytest.mark.bench
def test_rd_bounded_after_500_stints():
    """Invariant 2 — RD stays inside [rd_floor, rd_cap] after 500 stints.

    500 randomized stints (same 10 players all season, per the
    SINGLE_LINEUP_ALL_SEASON preset) partially decay every RD; three
    inactivity-decay pulses a year apart then slam O_rd/D_rd into the
    ``default_rd`` inactivity ceiling. Kalman process noise plus occasional
    shocks keep a noise floor above ``rd_floor``; every field must still sit
    inside ``[rd_floor, rd_cap]``, RD must have shrunk well below the prior,
    and inactivity must have restored O/D RD to the prior ceiling.
    """
    tracker = PlayerRatingTracker()
    rd_floor = float(tracker.cfg["rd_floor"])
    rd_cap = float(tracker.cfg["rd_cap"])
    rd_prior = float(tracker.cfg["default_rd"])
    lineup = SINGLE_LINEUP_ALL_SEASON
    all_ids = list(lineup["home_players"]) + list(lineup["away_players"])
    frame = make_stint_frame(
        n=500,
        seed=20261021,
        home_players=lineup["home_players"],
        away_players=lineup["away_players"],
    )
    rd_fields = ("O_rd", "D_rd", "D_rim_rd", "D_peri_rd")

    def _rd_extremes():
        vals = [tracker._get(p)[f] for p in all_ids for f in rd_fields]
        return min(vals), max(vals)

    # NB: pass datetime.date, matching the production caller
    # (game_features.py passes gdate.date()); a raw pd.Timestamp input makes
    # the days-since-last subtraction raise a swallowed TypeError and the
    # inflation silently no-ops.
    day0 = datetime.date(2025, 11, 1)
    tracker.apply_inactivity_decay(all_ids, day0)  # seeds last_date; no inflation yet

    lo_seen, hi_seen = math.inf, -math.inf
    for i, row in enumerate(frame.to_dict("records")):
        _feed(tracker, row)
        lo, hi = _rd_extremes()
        lo_seen, hi_seen = min(lo_seen, lo), max(hi_seen, hi)
        if i == 99:  # mid-decay; year-apart pulses slam the ceiling clamp.
            for extra_days in (365, 730, 1095):
                tracker.apply_inactivity_decay(all_ids, day0 + datetime.timedelta(days=extra_days))
                lo, hi = _rd_extremes()
                lo_seen, hi_seen = min(lo_seen, lo), max(hi_seen, hi)

    for pid in all_ids:
        for field in rd_fields:
            value = tracker.players[pid][field]
            assert math.isfinite(value)
            assert rd_floor <= value <= rd_cap, f"{pid}.{field} escaped bounds: {value}"

    # Non-vacuousness: RD collapsed below the prior, and inactivity/Kalman
    # restored the high side to at least default_rd.
    assert lo_seen >= rd_floor
    assert lo_seen < rd_prior
    assert hi_seen >= rd_prior


@pytest.mark.bench
def test_k_decay_monotonic_over_50_games():
    """Invariant 3 — effective K is non-increasing over 50 games.

    The effective K applied to one player is recovered from the *observed*
    rating delta, ``k = ΔO_mu / (err_A · wt · share)`` — a behavioural probe
    rather than a formula re-implementation — and each recovered value is
    cross-checked against the closed form ``K_kalman(rd, poss) ·
    k_mult(games)`` anchored to the tracker's own state, so the series being
    asserted non-increasing is the same object the update rule applies.
    """
    tracker = PlayerRatingTracker()
    home = make_player_ids(5, start=1, prefix="h")
    away = make_player_ids(5, start=6, prefix="a")
    target = home[0]

    poss, home_xpts, away_xpts = 1.0, 2.0, 1.0  # err_A ≈ +0.9, bounded away from 0
    season_progress = 0.5
    share = 1.0 / len(home)  # no usage dict → equal shares
    wt = poss * (0.7 + 0.3 * season_progress)  # period=1: no garbage/clutch/ctx multipliers

    ks = []
    for i in range(50):
        off_a = tracker.lineup_stats(home, opp_rim_rate=0.30, opp_three_rate=0.38)[0]
        def_b = tracker.lineup_stats(away, opp_rim_rate=0.30, opp_three_rate=0.38)[1]
        exp_ppp_a = (
            tracker.league_xppp
            + tracker.cfg["HOME_PPP_BOOST"]
            + (off_a - def_b) / tracker.cfg["ELO_SCALING_FACTOR"]
        )
        err_a = home_xpts / poss - exp_ppp_a
        assert abs(err_a) > 0.1  # keep the recovered K well-conditioned

        rd_before = tracker._get(target)["O_rd"]
        games_before = tracker.player_games[target]
        mu_before = tracker._get(target)["O_mu"]

        row = make_stint(
            game_id=f"kd{i:02d}", home_players=home, away_players=away,
            possessions=poss, home_xpts=home_xpts, away_xpts=away_xpts,
            period=1, stint_id=i,
        )
        _feed(tracker, row, season_progress=season_progress)

        mu_after = tracker._get(target)["O_mu"]
        k_applied = (mu_after - mu_before) / (err_a * wt * share)
        p_var = rd_before * rd_before
        r_obs = float(tracker.cfg.get("kalman_r_per_poss", KALMAN_R_PER_POSS)) / max(poss, 1e-9)
        gain = p_var / (p_var + r_obs)
        k_expected = gain * max(
            0.5,
            2.0 * math.exp(-games_before / max(tracker.cfg["k_mult_half_life"], 1.0)),
        )
        assert k_applied == pytest.approx(k_expected, rel=1e-9)
        ks.append(k_applied)

    ks = np.asarray(ks)
    assert np.all(np.isfinite(ks))
    assert np.all(np.diff(ks) <= 1e-12), f"effective K increased: {np.diff(ks).max()}"
    assert ks[1] < ks[0]  # non-vacuous: K genuinely decays, not merely constant


@pytest.mark.bench
def test_garbage_weight_idempotent_p0_8():
    """Invariant 4 — garbage-weight idempotence (P0.8).

    P0.8 (registry ``bench_p0_garbage_weight``): ``_weight_stint`` used to
    run an unguarded ``if abs(margin) >= 25: weight *= gt_w`` after the
    garbage/period ``elif`` chain, so a stint that was already flagged
    ``garbage=True`` *and* had a blowout margin got the garbage-time
    discount applied twice. Post-fix, identical calls with identical inputs
    must return the identical weight — exactly one ``gt_w`` application,
    never two — because the function is pure and the discount fires once.
    """
    tracker = PlayerRatingTracker()
    gt_w = tracker.cfg["garbage_time_weight"]
    preset = BLOWOUT_60  # period=4, garbage=True, in-stint margin only 10
    preset_margin = float(preset["HOME_SCORE_END"]) - float(preset["HOME_SCORE_START"])

    cases = [
        # Flag-only path: margin below the elif threshold, garbage flag set.
        (float(preset["possessions"]), int(preset["PERIOD"]), preset_margin, {"garbage": True}),
        # P0.8 regression path: garbage flag AND |margin| >= 25 — pre-fix code
        # applied the discount twice here.
        (100.0, 4, 30.0, {"garbage": True}),
    ]
    for poss, period, margin, ctx in cases:
        w1 = tracker._weight_stint(poss, period, margin, 1.0, ctx)
        w2 = tracker._weight_stint(poss, period, margin, 1.0, ctx)
        assert w1 == w2, "identical inputs must yield identical weights (no state, no re-discount)"
        assert w1 == pytest.approx(poss * gt_w, rel=1e-9), "exactly one garbage-time discount"
        assert w1 != pytest.approx(poss * gt_w * gt_w, rel=1e-9), "discount must not double-apply"


@pytest.mark.bench
@pytest.mark.parametrize(
    "period,margin,garbage,expect_discounted",
    [
        (4, 16.0, False, True),    # adversarial: elif must fire (15 <= 16 < 25)
        (4, 15.0, False, True),    # margin threshold is inclusive
        (4, 14.9, False, False),   # just below the threshold
        (4, -16.0, False, True),   # abs(margin): away-side blowout also fires
        (4, 10.0, True, True),     # flag fires below the margin threshold (BLOWOUT_60 shape)
        (4, 30.0, True, True),     # flag + blowout: exactly one discount (P0.8 regression)
        (5, 20.0, False, True),    # OT period still satisfies period >= 4
        (2, 30.0, False, False),   # adversarial: old unguarded |margin| >= 25 must NOT fire
        (1, 30.0, False, False),   # early-period blowout without flag → full weight
        (2, 30.0, True, True),     # flag path is period-independent
    ],
)
def test_weight_stint_garbage_branch_matrix(period, margin, garbage, expect_discounted):
    """Adversarial branch matrix for ``_weight_stint`` (P0.8).

    Pins the post-fix branch structure: the discount fires iff
    ``stint_ctx["garbage"]`` or ``period >= 4 and abs(margin) >= 15``, and
    the pre-fix unguarded ``abs(margin) >= 25`` branch must stay dead —
    verified by the ``period=2, margin=30`` case, which must keep full
    weight. ``season_progress=1.0`` makes the always-on season factor
    exactly 1.0 so the expectations are exact.
    """
    tracker = PlayerRatingTracker()
    gt_w = tracker.cfg["garbage_time_weight"]
    poss = 100.0
    weight = tracker._weight_stint(poss, period, margin, 1.0, {"garbage": garbage})
    expected = poss * gt_w if expect_discounted else poss
    assert weight == pytest.approx(expected, rel=1e-9)
    if expect_discounted:
        assert weight != pytest.approx(poss * gt_w * gt_w, rel=1e-9)


@pytest.mark.bench
def test_rd_responds_to_outcome_surprise():
    """P0.5 — RD must respond to outcome surprise, not just update count.

    A shocking upset stint should not shrink RD more than an expected-result
    stint — and Kalman uncertainty should *grow* after a shock, because the
    model just learned its prior was wrong.
    """
    t_exp = PlayerRatingTracker()
    t_shock = PlayerRatingTracker()
    league_xppp = t_exp.league_xppp
    h_boost = t_exp.cfg["HOME_PPP_BOOST"]
    rd0 = t_exp.cfg["default_rd"]

    # All-default lineups ⇒ expected ppp is league_xppp ± home boost, so this
    # stint's outcome error is ~0 (fully expected result)...
    expected_stint = make_stint(
        possessions=1.0, home_xpts=league_xppp + h_boost, away_xpts=league_xppp - h_boost, period=1,
    )
    # ...and this one is a shocking blowout no rating saw coming (abs_err >> 0.15 cap).
    shock_stint = make_stint(possessions=1.0, home_xpts=5.0, away_xpts=0.05, period=1)
    _feed(t_exp, expected_stint)
    _feed(t_shock, shock_stint)

    pid = "1"  # default home player, present in both trackers
    for field in ("O_rd", "D_rd"):
        rd_expected = t_exp.players[pid][field]
        rd_shock = t_shock.players[pid][field]
        # Weak half (holds today): the shock shrinks RD no more than the
        # expected result does.
        assert rd_shock >= rd_expected
        # Kalman half (fails today): a shock must *increase* uncertainty.
        assert rd_shock > rd0
