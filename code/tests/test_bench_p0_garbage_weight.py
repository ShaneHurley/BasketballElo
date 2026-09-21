"""Epic 11.2 / P0.8 — Fix _weight_stint garbage-time double-apply.

GitHub: P0.8 · Registry ID: ``bench_p0_garbage_weight``

``PlayerRatingTracker._weight_stint`` previously had an unguarded
``if abs(margin) >= 25: weight *= gt_w`` after the garbage/period elif chain,
so a Q4 blowout stint flagged ``garbage=True`` got the garbage-time weight
multiplier applied *twice*. The fix restructures the logic into a single
``if/elif`` chain so at most one garbage-time discount is ever applied per
stint, and non-garbage/non-blowout stints (Q1, or Q4 with a modest margin)
get no discount at all.
"""
from __future__ import annotations

import pytest

from pipeline.ratings import PlayerRatingTracker


def _tracker():
    return PlayerRatingTracker()


@pytest.mark.bench
def test_p0_8_q1_blowout_no_discount():
    """Q1 margin=30: not period>=4, no garbage flag ⇒ no discount at all."""
    tracker = _tracker()
    poss = 100.0
    weight = tracker._weight_stint(poss, period=1, margin=30, season_progress=1.0, stint_ctx=None)
    assert weight == pytest.approx(poss, rel=1e-9)


@pytest.mark.bench
def test_p0_8_q4_garbage_flag_applies_exactly_once():
    """Q4 margin=20, garbage=True ⇒ exactly one gt_w application."""
    tracker = _tracker()
    poss = 100.0
    gt_w = tracker.cfg.get("garbage_time_weight", 0.5)
    weight = tracker._weight_stint(
        poss, period=4, margin=20, season_progress=1.0, stint_ctx={"garbage": True}
    )
    expected_single = poss * gt_w
    expected_double = poss * gt_w * gt_w
    assert weight == pytest.approx(expected_single, rel=1e-9)
    assert weight != pytest.approx(expected_double, rel=1e-6)


@pytest.mark.bench
def test_p0_8_q4_extreme_blowout_garbage_not_double_applied():
    """Q4 margin=30 (>=25), garbage=True ⇒ still exactly one application, not two.

    This is the exact regression case for P0.8: before the fix, the
    unguarded ``if abs(margin) >= 25`` line ran *in addition to* the
    garbage-flag branch, discounting the stint's weight twice.
    """
    tracker = _tracker()
    poss = 100.0
    gt_w = tracker.cfg.get("garbage_time_weight", 0.5)
    weight = tracker._weight_stint(
        poss, period=4, margin=30, season_progress=1.0, stint_ctx={"garbage": True}
    )
    expected_single = poss * gt_w
    expected_double = poss * gt_w * gt_w
    assert weight == pytest.approx(expected_single, rel=1e-9)
    assert weight != pytest.approx(expected_double, rel=1e-6)
    # Same value regardless of whether margin crosses the old 25-point line.
    weight_20 = tracker._weight_stint(
        poss, period=4, margin=20, season_progress=1.0, stint_ctx={"garbage": True}
    )
    assert weight == pytest.approx(weight_20, rel=1e-9)


@pytest.mark.bench
def test_p0_8_q2_blowout_no_garbage_flag_never_discounted():
    """Adversarial regression case: Q2 margin=30, no garbage flag.

    Before the fix, the unguarded ``if abs(margin) >= 25`` line fired here
    even though period < 4 and no garbage flag was set — discounting a
    legitimate early-game blowout stint. It must NOT be discounted.
    """
    tracker = _tracker()
    poss = 100.0
    weight = tracker._weight_stint(poss, period=2, margin=30, season_progress=1.0, stint_ctx=None)
    assert weight == pytest.approx(poss, rel=1e-9)


@pytest.mark.bench
def test_p0_8_q4_modest_margin_no_discount():
    """Q4 margin=10, no garbage flag ⇒ below the period>=4 margin>=15 threshold, no discount."""
    tracker = _tracker()
    poss = 100.0
    weight = tracker._weight_stint(poss, period=4, margin=10, season_progress=1.0, stint_ctx=None)
    assert weight == pytest.approx(poss, rel=1e-9)


@pytest.mark.bench
def test_process_stint_garbage_discounted_once():
    """Live-path P0.8 follow-up: ``process_stint`` must not re-apply ``gt_w``.

    ``_weight_stint`` already discounts garbage stints once. ``_context_multiplier``
    used to multiply ``garbage_time_weight`` again, so a flagged stint was
    weighted at ``gt_w²`` (0.09 at the configured 0.30). The live applied
    weight is ``_weight_stint * _context_multiplier``; with other context
    boosts neutralized that product must equal one ``gt_w``, not two.
    """
    tracker = PlayerRatingTracker()
    gt_w = tracker.cfg["garbage_time_weight"]
    tracker.cfg["clutch_boost"] = 1.0
    tracker.cfg["tov_penalty"] = 1.0
    tracker.cfg["foul_draw_boost"] = 1.0
    tracker.cfg["variance_dampen"] = 1.0
    tracker.cfg["k_def_events"] = 0.0

    captured: dict[str, float] = {}
    orig_weight = tracker._weight_stint
    orig_ctx = tracker._context_multiplier

    def wrap_weight(*args, **kwargs):
        captured["wt"] = orig_weight(*args, **kwargs)
        return captured["wt"]

    def wrap_ctx(*args, **kwargs):
        m = orig_ctx(*args, **kwargs)
        captured["ctx"] = m
        return m

    tracker._weight_stint = wrap_weight
    tracker._context_multiplier = wrap_ctx

    home = [f"h{i}" for i in range(5)]
    away = [f"a{i}" for i in range(5)]
    poss = 100.0
    tracker.process_stint(
        home,
        away,
        poss,
        110.0,
        90.0,
        period=4,
        start_A=0,
        start_B=20,
        end_A=20,
        end_B=45,
        season_progress=1.0,
        stint_ctx={"garbage": True, "clutch": False},
    )
    applied = captured["wt"] * captured["ctx"]
    assert captured["wt"] == pytest.approx(poss * gt_w, rel=1e-9)
    assert captured["ctx"] == pytest.approx(1.0, rel=1e-9)
    assert applied == pytest.approx(poss * gt_w, rel=1e-9)
    assert applied != pytest.approx(poss * gt_w * gt_w, rel=1e-6)
