"""Epic 10.6 / P0.6 & Epic 11.5 — HAPM shrinkage must use real possession counts.

GitHub: P0.6 · Registry ID: ``bench_hapm_fake_shrink``

``HapmPriorTracker._lineup_prior`` multiplied every dyad/trio coefficient by
the *same* constant ``SHRINK / (SHRINK + 50)`` regardless of how many
possessions actually supported that coefficient — a duo observed for 10
possessions was trusted exactly as much as one observed for 5000. The fix
tracks per-key possession counts during ``fit`` and shrinks each
coefficient by the empirical-Bayes weight ``n / (n + SHRINK)``.
"""
from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

from pipeline.hapm import SHRINK, HapmPriorTracker


def _make_stints(pairs_and_poss, xh_offset=0.05, n_solo_players=6):
    """Build a synthetic stints_df where each home lineup features a
    designated duo playing together for a controlled number of total
    possessions (split across many small stints so `fit`'s >=100-row floor
    is satisfied), plus filler players to keep lineups 5-wide.
    """
    rows = []
    filler = [f"F{i}" for i in range(n_solo_players)]
    for (a, b), total_poss in pairs_and_poss:
        n_stints = max(1, min(100, int(total_poss)))
        poss_each = total_poss / n_stints
        for i in range(n_stints):
            lineup = [a, b] + filler[:3]
            rows.append({
                "HOME_players": "-".join(lineup),
                "possessions": poss_each,
                "home_xpts": poss_each * (1.10 + xh_offset),
            })
    return pd.DataFrame(rows)


@pytest.mark.bench
def test_p0_6_high_possession_duo_shrinks_less_than_low_possession_duo():
    """Acceptance: 5000-poss duo shrinks less than a 10-poss duo."""
    stints = _make_stints([
        (("HIGH1", "HIGH2"), 5000.0),
        (("LOW1", "LOW2"), 10.0),
    ])
    tracker = HapmPriorTracker(alpha=100.0)
    assert tracker.fit(stints)

    high_key = "HIGH1|HIGH2"
    low_key = "LOW1|LOW2"
    assert high_key in tracker._dyad_coef
    assert low_key in tracker._dyad_coef

    high_shrink = tracker._shrink_factor(tracker._dyad_poss[high_key])
    low_shrink = tracker._shrink_factor(tracker._dyad_poss[low_key])

    assert high_shrink > low_shrink
    # Explicit empirical-Bayes formula check.
    assert high_shrink == pytest.approx(5000.0 / (5000.0 + SHRINK))
    assert low_shrink == pytest.approx(10.0 / (10.0 + SHRINK))


@pytest.mark.bench
def test_p0_6_shrink_factor_not_constant_across_possession_counts():
    """Witness: old code used one constant factor regardless of possession count."""
    old_constant_factor = SHRINK / (SHRINK + 50)
    tracker = HapmPriorTracker()
    f_10 = tracker._shrink_factor(10.0)
    f_5000 = tracker._shrink_factor(5000.0)
    assert f_10 != pytest.approx(old_constant_factor, abs=1e-6) or f_5000 != pytest.approx(
        old_constant_factor, abs=1e-6
    )
    assert f_10 != pytest.approx(f_5000)


@pytest.mark.bench
def test_p0_6_lineup_prior_uses_possession_weighted_shrink():
    """The final lineup_prior for the high-possession duo should move further
    from zero (less shrunk) than the low-possession duo, given equal raw
    coefficients magnitude order (both duos share the same underlying signal
    strength by construction of _make_stints)."""
    stints = _make_stints([
        (("HIGH1", "HIGH2"), 5000.0),
        (("LOW1", "LOW2"), 10.0),
    ])
    tracker = HapmPriorTracker(alpha=100.0)
    assert tracker.fit(stints)

    prior_high = tracker._lineup_prior(["HIGH1", "HIGH2", "X1", "X2", "X3"])
    prior_low = tracker._lineup_prior(["LOW1", "LOW2", "X1", "X2", "X3"])

    coef_high = tracker._dyad_coef["HIGH1|HIGH2"]
    coef_low = tracker._dyad_coef["LOW1|LOW2"]

    # Effective shrink ratio applied (prior / raw coef * 100) must track
    # possession-based shrink factor, not a shared constant.
    shrink_high = tracker._shrink_factor(tracker._dyad_poss["HIGH1|HIGH2"])
    shrink_low = tracker._shrink_factor(tracker._dyad_poss["LOW1|LOW2"])
    assert prior_high == pytest.approx(coef_high * shrink_high * 100.0)
    assert prior_low == pytest.approx(coef_low * shrink_low * 100.0)
    assert shrink_high > shrink_low


@pytest.mark.bench
def test_p0_6_zero_possession_key_shrinks_to_zero():
    tracker = HapmPriorTracker()
    assert tracker._shrink_factor(0.0) == pytest.approx(0.0)
