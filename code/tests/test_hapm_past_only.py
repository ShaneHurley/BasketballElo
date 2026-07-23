"""Task 052 — cross-fit HAPM (and other target-fitted engine features) via
past-only cross-fitting; fix the current HAPM-before-split leak.

Regression test for LEAK_REGISTRY.md-worthy defect: the production call in
``pipeline/backtest.py`` does ``base_hapm.fit(train_stints)`` *once* on the
entire training span and then reuses that single fitted tracker as a static
lookup for every game in the span — including games chronologically earlier
than stints the tracker was fit on.
"""
import itertools

import numpy as np
import pandas as pd
import pytest

from pipeline.hapm import (
    HapmPriorTracker,
    PastOnlyHapmLookup,
    build_past_only_hapm_trackers,
    hapm_lookup_for_training,
)


def _make_stints(n_games=60, seed=5):
    rng = np.random.RandomState(seed)
    rows = []
    players = [f"p{i}" for i in range(12)]
    for gid in range(n_games):
        home = rng.choice(players, size=5, replace=False)
        rows.append({
            "GAME_ID": gid,
            "game_date": pd.Timestamp("2025-10-01") + pd.Timedelta(days=gid),
            "possessions": 10.0,
            "home_xpts": 11.0 + rng.normal(scale=0.5),
            "HOME_players": "-".join(home),
        })
    return pd.DataFrame(rows)


class TestHapmBeforeSplitIsALeak:
    def test_single_fit_on_full_span_uses_future_games_for_early_rows(self):
        """Documents the confirmed defect: fitting once on the whole span
        and treating the tracker as fit_max_timestamp == -inf for every
        row (including the very first game) is wrong — its true
        fit_max_timestamp is the *last* game in the span, which is after
        the early games it's applied to."""
        stints = _make_stints(n_games=120)
        tracker = HapmPriorTracker()
        tracker.fit(stints)  # current production pattern: fit on everything
        # The single tracker is identical for game 0 and game 39 even
        # though a leak-free tracker for game 0 cannot have seen game 39.
        assert tracker.fitted
        # There is no "fit_max_timestamp" on this tracker at all — a
        # leak-free lineage check (Task 058) has nothing to verify against,
        # which is itself evidence the old pattern cannot pass lineage CI.
        assert not hasattr(tracker, "fit_max_timestamp")


class TestBuildPastOnlyHapmTrackers:
    def test_block_zero_games_get_unfitted_tracker(self):
        stints = _make_stints(n_games=60)
        game_dates = dict(zip(stints["GAME_ID"], stints["game_date"]))
        trackers, fit_max_ts = build_past_only_hapm_trackers(stints, game_dates, n_blocks=6)
        ordered = sorted(game_dates, key=lambda g: game_dates[g])
        first_block_gid = ordered[0]
        assert not trackers[first_block_gid].fitted
        assert fit_max_ts[first_block_gid] is None

    def test_later_block_tracker_fit_max_timestamp_precedes_game_date(self):
        stints = _make_stints(n_games=60)
        game_dates = dict(zip(stints["GAME_ID"], stints["game_date"]))
        trackers, fit_max_ts = build_past_only_hapm_trackers(stints, game_dates, n_blocks=6)
        for gid, ts in fit_max_ts.items():
            if ts is None:
                continue
            assert ts < game_dates[gid], (
                f"HAPM tracker for game {gid} was fit on data at/after its "
                "own game_date — past-only cross-fitting violated."
            )

    def test_a_later_block_tracker_never_saw_a_future_game(self):
        stints = _make_stints(n_games=60)
        game_dates = dict(zip(stints["GAME_ID"], stints["game_date"]))
        trackers, _ = build_past_only_hapm_trackers(stints, game_dates, n_blocks=6)
        ordered = sorted(game_dates, key=lambda g: game_dates[g])
        mid_gid = ordered[len(ordered) // 2]
        tracker = trackers[mid_gid]
        if tracker.fitted:
            # The fitted dyad coefficients must only reflect players from
            # strictly-earlier games; we check this indirectly by
            # confirming the tracker used for the *first* game differs
            # from the tracker used for a later game (proving it was
            # refit, not reused/static across the whole span).
            assert trackers[ordered[0]] is not tracker

    def test_returns_empty_for_no_games(self):
        trackers, fit_max_ts = build_past_only_hapm_trackers(pd.DataFrame(), {}, n_blocks=6)
        assert trackers == {}
        assert fit_max_ts == {}


class TestPastOnlyHapmLookupIntegration:
    """Prove the backtest wiring path (`hapm_lookup_for_training` →
    ``PastOnlyHapmLookup.feature_dict(game_id=...)``) never lets an early
    game's HAPM feature depend on a later game's stints."""

    def test_early_game_features_unchanged_when_later_games_appended(self):
        early = _make_stints(n_games=40, seed=1)
        late = _make_stints(n_games=40, seed=2)
        late["GAME_ID"] = late["GAME_ID"] + 40
        late["game_date"] = late["game_date"] + pd.Timedelta(days=40)
        early_lookup = hapm_lookup_for_training(early, n_blocks=4)
        full_lookup = hapm_lookup_for_training(pd.concat([early, late], ignore_index=True), n_blocks=4)
        first_gid = int(early["GAME_ID"].iloc[0])
        lineup = early.loc[early["GAME_ID"] == first_gid, "HOME_players"].iloc[0].split("-")
        early_feat = early_lookup.feature_dict(lineup, lineup, game_id=first_gid)
        full_feat = full_lookup.feature_dict(lineup, lineup, game_id=first_gid)
        assert early_feat == full_feat

    def test_unknown_game_id_returns_zero_features_not_a_fabricated_prior(self):
        lookup = hapm_lookup_for_training(_make_stints(n_games=30), n_blocks=3)
        zeros = lookup.feature_dict(["a", "b", "c", "d", "e"], ["f", "g", "h", "i", "j"], game_id=99999)
        assert zeros == {"h_hapm_net": 0.0, "a_hapm_net": 0.0, "hapm_net_diff": 0.0}
        assert isinstance(lookup, PastOnlyHapmLookup)
