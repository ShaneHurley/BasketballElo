"""Parity / speed-path tests for hierarchical engine and incremental tuning CV."""
from __future__ import annotations

import numpy as np
import pytest

from pipeline.config import TUNING_INVALID_SCORE
from pipeline.hierarchical import HierarchicalPossessionEngine, lineup_combos
from pipeline.tuning import (
    _hier_eval_fold,
    _hier_warmup,
    _season_walkforward_mae,
    _season_walkforward_mae_incremental,
)


def _lineup(seed, n=5):
    rng = np.random.default_rng(seed)
    return [int(x) for x in rng.integers(1, 500, size=n)]


def _fake_records(n, seed=0):
    rng = np.random.default_rng(seed)
    recs = []
    for i in range(n):
        hp, ap = _lineup(seed + i * 2), _lineup(seed + i * 2 + 1)
        poss = float(rng.integers(2, 12))
        home_pts = float(rng.uniform(1.5, 4.0) * poss / 2)
        away_pts = float(rng.uniform(1.5, 4.0) * poss / 2)
        recs.append({
            "hp": hp,
            "ap": ap,
            "hp_cb": lineup_combos(hp),
            "ap_cb": lineup_combos(ap),
            "poss": poss,
            "home_pts": home_pts,
            "away_pts": away_pts,
            "home_xpts": home_pts * 0.98,
            "away_xpts": away_pts * 0.98,
            "season": 2020 + (i // max(1, n // 3)),
        })
    return recs


class TestHierarchicalSpeedPathParity:
    def test_max_5man_matches_scan(self):
        eng = HierarchicalPossessionEngine(min_poss_w5=50)
        for rec in _fake_records(80, seed=7):
            eng.update(
                rec["hp"], rec["ap"], rec["home_pts"], rec["away_pts"], rec["poss"],
                cb_off=rec["hp_cb"], cb_def=rec["ap_cb"],
            )
        scanned = max(
            (v for k, v in eng._combo_poss.items() if len(k) == 5),
            default=0.0,
        )
        assert eng._max_5man_poss == pytest.approx(scanned)

    def test_predict_and_update_matches_predict_then_update(self):
        a = HierarchicalPossessionEngine(w1=0.4, w2=0.2, w3=0.2, w5=0.2, k_off=0.3, k_def=0.2)
        b = HierarchicalPossessionEngine(w1=0.4, w2=0.2, w3=0.2, w5=0.2, k_off=0.3, k_def=0.2)
        for rec in _fake_records(40, seed=11):
            xo_a, xd_a, _, _ = a.predict_pts(
                rec["hp"], rec["ap"], rec["poss"],
                cb_off=rec["hp_cb"], cb_def=rec["ap_cb"],
            )
            a.update(
                rec["hp"], rec["ap"], rec["home_pts"], rec["away_pts"], rec["poss"],
                cb_off=rec["hp_cb"], cb_def=rec["ap_cb"],
            )
            xo_b, xd_b = b.predict_and_update(
                rec["hp"], rec["ap"], rec["home_pts"], rec["away_pts"], rec["poss"],
                cb_off=rec["hp_cb"], cb_def=rec["ap_cb"],
            )
            assert xo_a == pytest.approx(xo_b)
            assert xd_a == pytest.approx(xd_b)
        # State after many steps should match (keys + values).
        assert set(a.off) == set(b.off)
        assert set(a.dff) == set(b.dff)
        for k in a.off:
            assert a.off[k] == pytest.approx(b.off[k])
            assert a.dff[k] == pytest.approx(b.dff[k])
        assert a._max_5man_poss == pytest.approx(b._max_5man_poss)

    def test_dynamic_weights_threshold_identical(self):
        eng = HierarchicalPossessionEngine(w5=0.1, min_poss_w5=30)
        ln = [1, 2, 3, 4, 5]
        cb = lineup_combos(ln)
        for _ in range(10):
            eng.update(ln, ln, 10.0, 9.0, 5.0, cb_off=cb, cb_def=cb)
        W = eng._dynamic_weights()
        assert eng._max_5man_poss >= 30
        boosted = min(0.25, 0.1 * 1.8)
        total = 0.5 + 0.25 + 0.15 + boosted
        assert W[5] == pytest.approx(boosted / total)


class TestIncrementalSeasonCVParity:
    def test_incremental_matches_from_scratch_hier(self):
        recs = _fake_records(90, seed=21)
        by_season = {}
        for r in recs:
            by_season.setdefault(r["season"], []).append(r)
        seasons = sorted(by_season.keys())
        assert len(seasons) >= 2

        params = dict(w1=0.5, w2=0.2, w3=0.15, w5=0.15, k_off=0.25, k_def=0.2, league_rtg=112.0)

        def make_model():
            return HierarchicalPossessionEngine(
                params["w1"], params["w2"], params["w3"], params["w5"],
                k_off=params["k_off"], k_def=params["k_def"],
                league_avg_rtg=params["league_rtg"],
            )

        def fold_fn(train_recs, val_recs):
            eng = make_model()
            _hier_warmup(eng, train_recs)
            return _hier_eval_fold(eng, val_recs)

        scratch = _season_walkforward_mae(seasons, by_season, fold_fn)
        incr = _season_walkforward_mae_incremental(
            seasons, by_season, make_model, _hier_warmup, _hier_eval_fold,
        )
        assert scratch != TUNING_INVALID_SCORE
        assert incr == pytest.approx(scratch, rel=1e-12, abs=1e-12)
