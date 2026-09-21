"""Hierarchical possession engine."""
from __future__ import annotations

import itertools
import math
from collections import defaultdict
from functools import lru_cache

from pipeline.config import DEFAULT_LEAGUE_RTG, OFFSEASON_REVERSION


@lru_cache(maxsize=250_000)
def _combos_cached(ids: tuple[int, ...]) -> dict[int, list[tuple[int, ...]]]:
    return {
        1: [(p,) for p in ids],
        2: list(itertools.combinations(ids, 2)),
        3: list(itertools.combinations(ids, 3)),
        5: [tuple(ids)] if len(ids) == 5 else [],
    }


def lineup_ids(lineup) -> tuple[int, ...]:
    """Sorted player ids for a lineup (shared by tuning prep + engine)."""
    return tuple(sorted(int(x) for x in lineup if x is not None and str(x) != "nan"))


def lineup_combos(lineup) -> dict[int, list[tuple[int, ...]]]:
    """Combo keys for a lineup; cached across engines/trials."""
    return _combos_cached(lineup_ids(lineup))


class HierarchicalPossessionEngine:
    def __init__(self, w1=0.50, w2=0.25, w3=0.15, w5=0.10,
                 k_off=2.0, k_def=1.5, league_avg_rtg=DEFAULT_LEAGUE_RTG,
                 home_boost_rtg=1.5, update_mode="points", min_poss_w5=200):
        self.update_mode = update_mode  # "points" or "xppp"
        self.min_poss_w5 = min_poss_w5
        self._base_w5 = w5
        total = w1 + w2 + w3 + w5
        self.W = {1: w1/total, 2: w2/total, 3: w3/total, 5: w5/total}
        self._w1, self._w2, self._w3 = w1, w2, w3
        self._combo_poss = defaultdict(float)
        # O(1) stand-in for max(_combo_poss[k] for 5-man keys) — same threshold logic.
        self._max_5man_poss = 0.0

        self.K_off = {1: k_off, 2: k_off*0.50, 3: k_off*0.25, 5: k_off*0.10}
        self.K_def = {1: k_def, 2: k_def*0.50, 3: k_def*0.25, 5: k_def*0.10}

        self.lg = league_avg_rtg
        self.home_boost_rtg = home_boost_rtg
        self.off = defaultdict(float)
        self.dff = defaultdict(float)

    def _combos(self, lineup):
        return lineup_combos(lineup)

    @staticmethod
    def _mean(combos, store, single_combos=None):
        if not combos:
            return 0.0
        raw = sum(store[c] for c in combos) / len(combos)
        if single_combos and len(combos) < 3:
            single = (
                sum(store[c] for c in single_combos) / len(single_combos)
                if single_combos else raw
            )
            w = len(combos) / 3.0
            return w * raw + (1.0 - w) * single
        return raw

    def _dynamic_weights(self):
        """Increase 5-man weight when combo samples are rich."""
        w5 = self._base_w5
        if self._max_5man_poss >= self.min_poss_w5:
            w5 = min(0.25, self._base_w5 * 1.8)
        total = self._w1 + self._w2 + self._w3 + w5
        return {1: self._w1/total, 2: self._w2/total, 3: self._w3/total, 5: w5/total}

    def lineup_rating(self, lineup):
        cb = self._combos(lineup)
        if not cb[1]:
            return 0.0, 0.0
        W = self._dynamic_weights()
        singles = cb[1]
        off = sum(W[l] * self._mean(cb[l], self.off, singles) for l in W)
        dff = sum(W[l] * self._mean(cb[l], self.dff, singles) for l in W)
        return off, dff

    def _predict_core(self, cb_off, cb_def, possessions):
        W = self._dynamic_weights()
        singles_off, singles_def = cb_off[1], cb_def[1]
        os = sum(W[l] * self._mean(cb_off[l], self.off, singles_off) for l in W)
        ds = sum(W[l] * self._mean(cb_def[l], self.dff, singles_def) for l in W)
        o2 = sum(W[l] * self._mean(cb_def[l], self.off, singles_def) for l in W)
        d2 = sum(W[l] * self._mean(cb_off[l], self.dff, singles_off) for l in W)
        p = possessions / 100.0
        xo = (self.lg + os - ds + self.home_boost_rtg) * p
        xd = (self.lg + o2 - d2) * p
        return xo, xd, W

    def predict_pts(self, off_ln, def_ln, possessions, cb_off=None, cb_def=None):
        if cb_off is None:
            cb_off = self._combos(off_ln)
        if cb_def is None:
            cb_def = self._combos(def_ln)
        xo, xd, _ = self._predict_core(cb_off, cb_def, possessions)
        return xo, xd, cb_off, cb_def

    def _apply_update(self, cb_off, cb_def, eo, ed, possessions):
        for l in (1, 2, 3, 5):
            ko, kd = self.K_off[l], self.K_def[l]
            for c in cb_off[l]:
                self.off[c] += ko * eo
                self.dff[c] -= kd * ed
                self._combo_poss[c] += possessions
                if l == 5 and self._combo_poss[c] > self._max_5man_poss:
                    self._max_5man_poss = self._combo_poss[c]
            for c in cb_def[l]:
                self.off[c] += ko * ed
                self.dff[c] -= kd * eo
                self._combo_poss[c] += possessions
                if l == 5 and self._combo_poss[c] > self._max_5man_poss:
                    self._max_5man_poss = self._combo_poss[c]

    def update(self, off_ln, def_ln, pts_off, pts_def, possessions,
               xpts_off=None, xpts_def=None, cb_off=None, cb_def=None):
        if not math.isfinite(possessions) or possessions <= 0:
            return
        if self.update_mode == "xppp" and xpts_off is not None:
            pts_off = xpts_off
            pts_def = xpts_def if xpts_def is not None else pts_def
        if cb_off is None:
            cb_off = self._combos(off_ln)
        if cb_def is None:
            cb_def = self._combos(def_ln)
        xo, xd, _ = self._predict_core(cb_off, cb_def, possessions)
        self._apply_update(cb_off, cb_def, pts_off - xo, pts_def - xd, possessions)

    def predict_and_update(self, off_ln, def_ln, pts_off, pts_def, possessions,
                           xpts_off=None, xpts_def=None, cb_off=None, cb_def=None):
        """One predict + state update (avoids double-predict used by eval loops)."""
        if not math.isfinite(possessions) or possessions <= 0:
            return 0.0, 0.0
        upd_off, upd_def = pts_off, pts_def
        if self.update_mode == "xppp" and xpts_off is not None:
            upd_off = xpts_off
            upd_def = xpts_def if xpts_def is not None else pts_def
        if cb_off is None:
            cb_off = self._combos(off_ln)
        if cb_def is None:
            cb_def = self._combos(def_ln)
        xo, xd, _ = self._predict_core(cb_off, cb_def, possessions)
        self._apply_update(cb_off, cb_def, upd_off - xo, upd_def - xd, possessions)
        return xo, xd

    def offseason_revert(self):
        for store in (self.off, self.dff):
            for k in list(store):
                store[k] *= (1.0 - OFFSEASON_REVERSION)

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls.__new__(cls)
        obj.__dict__.update(data)
        if not hasattr(obj, "_max_5man_poss"):
            combo_poss = getattr(obj, "_combo_poss", {}) or {}
            obj._max_5man_poss = max(
                (v for k, v in combo_poss.items() if len(k) == 5),
                default=0.0,
            )
        return obj
