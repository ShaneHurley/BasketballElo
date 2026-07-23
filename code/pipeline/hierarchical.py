"""Hierarchical possession engine."""
import itertools
from collections import defaultdict

import numpy as np

from pipeline.config import DEFAULT_LEAGUE_RTG, OFFSEASON_REVERSION


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

        self.K_off = {1: k_off, 2: k_off*0.50, 3: k_off*0.25, 5: k_off*0.10}
        self.K_def = {1: k_def, 2: k_def*0.50, 3: k_def*0.25, 5: k_def*0.10}

        self.lg = league_avg_rtg
        self.home_boost_rtg = home_boost_rtg
        self.off = defaultdict(float)
        self.dff = defaultdict(float)

    def _combos(self, lineup):
        ids = sorted(int(x) for x in lineup if x is not None and str(x) != "nan")
        return {
            1: [(p,) for p in ids],
            2: list(itertools.combinations(ids, 2)),
            3: list(itertools.combinations(ids, 3)),
            5: [tuple(ids)] if len(ids) == 5 else [],
        }

    def _mean(self, combos, store, single_combos=None):
        if not combos:
            return 0.0
        raw = float(np.mean([store[c] for c in combos]))
        if single_combos and len(combos) < 3:
            single = float(np.mean([store[c] for c in single_combos])) if single_combos else raw
            w = len(combos) / 3.0
            return w * raw + (1.0 - w) * single
        return raw

    def _dynamic_weights(self):
        """Increase 5-man weight when combo samples are rich."""
        w5 = self._base_w5
        if self._combo_poss:
            max5 = max((v for k, v in self._combo_poss.items() if len(k) == 5), default=0)
            if max5 >= self.min_poss_w5:
                w5 = min(0.25, self._base_w5 * 1.8)
        total = self._w1 + self._w2 + self._w3 + w5
        return {1: self._w1/total, 2: self._w2/total, 3: self._w3/total, 5: w5/total}

    def lineup_rating(self, lineup):
        cb = self._combos(lineup)
        if not cb[1]: return 0.0, 0.0
        W = self._dynamic_weights()
        singles = cb[1]
        off = sum(W[l] * self._mean(cb[l], self.off, singles) for l in W)
        dff = sum(W[l] * self._mean(cb[l], self.dff, singles) for l in W)
        return off, dff

    def predict_pts(self, off_ln, def_ln, possessions):
        cb_off = self._combos(off_ln); cb_def = self._combos(def_ln)
        W = self._dynamic_weights()
        singles_off, singles_def = cb_off[1], cb_def[1]
        os = sum(W[l] * self._mean(cb_off[l], self.off, singles_off) for l in W)
        ds = sum(W[l] * self._mean(cb_def[l], self.dff, singles_def) for l in W)
        o2 = sum(W[l] * self._mean(cb_def[l], self.off, singles_def) for l in W)
        d2 = sum(W[l] * self._mean(cb_off[l], self.dff, singles_off) for l in W)
        p = possessions / 100.0
        return (self.lg + os - ds + self.home_boost_rtg) * p, (self.lg + o2 - d2) * p, cb_off, cb_def

    def update(self, off_ln, def_ln, pts_off, pts_def, possessions, xpts_off=None, xpts_def=None):
        if possessions <= 0: return
        if self.update_mode == "xppp" and xpts_off is not None:
            pts_off = xpts_off
            pts_def = xpts_def if xpts_def is not None else pts_def
        xo, xd, cb_off, cb_def = self.predict_pts(off_ln, def_ln, possessions)

        eo, ed = pts_off - xo, pts_def - xd

        W = self._dynamic_weights()
        for l in [1, 2, 3, 5]:
            ko, kd = self.K_off[l], self.K_def[l]
            for c in cb_off[l]:
                self.off[c] += ko * eo
                self.dff[c] -= kd * ed
                self._combo_poss[c] += possessions
            for c in cb_def[l]:
                self.off[c] += ko * ed
                self.dff[c] -= kd * eo
                self._combo_poss[c] += possessions

    def offseason_revert(self):
        for store in (self.off, self.dff):
            for k in list(store): store[k] *= (1.0 - OFFSEASON_REVERSION)


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
        return obj
