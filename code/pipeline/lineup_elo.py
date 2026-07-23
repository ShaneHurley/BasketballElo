"""Dedicated 5-man lineup Elo with shrinkage toward player-weighted mean."""
from __future__ import annotations

from collections import defaultdict

import numpy as np

MIN_POSSESSIONS = 50
SHRINK_K = 100.0


class LineupEloTracker:
    """Stores offensive/defensive rating per sorted 5-tuple with James-Stein shrinkage."""

    def __init__(self, league_xppp: float = 1.10, scaling: float = 1000.0, home_boost: float = 0.024):
        self.league_xppp = league_xppp
        self.scaling = scaling
        self.home_boost = home_boost
        self.off = defaultdict(float)
        self.dff = defaultdict(float)
        self.poss = defaultdict(float)
        self.chemistry = defaultdict(float)

    @staticmethod
    def _key(player_ids):
        ids = sorted(str(x) for x in player_ids if x and str(x) != "nan")
        return tuple(ids[:5]) if len(ids) >= 5 else tuple(ids)

    def _player_sum(self, player_ids, player_tracker, side="off"):
        if not player_tracker:
            return 1500.0, 1500.0
        off, dff, _ = player_tracker.lineup_stats(player_ids)
        return off, dff

    def lineup_rating(self, player_ids, player_tracker=None):
        key = self._key(player_ids)
        if not key:
            return 0.0, 0.0, 0.0, 0.0
        p_off, p_def = self._player_sum(key, player_tracker)
        n = self.poss.get(key, 0.0)
        if n >= MIN_POSSESSIONS and key in self.off:
            w = n / (n + SHRINK_K)
            off = w * self.off[key] + (1 - w) * (p_off - 1500.0)
            dff = w * self.dff[key] + (1 - w) * (p_def - 1500.0)
            chem = self.chemistry.get(key, 0.0) * w
            return off, dff, chem, n
        return (p_off - 1500.0) * 0.1, (p_def - 1500.0) * 0.1, 0.0, n

    def expected_margin(self, home_ids, away_ids, possessions, player_tracker=None, is_home=True):
        ho, hd, hc, _ = self.lineup_rating(home_ids, player_tracker)
        ao, ad, ac, _ = self.lineup_rating(away_ids, player_tracker)
        hb = self.home_boost if is_home else -self.home_boost
        ppp_h = self.league_xppp + hb + (1500 + ho - (1500 + ad)) / self.scaling
        ppp_a = self.league_xppp - hb + (1500 + ao - (1500 + hd)) / self.scaling
        margin = (ppp_h - ppp_a) * possessions
        chem_net = (hc - ac) * possessions / 100.0
        return margin + chem_net, chem_net

    def update_stint(self, off_ids, def_ids, xpts_off, xpts_def, possessions,
                     player_tracker=None, is_home_offense=True):
        if possessions <= 0:
            return
        key_off = self._key(off_ids)
        if len(key_off) < 5:
            return
        exp_margin, _ = self.expected_margin(off_ids, def_ids, possessions, player_tracker, is_home_offense)
        act_margin = (xpts_off - xpts_def) * possessions / max(possessions, 1)
        err = (xpts_off / possessions - self.league_xppp) * possessions if possessions else 0
        p_off, _ = self._player_sum(key_off, player_tracker)
        residual = err - (p_off - 1500.0) / self.scaling * possessions
        k = 0.15 * possessions / (self.poss[key_off] + possessions + SHRINK_K)
        self.off[key_off] += k * err
        self.dff[key_off] += k * (-err * 0.5)
        self.chemistry[key_off] = 0.9 * self.chemistry.get(key_off, 0) + 0.1 * residual
        self.poss[key_off] += possessions

    def feature_dict(self, home_ids, away_ids, player_tracker=None):
        ho, hd, hc, hn = self.lineup_rating(home_ids, player_tracker)
        ao, ad, ac, an = self.lineup_rating(away_ids, player_tracker)
        return {
            "h_lineup5_off": ho, "h_lineup5_def": hd, "h_lineup5_chem": hc,
            "a_lineup5_off": ao, "a_lineup5_def": ad, "a_lineup5_chem": ac,
            "lineup5_net": (ho - ad) - (ao - hd),
            "lineup5_chem_diff": hc - ac,
            "lineup5_sample_min": min(hn, an),
        }

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump(self.__dict__, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls()
        obj.__dict__.update(data)
        return obj
