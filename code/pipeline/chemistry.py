"""Lineup chemistry: 2/3-man residuals and on/off net ratings from stints."""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np

MIN_DUO_POSS = 50
SHRINK = 80.0


class ChemistryTracker:
    """Tracks pairwise/triple chemistry residuals with empirical-Bayes shrinkage."""

    def __init__(self, league_xppp: float = 1.10):
        self.league_xppp = league_xppp
        self.duo_off = defaultdict(float)
        self.duo_def = defaultdict(float)
        self.duo_poss = defaultdict(float)
        self.trio_off = defaultdict(float)
        self.trio_poss = defaultdict(float)
        self.on_off = defaultdict(lambda: {"on": 0.0, "off": 0.0, "on_p": 0.0, "off_p": 0.0})

    @staticmethod
    def _duo_key(a, b):
        return tuple(sorted((str(a), str(b))))

    @staticmethod
    def _trio_key(ids):
        return tuple(sorted(str(x) for x in ids if x))

    def _shrink(self, raw, n):
        return raw * (n / (n + SHRINK)) if n > 0 else 0.0

    def update_stint(self, off_ids, def_ids, xpts_off, xpts_def, possessions,
                     player_tracker=None, expected_ppp=None):
        if possessions <= 0:
            return
        act_ppp = xpts_off / possessions
        exp = expected_ppp if expected_ppp is not None else self.league_xppp
        residual = act_ppp - exp
        ids = [str(x) for x in off_ids if x and str(x) != "nan"]
        for a, b in itertools.combinations(ids, 2):
            k = self._duo_key(a, b)
            w = 0.08 * possessions / (self.duo_poss[k] + possessions + SHRINK)
            self.duo_off[k] += w * residual
            self.duo_poss[k] += possessions
        if len(ids) >= 3:
            for trio in itertools.combinations(ids, 3):
                tk = self._trio_key(trio)
                w = 0.05 * possessions / (self.trio_poss[tk] + possessions + SHRINK)
                self.trio_off[tk] += w * residual
                self.trio_poss[tk] += possessions
        for pid in ids:
            st = self.on_off[pid]
            st["on"] += act_ppp * possessions
            st["on_p"] += possessions

    def _duo_net(self, lineup):
        ids = [str(x) for x in lineup if x and str(x) != "nan"]
        if len(ids) < 2:
            return 0.0, 0.0
        vals, weights = [], []
        for a, b in itertools.combinations(ids, 2):
            k = self._duo_key(a, b)
            n = self.duo_poss.get(k, 0)
            if n >= MIN_DUO_POSS:
                vals.append(self._shrink(self.duo_off[k], n))
                weights.append(n)
        if not vals:
            return 0.0, 0.0
        return float(np.average(vals, weights=weights)), float(sum(weights))

    def _onoff_net(self, lineup):
        nets = []
        for pid in lineup:
            st = self.on_off.get(str(pid))
            if not st or st["on_p"] < 30:
                continue
            on_ppp = st["on"] / st["on_p"]
            nets.append(on_ppp - self.league_xppp)
        return float(np.mean(nets)) if nets else 0.0

    def _trio_net(self, lineup):
        ids = [str(x) for x in lineup if x and str(x) != "nan"]
        if len(ids) < 3:
            return 0.0, 0.0
        vals, weights = [], []
        for trio in itertools.combinations(ids, 3):
            tk = self._trio_key(trio)
            n = self.trio_poss.get(tk, 0)
            if n >= MIN_DUO_POSS:
                vals.append(self._shrink(self.trio_off[tk], n))
                weights.append(n)
        if not vals:
            return 0.0, 0.0
        return float(np.average(vals, weights=weights)), float(sum(weights))

    def lineup_chemistry(self, lineup, player_tracker=None):
        duo_net, duo_w = self._duo_net(lineup)
        trio_net, _ = self._trio_net(lineup)
        onoff = self._onoff_net(lineup)
        unc = 1.0 / (1.0 + duo_w / 200.0) if duo_w else 1.0
        return duo_net * 100.0, trio_net * 100.0, onoff * 100.0, unc

    def feature_dict(self, home_lineup, away_lineup, player_tracker=None):
        h_duo, h_trio, h_on, h_unc = self.lineup_chemistry(home_lineup, player_tracker)
        a_duo, a_trio, a_on, a_unc = self.lineup_chemistry(away_lineup, player_tracker)
        return {
            "h_chem_net": h_duo + h_on,
            "a_chem_net": a_duo + a_on,
            "chem_diff": (h_duo + h_on) - (a_duo + a_on),
            "h_chem_duo_net": h_duo,
            "a_chem_duo_net": a_duo,
            "h_chem_trio_net": h_trio,
            "a_chem_trio_net": a_trio,
            "h_onoff_net": h_on,
            "a_onoff_net": a_on,
            "chem_uncertainty": h_unc + a_unc,
        }

    def usage_conflict(self, weighted_ids):
        """High-usage overlap penalty when multiple high-minute players share floor."""
        if not weighted_ids:
            return 0.0
        pairs = sorted(weighted_ids, key=lambda x: -x[1])[:4]
        if len(pairs) < 2:
            return 0.0
        top_w = pairs[0][1]
        conflict = sum(w for _, w in pairs[1:] if w > 0.15 * top_w)
        return min(1.0, conflict)

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "duo_off": dict(self.duo_off), "duo_poss": dict(self.duo_poss),
                "trio_off": dict(self.trio_off), "trio_poss": dict(self.trio_poss),
                "on_off": {k: dict(v) for k, v in self.on_off.items()},
                "league_xppp": self.league_xppp,
            }, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(league_xppp=data.get("league_xppp", 1.10))
        obj.duo_off = defaultdict(float, data.get("duo_off", {}))
        obj.duo_poss = defaultdict(float, data.get("duo_poss", {}))
        obj.trio_off = defaultdict(float, data.get("trio_off", {}))
        obj.trio_poss = defaultdict(float, data.get("trio_poss", {}))
        obj.on_off = defaultdict(
            lambda: {"on": 0.0, "off": 0.0, "on_p": 0.0, "off_p": 0.0},
            {k: defaultdict(float, v) for k, v in data.get("on_off", {}).items()},
        )
        return obj
