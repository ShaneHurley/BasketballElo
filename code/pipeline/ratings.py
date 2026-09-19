"""Player-level Elo/xPPP rating tracker."""
import math
from collections import defaultdict
from datetime import date as date_type

import numpy as np
import pandas as pd

from pipeline.config import (
    ASSIST_SPLIT, OFFSEASON_REVERSION, USAGE_FLOOR,
    HOME_PPP_BOOST, K_MULT_HALF_LIFE, GARBAGE_TIME_WEIGHT,
    CLUTCH_BOOST, TOV_PENALTY, FOUL_DRAW_BOOST, VARIANCE_DAMPEN,
    XPPP_ACTUAL_BLEND, K_DEF_EVENTS, TOV_RATE_THRESHOLD,
    THREE_PA_RATE_THRESHOLD,
)
from pipeline.dates import is_valid_timestamp, to_py_date


class PlayerRatingTracker:
    """
    Glicko‑2 inspired rating tracker with separate offensive/defensive ratings.
    Maintains μ (rating), RD (uncertainty), and simplified volatility tracking.
    Context-aware stint multipliers use PBP-derived box stats passed via stint_ctx.

    P0.5 (bench_rd_games_played_proxy registry entry, docs-only per plan —
    no Kalman rewrite here): despite the Glicko-2 naming, ``*_rd`` is
    **not** a real Glicko-2 uncertainty estimate. It only ever decays
    multiplicatively toward ``rd_floor`` as a function of *how many times*
    ``_update_ratings`` has touched a player (see the ``0.99 + 0.02 *
    min(abs_err, 0.15)`` update in ``_update_ratings`` below, and the
    inactivity-based inflation in the offseason-reversion path), i.e. it is
    a games-played/inactivity counter reused as if it carried genuine
    posterior-variance information. A true Glicko-2/Kalman RD would grow
    with rating volatility and shrink with informative observations, not
    merely with elapsed update count. Any feature or calibration group
    (e.g. ``elo_calibration.py``'s "uncertainty" group,
    ``h_rating_uncertainty``/``a_rating_uncertainty``) that consumes this
    value should treat it as a proxy for experience/recency, not as a
    calibrated confidence interval. See Epic 9.1 (Kalman filter) for the
    real fix; deliberately out of scope for this fix (see plan phase 0).
    """

    DEFAULTS = {
        "tau": 0.5,
        "epsilon": 1e-6,
        "default_mu": 1500.0,
        "default_rd": 350.0,
        "default_sigma": 0.06,
        "HOME_PPP_BOOST": HOME_PPP_BOOST,
        "OFFSEASON_REVERSION": OFFSEASON_REVERSION,
        "USAGE_FLOOR": USAGE_FLOOR,
        "assist_split": ASSIST_SPLIT,
        "ELO_SCALING_FACTOR": 1000,
        "K_OFF": 0.9,
        "K_DEF": 0.9,
        "k_mult_half_life": K_MULT_HALF_LIFE,
        "rd_floor": 30.0,
        "garbage_time_weight": GARBAGE_TIME_WEIGHT,
        "clutch_boost": CLUTCH_BOOST,
        "tov_penalty": TOV_PENALTY,
        "foul_draw_boost": FOUL_DRAW_BOOST,
        "variance_dampen": VARIANCE_DAMPEN,
        "xppp_actual_blend": XPPP_ACTUAL_BLEND,
        "k_def_events": K_DEF_EVENTS,
        "tov_rate_threshold": TOV_RATE_THRESHOLD,
        "three_pa_rate_threshold": THREE_PA_RATE_THRESHOLD,
    }

    def __init__(self, config=None, league_xppp=1.10):
        self.cfg = {**self.DEFAULTS, **(config or {})}
        self.league_xppp = league_xppp
        self.players = {}
        self.player_games = defaultdict(int)

    def _get(self, pid):
        pid = str(pid)
        if pid not in self.players:
            self.players[pid] = {
                "O_mu": self.cfg["default_mu"],
                "D_mu": self.cfg["default_mu"],
                "D_rim_mu": self.cfg["default_mu"],
                "D_peri_mu": self.cfg["default_mu"],
                "O_rd": self.cfg["default_rd"],
                "D_rd": self.cfg["default_rd"],
                "D_rim_rd": self.cfg["default_rd"],
                "D_peri_rd": self.cfg["default_rd"],
                "O_sigma": self.cfg["default_sigma"],
                "D_sigma": self.cfg["default_sigma"],
                "Possessions": 0,
                "luck_pts": 0.0,
                "def_events": 0.0,
                "off_tov": 0.0,
                "last_date": None,
            }
        pl = self.players[pid]
        if "D_rim_mu" not in pl:
            pl["D_rim_mu"] = pl["D_mu"]
            pl["D_peri_mu"] = pl["D_mu"]
            pl["D_rim_rd"] = pl["D_rd"]
            pl["D_peri_rd"] = pl["D_rd"]
        return pl

    @property
    def O_Elo(self):
        return {pid: data["O_mu"] for pid, data in self.players.items()}

    @property
    def D_Elo(self):
        return {pid: data["D_mu"] for pid, data in self.players.items()}

    def lineup_uncertainty(self, player_ids):
        ids = [str(x) for x in player_ids if x and str(x) != "nan"]
        if not ids:
            return 350.0, 350.0
        o_rd = np.mean([self._get(p)["O_rd"] for p in ids])
        d_rd = np.mean([self._get(p)["D_rd"] for p in ids])
        return o_rd, d_rd

    def weighted_lineup_uncertainty(self, weighted_ids):
        pairs = [(str(p), float(w)) for p, w in weighted_ids if p and w > 0]
        if not pairs:
            return 350.0, 350.0
        ws = sum(w for _, w in pairs)
        o_rd = sum(self._get(p)["O_rd"] * w for p, w in pairs) / ws
        d_rd = sum(self._get(p)["D_rd"] * w for p, w in pairs) / ws
        return o_rd, d_rd

    def lineup_rolling_rates(self, player_ids, weights=None):
        """Lineup-level luck/def-event/tov rates from accumulated player stats."""
        ids = [str(x) for x in player_ids if x and str(x) != "nan"]
        if not ids:
            return 0.0, 0.0, 0.0

        if weights:
            pairs = [(str(p), float(w)) for p, w in weights if p and w > 0]
            if pairs:
                ws = sum(w for _, w in pairs)
                luck = sum(self._get(p)["luck_pts"] * w for p, w in pairs) / ws
                defe = sum(self._get(p)["def_events"] * w for p, w in pairs) / ws
                tov = sum(self._get(p)["off_tov"] * w for p, w in pairs) / ws
                poss = sum(self._get(p)["Possessions"] * w for p, w in pairs) / ws
                if poss > 0:
                    return luck / poss, defe / poss, tov / poss
                return 0.0, 0.0, 0.0

        poss = np.mean([max(self._get(p)["Possessions"], 1.0) for p in ids])
        luck = np.mean([self._get(p)["luck_pts"] for p in ids]) / poss
        defe = np.mean([self._get(p)["def_events"] for p in ids]) / poss
        tov = np.mean([self._get(p)["off_tov"] for p in ids]) / poss
        return float(luck), float(defe), float(tov)

    def apply_inactivity_decay(self, player_ids, date):
        if date is None:
            return
        try:
            if pd.isna(date):
                return
        except (TypeError, ValueError):
            pass

        if isinstance(date, date_type):
            py_date = date
        elif is_valid_timestamp(date):
            py_date = pd.Timestamp(date).date()
        else:
            py_date = to_py_date(date)
            if py_date is None:
                return

        for pid in player_ids:
            p = self._get(pid)
            last = p.get("last_date")
            if last is not None:
                try:
                    if isinstance(last, pd.Timestamp):
                        if pd.isna(last):
                            last = None
                        else:
                            last = last.date()
                    if last is not None:
                        days = (py_date - last).days
                        if days > 60:
                            p["O_rd"] = min(350.0, p["O_rd"] * (1 + 0.1 * (days / 7)))
                            p["D_rd"] = min(350.0, p["D_rd"] * (1 + 0.1 * (days / 7)))
                except (TypeError, ValueError):
                    pass
            p["last_date"] = py_date

    def offseason_revert(self, returning_player_ids=None):
        base_rev = self.cfg["OFFSEASON_REVERSION"]
        default_rd = self.cfg["default_rd"]
        default_sigma = self.cfg["default_sigma"]

        for pid, p in self.players.items():
            if returning_player_ids is not None and pid not in returning_player_ids:
                effective_rev = min(base_rev * 2.0, 0.50)
            else:
                effective_rev = base_rev

            p["O_mu"] = 1500.0 + (p["O_mu"] - 1500.0) * (1 - effective_rev)
            p["D_mu"] = 1500.0 + (p["D_mu"] - 1500.0) * (1 - effective_rev)
            p["D_rim_mu"] = 1500.0 + (p.get("D_rim_mu", p["D_mu"]) - 1500.0) * (1 - effective_rev)
            p["D_peri_mu"] = 1500.0 + (p.get("D_peri_mu", p["D_mu"]) - 1500.0) * (1 - effective_rev)
            p["O_rd"] = default_rd
            p["D_rd"] = default_rd
            p["D_rim_rd"] = default_rd
            p["D_peri_rd"] = default_rd
            p["O_sigma"] = default_sigma
            p["D_sigma"] = default_sigma
            p["Possessions"] = 0
            p["luck_pts"] = 0.0
            p["def_events"] = 0.0
            p["off_tov"] = 0.0

        self.player_games.clear()

    def lineup_stats(self, player_ids, weights=None, opp_rim_rate=None, opp_three_rate=None):
        ids = [str(x) for x in player_ids if x and str(x) != "nan"]
        if not ids:
            return 1500.0, 1500.0, 0.0
        if weights:
            pairs = [(str(p), float(w)) for p, w in weights if p and w > 0]
            if pairs:
                off, _, exp = self.weighted_lineup_stats(pairs)
                if opp_rim_rate is not None and opp_three_rate is not None:
                    dff = self.matchup_def_rating(ids, opp_rim_rate, opp_three_rate, weights=pairs)
                else:
                    _, dff, _ = self.weighted_lineup_stats(pairs)
                return off, dff, exp
        off = np.mean([self._get(p)["O_mu"] for p in ids])
        if opp_rim_rate is not None and opp_three_rate is not None:
            dff = self.matchup_def_rating(ids, opp_rim_rate, opp_three_rate)
        else:
            dff = np.mean([self._get(p)["D_mu"] for p in ids])
        exp = np.mean([self._get(p)["Possessions"] for p in ids])
        return off, dff, exp

    def _lineup_rim_peri(self, player_ids, weights=None):
        ids = [str(x) for x in player_ids if x and str(x) != "nan"]
        if not ids:
            return 1500.0, 1500.0, 1500.0
        if weights:
            pairs = [(str(p), float(w)) for p, w in weights if p and w > 0]
            if pairs:
                ws = sum(w for _, w in pairs)
                rim = sum(self._get(p)["D_rim_mu"] * w for p, w in pairs) / ws
                peri = sum(self._get(p)["D_peri_mu"] * w for p, w in pairs) / ws
                legacy = sum(self._get(p)["D_mu"] * w for p, w in pairs) / ws
                return float(rim), float(peri), float(legacy)
        rim = np.mean([self._get(p)["D_rim_mu"] for p in ids])
        peri = np.mean([self._get(p)["D_peri_mu"] for p in ids])
        legacy = np.mean([self._get(p)["D_mu"] for p in ids])
        return float(rim), float(peri), float(legacy)

    def matchup_def_rating(self, player_ids, opp_rim_rate, opp_three_rate, weights=None):
        """Blend rim/perimeter/mid defensive Elo by opponent shot profile."""
        rim, peri, legacy = self._lineup_rim_peri(player_ids, weights)
        rim_r = float(np.clip(opp_rim_rate, 0.05, 0.65))
        peri_r = float(np.clip(opp_three_rate, 0.05, 0.65))
        mid_r = max(0.12, 1.0 - rim_r - peri_r)
        total = rim_r + peri_r + mid_r
        rw, pw, mw = rim_r / total, peri_r / total, mid_r / total
        return rw * rim + pw * peri + mw * legacy

    def weighted_lineup_stats(self, weighted_ids, opp_rim_rate=None, opp_three_rate=None):
        pairs = [(str(p), float(w)) for p, w in weighted_ids if p and w > 0]
        if not pairs:
            return 1500.0, 1500.0, 0.0
        ws = sum(w for _, w in pairs)
        off = sum(self._get(p)["O_mu"] * w for p, w in pairs) / ws
        ids = [p for p, _ in pairs]
        if opp_rim_rate is not None and opp_three_rate is not None:
            dff = self.matchup_def_rating(ids, opp_rim_rate, opp_three_rate, weights=pairs)
        else:
            dff = sum(self._get(p)["D_mu"] * w for p, w in pairs) / ws
        exp = sum(self._get(p)["Possessions"] * w for p, w in pairs) / ws
        return off, dff, exp

    def _def_credit_weights(self, stint_ctx, defender_side: str):
        """Rim/perimeter/mid credit shares for a defending side ('home'|'away')."""
        stint_ctx = stint_ctx or {}
        if defender_side == "home":
            rim = float(stint_ctx.get("away_rim_rate", 0.30))
            peri = float(stint_ctx.get("away_3pa_rate", 0.38))
        else:
            rim = float(stint_ctx.get("home_rim_rate", 0.30))
            peri = float(stint_ctx.get("home_3pa_rate", 0.38))
        mid = max(0.12, 1.0 - rim - peri)
        total = rim + peri + mid
        return rim / total, peri / total, mid / total

    def _update_def_split(self, ids, error, poss, usage, wt, stint_ctx, defender_side, abs_err=0.0):
        rw, pw, mw = self._def_credit_weights(stint_ctx, defender_side)
        self._update_ratings(ids, error * rw, poss, usage, side="def_rim", abs_err=abs_err)
        self._update_ratings(ids, error * pw, poss, usage, side="def_peri", abs_err=abs_err)
        self._update_ratings(ids, error * mw, poss, usage, side="def", abs_err=abs_err)
        for p in ids:
            pl = self._get(p)
            pl["D_mu"] = 0.5 * (pl["D_rim_mu"] + pl["D_peri_mu"])

    def _context_multiplier(self, stint_ctx, side: str) -> float:
        """Boost/penalize stint weight from PBP box context (home=A, away=B)."""
        if not stint_ctx:
            return 1.0
        m = 1.0
        if stint_ctx.get("clutch"):
            m *= self.cfg.get("clutch_boost", 1.0)
        if stint_ctx.get("garbage"):
            m *= self.cfg.get("garbage_time_weight", 0.5)

        tov_thr = self.cfg.get("tov_rate_threshold", 0.15)
        if side == "home":
            tov_rate = stint_ctx.get("home_tov_rate", 0.0)
            foul_rate = stint_ctx.get("home_foul_draw_rate", 0.0)
            three_rate = stint_ctx.get("home_3pa_rate", 0.0)
        else:
            tov_rate = stint_ctx.get("away_tov_rate", 0.0)
            foul_rate = stint_ctx.get("away_foul_draw_rate", 0.0)
            three_rate = stint_ctx.get("away_3pa_rate", 0.0)

        if tov_rate > tov_thr:
            m *= self.cfg.get("tov_penalty", 1.0)
        if foul_rate > 0.12:
            m *= self.cfg.get("foul_draw_boost", 1.0)
        if three_rate > self.cfg.get("three_pa_rate_threshold", 0.45):
            m *= self.cfg.get("variance_dampen", 1.0)
        return m

    def _accumulate_player_context(self, ids, usage, poss, stint_ctx, side: str) -> None:
        if not stint_ctx or poss <= 0:
            return
        n = max(len(ids), 1)
        base_w = 1.0 / n
        total_usage = sum(usage.values()) if usage else 0
        u_floor = self.cfg["USAGE_FLOOR"]

        if side == "home":
            luck_share = stint_ctx.get("home_luck_ppp", 0.0) * poss
            def_ev = (stint_ctx.get("home_def_events", 0.0)) * poss
            tov_ev = stint_ctx.get("home_tov_rate", 0.0) * poss
        else:
            luck_share = stint_ctx.get("away_luck_ppp", 0.0) * poss
            def_ev = stint_ctx.get("away_def_events", 0.0) * poss
            tov_ev = stint_ctx.get("away_tov_rate", 0.0) * poss

        shares = []
        for p in ids:
            if total_usage > 0:
                raw = usage.get(p, 0) / total_usage
            else:
                raw = base_w
            shares.append(max(raw, base_w * u_floor))
        s = sum(shares) or 1.0

        for i, p in enumerate(ids):
            pl = self._get(p)
            sh = shares[i] / s
            pl["luck_pts"] += luck_share * sh
            pl["def_events"] += def_ev * sh
            pl["off_tov"] += tov_ev * sh

    def _apply_def_event_credit(self, ids, usage, poss, stint_ctx, side: str, wt: float) -> None:
        """Explicit micro-credit for steals/blocks/forced turnovers."""
        if not stint_ctx or poss <= 0:
            return
        k_def_ev = self.cfg.get("k_def_events", 0.0)
        if k_def_ev <= 0:
            return
        if side == "home":
            rate = stint_ctx.get("home_def_events", 0.0)
        else:
            rate = stint_ctx.get("away_def_events", 0.0)
        if rate <= 0:
            return
        credit = k_def_ev * rate * wt
        self._update_ratings(ids, credit, poss, usage, side="def", abs_err=0.0)

    def process_stint(self, ids_A, ids_B, poss, xpts_A, xpts_B,
                      usage_A=None, usage_B=None,
                      period=1, start_A=0, start_B=0,
                      end_A=0, end_B=0, season_progress=0.5,
                      stint_ctx=None):
        ast_split = self.cfg.get("assist_split", ASSIST_SPLIT)

        def collapse_usage(u_dict):
            if not u_dict:
                return {}
            return {p: (v[0] + v[1] * ast_split + v[2] * (1 - ast_split))
                    for p, v in u_dict.items()}

        flat_usage_A = collapse_usage(usage_A)
        flat_usage_B = collapse_usage(usage_B)

        def _usage_weights(u_dict, ids):
            if not u_dict:
                n = max(len(ids), 1)
                return [(p, 1.0 / n) for p in ids]
            total = sum(u_dict.values()) or 1.0
            return [(p, u_dict.get(p, 0) / total) for p in ids]

        wA = _usage_weights(flat_usage_A, ids_A)
        wB = _usage_weights(flat_usage_B, ids_B)

        opp_rim_for_home_d = float(stint_ctx.get("away_rim_rate", 0.30) if stint_ctx else 0.30)
        opp_peri_for_home_d = float(stint_ctx.get("away_3pa_rate", 0.38) if stint_ctx else 0.38)
        opp_rim_for_away_d = float(stint_ctx.get("home_rim_rate", 0.30) if stint_ctx else 0.30)
        opp_peri_for_away_d = float(stint_ctx.get("home_3pa_rate", 0.38) if stint_ctx else 0.38)

        off_A, def_A, _ = self.lineup_stats(
            ids_A, weights=wA, opp_rim_rate=opp_rim_for_away_d, opp_three_rate=opp_peri_for_away_d,
        )
        off_B, def_B, _ = self.lineup_stats(
            ids_B, weights=wB, opp_rim_rate=opp_rim_for_home_d, opp_three_rate=opp_peri_for_home_d,
        )
        margin_A = end_A - start_A

        if stint_ctx is None:
            stint_ctx = {}
        else:
            stint_ctx = dict(stint_ctx)
        stint_ctx.setdefault("clutch", period >= 4 and abs(start_A - start_B) < 10)
        stint_ctx.setdefault("garbage", period >= 4 and abs(margin_A) >= 15)

        wt = self._weight_stint(poss, period, margin_A, season_progress, stint_ctx)
        wt_a = wt * self._context_multiplier(stint_ctx, "home")
        wt_b = wt * self._context_multiplier(stint_ctx, "away")

        scaling = self.cfg["ELO_SCALING_FACTOR"]
        h_boost = self.cfg["HOME_PPP_BOOST"]

        exp_ppp_A = self.league_xppp + h_boost + (off_A - def_B) / scaling
        exp_ppp_B = self.league_xppp - h_boost + (off_B - def_A) / scaling

        act_ppp_A = xpts_A / poss if poss > 0 else 0
        act_ppp_B = xpts_B / poss if poss > 0 else 0

        blend = self.cfg.get("xppp_actual_blend", 0.0)
        if blend > 0 and poss > 0:
            act_pts_A = stint_ctx.get("home_pts", xpts_A) / poss
            act_pts_B = stint_ctx.get("away_pts", xpts_B) / poss
            act_ppp_A = (1.0 - blend) * act_ppp_A + blend * act_pts_A
            act_ppp_B = (1.0 - blend) * act_ppp_B + blend * act_pts_B

        err_A = act_ppp_A - exp_ppp_A
        err_B = act_ppp_B - exp_ppp_B
        # O/D residual signs documented in matchup_rating.residual_update_signs:
        # home O gets +err_A; away D gets credit opposite home scoring error.

        self._update_ratings(ids_A, err_A * wt_a, poss, flat_usage_A, side="off", abs_err=abs(err_A))
        self._update_def_split(
            ids_B, err_B * wt_b, poss, flat_usage_B, wt_b, stint_ctx, "away", abs_err=abs(err_B),
        )
        self._update_def_split(
            ids_A, -err_B * wt_a, poss, flat_usage_A, wt_a, stint_ctx, "home", abs_err=abs(err_B),
        )
        self._update_ratings(ids_B, -err_A * wt_b, poss, flat_usage_B, side="off", abs_err=abs(err_A))

        self._apply_def_event_credit(ids_A, flat_usage_A, poss, stint_ctx, "home", wt_a)
        self._apply_def_event_credit(ids_B, flat_usage_B, poss, stint_ctx, "away", wt_b)

        self._accumulate_player_context(ids_A, flat_usage_A, poss, stint_ctx, "home")
        self._accumulate_player_context(ids_B, flat_usage_B, poss, stint_ctx, "away")

    def _update_ratings(self, ids, error, poss, usage, side, abs_err=0.0):
        ids = [str(x) for x in ids if x and str(x) != "nan"]
        if not ids:
            return

        n = len(ids)
        base_w = 1.0 / n
        total_usage = sum(usage.values()) if usage else 0
        u_floor = self.cfg["USAGE_FLOOR"]
        half_life = self.cfg.get("k_mult_half_life", 15.0)
        rd_floor = self.cfg.get("rd_floor", 30.0)

        shares = []
        for p in ids:
            if total_usage > 0:
                raw = usage.get(p, 0) / total_usage
            else:
                raw = base_w
            shares.append(max(raw, base_w * u_floor))
        s = sum(shares)

        for i, p in enumerate(ids):
            pl = self._get(p)
            games = self.player_games[p]
            k_mult = max(0.5, 2.0 * math.exp(-games / max(half_life, 1.0)))
            k_base = self.cfg.get("K_OFF", 0.9) if side == "off" else self.cfg.get("K_DEF", 0.9)
            if side == "def_rim":
                rd = pl["D_rim_rd"]
            elif side == "def_peri":
                rd = pl["D_peri_rd"]
            elif side == "off":
                rd = pl["O_rd"]
            else:
                rd = pl["D_rd"]
            k_effective = k_base * (rd / 350.0)
            delta = error * k_effective * (shares[i] / s) * k_mult

            if side == "off":
                pl["O_mu"] += delta
                rd_new = pl["O_rd"] * (0.99 + 0.02 * min(abs_err, 0.15))
                pl["O_rd"] = max(rd_floor, min(350.0, rd_new))
                pl["O_sigma"] = min(0.15, pl["O_sigma"] * (1.0 + 0.05 * min(abs_err, 0.2)))
            elif side == "def_rim":
                pl["D_rim_mu"] += delta
                rd_new = pl["D_rim_rd"] * (0.99 + 0.02 * min(abs_err, 0.15))
                pl["D_rim_rd"] = max(rd_floor, min(350.0, rd_new))
            elif side == "def_peri":
                pl["D_peri_mu"] += delta
                rd_new = pl["D_peri_rd"] * (0.99 + 0.02 * min(abs_err, 0.15))
                pl["D_peri_rd"] = max(rd_floor, min(350.0, rd_new))
            else:
                pl["D_mu"] += delta
                rd_new = pl["D_rd"] * (0.99 + 0.02 * min(abs_err, 0.15))
                pl["D_rd"] = max(rd_floor, min(350.0, rd_new))
                pl["D_sigma"] = min(0.15, pl["D_sigma"] * (1.0 + 0.05 * min(abs_err, 0.2)))

            pl["Possessions"] += poss
            self.player_games[p] += 1

    def _weight_stint(self, poss, period, margin, season_progress, stint_ctx=None):
        weight = poss
        gt_w = self.cfg.get("garbage_time_weight", 0.5)
        if stint_ctx and stint_ctx.get("garbage"):
            weight *= gt_w
        elif period >= 4 and abs(margin) >= 15:
            weight *= gt_w
        weight *= (0.7 + 0.3 * season_progress)
        return weight

    def predict_game_margin(self, home_ids, away_ids, possessions,
                            weights_home=None, weights_away=None, is_home=True):
        if weights_home:
            ho, hd, _ = self.weighted_lineup_stats(weights_home)
        else:
            ho, hd, _ = self.lineup_stats(home_ids)
        if weights_away:
            ao, ad, _ = self.weighted_lineup_stats(weights_away)
        else:
            ao, ad, _ = self.lineup_stats(away_ids)

        scaling = self.cfg["ELO_SCALING_FACTOR"]
        hb = self.cfg["HOME_PPP_BOOST"] if is_home else -self.cfg["HOME_PPP_BOOST"]
        ppp_h = self.league_xppp + hb + (ho - ad) / scaling
        ppp_a = self.league_xppp - hb + (ao - hd) / scaling
        margin = (ppp_h - ppp_a) * float(possessions or 100.0)

        ho_rd, hd_rd = self.lineup_uncertainty(home_ids)
        ao_rd, ad_rd = self.lineup_uncertainty(away_ids)
        unc = (ho_rd + hd_rd + ao_rd + ad_rd) / 4.0
        return float(margin), float(unc)

    def implied_total(self, home_ids, away_ids, possessions,
                      weights_home=None, weights_away=None, league_avg_total=225.0):
        """Elo-pace implied game total."""
        if weights_home:
            ho, hd, _ = self.weighted_lineup_stats(weights_home)
        else:
            ho, hd, _ = self.lineup_stats(home_ids)
        if weights_away:
            ao, ad, _ = self.weighted_lineup_stats(weights_away)
        else:
            ao, ad, _ = self.lineup_stats(away_ids)
        scaling = self.cfg["ELO_SCALING_FACTOR"]
        hb = self.cfg["HOME_PPP_BOOST"]
        ppp_h = self.league_xppp + hb + (ho - ad) / scaling
        ppp_a = self.league_xppp - hb + (ao - hd) / scaling
        poss = float(possessions or 100.0)
        return float((ppp_h + ppp_a) * poss)

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({"players": self.players, "player_games": dict(self.player_games), "cfg": self.cfg}, f)

    @classmethod
    def load_state(cls, path, league_xppp=1.10):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        tracker = cls(config=data.get("cfg"), league_xppp=league_xppp)
        tracker.players = data["players"]
        tracker.player_games = defaultdict(int, data.get("player_games", {}))
        return tracker
