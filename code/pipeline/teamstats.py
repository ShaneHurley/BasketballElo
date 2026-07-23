"""Walk-forward team box-score form tracking (no leakage).

Adds non-leaky team-level rolling features on top of the Elo/xPPP system:

  - offensive / defensive rating (points per 100 possessions)
  - 3PT makes per game and 3PT%
  - blocks per game
  - forced turnovers per game

All rolling values use only games strictly before the current game date.
"""
from __future__ import annotations

from collections import defaultdict, deque

import numpy as np
import pandas as pd

# League-average fallbacks (used before a team has any history)
_DEFAULT_FORM = {
    "off_rtg": 113.0,
    "def_rtg": 113.0,
    "fg3m": 12.0,
    "fg3pct": 0.36,
    "blk": 5.0,
    "ftov": 13.5,
    # Four-factors + extra defensive/pressure metrics
    "efg": 0.53,        # effective FG%
    "oreb_pct": 0.25,   # offensive rebound rate
    "dreb_pct": 0.75,   # defensive rebound rate
    "tov_pct": 13.0,    # own turnovers per 100 possessions
    "ftr": 0.25,        # free-throw rate (FTA / FGA)
    "stl": 7.5,         # steals per game
    "fdrawn": 20.0,     # fouls drawn per game
    "luck_adj_off": 0.0,
    "pts_for": 110.0,
    "pts_against": 110.0,
    "win": 0.5,
    "pythag_win_pct": 0.5,
    "actual_win_pct": 0.5,
    "pythag_residual": 0.0,
}

# Multi-horizon rolling windows for form features (leak-free via get_window).
FORM_ROLL_WINDOWS = (3, 5, 10, 20)
FORM_MULTIWINDOW_KEYS = (
    "off_rtg", "def_rtg", "efg", "tov_pct", "oreb_pct", "dreb_pct", "ftr", "fg3pct",
)


def pythagorean_win_pct(pts_for: float, pts_against: float, exponent: float | None = None) -> float:
    """Expected win rate from points for/against (Morey NBA exponent default)."""
    from pipeline.config import PYTHAGOREAN_EXPONENT

    exp = float(PYTHAGOREAN_EXPONENT if exponent is None else exponent)
    pf = max(float(pts_for), 0.0)
    pa = max(float(pts_against), 0.0)
    if pf <= 0 and pa <= 0:
        return 0.5
    if pf <= 0:
        return 0.0
    if pa <= 0:
        return 1.0
    num = pf ** exp
    den = num + pa ** exp
    return float(num / den) if den > 0 else 0.5


def precompute_game_team_stats(stints_df: pd.DataFrame) -> dict:
    """
    Aggregate stint rows to one record per GAME_ID in a single vectorized pass.

    Returns dict: GAME_ID -> {act_h, act_a, tot_poss, home_poss, away_poss,
                              home_xpts, away_xpts, home_3pm, away_3pm,
                              home_3pa, away_3pa, home_blk, away_blk,
                              home_ftov, away_ftov}
    This replaces per-stint Python loops for game aggregation (big speedup).
    """
    sum_cols = {
        "home_pts": "act_h",
        "away_pts": "act_a",
        "possessions": "tot_poss",
        "home_poss": "home_poss",
        "away_poss": "away_poss",
        "home_xpts": "home_xpts",
        "away_xpts": "away_xpts",
        "home_3pm": "home_3pm",
        "away_3pm": "away_3pm",
        "home_3pa": "home_3pa",
        "away_3pa": "away_3pa",
        "home_blks": "home_blk",
        "away_blks": "away_blk",
        "home_tovs_forced": "home_ftov",
        "away_tovs_forced": "away_ftov",
        # Four-factors raw counts
        "home_oreb": "home_oreb",
        "away_oreb": "away_oreb",
        "home_dreb": "home_dreb",
        "away_dreb": "away_dreb",
        "home_fgm": "home_fgm",
        "away_fgm": "away_fgm",
        "home_fga": "home_fga",
        "away_fga": "away_fga",
        "home_tov": "home_tov",
        "away_tov": "away_tov",
        "home_fta": "home_fta",
        "away_fta": "away_fta",
        "home_fouls_drawn": "home_fdrawn",
        "away_fouls_drawn": "away_fdrawn",
        "home_stl": "home_stl",
        "away_stl": "away_stl",
        "home_xefg_sum": "home_xefg_sum",
        "away_xefg_sum": "away_xefg_sum",
        "home_rim_fga": "home_rim_fga",
        "away_rim_fga": "away_rim_fga",
        "home_three_fga": "home_three_fga",
        "away_three_fga": "away_three_fga",
        "home_shot_dist_sum": "home_shot_dist_sum",
        "away_shot_dist_sum": "away_shot_dist_sum",
    }
    present = {src: dst for src, dst in sum_cols.items() if src in stints_df.columns}
    agg = stints_df.groupby("GAME_ID", sort=False)[list(present.keys())].sum()
    agg = agg.rename(columns=present)

    # Task 014: when canonical, independently-verified final scores (Task
    # 006/007) have been attached to the stints frame, they override the
    # stint-summed act_h/act_a for *labels*. Weighted stint statistics
    # (possessions, xPoints, four-factors, etc.) are untouched and remain the
    # only source for ratings/features. Games without a canonical final
    # (e.g. seasons not yet covered by the canonical builder) keep the
    # stint-summed values as a documented fallback — never silently dropped.
    if "canonical_home_pts" in stints_df.columns and "canonical_away_pts" in stints_df.columns:
        canon = stints_df.groupby("GAME_ID", sort=False)[
            ["canonical_home_pts", "canonical_away_pts"]
        ].first()
        has_canon = canon["canonical_home_pts"].notna() & canon["canonical_away_pts"].notna()
        agg.loc[has_canon.index[has_canon], "act_h"] = canon.loc[has_canon, "canonical_home_pts"]
        agg.loc[has_canon.index[has_canon], "act_a"] = canon.loc[has_canon, "canonical_away_pts"]

    return agg.to_dict("index")


def team_game_form(game_stats: dict, side: str) -> dict:
    """Convert one game's aggregated stats into a team's box-form dict for `side`."""
    g = game_stats
    if side == "home":
        pts, opp_pts = g.get("act_h", 0.0), g.get("act_a", 0.0)
        poss = g.get("home_poss", 0.0)
        opp_poss = g.get("away_poss", 0.0)
        fg3m, fg3a = g.get("home_3pm", 0.0), g.get("home_3pa", 0.0)
        blk, ftov = g.get("home_blk", 0.0), g.get("home_ftov", 0.0)
        oreb, dreb = g.get("home_oreb", 0.0), g.get("home_dreb", 0.0)
        opp_oreb, opp_dreb = g.get("away_oreb", 0.0), g.get("away_dreb", 0.0)
        fgm, fga = g.get("home_fgm", 0.0), g.get("home_fga", 0.0)
        tov, fta = g.get("home_tov", 0.0), g.get("home_fta", 0.0)
        stl, fdrawn = g.get("home_stl", 0.0), g.get("home_fdrawn", 0.0)
    else:
        pts, opp_pts = g.get("act_a", 0.0), g.get("act_h", 0.0)
        poss = g.get("away_poss", 0.0)
        opp_poss = g.get("home_poss", 0.0)
        fg3m, fg3a = g.get("away_3pm", 0.0), g.get("away_3pa", 0.0)
        blk, ftov = g.get("away_blk", 0.0), g.get("away_ftov", 0.0)
        oreb, dreb = g.get("away_oreb", 0.0), g.get("away_dreb", 0.0)
        opp_oreb, opp_dreb = g.get("home_oreb", 0.0), g.get("home_dreb", 0.0)
        fgm, fga = g.get("away_fgm", 0.0), g.get("away_fga", 0.0)
        tov, fta = g.get("away_tov", 0.0), g.get("away_fta", 0.0)
        stl, fdrawn = g.get("away_stl", 0.0), g.get("away_fdrawn", 0.0)

    poss = poss if poss > 0 else g.get("tot_poss", 0.0) / 2.0
    opp_poss = opp_poss if opp_poss > 0 else g.get("tot_poss", 0.0) / 2.0
    xpts = g.get("home_xpts", pts) if side == "home" else g.get("away_xpts", pts)
    luck_adj = 100.0 * (xpts / poss - pts / poss) if poss > 0 else 0.0
    won = 1.0 if pts > opp_pts else 0.0
    return {
        "off_rtg": 100.0 * pts / poss if poss > 0 else _DEFAULT_FORM["off_rtg"],
        "def_rtg": 100.0 * opp_pts / opp_poss if opp_poss > 0 else _DEFAULT_FORM["def_rtg"],
        "pts_for": float(pts),
        "pts_against": float(opp_pts),
        "win": won,
        "fg3m": fg3m,
        "fg3pct": fg3m / fg3a if fg3a > 0 else _DEFAULT_FORM["fg3pct"],
        "blk": blk,
        "ftov": ftov,
        "efg": (fgm + 0.5 * fg3m) / fga if fga > 0 else _DEFAULT_FORM["efg"],
        "oreb_pct": oreb / (oreb + opp_dreb) if (oreb + opp_dreb) > 0 else _DEFAULT_FORM["oreb_pct"],
        "dreb_pct": dreb / (dreb + opp_oreb) if (dreb + opp_oreb) > 0 else _DEFAULT_FORM["dreb_pct"],
        "tov_pct": 100.0 * tov / poss if poss > 0 else _DEFAULT_FORM["tov_pct"],
        "ftr": fta / fga if fga > 0 else _DEFAULT_FORM["ftr"],
        "stl": stl,
        "fdrawn": fdrawn,
        "luck_adj_off": luck_adj,
    }


class TeamFormTracker:
    """Rolling, walk-forward team box-score form.

    Window of recent games (current season weight 1.0, previous season weighted).
    """

    def __init__(self, window=15, prev_season_weight=0.4):
        self.window = window
        self.prev_season_weight = prev_season_weight
        self.history = defaultdict(lambda: deque(maxlen=window * 2))
        # Rolling league pool of (off_rtg, def_rtg) for opponent/era adjustment.
        self.league = deque(maxlen=window * 30)

    def update(self, team, game_date, season, form: dict, opp_off=None, opp_def=None, weight: float = 1.0):
        # Store the opponent's pre-game ratings so `get` can opponent-adjust.
        w = max(0.0, float(weight))
        if w <= 0:
            return
        self.history[team].append((game_date, season, form, opp_off, opp_def, w))
        self.league.append((float(form.get("off_rtg", _DEFAULT_FORM["off_rtg"])),
                            float(form.get("def_rtg", _DEFAULT_FORM["def_rtg"]))))

    def league_avg(self):
        if not self.league:
            return _DEFAULT_FORM["off_rtg"], _DEFAULT_FORM["def_rtg"]
        arr = np.asarray(self.league, dtype=float)
        return float(arr[:, 0].mean()), float(arr[:, 1].mean())

    def get(self, team, current_season, current_date):
        base = dict(_DEFAULT_FORM)
        lg_off, lg_def = self.league_avg()
        base["off_rtg_adj"] = base["off_rtg"]
        base["def_rtg_adj"] = base["def_rtg"]
        if team not in self.history:
            return base
        acc = defaultdict(float)
        opp_off_sum = opp_def_sum = 0.0
        wsum = 0.0
        pts_for_sum = pts_against_sum = 0.0
        wins_sum = 0.0
        from pipeline.config import PYTHAG_MIN_GAMES

        for gd, season, form, oo, od, *rest in self.history[team]:
            w_extra = rest[0] if rest else 1.0
            if current_date is not None and gd is not None and gd >= current_date:
                continue
            w = (1.0 if season == current_season else self.prev_season_weight) * float(w_extra)
            for k, v in form.items():
                if k in ("pts_for", "pts_against", "win"):
                    continue
                acc[k] += v * w
            pts_for_sum += float(form.get("pts_for", 0.0)) * w
            pts_against_sum += float(form.get("pts_against", 0.0)) * w
            wins_sum += float(form.get("win", 0.0)) * w
            # opponent context (fall back to league avg when missing)
            opp_off_sum += (oo if oo is not None else lg_off) * w
            opp_def_sum += (od if od is not None else lg_def) * w
            wsum += w
        if wsum == 0:
            return base
        out = {k: acc[k] / wsum for k in _DEFAULT_FORM if k not in ("pts_for", "pts_against", "win", "pythag_win_pct", "actual_win_pct", "pythag_residual")}
        mean_opp_off = opp_off_sum / wsum
        mean_opp_def = opp_def_sum / wsum
        # Schedule-adjust: credit scoring vs strong defenses / stops vs strong offenses.
        out["off_rtg_adj"] = out["off_rtg"] + (lg_def - mean_opp_def)
        out["def_rtg_adj"] = out["def_rtg"] - (mean_opp_off - lg_off)
        if wsum >= float(PYTHAG_MIN_GAMES):
            out["actual_win_pct"] = float(wins_sum / wsum)
            out["pythag_win_pct"] = pythagorean_win_pct(pts_for_sum, pts_against_sum)
            out["pythag_residual"] = out["actual_win_pct"] - out["pythag_win_pct"]
        else:
            out["actual_win_pct"] = base["actual_win_pct"]
            out["pythag_win_pct"] = base["pythag_win_pct"]
            out["pythag_residual"] = base["pythag_residual"]
        return out

    def get_window(self, team, current_season, current_date, n_games: int):
        """Mean form over the most recent n_games before current_date (leak-free)."""
        base = dict(_DEFAULT_FORM)
        if team not in self.history or n_games <= 0:
            return base
        rows = []
        for gd, season, form, oo, od, *rest in self.history[team]:
            if current_date is not None and gd is not None and gd >= current_date:
                continue
            rows.append((gd, season, form, oo, od, rest[0] if rest else 1.0))
        if not rows:
            return base
        # Most recent first
        rows = sorted(rows, key=lambda x: x[0] if x[0] is not None else 0, reverse=True)[: int(n_games)]
        acc = defaultdict(float)
        wsum = 0.0
        for gd, season, form, oo, od, w_extra in rows:
            w = (1.0 if season == current_season else self.prev_season_weight) * float(w_extra)
            for k, v in form.items():
                if k in ("pts_for", "pts_against", "win"):
                    continue
                if k in _DEFAULT_FORM:
                    acc[k] += float(v) * w
            wsum += w
        if wsum <= 0:
            return base
        out = dict(base)
        for k in _DEFAULT_FORM:
            if k in ("pts_for", "pts_against", "win", "pythag_win_pct", "actual_win_pct", "pythag_residual"):
                continue
            if k in acc:
                out[k] = acc[k] / wsum
        return out

    def multi_window_feature_dict(self, home_team, away_team, current_season, current_date,
                                   windows=None) -> dict:
        """Emit roll_N suffix features for multi-horizon form (3/5/10/20 + season)."""
        windows = windows or FORM_ROLL_WINDOWS
        keys = FORM_MULTIWINDOW_KEYS
        out = {}
        # Season ≈ full tracker window (existing get)
        h_season = self.get(home_team, current_season, current_date)
        a_season = self.get(away_team, current_season, current_date)
        for k in keys:
            hk, ak = f"h_{k}", f"a_{k}"
            out[f"{hk}_roll_season"] = float(h_season.get(k, _DEFAULT_FORM.get(k, 0.0)))
            out[f"{ak}_roll_season"] = float(a_season.get(k, _DEFAULT_FORM.get(k, 0.0)))
            out[f"{k}_diff_roll_season"] = out[f"{hk}_roll_season"] - out[f"{ak}_roll_season"]
        for n in windows:
            hf = self.get_window(home_team, current_season, current_date, n)
            af = self.get_window(away_team, current_season, current_date, n)
            suf = f"roll_{n}"
            for k in keys:
                hk, ak = f"h_{k}", f"a_{k}"
                out[f"{hk}_{suf}"] = float(hf.get(k, _DEFAULT_FORM.get(k, 0.0)))
                out[f"{ak}_{suf}"] = float(af.get(k, _DEFAULT_FORM.get(k, 0.0)))
                out[f"{k}_diff_{suf}"] = out[f"{hk}_{suf}"] - out[f"{ak}_{suf}"]
            # Recent scoring variance proxy from pts_for if present in history
            h_pts = []
            a_pts = []
            for gd, season, form, *_rest in list(self.history.get(home_team, [])):
                if current_date is not None and gd is not None and gd >= current_date:
                    continue
                h_pts.append(float(form.get("pts_for", 0.0)))
            for gd, season, form, *_rest in list(self.history.get(away_team, [])):
                if current_date is not None and gd is not None and gd >= current_date:
                    continue
                a_pts.append(float(form.get("pts_for", 0.0)))
            h_pts = h_pts[-n:] if h_pts else []
            a_pts = a_pts[-n:] if a_pts else []
            out[f"h_pts_std_{suf}"] = float(np.std(h_pts)) if len(h_pts) >= 2 else 0.0
            out[f"a_pts_std_{suf}"] = float(np.std(a_pts)) if len(a_pts) >= 2 else 0.0
        return out

    def save_state(self, path):
        import pickle
        with open(path, "wb") as f:
            pickle.dump({
                "window": self.window,
                "prev_season_weight": self.prev_season_weight,
                "history": {k: list(v) for k, v in self.history.items()},
                "league": list(self.league),
            }, f)

    @classmethod
    def load_state(cls, path):
        import pickle
        with open(path, "rb") as f:
            data = pickle.load(f)
        obj = cls(window=data["window"], prev_season_weight=data["prev_season_weight"])
        for k, v in data["history"].items():
            obj.history[k] = deque(v, maxlen=obj.window * 2)
        obj.league = deque(data.get("league", []), maxlen=obj.window * 30)
        return obj


def form_feature_dict(h_form: dict, a_form: dict) -> dict:
    """Build the model-facing team-form feature block from two team form dicts."""
    def _g(d, k):
        return d.get(k, _DEFAULT_FORM[k])
    return {
        "h_off_rtg": h_form["off_rtg"],
        "h_def_rtg": h_form["def_rtg"],
        "a_off_rtg": a_form["off_rtg"],
        "a_def_rtg": a_form["def_rtg"],
        # Matchup nets: home offense vs away defense, etc.
        "off_rtg_net": (h_form["off_rtg"] - a_form["def_rtg"]) - (a_form["off_rtg"] - h_form["def_rtg"]),
        "h_fg3m": h_form["fg3m"],
        "a_fg3m": a_form["fg3m"],
        "fg3m_diff": h_form["fg3m"] - a_form["fg3m"],
        "h_fg3pct": h_form["fg3pct"],
        "a_fg3pct": a_form["fg3pct"],
        "fg3pct_diff": h_form["fg3pct"] - a_form["fg3pct"],
        "h_blk": h_form["blk"],
        "a_blk": a_form["blk"],
        "blk_diff": h_form["blk"] - a_form["blk"],
        "h_ftov": h_form["ftov"],
        "a_ftov": a_form["ftov"],
        "ftov_diff": h_form["ftov"] - a_form["ftov"],
        # ── Four Factors + extra defensive/pressure metrics ──
        "h_efg": _g(h_form, "efg"), "a_efg": _g(a_form, "efg"),
        "efg_diff": _g(h_form, "efg") - _g(a_form, "efg"),
        "h_oreb_pct": _g(h_form, "oreb_pct"), "a_oreb_pct": _g(a_form, "oreb_pct"),
        "oreb_pct_diff": _g(h_form, "oreb_pct") - _g(a_form, "oreb_pct"),
        "h_dreb_pct": _g(h_form, "dreb_pct"), "a_dreb_pct": _g(a_form, "dreb_pct"),
        "dreb_pct_diff": _g(h_form, "dreb_pct") - _g(a_form, "dreb_pct"),
        "h_tov_pct": _g(h_form, "tov_pct"), "a_tov_pct": _g(a_form, "tov_pct"),
        "tov_pct_diff": _g(h_form, "tov_pct") - _g(a_form, "tov_pct"),
        "h_ftr": _g(h_form, "ftr"), "a_ftr": _g(a_form, "ftr"),
        "ftr_diff": _g(h_form, "ftr") - _g(a_form, "ftr"),
        "h_stl": _g(h_form, "stl"), "a_stl": _g(a_form, "stl"),
        "stl_diff": _g(h_form, "stl") - _g(a_form, "stl"),
        "h_fdrawn": _g(h_form, "fdrawn"), "a_fdrawn": _g(a_form, "fdrawn"),
        "fdrawn_diff": _g(h_form, "fdrawn") - _g(a_form, "fdrawn"),
        # ── Opponent/schedule-adjusted ratings ──
        "h_off_rtg_adj": h_form.get("off_rtg_adj", h_form["off_rtg"]),
        "a_off_rtg_adj": a_form.get("off_rtg_adj", a_form["off_rtg"]),
        "h_def_rtg_adj": h_form.get("def_rtg_adj", h_form["def_rtg"]),
        "a_def_rtg_adj": a_form.get("def_rtg_adj", a_form["def_rtg"]),
        # Adjusted matchup net: home off vs away def minus away off vs home def.
        "off_rtg_adj_net": (
            (h_form.get("off_rtg_adj", h_form["off_rtg"]) - a_form.get("def_rtg_adj", a_form["def_rtg"]))
            - (a_form.get("off_rtg_adj", a_form["off_rtg"]) - h_form.get("def_rtg_adj", h_form["def_rtg"]))
        ),
        "h_luck_adj_off": _g(h_form, "luck_adj_off"),
        "a_luck_adj_off": _g(a_form, "luck_adj_off"),
        "luck_adj_diff": _g(h_form, "luck_adj_off") - _g(a_form, "luck_adj_off"),
        # ── Pythagorean expectation (rolling, pre-game) ──
        "h_pythag_win_pct": _g(h_form, "pythag_win_pct"),
        "a_pythag_win_pct": _g(a_form, "pythag_win_pct"),
        "pythag_win_pct_diff": _g(h_form, "pythag_win_pct") - _g(a_form, "pythag_win_pct"),
        "h_pythag_residual": _g(h_form, "pythag_residual"),
        "a_pythag_residual": _g(a_form, "pythag_residual"),
        "pythag_residual_diff": _g(h_form, "pythag_residual") - _g(a_form, "pythag_residual"),
    }


# Original team-form features (pre-Four-Factors); kept separate so the ablation
# harness can measure the incremental value of the new box-score block.
FORM_FEATURE_COLS_BASE = [
    "h_off_rtg", "h_def_rtg", "a_off_rtg", "a_def_rtg", "off_rtg_net",
    "h_fg3m", "a_fg3m", "fg3m_diff",
    "h_fg3pct", "a_fg3pct", "fg3pct_diff",
    "h_blk", "a_blk", "blk_diff",
    "h_ftov", "a_ftov", "ftov_diff",
]

# New Four-Factors + defensive/pressure block.
FOUR_FACTOR_COLS = [
    "h_efg", "a_efg", "efg_diff",
    "h_oreb_pct", "a_oreb_pct", "oreb_pct_diff",
    "h_dreb_pct", "a_dreb_pct", "dreb_pct_diff",
    "h_tov_pct", "a_tov_pct", "tov_pct_diff",
    "h_ftr", "a_ftr", "ftr_diff",
    "h_stl", "a_stl", "stl_diff",
    "h_fdrawn", "a_fdrawn", "fdrawn_diff",
]

# Opponent/schedule-adjusted rating block.
OPP_ADJ_COLS = [
    "h_off_rtg_adj", "a_off_rtg_adj", "h_def_rtg_adj", "a_def_rtg_adj", "off_rtg_adj_net",
]

LUCK_COLS = ["h_luck_adj_off", "a_luck_adj_off", "luck_adj_diff"]

PYTHAG_COLS = [
    "h_pythag_win_pct", "a_pythag_win_pct", "pythag_win_pct_diff",
    "h_pythag_residual", "a_pythag_residual", "pythag_residual_diff",
]

FORM_FEATURE_COLS = FORM_FEATURE_COLS_BASE + FOUR_FACTOR_COLS + OPP_ADJ_COLS + LUCK_COLS + PYTHAG_COLS

# Columns produced by TeamFormTracker.multi_window_feature_dict
MULTIWINDOW_FORM_COLS = []
for _k in FORM_MULTIWINDOW_KEYS:
    MULTIWINDOW_FORM_COLS.extend([
        f"h_{_k}_roll_season", f"a_{_k}_roll_season", f"{_k}_diff_roll_season",
    ])
    for _n in FORM_ROLL_WINDOWS:
        MULTIWINDOW_FORM_COLS.extend([
            f"h_{_k}_roll_{_n}", f"a_{_k}_roll_{_n}", f"{_k}_diff_roll_{_n}",
        ])
for _n in FORM_ROLL_WINDOWS:
    MULTIWINDOW_FORM_COLS.extend([f"h_pts_std_roll_{_n}", f"a_pts_std_roll_{_n}"])
