"""Epic 8.2 / 12.5 — sparse possession RAPM with optional informed priors.

Player-level Ridge RAPM on stint rows (HOME/AWAY_players, pts, possessions).
Lineup L-RAPM shrinks toward an informed prior π via residualization
``y' = y - Xπ`` then ``β = β' + π`` (sklearn Ridge shrinks to zero only).
"""
from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable, Sequence

import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge, RidgeCV

# Scale player O/D RAPM (PPP units) to roughly point-scale feature magnitudes
# comparable to hapm_net (~points).
_PPP_TO_POINTS = 100.0


def _parse_lineup(s) -> list[str]:
    return [p for p in str(s).split("-") if p and p != "nan"]


def _player_keys(ids: Sequence[str], side: str) -> list[str]:
    return [f"{side}:{p}" for p in ids]


class PlayerRapmTracker:
    """Offline-fit player offense/defense RAPM; lineup lookup at feature time.

    Design matrix columns are ``off:<pid>`` / ``def:<pid>``. Each home-offense
    stint contributes y = home_pts/poss (or home_xpts/poss), weight=possessions,
    with home players as offense (+1) and away as defense (+1 on def columns).
    Away-offense stints are the mirror.
    """

    def __init__(
        self,
        *,
        alphas: Sequence[float] | None = None,
        alpha: float | None = None,
        league_xppp: float = 1.10,
        use_xpts: bool = True,
        min_rows: int = 200,
        solver: str = "lsqr",
    ):
        self.alphas = list(alphas) if alphas is not None else [
            1.0, 10.0, 50.0, 100.0, 250.0, 500.0, 1000.0, 2500.0,
        ]
        self.fixed_alpha = alpha
        self.league_xppp = float(league_xppp)
        self.use_xpts = bool(use_xpts)
        self.min_rows = int(min_rows)
        self.solver = solver
        self.fitted = False
        self.alpha_: float | None = None
        self.off_coef: dict[str, float] = {}
        self.def_coef: dict[str, float] = {}
        self.poss_count: dict[str, float] = {}
        self.intercept_: float = 0.0
        self._col_index: dict[str, int] = {}

    def _stint_rows(
        self, stints_df
    ) -> tuple[list[dict[str, float]], list[float], list[float], dict[str, float]]:
        rows: list[dict[str, float]] = []
        y: list[float] = []
        w: list[float] = []
        poss_count: dict[str, float] = defaultdict(float)
        for _, row in stints_df.iterrows():
            poss = float(row.get("possessions", 0) or 0)
            if poss < 1e-9:
                continue
            home = _parse_lineup(row.get("HOME_players", ""))
            away = _parse_lineup(row.get("AWAY_players", ""))
            if len(home) < 1 or len(away) < 1:
                continue
            if self.use_xpts:
                h_pts = float(row.get("home_xpts", row.get("home_pts", 0)) or 0)
                a_pts = float(row.get("away_xpts", row.get("away_pts", 0)) or 0)
            else:
                h_pts = float(row.get("home_pts", 0) or 0)
                a_pts = float(row.get("away_pts", 0) or 0)
            # Home offense
            feat_h: dict[str, float] = {}
            for p in home:
                feat_h[f"off:{p}"] = 1.0
                poss_count[p] += poss
            for p in away:
                feat_h[f"def:{p}"] = 1.0
                poss_count[p] += poss
            rows.append(feat_h)
            y.append(h_pts / poss - self.league_xppp)
            w.append(poss)
            # Away offense
            feat_a: dict[str, float] = {}
            for p in away:
                feat_a[f"off:{p}"] = 1.0
            for p in home:
                feat_a[f"def:{p}"] = 1.0
            rows.append(feat_a)
            y.append(a_pts / poss - self.league_xppp)
            w.append(poss)
        return rows, y, w, dict(poss_count)

    def _dict_to_csr(
        self, rows: list[dict[str, float]], col_index: dict[str, int] | None = None
    ) -> tuple[sparse.csr_matrix, dict[str, int]]:
        if col_index is None:
            names: list[str] = sorted({k for r in rows for k in r})
            col_index = {n: i for i, n in enumerate(names)}
        n_cols = len(col_index)
        data, indices, indptr = [], [], [0]
        for r in rows:
            for k, v in r.items():
                j = col_index.get(k)
                if j is None:
                    continue
                data.append(float(v))
                indices.append(j)
            indptr.append(len(data))
        X = sparse.csr_matrix(
            (np.asarray(data, dtype=float), indices, indptr),
            shape=(len(rows), n_cols),
        )
        return X, col_index

    def fit(self, stints_df) -> bool:
        self.fitted = False
        self.off_coef = {}
        self.def_coef = {}
        if stints_df is None or getattr(stints_df, "empty", True):
            return False
        rows, y, w, poss_count = self._stint_rows(stints_df)
        if len(rows) < self.min_rows:
            return False
        X, col_index = self._dict_to_csr(rows)
        self._col_index = col_index
        y_arr = np.asarray(y, dtype=float)
        w_arr = np.asarray(w, dtype=float)
        if self.fixed_alpha is not None:
            model = Ridge(
                alpha=float(self.fixed_alpha),
                fit_intercept=True,
                solver=self.solver,
            )
            model.fit(X, y_arr, sample_weight=w_arr)
            self.alpha_ = float(self.fixed_alpha)
        else:
            model = RidgeCV(
                alphas=np.asarray(self.alphas, dtype=float),
                fit_intercept=True,
                scoring="neg_mean_absolute_error",
            )
            # RidgeCV uses dense solvers for some paths; force via underlying
            # by fitting Ridge after alpha selection when matrix is sparse.
            model.fit(X, y_arr, sample_weight=w_arr)
            self.alpha_ = float(model.alpha_)
            # Refit with explicit sparse-friendly solver at chosen alpha
            model = Ridge(
                alpha=self.alpha_,
                fit_intercept=True,
                solver=self.solver,
            )
            model.fit(X, y_arr, sample_weight=w_arr)
        self.intercept_ = float(model.intercept_)
        inv = {i: n for n, i in col_index.items()}
        for j, coef in enumerate(np.asarray(model.coef_, dtype=float)):
            name = inv[j]
            if name.startswith("off:"):
                self.off_coef[name[4:]] = float(coef)
            elif name.startswith("def:"):
                self.def_coef[name[4:]] = float(coef)
        self.poss_count = poss_count
        self.fitted = True
        return True

    def player_off(self, pid: str) -> float:
        return float(self.off_coef.get(str(pid), 0.0))

    def player_def(self, pid: str) -> float:
        return float(self.def_coef.get(str(pid), 0.0))

    def lineup_off_prior(self, lineup: Iterable) -> float:
        ids = [str(x) for x in lineup if x and str(x) != "nan"]
        if not ids:
            return 0.0
        return float(sum(self.player_off(p) for p in ids))

    def lineup_def_prior(self, lineup: Iterable) -> float:
        ids = [str(x) for x in lineup if x and str(x) != "nan"]
        if not ids:
            return 0.0
        return float(sum(self.player_def(p) for p in ids))

    def lineup_net_ppp(self, lineup: Iterable) -> float:
        """Offense PPP contribution minus opponent-facing defense (higher better)."""
        return self.lineup_off_prior(lineup) - self.lineup_def_prior(lineup)

    def feature_dict(self, home_lineup, away_lineup, game_id=None) -> dict[str, float]:
        if not self.fitted:
            return {"h_rapm_net": 0.0, "a_rapm_net": 0.0, "rapm_net_diff": 0.0}
        h = self.lineup_net_ppp(home_lineup) * _PPP_TO_POINTS
        a = self.lineup_net_ppp(away_lineup) * _PPP_TO_POINTS
        return {
            "h_rapm_net": float(h),
            "a_rapm_net": float(a),
            "rapm_net_diff": float(h - a),
        }


class LineupRapmTracker:
    """Epic 12.5 — lineup-level Ridge shrunk toward player-RAPM informed prior π.

    Fits on home-lineup stints with y = home_xpts/poss - league. Prior for
    lineup λ is sum of player off coefficients from a fitted ``PlayerRapmTracker``.
    Residualizes ``y' = y - π_row`` so standard Ridge(β') shrinks toward 0 in
    offset space; recover ``β = β' + π``.
    """

    def __init__(
        self,
        player_rapm: PlayerRapmTracker,
        *,
        alphas: Sequence[float] | None = None,
        alpha: float | None = 100.0,
        league_xppp: float = 1.10,
        min_rows: int = 100,
        solver: str = "lsqr",
    ):
        self.player_rapm = player_rapm
        self.alphas = list(alphas) if alphas is not None else [
            10.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
        ]
        self.fixed_alpha = alpha
        self.league_xppp = float(league_xppp)
        self.min_rows = int(min_rows)
        self.solver = solver
        self.fitted = False
        self.lineup_off: dict[str, float] = {}
        self.lineup_prior: dict[str, float] = {}
        self.alpha_: float | None = None

    @staticmethod
    def _lineup_key(ids: Sequence[str]) -> str:
        return "-".join(sorted(str(p) for p in ids))

    def fit(self, stints_df) -> bool:
        self.fitted = False
        self.lineup_off = {}
        self.lineup_prior = {}
        if not getattr(self.player_rapm, "fitted", False):
            return False
        if stints_df is None or getattr(stints_df, "empty", True):
            return False
        keys: list[str] = []
        y: list[float] = []
        w: list[float] = []
        pi: list[float] = []
        for _, row in stints_df.iterrows():
            poss = float(row.get("possessions", 0) or 0)
            if poss < 1e-9:
                continue
            home = _parse_lineup(row.get("HOME_players", ""))
            if len(home) < 2:
                continue
            key = self._lineup_key(home)
            xpts = float(row.get("home_xpts", row.get("home_pts", 0)) or 0)
            prior = self.player_rapm.lineup_off_prior(home)
            keys.append(key)
            y.append(xpts / poss - self.league_xppp)
            w.append(poss)
            pi.append(prior)
            self.lineup_prior[key] = prior
        if len(keys) < self.min_rows:
            return False
        uniq = sorted(set(keys))
        col = {k: i for i, k in enumerate(uniq)}
        data, indices, indptr = [], [], [0]
        for k in keys:
            data.append(1.0)
            indices.append(col[k])
            indptr.append(len(data))
        X = sparse.csr_matrix(
            (np.asarray(data), indices, indptr),
            shape=(len(keys), len(uniq)),
        )
        y_arr = np.asarray(y, dtype=float)
        pi_arr = np.asarray(pi, dtype=float)
        w_arr = np.asarray(w, dtype=float)
        # y' = y - Xπ  (each row has a single 1, so Xπ_i = π_i)
        y_prime = y_arr - pi_arr
        if self.fixed_alpha is not None:
            alpha = float(self.fixed_alpha)
        else:
            cv = RidgeCV(
                alphas=np.asarray(self.alphas, dtype=float),
                fit_intercept=False,
                scoring="neg_mean_absolute_error",
            )
            cv.fit(X, y_prime, sample_weight=w_arr)
            alpha = float(cv.alpha_)
        model = Ridge(
            alpha=alpha,
            fit_intercept=False,
            solver=self.solver,
        )
        model.fit(X, y_prime, sample_weight=w_arr)
        self.alpha_ = alpha
        beta_prime = np.asarray(model.coef_, dtype=float)
        for k, j in col.items():
            self.lineup_off[k] = float(beta_prime[j] + self.lineup_prior.get(k, 0.0))
        self.fitted = True
        return True

    def lineup_off_rating(self, lineup: Iterable) -> float:
        ids = [str(x) for x in lineup if x and str(x) != "nan"]
        if not ids:
            return 0.0
        key = self._lineup_key(ids)
        if key in self.lineup_off:
            return float(self.lineup_off[key])
        # Unseen lineup: fall back to informed prior only (prior dominates)
        return float(self.player_rapm.lineup_off_prior(ids))

    def feature_dict(self, home_lineup, away_lineup, game_id=None) -> dict[str, float]:
        if not self.fitted and not getattr(self.player_rapm, "fitted", False):
            return {
                "h_lrapm_net": 0.0,
                "a_lrapm_net": 0.0,
                "lrapm_net_diff": 0.0,
            }
        h = self.lineup_off_rating(home_lineup) * _PPP_TO_POINTS
        a = self.lineup_off_rating(away_lineup) * _PPP_TO_POINTS
        return {
            "h_lrapm_net": float(h),
            "a_lrapm_net": float(a),
            "lrapm_net_diff": float(h - a),
        }


def build_past_only_rapm_trackers(
    stints_df,
    game_dates: dict,
    *,
    n_blocks: int = 6,
    league_xppp: float = 1.10,
    with_lineup_prior: bool = True,
) -> tuple[dict, dict]:
    """Chronological past-only RAPM (+ optional L-RAPM) per game block."""
    if not game_dates:
        return {}, {}
    ordered = sorted(game_dates, key=lambda g: game_dates[g])
    n = len(ordered)
    n_blocks = max(1, min(n_blocks, n))
    sizes = [n // n_blocks + (1 if i < n % n_blocks else 0) for i in range(n_blocks)]
    bounds = [0]
    for s in sizes:
        bounds.append(bounds[-1] + s)
    blocks = [ordered[bounds[i]:bounds[i + 1]] for i in range(n_blocks)]
    game_id_to_tracker: dict[Any, Any] = {}
    block_fit_max_ts: dict[Any, Any] = {}
    prior_gids: list = []
    prior_max_ts = None
    for block in blocks:
        if prior_gids:
            player = PlayerRapmTracker(league_xppp=league_xppp)
            past = stints_df[stints_df["GAME_ID"].isin(prior_gids)]
            player.fit(past)
            if with_lineup_prior and player.fitted:
                lineup = LineupRapmTracker(player, league_xppp=league_xppp)
                lineup.fit(past)
                tracker: Any = _RapmBundle(player, lineup if lineup.fitted else None)
            else:
                tracker = player
        else:
            tracker = PlayerRapmTracker(league_xppp=league_xppp)
        for gid in block:
            game_id_to_tracker[gid] = tracker
            block_fit_max_ts[gid] = prior_max_ts
        prior_gids = prior_gids + list(block)
        block_max = max((game_dates[g] for g in block), default=prior_max_ts)
        prior_max_ts = block_max if prior_max_ts is None else max(prior_max_ts, block_max)
    return game_id_to_tracker, block_fit_max_ts


class _RapmBundle:
    """Feature facade emitting both player RAPM and L-RAPM columns."""

    def __init__(self, player: PlayerRapmTracker, lineup: LineupRapmTracker | None):
        self.player = player
        self.lineup = lineup
        self.fitted = bool(getattr(player, "fitted", False))

    def feature_dict(self, home_lineup, away_lineup, game_id=None) -> dict[str, float]:
        out = self.player.feature_dict(home_lineup, away_lineup, game_id=game_id)
        if self.lineup is not None:
            out.update(self.lineup.feature_dict(home_lineup, away_lineup, game_id=game_id))
        else:
            out.setdefault("h_lrapm_net", 0.0)
            out.setdefault("a_lrapm_net", 0.0)
            out.setdefault("lrapm_net_diff", 0.0)
        return out
