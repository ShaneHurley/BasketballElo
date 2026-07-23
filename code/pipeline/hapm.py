"""Batch HAPM (sparse Ridge) priors for dyad/trio chemistry — optional meta feature."""
from __future__ import annotations

import itertools
from collections import defaultdict

import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import Ridge

SHRINK = 80.0


class HapmPriorTracker:
    """Offline-fit dyad/trio coefficients; lookup at feature time."""

    def __init__(self, alpha: float = 100.0, league_xppp: float = 1.10):
        self.alpha = alpha
        self.league_xppp = league_xppp
        self.vectorizer = DictVectorizer(sparse=True)
        self.model = Ridge(alpha=alpha, fit_intercept=True)
        self.fitted = False
        self._dyad_coef = {}
        self._trio_coef = {}

    @staticmethod
    def _parse_lineup(s) -> list[str]:
        return [p for p in str(s).split("-") if p and p != "nan"]

    def fit(self, stints_df) -> bool:
        if stints_df is None or stints_df.empty:
            return False
        rows, y, w = [], [], []
        for _, row in stints_df.iterrows():
            poss = float(row.get("possessions", 0) or 0)
            if poss < 1:
                continue
            xh = float(row.get("home_xpts", 0) or 0) / poss
            feat = {}
            home = self._parse_lineup(row.get("HOME_players", ""))
            for a, b in itertools.combinations(home, 2):
                feat[f"d:{min(a,b)}|{max(a,b)}"] = 1
            if len(home) >= 3:
                for trio in itertools.combinations(sorted(home), 3):
                    feat[f"t:{trio[0]}|{trio[1]}|{trio[2]}"] = 1
            if not feat:
                continue
            rows.append(feat)
            y.append(xh - self.league_xppp)
            w.append(poss)
        if len(rows) < 100:
            return False
        X = self.vectorizer.fit_transform(rows)
        self.model.fit(X, np.array(y), sample_weight=np.array(w))
        self.fitted = True
        names = self.vectorizer.get_feature_names_out()
        for name, coef in zip(names, self.model.coef_):
            if name.startswith("d:"):
                self._dyad_coef[name[2:]] = float(coef)
            elif name.startswith("t:"):
                self._trio_coef[name[2:]] = float(coef)
        return True

    def _lineup_prior(self, lineup) -> float:
        if not self.fitted:
            return 0.0
        ids = [str(x) for x in lineup if x and str(x) != "nan"]
        vals = []
        for a, b in itertools.combinations(ids, 2):
            k = f"{min(a,b)}|{max(a,b)}"
            if k in self._dyad_coef:
                vals.append(self._dyad_coef[k] * (SHRINK / (SHRINK + 50)))
        if len(ids) >= 3:
            for trio in itertools.combinations(sorted(ids), 3):
                k = f"{trio[0]}|{trio[1]}|{trio[2]}"
                if k in self._trio_coef:
                    vals.append(self._trio_coef[k] * (SHRINK / (SHRINK + 50)))
        return float(np.mean(vals) * 100.0) if vals else 0.0

    def feature_dict(self, home_lineup, away_lineup, game_id=None) -> dict:
        # `game_id` is accepted (and ignored) so this tracker is a drop-in
        # replacement for `PastOnlyHapmLookup.feature_dict` below at every
        # call site (`pipeline/game_features.py`), regardless of whether a
        # single static tracker or a per-game past-only lookup is in use.
        h = self._lineup_prior(home_lineup)
        a = self._lineup_prior(away_lineup)
        return {"h_hapm_net": h, "a_hapm_net": a, "hapm_net_diff": h - a}


def build_past_only_hapm_trackers(stints_df, game_dates: dict, n_blocks: int = 6, alpha: float = 100.0):
    """Sequence of past-only-fit HAPM trackers (Task 052).

    ``HapmPriorTracker.fit`` is a single batch Ridge fit; calling it once on
    an entire season and then using the same fitted tracker as a static
    lookup for every game in that season (the current production pattern in
    ``pipeline/backtest.py``) means an early-season game's HAPM feature is
    informed by dyad/trio coefficients estimated from *later* games in the
    same span — the ``HAPM-before-split`` leak named in the plan's
    "Immediate trust assessment".

    This helper performs the "past-only cross-fitting" the plan calls for:
    games are grouped into ``n_blocks`` chronological blocks (by
    ``game_dates``); a fresh tracker is fit on stints from blocks strictly
    before block ``i`` and used only for games in block ``i``. Block 0 has
    no strictly-earlier block, so it gets an unfitted tracker (returns the
    same all-zero feature dict as ``use_hapm_priors=False`` — never a
    silently invented prior).

    Parameters
    ----------
    stints_df : DataFrame with ``GAME_ID`` and the stint columns
        ``HapmPriorTracker.fit`` consumes.
    game_dates : dict mapping GAME_ID -> orderable timestamp for every game
        that will request a tracker (not just games present in stints_df).
    n_blocks : number of chronological blocks.

    Returns
    -------
    (game_id_to_tracker, block_fit_max_timestamp) : dict, dict
        ``game_id_to_tracker[gid]`` is the tracker whose ``.feature_dict``
        must be used for that game; ``block_fit_max_timestamp[gid]`` is the
        max timestamp of stints used to fit it (``None`` for block 0 —
        Rule 5: never invent a prior instead of reporting "no fit yet").
    """
    if not game_dates:
        return {}, {}
    ordered_gids = sorted(game_dates, key=lambda g: game_dates[g])
    n = len(ordered_gids)
    n_blocks = max(1, min(n_blocks, n))
    block_sizes = [n // n_blocks + (1 if i < n % n_blocks else 0) for i in range(n_blocks)]
    boundaries = [0]
    for bs in block_sizes:
        boundaries.append(boundaries[-1] + bs)
    blocks = [ordered_gids[boundaries[i]:boundaries[i + 1]] for i in range(n_blocks)]

    game_id_to_tracker: dict = {}
    block_fit_max_timestamp: dict = {}
    prior_gids: list = []
    prior_max_ts = None
    for block in blocks:
        if prior_gids:
            tracker = HapmPriorTracker(alpha=alpha)
            past_stints = stints_df[stints_df["GAME_ID"].isin(prior_gids)]
            tracker.fit(past_stints)
        else:
            tracker = HapmPriorTracker(alpha=alpha)  # unfitted: zero features
        for gid in block:
            game_id_to_tracker[gid] = tracker
            block_fit_max_timestamp[gid] = prior_max_ts
        prior_gids = prior_gids + block
        block_max = max((game_dates[g] for g in block), default=prior_max_ts)
        prior_max_ts = block_max if prior_max_ts is None else max(prior_max_ts, block_max)
    return game_id_to_tracker, block_fit_max_timestamp


class PastOnlyHapmLookup:
    """Feature-time facade over the per-game past-only-fit trackers from
    ``build_past_only_hapm_trackers`` (integration fix for the
    ``hapm_before_split_leak`` registry entry).

    ``pipeline/game_features.py`` calls ``hapm_tracker.feature_dict(...)``
    with a single tracker shared across every game in a feature-generation
    pass. That single-tracker interface is exactly the leak: one Ridge fit
    reused as a static lookup for every game in the span, including games
    strictly earlier than some of the games it was fit on. This class keeps
    the same call signature but dispatches each call to whichever
    strictly-past-only block tracker is correct for that specific
    ``game_id``, so no early game's HAPM feature can depend on a later
    game's stints.

    ``self.fitted`` is always ``True`` so ``pipeline/game_features.py``'s
    ``getattr(hapm_tracker, "fitted", False)`` guard always dispatches into
    ``feature_dict`` — per-game "no strictly-earlier block yet" handling
    (Rule 5: never invent a prior) happens inside ``feature_dict`` itself,
    which returns the same all-zero dict as ``use_hapm_priors=False``.
    """

    def __init__(self, game_id_to_tracker: dict, fit_max_timestamp: dict):
        self.game_id_to_tracker = game_id_to_tracker
        self.fit_max_timestamp = fit_max_timestamp
        self.fitted = True

    def feature_dict(self, home_lineup, away_lineup, game_id=None) -> dict:
        tracker = self.game_id_to_tracker.get(game_id)
        if tracker is None or not tracker.fitted:
            return {"h_hapm_net": 0.0, "a_hapm_net": 0.0, "hapm_net_diff": 0.0}
        return tracker.feature_dict(home_lineup, away_lineup, game_id=game_id)


def hapm_lookup_for_training(stints_df, n_blocks: int = 6, alpha: float = 100.0) -> "PastOnlyHapmLookup":
    """Build the past-only HAPM lookup for ``pipeline/backtest.py``'s
    feature-generation call sites.

    Both the base-training slice and the calibration slice fed to
    ``pipeline.features.generate_features`` are subsets of the same
    ``train_stints`` span, so a single lookup built from the full span
    covers both call sites correctly — each game still only ever resolves
    to a tracker fit on strictly-earlier games within that span.
    """
    if stints_df is None or stints_df.empty:
        return PastOnlyHapmLookup({}, {})
    game_dates = (
        stints_df[["GAME_ID", "game_date"]]
        .drop_duplicates("GAME_ID")
        .set_index("GAME_ID")["game_date"]
        .to_dict()
    )
    trackers, fit_max_ts = build_past_only_hapm_trackers(
        stints_df, game_dates, n_blocks=n_blocks, alpha=alpha,
    )
    return PastOnlyHapmLookup(trackers, fit_max_ts)
