"""External player impact priors (EPM/RAPTOR-style CSV blend).

Task 032: every player-metric snapshot now carries an explicit
``observation_date`` (the date the external metric provider computed/
published that value). Snapshots are versioned per player -- a provider may
publish several dated EPM values for the same player over a season -- and an
as-of query (`as_of=gdate`) may only see a snapshot whose
``observation_date <= as_of``. A retrospectively updated/future metric value
must never be joinable to a game that happened before it existed (Rule 3).

When no snapshot qualifies as of the requested cutoff, callers get an
explicit "missing" signal (via `missing()` / the `*_missing` feature flags)
rather than a silent zero, so a genuinely-unavailable metric is
distinguishable from a metric whose real value happens to be zero.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from pipeline.config import STATE_DIR

EPM_CSV = STATE_DIR / "player_epm_priors.csv"
BLEND_WEIGHT = 0.35

REQUIRED_COLUMNS = ("player_id", "epm", "observation_date")


class EpmPriorSchemaError(ValueError):
    """Raised when a raw EPM snapshot CSV is missing required versioning columns."""


class EpmPriorTracker:
    """Blend internal player Elo with external, date-versioned impact metrics."""

    def __init__(self, csv_path=None, blend: float = BLEND_WEIGHT):
        self.blend = blend
        # player_id -> sorted list of (observation_date, epm), ascending.
        self.snapshots: Dict[str, List[Tuple[pd.Timestamp, float]]] = {}
        path = Path(csv_path) if csv_path else EPM_CSV
        if path.exists():
            try:
                df = pd.read_csv(path)
                self._load_versioned(df)
            except EpmPriorSchemaError:
                raise
            except Exception:
                self.snapshots = {}

    def _load_versioned(self, df: pd.DataFrame) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
        if missing:
            raise EpmPriorSchemaError(
                f"EPM prior CSV missing required versioning column(s) {missing}; "
                "every player-metric snapshot must carry an observation_date "
                "(Task 032) -- refusing to load an unversioned prior table"
            )
        d = df.copy()
        d["player_id"] = d["player_id"].astype(str)
        d["epm"] = pd.to_numeric(d["epm"], errors="coerce")
        d["observation_date"] = pd.to_datetime(d["observation_date"], errors="coerce")
        n_before = len(d)
        d = d.dropna(subset=["epm", "observation_date"])
        if len(d) < n_before:
            import warnings
            warnings.warn(
                f"EpmPriorTracker: dropped {n_before - len(d)} row(s) with "
                "unparseable epm/observation_date rather than fabricate one",
                stacklevel=2,
            )
        for pid, grp in d.groupby("player_id"):
            rows = sorted(
                (pd.Timestamp(r.observation_date), float(r.epm)) for r in grp.itertuples()
            )
            self.snapshots[pid] = rows

    @property
    def priors(self) -> Dict[str, float]:
        """Backward-compatible view: latest known value per player,
        ignoring any as-of cutoff. Prefer `get_as_of` for anything that
        touches a specific game/prediction."""
        return {pid: rows[-1][1] for pid, rows in self.snapshots.items() if rows}

    def get_as_of(self, player_id: str, as_of=None) -> Optional[float]:
        """Latest EPM snapshot with ``observation_date <= as_of``.

        `as_of=None` means "no cutoff" (latest available value) -- callers
        that have a real prediction timestamp must always pass it explicitly
        so a future snapshot cannot leak into a past game.
        """
        rows = self.snapshots.get(str(player_id))
        if not rows:
            return None
        if as_of is None:
            return rows[-1][1]
        cutoff = pd.Timestamp(as_of)
        eligible = [v for d, v in rows if d <= cutoff]
        if not eligible:
            return None
        return eligible[-1]

    def missing(self, player_id: str, as_of=None) -> bool:
        return self.get_as_of(player_id, as_of=as_of) is None

    def scaled_impact(self, player_id: str, as_of=None) -> float:
        epm = self.get_as_of(player_id, as_of=as_of)
        if epm is None:
            return 0.0
        return epm * 50.0  # scale to Elo-like units

    def blend_lineup_off(
        self, player_ids, internal_off: float, as_of=None, minutes=None,
    ) -> float:
        if not self.snapshots:
            return internal_off
        ids = [p for p in player_ids if p and not self.missing(p, as_of=as_of)]
        if not ids:
            return internal_off
        impacts = [self.scaled_impact(p, as_of=as_of) for p in ids]

        weights = None
        if minutes is not None:
            w = []
            complete = True
            getter = minutes.get if hasattr(minutes, "get") else None
            for p in ids:
                mv = getter(p) if getter is not None else None
                try:
                    mv_f = float(mv) if mv is not None and pd.notna(mv) else None
                except (TypeError, ValueError):
                    mv_f = None
                if mv_f is None or not np.isfinite(mv_f):
                    complete = False
                    break
                w.append(max(0.0, mv_f))
            if complete:
                weights = w

        if weights is None:
            ext = 1500.0 + float(np.mean(impacts))
            return (1 - self.blend) * internal_off + self.blend * ext

        wsum = float(sum(weights))
        if wsum <= 0:
            ext = 1500.0 + float(np.mean(impacts))
            return (1 - self.blend) * internal_off + self.blend * ext
        ext = 1500.0 + float(np.average(impacts, weights=weights))
        # Starter-load (~36 min) takes the full blend; bench minutes shrink
        # how much external EPM moves the lineup (Epic 7.4).
        starter_ref = 36.0
        avg_min = wsum / len(weights)
        eff_blend = float(np.clip(self.blend * (avg_min / starter_ref), 0.0, 1.0))
        return (1 - eff_blend) * internal_off + eff_blend * ext

    def feature_dict(self, home_ids, away_ids, ho_off, ao_off, as_of=None):
        if not self.snapshots:
            return {
                "epm_prior_diff": 0.0, "epm_blend_off_diff": 0.0,
                "epm_missing_frac_home": 1.0, "epm_missing_frac_away": 1.0,
            }
        home_ids = [p for p in (home_ids or []) if p]
        away_ids = [p for p in (away_ids or []) if p]

        def _side(ids):
            vals, n_missing = [], 0
            for p in ids:
                v = self.get_as_of(p, as_of=as_of)
                if v is None:
                    n_missing += 1
                else:
                    vals.append(self.scaled_impact(p, as_of=as_of))
            missing_frac = (n_missing / len(ids)) if ids else 1.0
            mean_val = float(np.mean(vals)) if vals else 0.0
            return mean_val, missing_frac

        h_mean, h_missing_frac = _side(home_ids)
        a_mean, a_missing_frac = _side(away_ids)
        return {
            "epm_prior_diff": h_mean - a_mean,
            "epm_blend_off_diff": self.blend * (h_mean - a_mean),
            "epm_missing_frac_home": h_missing_frac,
            "epm_missing_frac_away": a_missing_frac,
        }
