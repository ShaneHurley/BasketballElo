"""Calibration slice registry (Task 053).

Phase 4B of the plan requires exactly one calibration path per target
(engine margin, final margin, cover probability, ML win probability,
interval width) and forbids fitting Elo isotonic, MetaScore isotonic,
static spread calibration, MetaWin isotonic, and ATS isotonic calibrators
all on the identical tail slice (currently ``calib_features`` in
``pipeline/backtest.py``, reused byte-for-byte at every one of those call
sites).

This module does not decide *how* to split calibration data — that is a
larger backtest.py control-flow change (flagged as remaining risk in this
phase's handoff, since it also feeds the Elo/hierarchical tuning control
flow owned by a different phase). It provides the machinery to (a) declare
one calibration path per target and (b) fail loudly the moment two
different targets are calibrated on the exact same row set, so the defect
can no longer regress silently once the production wiring is repaired.
"""
from __future__ import annotations

import hashlib

import numpy as np

# Phase 4B's five single-purpose calibration targets.
CALIBRATION_TARGETS = (
    "engine_margin",
    "final_margin",
    "cover_probability",
    "ml_win_probability",
    "interval_width",
)


class CalibrationRoleError(ValueError):
    """Raised when a target's calibration path is registered twice, or two
    different targets share the exact same calibration row slice."""


def _slice_fingerprint(row_ids) -> str:
    arr = np.asarray(sorted(np.asarray(row_ids).tolist()))
    return hashlib.sha256(arr.tobytes()).hexdigest()


class CalibrationSliceRegistry:
    """Tracks which row-id slice calibrated which target."""

    def __init__(self):
        self._by_target: dict[str, str] = {}
        self._fingerprint_to_targets: dict[str, list[str]] = {}

    def register(self, target: str, row_ids) -> None:
        if target in self._by_target:
            raise CalibrationRoleError(
                f"Target {target!r} already has a registered calibration "
                f"path (Task 053: exactly one calibration path per target)."
            )
        fp = _slice_fingerprint(row_ids)
        self._by_target[target] = fp
        self._fingerprint_to_targets.setdefault(fp, []).append(target)
        others = self._fingerprint_to_targets[fp]
        if len(others) > 1:
            raise CalibrationRoleError(
                f"Targets {others!r} were all calibrated on the identical "
                "row slice. Phase 4B forbids reusing the same calibration "
                "slice across multiple calibrators (Elo isotonic, "
                "MetaScore isotonic, static spread calibration, MetaWin "
                "isotonic, and ATS isotonic must not all fit on the same "
                "tail slice)."
            )

    def registered_targets(self) -> list:
        return list(self._by_target.keys())

    def assert_all_targets_registered(self, targets=CALIBRATION_TARGETS) -> None:
        missing = [t for t in targets if t not in self._by_target]
        if missing:
            raise CalibrationRoleError(
                f"Missing a declared calibration path for: {missing!r}"
            )


# `pipeline/backtest.py`'s five calibrators that previously all fit on the
# identical tail `calib_features` slice (Task 053's confirmed
# `calibration_slice_reuse` defect).
BACKTEST_CALIBRATION_TARGETS = (
    "elo_isotonic",
    "metascore_isotonic",
    "static_spread_calibration",
    "metawin_isotonic",
    "ats_isotonic",
)


def chronological_game_id_partition(
    calib_df, targets=BACKTEST_CALIBRATION_TARGETS, date_col: str = "game_date",
    id_col: str = "GAME_ID",
) -> dict:
    """Partition ``calib_df``'s ``GAME_ID``s, ordered by ``date_col``, into
    one contiguous, mutually disjoint block per target.

    This is the "give each calibrator its own out-of-sample slice" fix for
    the `calibration_slice_reuse` leak: every target gets a distinct set of
    rows from the same calibration-tail span, so no two calibrators can be
    fit on the identical row set (which the registry would otherwise reject
    the moment both are ``register()``ed), while every slice remains
    strictly within the pre-test-season calibration window (no chronological
    leakage relative to `test_season`).

    Returns ``{target: np.ndarray of GAME_ID}``, in the same iteration order
    as ``targets``, with sizes as close to equal as possible (any remainder
    games go to the earliest targets).
    """
    targets = list(targets)
    if calib_df is None or len(calib_df) == 0:
        return {t: np.array([]) for t in targets}
    ordered_ids = (
        calib_df[[id_col, date_col]]
        .drop_duplicates(id_col)
        .sort_values(date_col)[id_col]
        .to_numpy()
    )
    n = len(ordered_ids)
    k = len(targets)
    sizes = [n // k + (1 if i < n % k else 0) for i in range(k)]
    boundaries = [0]
    for s in sizes:
        boundaries.append(boundaries[-1] + s)
    return {
        target: ordered_ids[boundaries[i]:boundaries[i + 1]]
        for i, target in enumerate(targets)
    }


def slice_by_game_ids(df, game_ids, id_col: str = "GAME_ID"):
    """Return the rows of ``df`` whose ``id_col`` is in ``game_ids``."""
    if df is None or len(df) == 0:
        return df
    return df[df[id_col].isin(set(game_ids))].copy()


def build_disjoint_calibration_slices(
    calib_df, registry: CalibrationSliceRegistry,
    targets=BACKTEST_CALIBRATION_TARGETS, date_col: str = "game_date",
    id_col: str = "GAME_ID",
) -> dict:
    """Partition ``calib_df`` into one disjoint sub-DataFrame per target and
    register each slice's row ids with ``registry`` in one step.

    Raises ``CalibrationRoleError`` (via ``registry.register``) if this is
    ever called twice for the same target, or if a future edit makes two
    targets' slices overlap.
    """
    id_partition = chronological_game_id_partition(
        calib_df, targets=targets, date_col=date_col, id_col=id_col,
    )
    slices = {}
    for target, ids in id_partition.items():
        registry.register(target, ids)
        slices[target] = slice_by_game_ids(calib_df, ids, id_col=id_col)
    return slices
