"""Central routing for cover/win probability sources (legacy vs simplified).

Canonical ATS probability is P(home covers vs the T-60 decision line).
Chosen-side probability is derived exactly once via ``ats_ev.cover_prob_for_side``.
"""
from __future__ import annotations

import numpy as np

from pipeline.config import (
    CALIBRATION_MODE,
    REQUIRE_META_WIN_WHEN_FITTED,
    WIN_PROB_SOURCE,
)


def select_cover_prob(
    mode: str | None,
    *,
    raw_cover: float,
    calibrated_cover: float,
    ats_classifier_prob: float | None = None,
    blend_weight: float = 0.5,
) -> float:
    """Select the side-specific cover probability used for stakes/gates.

    ``ats_classifier_prob`` must already be converted to the chosen side
    (never pass raw home-cover here without ``cover_prob_for_side``).
    In ``beta_ats`` mode, fail closed to the margin/raw path when the
    classifier probability is missing.
    """
    mode = mode or CALIBRATION_MODE
    if mode == "simplified":
        base = float(raw_cover)
    elif mode == "beta_ats":
        if ats_classifier_prob is not None and np.isfinite(float(ats_classifier_prob)):
            base = float(ats_classifier_prob)
        else:
            # Fail closed: no classifier → margin/raw path, not a silent default.
            base = float(raw_cover)
    else:
        base = float(calibrated_cover)
    return float(np.clip(base, 0.01, 0.99))


def blend_ats_classifier_cover(
    spread_cover: float,
    ats_prob: float | None,
    blend: float = 0.5,
) -> float:
    """Blend margin-based and classifier side-cover probabilities."""
    if ats_prob is None or not np.isfinite(ats_prob):
        return float(spread_cover)
    w = float(np.clip(blend, 0.0, 1.0))
    return float((1.0 - w) * spread_cover + w * ats_prob)


def resolve_win_prob(
    *,
    margin_for_win: float,
    win_model=None,
    calibrator=None,
    meta_calibrate_fn=None,
    pred_total=None,
) -> float:
    source = WIN_PROB_SOURCE
    if source == "auto":
        if win_model is not None and getattr(win_model, "fitted", False) and REQUIRE_META_WIN_WHEN_FITTED:
            source = "meta_win"
        elif calibrator is not None:
            source = "rolling_platt"
        else:
            source = "margin_isotonic"

    if source == "meta_win" and win_model is not None and getattr(win_model, "fitted", False):
        raise RuntimeError("resolve_win_prob expects precomputed win_model output at call site")

    if source == "rolling_platt" and calibrator is not None:
        try:
            return float(calibrator.predict(margin_for_win, total=pred_total))
        except TypeError:
            return float(calibrator.predict(margin_for_win))

    if meta_calibrate_fn is not None:
        return float(meta_calibrate_fn(margin_for_win))

    return float(1.0 / (1.0 + np.exp(-margin_for_win / 12.0)))


def should_use_bet_calibrator(mode: str | None = None) -> bool:
    from pipeline.config import BET_SELECTION_MODE, CONFIDENCE_MODE, CONFIDENCE_SELECTION_MODE
    if BET_SELECTION_MODE == "confidence_only":
        return True
    if CONFIDENCE_MODE == "unified" or CONFIDENCE_SELECTION_MODE == "min_score":
        return True
    mode = mode or CALIBRATION_MODE
    return mode == "legacy_stack"


def should_use_season1_calibrator() -> bool:
    """Fit confidence calibrator on training calib split before the first simulated season."""
    from pipeline.config import BET_SELECTION_MODE, CONFIDENCE_MODE
    return BET_SELECTION_MODE == "confidence_only" or CONFIDENCE_MODE == "unified"


def should_use_rolling_platt_when_win_model(mode: str | None = None) -> bool:
    mode = mode or CALIBRATION_MODE
    if WIN_PROB_SOURCE == "rolling_platt":
        return True
    if mode == "simplified":
        return False
    return True
