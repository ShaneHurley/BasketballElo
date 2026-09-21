"""Epic 10.2.4 — synthetic formula test bench: calibration invariants.

Four invariants across ``pipeline/elo_calibration.py``,
``pipeline/venn_abers.py``, and ``pipeline/calibration_registry.py``:

1. P0.4 — ``WalkForwardEloCalibrator`` interval responds to observed residuals
   (the ``predict_interval`` / ``update_residuals`` API still exists in
   ``elo_calibration.py``; it was NOT deleted in favor of
   ``market.SpreadCalibrator.predict_interval``, so the direct path is tested).
2. P0.9 — ``passes_venn_abers_filter`` fails closed on None/NaN width
   (independent re-assertion; deliberately does not import
   ``test_bench_p0_venn_abers.py``).
3. Isotonic min-sample floor — ``venn_abers.scores_to_interval`` raises
   ``ValueError`` when ``len(scores_cal) < 10`` (roadmap 7.6).
4. Disjoint-slice minimum-N — ``chronological_game_id_partition`` raises when
   any slice has ``len < 5`` (24 games / 5 targets); 25 games sits on the
   floor and is accepted (roadmap 7.6).

Adversarial case: isotonic fed 2 points with identical scores but different
labels must produce a sensible probability in [0, 1] (or a documented
degenerate step) without crashing.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.calibration_registry import (
    BACKTEST_CALIBRATION_TARGETS,
    chronological_game_id_partition,
)
from pipeline.elo_calibration import WalkForwardEloCalibrator
from pipeline.venn_abers import passes_venn_abers_filter, scores_to_interval

# ---------------------------------------------------------------------------
# 1. P0.4 — interval responds to residuals
# ---------------------------------------------------------------------------


@pytest.mark.bench
def test_interval_responds_to_residuals_p0_4():
    """Default interval is ±12 (width 24); after update_residuals it must move."""
    cal = WalkForwardEloCalibrator()
    lo0, hi0, w0 = cal.predict_interval({"elo_margin": 2.0})
    assert w0 == pytest.approx(24.0)
    assert lo0 == pytest.approx(2.0 - 12.0)
    assert hi0 == pytest.approx(2.0 + 12.0)

    cal.update_residuals([5.0] * 30)
    lo1, hi1, w1 = cal.predict_interval({"elo_margin": 2.0})

    # Core invariant: width must differ from the 24.0 default.
    assert w1 != pytest.approx(24.0)
    # Tightened (adversarial protocol): "differs from 24" alone would pass on a
    # garbage width (0, negative, NaN, off-center). Pin the exact contract:
    # 90th-percentile of |5.0| is 5.0, so width must be exactly 10.0.
    assert w1 == pytest.approx(10.0)
    assert np.isfinite(w1) and w1 > 0.0
    assert lo1 < hi1
    # Interval must stay centered on the point prediction.
    pred = cal.predict({"elo_margin": 2.0})
    assert lo1 == pytest.approx(pred - w1 / 2.0)
    assert hi1 == pytest.approx(pred + w1 / 2.0)


@pytest.mark.bench
def test_interval_update_is_sign_invariant_and_gated_p0_4():
    """Residual sign must not matter (quantile of |residual|), and the update
    must be gated at >= 20 residuals — otherwise 'width responds to residuals'
    is vacuous (any tiny sample would move it)."""
    cal_neg = WalkForwardEloCalibrator()
    cal_neg.update_residuals([-5.0] * 30)
    _, _, w_neg = cal_neg.predict_interval({"elo_margin": 0.0})
    assert w_neg == pytest.approx(10.0)  # same as +5.0 residuals

    cal_few = WalkForwardEloCalibrator()
    cal_few.update_residuals([100.0] * 19)  # below the documented min of 20
    _, _, w_few = cal_few.predict_interval({"elo_margin": 0.0})
    assert w_few == pytest.approx(24.0)  # default untouched

    # Determinism: two fresh calibrators agree exactly.
    a = WalkForwardEloCalibrator().predict_interval({"elo_margin": 1.0})
    b = WalkForwardEloCalibrator().predict_interval({"elo_margin": 1.0})
    assert a == b


@pytest.mark.bench
def test_predict_interval_honors_alpha():
    """After residuals exist, alpha must change the quantile band (not stay 90%)."""
    cal = WalkForwardEloCalibrator()
    cal.update_residuals([5.0] * 30)
    _, _, w90_const = cal.predict_interval({"elo_margin": 0.0}, alpha=0.10)
    assert w90_const == pytest.approx(10.0)

    mixed = list(range(1, 31))
    cal2 = WalkForwardEloCalibrator()
    cal2.update_residuals(mixed)
    _, _, w_lo = cal2.predict_interval({"elo_margin": 0.0}, alpha=0.10)
    _, _, w_hi = cal2.predict_interval({"elo_margin": 0.0}, alpha=0.50)
    assert w_lo > w_hi
    assert w_lo == pytest.approx(2.0 * float(np.quantile(np.abs(mixed), 0.90)))
    assert w_hi == pytest.approx(2.0 * float(np.quantile(np.abs(mixed), 0.50)))



# ---------------------------------------------------------------------------
# 2. P0.9 — Venn-Abers filter fails closed (independent assertion)
# ---------------------------------------------------------------------------


@pytest.mark.bench
def test_venn_abers_filter_fails_closed_p0_9():
    """A risk-limiting filter that cannot evaluate width must reject."""
    assert passes_venn_abers_filter(None, max_width=0.25) is False
    assert passes_venn_abers_filter(float("nan"), max_width=0.25) is False
    assert passes_venn_abers_filter(np.nan, max_width=0.25) is False
    assert passes_venn_abers_filter(float("inf"), max_width=0.25) is False
    # Guard against the trivial counterexample "always return False": a finite,
    # in-limit width must still be accepted.
    assert passes_venn_abers_filter(0.10, max_width=0.25) is True
    assert passes_venn_abers_filter(0.40, max_width=0.25) is False


# ---------------------------------------------------------------------------
# 3. Isotonic min-sample floor (roadmap 7.6 — currently missing)
# ---------------------------------------------------------------------------


@pytest.mark.bench
def test_scores_to_interval_enforces_min_samples():
    """With only 3 calibration points, scores_to_interval must refuse."""
    scores_cal = np.array([0.2, 0.5, 0.8])
    labels_cal = np.array([0, 1, 1])
    with pytest.raises(ValueError, match="at least 10"):
        scores_to_interval(scores_cal, labels_cal, np.array([0.55]))


# ---------------------------------------------------------------------------
# 4. Disjoint-slice minimum-N (roadmap 7.6 — currently missing)
# ---------------------------------------------------------------------------


def _shuffled_games_df(n_games: int, seed: int) -> pd.DataFrame:
    """Deterministic shuffled game table (proves the partition sorts by date)."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2025-01-01", periods=n_games, freq="D")
    ids = np.array([f"GAME{i:03d}" for i in range(n_games)])
    order = rng.permutation(n_games)
    return pd.DataFrame({"GAME_ID": ids[order], "game_date": dates[order]})


@pytest.mark.bench
def test_partition_enforces_per_slice_min_n():
    """24 games / 5 targets leaves a 4-game slice; must raise. 25 games is
    exactly min_slice_n=5 per target and is accepted."""
    assert len(BACKTEST_CALIBRATION_TARGETS) == 5
    df24 = _shuffled_games_df(24, seed=0)
    with pytest.raises(ValueError, match="min_slice_n"):
        chronological_game_id_partition(df24)
    df25 = _shuffled_games_df(25, seed=0)
    part = chronological_game_id_partition(df25)
    assert all(len(ids) == 5 for ids in part.values())


@pytest.mark.bench
def test_partition_is_disjoint_and_chronological_with_adequate_n():
    """Sanity (non-xfail): with adequate N the partition machinery itself is
    sound — disjoint, contiguous, date-ordered slices, equal sizes."""
    n_games = 50
    df = _shuffled_games_df(n_games, seed=3)
    part = chronological_game_id_partition(df)

    assert set(part.keys()) == set(BACKTEST_CALIBRATION_TARGETS)
    all_ids = np.concatenate(list(part.values()))
    assert len(all_ids) == n_games
    assert len(set(all_ids)) == n_games  # mutually disjoint
    assert all(len(v) == n_games // 5 for v in part.values())  # 10 each

    # Contiguous chronological blocks: concatenating slices in target order
    # must yield date-sorted GAME_IDs despite the shuffled input rows.
    date_of = dict(zip(df["GAME_ID"], df["game_date"]))
    seq = [date_of[g] for g in all_ids]
    assert seq == sorted(seq)


# ---------------------------------------------------------------------------
# Adversarial: identical scores, different labels
# ---------------------------------------------------------------------------


@pytest.mark.bench
def test_isotonic_identical_scores_different_labels_is_sensible():
    """10 calibration points at the same score with opposite labels. Must not
    crash and must yield probabilities in [0, 1]; the symmetric 50/50 point
    estimate and a non-negative width are pinned down."""
    # Floor is 10 scores; keep the identical-score mixed-label degeneracy.
    scores = np.array([0.5] * 10)
    labels = np.array([0, 1] * 5)
    p0, p1, opt, width = scores_to_interval(scores, labels, np.array([0.5]))
    for arr in (p0, p1, opt):
        assert np.all(np.isfinite(arr))
        assert np.all((0.0 <= arr) & (arr <= 1.0))
    # Balanced labels at one score: p0 (append 0) and p1 (append 1) stay
    # ordered, optimum is 0.5, width is non-negative.
    assert opt[0] == pytest.approx(0.5)
    assert 0.0 <= p0[0] <= p1[0] <= 1.0
    assert width[0] == pytest.approx(p1[0] - p0[0])
    assert width[0] >= 0.0
