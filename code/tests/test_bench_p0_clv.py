"""Epic 10.6.2 / P0.11 — CLV is dead in the latest full run: provenance or logic?

GitHub: P0.11 · Registry IDs: ``close_line_conflation``, ``quote_tip_proxy``,
``roi_without_odds_provenance``

P0.11: CLV is dead in the latest full run — ``n_actionable: 0``,
``n_finite_clv: 0``, ``mean_clv: NaN`` across 5,247 backtest games
(``output/20260914_224422_dash_standard`` review.json). This test isolates
whether dead CLV is *data-provenance* (decision≠close quote pairs never
materialize — the suspected ``quote_tip_proxy`` / ``close_line_conflation``
path) or *pipeline logic* (the CLV formula cannot produce finite values).

Verdict encoded below: given genuine decision≠close home-referenced spreads,
every layer of the CLV path — ``market_snapshots.point_clv``,
``bet_grading.bet_side_point_clv``, ``metrics.compute_clv``,
``metrics.add_clv_columns``, and the actual ``n_finite_clv`` metric inside
``metrics.compute_metrics`` — produces finite, correctly-signed values. The
dead run is therefore a **provenance** failure, and these tests pin the
fail-closed behaviors (NaN, never a silent 0) that keep such failures visible.

Adversarial notes (counterexamples attempted while writing this bench):

1. Sign-flipped spread convention (away-referenced lines fed as "home")
   passes any magnitude-only assertion with the wrong sign → pinned CLV sign
   against ``grade_spread_bet`` outcomes on margins strictly between the two
   lines (``test_p0_11_clv_sign_matches_beat_the_close_grading``).
2. Home/away perspective swap survives home-only assertions → exact-negation
   checks in both move directions plus a seeded sweep.
3. ``assert clv != 0.0`` is vacuous for NaN (NaN != everything) → provenance
   cases assert ``pd.isna`` / ``not np.isfinite`` and a frame-level
   ``(CLV == 0.0).sum() == 0`` guard against fillna(0) regressions.
4. decision == close returns an exact 0.0 at the raw formula layer — a
   *legitimate* no-move zero that would pass a naive "never 0" invariant.
   The real invariant is scoped: ``add_clv_columns`` must treat identical
   decision/close as single-snapshot conflation → NaN, while the formula
   layer keeps true zero representable. Both sides pinned.
5. ``compute_clv`` legacy side inference tie-breaks edge == 0 to "Home" — a
   bet that would never be placed still gets a signed CLV. Pinned as
   documented legacy behavior; the bench relies on the explicit-side path.
6. Infinite inputs yield non-finite CLV: the formula layer is NaN-safe but
   not inf-safe, so CLV finiteness is a data-provenance responsibility.
7. Feeding the raw lowercase factory schema into ``add_clv_columns`` maps no
   closing column → CLV fails closed to NaN rather than inventing a value.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.bet_grading import bet_side_point_clv, grade_spread_bet
from pipeline.market_snapshots import point_clv
from pipeline.metrics import add_clv_columns, compute_clv, compute_metrics
from tests.synth.factories import make_odds

SEED = 20260919


def _canonical_bet_row(odds: dict, direction: str, **extra) -> dict:
    """Map the lowercase factory odds schema onto the canonical backtest-export
    schema consumed by ``metrics.add_clv_columns`` (DECISION_SPREAD / CLOSING_SPREAD)."""
    row = {
        "MARKET_SPREAD": odds["market_spread"],
        "DECISION_SPREAD": odds["decision_spread"],
        "CLOSING_SPREAD": odds["closing_spread"],
        "DIRECTION": direction,
    }
    row.update(extra)
    return row


@pytest.mark.bench
def test_p0_11_point_clv_finite_when_line_moves_toward_home():
    """Acceptance: decision -3.5 → close -5.0 gives finite home CLV == +1.5
    at every layer of the CLV path (formula, bet-grading, scalar metrics)."""
    odds = make_odds(decision_spread=-3.5, closing_spread=-5.0)
    dec, clo = odds["decision_spread"], odds["closing_spread"]
    expected = dec - clo
    assert expected == pytest.approx(1.5, rel=0, abs=1e-12)

    for got in (
        point_clv(dec, clo, "home"),
        bet_side_point_clv(dec, clo, "Home"),
        # Explicit side: model_spread is unused by compute_clv (NaN proves it).
        compute_clv(np.nan, dec, clo, side="Home"),
    ):
        assert np.isfinite(got), f"CLV must be finite when decision != close, got {got}"
        assert got == pytest.approx(expected, rel=0, abs=1e-12)


@pytest.mark.bench
def test_p0_11_point_clv_finite_when_line_moves_toward_away():
    """Move in the other direction: decision -5.0 → close -3.5 gives home CLV
    == -1.5, and the away side is the exact negation (+1.5)."""
    odds = make_odds(decision_spread=-5.0, closing_spread=-3.5)
    dec, clo = odds["decision_spread"], odds["closing_spread"]

    home = point_clv(dec, clo, "home")
    away = point_clv(dec, clo, "away")
    assert np.isfinite(home) and np.isfinite(away)
    assert home == pytest.approx(-1.5, rel=0, abs=1e-12)
    assert away == pytest.approx(1.5, rel=0, abs=1e-12)
    assert away == pytest.approx(-home, rel=0, abs=1e-12)


@pytest.mark.bench
def test_p0_11_frame_path_and_n_finite_clv_metric_are_alive():
    """The P0.11 question at frame level: ``add_clv_columns`` and the real
    ``n_finite_clv`` metric inside ``compute_metrics`` must report finite CLV
    when genuine decision≠close pairs exist. If this is green while the full
    run shows ``n_finite_clv: 0``, the run's dead CLV is data provenance."""
    df = pd.DataFrame([
        _canonical_bet_row(make_odds(decision_spread=-3.5, closing_spread=-5.0), "Home",
                           PRED_SPREAD=-1.0, ACTUAL_MARGIN=-4.0),
        _canonical_bet_row(make_odds(decision_spread=-5.0, closing_spread=-3.5), "Away",
                           PRED_SPREAD=-7.0, ACTUAL_MARGIN=-2.0),
        _canonical_bet_row(make_odds(decision_spread=-3.5, closing_spread=-5.0), "Pass",
                           PRED_SPREAD=-3.5, ACTUAL_MARGIN=1.0),
    ])
    df["simulated_season_window"] = "2025-2026"

    out = add_clv_columns(df)
    assert out.loc[0, "CLV"] == pytest.approx(1.5, rel=0, abs=1e-12)   # home: dec - clo
    assert out.loc[1, "CLV"] == pytest.approx(1.5, rel=0, abs=1e-12)   # away: clo - dec
    assert pd.isna(out.loc[2, "CLV"])                                   # Pass → absent
    assert out["POINT_CLV"].equals(out["CLV"])
    assert np.isfinite(out["CLV"].dropna().mean())

    metrics = compute_metrics(out)
    all_row = metrics[metrics["season"] == "ALL"].iloc[0]
    assert int(all_row["n_finite_clv"]) == 2


@pytest.mark.bench
def test_p0_11_missing_closing_snapshot_is_nan_never_zero():
    """NaN provenance: a missing closing snapshot must propagate as NaN at
    every layer — never silently 0 (a fake zero would masquerade as a real
    'line did not move' observation and corrupt mean_clv)."""
    odds = make_odds(decision_spread=-3.5, closing_spread=np.nan)
    assert pd.isna(odds["closing_spread"])

    clv = point_clv(odds["decision_spread"], odds["closing_spread"], "home")
    assert pd.isna(clv)
    assert not np.isfinite(clv)
    assert clv != 0.0

    # Missing decision snapshot is equally absent.
    assert pd.isna(point_clv(np.nan, -5.0, "home"))
    assert pd.isna(bet_side_point_clv(-3.5, np.nan, "Away"))
    assert pd.isna(compute_clv(-6.0, -3.5, np.nan, side="Home"))

    # Frame layer: NaN, and no zero-valued row materializes.
    df = pd.DataFrame([
        _canonical_bet_row(odds, "Home", PRED_SPREAD=-1.0, ACTUAL_MARGIN=0.0),
    ])
    out = add_clv_columns(df)
    assert out["CLV"].isna().all()
    assert int((out["CLV"] == 0.0).sum()) == 0


@pytest.mark.bench
def test_p0_11_single_snapshot_conflation_is_nan_at_frame_layer():
    """The exact P0.11 mechanism: a single-snapshot feed (``make_odds`` default
    sets closing == decision, mirroring ``close_line_conflation``) yields a
    true 0.0 at the raw formula layer — a legitimate no-move zero — but the
    frame layer must treat identical decision/close as conflation → NaN, so a
    conflated feed can never inflate ``n_finite_clv`` with silent zeros."""
    odds = make_odds(spread=-3.5)  # decision_spread == closing_spread == -3.5
    assert odds["decision_spread"] == odds["closing_spread"]

    # Formula layer: true zero is representable (line genuinely did not move).
    assert point_clv(odds["decision_spread"], odds["closing_spread"], "home") == 0.0

    # Frame/provenance layer: conflation is NaN, never a silent 0.
    df = pd.DataFrame([
        _canonical_bet_row(odds, "Home", PRED_SPREAD=-1.0, ACTUAL_MARGIN=0.0),
        _canonical_bet_row(odds, "Away", PRED_SPREAD=-5.0, ACTUAL_MARGIN=0.0),
    ])
    out = add_clv_columns(df)
    assert out["CLV"].isna().all()
    assert int((out["CLV"] == 0.0).sum()) == 0


@pytest.mark.bench
def test_p0_11_clv_sign_matches_beat_the_close_grading():
    """Semantic pin against sign-flipped spread conventions: when the line
    moves toward home (decision -3.5 → close -5.0), a home bettor's decision
    number strictly beats the close — verified with ``grade_spread_bet`` on
    margins between the two lines — so home CLV must be positive. A
    convention flip (away-referenced lines) breaks this agreement."""
    odds = make_odds(decision_spread=-3.5, closing_spread=-5.0)
    for margin in (4.0, 4.5):  # home wins by 4–4.5: covers decision, fails close
        assert grade_spread_bet(margin, odds["decision_spread"], "Home") == "win"
        assert grade_spread_bet(margin, odds["closing_spread"], "Home") == "loss"
    assert point_clv(odds["decision_spread"], odds["closing_spread"], "home") > 0.0

    # Reverse move, away bettor: decision -5.0 → close -3.5.
    odds_rev = make_odds(decision_spread=-5.0, closing_spread=-3.5)
    for margin in (4.0, 4.5):  # away covers the decision number, not the close
        assert grade_spread_bet(margin, odds_rev["decision_spread"], "Away") == "win"
        assert grade_spread_bet(margin, odds_rev["closing_spread"], "Away") == "loss"
    assert point_clv(odds_rev["decision_spread"], odds_rev["closing_spread"], "away") > 0.0


@pytest.mark.bench
def test_p0_11_home_away_antisymmetry_seeded_sweep():
    """Seeded sweep: home CLV == decision - close exactly, away == its exact
    negation, finite for every finite distinct pair (deterministic rng)."""
    rng = np.random.default_rng(SEED)
    decision = rng.uniform(-15.0, 15.0, size=64)
    close = rng.uniform(-15.0, 15.0, size=64)
    for dec, clo in zip(decision, close):
        home = point_clv(dec, clo, "home")
        away = point_clv(dec, clo, "away")
        assert np.isfinite(home) and np.isfinite(away)
        assert home == pytest.approx(float(dec) - float(clo), rel=1e-12, abs=1e-12)
        assert away == pytest.approx(-(float(dec) - float(clo)), rel=1e-12, abs=1e-12)


@pytest.mark.bench
def test_p0_11_unknown_side_raises_no_silent_default():
    """An unrecognized side must raise — a silent default side would flip CLV
    signs for half of all bets without failing any NaN check."""
    with pytest.raises(ValueError):
        point_clv(-3.5, -5.0, "sideways")
    with pytest.raises(ValueError):
        bet_side_point_clv(-3.5, -5.0, "")


@pytest.mark.bench
def test_p0_11_legacy_compute_clv_zero_edge_tiebreak_documented():
    """Adversarial pin: legacy ``compute_clv`` side inference uses
    ``sign(model + bet)`` with ``edge >= 0 → Home``, so an exact zero-edge
    'bet' (which bet selection would never place) still gets Home-side CLV.
    Documents the tie-break so it cannot change silently; the bench's
    acceptance path always passes an explicit side."""
    # edge == 0 exactly → inferred Home → dec - clo == +1.5.
    got = compute_clv(3.5, -3.5, -5.0)
    assert got == pytest.approx(1.5, rel=0, abs=1e-12)
    # Negative edge → inferred Away → clo - dec == -1.5.
    got_away = compute_clv(1.0, -3.5, -5.0)
    assert got_away == pytest.approx(-1.5, rel=0, abs=1e-12)
    # No side and no model → absent, never guessed.
    assert pd.isna(compute_clv(np.nan, -3.5, -5.0))


@pytest.mark.bench
def test_p0_11_infinite_inputs_yield_non_finite_clv():
    """Boundary documentation: the formula layer guards NaN but not ±inf, so
    'point_clv is finite' holds only for finite market lines — finiteness of
    the input is a data-provenance responsibility, not a formula guarantee."""
    got = point_clv(np.inf, -5.0, "home")
    assert not np.isfinite(got)
    assert pd.isna(point_clv(np.inf, np.nan, "home"))  # NaN guard still wins


@pytest.mark.bench
def test_p0_11_raw_factory_schema_fails_closed_nan():
    """Provenance trap: feeding the raw lowercase factory/odds-loader schema
    straight into ``add_clv_columns`` maps no closing column (only uppercase
    ``CLOSING_SPREAD`` is read). The path must fail closed to NaN rather than
    compute against a wrong or invented column."""
    odds = make_odds(decision_spread=-3.5, closing_spread=-5.0)
    df = pd.DataFrame([{**odds, "DIRECTION": "Home"}])
    out = add_clv_columns(df)
    assert out["CLV"].isna().all()
    assert int((out["CLV"] == 0.0).sum()) == 0
