"""Epic 10.4.2 — every planted leak canary must be CAUGHT by its gate.

Each test pairs a canary from ``tests/synth/canaries.py`` with the
production gate that exists to stop that leak class. A gate that lets a
canary through is a failing test. Per the adversarial protocol, each test
also proves (a) the canary really is leaking (it would look *good* on the
metric it attacks) and (b) the gate's rejection reason matches the leak
type — via message checks, directional assertions, and a negative control
showing the same gate accepts honest input.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression

from pipeline.calibration_registry import CalibrationRoleError, CalibrationSliceRegistry
from pipeline.cv import ChronologicalPartitionCV, PastOnlyGroupCV
from pipeline.market_snapshots import (
    attach_tip_utc,
    build_quotes_frame,
    point_clv,
    reject_invalid_quotes,
    select_decision_quotes,
)
from pipeline.promotion_gates import (
    beats_market_and_baseline,
    interval_coverage_credible,
    margin_dispersion_ok,
    policy_retune_allowed,
)
from tests.synth.canaries import (
    FutureOddsQuote,
    PsychicFeature,
    SliceReuseAttack,
    TimeTravelerTracker,
)

SEED = 20261004


def _games_frame(rng: np.random.Generator, n_days: int = 40, games_per_day: int = 2) -> pd.DataFrame:
    dates = pd.date_range("2025-10-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        for g in range(games_per_day):
            rows.append({
                "game_id": f"{d:%Y%m%d}-{g}",
                "game_date": d,
                "future_signal": float(rng.normal(0.0, 1.0)),
            })
    return pd.DataFrame(rows)


@pytest.mark.bench
def test_past_only_group_cv_blocks_time_traveler_tracker():
    """Gate: PastOnlyGroupCV. Canary: TimeTravelerTracker (train on future).

    A tracker updated only on a fold's training rows must never have seen
    the game being predicted, and every fold must satisfy
    max(train_date) < min(val_date).
    """
    rng = np.random.default_rng(SEED)
    df = _games_frame(rng)
    X = df[["future_signal"]].to_numpy()
    groups = df["game_date"].to_numpy()

    gate_folds = list(PastOnlyGroupCV(n_splits=4).split(X, groups=groups))
    assert gate_folds, "gate produced no folds on well-formed input"
    for train_idx, val_idx in gate_folds:
        tracker = TimeTravelerTracker()
        # Update the tracker in chronological order over TRAIN rows only.
        for i in train_idx[np.argsort(groups[train_idx])]:
            tracker.peek_then_update(df["game_id"].iat[i], df["future_signal"].iat[i])
        assert tracker.last_seen_game_id is not None  # tracker really updated
        val_games = {df["game_id"].iat[j] for j in val_idx}
        assert tracker.last_seen_game_id not in val_games
        # Rejection reason == leak type: strictly temporal separation.
        assert groups[train_idx].max() < groups[val_idx].min()

    # Adversarial control: on this same data the deprecated leaky splitter
    # DOES train on future rows — so the invariant above is not vacuous, and
    # a TimeTravelerTracker fed by it would peek at validation-day games.
    future_trained_folds = 0
    for train_idx, val_idx in ChronologicalPartitionCV(n_splits=4).split(X, groups=groups):
        if len(train_idx) == 0 or len(val_idx) == 0:
            continue
        tracker = TimeTravelerTracker()
        for i in train_idx:
            tracker.peek_then_update(df["game_id"].iat[i], df["future_signal"].iat[i])
        if groups[train_idx].max() >= groups[val_idx].min():
            future_trained_folds += 1
    assert future_trained_folds > 0, (
        "control failed: leaky splitter did not train on future rows here, "
        "so this dataset cannot demonstrate the leak the gate blocks"
    )


@pytest.mark.bench
def test_promotion_gates_reject_psychic_feature_run():
    """Gate: promotion_gates sanity checks. Canary: PsychicFeature.

    A model given the psychic column wins every accuracy metric (and would
    be promoted by the accuracy-only gate), but perfect foresight is
    OVER-dispersed and OVER-covered versus any honest forecast — the sanity
    gates must reject it for exactly that reason.
    """
    rng = np.random.default_rng(SEED)
    n = 4000
    market_margin = rng.normal(-3.0, 6.0, n)            # market-implied home margin
    injury_signal = rng.normal(0.0, 3.0, n)             # real pre-game info
    actual_margin = market_margin + injury_signal + rng.normal(0.0, 12.0, n)

    df = pd.DataFrame({"ACTUAL_MARGIN": actual_margin, "MARKET_MARGIN": market_margin})
    df = PsychicFeature().attach(df, outcome_col="ACTUAL_MARGIN")
    # Canary really is perfectly correlated with the future label.
    assert np.array_equal(
        df["psychic_cover"].to_numpy(), (actual_margin > 0).astype(float)
    )

    # Psychic "model": the best predictor trainable from the leaked column
    # (OLS on a binary indicator = conditional means of the future outcome).
    psychic_pred = LinearRegression().fit(
        df[["psychic_cover"]], df["ACTUAL_MARGIN"]
    ).predict(df[["psychic_cover"]])
    # Honest baseline: shrunk market + pre-game signal; no future information.
    honest_pred = 0.75 * (market_margin + injury_signal)

    half_width_80 = 1.2815515655446004 * 12.0  # nominal 80% interval, honest noise scale

    def metrics(pred: np.ndarray) -> dict:
        err = actual_margin - pred
        return {
            "spread_mae": float(np.mean(np.abs(err))),
            "market_spread_mae": float(np.mean(np.abs(actual_margin - market_margin))),
            "margin_dispersion_ratio": float(np.std(pred) / np.std(market_margin)),
            "interval_coverage": float(np.mean(np.abs(err) <= half_width_80)),
            "market_error_corr": float(np.corrcoef(
                pred - market_margin, actual_margin - market_margin
            )[0, 1]),
        }

    psychic = metrics(psychic_pred)
    honest = metrics(honest_pred)

    # (a) The canary really is leaking: it dominates on accuracy and WOULD be
    # promoted by the accuracy-only gate — the sanity gates are the only stop.
    assert psychic["spread_mae"] < honest["spread_mae"]
    assert psychic["spread_mae"] < psychic["market_spread_mae"]
    assert beats_market_and_baseline(psychic, honest)

    # (b) Rejection reason matches the leak type: perfect foresight fails
    # HIGH (predicts at outcome scale / intervals over-cover), not low.
    assert psychic["margin_dispersion_ratio"] > 0.95  # over-dispersed, not collapsed
    assert not margin_dispersion_ok(psychic)
    assert psychic["interval_coverage"] > 0.80 + 0.08  # over-covered, not under
    assert not interval_coverage_credible(psychic)
    assert not policy_retune_allowed(psychic)

    # Adversarial control: the same gates PASS the honest run — rejection is
    # specific to the psychic leak, not gates that always say no.
    assert margin_dispersion_ok(honest)
    assert interval_coverage_credible(honest)
    assert policy_retune_allowed(honest)


@pytest.mark.bench
def test_calibration_slice_registry_raises_on_slice_reuse_attack():
    """Gate: CalibrationSliceRegistry. Canary: SliceReuseAttack (Task 053)."""
    rng = np.random.default_rng(SEED)
    row_ids = tuple(int(x) for x in rng.integers(10_000, 99_999, size=64))
    attack = SliceReuseAttack(row_ids=row_ids)
    assert attack.target_a != attack.target_b  # same rows, DIFFERENT calibrators

    registry = CalibrationSliceRegistry()
    with pytest.raises(CalibrationRoleError, match="identical row slice") as excinfo:
        attack.attempt(registry)

    # Rejection reason must be cross-target slice reuse — the registry also
    # raises for duplicate-target registration, which would prove nothing.
    assert "already has a registered calibration path" not in str(excinfo.value)
    # The raise names BOTH calibrators on the shared slice (reuse-specific).
    assert attack.target_a in str(excinfo.value)
    assert attack.target_b in str(excinfo.value)

    # Adversarial control: a DISJOINT slice for the second target is accepted,
    # so the raise above is specific to row-set reuse.
    control = CalibrationSliceRegistry()
    control.register(attack.target_a, row_ids)
    disjoint_ids = tuple(int(x) for x in rng.integers(200_000, 299_999, size=64))
    control.register(attack.target_b, disjoint_ids)  # must not raise
    assert set(control.registered_targets()) == {attack.target_a, attack.target_b}


@pytest.mark.bench
def test_odds_provenance_rejects_future_odds_quote():
    """Gate: market_snapshots T-60 provenance. Canary: FutureOddsQuote.

    The canary claims a "decision" quote 10 minutes before tip (inside the
    T-60 cutoff) and a "close" quote older than the decision. The quote is
    otherwise valid pregame data, so only the decision-cutoff provenance
    check can stop it — and it must.
    """
    tip = pd.Timestamp("2026-01-15T19:00:00Z")
    canary = FutureOddsQuote.inverted(tip)
    cutoff = 60.0
    # Canary really is leaking: broken chronology, decision inside the cutoff,
    # and the claimed pair would fabricate positive CLV (home side).
    assert canary.is_chronology_broken()
    assert canary.decision_ts > tip - pd.Timedelta(minutes=cutoff)
    assert point_clv(canary.decision_spread, canary.closing_spread, "home") > 0

    def quote_row(point: float, ts: pd.Timestamp) -> dict:
        return {
            "GAME_ID": "g1", "book": "pinnacle", "market": "spread", "side": "home",
            "point": point, "price_american": -110, "price_decimal": 1.909,
            "quote_timestamp": ts, "source_timestamp": ts,
            "ingestion_timestamp": ts + pd.Timedelta(minutes=1),
            "in_play": False,
        }

    legit_ts = tip - pd.Timedelta(minutes=90)
    rows = [
        quote_row(canary.decision_spread, canary.decision_ts),  # canary "decision" @ T-10
        quote_row(canary.closing_spread, canary.close_ts),      # canary "close" @ T-60
        quote_row(-3.0, legit_ts),                              # legit control @ T-90
    ]
    quotes = reject_invalid_quotes(attach_tip_utc(build_quotes_frame(rows), {"g1": tip}))
    decided = select_decision_quotes(quotes, cutoff_minutes=cutoff)

    # Rejection reason is the T-60 cutoff specifically: the canary's claimed
    # decision quote is valid pregame data (it survives reject_invalid_quotes)
    # yet is never ELIGIBLE for the decision slot.
    assert canary.decision_ts in set(quotes["quote_timestamp"])  # not dropped as invalid
    eligible = quotes[quotes["minutes_before_tip"] >= cutoff]
    assert canary.decision_ts not in set(eligible["quote_timestamp"])
    claimed = decided.loc[decided["point"] == canary.decision_spread].iloc[0]
    assert bool(claimed["decision_missing"])

    # The canary's "close" is what an honest pipeline actually uses as the
    # T-60 decision — the claimed chronology cannot survive provenance.
    close_group = decided.loc[decided["point"] == canary.closing_spread].iloc[0]
    assert not bool(close_group["decision_missing"])
    assert close_group["decision_quote_timestamp"] == canary.close_ts

    # Adversarial control: the legitimate T-90 quote IS selected, so the
    # refusal above is cutoff-specific, not a broken frame or pipeline.
    control = decided.loc[decided["point"] == -3.0].iloc[0]
    assert not bool(control["decision_missing"])
    assert control["decision_quote_timestamp"] == legit_ts

    # Invariant: no selected decision quote is ever younger than the cutoff.
    picked = decided.loc[~decided["decision_missing"]]
    assert (picked["decision_minutes_before_tip"] >= cutoff).all()
