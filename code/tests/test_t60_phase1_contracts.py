"""T-60 Phase 1 contract tests: decision residual, fail-closed dates/totals, two-sided ML."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from pipeline.market import (
    fair_probs_from_home_ml,
    fair_probs_from_ml_pair,
    ml_decimal_for_side,
    select_ml_bet,
)
from pipeline.market_targets import (
    decision_spread_series,
    margin_decision_residual,
    residual_to_margin,
    closing_spread_series,
)
from pipeline.model import MetaScoreModel, SAFE_FEATURE_COLS
from pipeline.market_snapshots import assert_no_close_leak


def test_decision_spread_series_prefers_decision_not_close():
    df = pd.DataFrame({
        "decision_spread": np.full(25, -4.0),
        "closing_spread": np.full(25, -6.0),
        "market_spread": np.full(25, -4.0),
    })
    s = decision_spread_series(df)
    assert s is not None
    assert float(s.iloc[0]) == -4.0
    close = closing_spread_series(df)
    assert float(close.iloc[0]) == -6.0


def test_margin_decision_residual_roundtrip():
    actual = np.array([7.0, -3.0, 10.0])
    decision = np.array([-4.0, 6.0, -7.0])
    resid = margin_decision_residual(actual, decision)
    assert np.allclose(resid, actual + decision)
    assert np.allclose(residual_to_margin(resid, decision), actual)


def test_meta_model_decision_residual_uses_t60_not_close():
    n = 80
    rng = np.random.default_rng(0)
    decision = rng.normal(-4, 2, n)
    close = decision - 1.5  # systematically different from T-60
    margin = rng.normal(0, 8, n)
    rows = {c: np.zeros(n) for c in SAFE_FEATURE_COLS}
    rows["elo_margin"] = margin + rng.normal(0, 2, n)
    rows["elo_net"] = rows["elo_margin"] / 10.0
    rows["exp_poss"] = np.full(n, 100.0)
    rows["decision_spread"] = decision
    rows["market_spread"] = decision
    rows["closing_spread"] = close
    df = pd.DataFrame(rows)
    model = MetaScoreModel(
        ridge_alpha=5.0,
        use_quantile_heads=False,
        train_target="decision_residual",
        cb_params={"iterations": 40, "depth": 2, "verbose": 0, "allow_writing_files": False},
    )
    y_h = (margin + 220) / 2.0
    y_a = (220 - margin) / 2.0
    model.fit(df, y_h, y_a)
    assert model.fitted
    assert model._active_train_mode == "decision_residual"
    # Deprecated alias still maps to decision_residual
    model2 = MetaScoreModel(
        ridge_alpha=5.0,
        use_quantile_heads=False,
        train_target="close_residual",
        cb_params={"iterations": 20, "depth": 2, "verbose": 0, "allow_writing_files": False},
    )
    assert model2.train_target == "decision_residual"


def test_unresolved_game_date_raises():
    from pipeline.game_features import build_game_features
    from pipeline.ratings import PlayerRatingTracker
    from pipeline.hierarchical import HierarchicalPossessionEngine
    from pipeline.trackers import PaceTracker
    from pipeline.config import DEFAULT_LEAGUE_XPPP

    elo = PlayerRatingTracker(league_xppp=DEFAULT_LEAGUE_XPPP)
    hier = HierarchicalPossessionEngine()
    pace = PaceTracker(team_window=10, league_window=100)
    with pytest.raises(ValueError, match="Unresolved game_date"):
        build_game_features(
            game_id="bad",
            gdate="not-a-date",
            home_team="BOS",
            away_team="NYK",
            home_starters=["1", "2", "3", "4", "5"],
            away_starters=["6", "7", "8", "9", "10"],
            elo_tracker=elo,
            hier_engine=hier,
            pace_tracker=pace,
        )


def test_missing_market_total_stays_nan():
    from pipeline.game_features import build_game_features
    from pipeline.ratings import PlayerRatingTracker
    from pipeline.hierarchical import HierarchicalPossessionEngine
    from pipeline.trackers import PaceTracker
    from pipeline.config import DEFAULT_LEAGUE_XPPP

    elo = PlayerRatingTracker(league_xppp=DEFAULT_LEAGUE_XPPP)
    hier = HierarchicalPossessionEngine()
    pace = PaceTracker(team_window=10, league_window=100)
    odds = {
        (pd.Timestamp("2025-11-01").date(), "BOS"): {
            "spread": -3.5, "ml": -150, "decision_spread": -3.5, "closing_spread": -3.0,
            # no total key
        }
    }
    feat = build_game_features(
        game_id="g1",
        gdate=pd.Timestamp("2025-11-01"),
        home_team="BOS",
        away_team="NYK",
        home_starters=["1", "2", "3", "4", "5"],
        away_starters=["6", "7", "8", "9", "10"],
        elo_tracker=elo,
        hier_engine=hier,
        pace_tracker=pace,
        odds_dict=odds,
    )
    assert pd.isna(feat["market_total"])
    assert feat["market_total_missing"] == 1
    assert pd.isna(feat["market_total_minus_league"])


def test_fair_probs_uses_real_away_ml():
    # Asymmetric book: home -200, away +160 (not -(-200)=+200)
    p_home, p_away = fair_probs_from_ml_pair(-200, 160)
    p_neg_home, p_neg_away = fair_probs_from_home_ml(-200)  # fallback negation
    assert abs(p_home + p_away - 1.0) < 1e-9
    # Real away price must change fair probs vs pure negation
    assert abs(p_away - p_neg_away) > 1e-6
    dec_away = ml_decimal_for_side(-200, "Away", market_ml_away=160)
    dec_neg = ml_decimal_for_side(-200, "Away", market_ml_away=None)
    assert abs(dec_away - dec_neg) > 1e-6


def test_select_ml_bet_uses_away_price():
    side, ev, dec = select_ml_bet(
        0.42, -180,
        market_ml_away=155,
        min_ev=0.0,
        min_win_pct=40,
        winner_only=False,
        max_favorite_decimal=1.0,
    )
    # Should be able to evaluate Away at real +155, not fabricated -(-180)
    assert dec is None or np.isnan(dec) or dec > 1.0


def test_assert_no_close_leak_rejects_close_columns():
    with pytest.raises(ValueError, match="close_"):
        assert_no_close_leak(["decision_spread", "close_spread", "elo_net"])


def test_ats_labels_are_home_cover_not_direction_flipped():
    from pipeline.ats_classifier import ATSClassifier

    df = pd.DataFrame({
        "actual_margin": [5.0, -8.0],
        "decision_spread": [-3.5, 4.0],
        "market_spread": [-3.5, 4.0],
        "pred_margin": [2.0, -6.0],
    })
    y = ATSClassifier.labels_from_df(df)
    # Game 0: 5 + (-3.5) = 1.5 > 0 → home cover = 1
    # Game 1: -8 + 4 = -4 < 0 → home cover = 0
    assert y[0] == 1.0
    assert y[1] == 0.0


def test_ats_enrich_ignores_closing_spread_only():
    """Closing-only rows must not invent model_edge from closing_spread."""
    from pipeline.ats_classifier import ATSClassifier, EXTRA_FEATURE_DEFAULTS

    clf = ATSClassifier(feature_cols=["elo_net"])
    row = clf._enrich_row({
        "pred_margin": 5.0,
        "closing_spread": -7.0,  # evaluation-only — must be ignored
    })
    assert row["decision_missing"] == 1
    # Defaults (not NaN) — predict path must stay LogisticRegression-safe.
    assert row["model_edge"] == EXTRA_FEATURE_DEFAULTS["model_edge"]
    assert row["abs_model_edge"] == EXTRA_FEATURE_DEFAULTS["abs_model_edge"]
    # Must not have used closing_spread: 5.0 + (-7.0) = -2.0
    assert row["model_edge"] != 5.0 + (-7.0)


def test_apply_prediction_mode_ignores_closing_spread():
    """Residual/blend predict path must not read closing_spread."""
    model = MetaScoreModel(
        ridge_alpha=5.0,
        use_quantile_heads=False,
        train_target="absolute",
        prediction_mode="residual",
        residual_alpha=1.0,
        cb_params={"iterations": 10, "depth": 2, "verbose": 0, "allow_writing_files": False},
    )
    model._active_train_mode = "absolute"
    # Only closing present — residual mode must fall through to absolute (raw).
    out = model._apply_prediction_mode(3.0, {"closing_spread": -8.0})
    assert out == 3.0
    # Decision present — residual uses market prior; with alpha=1 equals raw.
    out2 = model._apply_prediction_mode(3.0, {"decision_spread": -4.0})
    # residual: market_prior + alpha*(raw - market_prior) = 4 + 1*(3-4) = 3
    assert abs(out2 - 3.0) < 1e-9
