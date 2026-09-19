"""Unit tests for ATSClassifier labels and predict."""
import numpy as np
import pandas as pd

from pipeline.ats_classifier import ATSClassifier


def _mini_df():
    return pd.DataFrame({
        "actual_margin": [5, -3, 10, -8],
        "actual_home": [110, 100, 115, 95],
        "actual_away": [105, 103, 105, 103],
        "closing_spread": [-3.5, 2.0, -5.0, 4.0],
        "market_spread": [-3.5, 2.0, -5.0, 4.0],
        "pred_margin": [2.0, -1.0, 8.0, -6.0],
        "h_elo_off": [1500, 1500, 1500, 1500],
        "a_elo_off": [1500, 1500, 1500, 1500],
        "h_rating_uncertainty": [300, 300, 300, 300],
        "a_rating_uncertainty": [300, 300, 300, 300],
        "elo_margin": [1, -1, 3, -2],
        "exp_poss": [100, 100, 100, 100],
    })


def test_labels_from_df():
    y = ATSClassifier.labels_from_df(_mini_df())
    assert len(y) == 4
    assert np.isfinite(y).all()


def test_fit_predict_cover_prob():
    df = _mini_df()
    clf = ATSClassifier(feature_cols=["h_elo_off", "a_elo_off", "elo_margin"])
    calib = df.iloc[:2].copy()
    train = df.iloc[2:].copy()
    clf.fit(train, calib_df=calib)
    if clf.fitted:
        p = clf.predict_cover_prob(
            {"h_elo_off": 1500, "a_elo_off": 1500, "elo_margin": 2, "pred_margin": 3},
            "Home",
            -3.5,
        )
        assert 0.01 <= p <= 0.99


def test_predict_cover_prob_nan_safe():
    """NaN base features / missing line must not reach LogisticRegression."""
    df = pd.concat([_mini_df()] * 20, ignore_index=True)
    # Enough rows + class balance for fit() to succeed (needs >= 50).
    df.loc[::2, "actual_margin"] = 8
    df.loc[1::2, "actual_margin"] = -8
    clf = ATSClassifier(feature_cols=["h_elo_off", "a_elo_off", "elo_margin"])
    clf.fit(df.iloc[:60], calib_df=df.iloc[60:80])
    assert clf.fitted

    cases = [
        # (a) NaN base feature
        (
            {"h_elo_off": 1500, "a_elo_off": 1500, "elo_margin": np.nan, "pred_margin": 3},
            -3.5,
        ),
        # (b) no spread keys + NaN market_spread arg
        (
            {"h_elo_off": 1500, "a_elo_off": 1500, "elo_margin": 2, "pred_margin": 3},
            np.nan,
        ),
        # (c) NaN uncertainty fields
        (
            {
                "h_elo_off": 1500,
                "a_elo_off": 1500,
                "elo_margin": 2,
                "pred_margin": 3,
                "h_rating_uncertainty": np.nan,
                "a_rating_uncertainty": np.nan,
            },
            -3.5,
        ),
    ]
    for feat, spread in cases:
        p = clf.predict_cover_prob(feat, "Home", spread)
        assert np.isfinite(p)
        assert 0.01 <= p <= 0.99
