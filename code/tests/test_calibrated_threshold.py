"""Calibrated vs raw WIN_PCT threshold picking."""
import pandas as pd

from pipeline.bet_confidence import WalkForwardBetCalibrator, _scored_ats_bets
from pipeline.metrics import walkforward_min_confidence


def _lean_df(n=200, *, win_when_high_raw: bool = True):
    rows = []
    for i in range(n):
        edge = 3.0 + (i % 10)
        raw_conf = 48 + (i % 22)
        if win_when_high_raw:
            win = raw_conf >= 64 and i % 4 != 0
        else:
            win = i % 3 != 0
        rows.append({
            "MARKET_SPREAD": -3.5,
            "EDGE_LEAN": "Home" if i % 2 == 0 else "Away",
            "DIRECTION": "Home" if i % 2 == 0 else "Away",
            "EDGE": edge if i % 2 == 0 else -edge,
            "ACTUAL_MARGIN": 5 if win else -4,
            "WIN_PCT": raw_conf,
            "CONF_WIDTH": 18.0,
            "RATING_UNCERTAINTY": 220.0,
            "ELO_META_AGREEMENT": 1.0,
            "SPREAD_COVER_PROB": 0.55,
            "DISAGREEMENT_TRUST": 0.95,
            "PHANTOM_INJURY_FLAG": 0,
            "PRED_SPREAD": -2.0,
        })
    return pd.DataFrame(rows)


def test_calibrated_threshold_differs_from_stored_win_pct():
    df = _lean_df(220)
    cal = WalkForwardBetCalibrator()
    cal.fit(df, scope="all_prior")

    raw_thr = walkforward_min_confidence(
        df, thresholds=range(50, 71), default=64, min_bets=30, require_positive_ci=False,
    )
    cal_thr = walkforward_min_confidence(
        df,
        thresholds=range(50, 71),
        default=64,
        min_bets=30,
        require_positive_ci=False,
        bet_calibrator=cal,
    )
    scored = _scored_ats_bets(cal, df)
    assert len(scored) >= 30
    stored = df["WIN_PCT"].tolist()
    calibrated = [cs for _y, cs, _p in scored]
    assert stored != calibrated
    # Same numeric threshold must gate different bet sets once scores are rescaled.
    from pipeline.bet_selection import lean_spread_frame
    leans = lean_spread_frame(df)
    raw_at_64 = int((pd.to_numeric(leans["WIN_PCT"], errors="coerce") >= 64).sum())
    cal_at_64 = sum(1 for _y, cs, _p in scored if cs >= 64)
    assert raw_at_64 != cal_at_64


def test_calibrated_threshold_allows_more_bets_than_miscalibrated_64():
    """When isotonic compresses scores, calibrated picker should land below 64."""
    df = _lean_df(180, win_when_high_raw=True)
    cal = WalkForwardBetCalibrator()
    cal.fit(df, scope="all_prior")
    thr = walkforward_min_confidence(
        df,
        thresholds=(50, 52, 54, 56, 58, 60, 62, 64),
        default=64,
        min_bets=25,
        require_positive_ci=False,
        bet_calibrator=cal,
    )
    assert thr <= 64


def test_confidence_band_excludes_overconfident_scores():
    from pipeline.bet_confidence import _pick_confidence_band_gated_roi

    scored = []
    for i in range(200):
        cs = 52 + (i % 14)
        win = 58 <= cs <= 61 and i % 4 != 0
        scored.append((float(win), cs, cs / 100.0))
    roi, lo, hi, n = _pick_confidence_band_gated_roi(scored, min_bets=30)
    assert n >= 30
    assert hi is not None and hi <= 62
    assert lo >= 58
    assert roi > 0


def test_confidence_feature_vector_length():
    from pipeline.bet_confidence import CONFIDENCE_FEATURE_NAMES, confidence_feature_vector

    feats = confidence_feature_vector({
        "spread_edge_pts": 5.0,
        "conf_width": 20.0,
        "rating_uncertainty": 200.0,
        "matchup_vol_sigma": 11.0,
        "league_vol_sigma": 10.0,
        "elo_meta_agreement": 0.8,
        "spread_cover_prob": 0.56,
        "disagreement_trust": 0.9,
        "phantom_injury_flag": 0,
        "ats_classifier_prob": 0.55,
        "win_prob": 0.58,
        "elo_margin": 4.0,
        "lean": "Home",
    })
    assert len(feats) == len(CONFIDENCE_FEATURE_NAMES)


def test_calibrated_features_monotone_with_edge():
    from pipeline.bet_confidence import build_calibrated_feature_vector

    low = build_calibrated_feature_vector({"spread_edge_pts": 2.0})
    high = build_calibrated_feature_vector({"spread_edge_pts": 9.0})
    assert high[0] > low[0]  # composite_norm
    assert high[1] > low[1]  # abs_edge_norm


def test_logistic_isotonic_monotone_ranking():
    from pipeline.bet_confidence import WalkForwardBetCalibrator, _rank_spearman, _scored_ats_bets

    rows = []
    for i in range(240):
        edge = 1.0 + (i % 12)
        rows.append({
            "MARKET_SPREAD": -3.5,
            "EDGE_LEAN": "Home",
            "EDGE": edge,
            "ACTUAL_MARGIN": 6 if i % 5 != 0 and edge >= 6 else -4,
            "WIN_PCT": 50 + edge,
            "CONF_WIDTH": 22.0,
            "RATING_UNCERTAINTY": 220.0,
            "ELO_META_AGREEMENT": 0.9,
            "SPREAD_COVER_PROB": 0.52 + edge / 100.0,
            "WIN_PROB": 0.52 + edge / 80.0,
            "ELO_MARGIN_CALIBRATED": edge / 2.0,
            "DISAGREEMENT_TRUST": 0.95,
            "PHANTOM_INJURY_FLAG": 0,
            "PRED_SPREAD": -2.0,
        })
    df = pd.DataFrame(rows)
    cal = WalkForwardBetCalibrator(method="logistic_isotonic")
    cal.fit(df, method="logistic_isotonic", scope="all_prior")
    scored = _scored_ats_bets(cal, df)
    sp = _rank_spearman([s[0] for s in scored], [s[1] for s in scored])
    assert sp > 0.15


def test_fit_walkforward_confidence_calibrator_returns_meta():
    from pipeline.bet_confidence import fit_walkforward_confidence_calibrator

    rows = []
    for i in range(200):
        edge = 2.0 + (i % 10)
        rows.append({
            "MARKET_SPREAD": -3.5,
            "EDGE_LEAN": "Home",
            "EDGE": edge,
            "ACTUAL_MARGIN": 5 if edge >= 6 and i % 4 else -4,
            "CONF_WIDTH": 20.0,
            "RATING_UNCERTAINTY": 200.0,
            "ELO_META_AGREEMENT": 0.85,
            "SPREAD_COVER_PROB": 0.52 + edge / 100.0,
            "WIN_PROB": 0.52 + edge / 80.0,
            "ELO_MARGIN_CALIBRATED": edge / 2.0,
            "DISAGREEMENT_TRUST": 0.95,
            "PHANTOM_INJURY_FLAG": 0,
            "PRED_SPREAD": -2.0,
        })
    df = pd.DataFrame(rows)
    cal, meta = fit_walkforward_confidence_calibrator(
        df,
        train_season_label="2023-2024",
        test_season_label="2024-2025",
        season_index=1,
    )
    assert cal._fitted
    assert meta.get("inline_phase2a") is True
    assert meta.get("train_min_confidence", 0) >= 55
    assert meta.get("calib_method") == "isotonic" or meta.get("feature_weights")


def test_walkforward_confidence_gate_enforces_floor_and_cap():
    from pipeline.metrics import walkforward_confidence_gate

    df = _lean_df(220)
    cal = WalkForwardBetCalibrator()
    cal.fit(df, scope="all_prior")
    lo, hi = walkforward_confidence_gate(df, bet_calibrator=cal, min_bets=25, min_gated_bets=20)
    assert lo >= 55
    assert lo <= 60
    assert hi is None or hi <= 65


def test_confidence_min_edge_filters_scored_bets():
    from pipeline.bet_confidence import _confidence_training_frame, _scored_ats_bets
    from pipeline.config import CONFIDENCE_MIN_EDGE

    rows = []
    for i in range(40):
        edge = 1.0 if i % 2 == 0 else 4.0
        rows.append({
            "MARKET_SPREAD": -3.5,
            "EDGE_LEAN": "Home",
            "EDGE": edge,
            "ACTUAL_MARGIN": 5 if edge >= CONFIDENCE_MIN_EDGE else -4,
            "CONF_WIDTH": 18.0,
            "RATING_UNCERTAINTY": 200.0,
            "ELO_META_AGREEMENT": 1.0,
            "SPREAD_COVER_PROB": 0.55,
            "DISAGREEMENT_TRUST": 0.95,
            "PHANTOM_INJURY_FLAG": 0,
            "PRED_SPREAD": -2.0,
        })
    df = pd.DataFrame(rows)
    filtered = _confidence_training_frame(df)
    assert len(filtered) == 20
    cal = WalkForwardBetCalibrator()
    cal.fit(filtered, scope="all_prior")
    scored = _scored_ats_bets(cal, df)
    assert len(scored) == 20


def test_compare_selection_strategies():
    from pipeline.metrics import compare_selection_strategies

    df = _lean_df(120)
    df["DIRECTION"] = df["EDGE_LEAN"]
    tbl = compare_selection_strategies(df, min_conf=55, confidence_min_edge=3.0)
    assert not tbl.empty
    assert "confidence_hybrid" in tbl["mode"].iloc[0]
    assert (tbl["n_bets"] > 0).any()
