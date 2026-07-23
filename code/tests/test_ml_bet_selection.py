"""ML bet selection: predicted winner + EV gate."""
from pipeline.market import ml_is_coin_flip, ml_predicted_winner_side, select_ml_bet


def test_predicted_winner_side():
    assert ml_predicted_winner_side(0.55) == "Home"
    assert ml_predicted_winner_side(0.45) == "Away"
    assert ml_predicted_winner_side(0.5) == "Pass"


def test_coin_flip_detection():
    assert ml_is_coin_flip(0.52, band=0.03)
    assert ml_is_coin_flip(0.48, band=0.03)
    assert not ml_is_coin_flip(0.55, band=0.03)
    assert not ml_is_coin_flip(0.58, band=0.05)


def test_select_ml_bet_winner_only_bets_home_favorite_with_ev():
    # Home -150 (dec 1.67); model edge must clear shrink + MIN_ML_WIN_PCT in production
    side, ev, dec = select_ml_bet(
        0.68, -150,
        min_ev=0.03,
        min_win_pct=50,
        max_favorite_decimal=1.45,
        winner_only=True,
    )
    assert side == "Home"
    assert ev > 0.03


def test_select_ml_bet_winner_only_skips_dog_with_ev_but_model_picks_fav():
    # Model picks home (55%) but home is -300 dog has +EV — should NOT bet away
    side, _, _ = select_ml_bet(
        0.55, -300,
        min_ev=0.03,
        min_win_pct=50,
        max_favorite_decimal=1.45,
        winner_only=True,
    )
    # Home fav dec=1.33 < max_favorite_decimal 1.45 → Pass (odds filter)
    assert side == "Pass"


def test_select_ml_bet_winner_only_bets_model_underdog_when_picked():
    # Model picks away (35% home = 65% away), away +150 (dec 2.5)
    side, ev, dec = select_ml_bet(
        0.35, 150,
        min_ev=0.03,
        min_win_pct=50,
        max_favorite_decimal=1.45,
        winner_only=True,
    )
    assert side == "Away"
    assert ev > 0.03
    assert dec >= 1.45


def test_select_ml_bet_coin_flip_bets_high_ev_underdog():
    # Toss-up band but away is the dog; model leans away enough for EV + win%
    side, ev, _ = select_ml_bet(
        0.45, -130,
        min_ev=0.05,
        min_win_pct=50,
        max_favorite_decimal=1.45,
        winner_only=True,
        coin_flip_band=0.05,
        coin_flip_underdog_only=True,
        coin_flip_underdog_min_ev=0.05,
    )
    assert side == "Away"
    assert ev > 0.05


def test_select_ml_bet_coin_flip_skips_marginal_underdog_ev():
    # Toss-up with no real underdog edge (away +110)
    side, _, _ = select_ml_bet(
        0.52, -110,
        min_ev=0.05,
        min_win_pct=50,
        max_favorite_decimal=1.45,
        winner_only=True,
        coin_flip_band=0.03,
        coin_flip_underdog_only=True,
        coin_flip_underdog_min_ev=0.08,
    )
    assert side == "Pass"


def test_select_ml_bet_skips_when_ev_below_min():
    # Confident home pick but edge does not clear 5% bar
    side, _, _ = select_ml_bet(
        0.55, -110,
        min_ev=0.05,
        min_win_pct=50,
        max_favorite_decimal=1.45,
        winner_only=True,
    )
    assert side == "Pass"


def test_select_ml_bet_skips_when_win_pct_below_min():
    side, _, _ = select_ml_bet(
        0.56, -150,
        min_ev=0.03,
        min_win_pct=58,
        max_favorite_decimal=1.45,
        winner_only=True,
    )
    assert side == "Pass"


def test_select_ml_bet_legacy_best_ev_side():
    # Model picks home but away has higher EV — legacy mode bets away
    side, _, _ = select_ml_bet(
        0.28, 200,
        min_ev=0.03,
        min_win_pct=50,
        max_favorite_decimal=1.45,
        winner_only=False,
    )
    assert side == "Away"
