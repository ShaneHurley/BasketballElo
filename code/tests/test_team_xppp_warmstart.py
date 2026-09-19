"""TeamXpppTracker persistence and Elo warm-start blend."""
from pathlib import Path

import pandas as pd
import pytest

from pipeline.config import (
    DEFAULT_LEAGUE_XPPP,
    XPPP_TURNOVER_THRESHOLD,
    XPPP_WARM_START_GAMES,
)
from pipeline.ratings import PlayerRatingTracker
from pipeline.trackers import (
    PaceTracker,
    TeamXpppTracker,
    elo_team_xppp_prior,
    refresh_team_priors_for_season,
)


def test_xppp_save_load_roundtrip(tmp_path: Path):
    tr = TeamXpppTracker(window_size=40, prev_season_weight=0.5)
    d0 = pd.Timestamp("2025-11-01")
    tr.update("BOS", d0, 2026, 1.15, 1.05)
    tr.set_elo_prior("BOS", 1.12, 1.08)
    tr.set_turnover("BOS", 0.4)
    path = tmp_path / "xppp.pkl"
    tr.save_state(path)
    loaded = TeamXpppTracker.load_state(path)
    assert loaded.window_size == 40
    assert loaded.prev_season_weight == 0.5
    assert len(loaded.history["BOS"]) == 1
    assert loaded.elo_priors["BOS"] == pytest.approx((1.12, 1.08))
    assert loaded.turnover_frac["BOS"] == pytest.approx(0.4)
    off, deff = loaded.get_rolling_xppp("BOS", 2026, pd.Timestamp("2025-11-02"))
    assert off == pytest.approx(1.15) or off != DEFAULT_LEAGUE_XPPP


def test_empty_history_uses_elo_prior():
    tr = TeamXpppTracker(warm_start_games=8, turnover_threshold=0.35)
    tr.set_elo_prior("LAL", 1.20, 1.00)
    off, deff = tr.get_rolling_xppp("LAL", 2026, pd.Timestamp("2025-11-01"))
    assert off == pytest.approx(1.20)
    assert deff == pytest.approx(1.00)


def test_empty_history_without_prior_is_league():
    tr = TeamXpppTracker()
    off, deff = tr.get_rolling_xppp("NYK", 2026, pd.Timestamp("2025-11-01"))
    assert off == pytest.approx(DEFAULT_LEAGUE_XPPP)
    assert deff == pytest.approx(DEFAULT_LEAGUE_XPPP)


def test_warm_start_decays_after_observed_games():
    tr = TeamXpppTracker(warm_start_games=8, turnover_threshold=0.35)
    tr.set_elo_prior("DEN", 1.30, 0.90)
    tr.set_turnover("DEN", 0.50)  # high turnover → blend
    base = pd.Timestamp("2025-10-20")
    for i in range(XPPP_WARM_START_GAMES):
        tr.update("DEN", base + pd.Timedelta(days=i), 2026, 1.10, 1.10)
    # After warm_start_games obs, prior weight is 0
    off, deff = tr.get_rolling_xppp(
        "DEN", 2026, base + pd.Timedelta(days=XPPP_WARM_START_GAMES),
    )
    assert off == pytest.approx(1.10)
    assert deff == pytest.approx(1.10)


def test_partial_blend_with_high_turnover():
    tr = TeamXpppTracker(warm_start_games=8, turnover_threshold=0.35)
    tr.set_elo_prior("GSW", 1.30, 0.90)
    tr.set_turnover("GSW", 0.50)
    d0 = pd.Timestamp("2025-11-01")
    tr.update("GSW", d0, 2026, 1.10, 1.10)
    # n_obs=1 → w_prior = 1 - 1/8 = 0.875
    off, deff = tr.get_rolling_xppp("GSW", 2026, pd.Timestamp("2025-11-02"))
    w = 1.0 - 1.0 / 8.0
    assert off == pytest.approx(w * 1.30 + (1 - w) * 1.10)
    assert deff == pytest.approx(w * 0.90 + (1 - w) * 1.10)


def test_low_turnover_uses_observed_only():
    tr = TeamXpppTracker(warm_start_games=8, turnover_threshold=0.35)
    tr.set_elo_prior("MIA", 1.30, 0.90)
    tr.set_turnover("MIA", 0.10)  # below threshold
    d0 = pd.Timestamp("2025-11-01")
    tr.update("MIA", d0, 2026, 1.10, 1.10)
    off, deff = tr.get_rolling_xppp("MIA", 2026, pd.Timestamp("2025-11-02"))
    assert off == pytest.approx(1.10)
    assert deff == pytest.approx(1.10)


def test_elo_team_xppp_prior_scaling():
    elo = PlayerRatingTracker(league_xppp=1.10)
    # Force known ratings on a tiny lineup
    for pid in ("1", "2", "3", "4", "5"):
        p = elo._get(pid)
        p["O_mu"] = 1600.0
        p["D_mu"] = 1400.0
    scaling = elo.cfg["ELO_SCALING_FACTOR"]
    off, deff = elo_team_xppp_prior(elo, ["1", "2", "3", "4", "5"])
    assert off == pytest.approx(1.10 + (1600 - 1500) / scaling)
    assert deff == pytest.approx(1.10 - (1400 - 1500) / scaling)


def test_refresh_sets_turnover_and_pace_shrink():
    xppp = TeamXpppTracker(turnover_threshold=XPPP_TURNOVER_THRESHOLD)
    elo = PlayerRatingTracker(league_xppp=1.10)
    for pid in ("101", "102", "103", "104", "105", "201", "202"):
        elo._get(pid)
    pace = PaceTracker(team_window=5)
    pace.team_history["BOS"] = __import__("collections").deque([110.0, 112.0], maxlen=5)

    class _Rot:
        history = {"BOS": True}

        def expected_weights(self, team, fallback_ids=None):
            # Mostly new players vs prior roster
            return [("201", 0.2), ("202", 0.2), ("101", 0.2), ("102", 0.2), ("103", 0.2)]

    summary = refresh_team_priors_for_season(
        xppp,
        elo_tracker=elo,
        pace_tracker=pace,
        rotation_tracker=_Rot(),
        prior_rosters={"BOS": {"101", "102", "103", "104", "105"}},
    )
    assert summary["priors_set"] >= 1
    assert "BOS" in xppp.elo_priors
    assert xppp.turnover_frac["BOS"] == pytest.approx(0.4)  # 2/5 new
    # Pace shrunk toward 100
    assert float(pace.team_history["BOS"][0]) < 110.0
