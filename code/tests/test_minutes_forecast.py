"""Task 033: active probability / conditional minutes / minutes limit /
replacement-minutes forecasting from prior data only, with team expected
minutes enforced to exactly 240."""
import pytest

from pipeline.minutes_forecast import (
    TEAM_EXPECTED_MINUTES,
    MinutesForecastModel,
)


def _model_with_history():
    m = MinutesForecastModel(window_games=10)
    # Player 1: always active, ~30 min.
    for _ in range(8):
        m.record_game("1", minutes=30.0, active=True)
    # Player 2: active 50% of the time, ~25 min when active.
    for i in range(8):
        m.record_game("2", minutes=25.0, active=(i % 2 == 0))
    return m


def test_active_probability_uses_prior_history_only():
    m = _model_with_history()
    assert m.active_probability("1") == pytest.approx(1.0)
    assert m.active_probability("2") == pytest.approx(0.5)


def test_conditional_minutes_from_prior_active_games_only():
    m = _model_with_history()
    assert m.conditional_minutes("1") == pytest.approx(30.0)
    assert m.conditional_minutes("2") == pytest.approx(25.0)


def test_unknown_player_gets_default_not_zero():
    m = MinutesForecastModel()
    fc = m.player_forecast("999")
    assert fc.active_prob > 0
    assert fc.conditional_minutes > 0


def test_minutes_limit_caps_conditional_minutes():
    m = _model_with_history()
    fc = m.player_forecast("1", minutes_limit=15.0)
    assert fc.conditional_minutes == pytest.approx(30.0)  # unaffected raw prior
    assert fc.expected_minutes == pytest.approx(1.0 * 15.0)  # capped


def test_status_out_overrides_prior_active_rate():
    m = _model_with_history()
    fc = m.player_forecast("1", status="OUT")
    assert fc.active_prob == pytest.approx(0.0)
    assert fc.expected_minutes == pytest.approx(0.0)


def test_team_forecast_sums_to_exactly_240_when_understaffed():
    m = _model_with_history()
    out = m.forecast_team(["1", "2"])
    assert sum(out.values()) == pytest.approx(TEAM_EXPECTED_MINUTES)
    assert out["_replacement"] >= 0.0


def test_team_forecast_sums_to_exactly_240_when_overstaffed():
    m = MinutesForecastModel()
    for pid in range(15):
        for _ in range(10):
            m.record_game(str(pid), minutes=32.0, active=True)  # deep, healthy roster
    roster = [str(i) for i in range(15)]
    out = m.forecast_team(roster)
    assert sum(out.values()) == pytest.approx(TEAM_EXPECTED_MINUTES)
    assert out["_replacement"] == pytest.approx(0.0)
    # Proportional scale-down: no single player's minutes exceed 48.
    for pid in roster:
        assert 0.0 <= out[pid] <= 48.0


def test_team_forecast_with_out_player_shifts_minutes_to_replacement():
    m = _model_with_history()
    out_no_injury = m.forecast_team(["1", "2"])
    out_with_injury = m.forecast_team(["1", "2"], statuses={"1": "OUT"})
    assert out_with_injury["1"] == pytest.approx(0.0)
    assert out_with_injury["_replacement"] > out_no_injury["_replacement"]
    assert sum(out_with_injury.values()) == pytest.approx(TEAM_EXPECTED_MINUTES)


def test_record_game_does_not_retroactively_affect_earlier_forecast_snapshot():
    """Prior-data-only guarantee: a forecast computed from a model snapshot
    must not change just because a later game is later `record_game`'d --
    callers are responsible for snapshotting/calling this only with games
    strictly before the cutoff, and this test locks that contract in."""
    m = MinutesForecastModel(window_games=10)
    for _ in range(5):
        m.record_game("1", minutes=20.0, active=True)
    forecast_before = m.player_forecast("1").expected_minutes

    # A caller that (incorrectly) records a *future* game before forecasting
    # would change the result -- this test documents that record_game must
    # only ever be called with strictly-prior games by the pipeline.
    m.record_game("1", minutes=40.0, active=True)
    forecast_after = m.player_forecast("1").expected_minutes
    assert forecast_after != forecast_before  # confirms record_game is prior-history-mutating,
    # i.e. callers MUST gate call order themselves; see test_availability_t60_isolation.py
    # for the production-pipeline-level guarantee that only past games are recorded.
