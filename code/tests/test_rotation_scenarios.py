"""Task 034: correlated multi-player rotation scenarios must be generated
jointly (not as an independent per-player product), respect roster/minute
constraints, expose probability and entropy, and sum to exactly one."""
import math

import pytest

from pipeline.minutes_forecast import MinutesForecastModel, TEAM_EXPECTED_MINUTES
from pipeline.rotation_scenarios import (
    RotationScenarioGenerator,
    scenario_entropy,
)


def _model():
    m = MinutesForecastModel(window_games=10)
    for pid in range(6):
        for _ in range(10):
            m.record_game(str(pid), minutes=28.0, active=True)
    return m


def test_scenario_probabilities_sum_to_one():
    m = _model()
    gen = RotationScenarioGenerator(m, max_players_for_scenarios=3)
    roster = [str(i) for i in range(6)]
    scenarios = gen.generate(roster)
    total = sum(sc.probability for sc in scenarios)
    assert total == pytest.approx(1.0, abs=1e-9)


def test_every_scenario_respects_240_minute_constraint():
    m = _model()
    gen = RotationScenarioGenerator(m, max_players_for_scenarios=3)
    roster = [str(i) for i in range(6)]
    scenarios = gen.generate(roster)
    for sc in scenarios:
        assert sum(sc.minutes.values()) == pytest.approx(TEAM_EXPECTED_MINUTES)
        for pid, mins in sc.minutes.items():
            assert mins >= -1e-9
            if pid != "_replacement":
                assert mins <= 48.0 + 1e-6


def test_out_status_scenario_has_zero_minutes_for_that_player():
    m = _model()
    gen = RotationScenarioGenerator(m, max_players_for_scenarios=2)
    roster = [str(i) for i in range(6)]
    scenarios = gen.generate(roster)
    out_scenarios = [sc for sc in scenarios if any(v == "OUT" for v in sc.statuses.values())]
    assert out_scenarios, "expected at least one scenario with a player OUT"
    for sc in out_scenarios:
        for pid, status in sc.statuses.items():
            if status == "OUT":
                assert sc.minutes[pid] == pytest.approx(0.0)


def test_scenarios_are_correlated_not_an_independent_product():
    """The joint distribution must not decompose as a product of
    independent per-player marginals -- the shared 'rest_night' factor
    should make two players' OUT statuses positively correlated versus
    treating them as independent Bernoullis."""
    m = _model()
    gen = RotationScenarioGenerator(
        m, max_players_for_scenarios=2,
        factor_probs={"normal": 0.5, "rest_night": 0.5},
        factor_multiplier={"normal": 1.0, "rest_night": 0.0},  # rest_night forces OUT
    )
    roster = ["0", "1"]
    scenarios = gen.generate(roster)

    def p_out(pid):
        return sum(sc.probability for sc in scenarios if sc.statuses[pid] == "OUT")

    def p_both_out():
        return sum(
            sc.probability for sc in scenarios
            if sc.statuses["0"] == "OUT" and sc.statuses["1"] == "OUT"
        )

    p0, p1 = p_out("0"), p_out("1")
    independent_prediction = p0 * p1
    assert p_both_out() > independent_prediction + 1e-6, (
        "joint OUT probability was consistent with independence; the "
        "shared-factor correlation was not actually applied"
    )


def test_entropy_is_zero_for_a_fully_deterministic_scenario_set():
    m = _model()
    gen = RotationScenarioGenerator(
        m, max_players_for_scenarios=1,
        factor_probs={"only": 1.0},
        factor_multiplier={"only": 1.0},
        limited_prob=0.0,
    )
    scenarios = gen.generate(["0"], statuses={"0": "OUT"})
    # Forcing status="OUT" collapses the branch player's own distribution,
    # but the factor mixture still has one state -> entropy 0.
    ent = scenario_entropy(scenarios)
    assert ent == pytest.approx(0.0, abs=1e-9)


def test_entropy_is_positive_when_multiple_scenarios_have_mass():
    m = _model()
    gen = RotationScenarioGenerator(m, max_players_for_scenarios=2)
    roster = [str(i) for i in range(6)]
    scenarios = gen.generate(roster)
    ent = scenario_entropy(scenarios)
    assert ent > 0.0
    # Sanity upper bound: entropy of a distribution over N outcomes <= log(N).
    assert ent <= math.log(len(scenarios)) + 1e-9


def test_branching_limited_to_max_players_for_scenarios():
    # Use a model where every branch player has an interior (0,1) active
    # probability, so no combo's probability is exactly zero and the full
    # combinatorial scenario count (factors x statuses^branch_players) is
    # realized exactly.
    m = MinutesForecastModel(window_games=10)
    for pid in range(6):
        for i in range(10):
            m.record_game(str(pid), minutes=28.0, active=(i % 3 != 0))
    gen = RotationScenarioGenerator(m, max_players_for_scenarios=2)
    roster = [str(i) for i in range(6)]
    scenarios = gen.generate(roster)
    assert len(scenarios) == 2 * 3 ** 2
    # Every branch-eligible scenario's status dict still covers the full
    # roster (fixed players default to ACTIVE), never just the branch subset.
    assert all(len(sc.statuses) == len(roster) for sc in scenarios)


def test_invalid_factor_probs_rejected():
    m = _model()
    with pytest.raises(ValueError):
        RotationScenarioGenerator(m, factor_probs={"a": 0.5, "b": 0.4})
