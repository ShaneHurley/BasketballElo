"""Task 035: posterior-predictive scenario mixer -- validated against a
hand-calculated toy example (mean/variance via the law of total variance)
and a Monte Carlo cross-check."""
import math

import numpy as np
import pytest

from pipeline.minutes_forecast import MinutesForecastModel
from pipeline.rotation_scenarios import RotationScenario
from pipeline.scenario_mixer import (
    MixedForecast,
    ScenarioPrediction,
    mix_scenario_predictions,
    predict_and_mix_scenarios,
)


def _toy_scenarios(p_a=0.5, p_b=0.5):
    sc_a = RotationScenario(scenario_id="A", statuses={}, minutes={}, probability=p_a)
    sc_b = RotationScenario(scenario_id="B", statuses={}, minutes={}, probability=p_b)
    return sc_a, sc_b


def test_hand_calculated_toy_example_mean_and_variance():
    """Two equally-likely scenarios, means +10 and -10, both variance 4.
    By hand:
      mean = 0.5*10 + 0.5*(-10) = 0
      Var  = E[Var|S] + Var[E|S]
           = (0.5*4 + 0.5*4) + (0.5*(10-0)^2 + 0.5*(-10-0)^2)
           = 4 + 100 = 104
    """
    sc_a, sc_b = _toy_scenarios()
    preds = [
        ScenarioPrediction(scenario=sc_a, mean_margin=10.0, var_margin=4.0),
        ScenarioPrediction(scenario=sc_b, mean_margin=-10.0, var_margin=4.0),
    ]
    mixed = mix_scenario_predictions(preds)
    assert mixed.mean == pytest.approx(0.0, abs=1e-9)
    assert mixed.variance == pytest.approx(104.0, abs=1e-9)
    assert mixed.std == pytest.approx(math.sqrt(104.0), abs=1e-9)


def test_hand_calculated_toy_example_unequal_weights():
    """p=0.8 mean=5 var=1; p=0.2 mean=-15 var=9.
    mean = 0.8*5 + 0.2*(-15) = 4 - 3 = 1
    Var  = 0.8*1 + 0.2*9 + [0.8*(5-1)^2 + 0.2*(-15-1)^2]
         = (0.8 + 1.8) + (0.8*16 + 0.2*256)
         = 2.6 + (12.8 + 51.2) = 2.6 + 64.0 = 66.6
    """
    sc_a, sc_b = _toy_scenarios(p_a=0.8, p_b=0.2)
    preds = [
        ScenarioPrediction(scenario=sc_a, mean_margin=5.0, var_margin=1.0),
        ScenarioPrediction(scenario=sc_b, mean_margin=-15.0, var_margin=9.0),
    ]
    mixed = mix_scenario_predictions(preds)
    assert mixed.mean == pytest.approx(1.0, abs=1e-9)
    assert mixed.variance == pytest.approx(66.6, abs=1e-9)


def test_monte_carlo_matches_closed_form_mixing():
    """Cross-check the closed-form mixing formula against a large Monte
    Carlo sample drawn from the actual Gaussian mixture."""
    rng = np.random.default_rng(0)
    scenarios_spec = [
        (0.3, 2.0, 1.5),
        (0.5, -4.0, 3.0),
        (0.2, 12.0, 0.5),
    ]
    sc_objs = [
        RotationScenario(scenario_id=str(i), statuses={}, minutes={}, probability=p)
        for i, (p, _, _) in enumerate(scenarios_spec)
    ]
    preds = [
        ScenarioPrediction(scenario=sc, mean_margin=mean, var_margin=var)
        for sc, (p, mean, var) in zip(sc_objs, scenarios_spec)
    ]
    mixed = mix_scenario_predictions(preds)

    n = 2_000_000
    probs = [p for p, _, _ in scenarios_spec]
    choice = rng.choice(len(scenarios_spec), size=n, p=probs)
    samples = np.empty(n)
    for i, (p, mean, var) in enumerate(scenarios_spec):
        mask = choice == i
        samples[mask] = rng.normal(mean, math.sqrt(var), size=mask.sum())

    assert mixed.mean == pytest.approx(float(np.mean(samples)), abs=0.02)
    assert mixed.variance == pytest.approx(float(np.var(samples)), rel=0.01)


def test_rejects_scenarios_not_summing_to_one():
    sc_a, sc_b = _toy_scenarios(p_a=0.5, p_b=0.4)  # sums to 0.9, invalid
    preds = [
        ScenarioPrediction(scenario=sc_a, mean_margin=1.0, var_margin=1.0),
        ScenarioPrediction(scenario=sc_b, mean_margin=2.0, var_margin=1.0),
    ]
    with pytest.raises(ValueError):
        mix_scenario_predictions(preds)


def test_rejects_negative_component_variance():
    sc_a, sc_b = _toy_scenarios()
    preds = [
        ScenarioPrediction(scenario=sc_a, mean_margin=1.0, var_margin=-1.0),
        ScenarioPrediction(scenario=sc_b, mean_margin=2.0, var_margin=1.0),
    ]
    with pytest.raises(ValueError):
        mix_scenario_predictions(preds)


def test_predict_and_mix_scenarios_end_to_end_with_real_rotation_scenarios():
    """End-to-end: real Task 034 scenarios (from a real MinutesForecastModel)
    fed through a toy player/team `predict_fn`, mixed by scenario
    probability -- checks the pipeline wiring, not just the raw formula."""
    from pipeline.rotation_scenarios import RotationScenarioGenerator

    m = MinutesForecastModel(window_games=10)
    for pid in range(4):
        for i in range(10):
            m.record_game(str(pid), minutes=28.0, active=(i % 3 != 0))
    gen = RotationScenarioGenerator(m, max_players_for_scenarios=2)
    roster = [str(i) for i in range(4)]
    scenarios = gen.generate(roster)

    def toy_predict_fn(sc):
        # A toy "team model": each OUT starter costs 4 expected margin
        # points and adds 2 to variance; otherwise a flat baseline.
        n_out = sum(1 for s in sc.statuses.values() if s == "OUT")
        mean_margin = 3.0 - 4.0 * n_out
        var_margin = 25.0 + 2.0 * n_out
        return mean_margin, var_margin

    mixed = predict_and_mix_scenarios(scenarios, toy_predict_fn)
    assert isinstance(mixed, MixedForecast)
    assert mixed.variance > 0
    # More OUT-heavy scenarios pull the mean below the zero-OUT baseline.
    assert mixed.mean < 3.0
