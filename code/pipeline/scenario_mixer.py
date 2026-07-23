"""Task 035: posterior-predictive rotation-scenario mixer.

Feeds each valid rotation scenario (Task 034) through a player/team margin
model, then mixes the resulting per-scenario margin distributions into one
posterior-predictive margin distribution, weighted by scenario probability.

Mixing uses the exact law of total expectation/variance for a finite
mixture (no approximation):

  E[margin]   = sum_i p_i * E[margin | scenario_i]
  Var[margin] = E[Var[margin | scenario_i]] + Var[E[margin | scenario_i]]
              = sum_i p_i * (var_i + mean_i^2) - E[margin]^2

This is exact for any finite mixture of distributions with finite first and
second moments, regardless of each component's shape (Gaussian components
are the common case here, but the mixing formula itself does not assume
normality).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

from pipeline.rotation_scenarios import RotationScenario


@dataclass
class ScenarioPrediction:
    scenario: RotationScenario
    mean_margin: float
    var_margin: float


@dataclass
class MixedForecast:
    mean: float
    variance: float
    components: List[ScenarioPrediction]

    @property
    def std(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


def mix_scenario_predictions(predictions: Sequence[ScenarioPrediction]) -> MixedForecast:
    """Exact posterior-predictive mixing given each scenario's own
    (probability, mean, variance)."""
    if not predictions:
        raise ValueError("mix_scenario_predictions: no scenario predictions supplied")

    probs = [sp.scenario.probability for sp in predictions]
    total_prob = sum(probs)
    if abs(total_prob - 1.0) > 1e-6:
        raise ValueError(
            f"scenario probabilities must sum to 1.0 before mixing, got {total_prob}"
        )
    if any(sp.var_margin < 0 for sp in predictions):
        raise ValueError("mix_scenario_predictions: a component variance was negative")

    mean = sum(p * sp.mean_margin for p, sp in zip(probs, predictions))
    second_moment = sum(p * (sp.var_margin + sp.mean_margin ** 2) for p, sp in zip(probs, predictions))
    variance = second_moment - mean ** 2
    # Floating-point mixtures of near-degenerate components can produce a
    # tiny negative variance; clip rather than propagate a nonsensical value.
    variance = max(variance, 0.0)
    return MixedForecast(mean=mean, variance=variance, components=list(predictions))


def predict_and_mix_scenarios(
    scenarios: Sequence[RotationScenario],
    predict_fn: Callable[[RotationScenario], Tuple[float, float]],
) -> MixedForecast:
    """Run every scenario through `predict_fn` (the player/team margin
    model applied to that scenario's rotation/minutes) and mix the results.

    `predict_fn(scenario) -> (mean_margin, var_margin)`.
    """
    predictions = [
        ScenarioPrediction(scenario=sc, mean_margin=m, var_margin=v)
        for sc in scenarios
        for m, v in [predict_fn(sc)]
    ]
    return mix_scenario_predictions(predictions)
