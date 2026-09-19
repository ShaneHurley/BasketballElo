"""Elo blend cap protects lean |edge| from Optuna market-hug."""
from pipeline.config import ELO_BLEND_ALPHA_MAX
from pipeline.model import MetaScoreModel


def test_elo_blend_alpha_max_clips_init():
    m = MetaScoreModel(elo_blend_alpha=0.55)
    assert m.elo_blend_alpha <= float(ELO_BLEND_ALPHA_MAX) + 1e-9


def test_dynamic_blend_respects_max():
    m = MetaScoreModel(elo_blend_alpha=0.55, dynamic_elo_blend=True)
    a = m._dynamic_elo_blend_alpha({"uncertainty_diff": 400.0})
    assert a <= float(ELO_BLEND_ALPHA_MAX) + 1e-9
