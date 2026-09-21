"""Ablation 8.4 — RAPM vs HAPM/chem configs + prune gate."""
from __future__ import annotations

import pandas as pd
import pytest

from pipeline.ablation import (
    hapm_chem_prune_cols,
    rapm_hapm_ablation_configs,
    should_prune_hapm_chem,
)
from pipeline.model import EPVA_COLS, HAPM_COLS, RAPM_COLS

pytestmark = pytest.mark.bench


def test_rapm_hapm_ablation_configs_four_arms():
    cfgs = rapm_hapm_ablation_configs()
    assert set(cfgs) == {"rapm_only", "hapm_chem", "all", "rapm_only_retuned"}
    assert "cb_params_scale" in (cfgs["rapm_only_retuned"].get("model_kwargs") or {})
    # rapm_only drops HAPM; hapm_chem drops RAPM; all arms gate EPVA off
    rapm_only_feats = set(cfgs["rapm_only"]["feature_cols"])
    assert not (rapm_only_feats & set(HAPM_COLS))
    assert not (rapm_only_feats & set(EPVA_COLS))
    assert set(RAPM_COLS).issubset(rapm_only_feats)
    hapm_feats = set(cfgs["hapm_chem"]["feature_cols"])
    assert not (hapm_feats & set(RAPM_COLS))
    assert not (hapm_feats & set(EPVA_COLS))
    assert not (set(cfgs["all"]["feature_cols"]) & set(EPVA_COLS))
    for name in cfgs:
        assert cfgs[name]["backtest_kwargs"].get("use_epva_features") is False


def test_should_prune_hapm_chem_requires_retune_match():
    # all better mean → keep
    summary = pd.DataFrame([
        {"config": "rapm_only", "season": 2022, "spread_mae": 11.3},
        {"config": "rapm_only", "season": 2023, "spread_mae": 11.4},
        {"config": "all", "season": 2022, "spread_mae": 11.0},
        {"config": "all", "season": 2023, "spread_mae": 11.1},
        {"config": "rapm_only_retuned", "season": 2022, "spread_mae": 11.25},
        {"config": "rapm_only_retuned", "season": 2023, "spread_mae": 11.35},
    ])
    d = should_prune_hapm_chem(summary)
    assert d["prune"] is False
    assert "keep HAPM" in d["reason"] or "improves mean" in d["reason"]

    # retuned matches all fold-std at equal MAE → prune
    summary2 = pd.DataFrame([
        {"config": "rapm_only", "season": 2022, "spread_mae": 11.20},
        {"config": "rapm_only", "season": 2023, "spread_mae": 11.30},
        {"config": "all", "season": 2022, "spread_mae": 11.18},
        {"config": "all", "season": 2023, "spread_mae": 11.22},
        {"config": "rapm_only_retuned", "season": 2022, "spread_mae": 11.17},
        {"config": "rapm_only_retuned", "season": 2023, "spread_mae": 11.21},
    ])
    d2 = should_prune_hapm_chem(summary2)
    assert d2["prune"] is True
    cols = hapm_chem_prune_cols()
    assert set(HAPM_COLS).issubset(cols)
    assert "chem_diff" in cols
