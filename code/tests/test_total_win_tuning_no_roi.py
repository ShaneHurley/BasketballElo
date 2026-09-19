"""Total / MetaWin Optuna objectives must be proper-scoring only (no ROI)."""
from __future__ import annotations

import inspect

from pipeline.model import _meta_win_tuning_objective, tune_total_model


def test_meta_win_objective_has_no_roi_term():
    src = inspect.getsource(_meta_win_tuning_objective)
    for banned in ("ml_roi", "META_WIN_LOSS_ML_ROI_WEIGHT", "fair_home_win_prob", "best_r"):
        assert banned not in src, f"found banned ROI term {banned!r} in meta-win loss"


def test_tune_total_model_source_has_no_ou_roi():
    src = inspect.getsource(tune_total_model)
    for banned in ("ou_roi", "TOTAL_LOSS_OU_ROI_WEIGHT", "best_r"):
        assert banned not in src, f"found banned ROI term {banned!r} in total-model loss"
