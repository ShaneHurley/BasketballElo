"""Negative-control suite smoke tests."""
from __future__ import annotations

import numpy as np

from pipeline.negative_controls import (
    future_line_injection_should_improve,
    quote_after_cutoff_rejected,
    run_negative_control_suite,
    shuffled_outcomes_destroy_edge,
)


def test_shuffled_outcomes_control():
    rng = np.random.default_rng(0)
    y = rng.normal(size=200)
    pred = y + rng.normal(scale=0.1, size=200)

    def mae(a, b):
        return float(np.mean(np.abs(a - b)))

    out = shuffled_outcomes_destroy_edge(y, pred, metric_fn=mae, n_perm=20)
    assert out["passed"] is True
    assert out["null_mean"] > out["base_metric"]


def test_future_line_injection_control():
    out = future_line_injection_should_improve(13.0, 10.0)
    assert out["passed"] is True


def test_quote_after_cutoff_rejects_late_quote():
    out = quote_after_cutoff_rejected()
    assert out["passed"] is True
    assert abs(out["selected_price"] - (-110.0)) < 1e-9


def test_run_suite_includes_close_guard_and_cutoff():
    report = run_negative_control_suite()
    assert report["head"] == "negative_controls"
    assert report["metrics"]["close_column_guard"]["passed"] is True
    assert report["metrics"]["quote_after_cutoff"]["passed"] is True
