"""Epic 10.6 / P0.9 — Venn-Abers filter must fail closed on invalid width.

GitHub: P0.9 · Registry ID: ``bench_venn_abers_fail_open`` (fixed)

A risk-limiting filter that cannot evaluate an interval must **reject** the
bet, not approve it.
"""
from __future__ import annotations

import math

import pytest

from pipeline.venn_abers import passes_venn_abers_filter


@pytest.mark.bench
@pytest.mark.parametrize("width", [None, float("nan"), float("inf"), float("-inf")])
def test_p0_9_venn_abers_fails_closed_on_invalid_width(width):
    """Acceptance: invalid width ⇒ filter rejects (returns False)."""
    assert passes_venn_abers_filter(width, max_width=0.25) is False


@pytest.mark.bench
def test_p0_9_finite_width_still_compares_to_max():
    """Finite widths keep the normal comparison."""
    assert passes_venn_abers_filter(0.10, max_width=0.25) is True
    assert passes_venn_abers_filter(0.40, max_width=0.25) is False
    assert math.isfinite(0.10)


@pytest.mark.bench
def test_p0_9_nan_rejects():
    """Witness: NaN must reject (fail-closed)."""
    assert passes_venn_abers_filter(float("nan"), max_width=0.25) is False
