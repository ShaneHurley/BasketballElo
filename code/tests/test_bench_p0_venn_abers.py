"""Epic 10.6 / P0.9 — Venn-Abers filter must fail closed on invalid width.

GitHub: P0.9 · Planned registry ID: ``bench_venn_abers_fail_open``

A risk-limiting filter that cannot evaluate an interval must **reject** the
bet, not approve it. Bug: ``passes_venn_abers_filter`` returns ``True`` when
width is ``None`` or non-finite (``venn_abers.py:35-37``).
"""
from __future__ import annotations

import math

import pytest

from pipeline.venn_abers import passes_venn_abers_filter


@pytest.mark.bench
@pytest.mark.xfail(
    strict=True,
    reason=(
        "P0.9: passes_venn_abers_filter fails open on None/NaN "
        "(venn_abers.py:35-37 returns True)"
    ),
)
@pytest.mark.parametrize("width", [None, float("nan"), float("inf"), float("-inf")])
def test_p0_9_venn_abers_fails_closed_on_invalid_width(width):
    """Acceptance: invalid width ⇒ filter rejects (returns False)."""
    assert passes_venn_abers_filter(width, max_width=0.25) is False


@pytest.mark.bench
def test_p0_9_finite_width_still_compares_to_max():
    """Finite widths keep the normal comparison (not part of the bug)."""
    assert passes_venn_abers_filter(0.10, max_width=0.25) is True
    assert passes_venn_abers_filter(0.40, max_width=0.25) is False
    assert math.isfinite(0.10)


@pytest.mark.bench
def test_p0_9_documents_bug_nan_currently_passes():
    """Regression witness: NaN currently returns True (fail-open)."""
    result = passes_venn_abers_filter(float("nan"), max_width=0.25)
    if result is True:
        # Bug still present — documented.
        return
    # Once fixed, NaN must reject.
    assert result is False
