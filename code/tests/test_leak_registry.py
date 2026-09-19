"""Task 005: the leak registry document lists every currently confirmed leak."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
REGISTRY_PATH = REPO_ROOT / "LEAK_REGISTRY.md"

REQUIRED_LEAK_IDS = (
    "score_corruption",
    "date_corruption",
    "close_line_conflation",
    "stack_cv_future_leakage",
    "clv_definition_bug",
    "push_as_loss_bug",
    "cap_after_profit_bug",
    "ml_ev_still_negates_away_price",
    "close_residual_vs_decision_residual",
    "quote_tip_proxy",
    "missing_total_zero_sentinel",
    "unresolved_date_now_fallback",
)


def test_leak_registry_exists():
    assert REGISTRY_PATH.exists(), "LEAK_REGISTRY.md is missing"


def test_leak_registry_lists_all_required_defects():
    text = REGISTRY_PATH.read_text()
    for leak_id in REQUIRED_LEAK_IDS:
        assert f"`{leak_id}`" in text, f"leak registry missing entry for {leak_id!r}"


def test_leak_registry_has_regression_test_and_status_columns():
    text = REGISTRY_PATH.read_text()
    assert "Regression test" in text
    assert "Status" in text
    assert "Affected artifacts" in text
