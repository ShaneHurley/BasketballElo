"""Regression: backtest must import every class it constructs at runtime."""
from __future__ import annotations

import ast
from pathlib import Path


def test_ats_classifier_is_imported_in_backtest():
    src = Path(__file__).resolve().parents[1] / "pipeline" / "backtest.py"
    if not src.exists():
        # When tests live under BasketballElo/code/tests
        src = Path(__file__).resolve().parents[1] / "pipeline" / "backtest.py"
    tree = ast.parse(src.read_text())
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            for alias in node.names:
                imported.add(alias.asname or alias.name)
    assert "ATSClassifier" in imported, (
        "pipeline.backtest constructs ATSClassifier but does not import it "
        "(NameError after MetaWin tuning in walk-forward)."
    )


def test_ats_classifier_import_resolves():
    from pipeline.backtest import ATSClassifier as Imported
    from pipeline.ats_classifier import ATSClassifier as Canonical

    assert Imported is Canonical
