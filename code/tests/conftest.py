"""Pytest config for the BasketballElo test suite."""
from __future__ import annotations

import pytest


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "bench: synthetic formula test bench (Epic 10) — extreme/adversarial inputs",
    )
