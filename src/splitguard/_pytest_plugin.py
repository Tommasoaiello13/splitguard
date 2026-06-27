"""pytest integration: fail a test when data leakage occurs during its execution.

Usage
-----
Per-test, request the ``no_leakage`` fixture::

    def test_pipeline(no_leakage):
        ...  # any leakage raises LeakageError -> the test fails

Project-wide, enable the autouse guard by adding to ``pytest.ini`` / ``pyproject.toml``::

    [tool.pytest.ini_options]
    splitguard = true
"""

from __future__ import annotations

import pytest

from . import _patch


def pytest_addoption(parser):
    parser.addini("splitguard", "guard every test against data leakage", type="bool", default=False)


@pytest.fixture
def no_leakage():
    """Fail the requesting test if any fitted estimator sees held-out rows."""
    with _patch.guard(policy="raise"):
        yield


@pytest.fixture(autouse=True)
def _splitguard_autouse(request):
    if not request.config.getini("splitguard"):
        yield
        return
    with _patch.guard(policy="raise"):
        yield
