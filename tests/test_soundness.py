"""Soundness invariants: the detector must never break or alter the user's program."""

from __future__ import annotations

import logging

import numpy as np
import pytest
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

from splitguard import _core, _patch


@pytest.fixture(autouse=True)
def _clean_state():
    _core._state.reset()
    _core._state.policy_override = None
    yield
    if _patch._installed:
        _patch.uninstall()
    _core._state.reset()


def _data(n=40, p=3, seed=1):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), (rng.random(n) > 0.5).astype(int)


def test_user_fit_survives_internal_error(monkeypatch):
    X, _ = _data()
    _patch.install()

    def boom(*_args, **_kwargs):
        raise RuntimeError("internal detector fault")

    monkeypatch.setattr(_core, "on_fit", boom)
    scaler = StandardScaler()
    returned = scaler.fit(X)  # must not raise despite the internal fault
    _patch.uninstall()

    assert returned is scaler


def test_import_does_not_pull_pandas():
    # importing the package must not import pandas eagerly
    import importlib
    import sys

    had_pandas = "pandas" in sys.modules
    sys.modules.pop("pandas", None)
    importlib.import_module("splitguard")
    if not had_pandas:
        assert "pandas" not in sys.modules, "splitguard must not import pandas at import time"


def test_log_policy_does_not_raise(caplog):
    X, y = _data()
    with caplog.at_level(logging.WARNING, logger="splitguard"):
        with _patch.guard(policy="log"):
            StandardScaler().fit(X)
            model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
    assert any("leak" in r.message.lower() for r in caplog.records)


def test_raise_policy_raises_on_leak():
    X, y = _data()
    with pytest.raises(_core.LeakageError):
        with _patch.guard(policy="raise"):
            StandardScaler().fit(X)
            model_selection.train_test_split(X, y, test_size=0.25, random_state=0)


def test_config_does_not_bleed_across_guards():
    # a non-default config in one guard must not survive into the next
    with _patch.guard(policy="warn", min_leaked_rows=10):
        pass
    with _patch.guard(policy="log"):
        assert _core._state.min_leaked_rows == 1
        assert _core._state.policy_override == "log"


def test_clean_run_under_guard_is_silent():
    X, y = _data()
    with _patch.guard(policy="raise"):
        X_tr, _, _, _ = model_selection.train_test_split(X, y, test_size=0.25, random_state=0)
        StandardScaler().fit(X_tr)  # train-only fit, no leak -> no raise
    assert _core.report() == []
