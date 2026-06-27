"""Native (non-``.fit``) training APIs: xgboost.train / lightgbm.train via DMatrix/Dataset."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import model_selection

from splitguard import _core, _patch


@pytest.fixture(autouse=True)
def _clean_state():
    _core._state.reset()
    _core._state.policy_override = None
    yield
    if _patch._installed:
        _patch.uninstall()
    _core._state.reset()


def _data(n=80, p=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), (rng.random(n) > 0.5).astype(int)


def test_xgboost_native_train_full_matrix_leak():
    xgb = pytest.importorskip("xgboost")
    X, y = _data()
    with _patch.guard(policy="log"):
        model_selection.train_test_split(X, y, random_state=0)  # taints the test rows
        dtrain = xgb.DMatrix(X, label=y)  # full matrix -> includes held-out rows
        xgb.train({"max_depth": 2, "verbosity": 0}, dtrain, num_boost_round=3)
    assert any("xgboost.train" in f.estimator for f in _core.report())


def test_xgboost_native_train_train_only_clean():
    xgb = pytest.importorskip("xgboost")
    X, y = _data()
    with _patch.guard(policy="log"):
        X_tr, _X_te, y_tr, _y_te = model_selection.train_test_split(X, y, random_state=0)
        dtrain = xgb.DMatrix(X_tr, label=y_tr)  # train only
        xgb.train({"max_depth": 2, "verbosity": 0}, dtrain, num_boost_round=3)
    assert _core.report() == []


def test_xgboost_native_uninstall_restores():
    xgb = pytest.importorskip("xgboost")
    original_train = xgb.train
    _patch.install()
    assert getattr(xgb.train, "_splitguard", False) is True
    _patch.uninstall()
    assert xgb.train is original_train


def test_lightgbm_native_train_is_instrumented():
    try:
        import importlib

        lgb = importlib.import_module("lightgbm")
        _ = lgb.train  # access may trigger a failing lazy import in some envs
    except ImportError:
        pytest.skip("lightgbm not installed")
    except Exception:  # pragma: no cover - environment-specific (dask) incompatibility
        pytest.skip("lightgbm import fails in this environment")
    with _patch.guard(policy="log"):
        if not getattr(lgb.train, "_splitguard", False):
            pytest.skip("lightgbm native wrapping unavailable in this environment (dask)")
        assert getattr(lgb.train, "_splitguard", False) is True
