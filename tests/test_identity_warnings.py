"""GAP 1: splitguard warns when row identity is value-based and unreliable."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

from splitguard import _core, _patch


@pytest.fixture(autouse=True)
def _clean_state():
    _core._state.reset()
    _core._warned_mixed_identity = False
    _core._warned_value_dupes = False
    _core._warned_pre_split_transform = False
    _core._warned_no_split = False
    yield
    if _patch._installed:
        _patch.uninstall()
    _core._state.reset()


def test_numpy_duplicate_held_out_rows_warns():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(120, 4))
    X[90:] = X[:30]  # duplicate rows -> value identity is ambiguous
    y = (rng.random(120) > 0.5).astype(int)
    _patch.install()
    with pytest.warns(UserWarning, match="duplicate"):
        model_selection.train_test_split(X, y, random_state=0)


def test_mixed_pandas_split_numpy_fit_warns():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 4))
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(4)])
    y = pd.Series((rng.random(120) > 0.5).astype(int))
    _patch.install()
    with pytest.warns(UserWarning, match="identit"):
        model_selection.train_test_split(df, y, random_state=0)  # index-based taint
        StandardScaler().fit(X)  # NumPy fit -> identities cannot match


def test_consistent_pandas_does_not_warn():
    rng = np.random.default_rng(2)
    df = pd.DataFrame(rng.normal(size=(120, 4)), columns=[f"f{i}" for i in range(4)])
    y = pd.Series((rng.random(120) > 0.5).astype(int))
    _patch.install()
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("error")  # any splitguard warning would fail the test
        X_tr, _X_te, y_tr, _y_te = model_selection.train_test_split(df, y, random_state=0)
        StandardScaler().fit(X_tr)


def test_strict_transforms_warns_on_fit_transform_then_split():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(120, 5))
    y = (rng.random(120) > 0.5).astype(int)
    with pytest.warns(UserWarning, match="strict"):  # GAP 2: opt-in pre-split transformer
        with _patch.guard(policy="log", strict_transforms=True):
            X2 = StandardScaler().fit_transform(X)  # transformer fitted before the split
            model_selection.train_test_split(X2, y, random_state=0)


def test_warns_when_fit_but_no_split_tracked():
    # the import-order trap: fits happen but train_test_split was never intercepted
    rng = np.random.default_rng(5)
    X = rng.normal(size=(120, 5))
    with pytest.warns(UserWarning, match="no held-out split was tracked"):
        with _patch.guard(policy="log"):
            StandardScaler().fit(X)  # a fit, but no split -> nothing could be checked


def test_no_warning_when_split_is_tracked():
    import warnings

    rng = np.random.default_rng(6)
    X = rng.normal(size=(120, 5))
    y = (rng.random(120) > 0.5).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # a correctly tracked split must not trigger the warning
        with _patch.guard(policy="log"):
            X_tr, _X_te, y_tr, _y_te = model_selection.train_test_split(X, y, random_state=0)
            StandardScaler().fit(X_tr)


def test_fit_transform_silent_without_strict():
    import warnings

    rng = np.random.default_rng(4)
    X = rng.normal(size=(120, 5))
    y = (rng.random(120) > 0.5).astype(int)
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # default must not warn (no false positives)
        with _patch.guard(policy="log"):
            X2 = StandardScaler().fit_transform(X)
            model_selection.train_test_split(X2, y, random_state=0)
