"""Index-identity for pandas: duplicate-value rows must not cause false positives."""

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
    _core._state.policy_override = None
    yield
    if _patch._installed:
        _patch.uninstall()
    _core._state.reset()


def _df_with_duplicate_values(n=200, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    X[150:] = X[:50]  # 50 legitimately identical rows (same values, different index)
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(5)])
    y = pd.Series((rng.random(n) > 0.5).astype(int))
    return df, y


def test_pandas_duplicate_rows_no_false_positive():
    df, y = _df_with_duplicate_values()
    _patch.install()
    X_tr, _X_te, _y_tr, _y_te = model_selection.train_test_split(df, y, random_state=0)
    StandardScaler().fit(X_tr)  # correct: fit on the train split only
    _patch.uninstall()
    assert _core.report() == []  # identical VALUES at distinct indices -> not leakage


def test_pandas_real_leak_still_caught():
    df, y = _df_with_duplicate_values()
    _patch.install()
    StandardScaler().fit(df)  # fit on the full frame before the split -> leak
    model_selection.train_test_split(df, y, random_state=0)
    _patch.uninstall()
    findings = _core.report()
    assert len(findings) == 1
    assert findings[0].estimator == "StandardScaler"


def test_pandas_clean_is_silent():
    df, y = _df_with_duplicate_values(seed=1)
    _patch.install()
    X_tr, _X_te, y_tr, _y_te = model_selection.train_test_split(df, y, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    pd.DataFrame(scaler.transform(X_tr), index=X_tr.index)  # train-only transform
    _patch.uninstall()
    assert _core.report() == []
