"""Ground-truth oracle tests for the leakage detector."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import splitguard
from splitguard import _core, _patch


@pytest.fixture(autouse=True)
def _clean_state():
    _core._state.reset()
    _core._state.policy_override = None
    _core._state.min_leaked_rows = 1
    yield
    if _patch._installed:
        _patch.uninstall()
    _core._state.reset()


def _data(n=60, p=4, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, p))
    y = (rng.random(n) > 0.5).astype(int)
    return X, y


def test_leak_fit_before_split():
    X, y = _data()
    _patch.install()
    StandardScaler().fit(X)  # fit before the split -> caught retroactively
    _, X_te, _, _ = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    _patch.uninstall()

    findings = _core.report()
    assert len(findings) == 1
    assert findings[0].pattern == "fit_before_split"
    assert findings[0].estimator == "StandardScaler"
    assert findings[0].leaked_rows == X_te.shape[0]


def test_leak_fit_after_split_on_full():
    X, y = _data()
    _patch.install()
    model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    LogisticRegression(max_iter=200).fit(X, y)  # fit on full matrix after the split
    _patch.uninstall()

    findings = _core.report()
    assert any(
        f.pattern == "fit_after_split" and f.estimator == "LogisticRegression" for f in findings
    )


def test_clean_pipeline_no_findings():
    X, y = _data()
    _patch.install()
    X_tr, _, y_tr, _ = model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    scaler = StandardScaler().fit(X_tr)
    LogisticRegression(max_iter=200).fit(scaler.transform(X_tr), y_tr)
    _patch.uninstall()

    assert _core.report() == []


def test_mark_test_explicit():
    X, _ = _data()
    _patch.install()
    splitguard.mark_test(X[:20])  # manual hold-out, no train_test_split
    StandardScaler().fit(X)  # full matrix overlaps the 20 held-out rows
    _patch.uninstall()

    findings = _core.report()
    assert len(findings) == 1
    assert findings[0].leaked_rows == 20


def test_no_mutation_and_return_identity():
    X, _ = _data()
    X_before = X.copy()
    _patch.install()
    scaler = StandardScaler()
    returned = scaler.fit(X)
    _patch.uninstall()

    assert returned is scaler
    assert np.array_equal(X, X_before)


def test_non_feature_inputs_are_ignored():
    _patch.install()
    assert _core._row_hashes(np.array([1, 2, 3])) is None  # 1-D target-like
    assert _core._row_hashes(["a", "b", "c"]) is None  # text tokens
    _patch.uninstall()
    assert _core.report() == []


def test_uninstall_restores_originals():
    orig_split = model_selection.train_test_split
    _patch.install()
    assert getattr(model_selection.train_test_split, "_splitguard", False) is True
    _patch.uninstall()
    assert model_selection.train_test_split is orig_split
    assert getattr(StandardScaler.fit, "_splitguard", False) is False


def test_object_dtype_is_not_tracked():
    _patch.install()
    x_obj = np.array([["a", 1], ["b", 2]], dtype=object)
    assert _core._row_hashes(x_obj) is None  # identity hashing avoided
    _patch.uninstall()


def test_nested_guard_keeps_outer_instrumentation():
    X, y = _data()
    with _patch.guard(policy="log", reset=True):
        with _patch.guard(policy="log", reset=False):
            pass  # inner exit must not tear down the outer guard
        assert _patch._installed is True
        assert _core._state.active is True
        StandardScaler().fit(X)
        model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
        assert len(_core.report()) == 1
    assert _patch._installed is False
    assert _patch._install_depth == 0


def test_three_way_split_taints_val_and_test():
    X, y = _data(n=120)
    _patch.install()
    # train / validation / test via two splits
    X_tmp, X_te, y_tmp, y_te = model_selection.train_test_split(X, y, test_size=0.2, random_state=0)
    X_tr, X_val, y_tr, y_val = model_selection.train_test_split(
        X_tmp, y_tmp, test_size=0.25, random_state=0
    )
    StandardScaler().fit(X)  # full-matrix fit overlaps BOTH val and test
    _patch.uninstall()

    findings = _core.report()
    assert len(findings) == 1
    assert findings[0].leaked_rows == X_te.shape[0] + X_val.shape[0]


def test_multiple_dynamic_splits_each_tracked():
    X, y = _data(n=100)
    leaks = []
    for seed in range(3):  # different splits ("dynamic")
        with _patch.guard(policy="log"):
            model_selection.train_test_split(X, y, test_size=0.3, random_state=seed)
            StandardScaler().fit(X)  # leaks on every split
            leaks.append(len(_core.report()))
    assert leaks == [1, 1, 1]


def test_min_leaked_rows_threshold():
    X, _ = _data()
    _patch.install()
    _core.configure(min_leaked_rows=5)
    splitguard.mark_test(X[:3])  # only 3 held-out rows
    StandardScaler().fit(X)
    _patch.uninstall()
    assert _core.report() == []  # below the threshold of 5
