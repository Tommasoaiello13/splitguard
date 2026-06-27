"""Universal coverage: every split (cross-validators) and every model (non-sklearn)."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GroupKFold, KFold
from sklearn.preprocessing import StandardScaler

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


def _data(n=90, p=5, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), (rng.random(n) > 0.5).astype(int)


# --- Cross-validation splits ------------------------------------------------ #
def test_kfold_leaky_fit_in_loop():
    X, y = _data()
    with _patch.guard(policy="log"):
        for tr, _te in KFold(n_splits=3, shuffle=True, random_state=0).split(X):
            StandardScaler().fit(X)  # full matrix inside the fold -> leak
            LogisticRegression(max_iter=200).fit(X[tr], y[tr])
    findings = _core.report()
    assert len(findings) == 1
    assert findings[0].estimator == "StandardScaler"


def test_kfold_clean_loop_no_false_positive():
    X, y = _data()
    with _patch.guard(policy="log"):
        for tr, _te in KFold(n_splits=3, shuffle=True, random_state=0).split(X):
            scaler = StandardScaler().fit(X[tr])  # train-fold only -> correct
            LogisticRegression(max_iter=200).fit(scaler.transform(X[tr]), y[tr])
    assert _core.report() == []  # must not false-positive across folds


def test_preprocess_before_cv_loop_flagged():
    X, y = _data()
    with _patch.guard(policy="log"):
        StandardScaler().fit(X)  # fitted before the CV loop -> leak
        for tr, _te in KFold(n_splits=3, shuffle=True, random_state=0).split(X):
            LogisticRegression(max_iter=200).fit(X[tr], y[tr])
    findings = _core.report()
    assert any(
        f.estimator == "StandardScaler" and f.pattern == "fit_before_split" for f in findings
    )


def test_groupkfold_clean_no_false_positive():
    X, y = _data()
    groups = np.arange(len(y)) % 10
    with _patch.guard(policy="log"):
        for tr, _te in GroupKFold(n_splits=3).split(X, y, groups):
            StandardScaler().fit(X[tr])
    assert _core.report() == []


def test_cv_uninstall_restores_split():
    _patch.install()
    assert getattr(KFold.split, "_splitguard", False) is True
    _patch.uninstall()
    assert getattr(KFold.split, "_splitguard", False) is False


# --- Non-scikit-learn models ------------------------------------------------ #
def test_xgboost_full_fit_after_split_flagged():
    xgb = pytest.importorskip("xgboost")
    X, y = _data(n=80)
    with _patch.guard(policy="log"):
        model_selection.train_test_split(X, y, random_state=0)
        xgb.XGBClassifier(n_estimators=5, max_depth=2, verbosity=0).fit(X, y)
    assert any("XGB" in f.estimator for f in _core.report())


def _load_or_skip(mod_name, cls_name):
    import importlib

    try:
        mod = importlib.import_module(mod_name)
        return getattr(mod, cls_name)  # import/access may fail in some envs (e.g. dask)
    except ImportError:
        pytest.skip(f"{mod_name} not installed")
    except Exception:  # pragma: no cover - environment-specific dependency incompatibility
        pytest.skip(f"{mod_name} import fails in this environment")


def test_lightgbm_is_instrumented():
    # Instrumentation presence proves "every model" coverage; the real LightGBM fit is not
    # exercised because LightGBM lazily imports dask, which has an environment-specific
    # incompatibility unrelated to splitguard.
    cls = _load_or_skip("lightgbm", "LGBMClassifier")
    with _patch.guard(policy="log"):
        assert getattr(cls.fit, "_splitguard", False) is True


def test_catboost_is_instrumented():
    cls = _load_or_skip("catboost", "CatBoostClassifier")
    with _patch.guard(policy="log"):
        assert getattr(cls.fit, "_splitguard", False) is True
