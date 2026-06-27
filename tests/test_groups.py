"""Group leakage: the same entity (patient/user/store) appearing in train and test."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn import model_selection
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, KFold

import splitguard
from splitguard import _core, _patch


@pytest.fixture(autouse=True)
def _clean_state():
    _core._state.reset()
    _core._state.policy_override = None
    yield
    if _patch._installed:
        _patch.uninstall()
    _core._state.reset()


def _grouped(n_groups=30, per=4, seed=0):
    rng = np.random.default_rng(seed)
    n = n_groups * per
    X = rng.normal(size=(n, 5))
    y = (rng.random(n) > 0.5).astype(int)
    groups = np.repeat(np.arange(n_groups), per)
    return X, y, groups


def _has_group_leak():
    return any(f.pattern == "group_leakage" for f in _core.report())


def test_random_split_group_leakage_flagged():
    X, y, groups = _grouped()
    with _patch.guard(policy="log"):
        splitguard.mark_groups(X, groups)
        model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    assert _has_group_leak()


def test_group_shuffle_split_no_leakage():
    X, _y, groups = _grouped(seed=1)
    with _patch.guard(policy="log"):
        splitguard.mark_groups(X, groups)
        for _tr, _te in GroupShuffleSplit(n_splits=2, test_size=0.3, random_state=0).split(
            X, groups=groups
        ):
            pass
    assert not _has_group_leak()


def test_kfold_group_leakage_flagged():
    X, _y, groups = _grouped(seed=2)
    with _patch.guard(policy="log"):
        splitguard.mark_groups(X, groups)
        for _tr, _te in KFold(n_splits=4, shuffle=True, random_state=0).split(X):
            pass
    assert _has_group_leak()


def test_groupkfold_no_group_leakage():
    X, _y, groups = _grouped(seed=3)
    with _patch.guard(policy="log"):
        splitguard.mark_groups(X, groups)
        for _tr, _te in GroupKFold(n_splits=4).split(X, groups=groups):
            pass
    assert not _has_group_leak()


def test_guard_groups_param_catches_group_leak():
    X, y, groups = _grouped(seed=5)
    with _patch.guard(policy="log", groups=(X, groups)):  # registered, survives reset
        model_selection.train_test_split(X, y, random_state=0)
    assert _has_group_leak()


def test_no_groups_marked_is_silent():
    X, y, _groups = _grouped(seed=4)
    with _patch.guard(policy="log"):
        model_selection.train_test_split(X, y, test_size=0.3, random_state=0)
    assert not _has_group_leak()  # no mark_groups -> no group check
