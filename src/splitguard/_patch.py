"""Runtime hooks into scikit-learn: wrap every estimator ``fit`` and ``train_test_split``.

Patching is reversible and fully restored by :func:`uninstall`. Wrappers are
exception-contained and always return the wrapped callable's original result, so an
instrumented program behaves identically to an uninstrumented one.
"""

from __future__ import annotations

import contextlib
import functools
import sys
import weakref

from . import _core

# Row hashes of native data containers (DMatrix/Dataset), keyed weakly so the user's object is
# never mutated and the association is dropped when the container is garbage-collected.
_native_hashes: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()

# (class, had_own_fit, original_fit_callable) for every patched estimator class.
_patched_fits: list = []
# (class, had_own_split, original_split_callable) for every patched cross-validator class.
_patched_splits: list = []
# (owner_object, attribute_name, original_callable) for native trainers (xgboost/lightgbm).
_patched_native: list = []
_orig_train_test_split = None
_installed = False
_install_depth = 0  # reference count so nested guard() blocks restore only at the outermost

# Non-scikit-learn estimators that follow the sklearn ``.fit(X, y)`` API but are absent from
# ``all_estimators()``. Wrapped if (and when) their library is imported.
_EXTERNAL_ESTIMATORS = (
    ("xgboost", ("XGBModel", "XGBClassifier", "XGBRegressor", "XGBRanker")),
    ("lightgbm", ("LGBMModel", "LGBMClassifier", "LGBMRegressor", "LGBMRanker")),
    ("catboost", ("CatBoost", "CatBoostClassifier", "CatBoostRegressor")),
)


def _wrap_external_estimators() -> None:
    """Wrap ``.fit`` of XGBoost/LightGBM/CatBoost estimators that are currently imported."""
    with _core._state.lock:  # serialise the check-then-patch against concurrent hooks
        for mod_name, class_names in _EXTERNAL_ESTIMATORS:
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            for cls_name in class_names:
                try:
                    # accessing some libraries' attributes triggers lazy imports that may fail;
                    # this must never propagate into the user's program.
                    cls = getattr(mod, cls_name, None)
                    if cls is None:
                        continue
                    fit_fn = cls.__dict__.get("fit")
                    if fit_fn is None or getattr(fit_fn, "_splitguard", False):
                        continue
                    cls.fit = _make_fit_wrapper(fit_fn)
                    _patched_fits.append((cls, True, fit_fn))
                except Exception:
                    continue


# Native (non-``.fit``) training APIs: a data container constructed from arrays, then a
# module-level ``train(params, data, ...)``. (module, data-class, train-fn, data-arg, train-arg)
_NATIVE_TRAINERS = (
    ("xgboost", "DMatrix", "train", "data", "dtrain"),
    ("lightgbm", "Dataset", "train", "data", "train_set"),
)


def _make_data_init_wrapper(original, data_arg):
    @functools.wraps(original)
    def __init__(self, *args, **kwargs):
        original(self, *args, **kwargs)
        if _core._state.active:
            try:
                data = args[0] if args else kwargs.get(data_arg)
                hashes = _core._row_hashes(data)
                if hashes:
                    _native_hashes[self] = frozenset(hashes)  # weak: never mutate the container
            except Exception:
                pass

    __init__._splitguard = True
    return __init__


def _make_native_train_wrapper(original, label, train_arg):
    @functools.wraps(original)
    def train(*args, **kwargs):
        if _core._state.active:
            try:
                data = args[1] if len(args) > 1 else kwargs.get(train_arg)
                hashes = _native_hashes.get(data) if data is not None else None
                if hashes:
                    _core.on_fit_with_hashes(label, hashes, _core.user_call_site())
            except Exception:
                pass
        return original(*args, **kwargs)

    train._splitguard = True
    return train


def _wrap_native_trainers() -> None:
    """Hook ``DMatrix``/``Dataset`` construction and ``xgboost.train``/``lightgbm.train``."""
    with _core._state.lock:  # serialise the check-then-patch against concurrent hooks
        for mod_name, data_cls, train_fn, data_arg, train_arg in _NATIVE_TRAINERS:
            mod = sys.modules.get(mod_name)
            if mod is None:
                continue
            # Independent try blocks: a failing lazy import on the data class must not block the
            # train-function hook (and vice versa).
            try:
                cls = getattr(mod, data_cls, None)
                init = cls.__dict__.get("__init__") if cls is not None else None
                if init is not None and not getattr(init, "_splitguard", False):
                    cls.__init__ = _make_data_init_wrapper(init, data_arg)  # type: ignore[misc]
                    _patched_native.append((cls, "__init__", init))
            except Exception:
                pass
            try:
                fn = getattr(mod, train_fn, None)
                if fn is not None and not getattr(fn, "_splitguard", False):
                    label = f"{mod_name}.{train_fn}"
                    setattr(mod, train_fn, _make_native_train_wrapper(fn, label, train_arg))
                    _patched_native.append((mod, train_fn, fn))
            except Exception:
                pass


def _make_fit_wrapper(original):
    @functools.wraps(original)
    def fit(self, *args, **kwargs):
        if _core._state.active:
            try:
                X = args[0] if args else kwargs.get("X")
                if X is not None:
                    _core.on_fit(
                        type(self).__name__,
                        X,
                        _core.user_call_site(),
                        is_transformer=hasattr(type(self), "transform"),
                    )
            except Exception:
                pass  # instrumentation must never break a user fit
        return original(self, *args, **kwargs)

    fit._splitguard = True
    return fit


def _make_split_wrapper(original):
    @functools.wraps(original)
    def train_test_split(*args, **kwargs):
        result = original(*args, **kwargs)
        if _core._state.active:
            try:
                _wrap_external_estimators()  # the model library is imported by now
                _wrap_native_trainers()
                # train_test_split returns 2*n arrays; train at even, test at odd indices.
                site = _core.user_call_site()
                _core.on_split(list(result[1::2]), site)
                _core.check_group_split(list(result[0::2]), list(result[1::2]), site)
            except Exception:
                pass
        return result

    train_test_split._splitguard = True
    return train_test_split


def _make_cv_split_wrapper(original):
    """Wrap a cross-validator ``.split`` so each fold's held-out rows are the active taint.

    Taint is REPLACED per fold (never accumulated across folds), so a correct per-fold
    ``fit(X[train])`` stays silent while a ``fit(X)`` inside the loop, or a preprocessing fit
    before the loop, is flagged.
    """

    @functools.wraps(original)
    def split(self, X, y=None, groups=None):
        gen = original(self, X, y, groups)
        if not _core._state.active:
            yield from gen
            return
        _wrap_external_estimators()
        _wrap_native_trainers()
        site = _core.user_call_site()
        st = _core._state
        with st.lock:
            base_taint = set(st.taint)
            base_fits = len(st.fits)
        try:
            for tr_idx, te_idx in gen:
                fold_hashes = None
                try:
                    fold_hashes = _core._row_hashes(_core.index_rows(X, te_idx))
                except Exception:
                    fold_hashes = None
                with st.lock:
                    del st.fits[base_fits:]  # discard the previous fold's fit records
                    new_taint = set(base_taint)
                    if fold_hashes:
                        new_taint.update(fold_hashes)
                    _core.set_fold_taint(new_taint)
                try:
                    _core.check_group_split(
                        [_core.index_rows(X, tr_idx)], [_core.index_rows(X, te_idx)], site
                    )
                except Exception:
                    pass
                yield tr_idx, te_idx
        finally:
            with st.lock:
                del st.fits[base_fits:]
                st.taint = base_taint

    split._splitguard = True
    return split


def install() -> None:
    """Patch all loaded scikit-learn estimators' ``fit`` and ``train_test_split``."""
    global _orig_train_test_split, _installed, _install_depth
    _install_depth += 1
    if _installed:
        _core._state.active = True
        return
    try:
        from sklearn import model_selection
        from sklearn.utils import all_estimators

        seen = set()
        for _name, cls in all_estimators():
            if cls in seen or not hasattr(cls, "fit"):
                continue
            original = cls.fit
            if getattr(original, "_splitguard", False):
                continue
            had_own = "fit" in cls.__dict__
            try:
                cls.fit = _make_fit_wrapper(original)
            except (TypeError, AttributeError):
                continue  # immutable / extension class
            _patched_fits.append((cls, had_own, original))
            seen.add(cls)

        _orig_train_test_split = model_selection.train_test_split
        model_selection.train_test_split = _make_split_wrapper(_orig_train_test_split)

        _wrap_external_estimators()
        _wrap_native_trainers()
        _wrap_cross_validators(model_selection)
    except Exception:
        _install_depth = max(0, _install_depth - 1)  # do not leak the reference count
        raise

    _installed = True
    _core._state.active = True


def _wrap_cross_validators(model_selection) -> None:
    import inspect

    bases = tuple(
        b
        for b in (
            getattr(model_selection, "BaseCrossValidator", None),
            getattr(model_selection, "BaseShuffleSplit", None),
        )
        if b is not None
    )
    if not bases:
        return
    for _name, cls in inspect.getmembers(model_selection, inspect.isclass):
        if not issubclass(cls, bases):
            continue
        original = cls.split
        if getattr(original, "_splitguard", False):
            continue
        had_own = "split" in cls.__dict__
        try:
            cls.split = _make_cv_split_wrapper(original)
        except (TypeError, AttributeError):
            continue
        _patched_splits.append((cls, had_own, original))


def uninstall() -> None:
    """Restore every patched ``fit`` and ``train_test_split`` to its original.

    Reference-counted: with nested ``guard()`` blocks, only the outermost actually restores.
    """
    global _orig_train_test_split, _installed, _install_depth
    _install_depth = max(0, _install_depth - 1)
    if _install_depth > 0:
        return  # still inside an outer guard; keep instrumentation in place

    for cls, had_own, original in _patched_fits:
        try:
            if had_own:
                cls.fit = original
            elif "fit" in cls.__dict__ and getattr(cls.__dict__["fit"], "_splitguard", False):
                delattr(cls, "fit")  # reveal the inherited fit again
        except (TypeError, AttributeError):
            pass
    _patched_fits.clear()

    for cls, had_own, original in _patched_splits:
        try:
            if had_own:
                cls.split = original
            elif "split" in cls.__dict__ and getattr(cls.__dict__["split"], "_splitguard", False):
                delattr(cls, "split")
        except (TypeError, AttributeError):
            pass
    _patched_splits.clear()

    for owner, attr, original in _patched_native:
        try:
            setattr(owner, attr, original)
        except (TypeError, AttributeError):
            pass
    _patched_native.clear()

    if _orig_train_test_split is not None:
        ms = sys.modules.get("sklearn.model_selection")  # avoid importing at interpreter exit
        if ms is not None:
            ms.train_test_split = _orig_train_test_split  # type: ignore[attr-defined]
        _orig_train_test_split = None

    _installed = False
    _core._state.active = False


@contextlib.contextmanager
def guard(
    policy: str | None = None,
    min_leaked_rows: int | None = None,
    reset: bool = True,
    strict_transforms: bool = False,
    groups: tuple | None = None,
):
    """Install hooks for the duration of the block, then report any leakage on exit.

    Parameters
    ----------
    policy:
        ``"raise"`` (default in scripts/pytest), ``"warn"`` (notebooks) or ``"log"``
        (non-TTY). When ``None`` the policy is auto-detected from the runtime context.
    min_leaked_rows:
        Minimum number of held-out rows a fit must touch before it is reported.
    reset:
        Clear any previously accumulated state on entry.
    strict_transforms:
        Opt-in: warn when a transformer is fitted before the split (catches the
        ``fit_transform`` then split-on-transformed-data pattern that row identity cannot see).
    groups:
        Optional ``(X, group_ids)`` to register for group-leakage detection. Pass it here so it
        survives the entry ``reset()`` — equivalent to calling ``mark_groups(X, group_ids)`` as
        the first statement inside the block.
    """
    if reset:
        _core._state.reset()
    _core.configure(
        policy=policy, min_leaked_rows=min_leaked_rows, strict_transforms=strict_transforms
    )
    install()
    if groups is not None:
        _core.mark_groups(groups[0], groups[1])
    try:
        yield _core._state
    finally:
        uninstall()
        _core.finalize_and_report()


def across_splits(run, seeds, policy: str = "log") -> dict:
    """Run ``run(seed)`` under a fresh guard for each seed; return ``{seed: findings}``.

    Detects overlap leakage independently on every (dynamic) split. This reports leakage
    per split only; it does NOT measure score stability and does NOT detect multi-test
    leakage (test-set reuse for model selection), which is a decision-flow problem outside
    a row-identity engine.
    """
    results = {}
    for seed in seeds:
        with guard(policy=policy):
            run(seed)
            results[seed] = _core.report()
    return results
