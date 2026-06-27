"""Core leakage-detection engine: test-set taint tracking by row identity.

The engine observes a real execution. It records a content hash of every held-out
("test") feature row, and a content hash of the rows passed to every estimator ``fit``.
A leak is any non-empty intersection between the rows a fitted estimator has seen and
the held-out rows. Both temporal orders are handled:

* fit *after* the split, on data overlapping the test set -> detected at fit time;
* fit *before* the split (e.g. ``scaler.fit(X)`` then ``train_test_split``) -> detected
  retroactively when the split taints rows the earlier fit already consumed.

The engine never mutates user data, never alters estimator return values, and never
raises out of its hooks (all hook bodies are exception-contained). It is a
coverage-bounded detector: it reports leakage that actually occurs in the executed run;
it cannot prove a pipeline is leak-free on unexercised paths.
"""

from __future__ import annotations

import hashlib
import logging
import os
import sys
import threading
import warnings
from dataclasses import dataclass, field

_PKG_DIR = os.path.normcase(os.path.dirname(os.path.abspath(__file__)))  # normcase for Windows
_HASH_DIGEST_SIZE = 16  # 16-byte BLAKE2b; birthday-collision prob < 1e-17 for < 1e6 rows
_sparse_warned = False  # one-shot guard for the scipy.sparse "not tracked" warning
_warned_mixed_identity = False  # one-shot: value-based fit checked against index-based taint
_warned_value_dupes = False  # one-shot: value identity made ambiguous by duplicate rows
_warned_pre_split_transform = False  # one-shot (strict mode): transformer fitted before the split
_warned_no_split = False  # one-shot: fits happened but no split was tracked (import-order trap)


class LeakageError(RuntimeError):
    """Raised when data leakage is detected and the active policy is ``"raise"``."""


@dataclass
class Finding:
    """A single detected leak: a fitted estimator that has seen held-out rows."""

    estimator: str
    leaked_rows: int
    fit_rows: int
    pattern: str  # "fit_after_split" or "fit_before_split"
    call_site: str

    @property
    def fraction(self) -> float:
        return self.leaked_rows / self.fit_rows if self.fit_rows else 0.0


@dataclass
class _FitRecord:
    estimator: str
    row_hashes: frozenset[bytes]
    n_rows: int
    call_site: str
    flagged: bool = False
    is_transformer: bool = False  # has a transform() method
    pre_split: bool = False  # fitted while no split had happened yet


@dataclass
class _State:
    """Process-wide detector state, guarded by a re-entrant lock."""

    lock: threading.RLock = field(default_factory=threading.RLock)
    taint: set = field(default_factory=set)
    fits: list = field(default_factory=list)
    findings: list = field(default_factory=list)
    row_group: dict = field(default_factory=dict)  # row hash -> group id (for group leakage)
    taint_kind: str | None = None  # "index" (pandas) or "value" (numpy) — identity reliability
    active: bool = False
    min_leaked_rows: int = 1
    strict_transforms: bool = False  # opt-in: warn on a transformer fitted before the split
    policy_override: str | None = None  # "raise" | "warn" | "log" | None (auto)

    def reset(self) -> None:
        with self.lock:
            self.taint.clear()
            self.fits.clear()
            self.findings.clear()
            self.row_group.clear()
            self.taint_kind = None
            self.strict_transforms = False
            self.policy_override = None  # must not bleed across guard() calls
            self.min_leaked_rows = 1


_state = _State()


# --------------------------------------------------------------------------- #
# Row identity
# --------------------------------------------------------------------------- #
def _warn_if_sparse(X) -> None:
    """Warn once if a scipy sparse matrix is seen: sparse inputs are not tracked."""
    global _sparse_warned
    if _sparse_warned:
        return
    sp = sys.modules.get("scipy.sparse")
    if sp is not None and sp.issparse(X):
        _sparse_warned = True
        warnings.warn(
            "splitguard: sparse matrix passed to fit(); sparse inputs are not tracked, "
            "so leakage on sparse data will not be reported.",
            stacklevel=4,
        )


def _identity_kind(X) -> str | None:
    """``"index"`` for a pandas DataFrame, ``"value"`` for a numpy/array feature matrix."""
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(X, pd.DataFrame):
        return "index"
    if _as_feature_matrix(X) is not None:
        return "value"
    return None


def _warn_mixed_identity() -> None:
    global _warned_mixed_identity
    if not _warned_mixed_identity:
        _warned_mixed_identity = True
        warnings.warn(
            "splitguard: a NumPy fit is being checked against pandas index-based held-out rows; "
            "the identities cannot match and a leak may be missed. Keep the pipeline consistently "
            "pandas (or use a Pipeline).",
            stacklevel=2,
        )


def _warn_value_dupes() -> None:
    global _warned_value_dupes
    if not _warned_value_dupes:
        _warned_value_dupes = True
        warnings.warn(
            "splitguard: held-out NumPy rows contain duplicate values; without an index, row "
            "identity is value-based and may report false positives. Pass a pandas DataFrame for "
            "index-based identity, or raise configure(min_leaked_rows=...).",
            stacklevel=2,
        )


def _warn_pre_split_transform(estimator: str, call_site: str) -> None:
    global _warned_pre_split_transform
    if not _warned_pre_split_transform:
        _warned_pre_split_transform = True
        warnings.warn(
            f"splitguard (strict): transformer {estimator} ({call_site}) was fitted before the "
            "split. If it was fitted on the full data this is preprocessing leakage that the "
            "row-identity check cannot confirm once the split is on transformed data. Fit inside "
            "a Pipeline or on the training split only.",
            stacklevel=2,
        )


def _warn_no_split_tracked() -> None:
    global _warned_no_split
    if not _warned_no_split:
        _warned_no_split = True
        warnings.warn(
            "splitguard: estimators were fitted but no held-out split was tracked this run, so "
            "NO leakage could be detected. This usually means train_test_split was imported "
            "before splitguard (a stale binding bypasses the hook). Put `import splitguard.auto` "
            "first, call `train_test_split` via the module, or use `splitguard.mark_test(...)`.",
            stacklevel=2,
        )


def _as_feature_matrix(X):
    """Return a contiguous 2-D numeric array of feature rows, or ``None`` to skip.

    Skipped: 1-D inputs (targets ``y``), object/mixed dtype (hashing would compare object
    identity, not content), sparse matrices, and non array-like inputs (e.g. text tokens).
    """
    import numpy as np

    arr = None
    pd = sys.modules.get("pandas")  # read pandas only if the caller already uses it
    if pd is not None:
        if isinstance(X, pd.DataFrame):
            arr = X.to_numpy()
        elif isinstance(X, pd.Series):
            return None  # 1-D target-like
    if arr is None:
        if isinstance(X, np.ndarray):
            if X.ndim == 1:
                return None  # target-like
            arr = X
        elif isinstance(X, (list, tuple)):
            try:
                arr = np.asarray(X)
            except (ValueError, TypeError):
                return None
        else:
            _warn_if_sparse(X)
            return None

    if arr.ndim != 2 or arr.dtype.kind == "O":
        return None  # object/mixed dtype hashes identity, not content
    return np.ascontiguousarray(arr)


def _row_hashes(X):
    """Per-row identity hashes, or ``None`` if *X* is not a feature matrix.

    For pandas the identity is the row's INDEX label, which is a sample's true identity and is
    preserved by ``train_test_split``. Two legitimately identical rows (same values, different
    index) are therefore NOT confused as leakage; only the same sample (same index in both train
    and the held-out set) is flagged. NumPy arrays carry no index, so values are hashed and
    genuine duplicate rows can register as overlap (an information-theoretic limit without ids).
    """
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(X, pd.DataFrame):
        return [
            hashlib.blake2b(b"idx:" + repr(ix).encode(), digest_size=_HASH_DIGEST_SIZE).digest()
            for ix in X.index
        ]
    if pd is not None and isinstance(X, pd.Series):
        return None  # 1-D target-like
    arr = _as_feature_matrix(X)
    if arr is None:
        return None
    salt = f"{arr.dtype.str}|{arr.shape[1]}".encode()
    return [
        hashlib.blake2b(salt + arr[i].tobytes(), digest_size=_HASH_DIGEST_SIZE).digest()
        for i in range(arr.shape[0])
    ]


def user_call_site() -> str:
    """Nearest user-code frame (outside splitguard, sklearn, site-packages, frozen frames).

    Walks ``f_back`` directly via ``sys._getframe`` rather than ``traceback.walk_stack``,
    which on CPython 3.13 assumes a fixed stack depth and fails for shallow (module-level /
    notebook-cell) call sites.
    """
    frame = sys._getframe().f_back  # caller of user_call_site
    while frame is not None:
        fn = frame.f_code.co_filename
        norm = os.path.normcase(fn)  # case-insensitive match on Windows
        skip = (
            fn.startswith("<")  # frozen frames (<frozen runpy>, <string>)
            or norm.startswith(_PKG_DIR)
            or "site-packages" in norm
            or "dist-packages" in norm
            or f"{os.sep}sklearn{os.sep}" in norm
        )
        if not skip:
            return f"{os.path.basename(fn)}:{frame.f_lineno} in {frame.f_code.co_name}()"
        frame = frame.f_back
    return "<unknown>"


# --------------------------------------------------------------------------- #
# Event hooks (called from the patched fit / train_test_split)
# --------------------------------------------------------------------------- #
def on_fit(estimator: str, X, call_site: str, is_transformer: bool = False) -> None:
    """Record a fit and flag it if it has already seen any tainted (held-out) row."""
    hashes = _row_hashes(X)
    if hashes is None:
        return
    if _state.taint and _state.taint_kind == "index" and _identity_kind(X) == "value":
        _warn_mixed_identity()  # value fit vs index taint -> identities cannot match
    on_fit_with_hashes(estimator, hashes, call_site, is_transformer=is_transformer)


def on_fit_with_hashes(
    estimator: str, hashes, call_site: str, is_transformer: bool = False
) -> None:
    """Record a fit from precomputed row hashes (used by native trainers, e.g. xgboost.train)."""
    hset = frozenset(hashes)
    n_rows = len(hashes)
    with _state.lock:
        record = _FitRecord(
            estimator,
            hset,
            n_rows,
            call_site,
            is_transformer=is_transformer,
            pre_split=not _state.taint,
        )
        _state.fits.append(record)
        overlap = hset & _state.taint
        if len(overlap) >= _state.min_leaked_rows:
            record.flagged = True
            _state.findings.append(
                Finding(estimator, len(overlap), n_rows, "fit_after_split", call_site)
            )


def on_split(test_arrays, call_site: str) -> None:
    """Taint the held-out feature rows and retroactively flag earlier fits."""
    new_taint: set = set()
    kind = None
    for arr in test_arrays:
        hashes = _row_hashes(arr)
        if not hashes:
            continue
        new_taint.update(hashes)
        kind = _identity_kind(arr) or kind
        if _identity_kind(arr) == "value" and len(set(hashes)) < len(hashes):
            _warn_value_dupes()  # duplicate rows make value identity ambiguous
    if not new_taint:
        return
    with _state.lock:
        _state.taint.update(new_taint)
        if kind:
            _state.taint_kind = kind
        for record in _state.fits:
            if record.flagged:
                continue
            overlap = record.row_hashes & new_taint
            if len(overlap) >= _state.min_leaked_rows:
                record.flagged = True
                _state.findings.append(
                    Finding(
                        record.estimator,
                        len(overlap),
                        record.n_rows,
                        "fit_before_split",
                        record.call_site,
                    )
                )
        if _state.strict_transforms:
            for record in _state.fits:
                if record.is_transformer and record.pre_split and not record.flagged:
                    _warn_pre_split_transform(record.estimator, record.call_site)
                    break


def mark_test(*arrays) -> None:
    """Explicitly taint held-out arrays (use when not calling ``train_test_split``)."""
    on_split(list(arrays), user_call_site())


def index_rows(X, idx):
    """Return ``X[idx]`` for ndarray/list or ``X.iloc[idx]`` for pandas (used by CV folds)."""
    pd = sys.modules.get("pandas")
    if pd is not None and isinstance(X, (pd.DataFrame, pd.Series)):
        return X.iloc[idx]
    return X[idx]


def set_fold_taint(new_taint: set) -> None:
    """Replace the active taint with a fold's held-out rows and re-check earlier fits.

    Used by cross-validator hooks: each fold's held-out set replaces the previous one
    (no cross-fold accumulation, which would false-positive on correct per-fold fits),
    while fits made before the loop are still flagged retroactively.
    """
    with _state.lock:
        _state.taint = new_taint
        for record in _state.fits:
            if record.flagged:
                continue
            overlap = record.row_hashes & new_taint
            if len(overlap) >= _state.min_leaked_rows:
                record.flagged = True
                _state.findings.append(
                    Finding(
                        record.estimator,
                        len(overlap),
                        record.n_rows,
                        "fit_before_split",
                        record.call_site,
                    )
                )


# --------------------------------------------------------------------------- #
# Configuration and reporting
# --------------------------------------------------------------------------- #
def mark_groups(X, groups) -> None:
    """Register the group id of each row of feature matrix *X* (for group-leakage checks).

    A "group" is an entity that must not span train and test (patient, user, store, ...).
    After registering, a split that puts the same group in both train and test is flagged.
    """
    hashes = _row_hashes(X)
    if hashes is None:
        return
    group_list = list(groups)
    if len(group_list) != len(hashes):
        return  # length mismatch -> skip safely
    with _state.lock:
        for h, g in zip(hashes, group_list, strict=True):
            _state.row_group[h] = g


def _groups_of(arrays) -> set:
    found: set = set()
    for arr in arrays:
        hashes = _row_hashes(arr)
        if not hashes:
            continue
        for h in hashes:
            g = _state.row_group.get(h)
            if g is not None:
                found.add(g)
    return found


def check_group_split(train_arrays, test_arrays, call_site: str) -> None:
    """Flag group leakage: a group present in both the train and the held-out arrays."""
    with _state.lock:
        if not _state.row_group:
            return
        train_groups = _groups_of(train_arrays)
        test_groups = _groups_of(test_arrays)
        shared = train_groups & test_groups
        if shared:
            _state.findings.append(
                Finding(
                    "<split>",
                    len(shared),
                    len(test_groups) or len(shared),
                    "group_leakage",
                    call_site,
                )
            )


def configure(
    policy: str | None = None,
    min_leaked_rows: int | None = None,
    strict_transforms: bool | None = None,
) -> None:
    """Override the violation policy, the minimum leaked-row count, or strict-transforms."""
    if policy is not None:
        if policy not in ("raise", "warn", "log"):
            raise ValueError("policy must be 'raise', 'warn' or 'log'")
        _state.policy_override = policy
    if min_leaked_rows is not None:
        if min_leaked_rows < 1:
            raise ValueError("min_leaked_rows must be >= 1")
        _state.min_leaked_rows = min_leaked_rows
    if strict_transforms is not None:
        _state.strict_transforms = strict_transforms


def report() -> list:
    """Return the findings so far, de-duplicated by (estimator, call site, pattern).

    A leaking fit inside a cross-validation loop fires once per fold; those collapse to a
    single finding (keeping the largest leaked-row count).
    """
    with _state.lock:
        best: dict = {}
        for f in _state.findings:
            key = (f.estimator, f.call_site, f.pattern)
            if key not in best or f.leaked_rows > best[key].leaked_rows:
                best[key] = f
        return list(best.values())


def _resolve_policy() -> str:
    if _state.policy_override is not None:
        return _state.policy_override
    if os.environ.get("PYTEST_CURRENT_TEST"):  # set by pytest during each test
        return "raise"
    if "ipykernel" in sys.modules:  # Jupyter kernel: warn, do not interrupt the cell
        return "warn"
    if not getattr(sys.stdout, "isatty", lambda: False)():  # CI / pipe: structured log
        return "log"
    return "raise"  # interactive terminal default


def _summary(findings: list) -> str:
    n = len(findings)
    rows = sum(f.leaked_rows for f in findings)
    return f"splitguard: {n} leak(s) detected; {rows} held-out row(s) reached fitted estimators"


def finalize_and_report() -> list:
    """Render findings and apply the active policy (raise / warn / log)."""
    findings = report()
    if not findings:
        with _state.lock:
            # fits happened but nothing was held out -> the split was never intercepted
            if _state.fits and not _state.taint:
                _warn_no_split_tracked()
        return findings

    from . import _report

    policy = _resolve_policy()
    summary = _summary(findings)
    if policy == "log":
        logging.getLogger("splitguard").warning("%s\n%s", summary, _report.render(findings))
        return findings

    _report.emit(findings)  # rich panel if available, else structured plain text
    if policy == "raise":
        raise LeakageError(summary)
    warnings.warn(summary, stacklevel=2)
    return findings
