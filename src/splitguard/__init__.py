"""splitguard — catch machine-learning train-test data leakage at runtime, zero config.

Quickstart
----------
>>> import splitguard
>>> with splitguard.guard():
...     X_tr, X_te, y_tr, y_te = train_test_split(X, y)
...     scaler.fit(X)            # leak: scaler saw the held-out rows -> reported on exit

splitguard observes a real run and reports any fitted estimator that has seen held-out
rows. It is a coverage-bounded detector: it flags leakage that actually occurs in the
executed run; it does not prove a pipeline is leak-free on unexercised paths.
"""

from __future__ import annotations

from ._core import Finding, LeakageError, configure, mark_groups, mark_test, report
from ._patch import across_splits, guard, install, uninstall

__version__ = "0.1.0"

__all__ = [
    "guard",
    "install",
    "uninstall",
    "across_splits",
    "mark_test",
    "mark_groups",
    "configure",
    "report",
    "Finding",
    "LeakageError",
    "__version__",
]
