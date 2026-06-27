"""Why value-inspecting guards miss data leakage -- a live side-by-side.

Leakage is a data-FLOW bug (a fitted step saw held-out rows), not a data-VALUE bug. Tools
that inspect values -- floating-point error traps, NaN/Inf asserts, schema validators in the
pandera / Great Expectations style -- all pass the leaky pipeline silently, because the data
itself is perfectly valid. splitguard tracks the held-out rows through the run, so it is the
one approach that catches it.

Run:  python examples/comparison.py
"""

from __future__ import annotations

import numpy as np
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest, f_classif

import splitguard

rng = np.random.default_rng(0)
N, P, K = 160, 5000, 20
X = rng.normal(size=(N, P))
y = (rng.random(N) > 0.5).astype(int)


def _run_leaky_pipeline():
    """A feature selector fitted on the full matrix -- the held-out rows leak in."""
    model_selection.train_test_split(X, y, test_size=0.4, random_state=0)
    SelectKBest(f_classif, k=K).fit(X, y)


def detects_with_errstate() -> bool:
    try:
        with np.errstate(all="raise"):
            _run_leaky_pipeline()
        return False
    except FloatingPointError:
        return True


def detects_with_nan_assert() -> bool:
    _run_leaky_pipeline()
    return bool(np.isnan(X).any() or np.isinf(X).any())


def detects_with_schema_validation() -> bool:
    # pandera / Great Expectations validate dtype, nullability and value ranges.
    _run_leaky_pipeline()
    schema_ok = X.dtype.kind == "f" and np.isfinite(X).all() and -10 < X.min() and X.max() < 10
    return not schema_ok  # the schema is satisfied, so the leak is not detected


def detects_with_splitguard() -> bool:
    with splitguard.guard(policy="log"):
        _run_leaky_pipeline()
    return len(splitguard.report()) > 0


def main() -> None:
    checks = [
        ("numpy errstate(all='raise')", detects_with_errstate),
        ("manual assert no NaN/Inf", detects_with_nan_assert),
        ("schema validation (pandera/GE style)", detects_with_schema_validation),
        ("splitguard", detects_with_splitguard),
    ]
    print()
    print(f"{'Approach':<40} {'Detects leakage?':>16}")
    print("-" * 58)
    for name, fn in checks:
        print(f"{name:<40} {('YES' if fn() else 'NO'):>16}")
    print()
    print("Leakage is about data flow, not data values: only splitguard tracks the")
    print("held-out rows through fit() and reports the step that saw them.")


if __name__ == "__main__":
    main()
