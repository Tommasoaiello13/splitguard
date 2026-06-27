"""Validate splitguard against the published Yang et al. (ASE'22) leakage benchmark.

Benchmark: github.com/malusamayo/leakage-analysis (MIT), the tool of "Data Leakage in
Notebooks: Static Detection and Better Processes". Its ``tests/inputs/*.py`` are labelled
with the ground-truth ``Model.fit`` calls that leak (see its ``tests/test_preprocessing.py``
and ``tests/test_overlap.py``).

This harness runs the *runnable* benchmark cases under ``splitguard.guard`` on synthesized
compatible data, plus deterministic overlap cases, and prints a per-category confusion matrix.
It is deliberately honest: splitguard detects row-identity OVERLAP, so it is expected to miss
statistics-based preprocessing leaks (np.mean(df), preprocessing.scale(df)) that put no test
row into a fit.

Run:  python validation/run_benchmark.py  [path-to-leakage-analysis-repo]
"""

from __future__ import annotations

import contextlib
import io
import os
import sys
import tempfile

import numpy as np
import pandas as pd

import splitguard
from splitguard import _core

# Ground-truth: True = the file contains leakage (label set is non-empty in the benchmark).
BENCHMARK_PREPROCESSING = {
    "test0.py": True,  # df.fillna(np.mean(df)) on full data before split  -> statistics leak
    "test1.py": True,  # per-split fillna (labelled leaky by the benchmark)
    "test2.py": True,  # X_train.fillna(np.mean(df)) -> statistics leak
    "test3.py": False,  # scale() result is unused; trains on raw df -> clean
}


def _make_titanic_csv(path: str, n: int = 200, seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    fare = rng.gamma(2.0, 15.0, size=n)
    fare[rng.random(n) < 0.1] = np.nan  # some missing Fare (the cases impute it)
    df = pd.DataFrame(
        {
            "Survived": (rng.random(n) > 0.5).astype(int),
            "Pclass": rng.integers(1, 4, size=n),
            "Age": rng.uniform(1, 80, size=n).round(1),
            "SibSp": rng.integers(0, 4, size=n),
            "Parch": rng.integers(0, 3, size=n),
            "Fare": fare,
        }
    )
    df.to_csv(path, index=False)


def _run_source(code: str, workdir: str) -> tuple[list, str]:
    """Exec *code* under splitguard in *workdir*; return (findings, error-or-empty)."""
    cwd = os.getcwd()
    os.chdir(workdir)
    _core._state.reset()
    splitguard.install()
    err = ""
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            exec(compile(code, "<case>", "exec"), {"__name__": "__case__"})
    except Exception as exc:  # benchmark files may need data we did not synthesize
        err = f"{type(exc).__name__}: {exc}"
    finally:
        splitguard.uninstall()
        os.chdir(cwd)
    return splitguard.report(), err


def _run_benchmark_cases(repo: str, workdir: str) -> list[tuple]:
    rows = []
    inputs = os.path.join(repo, "tests", "inputs")
    for fname, is_leaky in BENCHMARK_PREPROCESSING.items():
        fpath = os.path.join(inputs, fname)
        if not os.path.exists(fpath):
            rows.append((fname, is_leaky, None, "missing"))
            continue
        with open(fpath, encoding="utf-8") as fh:
            findings, err = _run_source(fh.read(), workdir)
        detected = len(findings) > 0 if not err else None
        rows.append((fname, is_leaky, detected, err))
    return rows


# Deterministic overlap cases -- splitguard's designed sweet spot (no benchmark data needed).
_OVERLAP_CASES = {
    "scaler.fit(full) before split": (
        True,
        """
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(0)
X = rng.normal(size=(120, 6)); y = (rng.random(120) > 0.5).astype(int)
scaler = StandardScaler().fit(X)
X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, random_state=0)
LogisticRegression(max_iter=500).fit(scaler.transform(X_tr), y_tr)
""",
    ),
    "selector.fit(full, y)": (
        True,
        """
import numpy as np
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest, f_classif
rng = np.random.default_rng(1)
X = rng.normal(size=(120, 50)); y = (rng.random(120) > 0.5).astype(int)
X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, random_state=0)
SelectKBest(f_classif, k=10).fit(X, y)
""",
    ),
    "oversample (row duplication) before split": (
        True,
        """
import numpy as np
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(2)
X = rng.normal(size=(80, 6)); y = (rng.random(80) > 0.5).astype(int)
X = np.vstack([X, X[:40]]); y = np.concatenate([y, y[:40]])  # duplicate rows
X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, random_state=0)
LogisticRegression(max_iter=500).fit(X_tr, y_tr)
""",
    ),
    "correct: scaler.fit(train) only": (
        False,
        """
import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
rng = np.random.default_rng(3)
X = rng.normal(size=(120, 6)); y = (rng.random(120) > 0.5).astype(int)
X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, random_state=0)
scaler = StandardScaler().fit(X_tr)
LogisticRegression(max_iter=500).fit(scaler.transform(X_tr), y_tr)
""",
    ),
}


def _run_overlap_cases(workdir: str) -> list[tuple]:
    rows = []
    for name, (is_leaky, code) in _OVERLAP_CASES.items():
        findings, err = _run_source(code, workdir)
        detected = len(findings) > 0 if not err else None
        rows.append((name, is_leaky, detected, err))
    return rows


def _confusion(rows: list[tuple]) -> dict:
    tp = fp = tn = fn = skipped = 0
    for _name, leaky, detected, _err in rows:
        if detected is None:
            skipped += 1
            continue
        if leaky and detected:
            tp += 1
        elif leaky and not detected:
            fn += 1
        elif not leaky and detected:
            fp += 1
        else:
            tn += 1
    return {"TP": tp, "FP": fp, "TN": tn, "FN": fn, "skipped": skipped}


def _print_rows(title: str, rows: list[tuple]) -> None:
    print(f"\n{title}")
    print("-" * 78)
    print(f"{'case':<46}{'leaky?':<8}{'detected?':<11}note")
    for name, leaky, detected, err in rows:
        det = "skip" if detected is None else ("YES" if detected else "no")
        note = err[:22] if err else ("✓" if (leaky == detected) else "MISMATCH")
        print(f"{name:<46}{('yes' if leaky else 'no'):<8}{det:<11}{note}")


def main() -> None:
    repo = (
        sys.argv[1]
        if len(sys.argv) > 1
        else os.path.join(
            os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
            "refs",
            "leakage-analysis",
        )
    )
    with tempfile.TemporaryDirectory() as workdir:
        _make_titanic_csv(os.path.join(workdir, "data.csv"))
        bench = _run_benchmark_cases(repo, workdir)
        overlap = _run_overlap_cases(workdir)

    _print_rows("A. Benchmark Preprocessing cases (Yang et al. ASE'22, statistics-based)", bench)
    _print_rows("B. Overlap cases (splitguard's row-identity sweet spot)", overlap)

    cb, co = _confusion(bench), _confusion(overlap)
    print("\nConfusion matrix")
    print("-" * 78)
    print(f"  Preprocessing (statistics):  {cb}")
    print(f"  Overlap (row identity)    :  {co}")
    print("\nHonest reading: splitguard is an OVERLAP detector. It misses statistics-based")
    print("preprocessing leaks (no test row enters a fit) and catches row-identity overlap.")


if __name__ == "__main__":
    main()
