"""Dynamic splits: check leakage independently across many different splits.

``across_splits`` reruns a pipeline under a fresh guard for each seed and reports whether
leakage occurred on that split. This catches leakage that depends on the split, and confirms
that a *corrected* pipeline stays clean across all of them. (It reports leakage per split
only -- not score stability, and not multi-test leakage, which a row-identity engine cannot
detect.)

Run:  python examples/across_splits.py
"""

from __future__ import annotations

import numpy as np
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest, f_classif

import splitguard

rng = np.random.default_rng(0)
X = rng.normal(size=(160, 400))
y = (rng.random(160) > 0.5).astype(int)


def leaky(seed: int) -> None:
    model_selection.train_test_split(X, y, test_size=0.3, random_state=seed)
    SelectKBest(f_classif, k=15).fit(X, y)  # selection on the full matrix -> leak


def clean(seed: int) -> None:
    X_tr, X_te, y_tr, y_te = model_selection.train_test_split(
        X, y, test_size=0.3, random_state=seed
    )
    SelectKBest(f_classif, k=15).fit(X_tr, y_tr)  # selection on train only -> clean


def _table(title: str, results: dict) -> None:
    print(f"\n{title}")
    print(f"{'seed':<8}{'leakage detected':<20}{'leaked rows'}")
    for seed, findings in results.items():
        rows = findings[0].leaked_rows if findings else 0
        print(f"{seed:<8}{('YES' if findings else 'no'):<20}{rows}")


def main() -> None:
    seeds = range(5)
    _table("Leaky pipeline across splits", splitguard.across_splits(leaky, seeds))
    _table("Corrected pipeline across splits", splitguard.across_splits(clean, seeds))


if __name__ == "__main__":
    main()
