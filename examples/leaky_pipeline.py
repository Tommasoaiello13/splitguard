"""Feature-selection leakage: an optimistic holdout score that is pure illusion.

The data is pure noise with random labels, so the only honest accuracy is ~50%. Selecting
features on the FULL matrix (training + test) lets information from the held-out labels leak
into the model, inflating the holdout score. splitguard flags the offending fit; re-running
the selection on the training split only restores the honest score.

Run:  python examples/leaky_pipeline.py
"""

from __future__ import annotations

import numpy as np
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

import splitguard

rng = np.random.default_rng(0)
N, P, K = 160, 5000, 20
X = rng.normal(size=(N, P))  # pure noise features
y = (rng.random(N) > 0.5).astype(int)  # random labels: no real signal exists


def main() -> None:
    # --- Leaky pipeline: feature selection fitted on the FULL matrix (sees the test set) ---
    # Zero config: the split happens inside guard(), so the held-out rows are tracked
    # automatically -- no annotations, no schema.
    with splitguard.guard(policy="warn"):
        X_tr, X_te, y_tr, y_te = model_selection.train_test_split(
            X, y, test_size=0.4, random_state=0
        )
        selector = SelectKBest(f_classif, k=K).fit(X, y)  # <-- leak: peeks at X_te, y_te
        clf = LogisticRegression(max_iter=1000).fit(selector.transform(X_tr), y_tr)
        leaky_acc = accuracy_score(y_te, clf.predict(selector.transform(X_te)))

    # --- Correct pipeline: selection fitted on the training split only ---
    selector_ok = SelectKBest(f_classif, k=K).fit(X_tr, y_tr)
    clf_ok = LogisticRegression(max_iter=1000).fit(selector_ok.transform(X_tr), y_tr)
    honest_acc = accuracy_score(y_te, clf_ok.predict(selector_ok.transform(X_te)))

    print()
    print(f"Leaky holdout accuracy   : {leaky_acc:.1%}   (looks great -- it is a lie)")
    print(f"Honest holdout accuracy  : {honest_acc:.1%}   (the truth: random data)")
    print(f"Inflation from leakage   : {(leaky_acc - honest_acc) * 100:+.1f} points")


if __name__ == "__main__":
    main()
