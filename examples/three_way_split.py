"""Three-way split (train / validation / test): leakage into EITHER held-out set is caught.

splitguard taints every held-out portion of every ``train_test_split`` call, so a two-step
train/val/test split tracks both the validation and the test rows. A fit on the full matrix
leaks into both.

Run:  python examples/three_way_split.py
"""

from __future__ import annotations

import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

import splitguard

rng = np.random.default_rng(0)
X = rng.normal(size=(200, 6))
y = (rng.random(200) > 0.5).astype(int)


def main() -> None:
    with splitguard.guard(policy="warn"):
        X_tmp, X_test, y_tmp, y_test = model_selection.train_test_split(
            X, y, test_size=0.2, random_state=0
        )
        X_train, X_val, y_train, y_val = model_selection.train_test_split(
            X_tmp, y_tmp, test_size=0.25, random_state=0
        )
        StandardScaler().fit(X)  # leaks into BOTH validation and test

    leaked = splitguard.report()[0].leaked_rows
    print(
        f"\nHeld-out rows that reached the fit: {leaked} "
        f"(validation {len(X_val)} + test {len(X_test)})"
    )


if __name__ == "__main__":
    main()
