"""The zero-config experience: a single import guards the whole script.

Run:  python examples/one_import.py
The leakage report card is printed automatically when the script exits.
"""

from __future__ import annotations

# ruff: noqa: I001, E402  (the one-import line must come first, on purpose)
import splitguard.auto  # noqa: F401  <- the only line you add

import numpy as np
from sklearn import model_selection
from sklearn.preprocessing import StandardScaler

rng = np.random.default_rng(0)
X = rng.normal(size=(120, 6))
y = (rng.random(120) > 0.5).astype(int)

scaler = StandardScaler().fit(X)  # fitted before the split -> leak
X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, random_state=0)
# ... rest of a normal training script; splitguard reports at exit.
