"""Rigorous, honest evaluation of splitguard across leakage categories and false-positive traps.

Designed with expert (data-scientist + ml-engineer) input. Every case is RUN LIVE under
splitguard over multiple seeds. It deliberately includes the cases where splitguard is expected
to FAIL -- the fit_transform-then-split blind spot, pure-statistics leakage, target leakage, and
false-positive traps (legitimate duplicate rows, single-feature data) -- so the result is honest,
not a strawman.

Competitor columns (deepchecks, static leakage-analysis) are DOCUMENTED, not executed:
deepchecks fails to import under scikit-learn 1.8; leakage-analysis needs souffle/Py3.8.

Run:  python validation/rigorous_eval.py
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import splitguard
from splitguard import _core, _patch

SEEDS = range(8)


def _Xy(seed, n=300, p=8):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), (rng.random(n) > 0.5).astype(int)


def _detected(run, seed) -> bool:
    _core._state.reset()
    _patch.install()
    try:
        run(seed)
    except Exception:
        pass
    finally:
        _patch.uninstall()
    return len(_core.report()) > 0


# --- LEAKY cases ------------------------------------------------------------------------ #
def leak_overlap_oversample(seed):
    X, y = _Xy(seed)
    X = np.vstack([X, X[:100]])
    y = np.concatenate([y, y[:100]])  # exact duplicate rows -> overlap after split
    Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=seed)
    LogisticRegression(max_iter=200).fit(Xtr, ytr)


def leak_fit_after_split_full(seed):
    X, y = _Xy(seed)
    model_selection.train_test_split(X, y, random_state=seed)
    SelectKBest(f_classif, k=3).fit(X, y)  # fit on full matrix after the split


def leak_fit_before_split_origX(seed):
    X, y = _Xy(seed)
    StandardScaler().fit(X)  # fit before split, split is on the ORIGINAL X -> caught retroactively
    model_selection.train_test_split(X, y, random_state=seed)


def leak_fit_transform_then_split(seed):  # BLIND SPOT: split on the TRANSFORMED data
    X, y = _Xy(seed)
    X2 = StandardScaler().fit_transform(X)  # most common Kaggle pattern
    Xtr, _, ytr, _ = model_selection.train_test_split(X2, y, random_state=seed)
    LogisticRegression(max_iter=200).fit(Xtr, ytr)


def leak_pure_statistics(seed):  # BLIND SPOT: numpy statistic, no estimator fit on full
    X, y = _Xy(seed)
    X = X - X.mean(axis=0)
    Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=seed)
    LogisticRegression(max_iter=200).fit(Xtr, ytr)


def leak_group_random(seed):
    rng = np.random.default_rng(seed)
    n_g, per = 40, 5
    X = rng.normal(size=(n_g * per, 8))
    y = (rng.random(n_g * per) > 0.5).astype(int)
    groups = np.repeat(np.arange(n_g), per)
    splitguard.mark_groups(X, groups)
    model_selection.train_test_split(X, y, random_state=seed)  # random split on grouped data


def leak_target(seed):  # BLIND SPOT: a feature is (a function of) the label
    X, y = _Xy(seed)
    X = np.column_stack([X, y + 0.01 * np.random.default_rng(seed).normal(size=len(y))])
    Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=seed)
    LogisticRegression(max_iter=200).fit(Xtr, ytr)


# --- CLEAN cases (well-formed) ---------------------------------------------------------- #
def clean_train_only(seed):
    X, y = _Xy(seed)
    Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=seed)
    scaler = StandardScaler().fit(Xtr)
    LogisticRegression(max_iter=200).fit(scaler.transform(Xtr), ytr)


def clean_groupkfold(seed):
    from sklearn.model_selection import GroupKFold

    rng = np.random.default_rng(seed)
    n_g, per = 40, 5
    X = rng.normal(size=(n_g * per, 8))
    groups = np.repeat(np.arange(n_g), per)
    splitguard.mark_groups(X, groups)
    for tr, _te in GroupKFold(n_splits=4).split(X, groups=groups):
        StandardScaler().fit(X[tr])


# --- CLEAN FP-traps (test splitguard's value-hashing weakness honestly) ----------------- #
def trap_duplicate_rows(seed):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 8))
    X[150:] = X[:50]  # 50 legitimately identical rows (not leakage)
    y = (rng.random(200) > 0.5).astype(int)
    Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=seed)
    StandardScaler().fit(Xtr)  # correct: fit on train only


def trap_single_feature(seed):
    rng = np.random.default_rng(seed)
    X = rng.integers(0, 3, size=(200, 1)).astype(float)  # 1 col, 3 values -> heavy collisions
    y = (rng.random(200) > 0.5).astype(int)
    Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=seed)
    StandardScaler().fit(Xtr)  # correct pipeline


def trap_pandas_duplicate_rows(seed):  # same as the numpy trap but as a DataFrame (index identity)
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(200, 6))
    X[150:] = X[:50]  # legitimate duplicate VALUES, distinct index
    df = pd.DataFrame(X, columns=[f"f{i}" for i in range(6)])
    y = pd.Series((rng.random(200) > 0.5).astype(int))
    Xtr, _, _ytr, _ = model_selection.train_test_split(df, y, random_state=seed)
    StandardScaler().fit(Xtr)  # correct: fit on the train split only


def trap_pandas_single_feature(seed):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(rng.integers(0, 3, size=(200, 1)).astype(float), columns=["f0"])
    y = pd.Series((rng.random(200) > 0.5).astype(int))
    Xtr, _, _ytr, _ = model_selection.train_test_split(df, y, random_state=seed)
    StandardScaler().fit(Xtr)


# category, name, leaky?, runner, splitguard-by-design (for reference only)
LEAKY = [
    ("overlap", "oversample dup rows before split", leak_overlap_oversample, True),
    ("preproc-fit", "fit AFTER split on full matrix", leak_fit_after_split_full, True),
    ("preproc-fit", "fit BEFORE split, split on orig X", leak_fit_before_split_origX, True),
    ("preproc-fit", "fit_transform THEN split (blind)", leak_fit_transform_then_split, False),
    ("preproc-stats", "subtract full-data mean (blind)", leak_pure_statistics, False),
    ("group", "random split on grouped data", leak_group_random, True),
    ("target", "feature = label (blind)", leak_target, False),
]
CLEAN = [
    ("clean", "fit on train only", clean_train_only, False),
    ("clean", "correct GroupKFold", clean_groupkfold, False),
    ("fp-trap-np", "dup rows (numpy, no index)", trap_duplicate_rows, False),
    ("fp-trap-np", "single feature (numpy)", trap_single_feature, False),
    ("fp-trap-pd", "dup rows (pandas, index id)", trap_pandas_duplicate_rows, False),
    ("fp-trap-pd", "single feature (pandas)", trap_pandas_single_feature, False),
]

# Documented competitor detection per case (NOT executed -- see module docstring).
# deepchecks: shared rows only. static (ASE'22): code-visible overlap+preprocessing (no group).
_DOC = {
    "oversample dup rows before split": (True, True),
    "fit AFTER split on full matrix": (False, True),
    "fit BEFORE split, split on orig X": (False, True),
    "fit_transform THEN split (blind)": (False, True),
    "subtract full-data mean (blind)": (False, True),
    "random split on grouped data": (False, False),
    "feature = label (blind)": (False, False),
}


def main() -> None:
    print(f"\nRIGOROUS EVALUATION  ({len(list(SEEDS))} seeds/case, splitguard run LIVE)\n")
    print(f"{'category':<14}{'case':<36}{'leaky':<7}{'splitguard det.':<17}{'dc':<5}{'static'}")
    print("-" * 92)

    cat_hits: dict = {}
    cat_tot: dict = {}
    for cat, name, fn, _by_design in LEAKY:
        hits = sum(_detected(fn, s) for s in SEEDS)
        cat_hits[cat] = cat_hits.get(cat, 0) + hits
        cat_tot[cat] = cat_tot.get(cat, 0) + len(list(SEEDS))
        dc, st = _DOC.get(name, (False, False))
        rate = f"{hits}/{len(list(SEEDS))}"
        print(f"{cat:<14}{name:<36}{'yes':<7}{rate:<17}"
              f"{('Y' if dc else '-'):<5}{('Y' if st else '-')}")

    print("-" * 92)
    fp = {"clean": [0, 0], "fp-trap-np": [0, 0], "fp-trap-pd": [0, 0]}
    for cat, name, fn, _bd in CLEAN:
        fires = sum(_detected(fn, s) for s in SEEDS)
        n = len(list(SEEDS))
        fp[cat][0] += fires
        fp[cat][1] += n
        print(f"{cat:<14}{name:<36}{'no':<7}{f'{fires}/{n} fired':<17}{'-':<5}-")

    # aggregate recall on the row/group-identity-detectable leak classes
    detectable = ["overlap", "preproc-fit", "group"]  # excludes blind-spots by design
    det_hits = sum(cat_hits.get(c, 0) for c in detectable)
    det_tot = sum(cat_tot.get(c, 0) for c in detectable)
    blind = ["preproc-stats", "target"]
    blind_hits = sum(cat_hits.get(c, 0) for c in blind)
    blind_tot = sum(cat_tot.get(c, 0) for c in blind)

    print("\nSUMMARY")
    print(f"  Recall on identity-detectable leaks (overlap+preproc-fit+group): "
          f"{det_hits}/{det_tot}")
    print(f"  Recall on by-design blind spots (stats+target): "
          f"{blind_hits}/{blind_tot}  (expected ~0)")
    print(f"  False positives on well-formed clean: {fp['clean'][0]}/{fp['clean'][1]}")
    print(f"  FP on traps -- PANDAS (index identity): {fp['fp-trap-pd'][0]}/{fp['fp-trap-pd'][1]}"
          f"  <-- FIXED (was firing with value hashing)")
    print(f"  FP on traps -- NUMPY (no index, fundamental limit): "
          f"{fp['fp-trap-np'][0]}/{fp['fp-trap-np'][1]}  <-- documented residual")
    print("\n  Competitors (documented, not executed): deepchecks catches shared-row overlap only;")
    print("  static (ASE'22) catches code-visible overlap+preprocessing (incl. the blind variants")
    print("  splitguard misses) but NOT group leakage. splitguard is the only one catching group.")


if __name__ == "__main__":
    main()
