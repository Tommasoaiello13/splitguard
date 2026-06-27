"""Head-to-head KPI comparison: splitguard vs deepchecks vs static analysis.

For each labelled pipeline we record what each method detects. The methods are fundamentally
different and that is the point:

* splitguard  observes the COMPUTATION at runtime (which rows reached a fit);
* deepchecks   inspects the assembled train/test DATA (overlapping samples, drift);
* leakage-analysis (Yang et al. ASE'22) reads the CODE statically (souffle/Datalog).

leakage-analysis needs souffle + Python 3.8 and is not run here; its column is filled from its
published taxonomy (Overlap / Preprocessing / Multi-test) applied to each case -- an honest
reference for what a static tool detects, including the cases where it beats splitguard.

Run:  python validation/compare_methods.py
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import splitguard
from splitguard import _core, _patch

RNG = np.random.default_rng(0)
COLS = [f"f{i}" for i in range(6)]


def _Xy(n=160, p=6, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(size=(n, p)), (rng.random(n) > 0.5).astype(int)


def _df(X, y):
    d = pd.DataFrame(X, columns=COLS[: X.shape[1]])
    d["y"] = y
    return d


# --- splitguard runners (run the real pipeline under a guard) --------------------------- #
def _sg(run) -> bool:
    _core._state.reset()
    _patch.install()
    try:
        run()
    except Exception:
        pass
    finally:
        _patch.uninstall()
    return len(_core.report()) > 0


def sg_overlap():
    def run():
        X, y = _Xy()
        X = np.vstack([X, X[:60]])
        y = np.concatenate([y, y[:60]])  # duplicate rows -> overlap after split
        Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=0)
        LogisticRegression(max_iter=300).fit(Xtr, ytr)

    return _sg(run)


def sg_preprocess_full_fit():
    def run():
        X, y = _Xy()
        scaler = StandardScaler().fit(X)  # fit on full matrix before split
        Xtr, Xte, ytr, yte = model_selection.train_test_split(X, y, random_state=0)
        LogisticRegression(max_iter=300).fit(scaler.transform(Xtr), ytr)

    return _sg(run)


def sg_preprocess_stats():
    def run():
        X, y = _Xy()
        col_mean = X.mean(axis=0)  # statistic over the FULL data (numpy, no estimator fit)
        X = np.where(np.isnan(X), col_mean, X)
        Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=0)
        LogisticRegression(max_iter=300).fit(Xtr, ytr)

    return _sg(run)


def sg_group():
    def run():
        n_g, per = 40, 4
        X = RNG.normal(size=(n_g * per, 6))
        y = (RNG.random(n_g * per) > 0.5).astype(int)
        groups = np.repeat(np.arange(n_g), per)
        splitguard.mark_groups(X, groups)
        model_selection.train_test_split(X, y, random_state=0)  # random split on grouped data

    return _sg(run)


def sg_clean():
    def run():
        X, y = _Xy()
        Xtr, Xte, ytr, yte = model_selection.train_test_split(X, y, random_state=0)
        scaler = StandardScaler().fit(Xtr)
        LogisticRegression(max_iter=300).fit(scaler.transform(Xtr), ytr)

    return _sg(run)


# --- deepchecks runner: inspects the assembled train/test data -------------------------- #
def _make_datasets(case: str):
    """Return (train_df, test_df) as the user would have them after the (leaky) pipeline."""
    if case == "overlap":
        X, y = _Xy()
        X = np.vstack([X, X[:60]])
        y = np.concatenate([y, y[:60]])
        Xtr, Xte, ytr, yte = model_selection.train_test_split(X, y, random_state=0)
        return _df(Xtr, ytr), _df(Xte, yte)
    if case == "group":
        n_g, per = 40, 4
        X = RNG.normal(size=(n_g * per, 6))
        y = (RNG.random(n_g * per) > 0.5).astype(int)
        Xtr, Xte, ytr, yte = model_selection.train_test_split(X, y, random_state=0)
        return _df(Xtr, ytr), _df(Xte, yte)  # disjoint rows, shared groups
    # preprocess_full_fit / preprocess_stats / clean all yield disjoint, clean-looking data
    X, y = _Xy()
    Xtr, Xte, ytr, yte = model_selection.train_test_split(X, y, random_state=0)
    return _df(Xtr, ytr), _df(Xte, yte)


def deepchecks_detects(case: str) -> bool | None:
    try:
        from deepchecks.tabular import Dataset
        from deepchecks.tabular.checks import TrainTestSamplesMix
    except Exception:
        return None  # not installed
    try:
        train_df, test_df = _make_datasets(case)
        res = TrainTestSamplesMix().run(
            train_dataset=Dataset(train_df, label="y"),
            test_dataset=Dataset(test_df, label="y"),
        )
        val = res.value
        ratio = val.get("ratio", 0) if isinstance(val, dict) else 0
        return bool(ratio and ratio > 0)
    except Exception:
        return None


# leakage-analysis (static, ASE'22) -- from its taxonomy: Overlap + Preprocessing (+ Multi-test);
# group leakage is NOT in its taxonomy. Reference values, not executed (needs souffle/Py3.8).
_STATIC = {
    "overlap": True,
    "preprocess_full_fit": True,
    "preprocess_stats": True,
    "group": False,
    "clean": False,
}

# deepchecks could not be executed here: its tabular module fails to import under scikit-learn
# 1.8 (references the removed 'max_error' scorer). Reference values are from its documented
# mechanism -- TrainTestSamplesMix compares the assembled datasets and detects test ROWS that
# appear in train, i.e. only the shared-row overlap case.
_DEEPCHECKS = {
    "overlap": True,
    "preprocess_full_fit": False,
    "preprocess_stats": False,
    "group": False,
    "clean": False,
}

CASES = [
    ("overlap (shared rows)", "overlap", True, sg_overlap),
    ("preprocessing: fit on full matrix", "preprocess_full_fit", True, sg_preprocess_full_fit),
    ("preprocessing: full-data statistic", "preprocess_stats", True, sg_preprocess_stats),
    ("group leakage (random split)", "group", True, sg_group),
    ("clean pipeline", "clean", False, sg_clean),
]


def _overhead() -> tuple[float, float]:
    """Median wall-time of a small training run, unguarded vs guarded."""
    X, y = _Xy(n=400, p=20)

    def workload():
        for _ in range(20):
            Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=0)
            LogisticRegression(max_iter=100).fit(Xtr, ytr)

    def timed(fn):
        ts = []
        for _ in range(3):
            t0 = time.perf_counter()
            fn()
            ts.append(time.perf_counter() - t0)
        return sorted(ts)[1]

    base = timed(workload)
    _core._state.reset()
    _patch.install()
    try:
        guarded = timed(workload)
    finally:
        _patch.uninstall()
    return base, guarded


def main() -> None:
    dc_live = deepchecks_detects("overlap")  # probe whether deepchecks runs at all
    dc_executed = dc_live is not None

    rows = []
    for label, case, leaky, sg_fn in CASES:
        sg = sg_fn()
        dc = deepchecks_detects(case) if dc_executed else _DEEPCHECKS[case]
        rows.append((label, leaky, sg, dc, _STATIC[case]))

    def mark(detected, leaky):
        if detected is None:
            return "n/a"
        if leaky:
            return "DETECT" if detected else "miss"
        return "ok(silent)" if not detected else "FALSE+"

    print("\nDETECTION BY LEAKAGE TYPE")
    print(f"{'case':<38}{'leaky':<7}{'splitguard':<12}{'deepchecks':<12}{'static(ASE22)'}")
    print("-" * 90)
    for label, leaky, sg, dc, st in rows:
        print(
            f"{label:<38}{('yes' if leaky else 'no'):<7}"
            f"{mark(sg, leaky):<12}{mark(dc, leaky):<12}{mark(st, leaky)}"
        )

    leaky_rows = [(sg, dc, st) for _l, lk, sg, dc, st in rows if lk]
    n = len(leaky_rows)
    false_pos = [sg for _l, lk, sg, _dc, _st in rows if not lk and sg]

    def recall(idx):
        return f"{sum(1 for r in leaky_rows if r[idx] is True)}/{n}"

    print(f"\nRecall on leaky cases:  splitguard {recall(0)} · "
          f"deepchecks {recall(1)} · static {recall(2)}")
    print(f"False positives on the clean pipeline (splitguard): {len(false_pos)}")
    print("splitguard = run live; deepchecks = NOT executable here (sklearn-1.8 incompat),"
          " documented mechanism; static = documented (needs souffle/Py3.8).")

    base, guarded = _overhead()
    print(f"\nRUNTIME OVERHEAD (20 fits): unguarded {base * 1e3:.0f} ms -> "
          f"guarded {guarded * 1e3:.0f} ms ({(guarded / base - 1) * 100:+.0f}%)")

    print("\nQUALITATIVE KPIs")
    print(f"{'KPI':<24}{'splitguard':<26}{'deepchecks':<26}{'static(ASE22)'}")
    print("-" * 100)
    kpis = [
        ("setup", "1 line (import .auto)", "wrap Dataset + suite.run", "souffle + CLI/Docker"),
        ("when", "runtime (live)", "after assembling data", "static (pre-run)"),
        ("localizes line", "yes (file:line)", "no", "yes (static)"),
        ("group leakage", "yes (mark_groups)", "no", "no"),
        ("leaves-data-clean leak", "yes (sees the fit)", "no (sees data)", "yes (sees code)"),
        ("frameworks", "sklearn+xgb/lgbm/cat", "tabular datasets", "notebooks (sklearn)"),
        ("hard deps", "numpy", "pandas+plotly+…(heavy)", "souffle+py3.8"),
    ]
    for k, a, b, c in kpis:
        print(f"{k:<24}{a:<26}{b:<26}{c}")

    print("\nMethodology: splitguard observes the FIT (catches leaks that leave the data clean:")
    print("preprocessing-fit, group); deepchecks inspects the DATA (only shared rows); static")
    print("analysis reads the CODE (catches preprocessing incl. statistics, no group, no runtime).")


if __name__ == "__main__":
    main()
