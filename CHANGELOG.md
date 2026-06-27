# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/) and the project adheres to semantic versioning.

## [0.1.0] - 2026-06-27

### Added
- Runtime train–test leakage detection by test-set row-identity taint tracking.
- Automatic scikit-learn hooks: every estimator `fit` and `train_test_split` are instrumented.
- Detection of both temporal orders: fit-after-split on overlapping data, and fit-before-split
  (flagged retroactively).
- `guard()` context manager, `install()` / `uninstall()`, `mark_test()`, `configure()`,
  `report()` and the `LeakageError` type.
- Context-aware policy: raise (scripts/pytest), warn (notebooks), log (non-interactive).
- Optional Rich terminal report with a structured plain-text fallback.
- pytest plugin: `no_leakage` fixture and a project-wide `splitguard = true` option.
- Examples (`leaky_pipeline.py`, `comparison.py`) and README figures generated from live runs.
- Three-way (train/validation/test) split tracking and `across_splits()` for per-split leakage
  across dynamic splits, with examples (`three_way_split.py`, `across_splits.py`).
- Validation against the published Yang et al. (ASE'22) `leakage-analysis` benchmark
  (`validation/run_benchmark.py`): full recall/precision on Overlap leakage; honest miss on
  statistics-based Preprocessing leakage. Scope and results documented in the README.
- **Zero-config mode: `import splitguard.auto`** — one line guards the whole run and prints a
  leakage report card at exit (the runtime "firewall" experience). Example `one_import.py`.
- **Every split:** cross-validators (`KFold`, `StratifiedKFold`, `GroupKFold`, `TimeSeriesSplit`,
  `ShuffleSplit`, …) are instrumented with per-fold taint (no cross-fold false positives), plus
  three-way train/val/test and `across_splits()` for dynamic splits.
- **Every model:** XGBoost, LightGBM and CatBoost sklearn-style `.fit` are instrumented in
  addition to all scikit-learn estimators. External-estimator hooks are fully exception-contained.
- **Group leakage:** `mark_groups(X, groups)` flags any group (entity) that spans train and test
  — across `train_test_split` and cross-validation folds — and confirms `GroupKFold`/
  `GroupShuffleSplit` stay clean. This is leakage `Pipeline` does not prevent.

### Fixed
- **No more silent false-clean (import-order trap):** if estimators are fitted but no held-out
  split was ever tracked — typically because `train_test_split` was imported *before* splitguard
  (a stale binding bypasses the hook) — splitguard now warns loudly that NO leakage could be
  detected, instead of reporting a misleading clean result. (Flagged independently by three ML
  engineers trying the tool.)
- Call-site resolution for module-level / notebook-cell fits: replaced
  `traceback.walk_stack` (depth-fragile on CPython 3.13) with a direct `sys._getframe` walk.
- **False positives on legitimate duplicate rows (pandas):** row identity is now the pandas
  index label (preserved by `train_test_split`), not feature values — eliminating false alarms on
  genuinely repeated rows while still catching the same sample appearing in train and test.
  Validated: FP on duplicate-row pandas traps 16/16 → 0/16. (NumPy without an index is a residual
  information-theoretic limit, documented.)
- `guard(groups=(X, group_ids))`: register groups for group-leakage detection without the
  "must call `mark_groups` inside the block" footgun (the entry `reset()` no longer drops them).
- `guard(strict_transforms=True)` / `configure(strict_transforms=...)`: opt-in warning when a
  transformer is fitted before the split (the `fit_transform`-then-split blind spot that row
  identity cannot confirm). Off by default to avoid false positives.
- Identity-reliability warnings: splitguard now warns (once) when a NumPy fit is checked against
  pandas index-based held-out rows (identities can't match → leak missed), or when held-out NumPy
  rows contain duplicate values (value identity is ambiguous → possible false positive). Silent
  gaps are now visible.
- Configuration no longer bleeds across `guard()` calls (`reset()` now clears the policy and
  threshold); native trainers no longer mutate the user's `DMatrix`/`Dataset` (weak-keyed map);
  the install reference count is restored if `install()` raises; concurrent hook-time patching is
  lock-guarded; `uninstall()` no longer imports sklearn at interpreter exit; call-site filtering is
  case-insensitive on Windows. (Found by a code-review pass on the grown source.)
