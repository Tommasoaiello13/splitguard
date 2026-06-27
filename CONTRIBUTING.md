# Contributing to splitguard

Thanks for your interest in improving splitguard.

## Development setup

```bash
git clone https://github.com/Tommasoaiello13/splitguard
cd splitguard
pip install -e ".[dev]"
```

## Before opening a pull request

```bash
ruff check src tests        # lint
ruff format --check src tests
mypy src                    # types
pytest                      # tests (must stay green)
```

New behaviour needs a test. Detection changes should add a ground-truth oracle case to
`tests/` (a known-leaky pipeline that must be flagged, or a known-clean one that must stay
silent).

## Design invariants (do not break)

- **Never mutate user data** and **never change an estimator's return value.**
- **Never raise out of a hook** — instrumentation failures are swallowed; the user's program
  always runs.
- **Be honest** — report "boundary" vs "birthplace" accurately; do not over-claim. splitguard is
  a coverage-bounded detector, not a proof.

## Good first contributions

- Native adapters beyond scikit-learn (e.g. a thin wrapper for a `fit`-style API).
- An index-identity mode to remove duplicate-row false positives.
- Additional detectors in the Silent Pipeline Correctness family (schema drift, merge explosion).
