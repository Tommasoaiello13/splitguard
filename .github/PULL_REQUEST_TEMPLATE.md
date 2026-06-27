## Summary

What does this change and why?

## Checklist

- [ ] Tests added/updated (a ground-truth oracle case for detection changes)
- [ ] `ruff check` and `ruff format --check` pass
- [ ] `mypy src` passes
- [ ] `pytest` is green
- [ ] Design invariants respected: no data mutation, no change to estimator results, hooks never
      raise, honest reporting
