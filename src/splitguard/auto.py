"""Zero-config switch: ``import splitguard.auto`` guards the whole run.

Put a single line at the top of any training script or notebook::

    import splitguard.auto

The runtime guard is installed immediately and a leakage report card is printed when the
process exits: a report if any held-out row reached a fit, a plain confirmation line if a
split was seen and stayed clean, and nothing at all if splitguard was never exercised.

Set ``SPLITGUARD_DISABLE=1`` to make the import a no-op.
"""

from __future__ import annotations

import atexit
import os
import sys

from . import _core, _patch

_ENABLED = os.environ.get("SPLITGUARD_DISABLE", "") not in ("1", "true", "True")

if _ENABLED:
    _patch.install()


def _report_at_exit() -> None:
    findings = _core.report()
    try:
        if findings:
            from . import _report

            _report.emit(findings)
        elif _core._state.taint:
            n = len(_core._state.taint)
            print(
                f"splitguard: no leakage detected ({n} held-out rows tracked this run).",
                file=sys.stderr,
            )
    finally:
        _patch.uninstall()


if _ENABLED:
    atexit.register(_report_at_exit)
