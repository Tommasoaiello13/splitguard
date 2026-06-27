"""The one-import zero-config mode (`import splitguard.auto`), tested in a subprocess.

A subprocess is used because ``import splitguard.auto`` installs a process-wide guard and an
atexit report card, which must not leak into the rest of the test session.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

_LEAKY = textwrap.dedent(
    """
    import splitguard.auto
    import numpy as np
    from sklearn import model_selection
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5)); y = (rng.random(80) > 0.5).astype(int)
    StandardScaler().fit(X)
    model_selection.train_test_split(X, y, random_state=0)
    """
)

_CLEAN = textwrap.dedent(
    """
    import splitguard.auto
    import numpy as np
    from sklearn import model_selection
    from sklearn.preprocessing import StandardScaler
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 5)); y = (rng.random(80) > 0.5).astype(int)
    X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, random_state=0)
    StandardScaler().fit(X_tr)
    """
)


def _run(code: str, disable: bool = False) -> str:
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join(p for p in sys.path if p)
    env["PYTHONIOENCODING"] = "utf-8"
    if disable:
        env["SPLITGUARD_DISABLE"] = "1"
    proc = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        env=env,
        timeout=180,
    )
    return (proc.stdout + proc.stderr).lower()


def test_auto_reports_leak_at_exit():
    assert "leak" in _run(_LEAKY)


def test_auto_confirms_clean_run():
    out = _run(_CLEAN)
    assert "no leakage detected" in out


def test_auto_disabled_is_silent():
    out = _run(_LEAKY, disable=True)
    assert "leak" not in out
