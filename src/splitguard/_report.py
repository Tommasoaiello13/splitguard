"""Human-readable rendering of leakage findings.

A Rich panel is used when ``rich`` is installed and the stream is interactive; otherwise
a structured plain-text report is produced (CI- and log-friendly). Rendering never raises.
"""

from __future__ import annotations

_PATTERN_LABEL = {
    "fit_before_split": "fit BEFORE split (saw rows later held out)",
    "fit_after_split": "fit AFTER split on data overlapping the test set",
    "group_leakage": "GROUP leakage (same group in train and test)",
}

_SUGGESTION = {
    "fit_before_split": (
        "Move the split before any fit: split first, then fit/transform on the TRAIN part "
        "only and apply the fitted object to the test part."
    ),
    "fit_after_split": (
        "Fit on the training subset only (e.g. estimator.fit(X_train)); never fit on the "
        "full matrix or on X_test."
    ),
    "group_leakage": (
        "Split by group so an entity never spans train and test: use GroupKFold or "
        "GroupShuffleSplit (or pass groups= to your splitter) instead of a random split."
    ),
}


def _lines(findings: list) -> list[str]:
    out = ["", "splitguard: data leakage detected", "=" * 52]
    for i, f in enumerate(findings, 1):
        pct = f.fraction * 100.0
        out += [
            f"[{i}] {f.estimator}",
            f"    pattern     {_PATTERN_LABEL.get(f.pattern, f.pattern)}",
            f"    leaked      {f.leaked_rows}/{f.fit_rows} held-out rows reached this fit "
            f"({pct:.0f}%)",
            f"    call site   {f.call_site}",
            f"    fix         {_SUGGESTION.get(f.pattern, '')}",
            "",
        ]
    out.append(
        "Note: coverage-bounded -- this reports leakage that occurred in this run; it does "
        "not prove the pipeline is leak-free on unexercised paths."
    )
    return out


def render(findings: list) -> str:
    """Return the plain-text report for *findings*."""
    return "\n".join(_lines(findings))


def emit(findings: list) -> None:
    """Print the report: a Rich panel when available, else structured plain text."""
    try:
        from rich.console import Console
        from rich.panel import Panel
        from rich.table import Table

        console = Console(stderr=True)
        table = Table(show_header=True, header_style="bold red", box=None)
        table.add_column("#", justify="right")
        table.add_column("estimator", style="bold")
        table.add_column("pattern")
        table.add_column("leaked", justify="right")
        table.add_column("call site")
        for i, f in enumerate(findings, 1):
            table.add_row(
                str(i),
                f.estimator,
                _PATTERN_LABEL.get(f.pattern, f.pattern),
                f"{f.leaked_rows}/{f.fit_rows} ({f.fraction * 100:.0f}%)",
                f.call_site,
            )
        fixes = "\n".join(f"• {f.estimator}: {_SUGGESTION.get(f.pattern, '')}" for f in findings)
        console.print(
            Panel(
                table,
                title="[bold red]splitguard — data leakage detected[/]",
                subtitle="coverage-bounded: reports leaks that occurred in this run",
                border_style="red",
            )
        )
        console.print(Panel(fixes, title="suggested fixes", border_style="yellow"))
    except Exception:
        import sys

        print(render(findings), file=sys.stderr)
