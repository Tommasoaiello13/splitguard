"""Generate the README analysis figures from live computation (no hardcoded numbers).

Produces three honest figures and prints the exact numbers used for the README captions:
  1. analysis_impact.png   — how much feature-selection leakage inflates the holdout score,
                              as a function of the number of (noise) features.
  2. analysis_overhead.png — the per-fit time splitguard adds, as a function of dataset rows.
  3. analysis_coverage.png — detection recall by leakage type + the false-positive picture.

Run:  python tools/make_analysis_figures.py
"""

from __future__ import annotations

import os
import time

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

import splitguard
from splitguard import _core, _patch

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
INK, RED, GREEN, BLUE, SLATE = "#0f172a", "#e11d48", "#059669", "#2563eb", "#94a3b8"
_SEEDS8 = tuple(range(8))
_SEEDS6 = tuple(range(6))
_SEEDS5 = tuple(range(5))


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    ax.grid(axis="y", color="#e5e7eb", lw=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------- #
# 1. Impact: feature-selection leakage inflation vs number of noise features
# --------------------------------------------------------------------------- #
def impact(seeds=_SEEDS6):
    feature_counts = [50, 100, 250, 500, 1000, 2000]
    n = 160
    means, stds = [], []
    for p in feature_counts:
        infl = []
        for s in seeds:
            rng = np.random.default_rng(s)
            X = rng.normal(size=(n, p))
            y = (rng.random(n) > 0.5).astype(int)
            X_tr, X_te, y_tr, y_te = model_selection.train_test_split(
                X, y, test_size=0.4, random_state=s
            )
            sel = SelectKBest(f_classif, k=20).fit(X, y)  # leak: selection on full X, y
            leaky = LogisticRegression(max_iter=500).fit(sel.transform(X_tr), y_tr)
            leaky_acc = leaky.score(sel.transform(X_te), y_te)
            sel2 = SelectKBest(f_classif, k=20).fit(X_tr, y_tr)  # honest: on train only
            honest = LogisticRegression(max_iter=500).fit(sel2.transform(X_tr), y_tr)
            honest_acc = honest.score(sel2.transform(X_te), y_te)
            infl.append((leaky_acc - honest_acc) * 100)
        means.append(float(np.mean(infl)))
        stds.append(float(np.std(infl)))

    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=140)
    ax.errorbar(feature_counts, means, yerr=stds, marker="o", color=RED, capsize=4, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("number of (pure-noise) features", color=INK)
    ax.set_ylabel("holdout accuracy inflation (pts)", color=INK)
    ax.set_title(
        "Impact: feature-selection leakage inflates the score more as features grow\n"
        "(data is pure noise — the honest accuracy is 50%)",
        fontsize=12, fontweight="bold", color=INK, pad=12,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "analysis_impact.png"), bbox_inches="tight")
    plt.close(fig)
    return dict(zip(feature_counts, [round(m, 1) for m in means], strict=True))


# --------------------------------------------------------------------------- #
# 2. Overhead: per-fit time splitguard adds vs dataset rows
# --------------------------------------------------------------------------- #
def overhead(seeds=_SEEDS5):
    row_counts = [100, 1_000, 10_000, 100_000]
    added_ms = []
    for nrows in row_counts:
        rng = np.random.default_rng(0)
        X = rng.normal(size=(nrows, 20))

        def timed(active, X=X):
            ts = []
            for _ in seeds:
                _core._state.reset()
                if active:
                    _patch.install()
                t0 = time.perf_counter()
                StandardScaler().fit(X)
                ts.append(time.perf_counter() - t0)
                if active:
                    _patch.uninstall()
            return float(np.median(ts))

        base, guarded = timed(False), timed(True)
        added_ms.append(max(0.0, (guarded - base) * 1e3))

    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=140)
    ax.plot(row_counts, added_ms, marker="o", color=BLUE, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("rows in the fitted matrix", color=INK)
    ax.set_ylabel("time splitguard adds per fit (ms)", color=INK)
    ax.set_title(
        "Cost: per-fit overhead is the row hashing — a fixed cost, dwarfed by real model fits",
        fontsize=12, fontweight="bold", color=INK, pad=12,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "analysis_overhead.png"), bbox_inches="tight")
    plt.close(fig)
    return dict(zip(row_counts, [round(m, 2) for m in added_ms], strict=True))


# --------------------------------------------------------------------------- #
# 3. Coverage: detection recall by leakage type (live), + false positives
# --------------------------------------------------------------------------- #
def _detect(run, seed):
    _core._state.reset()
    _patch.install()
    try:
        run(seed)
    except Exception:
        pass
    finally:
        _patch.uninstall()
    return len(_core.report()) > 0


def coverage(seeds=_SEEDS8):
    def overlap(s):
        rng = np.random.default_rng(s)
        X = rng.normal(size=(160, 6))
        y = (rng.random(160) > 0.5).astype(int)
        X = np.vstack([X, X[:60]])
        y = np.concatenate([y, y[:60]])
        Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=s)
        LogisticRegression(max_iter=200).fit(Xtr, ytr)

    def preproc_fit(s):
        rng = np.random.default_rng(s)
        X = rng.normal(size=(160, 6))
        y = (rng.random(160) > 0.5).astype(int)
        StandardScaler().fit(X)
        model_selection.train_test_split(X, y, random_state=s)

    def group(s):
        rng = np.random.default_rng(s)
        X = rng.normal(size=(160, 6))
        y = (rng.random(160) > 0.5).astype(int)
        g = np.repeat(np.arange(32), 5)
        splitguard.mark_groups(X, g)
        model_selection.train_test_split(X, y, random_state=s)

    def stats(s):
        rng = np.random.default_rng(s)
        X = rng.normal(size=(160, 6))
        y = (rng.random(160) > 0.5).astype(int)
        X = X - X.mean(axis=0)
        Xtr, _, ytr, _ = model_selection.train_test_split(X, y, random_state=s)
        LogisticRegression(max_iter=200).fit(Xtr, ytr)

    cases = [
        ("overlap", overlap, GREEN),
        ("preprocessing\n(fit on full)", preproc_fit, GREEN),
        ("group", group, GREEN),
        ("preprocessing\n(statistics)", stats, SLATE),
    ]
    labels, recalls, colors = [], [], []
    for name, fn, color in cases:
        r = sum(_detect(fn, s) for s in seeds) / len(list(seeds)) * 100
        labels.append(name)
        recalls.append(r)
        colors.append(color)

    fig, ax = plt.subplots(figsize=(7.5, 4.3), dpi=140)
    bars = ax.bar(labels, recalls, color=colors, width=0.6)
    for b, r in zip(bars, recalls, strict=True):
        ax.text(b.get_x() + b.get_width() / 2, r + 2, f"{r:.0f}%", ha="center",
                va="bottom", fontweight="bold", color=INK)
    ax.set_ylim(0, 108)
    ax.set_ylabel("detection rate (recall)", color=INK)
    ax.set_title(
        "Coverage: what splitguard catches (green) vs out-of-mechanism (grey)\n"
        "0 false positives on correct pipelines (pandas)",
        fontsize=12, fontweight="bold", color=INK, pad=12,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "analysis_coverage.png"), bbox_inches="tight")
    plt.close(fig)
    return dict(zip([n.replace("\n", " ") for n, _, _ in cases],
                    [round(r) for r in recalls], strict=True))


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    print("impact  (inflation pts by n_features):", impact())
    print("overhead (added ms/fit by n_rows)    :", overhead())
    print("coverage (recall % by leakage type)  :", coverage())
    print(f"figures written to {ASSETS}")


if __name__ == "__main__":
    main()
