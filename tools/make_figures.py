"""Generate the README figures from live results (no hardcoded numbers).

Run:  python tools/make_figures.py
Outputs: assets/hero_accuracy.png, assets/comparison.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
INK = "#0f172a"
RED = "#e11d48"
GREEN = "#059669"
SLATE = "#94a3b8"


def _measure():
    rng = np.random.default_rng(0)
    n, p, k = 160, 5000, 20
    X = rng.normal(size=(n, p))
    y = (rng.random(n) > 0.5).astype(int)
    X_tr, X_te, y_tr, y_te = model_selection.train_test_split(X, y, test_size=0.4, random_state=0)

    sel = SelectKBest(f_classif, k=k).fit(X, y)  # leak
    clf = LogisticRegression(max_iter=1000).fit(sel.transform(X_tr), y_tr)
    leaky = accuracy_score(y_te, clf.predict(sel.transform(X_te)))

    sel2 = SelectKBest(f_classif, k=k).fit(X_tr, y_tr)
    clf2 = LogisticRegression(max_iter=1000).fit(sel2.transform(X_tr), y_tr)
    honest = accuracy_score(y_te, clf2.predict(sel2.transform(X_te)))
    return leaky, honest


def _style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)


def fig_hero(leaky: float, honest: float) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    labels = [
        "Leaky pipeline\n(feature selection on full data)",
        "Honest pipeline\n(selection on train only)",
    ]
    bars = ax.bar(labels, [leaky * 100, honest * 100], color=[RED, GREEN], width=0.55)
    for b, v in zip(bars, (leaky, honest), strict=True):
        ax.text(
            b.get_x() + b.get_width() / 2,
            v * 100 + 1.5,
            f"{v:.0%}",
            ha="center",
            va="bottom",
            fontsize=26,
            fontweight="bold",
            color=INK,
        )
    ax.set_ylim(0, 100)
    ax.set_ylabel("Holdout accuracy", fontsize=11, color=INK)
    ax.set_title(
        "Same model, same random-noise data — one score is a lie",
        fontsize=14,
        fontweight="bold",
        color=INK,
        pad=14,
    )
    _style(ax)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "hero_accuracy.png"), bbox_inches="tight")
    plt.close(fig)


def fig_comparison() -> None:
    approaches = [
        "numpy errstate(all='raise')",
        "manual assert no NaN/Inf",
        "schema validation (pandera / GE)",
        "splitguard",
    ]
    detects = [False, False, False, True]
    colors = [SLATE, SLATE, SLATE, GREEN]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    yps = np.arange(len(approaches))[::-1]
    ax.barh(yps, [1] * len(approaches), color=colors, height=0.6)
    for y, ok in zip(yps, detects, strict=True):
        ax.text(
            0.5,
            y,
            "DETECTS LEAK" if ok else "MISSES IT",
            ha="center",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="white",
        )
    ax.set_yticks(yps)
    ax.set_yticklabels(approaches, fontsize=11, color=INK)
    ax.set_xticks([])
    ax.set_xlim(0, 1)
    ax.set_title(
        "Value-inspecting tools miss leakage; splitguard tracks data flow",
        fontsize=13,
        fontweight="bold",
        color=INK,
        pad=14,
    )
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "comparison.png"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    leaky, honest = _measure()
    fig_hero(leaky, honest)
    fig_comparison()
    print(f"Wrote assets to {ASSETS}")
    print(f"  hero_accuracy.png  (leaky {leaky:.1%} vs honest {honest:.1%})")
    print("  comparison.png")


if __name__ == "__main__":
    main()
