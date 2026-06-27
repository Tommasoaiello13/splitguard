"""Generate the splitguard logo: a shield (the guard) split into two halves (train | test).

Run:  python tools/make_logo.py   ->   assets/logo.png
"""

from __future__ import annotations

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

ASSETS = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets")
INK = "#0f172a"
TRAIN = "#dbeafe"  # faint blue
TEST = "#dcfce7"  # faint green
ACCENT = "#e11d48"  # the "leak" being stopped


def main() -> None:
    os.makedirs(ASSETS, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4, 4), dpi=160)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.axis("off")

    # shield silhouette (flat top, tapering to a point) split down the middle
    left = [(0.20, 0.84), (0.50, 0.84), (0.50, 0.14), (0.20, 0.52)]
    right = [(0.50, 0.84), (0.80, 0.84), (0.80, 0.52), (0.50, 0.14)]
    outline = [(0.20, 0.84), (0.80, 0.84), (0.80, 0.52), (0.50, 0.14), (0.20, 0.52)]

    ax.add_patch(Polygon(left, closed=True, facecolor=TRAIN, edgecolor="none", zorder=1))
    ax.add_patch(Polygon(right, closed=True, facecolor=TEST, edgecolor="none", zorder=1))
    ax.add_patch(Polygon(outline, closed=True, fill=False, edgecolor=INK, lw=7,
                         joinstyle="round", zorder=3))
    # the split line (the boundary that must hold)
    ax.plot([0.50, 0.50], [0.14, 0.84], color=INK, lw=5, zorder=3)
    # a small mark: a "leak" drop stopped at the split
    ax.plot(0.50, 0.62, marker="o", markersize=11, color=ACCENT, zorder=4)
    ax.plot(0.50, 0.62, marker="o", markersize=4, color="white", zorder=5)

    fig.savefig(os.path.join(ASSETS, "logo.png"), bbox_inches="tight", transparent=True)
    plt.close(fig)
    print(f"wrote {os.path.join(ASSETS, 'logo.png')}")


if __name__ == "__main__":
    main()
