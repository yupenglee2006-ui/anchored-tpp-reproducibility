"""Shared setup for experiment scripts."""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

plt.rcParams.update({"font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
                     "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 150,
                     # Springer production rejects Type 3 fonts.  Matplotlib
                     # emits Type 3 by default; 42 selects embedded TrueType.
                     "pdf.fonttype": 42, "ps.fonttype": 42})

FIGDIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                      "figures")
os.makedirs(FIGDIR, exist_ok=True)


def save(fig, name):
    path = os.path.join(FIGDIR, name)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}")
    return path


def panel_letter(ax, letter, x=-0.14, y=1.16, size=11):
    """Stamp a bold panel letter, e.g. "(a)", outside the top-left of an axes.

    MDPI (and most journals) key multi-panel captions to letters rather than to
    positions, because a position word silently becomes wrong the moment the
    layout is rearranged -- which is exactly what happened when these figures
    moved from a 1x3 strip to a 2x2 grid.  Coordinates are in axes fraction, so
    the letter tracks the panel regardless of figure size.
    """
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=size,
            fontweight="bold", va="top", ha="left")
