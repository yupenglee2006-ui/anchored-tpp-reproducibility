"""
Paper Figure 1 -- the anchored TPP workflow diagram.

A schematic, not a result: it takes no data and depends on no experiment.  It
lives here so that every figure in the manuscript has a generator in this
package and none has to be redrawn by hand.

    python experiments/make_fig_workflow.py

Output: figures/fig_workflow.pdf
"""
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _common import plt, FIGDIR

# ----------------------------------------------------------------------
# style
# ----------------------------------------------------------------------
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.grid": False,
    "pdf.fonttype": 42,   # embed TrueType, not Type 3 -- required by Springer production
    "ps.fonttype": 42,
})

FILL = {
    "data":   "#dce6f2",   # muted blue   -- inputs / computation
    "analyst": "#fbe0c4",  # muted orange -- the human step
    "output": "#d8ecd8",   # muted green  -- deliverable
}
EDGE = "#33475b"
ARROW = "#33475b"

# ----------------------------------------------------------------------
# geometry (figure coordinates, inches)
# ----------------------------------------------------------------------
FIG_W, FIG_H = 6.62, 2.24   # tight bbox lands at ~174 mm, Springer max width
BOX_W, BOX_H = 1.56, 0.92
GAP = 0.35
Y_BOX = 1.18                      # bottom edge of the box row
X0 = (FIG_W - (4 * BOX_W + 3 * GAP)) / 2.0

fig = plt.figure(figsize=(FIG_W, FIG_H))
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, FIG_W)
ax.set_ylim(0, FIG_H)
ax.axis("off")

boxes = [
    dict(fill=FILL["data"],
         main="High-dimensional\ngeochemical data",
         sub=r"$X \in \mathbb{R}^{\,n \times p}$"),
    dict(fill=FILL["analyst"],
         main="Analyst drags\n$m \\ll n$ anchors",
         sub="the hypothesis", sub_italic=True),
    dict(fill=FILL["data"],
         main="Closed-form\nprojection",
         sub=r"$\hat{W} = UV^{\top}$"),
    dict(fill=FILL["output"],
         main="Ranked review list\nand loading vector",
         sub=None),
]

centres = []
for i, b in enumerate(boxes):
    x = X0 + i * (BOX_W + GAP)
    cx, cy = x + BOX_W / 2.0, Y_BOX + BOX_H / 2.0
    centres.append((x, cx, cy))

    ax.add_patch(FancyBboxPatch(
        (x, Y_BOX), BOX_W, BOX_H,
        boxstyle="round,pad=0.012,rounding_size=0.075",
        linewidth=0.9, edgecolor=EDGE, facecolor=b["fill"],
        mutation_aspect=1.0, zorder=2))

    has_sub = b["sub"] is not None
    ax.text(cx, cy + (0.13 if has_sub else 0.0), b["main"],
            ha="center", va="center", fontsize=8.6, color="#1b2733",
            linespacing=1.35, zorder=3)
    if has_sub:
        ax.text(cx, cy - 0.22, b["sub"],
                ha="center", va="center", fontsize=8.6, color="#1b2733",
                style="italic" if b.get("sub_italic") else "normal", zorder=3)

# ----------------------------------------------------------------------
# forward arrows
# ----------------------------------------------------------------------
for i in range(3):
    x_start = centres[i][0] + BOX_W
    x_end = centres[i + 1][0]
    y = Y_BOX + BOX_H / 2.0
    ax.add_patch(FancyArrowPatch(
        (x_start + 0.05, y), (x_end - 0.05, y),
        arrowstyle="-|>", mutation_scale=11,
        linewidth=1.0, color=ARROW, shrinkA=0, shrinkB=0, zorder=2))

# ----------------------------------------------------------------------
# dashed refinement loop: box 4 -> down -> left -> up into box 2
# ----------------------------------------------------------------------
y_loop = 0.44
x_from = centres[3][1]
x_to = centres[1][1]

ax.add_patch(FancyArrowPatch(
    (x_from, Y_BOX - 0.02), (x_to, Y_BOX - 0.02),
    connectionstyle=f"bar,fraction=-{(Y_BOX - y_loop) / abs(x_from - x_to):.4f}",
    arrowstyle="-|>", mutation_scale=11, linewidth=1.0,
    linestyle=(0, (4.5, 2.6)), color=ARROW,
    shrinkA=0, shrinkB=0, zorder=1))

ax.text((x_from + x_to) / 2.0, y_loop - 0.16,
        "interactive refinement: revise the anchor set, refit",
        ha="center", va="center", fontsize=8.0, color="#33475b")

# save without bbox_inches="tight": this figure fills its canvas by design, so
# a tight bbox expands it past the 174 mm limit rather than trimming it
import os
_out = os.path.join(FIGDIR, "fig_workflow.pdf")
fig.savefig(_out)
plt.close(fig)
print("  wrote figures/fig_workflow.pdf")
