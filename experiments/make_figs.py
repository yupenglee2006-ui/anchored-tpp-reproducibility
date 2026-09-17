"""
Fully synthetic benchmark figures.

Outputs, and where they land in the manuscript (v4.0):
    fig_compare.pdf   paper Fig. 3   detection comparison
    fig_loadings.pdf  paper Fig. 4   recovered loading vector
    fig_robust.pdf    paper Fig. 5   anchor-misspecification robustness

Figures are referred to by filename, not by number, throughout this package:
paper numbering has changed twice during revision and will change again.

Changed in v2.2: this script no longer exec()s cells out of the notebook by
index.  v2.0 did, which meant (a) reordering or inserting a notebook cell
silently broke figure generation, and (b) producing static PDFs required
ipywidgets to be installed, since the notebook's setup cell imports it.  Both
now import the `atpp` package instead, and the notebook does the same, so the
two cannot drift apart.
"""
import numpy as np
from _common import panel_letter, plt, save

from atpp import (AnchoredTPP, ELEMENTS, IDX, generate_core, review_list,
                  best_sign_review, CLEAN, HIDDEN_MUD, VISIBLE_MUD,
                  NATURAL_BARITE)
from atpp.baselines import pca_scores, single_element, single_element_anchor_selected

STYLES = [(CLEAN, "lightgrey", "o", 10, "clean formation"),
          (NATURAL_BARITE, "#1f77b4", "s", 30, "natural barite (confounder)"),
          (HIDDEN_MUD, "#ff7f0e", "s", 48, "hidden contamination"),
          (VISIBLE_MUD, "#d62728", "^", 48, "analyst-flagged")]


def panel(ax, Y, truth, anchors=None, legend=False):
    for code, col, mk, sz, lab in STYLES:
        m = truth == code
        if m.any():
            ax.scatter(Y[m, 0], Y[m, 1], c=col, marker=mk, s=sz, alpha=.75,
                       edgecolors="k", linewidth=.35, label=lab)
    if anchors is not None:
        ax.scatter(Y[anchors, 0], Y[anchors, 1], facecolors="none",
                   edgecolors="lime", s=130, linewidth=1.6,
                   label="dragged anchors", zorder=5)
    if legend:
        ax.legend(fontsize=6.5, loc="upper left", framealpha=.92)


def main():
    X, truth = generate_core(seed=42)
    eng = AnchoredTPP(missing="zero").fit_data(X, feature_names=ELEMENTS)
    flagged = np.where(truth == VISIBLE_MUD)[0]
    eng.fit(flagged[:20], seed=0)
    Xs = eng.Xs_

    ba = single_element(Xs, IDX["Ba"])
    su = single_element(Xs, IDX["S"])
    pcs = pca_scores(Xs)

    # Baselines get the better of the two signs -- a deliberate handicap to us.
    r_ba = best_sign_review(ba, truth, 15)
    r_pc = best_sign_review(pcs[:, 0], truth, 15)
    r_tp = review_list(eng.scores(), truth, 15)

    # ---------- three-way comparison ----------
    # 2x2 at 174 mm: three data panels plus a legend cell,
    # so that lettering survives typesetting at ~9 pt instead of being scaled
    # down to ~5 pt from a 290 mm-wide 1x3 strip.
    fig, axes2 = plt.subplots(2, 2, figsize=(6.85, 6.0))
    axes = [axes2[0, 0], axes2[0, 1], axes2[1, 0]]
    legend_ax = axes2[1, 1]
    panel(axes[0], np.column_stack([ba, su]), truth)
    axes[0].set_xlabel("Ba (standardised)"); axes[0].set_ylabel("S (standardised)")
    axes[0].set_title(f"Single element (Ba)\nfound {r_ba['found_hidden_mud']}/15  |  "
                      f"barite false-pos {r_ba['false_pos_barite']}", fontweight="bold")

    panel(axes[1], pcs, truth)
    axes[1].set_xlabel("PC1"); axes[1].set_ylabel("PC2")
    axes[1].set_title(f"Blind PCA\nfound {r_pc['found_hidden_mud']}/15  |  "
                      f"barite false-pos {r_pc['false_pos_barite']}", fontweight="bold")

    panel(axes[2], eng.Y_, truth, anchors=eng.anchor_idx_, legend=False)
    axes[2].set_xlabel("TPP axis 1"); axes[2].set_ylabel("TPP axis 2")
    axes[2].set_title(f"Anchored TPP (20 anchors)\nfound {r_tp['found_hidden_mud']}/15  |  "
                      f"barite false-pos {r_tp['false_pos_barite']}", fontweight="bold")

    for a, L in zip(axes, ("(a)", "(b)", "(c)")):
        panel_letter(a, L)

    # fourth cell carries the shared legend, freeing the data panels of it
    handles, labels = axes[2].get_legend_handles_labels()
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", fontsize=8.5,
                     frameon=True, framealpha=.95, title="sample classes")
    plt.tight_layout()
    save(fig, "fig_compare.pdf")

    # ---------- robustness to anchor misspecification ----------
    clean_pool = np.where(truth == CLEAN)[0]
    fracs = np.arange(0, 0.85, 0.05)
    mu, sd = [], []
    for f in fracs:
        trials = []
        for s in range(12):
            rng = np.random.default_rng(s)
            a = list(rng.permutation(flagged)[:20])
            n_bad = int(round(f * len(a)))
            if n_bad:
                a[:n_bad] = list(rng.choice(clean_pool, n_bad, replace=False))
            eng.fit(np.array(a), seed=s)
            trials.append(review_list(eng.scores(), truth, 15)["found_hidden_mud"])
        mu.append(np.mean(trials)); sd.append(np.std(trials))
    mu, sd = np.array(mu), np.array(sd)

    fig, ax = plt.subplots(figsize=(6.85, 3.8))   # 174 mm
    ax.plot(fracs * 100, mu, "o-", color="#2c7fb8", lw=2, ms=4.5, label="Anchored TPP")
    ax.fill_between(fracs * 100, mu - sd, mu + sd, color="#2c7fb8", alpha=.2,
                    label=r"$\pm$1 s.d. over 12 draws")
    ax.axhline(r_pc["found_hidden_mud"], color="#d95f02", ls="--", lw=1.6,
               label="blind PCA")
    ax.set_xlabel("% of dragged anchors that were actually wrong")
    ax.set_ylabel("hidden samples recovered\n(15-sample review list)")
    ax.legend(fontsize=8)
    plt.tight_layout()
    save(fig, "fig_robust.pdf")
    print(f"  robustness at 0/25/50/75%: "
          f"{[round(mu[i], 1) for i in [0, 5, 10, 15]]}")

    # ---------- recovered loading vector ----------
    eng.fit(flagged[:20], seed=0)
    load = eng.loadings(top=10)
    names = [e for e, _ in load][::-1]
    vals = [w for _, w in load][::-1]
    cols = ["#2c7fb8" if v > 0 else "#d95f02" for v in vals]
    fig, ax = plt.subplots(figsize=(5.0, 4.0))
    ax.barh(names, vals, color=cols, edgecolor="k", linewidth=.4)
    ax.axvline(0, color="k", lw=.8)
    ax.set_xlabel("TPP axis-1 loading")
    ax.grid(alpha=.3, axis="x")
    plt.tight_layout()
    save(fig, "fig_loadings.pdf")
    print(f"  loadings: {[(e, round(w, 3)) for e, w in load[:6]]}")

    # ---------- integrity note ----------
    sc = single_element_anchor_selected(Xs, flagged[:20])
    r_1e = review_list(sc, truth, 15)
    print()
    print(f"  NOTE: anchor-selected single-element rule recovers "
          f"{r_1e['found_hidden_mud']}/15 vs anchored TPP "
          f"{r_tp['found_hidden_mud']}/15.")
    print("  Run exp_single_element.py before quoting the synthetic result.")


if __name__ == "__main__":
    print("Main benchmark figures (paper Figs. 2, 3, 8, 9)")
    main()
