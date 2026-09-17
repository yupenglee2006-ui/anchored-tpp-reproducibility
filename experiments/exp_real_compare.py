"""
Paper Figures 8 and 9 -- the semi-synthetic detection panels and the
contamination-strength sweep on a real geochemical background.

New in v3.0.  Before this release these two figures had no generator in the
package: they were produced ad hoc, so `Code Availability` overstated what was
reproducible.  Both are now built from the same tested `semisynth` primitives
as every other real-data result.

Requires the Hall (2018) XRF CSV:

    python experiments/exp_real_compare.py --csv experiments/XRF_dataset.csv

To check the pipeline runs before you have that file, generate a stand-in:

    python experiments/_make_demo_csv.py --out /tmp/demo.csv
    python experiments/exp_real_compare.py --csv /tmp/demo.csv --n-real 4

The stand-in reproduces the column schema and rough covariance of the real
data but is NOT the real data; numbers from it must never be quoted.

Outputs
    figures/fig_real_compare.pdf   three-panel detection comparison
    figures/fig_real_strength.pdf  recovery vs planted contamination strength
    results_real_compare.json      the numbers both figures are drawn from
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from _common import panel_letter, plt, save

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semisynth import (load_csv, build_features, plant_contamination, review,
                       best_sign, CLEAN, HIDDEN, VISIBLE)
from atpp import AnchoredTPP
from atpp.baselines import pca_scores, single_element

warnings.filterwarnings("ignore")

K = 12              # review-list length == number of hidden samples
M_ANCHOR = 15       # anchors used for the panel figure
STRENGTH = 1.3      # reference planted strength
STRENGTHS = [0.6, 0.8, 1.0, 1.3, 1.6, 2.0]
CHANCE = None       # computed from the data: K * K / n_candidates


# ----------------------------------------------------------------------
# panel figure
# ----------------------------------------------------------------------

def _require_csv(path):
    """Fail with an actionable message instead of a bare FileNotFoundError.

    The Hall (2018) cuttings dataset is third-party and is NOT redistributed in
    this archive.  See the README section "Getting the real dataset".
    """
    import os
    import sys
    if not os.path.exists(path):
        sys.exit(
            "\nThe semi-synthetic experiments need the published cuttings XRF\n"
            "dataset of Hall (2018), CSEG Recorder, which is not redistributed\n"
            "in this archive for licensing reasons.\n\n"
            f"  looked for: {path}\n\n"
            "Obtain it from the source cited in the README, then re-run with\n"
            "  --csv /path/to/XRF_dataset.csv\n\n"
            "To check the pipeline executes without it:\n"
            "  python experiments/_make_demo_csv.py --out /tmp/demo.csv\n"
            "  python experiments/exp_real_compare.py --csv /tmp/demo.csv --n-real 4\n"
            "Numbers from the stand-in must never be quoted.\n")
    return path


def _scatter(ax, Y, truth, flagged, anchors=None, legend=False):
    """Common scatter: unaffected, hidden contamination, analyst-flagged."""
    styles = [(CLEAN, "lightgrey", "o", 12, "unaffected"),
              (HIDDEN, "#ff7f0e", "s", 42, "hidden contamination"),
              (VISIBLE, "#d62728", "^", 42, "analyst-flagged")]
    for code, col, mk, sz, lab in styles:
        m = truth == code
        if m.any():
            ax.scatter(Y[m, 0], Y[m, 1], c=col, marker=mk, s=sz, alpha=.8,
                       edgecolors="k", linewidth=.35, label=lab)
    if anchors is not None:
        ax.scatter(Y[anchors, 0], Y[anchors, 1], facecolors="none",
                   edgecolors="lime", s=120, linewidth=1.5,
                   label="dragged anchors", zorder=5)
    if legend:
        ax.legend(fontsize=6.5, loc="upper left", framealpha=.92)


def panel_figure(raw, seed=0):
    """Single representative realisation, three detection strategies."""
    df, truth, vis, hid = plant_contamination(raw, strength=STRENGTH, seed=seed)
    X, nm = build_features(df, use_clr=True)
    anchors = vis[:M_ANCHOR]

    # (a) two-element view: the univariate rule a geologist would reach for
    k_idx, al_idx = nm.index("K2O"), nm.index("Al2O3")
    r_1el = best_sign(single_element(X, k_idx), truth, K)

    # (b) blind PCA
    P = pca_scores(X)
    r_pca = best_sign(P[:, 0], truth, K)

    # (c) anchored TPP
    eng = AnchoredTPP(standardise=False, missing="error")
    eng.fit_data(X, feature_names=nm).fit(anchors, seed=seed)
    Y = eng.Y_
    r_tpp = review(eng.scores(), truth, K)

    # 2x2 at 174 mm: three data panels plus a legend cell.  Panels are keyed
    # by letter, not by position -- see _common.panel_letter.
    fig, ax2 = plt.subplots(2, 2, figsize=(6.85, 6.2))
    ax = [ax2[0, 0], ax2[0, 1], ax2[1, 0]]
    legend_ax = ax2[1, 1]
    for a in ax:
        a.grid(alpha=.3)

    _scatter(ax[0], np.column_stack([X[:, k_idx], X[:, al_idx]]), truth, None)
    ax[0].set_xlabel("K2O (clr, std)")
    ax[0].set_ylabel("Al2O3 (clr, std)")
    ax[0].set_title(f"Single element (K2O vs Al2O3)\n"
                    f"found {r_1el['found']}/{K}  |  false-pos "
                    f"{K - r_1el['found']}", fontsize=10)

    _scatter(ax[1], P[:, :2], truth, None)
    ax[1].set_xlabel("PC1")
    ax[1].set_ylabel("PC2")
    ax[1].set_title(f"Blind PCA\nfound {r_pca['found']}/{K}  |  false-pos "
                    f"{K - r_pca['found']}", fontsize=10)

    _scatter(ax[2], Y, truth, None, anchors=anchors, legend=False)
    ax[2].set_xlabel("TPP axis 1")
    ax[2].set_ylabel("TPP axis 2")
    ax[2].set_title(f"Anchored TPP ({M_ANCHOR} anchors)\n"
                    f"found {r_tpp['found']}/{K}  |  false-pos "
                    f"{K - r_tpp['found']}", fontsize=10)

    for a, L in zip(ax, ("(a)", "(b)", "(c)")):
        panel_letter(a, L)

    handles, labels = ax[2].get_legend_handles_labels()
    legend_ax.axis("off")
    legend_ax.legend(handles, labels, loc="center", fontsize=8.5,
                     frameon=True, framealpha=.95, title="sample classes")
    fig.tight_layout()
    save(fig, "fig_real_compare.pdf")
    return {"seed": seed, "m": M_ANCHOR, "strength": STRENGTH,
            "single_element_K2O": int(r_1el["found"]),
            "blind_pca": int(r_pca["found"]),
            "anchored_tpp": int(r_tpp["found"])}


# ----------------------------------------------------------------------
# strength sweep
# ----------------------------------------------------------------------
def strength_sweep(raw, n_real):
    """Recovery vs planted strength, for clr and raw-wt% feature spaces."""
    out = {k: [] for k in ("tpp_clr", "tpp_raw", "k2o", "pca")}
    for st in STRENGTHS:
        acc = {k: [] for k in out}
        for s in range(n_real):
            df, truth, vis, _ = plant_contamination(raw, strength=st, seed=s)
            anchors = vis[:M_ANCHOR]

            X, nm = build_features(df, use_clr=True)
            eng = AnchoredTPP(standardise=False, missing="error")
            eng.fit_data(X, feature_names=nm).fit(anchors, seed=s)
            acc["tpp_clr"].append(review(eng.scores(), truth, K)["found"])
            acc["k2o"].append(best_sign(single_element(X, nm.index("K2O")),
                                        truth, K)["found"])
            acc["pca"].append(best_sign(pca_scores(X)[:, 0], truth, K)["found"])

            Xr, nmr = build_features(df, use_clr=False)
            engr = AnchoredTPP(standardise=False, missing="error")
            engr.fit_data(Xr, feature_names=nmr).fit(anchors, seed=s)
            acc["tpp_raw"].append(review(engr.scores(), truth, K)["found"])

        for k in out:
            out[k].append(float(np.mean(acc[k])))
    return out


def strength_figure(sweep, chance):
    series = [("tpp_clr", "Anchored TPP (clr)", "o", "-", "#1f77b4"),
              ("tpp_raw", "Anchored TPP (raw wt%)", "s", "--", "#17becf"),
              ("k2o", "Single-element K2O", "^", ":", "#7f7fbf"),
              ("pca", "Blind PCA", "v", ":", "#ff7f0e")]
    fig, ax = plt.subplots(figsize=(6.85, 4.3))   # 174 mm
    for key, lab, mk, ls, col in series:
        ax.plot(STRENGTHS, sweep[key], marker=mk, linestyle=ls, color=col,
                linewidth=2, markersize=7, label=lab)
    ax.axhline(chance, color="grey", linewidth=1)
    ax.text(STRENGTHS[-1], chance, " chance", va="bottom", ha="right",
            fontsize=8, color="grey")
    ax.set_xlabel("planted contamination strength (relative)")
    ax.set_ylabel(f"hidden samples recovered\n({K}-sample review list)")
    ax.set_ylim(0, K)
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    save(fig, "fig_real_strength.pdf")


# ----------------------------------------------------------------------
def main(csv, n_real, seed):
    raw = load_csv(csv)
    n_candidates = len(raw) - 25          # visible anchors excluded from ranking
    chance = K * K / n_candidates

    print(f"real background: n={len(raw)}, candidates={n_candidates}, "
          f"chance={chance:.2f}")

    print("\npanel figure ...")
    panel = panel_figure(raw, seed=seed)
    print("   ", panel)

    print(f"\nstrength sweep ({n_real} realisations per level) ...")
    sweep = strength_sweep(raw, n_real)
    strength_figure(sweep, chance)
    for k, v in sweep.items():
        print(f"    {k:9s}", " ".join(f"{x:5.2f}" for x in v))

    res = {"k": K, "m": M_ANCHOR, "n_real": n_real, "chance": round(chance, 3),
           "strengths": STRENGTHS, "panel": panel, "sweep": sweep}
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "results_real_compare.json"), "w") as fh:
        json.dump(res, fh, indent=2)
    print("\nwrote results_real_compare.json")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True, help="Hall (2018) XRF CSV")
    ap.add_argument("--n-real", type=int, default=12,
                    help="realisations per strength level (paper: 12)")
    ap.add_argument("--seed", type=int, default=0,
                    help="realisation shown in the panel figure")
    a = ap.parse_args()
    _require_csv(a.csv)
    main(a.csv, a.n_real, a.seed)
