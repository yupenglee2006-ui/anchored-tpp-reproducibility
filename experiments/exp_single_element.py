"""
NEW IN v2.2 -- benchmark integrity check.  Read the output before quoting the
fully synthetic results anywhere.

The paper's headline comparison is against a single-element BARIUM criterion,
chosen a priori because the planted confounder (natural barite) was designed to
defeat it.  It does defeat it: Ba recovers 0 of 15.

But nothing in the design defeats a single-element CHLORINE criterion.  In the
generator, Cl receives no natural geological loading at all -- its only variance
is the 0.35 noise floor -- so a planted +2.2 / +1.32 shift makes it a ~4-sigma
pure marker of contamination.

An analyst does not need labels to find that.  Flag a few samples, look at which
element they are most obviously enriched in, sort on it.  That rule
(`single_element_anchor_selected`) uses exactly the information anchored TPP
uses, and on the published benchmark it OUTPERFORMS anchored TPP.

This script quantifies the problem and shows what repairing it costs.  Giving Cl
realistic down-hole salinity variation (`cl_natural_sd`) removes the shortcut,
but makes the benchmark much harder for everything, so contamination strength
has to rise to keep the comparison informative.

Bottom line for interpretation: the fully synthetic benchmark supports the claim
that anchored TPP beats BLIND and A-PRIORI-SINGLE-ELEMENT methods.  It does not
support a claim of superiority over methods that also use the anchors.

RESOLVED IN v2.3.  With the real dataset available, `exp_real_semisup.py` shows
the ordering REVERSES on real background: the anchor-selected single-element rule
recovers 3.93 +- 0.18 of 12 against anchored TPP's 7.03 +- 0.26 (paired
p < 1e-4).  The shortcut documented below is a property of this generator, not of
the problem.  The real-background experiment is the primary evidence; this script
remains as the audit that established the synthetic benchmark's limits.
"""
import warnings

import numpy as np
from _common import plt, save

from atpp import AnchoredTPP, generate_core, review_list, VISIBLE_MUD, IDX
from atpp.baselines import (single_element, single_element_anchor_selected,
                            single_element_oracle, pu_logistic_scores)

warnings.filterwarnings("ignore")

GRID = [(0.0, 1.0), (0.7, 1.5), (1.0, 1.5), (1.0, 2.0), (1.4, 2.0)]
N_REAL = 30
M = 20


def natural_spread():
    """Report the natural (pre-contamination) standard deviation per element."""
    X0, _ = generate_core(seed=42, contamination_strength=0.0)
    return {e: float(X0[:, i].std()) for e, i in IDX.items()}


def main():
    sd = natural_spread()
    print("  natural s.d. of key channels (contamination_strength=0):")
    for e in ["Si", "K", "Sr", "Ba", "S", "Cl", "Na"]:
        note = "  <-- no geological loading" if sd[e] < 0.4 else ""
        print(f"    {e:<3s} {sd[e]:.3f}{note}")
    print()

    rows = []
    for cl_sd, strength in GRID:
        acc = {k: [] for k in ("Ba", "anchor1e", "oracle1e", "TPP", "PU")}
        chosen = []
        for s in range(N_REAL):
            X, truth = generate_core(seed=2000 + s, cl_natural_sd=cl_sd,
                                     contamination_strength=strength)
            rng = np.random.default_rng(s)
            anchors = rng.choice(np.where(truth == VISIBLE_MUD)[0], M, replace=False)
            eng = AnchoredTPP(missing="zero").fit_data(X).fit(anchors, seed=s)
            Xs = eng.Xs_
            acc["TPP"].append(review_list(eng.scores(), truth, 15)["found_hidden_mud"])
            acc["Ba"].append(
                review_list(single_element(Xs, IDX["Ba"]), truth, 15)["found_hidden_mud"])
            sc, j = single_element_anchor_selected(Xs, anchors, return_index=True)
            chosen.append(j)
            acc["anchor1e"].append(review_list(sc, truth, 15)["found_hidden_mud"])
            best, _ = single_element_oracle(Xs, truth, review_list, 15)
            acc["oracle1e"].append(best)
            acc["PU"].append(
                review_list(pu_logistic_scores(Xs, anchors), truth, 15)["found_hidden_mud"])
        from atpp import ELEMENTS
        pick = ELEMENTS[max(set(chosen), key=chosen.count)]
        rows.append((cl_sd, strength, {k: float(np.mean(v)) for k, v in acc.items()}, pick))

    print("  recovery of 15 hidden samples, mean of "
          f"{N_REAL} realisations, m={M} anchors")
    print(f"  {'Cl s.d.':>8} {'strength':>9} | {'Ba':>6} {'anchor-1el':>11} "
          f"{'oracle-1el':>11} {'TPP':>6} {'PU':>6}   element picked")
    for cl_sd, strength, a, pick in rows:
        print(f"  {cl_sd:8.1f} {strength:9.1f} | {a['Ba']:6.1f} {a['anchor1e']:11.1f} "
              f"{a['oracle1e']:11.1f} {a['TPP']:6.1f} {a['PU']:6.1f}   {pick}")

    print()
    print("  READ THIS: on the published configuration (Cl s.d. 0.0, strength 1.0)")
    print("  the label-free anchor-selected single-element rule beats anchored TPP.")
    print("  It selects Cl, which the generator left as a pure marker. Anchored TPP")
    print("  overtakes it only once Cl is given realistic natural variance.")
    print("  PU logistic regression outperforms anchored TPP in every configuration.")

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = np.arange(len(rows))
    w = 0.2
    for k, off, c, lab in [("Ba", -1.5, "#7f7f7f", "single-element Ba (a priori)"),
                           ("anchor1e", -0.5, "#9467bd", "single-element, anchor-selected"),
                           ("TPP", 0.5, "#1f77b4", "anchored TPP"),
                           ("PU", 1.5, "#2ca02c", "PU logistic")]:
        ax.bar(x + off * w, [r[2][k] for r in rows], w, color=c, label=lab)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Cl sd {a}\nstr {b}" for a, b, _, _ in rows], fontsize=7.5)
    ax.set_ylabel("hidden samples recovered\n(15-sample review list)")
    ax.set_ylim(0, 15)
    ax.legend(fontsize=7.5)
    ax.set_title("Benchmark integrity: is a single element enough?", fontsize=10)
    plt.tight_layout()
    save(fig, "fig_single_element_check.pdf")


if __name__ == "__main__":
    print("Benchmark integrity check -- single-element baselines (NEW in v2.2)")
    main()
