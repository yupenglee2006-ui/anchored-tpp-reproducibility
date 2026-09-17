"""
Paper Section 5.5, Figure 5 -- spatial autocorrelation and anchor placement.

The published benchmark draws samples independently, which is unrepresentative
of high-resolution core and cuttings scanning.  Here the latent factors follow
an AR(1) process down-hole and the planted populations occupy contiguous
intervals.  Two anchor strategies are compared:

  spread    -- anchors drawn from across several affected intervals
  clustered -- anchors taken as one contiguous run from a single interval

The second is what an analyst naturally does (pick the most obviously stained
run).

FINDING (v2.2, corrected): the penalty for clustered anchors is NON-MONOTONIC in
phi.  It is absent for independent samples, real and resolvable at moderate
autocorrelation (phi ~ 0.7-0.85), and vanishes again at phi = 0.95 -- not because
clustering stops mattering but because recovery collapses for both strategies and
the difference is swamped.  An earlier draft claimed the penalty "grows with
autocorrelation"; that is wrong.

The robust finding is the degradation with autocorrelation itself: recovery falls
by roughly half from independent to phi = 0.95, and realisation-to-realisation
scatter roughly doubles.
"""
import numpy as np
from _common import plt, save

from atpp import AnchoredTPP, generate_core, review_list, VISIBLE_MUD

CONFIGS = [(0.0, 1, "i.i.d."), (0.7, 8, r"$\phi$=0.7"),
           (0.85, 12, r"$\phi$=0.85"), (0.95, 25, r"$\phi$=0.95")]
N_REAL = 80   # raised from 24: the between-strategy difference is
              # small relative to realisation scatter, and 24 draws could not
              # resolve it. Standard ERRORS are reported below, not s.d.
M = 20


def run(phi, block, mode, n_real=N_REAL, m=M):
    vals = []
    for s in range(n_real):
        X, truth = generate_core(seed=100 + s, ar_phi=phi, block=block)
        rng = np.random.default_rng(s)
        vis = np.sort(np.where(truth == VISIBLE_MUD)[0])
        if mode == "spread":
            anchors = rng.choice(vis, m, replace=False)
        else:
            start = int(rng.integers(0, max(1, len(vis) - m)))
            anchors = vis[start:start + m]
        eng = AnchoredTPP(missing="zero").fit_data(X).fit(anchors, seed=s)
        vals.append(review_list(eng.scores(), truth, 15)["found_hidden_mud"])
    vals = np.asarray(vals, dtype=float)
    return float(vals.mean()), float(vals.std()), float(vals.std() / np.sqrt(len(vals)))


def main():
    sp, spsd, cl, clsd = [], [], [], []
    print(f"  ({N_REAL} realisations; +- is the STANDARD ERROR of the mean)")
    for phi, block, lab in CONFIGS:
        a, asd, ase = run(phi, block, "spread")
        c, csd, cse = run(phi, block, "clustered")
        sp.append(a); spsd.append(asd); cl.append(c); clsd.append(csd)
        diff = a - c
        dse = float(np.hypot(ase, cse))
        flag = "significant" if abs(diff) > 2 * dse else "not resolved"
        print(f"  {lab:<12} spread {a:5.2f}+-{ase:.2f}   clustered {c:5.2f}+-{cse:.2f}"
              f"   diff {diff:+5.2f}+-{dse:.2f}  ({flag})")

    fig, ax = plt.subplots(figsize=(5.6, 3.4))
    x = np.arange(len(CONFIGS))
    ax.bar(x - .19, sp, .38, yerr=spsd, capsize=3, color="#1f77b4",
           label="anchors spread over intervals")
    ax.bar(x + .19, cl, .38, yerr=clsd, capsize=3, color="#ff7f0e",
           label="anchors from a single interval")
    ax.set_xticks(x)
    ax.set_xticklabels([c[2] for c in CONFIGS])
    ax.set_xlabel("down-hole AR(1) coefficient of the latent factors")
    ax.set_ylabel("hidden samples recovered\n(15-sample review list)")
    ax.set_ylim(0, 15)
    ax.legend(fontsize=8)
    plt.tight_layout()
    save(fig, "fig_spatial.pdf")


if __name__ == "__main__":
    print("Spatial autocorrelation experiment (paper Fig. 5)")
    main()
