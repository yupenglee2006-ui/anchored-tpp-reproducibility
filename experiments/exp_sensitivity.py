"""
Paper Section 3.3, Figure 1 -- hyperparameter invariance.

Demonstrates Proposition 1: the closed-form solution does not depend on the
background weight omega_bg or on the pull magnitude rho, at all.  Panel (c)
contrasts the closed form against direct iterative minimisation of the full
least-squares objective, which DOES depend on omega_bg.
"""
import numpy as np
from _common import plt, save

from atpp import AnchoredTPP, generate_core, review_list, VISIBLE_MUD


def main():
    X, truth = generate_core(seed=42)
    anchors = np.where(truth == VISIBLE_MUD)[0][:20]
    base = AnchoredTPP(missing="zero").fit_data(X).fit(anchors, seed=7)
    W0 = base.W_

    wbs = np.logspace(-4, 0, 17)
    rhos = np.logspace(-1, 2, 16)
    rec_w, dev_w, rec_r, dev_r = [], [], [], []

    for wb in wbs:
        e = AnchoredTPP(missing="zero").fit_data(X).fit(
            anchors, background_weight=wb, seed=7)
        rec_w.append(review_list(e.scores(), truth, 15)["found_hidden_mud"])
        dev_w.append(max(float(np.abs(e.W_ - W0).max()), 1e-17))

    for rh in rhos:
        e = AnchoredTPP(missing="zero").fit_data(X).fit(
            anchors, pull=rh, jitter=0.3 * rh / 3.0, seed=7)
        rec_r.append(review_list(e.scores(), truth, 15)["found_hidden_mud"])
        dev_r.append(max(float(np.abs(e.W_ - W0).max()), 1e-17))

    wb_ex = np.logspace(-3, -0.3, 7)
    rec_ex, cos_ex = [], []
    for wb in wb_ex:
        e = AnchoredTPP(missing="zero").fit_data(X).fit_least_squares(
            anchors, background_weight=wb, seed=7)
        rec_ex.append(review_list(e.scores(), truth, 15)["found_hidden_mud"])
        cos_ex.append(float(abs(e.W_[:, 0] @ W0[:, 0])))

    # 2x2 at 174 mm (Springer max width); the fourth cell is unused because
    # panels (a) and (b) carry twin axes that do not share a legend
    fig, _axg = plt.subplots(2, 2, figsize=(6.85, 5.6))
    ax = [_axg[0, 0], _axg[0, 1], _axg[1, 0]]
    _axg[1, 1].axis("off")
    for a, xs, rec, dev, lab, ttl in [
            (ax[0], wbs, rec_w, dev_w, r"background weight $\omega_{\mathrm{bg}}$",
             r"(a) invariance to $\omega_{\mathrm{bg}}$"),
            (ax[1], rhos, rec_r, dev_r, r"pull magnitude $\rho$",
             r"(b) invariance to $\rho$")]:
        a.semilogx(xs, rec, "o-", color="#1f77b4")
        a.set_xlabel(lab)
        a.set_ylim(0, 15)
        a.set_title(ttl)
        a.set_ylabel("hidden samples recovered")
        t = a.twinx()
        t.semilogx(xs, dev, "s--", color="#d62728", ms=3)
        t.set_yscale("log")
        t.set_ylim(1e-18, 1e-8)
        t.grid(False)
        t.set_ylabel(r"$\max|\hat W-\hat W_0|$", color="#d62728")

    ax[2].semilogx(wb_ex, rec_ex, "o-", color="#2ca02c",
                   label="iterative residual min.")
    ax[2].axhline(rec_w[0], ls="--", color="#1f77b4",
                  label=r"closed form (all $\omega_{bg}$)")
    ax[2].set_xlabel(r"$\omega_{\mathrm{bg}}$")
    ax[2].set_ylim(0, 15)
    ax[2].set_title("(c) alignment vs residual objective")
    ax[2].set_ylabel("hidden samples recovered")
    ax[2].legend(fontsize=7, loc="lower left")
    plt.tight_layout()
    save(fig, "fig_sensitivity.pdf")

    cos_c = []
    for s in range(16):
        Xa, ta = generate_core(seed=42 + s)
        ea = AnchoredTPP(missing="zero").fit_data(Xa).fit(
            np.where(ta == VISIBLE_MUD)[0][:20], seed=s)
        cos_c.append(ea.centroid_alignment())

    print(f"  omega_bg : recovery {min(rec_w)}-{max(rec_w)}, "
          f"max|dW| = {max(dev_w):.2e}")
    print(f"  rho      : recovery {min(rec_r)}-{max(rec_r)}, "
          f"max|dW| = {max(dev_r):.2e}")
    print(f"  iterative LS recovery {rec_ex}, cos vs closed form "
          f"{min(cos_ex):.3f}-{max(cos_ex):.3f}")
    print(f"  centroid alignment: {np.mean(cos_c):.4f} +- {np.std(cos_c):.4f}")


if __name__ == "__main__":
    print("Sensitivity / invariance experiment (paper Fig. 1)")
    main()
