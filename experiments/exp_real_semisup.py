"""
Paper Section 6 -- semi-synthetic comparison against all baselines, plus the
real-data test of the anchor-placement recommendation of Section 5.5.

New in v2.3.  Requires the Hall (2018) XRF CSV.

    python experiments/exp_real_semisup.py --csv experiments/XRF_dataset.csv

This is the experiment the paper's case principally rests on: the background is
real, so every element carries genuine geological variability and no channel is
an accidental pure marker of the planted signature (contrast Section 5.8).
"""
import argparse
import json
import os
import sys
import warnings

import numpy as np
from scipy import stats
from _common import plt, save

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from semisynth import (load_csv, build_features, plant_contamination, review,
                       best_sign)
from atpp import AnchoredTPP
from atpp.baselines import (pca_scores, single_element,
                            single_element_anchor_selected, ocsvm_scores,
                            pu_logistic_scores)

warnings.filterwarnings("ignore")

K = 12          # review-list length = number of hidden samples
STRENGTH = 1.3
METHODS = [("PCA", "blind PCA (PC1)"),
           ("K2O", "single-element K$_2$O (a priori)"),
           ("1el", "single-element, anchor-selected"),
           ("OC", "one-class SVM"),
           ("PU", "PU logistic"),
           ("TPP", "anchored TPP")]



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


def run_panel(raw, n_real, m, use_clr=True):
    acc = {k: [] for k, _ in METHODS}
    for s in range(n_real):
        df, truth, vis, _ = plant_contamination(raw, strength=STRENGTH, seed=s)
        X, nm = build_features(df, use_clr=use_clr)
        a = vis[:m]
        eng = AnchoredTPP(standardise=False, missing="error")
        eng.fit_data(X, feature_names=nm).fit(a, seed=s)
        acc["PCA"].append(best_sign(pca_scores(X)[:, 0], truth, K)["found"])
        acc["K2O"].append(best_sign(single_element(X, nm.index("K2O")), truth, K)["found"])
        acc["1el"].append(review(single_element_anchor_selected(X, a), truth, K)["found"])
        acc["OC"].append(review(ocsvm_scores(X, a), truth, K)["found"])
        acc["PU"].append(review(pu_logistic_scores(X, a), truth, K)["found"])
        acc["TPP"].append(review(eng.scores(), truth, K)["found"])
    return {k: np.asarray(v, float) for k, v in acc.items()}


def main(csv, n_real):
    raw = load_csv(csv)
    R = {"n_real": n_real, "k": K, "strength": STRENGTH}

    print(f"  real background: {len(raw)} samples")
    r_raw = float(np.corrcoef(raw.K2O, raw.Al2O3)[0, 1])
    Xc, nm = build_features(raw, use_clr=True)
    r_clr = float(np.corrcoef(Xc[:, nm.index("K2O")], Xc[:, nm.index("Al2O3")])[0, 1])
    print(f"  K2O-Al2O3 correlation: {r_raw:.3f} (raw wt%), {r_clr:.3f} (clr)")
    R["r_K2O_Al2O3"] = {"raw": round(r_raw, 3), "clr": round(r_clr, 3)}

    # measured down-hole autocorrelation of the real background
    order = np.argsort(raw["Depth"].values)
    ac = [np.corrcoef(Xc[order, j][:-1], Xc[order, j][1:])[0, 1]
          for j in range(Xc.shape[1]) if Xc[:, j].std() > 0]
    R["lag1_autocorr_median"] = round(float(np.median(ac)), 2)
    print(f"  measured lag-1 autocorrelation of clr features: "
          f"{np.median(ac):.2f} (median)")

    # ---- headline comparison, clr and raw ---------------------------
    print(f"\n  headline comparison ({n_real} realisations, m=15, of {K} hidden)")
    for tag, use_clr in [("clr", True), ("raw", False)]:
        acc = run_panel(raw, n_real, m=15, use_clr=use_clr)
        R[tag] = {k: [round(float(v.mean()), 2),
                      round(float(v.std() / np.sqrt(n_real)), 2)]
                  for k, v in acc.items()}
        if use_clr:
            acc_clr = acc
        print(f"    --- {tag} ---")
        for k, lab in METHODS:
            print(f"      {lab:<34} {acc[k].mean():5.2f} +- "
                  f"{acc[k].std() / np.sqrt(n_real):.2f}")
    R["clr_benefit_factor"] = round(R["clr"]["TPP"][0] / R["raw"]["TPP"][0], 2)
    print(f"    clr benefit factor: {R['clr_benefit_factor']:.2f}x")

    # ---- paired tests against anchored TPP --------------------------
    print("\n  paired differences (anchored TPP minus each), clr:")
    R["paired"] = {}
    t = acc_clr["TPP"]
    for k, lab in METHODS:
        if k == "TPP":
            continue
        d = t - acc_clr[k]
        p = float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.0
        R["paired"][k] = {"diff": round(float(d.mean()), 2),
                          "se": round(float(d.std() / np.sqrt(n_real)), 2),
                          "p": round(p, 5)}
        print(f"    TPP - {lab:<34} {d.mean():+5.2f} +- "
              f"{d.std() / np.sqrt(n_real):.2f}   p={p:.4f}")

    # ---- anchor sweep ------------------------------------------------
    print("\n  anchor sweep (clr):")
    ms = [1, 2, 3, 5, 8, 12, 15, 18, 25]
    sweep = {k: [] for k in ("TPP", "PU", "OC", "1el")}
    sweep_p = []
    for m in ms:
        acc = run_panel(raw, n_real, m=m, use_clr=True)
        for k, src in [("TPP", "TPP"), ("PU", "PU"), ("OC", "OC"), ("1el", "1el")]:
            sweep[k].append(round(float(acc[src].mean()), 2))
        d = acc["TPP"] - acc["PU"]
        p = float(stats.wilcoxon(d).pvalue) if np.any(d) else 1.0
        sweep_p.append(round(p, 4))
        print(f"    m={m:<3d} TPP {acc['TPP'].mean():5.2f}  PU {acc['PU'].mean():5.2f}  "
              f"OCSVM {acc['OC'].mean():5.2f}  1el {acc['1el'].mean():5.2f}   "
              f"TPP-PU {d.mean():+5.2f} (p={p:.3f})")
    R["sweep"] = {"m": ms, **sweep, "tpp_minus_pu_p": sweep_p}

    # ---- anchor placement on real background -------------------------
    print("\n  anchor placement, contiguous contamination (block=10), m=15:")
    rankpos = {p: i for i, p in enumerate(order)}
    place = {"spread": [], "clustered": []}
    for s in range(max(n_real, 100)):
        df, truth, vis, _ = plant_contamination(raw, strength=STRENGTH, seed=s, block=10)
        X, nm2 = build_features(df, use_clr=True)
        vs = np.array(sorted(vis, key=lambda p: rankpos[p]))
        for mode in place:
            rng = np.random.default_rng(s)
            a = (rng.choice(vs, 15, replace=False) if mode == "spread"
                 else vs[int(rng.integers(0, max(1, len(vs) - 15))):][:15])
            eng = AnchoredTPP(standardise=False, missing="error")
            eng.fit_data(X, feature_names=nm2).fit(a, seed=s)
            place[mode].append(review(eng.scores(), truth, K)["found"])
    sp = np.asarray(place["spread"], float)
    cl = np.asarray(place["clustered"], float)
    d = sp - cl
    n_pl = len(sp)
    R["placement"] = {"n": n_pl,
                      "spread": [round(float(sp.mean()), 2), round(float(sp.std() / np.sqrt(n_pl)), 2)],
                      "clustered": [round(float(cl.mean()), 2), round(float(cl.std() / np.sqrt(n_pl)), 2)],
                      "diff": round(float(d.mean()), 2),
                      "se": round(float(d.std() / np.sqrt(n_pl)), 2),
                      "p": round(float(stats.wilcoxon(d).pvalue), 6)}
    print(f"    spread {sp.mean():5.2f} +- {sp.std() / np.sqrt(n_pl):.2f}   "
          f"clustered {cl.mean():5.2f} +- {cl.std() / np.sqrt(n_pl):.2f}   "
          f"diff {d.mean():+.2f} +- {d.std() / np.sqrt(n_pl):.2f}  "
          f"p={R['placement']['p']:.5f}")

    # ---- loading reproducibility -------------------------------------
    W = []
    for s in range(n_real):
        df, truth, vis, _ = plant_contamination(raw, strength=STRENGTH, seed=s)
        X, nm2 = build_features(df, use_clr=True)
        eng = AnchoredTPP(standardise=False, missing="error")
        eng.fit_data(X, feature_names=nm2).fit(vis[:15], seed=s)
        w = eng.W_[:, 0]
        W.append(w * np.sign(w[nm2.index("K2O")]))   # fix sign by a planted element
    W = np.array(W)
    mu, sd = W.mean(0), W.std(0)
    ordr = np.argsort(-np.abs(mu))[:9]
    R["loadings"] = {nm2[i]: [round(float(mu[i]), 2), round(float(sd[i]), 2)] for i in ordr}
    print("\n  mean loadings over realisations (sign-aligned):")
    for i in ordr:
        print(f"    {nm2[i]:<20} {mu[i]:+.2f} +- {sd[i]:.2f}")
    R["Al2O3_rank"] = int(list(np.argsort(-np.abs(mu))).index(nm2.index("Al2O3")) + 1)
    print(f"    Al2O3 ranks {R['Al2O3_rank']} by |loading|")

    # ---- figure -------------------------------------------------------
    # stacked at 174 mm (Springer max width) rather than a 252 mm-wide pair:
    # the bar labels in (a) are long and do not survive being scaled down
    fig, ax = plt.subplots(2, 1, figsize=(6.85, 7.2))
    labs = [lab for _, lab in METHODS]
    vals = [R["clr"][k][0] for k, _ in METHODS]
    errs = [R["clr"][k][1] for k, _ in METHODS]
    cols = ["#7f7f7f", "#7f7f7f", "#9467bd", "#9467bd", "#2ca02c", "#1f77b4"]
    ax[0].barh(range(len(labs)), vals, xerr=errs, color=cols, edgecolor="k", linewidth=.4)
    ax[0].set_yticks(range(len(labs)))
    ax[0].set_yticklabels(labs, fontsize=8)
    ax[0].invert_yaxis()
    ax[0].axvline(0.59, ls=":", color="k", lw=1)
    ax[0].text(0.7, len(labs) - 0.4, "chance", fontsize=7, color="k")
    ax[0].set_xlabel(f"hidden samples recovered (of {K})")
    ax[0].set_xlim(0, 12)
    ax[0].set_title("(a) real background, $m=15$ anchors", fontsize=10)

    for k, c, mk, lab in [("TPP", "#1f77b4", "o", "anchored TPP"),
                          ("PU", "#2ca02c", "s", "PU logistic"),
                          ("OC", "#9467bd", "^", "one-class SVM"),
                          ("1el", "#8c564b", "v", "single-element, anchor-selected")]:
        ax[1].plot(ms, sweep[k], mk + "-", color=c, label=lab)
    ax[1].set_xlabel("number of anchor points $m$")
    ax[1].set_ylabel(f"hidden samples recovered (of {K})")
    ax[1].set_ylim(0, 12)
    ax[1].legend(fontsize=7.5, loc="upper left")
    ax[1].set_title("(b) anchor economy on real background", fontsize=10)
    plt.tight_layout()
    save(fig, "fig_real_semisup.pdf")

    out = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                       "results_semisynth_v23.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"  wrote {os.path.basename(out)}")


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=os.path.join(here, "XRF_dataset.csv"))
    ap.add_argument("--n-real", type=int, default=60)
    a = ap.parse_args()
    _require_csv(a.csv)
    print("Semi-synthetic comparison on real background (paper Section 6)")
    main(a.csv, a.n_real)
