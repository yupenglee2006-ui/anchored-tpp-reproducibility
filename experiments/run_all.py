"""
Run every experiment and record the numbers the paper quotes.

    python experiments/run_all.py            # everything except semi-synthetic
    python experiments/run_all.py --csv X.csv  # include semi-synthetic

Writes figures/ and results_v30.json.  If a number in the manuscript disagrees
with results_v30.json, the manuscript is wrong -- regenerate, do not edit the
JSON.

Scripts marked SUPPORTING below produce results that are no longer reported in
the manuscript.  They are retained because they were run, and because a
reviewer may ask; do not cite figures from them as paper figures.
"""
import argparse
import json
import os
import subprocess
import sys
import time
import warnings

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

warnings.filterwarnings("ignore")


def collect():
    """Recompute the quoted numbers directly, so the JSON is self-contained."""
    from atpp import (AnchoredTPP, generate_core, review_list, best_sign_review,
                      VISIBLE_MUD, IDX, ELEMENTS)
    from atpp.baselines import (pca_scores, single_element, mcd_scores,
                                single_element_anchor_selected,
                                single_element_oracle, isolation_forest_scores,
                                ocsvm_scores, pu_logistic_scores)
    R = {}

    # ---- headline detection result -----------------------------------
    X, truth = generate_core(seed=42)
    eng = AnchoredTPP(missing="zero").fit_data(X, feature_names=ELEMENTS)
    anchors = np.where(truth == VISIBLE_MUD)[0][:20]
    eng.fit(anchors, seed=0)
    Xs = eng.Xs_
    R["headline"] = {
        "tpp": review_list(eng.scores(), truth, 15)["found_hidden_mud"],
        "ba": best_sign_review(single_element(Xs, IDX["Ba"]), truth, 15)["found_hidden_mud"],
        "pca": best_sign_review(pca_scores(Xs)[:, 0], truth, 15)["found_hidden_mud"],
        "loadings": {e: round(w, 3) for e, w in eng.loadings(top=6)},
    }

    # ---- hyperparameter invariance ----------------------------------------
    W0 = eng.W_
    dw = dr = 0.0
    recs = set()
    for wb in np.logspace(-4, 0, 17):
        e = AnchoredTPP(missing="zero").fit_data(X).fit(anchors, background_weight=wb, seed=7)
        dw = max(dw, float(np.abs(e.W_ - AnchoredTPP(missing="zero").fit_data(X)
                                  .fit(anchors, seed=7).W_).max()))
        recs.add(review_list(e.scores(), truth, 15)["found_hidden_mud"])
    base7 = AnchoredTPP(missing="zero").fit_data(X).fit(anchors, seed=7)
    for c in np.logspace(-1, 2, 16):
        e = AnchoredTPP(missing="zero").fit_data(X).fit(
            anchors, pull=c, jitter=0.3 * c / 3.0, seed=7)
        dr = max(dr, float(np.abs(e.W_ - base7.W_).max()))
    rec_ex, cos_ex = [], []
    for wb in np.logspace(-3, -0.3, 7):
        e = AnchoredTPP(missing="zero").fit_data(X).fit_least_squares(
            anchors, background_weight=wb, seed=7)
        rec_ex.append(review_list(e.scores(), truth, 15)["found_hidden_mud"])
        cos_ex.append(round(float(abs(e.W_[:, 0] @ base7.W_[:, 0])), 3))
    cos_c = []
    for s in range(16):
        Xa, ta = generate_core(seed=42 + s)
        cos_c.append(AnchoredTPP(missing="zero").fit_data(Xa)
                     .fit(np.where(ta == VISIBLE_MUD)[0][:20], seed=s).centroid_alignment())
    R["invariance"] = {
        "recovery_over_omega_bg": sorted(recs),
        "max_dev_omega_bg": dw, "max_dev_rho": dr,
        "iterative_ls_recovery": rec_ex, "iterative_ls_cos": cos_ex,
        "centroid_cos_mean": round(float(np.mean(cos_c)), 4),
        "centroid_cos_sd": round(float(np.std(cos_c)), 4),
    }

    # ---- spatial autocorrelation -------------------------------------------
    def spatial(phi, block, mode, n=80, m=20):
        # NOTE: seed base MUST match exp_spatial.py (100 + s).  Until v3.0 this
        # used 1000 + s, an independent draw of the same experiment, so the
        # JSON and fig_spatial.pdf reported different means for the same
        # quantity -- around 0.5 samples apart, which is of the same order as
        # the standard errors either of them quote.
        v = []
        for s in range(n):
            Xg, tg = generate_core(seed=100 + s, ar_phi=phi, block=block)
            rng = np.random.default_rng(s)
            vis = np.sort(np.where(tg == VISIBLE_MUD)[0])
            a = (rng.choice(vis, m, replace=False) if mode == "spread"
                 else vis[int(rng.integers(0, max(1, len(vis) - m))):][:m])
            e = AnchoredTPP(missing="zero").fit_data(Xg).fit(a, seed=s)
            v.append(review_list(e.scores(), tg, 15)["found_hidden_mud"])
        v = np.asarray(v, float)
        return round(float(v.mean()), 2), round(float(v.std() / np.sqrt(n)), 2)

    R["spatial"] = {}
    for phi, blk, lab in [(0.0, 1, "iid"), (0.7, 8, "0.70"),
                          (0.85, 12, "0.85"), (0.95, 25, "0.95")]:
        sp, spe = spatial(phi, blk, "spread")
        cl, cle = spatial(phi, blk, "clustered")
        R["spatial"][lab] = {"spread": sp, "spread_se": spe,
                             "clustered": cl, "clustered_se": cle,
                             "diff": round(sp - cl, 2),
                             "diff_se": round(float(np.hypot(spe, cle)), 2)}

    # ---- benchmark integrity ----------------------------------------
    R["single_element"] = []
    for cl_sd, strength in [(0.0, 1.0), (0.7, 1.5), (1.0, 1.5), (1.0, 2.0), (1.4, 2.0)]:
        acc = {k: [] for k in ("Ba", "anchor1e", "oracle1e", "TPP", "PU")}
        for s in range(30):
            Xg, tg = generate_core(seed=2000 + s, cl_natural_sd=cl_sd,
                                   contamination_strength=strength)
            rng = np.random.default_rng(s)
            a = rng.choice(np.where(tg == VISIBLE_MUD)[0], 20, replace=False)
            e = AnchoredTPP(missing="zero").fit_data(Xg).fit(a, seed=s)
            Z = e.Xs_
            acc["TPP"].append(review_list(e.scores(), tg, 15)["found_hidden_mud"])
            acc["Ba"].append(review_list(single_element(Z, IDX["Ba"]), tg, 15)["found_hidden_mud"])
            acc["anchor1e"].append(review_list(single_element_anchor_selected(Z, a), tg, 15)["found_hidden_mud"])
            acc["oracle1e"].append(single_element_oracle(Z, tg, review_list, 15)[0])
            acc["PU"].append(review_list(pu_logistic_scores(Z, a), tg, 15)["found_hidden_mud"])
        R["single_element"].append({
            "cl_natural_sd": cl_sd, "strength": strength,
            **{k: round(float(np.mean(v)), 1) for k, v in acc.items()}})

    X0, _ = generate_core(seed=42, contamination_strength=0.0)
    R["natural_sd"] = {e_: round(float(X0[:, i].std()), 3)
                       for e_, i in IDX.items() if e_ in ("Si", "K", "Ba", "S", "Cl", "Na")}
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="Hall (2018) XRF CSV; enables the semi-synthetic run")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()

    if not args.skip_figures:
        for script in ["make_fig_workflow.py",  # paper Fig. 1
                       "exp_sensitivity.py",    # paper Fig. 2
                       "make_figs.py",          # paper Figs. 3-5
                       "exp_spatial.py",        # paper Fig. 6
                       "exp_anchor_economy.py",  # anchor-economy numbers in text
                       "exp_single_element.py"]:  # paper Sect. 5.5 numbers
            print(f"\n--- {script} " + "-" * (60 - len(script)))
            subprocess.run([sys.executable, os.path.join(HERE, script)],
                           cwd=HERE, check=True)

    print("\n--- collecting numbers " + "-" * 45)
    R = collect()
    out = os.path.join(ROOT, "results_v30.json")
    json.dump(R, open(out, "w"), indent=1)
    print(f"  wrote {os.path.relpath(out, ROOT)}")

    if args.csv:
        csv_path = os.path.abspath(args.csv)   # subprocesses run with cwd=HERE
        if not os.path.exists(csv_path):
            raise SystemExit(f"--csv path not found: {csv_path}")
        for script in ["semisynth.py", "exp_real_compare.py",
                       "exp_real_semisup.py"]:
            print(f"\n--- {script} " + "-" * (60 - len(script)))
            subprocess.run([sys.executable, os.path.join(HERE, script),
                            "--csv", csv_path], cwd=HERE, check=True)
    else:
        print("\n  (semi-synthetic experiments skipped: pass --csv to include them)")
        print("  These are the PRIMARY evidence for the method -- see README.")


if __name__ == "__main__":
    main()
