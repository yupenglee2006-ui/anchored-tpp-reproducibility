"""
Semi-synthetic validation of anchored TPP on real published XRF data
(paper Section 6).

Background covariance: real cuttings XRF data (Hall 2018, CSEG Recorder;
269 samples from the lateral of an unconventional well).
Planted signal: KCl-brine drilling-mud contamination (K2O + Na2O + Cl), added in
RAW concentration space so that compositional closure is respected.
Natural confounder: NOT planted -- clay-hosted potassium is already present in
the real data (K2O vs Al2O3, r = +0.94), which defeats any single-element K2O
criterion.

THIS IS THE EXPERIMENT TO TRUST.  Unlike the fully synthetic benchmark, every
element here has genuine geological variability, so no channel is an accidental
pure marker of the planted signal.  See exp_single_element.py for why that
distinction matters.

New in v2.2:
  * reports the anchor-selected single-element baseline and the PU / one-class
    baselines alongside the original three;
  * averages over realisations instead of reporting a single draw;
  * fails with an actionable message when the CSV is absent.

DATA
----
Place `XRF_dataset.csv` beside this script (or pass --csv).  It must contain the
nine normative mineral columns, twelve major-element oxides and Zr, as published
with Hall (2018).  The file is not redistributed here.
"""
from __future__ import annotations

import argparse
import os
import sys
import warnings

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from atpp import AnchoredTPP
from atpp.compositional import clr
from atpp.baselines import (pca_scores, single_element,
                            single_element_anchor_selected,
                            ocsvm_scores, pu_logistic_scores)

warnings.filterwarnings("ignore")

MINERAL = ['Quartz', 'K-Feldspar', 'Plagioclase', 'Chlorite', 'IlliteSmectiteMica',
           'Calcite', 'Ankerite/Dolomite', 'Pyrite', 'Organics']
OXIDE = ['Al2O3', 'SiO2', 'TiO2', 'Fe2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O',
         'P2O5', 'SO3', 'Cl']

CLEAN, HIDDEN, VISIBLE = 0, 1, 2



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


def load_csv(path):
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas is required for the semi-synthetic experiment: "
                         "pip install pandas")
    if not os.path.exists(path):
        raise SystemExit(
            f"\nCannot find '{path}'.\n\n"
            "The semi-synthetic experiment needs the published cuttings XRF\n"
            "dataset of Hall (2018), CSEG Recorder. It is not redistributed\n"
            "with this code. Save it as a CSV containing the columns:\n"
            f"  minerals : {', '.join(MINERAL)}\n"
            f"  oxides   : {', '.join(OXIDE)}\n"
            "  trace    : Zr\n"
            "and re-run with --csv /path/to/file.csv\n\n"
            "Every other experiment in this package runs without it.")
    df = pd.read_csv(path)
    missing = [c for c in MINERAL + OXIDE + ['Zr'] if c not in df.columns]
    if missing:
        raise SystemExit(f"'{path}' is missing required columns: {missing}")
    return df


def build_features(df, use_clr=True):
    """Standardised feature matrix. Mineral and oxide blocks are separate
    compositions and are clr-transformed independently; Zr is on an open scale
    (ppm) and is log-transformed rather than clr-transformed."""
    if use_clr:
        blocks = [clr(df[OXIDE].values), clr(df[MINERAL].values),
                  np.log(df[['Zr']].values)]
    else:
        blocks = [df[OXIDE].values, df[MINERAL].values, df[['Zr']].values]
    names = list(OXIDE) + list(MINERAL) + ['Zr']
    X = np.hstack(blocks)
    s = X.std(axis=0, keepdims=True)
    s[s == 0] = 1
    return (X - X.mean(axis=0, keepdims=True)) / s, names


def plant_contamination(df, strength=1.3, n_visible=25, n_hidden=12,
                        hidden_amp=0.55, seed=0, block=1, depth_col="Depth"):
    """Add KCl-brine mud to selected samples, in RAW concentration space.

    A mud-affected cutting carries added K, Na and Cl but no extra clay, so the
    contrast against clay-hosted potassium is Al2O3.

    block : int, default 1
        1 reproduces the published design (contaminated samples drawn at random).
        If > 1, contamination is planted in contiguous DOWN-HOLE intervals of this
        length, which is what mud invasion actually does. The real background has
        a median lag-1 autocorrelation of 0.61 in clr space, so contiguous
        planting is the realistic case and the one in which anchor placement
        matters (paper Section 5.5).
    """
    rng = np.random.default_rng(seed)
    out = df.copy()
    n = len(out)
    truth = np.zeros(n, dtype=int)

    if block > 1:
        order = (np.argsort(df[depth_col].values) if depth_col in df.columns
                 else np.arange(n))
        picks, taken = [], set()
        need = n_visible + n_hidden
        while len(picks) < need:
            st = int(rng.integers(0, max(1, n - block)))
            for j in range(st, min(st + block, n)):
                pos = int(order[j])
                if pos not in taken and len(picks) < need:
                    picks.append(pos); taken.add(pos)
        pick = np.array(picks, dtype=int)
    else:
        pick = rng.choice(n, n_visible + n_hidden, replace=False)

    vis, hid = pick[:n_visible], pick[n_visible:]
    truth[vis], truth[hid] = VISIBLE, HIDDEN

    add = {'K2O': 0.55, 'Na2O': 0.30, 'Cl': 0.22}
    for grp, amp in [(vis, 1.0), (hid, hidden_amp)]:
        for el, base in add.items():
            jitter = rng.normal(1.0, 0.15, len(grp)).clip(0.5, 1.5)
            out.loc[out.index[grp], el] += base * amp * strength * jitter
    return out, truth, vis, hid


def review(score, truth, k=12):
    cand = np.where(truth != VISIBLE)[0]
    top = cand[np.argsort(-np.asarray(score)[cand])][:k]
    return dict(found=int((truth[top] == HIDDEN).sum()),
                false=int((truth[top] == CLEAN).sum()))


def best_sign(score, truth, k=12):
    a, b = review(score, truth, k), review(-np.asarray(score), truth, k)
    return a if a["found"] >= b["found"] else b


def main(csv, n_real, strength, n_anchor):
    raw = load_csv(csv)
    print(f"real background: {raw.shape[0]} samples, "
          f"{len(MINERAL) + len(OXIDE) + 1} features")
    K = 12

    for tag, use_clr in [("CLR (compositional)", True), ("raw wt% (no CLR)", False)]:
        acc = {k: [] for k in ("K2O", "anchor1e", "PCA", "TPP", "PU", "OCSVM")}
        for s in range(n_real):
            df, truth, vis, hid = plant_contamination(raw, strength=strength, seed=s)
            X, names = build_features(df, use_clr=use_clr)
            anchors = vis[:n_anchor]
            eng = AnchoredTPP(standardise=False, missing="error")
            eng.fit_data(X, feature_names=names).fit(anchors, seed=s)

            acc["K2O"].append(best_sign(single_element(X, names.index('K2O')), truth, K)["found"])
            acc["anchor1e"].append(review(single_element_anchor_selected(X, anchors), truth, K)["found"])
            acc["PCA"].append(best_sign(pca_scores(X)[:, 0], truth, K)["found"])
            acc["TPP"].append(review(eng.scores(), truth, K)["found"])
            acc["PU"].append(review(pu_logistic_scores(X, anchors), truth, K)["found"])
            acc["OCSVM"].append(review(ocsvm_scores(X, anchors), truth, K)["found"])

        print(f"\n=== {tag} === (mean of {n_real} realisations, "
              f"{n_anchor} anchors, of 12 hidden)")
        for k, lab in [("PCA", "blind PCA (PC1)"),
                       ("K2O", "single-element K2O (a priori)"),
                       ("anchor1e", "single-element, anchor-selected"),
                       ("OCSVM", "one-class SVM on anchors"),
                       ("PU", "PU logistic (anchors positive)"),
                       ("TPP", "anchored TPP")]:
            v = np.array(acc[k], float)
            print(f"  {lab:<34} {v.mean():5.2f} +- {v.std():.2f}")

        if use_clr:
            df, truth, vis, hid = plant_contamination(raw, strength=strength, seed=0)
            X, names = build_features(df, use_clr=True)
            eng = AnchoredTPP(standardise=False, missing="error")
            eng.fit_data(X, feature_names=names).fit(vis[:n_anchor], seed=0)
            print("  TPP loadings:",
                  ", ".join(f"{n}{w:+.2f}" for n, w in eng.loadings(top=7)))
            st = eng.anchor_stability(n_splits=12, seed=0)
            print(f"  anchor stability: mean|cos| {st['mean_abs_cos']:.3f}, "
                  f"top-15 overlap {st['mean_top15_overlap']:.2f}")

    print("\nNOTE: 'single-element, anchor-selected' and the PU / one-class rows")
    print("are new in v2.2. They consume the same anchor information anchored TPP")
    print("does and are the fair comparison; PCA and the a-priori K2O criterion do not.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    here = os.path.dirname(os.path.abspath(__file__))
    ap.add_argument("--csv", default=os.path.join(here, "XRF_dataset.csv"))
    ap.add_argument("--n-real", type=int, default=12)
    ap.add_argument("--strength", type=float, default=1.3)
    ap.add_argument("--anchors", type=int, default=15)
    a = ap.parse_args()
    _require_csv(a.csv)
    main(a.csv, a.n_real, a.strength, a.anchors)
