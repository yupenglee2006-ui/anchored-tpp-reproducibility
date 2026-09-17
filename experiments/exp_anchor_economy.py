"""
Anchor economy on the fully synthetic benchmark.

The manuscript states how detection recovery depends on the number of anchors
m.  Until v3.2 that claim had no generator in this package and the quoted
figures were wrong: they came from a single anchor draw and reported the curve
as flat beyond m = 5.  It is not.  Recovery climbs steadily from m = 1 to
m = 40, and a single draw is far too noisy to read a curve from.

This script fixes the dataset (the headline configuration, seed 0) and averages
over N_DRAW independent anchor draws, so the only quantity varying is which
visible-contamination samples the analyst happened to pick.  Standard errors
are over anchor draws.

Writes results_anchor_economy.json.  No figure: the manuscript reports these
numbers in text only.
"""
import json
import os
import warnings

import numpy as np

from _common import plt  # noqa: F401  (path setup)
from atpp import AnchoredTPP, generate_core, review_list, VISIBLE_MUD

MS = [1, 2, 3, 5, 8, 10, 15, 20, 30, 40]
N_DRAW = 30
K = 15          # review-list length, matches the protocol elsewhere
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def sweep(ms=MS, n_draw=N_DRAW, k=K, data_seed=0):
    X, truth = generate_core(seed=data_seed)
    vis = np.where(truth == VISIBLE_MUD)[0]
    out = {}
    for m in ms:
        vals = []
        for s in range(n_draw):
            rng = np.random.default_rng(s)
            anchors = rng.choice(vis, m, replace=False)
            with warnings.catch_warnings():
                # m = 1 gives a rank-1 target; axis 2 is arbitrary but the
                # ranking axis, which is all this experiment uses, is defined.
                warnings.simplefilter("ignore")
                eng = AnchoredTPP(missing="zero").fit_data(X).fit(anchors, seed=s)
            vals.append(review_list(eng.scores(), truth, k)["found_hidden_mud"])
        v = np.asarray(vals, float)
        out[m] = (round(float(v.mean()), 2),
                  round(float(v.std() / np.sqrt(n_draw)), 2))
    return out


def main():
    res = sweep()
    print(f"  anchor economy, fully synthetic benchmark "
          f"({N_DRAW} anchor draws, fixed dataset; +- is the STANDARD ERROR)")
    print(f"  {'m':>4}  {'recovered of 15':>16}")
    for m, (mu, se) in res.items():
        print(f"  {m:>4}  {mu:>10.2f} +- {se:.2f}")
    print("\n  READ THIS: the curve is NOT flat beyond a handful of anchors.")
    print("  It rises monotonically across the whole range tested.  Quote the")
    print("  shape, not a saturation point.")

    out = os.path.join(ROOT, "results_anchor_economy.json")
    json.dump({"n_draw": N_DRAW, "k": K, "data_seed": 0,
               "m": list(res), "mean": [v[0] for v in res.values()],
               "se": [v[1] for v in res.values()]},
              open(out, "w"), indent=1)
    print(f"  wrote {os.path.relpath(out, ROOT)}")


if __name__ == "__main__":
    main()
