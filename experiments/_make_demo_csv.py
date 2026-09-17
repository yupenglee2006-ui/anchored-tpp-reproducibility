"""
Generate a stand-in CSV with the column schema the semi-synthetic experiments
expect, so that the pipeline can be smoke-tested before the real dataset is in
hand.

    python experiments/_make_demo_csv.py --out /tmp/demo.csv

WHAT THIS IS NOT
    It is not the Hall (2018) dataset, and it is not a substitute for it.  The
    covariance here is a crude imitation: a few latent factors driving mineral
    and oxide blocks, with K2O tied to the clay factor so that the K2O/Al2O3
    confounding the paper relies on is present in kind, but not in degree.
    Numbers produced from this file are meaningless and must never be quoted,
    compared against the manuscript, or included in a submission.

WHAT IT IS FOR
    Checking that `exp_real_compare.py`, `exp_real_semisup.py` and
    `semisynth.py` execute end to end -- import paths, column names, array
    shapes, figure writing -- on a machine that does not yet have the real CSV.
"""
import argparse

import numpy as np

MINERAL = ['Quartz', 'K-Feldspar', 'Plagioclase', 'Chlorite',
           'IlliteSmectiteMica', 'Calcite', 'Ankerite/Dolomite', 'Pyrite',
           'Organics']
OXIDE = ['Al2O3', 'SiO2', 'TiO2', 'Fe2O3', 'MnO', 'MgO', 'CaO', 'Na2O', 'K2O',
         'P2O5', 'SO3', 'Cl']


def build(n=269, seed=0):
    try:
        import pandas as pd
    except ImportError:
        raise SystemExit("pandas is required: pip install pandas")
    rng = np.random.default_rng(seed)

    depth = np.sort(rng.uniform(2000, 2600, n))
    # three latent controls with down-hole persistence, as in a real profile
    def ar1(phi=0.85):
        z = np.zeros(n)
        e = rng.standard_normal(n)
        for i in range(1, n):
            z[i] = phi * z[i - 1] + np.sqrt(1 - phi ** 2) * e[i]
        return z

    clay, carb, org = ar1(), ar1(), ar1(0.7)

    mineral = np.column_stack([
        40 + 8 * -clay + rng.normal(0, 2, n),      # Quartz
        6 + 2 * clay + rng.normal(0, .8, n),       # K-Feldspar
        5 + 1.5 * clay + rng.normal(0, .7, n),     # Plagioclase
        4 + 2 * clay + rng.normal(0, .6, n),       # Chlorite
        22 + 9 * clay + rng.normal(0, 2, n),       # IlliteSmectiteMica
        10 + 7 * carb + rng.normal(0, 2, n),       # Calcite
        6 + 4 * carb + rng.normal(0, 1.2, n),      # Ankerite/Dolomite
        2 + .8 * org + rng.normal(0, .3, n),       # Pyrite
        5 + 2.5 * org + rng.normal(0, .6, n)])     # Organics
    mineral = np.clip(mineral, 0.05, None)
    mineral = 100 * mineral / mineral.sum(1, keepdims=True)

    # K2O deliberately tracks the clay factor: this is the confounding the
    # paper's univariate baseline falls foul of.
    oxide = np.column_stack([
        14 + 4.5 * clay + rng.normal(0, .5, n),    # Al2O3
        58 - 7 * clay - 4 * carb + rng.normal(0, 1.5, n),   # SiO2
        0.7 + .2 * clay + rng.normal(0, .05, n),   # TiO2
        5 + 1.6 * clay + rng.normal(0, .4, n),     # Fe2O3
        0.08 + .03 * carb + rng.normal(0, .01, n),  # MnO
        2.2 + 1.1 * carb + rng.normal(0, .2, n),   # MgO
        7 + 5 * carb + rng.normal(0, .9, n),       # CaO
        1.3 + .3 * clay + rng.normal(0, .12, n),   # Na2O
        2.8 + 1.25 * clay + rng.normal(0, .18, n),  # K2O  <- clay-hosted
        0.15 + .04 * org + rng.normal(0, .02, n),  # P2O5
        1.1 + .5 * org + rng.normal(0, .15, n),    # SO3
        0.05 + .02 * rng.standard_normal(n)])      # Cl
    oxide = np.clip(oxide, 0.005, None)
    oxide = 100 * oxide / oxide.sum(1, keepdims=True)

    zr = np.clip(120 + 45 * -clay + rng.normal(0, 12, n), 5, None)

    df = pd.DataFrame(np.column_stack([mineral, oxide, zr, depth]),
                      columns=MINERAL + OXIDE + ['Zr', 'Depth'])
    return df


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--n", type=int, default=269)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    df = build(a.n, a.seed)
    df.to_csv(a.out, index=False)
    r = np.corrcoef(df["K2O"], df["Al2O3"])[0, 1]
    print(f"wrote {a.out}  ({len(df)} rows)")
    print(f"K2O-Al2O3 correlation r = {r:.2f}  "
          f"(real data: 0.94 -- this is an imitation, do not quote)")
    print("NOTE: method orderings on this file may invert relative to the "
          "paper\n      (clr vs raw wt%, for instance). That is a property of "
          "the imitation,\n      not a result. Use it only to check that the "
          "scripts execute.")
