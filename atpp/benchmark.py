"""
Synthetic XRF core-scan benchmark (paper Section 4).

REPRODUCIBILITY GUARANTEE
-------------------------
`generate_core()` called with default arguments reproduces the v2.0 generator
bit-for-bit, and therefore the published figures and numbers of v2.1/v2.2.  The
new arguments (`ar_phi`, `block`, `cl_natural_sd`) all default to values that
disable them.  Do not change the defaults without regenerating every figure.
"""
from __future__ import annotations

import numpy as np

ELEMENTS = ['Si', 'Ti', 'Al', 'K', 'Zr', 'Rb', 'Nb',        # detrital
            'Ca', 'Sr', 'Mg', 'Mn',                          # carbonate
            'Fe', 'V', 'Mo', 'U', 'Ni', 'Cu', 'Zn', 'Cr',    # redox
            'Ba', 'S', 'Cl', 'P', 'Na', 'Th']                # mud / sulphide / misc
IDX = {e: i for i, e in enumerate(ELEMENTS)}

# truth codes -- used for SCORING ONLY, never for fitting
CLEAN, HIDDEN_MUD, VISIBLE_MUD, NATURAL_BARITE = 0, 1, 2, 3

__all__ = ["ELEMENTS", "IDX", "CLEAN", "HIDDEN_MUD", "VISIBLE_MUD",
           "NATURAL_BARITE", "generate_core"]


def _ar1(rng, n, phi):
    """AR(1) series with unit marginal variance."""
    if phi <= 0:
        return rng.normal(0, 1, n)
    z = rng.normal(0, 1, n)
    out = np.empty(n)
    out[0] = z[0]
    s = np.sqrt(1.0 - phi ** 2)
    for i in range(1, n):
        out[i] = phi * out[i - 1] + s * z[i]
    return out


def generate_core(n_samples=800, contamination_strength=1.0,
                  n_visible=40, n_hidden=15, n_barite=35, seed=42,
                  ar_phi=0.0, block=1, cl_natural_sd=0.0):
    """Synthetic XRF core scan with planted mud contamination + natural barite.

    Parameters
    ----------
    ar_phi : float, default 0.0
        AR(1) coefficient applied down-hole to the three latent factors.
        0.0 = independent samples, i.e. the published v2.0 design.  Real
        high-resolution core scans are strongly autocorrelated; see paper
        Section 5.5.
    block : int, default 1
        If > 1, planted populations are laid down as contiguous down-hole
        intervals of this length instead of at random.  1 = published design.
    cl_natural_sd : float, default 0.0
        Standard deviation of a formation-brine factor loading on Cl (and, at
        0.9x, on Na).  0.0 = published design.

        *** READ THIS BEFORE USING THE DEFAULT AS A METHOD COMPARISON ***
        With cl_natural_sd = 0.0, chlorine has no natural geological variation
        at all: its only variance is the 0.35 noise floor, so a planted +2.2
        (visible) / +1.32 (hidden) shift makes Cl a ~4-sigma near-perfect
        single-element marker.  The planted confounder (natural barite) was
        designed to defeat a *barium* criterion, and it does -- but nothing in
        the design defeats a *chlorine* criterion.

        Consequence: a label-free analyst rule that simply picks the element
        whose anchor mean deviates most (see
        `atpp.baselines.single_element_anchor_selected`) selects Cl and recovers
        ~12.8 of 15, against ~9.0 for anchored TPP.  The headline "11 vs 0"
        comparison in the paper holds only against the *barium* baseline that
        was specified in advance.

        Setting cl_natural_sd to ~0.7-1.0 gives chlorine realistic variability
        (formation-water salinity varies down-hole) and removes the shortcut.
        Note that this makes the benchmark much harder for every method, so
        `contamination_strength` needs raising to ~1.5-2.0 to keep the
        comparison informative.  See experiments/exp_single_element.py.
    """
    rng = np.random.default_rng(seed)
    p = len(ELEMENTS)

    # --- latent geological factors (dominant variance) ---
    if ar_phi > 0:
        detrital = _ar1(rng, n_samples, ar_phi)
        carbonate = -0.6 * detrital + 0.8 * _ar1(rng, n_samples, ar_phi)
        redox = _ar1(rng, n_samples, ar_phi)
    else:
        # exact v2.0 RNG call sequence -- do not reorder
        detrital = rng.normal(0, 1, n_samples)
        carbonate = -0.6 * detrital + rng.normal(0, 0.8, n_samples)
        redox = rng.normal(0, 1, n_samples)

    X = rng.normal(0, 0.35, (n_samples, p))
    for e in ['Si', 'Ti', 'Al', 'K', 'Zr', 'Rb', 'Nb']:
        X[:, IDX[e]] += 1.6 * detrital
    for e in ['Ca', 'Sr', 'Mg', 'Mn']:
        X[:, IDX[e]] += 1.6 * carbonate
    for e in ['Fe', 'V', 'Mo', 'U', 'Ni', 'Cu', 'Zn', 'Cr']:
        X[:, IDX[e]] += 1.5 * redox

    # Ba and S carry natural geological variation too -- no free lunch
    X[:, IDX['Ba']] += 0.9 * detrital + rng.normal(0, 0.6, n_samples)
    X[:, IDX['S']] += 0.8 * redox

    # --- optional: give Cl natural variation (see docstring) ---
    if cl_natural_sd > 0:
        brine = _ar1(rng, n_samples, ar_phi) if ar_phi > 0 \
            else rng.normal(0, 1, n_samples)
        X[:, IDX['Cl']] += cl_natural_sd * brine
        X[:, IDX['Na']] += 0.9 * cl_natural_sd * brine

    # --- plant the populations ---
    truth = np.zeros(n_samples, dtype=int)
    if block > 1:
        taken, groups = set(), []
        for k in (n_visible, n_hidden, n_barite):
            picks = []
            while len(picks) < k:
                st = int(rng.integers(0, max(1, n_samples - block)))
                for j in range(st, min(st + block, n_samples)):
                    if j not in taken and j not in picks and len(picks) < k:
                        picks.append(j)
            taken |= set(picks)
            groups.append(np.array(picks, dtype=int))
        vis, hid, bar = groups
    else:
        # exact v2.0 behaviour
        pick = rng.choice(n_samples, n_visible + n_hidden + n_barite, replace=False)
        vis = pick[:n_visible]
        hid = pick[n_visible:n_visible + n_hidden]
        bar = pick[n_visible + n_hidden:]

    truth[vis], truth[hid], truth[bar] = VISIBLE_MUD, HIDDEN_MUD, NATURAL_BARITE

    # drilling mud: Ba + K + Cl + Sr together
    for grp, amp in [(vis, 1.0), (hid, 0.6)]:
        a = amp * contamination_strength
        X[grp, IDX['Ba']] += 3.0 * a
        X[grp, IDX['K']] += 2.0 * a
        X[grp, IDX['Cl']] += 2.2 * a
        X[grp, IDX['Sr']] += 1.0 * a

    # natural barite: high Ba AND S, but no Cl / K excess
    X[bar, IDX['Ba']] += 3.2 * contamination_strength
    X[bar, IDX['S']] += 2.4 * contamination_strength

    return X, truth
