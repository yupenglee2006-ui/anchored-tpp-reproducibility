"""Compositional pre-treatment (paper Section 3.5).

XRF concentrations are compositional: non-negative and carrying only relative
information.  Covariances on raw concentrations are distorted and spurious
negative correlations arise, and because anchored TPP operates directly on
X^T Omega T, that distortion propagates into the projection.

On the semi-synthetic experiment the clr transform improves recovery by roughly
a factor of 1.8.  Treat it as a requirement, not a refinement.
"""
from __future__ import annotations

import numpy as np

__all__ = ["multiplicative_replacement", "clr", "close"]


def close(A):
    """Close each row to sum 1."""
    A = np.asarray(A, dtype=float)
    s = A.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return A / s


def multiplicative_replacement(A, delta=None):
    """Replace zeros / below-detection values without destroying other ratios.

    This is the right way to handle below-detection-limit XRF returns before a
    log-ratio transform -- substituting 0 is inadmissible under a logarithm, and
    substituting the column mean destroys the compositional information you are
    about to use.

    delta : float, optional
        Replacement value. Defaults to 0.65 x the smallest positive value
        present, the usual rule of thumb. If you know the actual detection
        limit per element, pass 0.65 x that instead -- it is strictly better.
    """
    A = np.asarray(A, dtype=float).copy()
    if np.isnan(A).any():
        raise ValueError(
            "NaNs present. Decide explicitly whether they are "
            "below-detection (replace with a detection-limit-based delta) or "
            "not-measured (drop the channel) before calling this."
        )
    if (A < 0).any():
        raise ValueError("negative values are not compositional")
    if delta is None:
        pos = A[A > 0]
        delta = 0.65 * pos.min() if pos.size else 1e-6
    for i in range(A.shape[0]):
        row = A[i]
        z = row == 0
        if z.any():
            if (~z).sum() == 0:
                raise ValueError(f"row {i} is entirely zero")
            row[z] = delta
            row[~z] = row[~z] * (1.0 - z.sum() * delta / row[~z].sum())
        A[i] = row
    return A


def clr(A, delta=None):
    """Centred log-ratio transform of a (sub)composition.

    Returns log(x_j / g(x)) where g is the row geometric mean.

    INTERPRETING LOADINGS ON clr VARIABLES: the resulting weights are on
    log-ratios relative to the compositional centre, NOT on weight percentages.
    A positive loading means enrichment relative to the sample's own geometric
    mean. clr components sum to zero by construction, so loadings are not
    independent, and a scatter of small negative loadings across unrelated
    variables is an artefact of closure rather than a set of findings.
    """
    A = multiplicative_replacement(A, delta=delta)
    A = close(A)
    L = np.log(A)
    return L - L.mean(axis=1, keepdims=True)
