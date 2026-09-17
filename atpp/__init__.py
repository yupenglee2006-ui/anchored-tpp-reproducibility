"""
Anchored Targeted Projection Pursuit -- reference implementation, v2.3.

Companion code for Li & Yuan, "Anchored Targeted Projection Pursuit:
Label-Free, Hypothesis-Driven Quality Control of Multivariate Geochemical Data".

Quick start
-----------
    from atpp import AnchoredTPP, selftest_invariance
    selftest_invariance()                    # verify the algebra on your machine

    eng = AnchoredTPP().fit_data(X, feature_names=elements)
    eng.fit(anchor_indices)                  # a handful of confident examples
    review = np.argsort(-eng.scores())[:15]  # the samples to re-inspect
    eng.loadings(top=6)                      # the criterion, in interpretable units
    eng.anchor_stability()                   # REPORT THIS alongside any result

Before deploying on real data, read the "Deployment checklist" section of
README.md -- particularly the notes on compositional pre-treatment, missing
values, and applying a fitted projection to a second well.
"""
__version__ = "2.3.0"

from .core import AnchoredTPP, Standardiser, selftest_invariance
from .benchmark import (ELEMENTS, IDX, CLEAN, HIDDEN_MUD, VISIBLE_MUD,
                        NATURAL_BARITE, generate_core)
from .protocol import review_list, best_sign_review
from . import baselines, compositional

__all__ = ["AnchoredTPP", "Standardiser", "selftest_invariance",
           "ELEMENTS", "IDX", "CLEAN", "HIDDEN_MUD", "VISIBLE_MUD",
           "NATURAL_BARITE", "generate_core", "review_list",
           "best_sign_review", "baselines", "compositional"]
