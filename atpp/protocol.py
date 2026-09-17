"""Review-list evaluation protocol (paper Section 4.4).

The protocol mirrors operational practice: the analyst has already flagged the
visible samples, so re-finding those is not a discovery.  Exclude them, rank the
rest, take the top k, and report what the top k actually turn out to be.
"""
from __future__ import annotations

import numpy as np

from .benchmark import CLEAN, HIDDEN_MUD, VISIBLE_MUD, NATURAL_BARITE

__all__ = ["review_list", "best_sign_review"]


def review_list(score, truth, k=15):
    """Rank the samples the analyst did NOT flag; report what the top k are.

    `score` is ranked in DESCENDING order. If your statistic points the other
    way, negate it -- or use `best_sign_review`, which is the fair thing to do
    for sign-ambiguous statistics such as principal components.
    """
    score = np.asarray(score, dtype=float)
    truth = np.asarray(truth)
    candidates = np.where(truth != VISIBLE_MUD)[0]
    if k > candidates.size:
        raise ValueError(f"k={k} exceeds {candidates.size} candidates")
    top = candidates[np.argsort(-score[candidates])][:k]
    return dict(
        found_hidden_mud=int((truth[top] == HIDDEN_MUD).sum()),
        false_pos_barite=int((truth[top] == NATURAL_BARITE).sum()),
        false_pos_clean=int((truth[top] == CLEAN).sum()),
        total_hidden=int((truth == HIDDEN_MUD).sum()),
        top_idx=top,
    )


def best_sign_review(score, truth, k=15):
    """Review list using whichever SIGN of the statistic performs better.

    Principal components have arbitrary sign: `svd` may return -PC1 just as
    easily as +PC1, so scoring only +PC1 can flatter a comparison by accident.
    Giving the baseline the better of the two signs is an oracle advantage, and
    therefore conservative -- if the baseline still loses, the result is safe.

    On the published benchmark both signs of PC1 and of Ba recover 0 of 15, so
    the paper's headline comparison is not an artefact of sign choice.  This
    function exists so that remains checkable rather than assumed.
    """
    score = np.asarray(score, dtype=float)
    a = review_list(score, truth, k)
    b = review_list(-score, truth, k)
    return a if a["found_hidden_mud"] >= b["found_hidden_mud"] else b
