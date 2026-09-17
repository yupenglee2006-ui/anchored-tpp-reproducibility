"""
Comparison baselines.

Two families:

  * UNSUPERVISED -- use no anchor information at all: PCA, isolation forest,
    robust Mahalanobis (MCD).  These answer "which samples are unusual", not
    "unusual in what respect".

  * ANCHOR-USING -- consume the same information anchored TPP does (a few
    positives, no negatives): one-class SVM, PU learning, and the
    single-element rule below.  These are the fair comparison.

`single_element_anchor_selected` was added in v2.2 and matters.  See its
docstring.
"""
from __future__ import annotations

import numpy as np

__all__ = ["pca_scores", "single_element", "single_element_anchor_selected",
           "single_element_oracle", "isolation_forest_scores", "mcd_scores",
           "ocsvm_scores", "pu_logistic_scores"]


# ------------------------------------------------------------------ unsupervised
def pca_scores(Xs, n_components=2):
    """Blind PCA on already-standardised data. Returns (n, n_components)."""
    Xc = Xs - Xs.mean(axis=0)
    _, _, Vt = np.linalg.svd(Xc, full_matrices=False)
    return Xc @ Vt[:n_components].T


def isolation_forest_scores(Xs, seed=0):
    """Outlyingness (higher = more anomalous)."""
    from sklearn.ensemble import IsolationForest
    return -IsolationForest(random_state=seed).fit(Xs).score_samples(Xs)


def mcd_scores(Xs, seed=0):
    """Robust Mahalanobis distance via minimum covariance determinant.

    Requires n > p; raises otherwise.  Anchored TPP has no such requirement
    (it never inverts a covariance matrix) -- that contrast is the point of
    paper Section 5.6.
    """
    from sklearn.covariance import MinCovDet
    n, p = Xs.shape
    if n <= p:
        raise ValueError(
            f"MCD needs n > p; got n={n}, p={p}. This is exactly the regime "
            "where anchored TPP remains well posed and MCD does not."
        )
    return MinCovDet(random_state=seed).fit(Xs).mahalanobis(Xs)


# ------------------------------------------------------------------ single element
def single_element(Xs, idx):
    """Rank by one standardised channel, chosen a priori (e.g. Ba)."""
    return Xs[:, idx]


def single_element_anchor_selected(Xs, anchors, return_index=False):
    """Rank by the single channel whose ANCHOR MEAN deviates most from zero.

    This is label-free -- it uses only the anchors, exactly the information
    anchored TPP uses -- and it is what a competent analyst would actually do
    after flagging a few samples: look at which element those samples are most
    obviously enriched (or depleted) in, and sort on it.  It is therefore the
    honest strongest simple alternative, and should be reported alongside the
    a-priori single-element criterion.

    *** WHY THIS MATTERS ***
    On the published synthetic benchmark (cl_natural_sd=0) this rule selects Cl
    and recovers ~12.8 of 15, versus ~9.0 for anchored TPP and 0 for the
    a-priori Ba criterion.  The benchmark's confounder was designed to defeat a
    barium criterion; it does not defeat a chlorine one, because chlorine was
    given no natural variance in the generator.  The paper's claim that the
    benchmark "cannot be isolated by any single-element criterion" holds for the
    pre-specified Ba baseline but not for this one.

    The semi-synthetic experiment (paper Section 6), where the background is
    real data and every element has genuine geological variability, is the
    setting where this baseline is a meaningful test of the method.  RUN IT
    THERE -- see experiments/semisynth.py, which now reports it.
    """
    Xs = np.asarray(Xs, dtype=float)
    a = np.asarray(anchors, dtype=int)
    d = Xs[a].mean(axis=0)
    j = int(np.argmax(np.abs(d)))
    score = np.sign(d[j]) * Xs[:, j]
    return (score, j) if return_index else score


def single_element_oracle(Xs, truth, review_fn, k=15):
    """Best achievable single-element result, chosen USING the hidden labels.

    Not achievable in practice -- reported only as an upper bound, to show how
    much of the gap to `single_element_anchor_selected` is label information.
    """
    best, best_j = -1, None
    for j in range(Xs.shape[1]):
        r = max(review_fn(Xs[:, j], truth, k)["found_hidden_mud"],
                review_fn(-Xs[:, j], truth, k)["found_hidden_mud"])
        if r > best:
            best, best_j = r, j
    return best, best_j


# ------------------------------------------------------------------ anchor-using ML
def ocsvm_scores(Xs, anchors, nu=0.2, gamma="scale"):
    """One-class SVM fitted on the anchors; higher = closer to the anchor class."""
    from sklearn.svm import OneClassSVM
    a = np.asarray(anchors, dtype=int)
    return OneClassSVM(nu=nu, gamma=gamma).fit(Xs[a]).decision_function(Xs)


def pu_logistic_scores(Xs, anchors, C=1.0, max_iter=2000):
    """Positive--unlabelled logistic classifier: anchors positive, rest unlabelled.

    The naive (non-traditional) PU formulation of Elkan & Noto (2008): train
    positive-vs-unlabelled directly and rank by the resulting probability, which
    is monotone in the true posterior and therefore adequate for ranking.
    """
    from sklearn.linear_model import LogisticRegression
    a = np.asarray(anchors, dtype=int)
    y = np.zeros(len(Xs))
    y[a] = 1
    clf = LogisticRegression(max_iter=max_iter, C=C).fit(Xs, y)
    return clf.predict_proba(Xs)[:, 1]
