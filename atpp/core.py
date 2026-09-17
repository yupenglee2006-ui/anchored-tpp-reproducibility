"""
Anchored Targeted Projection Pursuit -- core engine.

Reference: Li & Yuan, "Anchored Targeted Projection Pursuit: Label-Free,
Hypothesis-Driven Quality Control of Multivariate Geochemical Data", v2.3.

THE ESTIMATOR
-------------
Given a standardised data matrix X (n x p) and an anchor set A of m << n samples
that the analyst believes share a property, build a target matrix T (n x 2) with

    (T[i,0], T[i,1]) = (rho, eps_i)   for i in A
                     = (0,   0)       otherwise

and diagonal weights Omega with omega_i = 1 on A and omega_bg elsewhere.  The
projection is the orthonormal polar factor of the weighted cross-product matrix

    M = X^T Omega T,    M = U S V^T,    W_hat = U V^T.

W_hat maximises the ALIGNMENT objective  tr(W^T X^T Omega T)  subject to
W^T W = I, exactly, for rectangular W (von Neumann's trace inequality).

IMPORTANT -- corrected in v2.2
------------------------------
W_hat is NOT the exact minimiser of the weighted least-squares objective
|| Omega^{1/2} (XW - T) ||_F^2.  Expanding that objective gives

    tr(W^T X^T Omega X W) - 2 tr(W^T X^T Omega T) + tr(T^T Omega T)

whose first (dispersion) term is constant only when W is square (the balanced
Procrustes problem of Schoenemann 1966) or when X^T Omega X is proportional to
the identity.  For W of shape (p, 2) with p > 2 it varies over the Stiefel
manifold, and the problem -- "Procrustes projection" in the terminology of
Gower & Dijksterhuis (2004) -- has no closed-form solution.  Versions up to
v2.0 of this code claimed the closed form solved the least-squares objective.
It does not; it solves the alignment objective.  Use `fit_least_squares` for the
iterative alternative, and see `atpp.experiments.exp_sensitivity` for the
comparison.

TWO PROPERTIES WORTH KNOWING BEFORE DEPLOYING
---------------------------------------------
1. INVARIANCE.  Rows of T outside A are identically zero, so omega_bg multiplies
   zero and cancels exactly: W_hat does not depend on omega_bg at all, and is
   invariant to any common positive rescaling of (rho, sigma_eps).  There is
   nothing to tune.  `selftest_invariance()` verifies this at runtime.

2. ANCHOR LOCALITY.  M = sum_{i in A} x_i T[i,:]^T, so W_hat depends on X ONLY
   through the m anchor rows.  Consequences: cost is O(m p); no covariance matrix
   is formed or inverted, so the method is well posed for p >> n; and the FIT is
   insensitive to background values.  The SCORES are not -- the samples you are
   trying to find are themselves background rows, so their data quality matters
   more than the anchors'.  See `AnchoredTPP.transform`.

   A corollary is that the first axis is very close to the direction of the
   anchor centroid in standardised space (cos ~ 0.9999 on the benchmark).  The
   estimator is simple; the contribution is the formulation, the interactive
   loop and the review-list protocol.  Read `centroid_alignment()` for the
   diagnostic.
"""
from __future__ import annotations

import numpy as np

__all__ = ["Standardiser", "AnchoredTPP", "selftest_invariance"]


# --------------------------------------------------------------------------
# Standardisation with stored state
# --------------------------------------------------------------------------
class Standardiser:
    """Column-wise z-scoring that REMEMBERS its parameters.

    This exists because of a bug in v2.0 (`tpp_with_unknown_samples`), which
    standardised the training and application sets separately.  A projection
    fitted in one z-scale was then applied to data in a different z-scale,
    silently producing meaningless scores.  When you fit on one well and score
    another, you must reuse the fitted centre and scale -- that is what this
    class enforces.
    """

    def __init__(self, ddof: int = 0):
        self.ddof = ddof
        self.mean_ = None
        self.scale_ = None

    def fit(self, X) -> "Standardiser":
        X = np.asarray(X, dtype=float)
        self.mean_ = np.nanmean(X, axis=0)
        s = np.nanstd(X, axis=0, ddof=self.ddof)
        s = np.where((s == 0) | ~np.isfinite(s), 1.0, s)
        self.scale_ = s
        return self

    def transform(self, X):
        if self.mean_ is None:
            raise RuntimeError("Standardiser.fit must be called before transform.")
        X = np.asarray(X, dtype=float)
        if X.shape[1] != self.mean_.shape[0]:
            raise ValueError(
                f"expected {self.mean_.shape[0]} columns, got {X.shape[1]}"
            )
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)


# --------------------------------------------------------------------------
# Missing values
# --------------------------------------------------------------------------
def _handle_missing(Xs, policy, where):
    """Apply a missing-value policy to standardised data.

    policy : 'error' | 'mean' | 'zero'
        'mean'/'zero' are the same thing after standardisation (the column mean
        is 0), and are provided only so the intent is explicit in user code.
    """
    if not np.isnan(Xs).any():
        return Xs
    n_bad = int(np.isnan(Xs).sum())
    if policy == "error":
        raise ValueError(
            f"{n_bad} missing value(s) in {where}. Pass missing='mean' to impute "
            "to the column mean, or handle below-detection values with "
            "atpp.compositional.multiplicative_replacement before standardising "
            "(preferred for compositional data)."
        )
    if policy not in ("mean", "zero"):
        raise ValueError(f"unknown missing policy {policy!r}")
    return np.where(np.isnan(Xs), 0.0, Xs)


# --------------------------------------------------------------------------
# Engine
# --------------------------------------------------------------------------
class AnchoredTPP:
    """Anchored targeted projection pursuit.

    Parameters
    ----------
    standardise : bool
        Z-score the input. Standardisation parameters are stored so that
        `transform` can be applied consistently to new data.
    missing : {'error', 'mean', 'zero'}
        What to do with NaNs. Default 'error' -- silent imputation on real
        assay data is how bad QC results happen.

    Examples
    --------
    >>> eng = AnchoredTPP().fit_data(X)          # X : (n, p) raw
    >>> eng.fit(anchor_indices)                  # anchors the analyst dragged
    >>> ranking = eng.scores()                   # rank by this
    >>> eng.loadings(top=6)                      # the interpretable criterion
    """

    def __init__(self, standardise: bool = True, missing: str = "error"):
        self.standardise = standardise
        self.missing = missing
        self.scaler_ = Standardiser() if standardise else None
        self.Xs_ = None
        self.W_ = None
        self.Y_ = None
        self.anchor_idx_ = None
        self.feature_names_ = None

    # -------------------------------------------------- data
    def fit_data(self, X, feature_names=None) -> "AnchoredTPP":
        """Register the dataset to be screened (and fit standardisation)."""
        X = np.asarray(X, dtype=float)
        if X.ndim != 2:
            raise ValueError(f"X must be 2-D, got shape {X.shape}")
        if X.shape[0] < 2:
            raise ValueError("need at least 2 samples")
        Xs = self.scaler_.fit_transform(X) if self.standardise else X.copy()
        self.Xs_ = _handle_missing(Xs, self.missing, "X")
        self.n_, self.p_ = self.Xs_.shape
        if feature_names is not None:
            if len(feature_names) != self.p_:
                raise ValueError(
                    f"feature_names has {len(feature_names)} entries, "
                    f"X has {self.p_} columns"
                )
            self.feature_names_ = list(feature_names)
        else:
            self.feature_names_ = [f"v{i}" for i in range(self.p_)]
        return self

    # -------------------------------------------------- fit
    def fit(self, anchor_idx, pull: float = 3.0, background_weight: float = 0.05,
            jitter: float = 0.3, seed: int = 0) -> "AnchoredTPP":
        """Fit the projection from a set of anchor indices.

        `background_weight` and `pull` have NO effect on the result (see module
        docstring, Proposition 1).  They are retained for interface
        compatibility with v2.0 and because they document intent.
        """
        if self.Xs_ is None:
            raise RuntimeError("call fit_data(X) before fit(anchors)")
        a = np.asarray(anchor_idx, dtype=int).ravel()
        # De-duplicate while PRESERVING ORDER. Order matters: the axis-2 jitter
        # is drawn sequentially and assigned to a[0], a[1], ..., so sorting the
        # anchors would silently change the fitted projection and break
        # bit-for-bit reproducibility against v2.0.
        _, first = np.unique(a, return_index=True)
        a = a[np.sort(first)]
        if a.size == 0:
            raise ValueError("anchor set is empty")
        if a.min() < 0 or a.max() >= self.n_:
            raise ValueError(
                f"anchor indices must lie in [0, {self.n_ - 1}]; "
                f"got [{a.min()}, {a.max()}]"
            )
        if a.size < 2:
            # Still well defined on axis 1; axis 2 is degenerate. Warn, don't fail.
            import warnings
            warnings.warn(
                "fewer than 2 anchors: rank(M) <= 1, so TPP axis 2 is arbitrary. "
                "Axis 1 (used for ranking) remains well defined.",
                stacklevel=2,
            )

        rng = np.random.default_rng(seed)
        T = np.zeros((self.n_, 2))
        omega = np.full(self.n_, float(background_weight))
        T[a, 0] = pull
        T[a, 1] = rng.normal(0.0, jitter, a.size)
        omega[a] = 1.0

        M = (self.Xs_ * omega[:, None]).T @ T
        U, _, Vt = np.linalg.svd(M, full_matrices=False)
        self.W_ = U @ Vt
        self.Y_ = self.Xs_ @ self.W_
        self.anchor_idx_ = a
        self.pull_, self.jitter_, self.seed_ = pull, jitter, seed
        return self

    def fit_least_squares(self, anchor_idx, pull: float = 3.0,
                          background_weight: float = 0.05, jitter: float = 0.3,
                          seed: int = 0, n_iter: int = 3000, step: float = 1e-4):
        """Iterative alternative: minimise the FULL weighted least-squares
        objective (Equation 2 of the paper) by projected gradient descent on the
        Stiefel manifold, rather than the alignment objective.

        Unlike `fit`, this DOES depend on `background_weight`.  Provided so the
        difference can be examined; `fit` remains the recommended entry point.
        """
        if self.Xs_ is None:
            raise RuntimeError("call fit_data(X) before fit_least_squares")
        a = np.unique(np.asarray(anchor_idx, dtype=int).ravel())
        rng = np.random.default_rng(seed)
        T = np.zeros((self.n_, 2))
        w = np.full(self.n_, float(background_weight))
        T[a, 0] = pull
        T[a, 1] = rng.normal(0.0, jitter, a.size)
        w[a] = 1.0

        def polar(A):
            U, _, Vt = np.linalg.svd(A, full_matrices=False)
            return U @ Vt

        W = polar(self.Xs_.T @ (w[:, None] * T))
        for _ in range(n_iter):
            G = 2 * self.Xs_.T @ (w[:, None] * (self.Xs_ @ W - T))
            W = polar(W - step * G)
        self.W_ = W
        self.Y_ = self.Xs_ @ W
        self.anchor_idx_ = a
        return self

    # -------------------------------------------------- use
    def scores(self, axis: int = 0):
        """Ranking statistic. Higher = more like the anchors."""
        self._check_fitted()
        return self.Y_[:, axis]

    def transform(self, X_new):
        """Apply the fitted projection to NEW samples (e.g. another well).

        Uses the standardisation parameters learned in `fit_data`, which is the
        only correct thing to do -- re-standardising the new data on its own
        moments silently invalidates the projection.  This was the v2.0 bug.
        """
        self._check_fitted()
        X_new = np.asarray(X_new, dtype=float)
        Xs = self.scaler_.transform(X_new) if self.standardise else X_new
        Xs = _handle_missing(Xs, self.missing, "X_new")
        return Xs @ self.W_

    def loadings(self, axis: int = 0, top: int | None = None):
        """(name, weight) pairs, sorted by |weight| descending.

        NOTE for compositional data: if the features were clr-transformed, these
        are weights on LOG-RATIOS relative to the geometric mean of the
        (sub)composition, not on concentrations.  A positive loading means
        enrichment relative to the sample's own compositional centre.  Because
        clr components sum to zero, small negative loadings on unrelated
        variables are an artefact of closure, not findings.
        """
        self._check_fitted()
        w = self.W_[:, axis]
        order = np.argsort(-np.abs(w))
        if top is not None:
            order = order[:top]
        return [(self.feature_names_[i], float(w[i])) for i in order]

    # -------------------------------------------------- diagnostics
    def centroid_alignment(self) -> float:
        """|cos| between TPP axis 1 and the standardised anchor-centroid direction.

        Typically ~0.9999.  A value materially below 1 means the jitter is large
        relative to `pull`, or the anchor centroid is near the origin (anchors
        not actually distinctive) -- both worth investigating.
        """
        self._check_fitted()
        c = self.Xs_[self.anchor_idx_].mean(axis=0)
        nrm = np.linalg.norm(c)
        if nrm == 0:
            return float("nan")
        return float(abs(self.W_[:, 0] @ (c / nrm)))

    def anchor_stability(self, n_splits: int = 12, frac: float = 0.6, seed: int = 0):
        """Anchor-stability diagnostic recommended in the paper (Sections 5.4, 5.5).

        Refits on random subsets of the anchors and reports how consistent the
        loading vector is.  Report this alongside any operational result: an
        unstable loading vector means the anchor set does not define a coherent
        contrast, and the review list should not be trusted.

        For down-hole data with strong autocorrelation, prefer
        `anchor_stability_blocked`, which resamples contiguous depth blocks --
        random subsetting is optimistic when neighbouring samples are near
        duplicates.

        Returns
        -------
        dict with mean/min pairwise |cos| between loading vectors, the per-split
        loading matrix, and the mean Spearman-free rank overlap of the top-k lists.
        """
        self._check_fitted()
        rng = np.random.default_rng(seed)
        a = self.anchor_idx_
        k = max(2, int(round(frac * a.size)))
        Ws, tops = [], []
        for s in range(n_splits):
            sub = rng.choice(a, size=min(k, a.size), replace=False)
            e = AnchoredTPP(standardise=False, missing="zero")
            e.Xs_ = self.Xs_
            e.n_, e.p_ = self.Xs_.shape
            e.feature_names_ = self.feature_names_
            e.fit(sub, pull=self.pull_, jitter=self.jitter_, seed=s)
            Ws.append(e.W_[:, 0])
            tops.append(set(np.argsort(-e.Y_[:, 0])[:15].tolist()))
        Wm = np.array(Ws)
        cos = np.abs(Wm @ Wm.T)
        iu = np.triu_indices(len(Ws), 1)
        overlaps = [len(tops[i] & tops[j]) / 15 for i, j in zip(*iu)]
        return {
            "mean_abs_cos": float(cos[iu].mean()),
            "min_abs_cos": float(cos[iu].min()),
            "mean_top15_overlap": float(np.mean(overlaps)),
            "loadings": Wm,
        }

    def anchor_stability_blocked(self, depth=None, n_splits: int = 12,
                                 frac: float = 0.6, seed: int = 0):
        """Depth-blocked variant of `anchor_stability`.

        Drops whole contiguous runs of anchors rather than random individuals.
        Use this whenever samples are autocorrelated down-hole -- the paper
        (Section 5.5) shows that clustered anchors give materially worse and far
        more variable recovery, which random subsetting hides.
        """
        self._check_fitted()
        a = np.sort(self.anchor_idx_) if depth is None else \
            self.anchor_idx_[np.argsort(np.asarray(depth)[self.anchor_idx_])]
        rng = np.random.default_rng(seed)
        keep = max(2, int(round(frac * a.size)))
        Ws = []
        for s in range(n_splits):
            start = int(rng.integers(0, max(1, a.size - keep + 1)))
            sub = a[start:start + keep]
            e = AnchoredTPP(standardise=False, missing="zero")
            e.Xs_ = self.Xs_
            e.n_, e.p_ = self.Xs_.shape
            e.feature_names_ = self.feature_names_
            e.fit(sub, pull=self.pull_, jitter=self.jitter_, seed=s)
            Ws.append(e.W_[:, 0])
        Wm = np.array(Ws)
        cos = np.abs(Wm @ Wm.T)
        iu = np.triu_indices(len(Ws), 1)
        return {"mean_abs_cos": float(cos[iu].mean()),
                "min_abs_cos": float(cos[iu].min()),
                "loadings": Wm}

    # -------------------------------------------------- internals
    def _check_fitted(self):
        if self.W_ is None:
            raise RuntimeError("call fit(anchors) first")


# --------------------------------------------------------------------------
# Runtime verification of the two propositions
# --------------------------------------------------------------------------
def selftest_invariance(X=None, anchors=None, verbose: bool = True) -> dict:
    """Verify Propositions 1 and 2 numerically on the supplied (or a random) X.

    Run this once on YOUR data before deploying: it confirms that the projection
    is unaffected by omega_bg and rho, and that it depends only on anchor rows.
    """
    rng = np.random.default_rng(0)
    if X is None:
        X = rng.normal(size=(400, 20))
        X[:15] += 2.0
    if anchors is None:
        anchors = np.arange(15)

    base = AnchoredTPP(missing="zero").fit_data(X).fit(anchors, seed=7)
    W0 = base.W_

    dev_w = 0.0
    for wb in [1e-4, 1e-3, 1e-2, 0.05, 0.1, 0.5, 1.0]:
        e = AnchoredTPP(missing="zero").fit_data(X).fit(
            anchors, background_weight=wb, seed=7)
        dev_w = max(dev_w, float(np.abs(e.W_ - W0).max()))

    dev_r = 0.0
    for c in [0.25, 1.0, 4.0, 25.0]:
        e = AnchoredTPP(missing="zero").fit_data(X).fit(
            anchors, pull=3.0 * c, jitter=0.3 * c, seed=7)
        dev_r = max(dev_r, float(np.abs(e.W_ - W0).max()))

    # locality: perturb every non-anchor row, refit with the SAME standardiser
    Xs = base.Xs_.copy()
    mask = np.ones(len(Xs), bool)
    mask[np.asarray(anchors)] = False
    Xs[mask] += 5.0 * rng.normal(size=Xs[mask].shape)
    e = AnchoredTPP(standardise=False, missing="zero").fit_data(Xs).fit(anchors, seed=7)
    dev_loc = float(np.abs(e.W_ - W0).max())

    out = {"max_dev_omega_bg": dev_w, "max_dev_rho": dev_r,
           "max_dev_background_perturbation": dev_loc,
           "centroid_alignment": base.centroid_alignment()}
    if verbose:
        print("Anchored TPP self-test")
        print(f"  Prop.1  max|W-W0| over omega_bg in [1e-4, 1] : {dev_w:.3e}")
        print(f"  Prop.1  max|W-W0| over rho scaling x100      : {dev_r:.3e}")
        print(f"  Prop.2  max|W-W0| perturbing all background  : {dev_loc:.3e}")
        print(f"  cos(axis 1, anchor centroid)                 : "
              f"{out['centroid_alignment']:.6f}")
        ok = max(dev_w, dev_r, dev_loc) < 1e-10
        print(f"  RESULT: {'PASS' if ok else 'FAIL'}")
    return out
